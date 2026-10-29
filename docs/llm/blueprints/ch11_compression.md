# Chapter 11: Model Compression — Blueprint

**Label:** `chap_lm_compression` · **Arc:** VI. Deploying · **Sections:** 11.1–11.6 (`sec_lm_quant_inference`, `sec_lm_outliers`, `sec_lm_gptq`, `sec_lm_rotations`, `sec_lm_qat`, `sec_lm_pruning`) · **Tiers:** all T1 (one Ada GPU, ≤ 15 GiB, ≤ ~15 min per notebook); no T2+ section, so no reduced-config flags are needed, but every notebook has a `FAST` flag that shrinks calibration and evaluation sizes for CI.

## Chapter overview

### Draft intro paragraph

A language model that wins a benchmark is useless if it does not fit, or if it decodes too slowly to serve. Chapter 7 introduced quantization for one narrow purpose: storing frozen weights in fewer bits so that a fine-tuning run fits in memory (`sec_lm_qlora`). Inference changes three things. The binding resource is now memory *traffic* as much as capacity, because decode reads every weight once per token (`eq_hw_decode_bound`). Activations are quantized too, so that integer or low-precision tensor cores can do the arithmetic. And a recipe is judged on held-out quality at a stated speed, not on reconstruction error. This chapter builds the inference toolbox in the order in which the difficulty arises. It starts with formats and kernels and a roofline that predicts when each pays off (11.1). It then covers activation outliers and the equivalent rescalings that tame them (11.2). Next come second-order, reconstruction-based rounding (GPTQ, 11.3), rotations and vector codebooks for 4-bit activations (11.4), and quantization-aware training with native microscaling formats (11.5). The chapter ends with pruning, sparsity and lossless coding (11.6). Every method is implemented from scratch, checked against brute force or an exact identity, and then compared in one common report with error bars.

### Learning objectives

After the chapter a reader can:

1. Predict, from a layer's shape, a format's bits per parameter (scales included) and a GPU's bandwidth and peak rates, the decode and prefill speedup of W8A16, W4A16 and W8A8, and name the batch size at which each stops helping (11.1).
2. Explain, with the invariant product `a_j·w_j`, why per-channel activation outliers break per-tensor activation quantization and how SmoothQuant (balance) and AWQ (salience) move difficulty between activations and weights without changing the function (11.2).
3. Derive the OBS update and its Cholesky form, implement GPTQ with lazy block updates, and verify it against exhaustive search on a 6-weight layer (11.3).
4. Prove that a randomized Hadamard rotation gives `max/RMS = O(√log d)` with high probability, fuse rotations into a Qwen3 checkpoint with logits unchanged to floating-point tolerance, and measure the W4A4 effect (11.4).
5. Train a fake-quantized model with a clipped straight-through estimator and compare it with post-training quantization at a matched data and update budget; read a native MXFP4 checkpoint tensor by tensor (11.5).
6. Apply unstructured, 2:4 and structured pruning, state which of them can speed up which phase, and verify a lossless exponent coder bit for bit (11.6).
7. Choose, for a stated deployment target (memory budget, batch regime, GPU generation), among PTQ, QAT, pruning and native low-bit formats, and defend the choice with a table of quality with confidence intervals, bytes and measured speed.

### Prerequisites

Book labels:

- `sec_numerics`: float formats; E4M3/E5M2/E2M1 layouts. Not re-taught.
- `sec_memory_precision`: bytes per parameter; the "halving wins twice" rule.
- `sec_hardware` (`tab_gpu_specs`, `eq_hw_decode_bound`, `subsec_hw-two-workloads`): ridge points; decode and prefill rooflines.
- `sec_perf_model`: `d2l.Benchmark`.
- `sec_compilation`: `torch.compile`, CUDA graphs for the decode loop.
- `sec_kv-cache`: GQA cache bytes (not reduced here; 12.4).
- `sec_gpt`, `sec_transformer-block`: `d2l.GPT`, RMSNorm, QK-norm hooks.
- `sec_custom_layer` (11.5): the straight-through estimator and `RoundSTE`.
- `sec_mdl-svd-low-rank` (11.4, 11.6): low rank and singular spectra.
- `sec_mdl-numerical-stability-conditioning` (11.3): κ(XᵀX)=κ(X)², damping.
- `subsec_mdl-psd` in `sec_mdl-eigendecompositions` (11.3): positive definiteness and the Cholesky factorization.
- `sec_mdl-concentration-generalization` (11.4): Rademacher/sub-Gaussian tails (`eq_mdl-subgaussian`).
- `sec_mdl-geometry-linear-algebraic-ops` (11.4): orthogonal maps preserve norms.
- `sec_mdl-information_theory` (11.4, 11.6): entropy, rate–distortion.
- `sec_mdl-statistics`: the bootstrap.

Earlier sections of this part:

- 1.2 (`d2l.load_qwen3`, `d2l.generate`).
- 1.5 (`d2l.EvalSuite`, `d2l.paired_bootstrap`).
- 7.4 (LoRA, used for recovery in 11.5 and 11.6).
- 7.6 (the quantizers and the Δ²/12 error).
- 4.4 (block-scaled formats; FP8 on Ada; NVFP4 simulation).
- 5.2 (the batch at which decode turns compute-bound).
- 9.1 (distillation loss, used in 11.6).

### Section dependency graph

```
7.6 quantizers ─► 11.1 formats, kernels, roofline ─► 11.2 outliers: SmoothQuant, AWQ ──┐
                      │                                    │                           ▼
                      │                                    └──────────────► 11.4 rotations, codebooks (W4A4)
                      ├──────────────► 11.3 GPTQ ─────────────────────────────┤        │
                      │                   │                                     ▼        ▼
                      │                   └─► 11.6 pruning (SparseGPT reuses 11.3) ◄── 11.5 QAT, MXFP4/NVFP4
                      └──────────────────────────────────────────────────────────────────┘
11.1 supplies fake_quant, window_nll, calib_tokens, compression_row to all later sections.
11.3's gptq_quantize is reused by 11.4 (rotate-then-GPTQ), 11.5 (PTQ comparator) and 11.6 (SparseGPT).
```

Recommended reading orders: 11.1 → 11.2 → 11.3 is the core (this is the "Large Language Models" course path: 11.1 and 11.3); 11.4 needs 11.1–11.2; 11.5 and 11.6 need 11.1 and 11.3 only.

### Running examples and shared artifacts

Used:

- R1 (`d2l.load_qwen3`, a `d2l.GPT`) for from-scratch surgery (rotations, QAT).
- Hugging Face Transformers `Qwen/Qwen3-1.7B` for kernels and benchmarks.
- `Qwen/Qwen3-0.6B`, `-4B`, `-8B` where a size sweep is needed.
- `openai/gpt-oss-20b` (11.5, tensor inspection).
- Calibration data `HuggingFaceFW/fineweb-edu` (`sample-10BT`, ODC-By); evaluation text `Salesforce/wikitext` (`wikitext-2-raw-v1`, CC BY-SA 3.0 / GFDL).
- The 1.5 `EvalSuite`.

Produced (all committed):

- `data/compression_report.jsonl`: one row per compressed checkpoint, written by `d2l.compression_row` and appended by every section. This is the chapter's main artifact. Fields: `name`, `base`, `method`, `bits_w`, `bits_a`, `group`, `eff_bits_per_weight`, `weight_bytes`, `ppl_wikitext2`, `ppl_ci95`, `task_scores`, `prefill_tok_s`, `decode_tok_s_b1`, `peak_mem_gib`, `gpu`, `torch`, `torchao`, `date`.
- DATA_HUB keys:
  - `lm-c11-calib-fineweb-128x2048`: `int32` NumPy array `[128, 2048]` of Qwen3 token ids from FineWeb-Edu `sample-10BT`, seed 0 (~1 MB).
  - `lm-c11-act-stats-qwen3-1p7b`: `.npz` of per-site, per-channel `absmax`, `rms`, `mean_abs` for all 28 layers (11.2).
  - `lm-c11-qwen3-1p7b-gptq-w4g128`: packed GPTQ checkpoint (optional, ~1.1 GB; reused by 12.x as a quantized target if a quantized fast decoder is wanted).
- Figures and tables in each section; the final comparison table `tab_lm_comp_all` is built in 11.6 from the JSONL rows.

### Notation used throughout the chapter

- Linear layers follow `nn.Linear`: weight `W ∈ R^{d_out×d_in}`, input rows `X ∈ R^{T×d_in}`, output `Y = X Wᵀ`. v5's identity `XW = (X diag(s)⁻¹)(diag(s) W)` is written here as `X Wᵀ = (X diag(s)⁻¹)(W diag(s))ᵀ` (scale *columns* of `W`, i.e. input channels). This is the same identity with `W` stored transposed; say so once in 11.2.
- `b` bits of a code, `g` group size, `b_eff = b + (b_scale + b_zero)/g` effective bits per weight.
- `π` peak FLOP/s (or OP/s) of the unit used, `β` memory bandwidth, `B` batch of tokens in flight (rows of `X`).
- `H = XᵀX ∈ R^{d_in×d_in}` is the layer Hessian of the reconstruction loss (11.3).
- "Fake quantization" means quantize then dequantize in floating point (simulation). "Real kernel" means the packed tensor is consumed by a dedicated GPU kernel.

### Compute plan

| § | Tier | GPU-time per full run (to be measured; target) | Peak GPU memory | Committed artifacts |
|---|---|---|---|---|
| 11.1 | T1 | ≤ 15 min (10 configs × ppl on ~160 windows, plus timing) | ≤ 9 GiB (1.7B BF16 ≈ 3.4 GB weights); 8B only quantized on load | report rows; `fig_lm_q_pareto`; layer-speed CSV |
| 11.2 | T1 | ≤ 12 min (stats pass + 3 methods) | ≤ 9 GiB | `lm-c11-act-stats-qwen3-1p7b`; rows |
| 11.3 | T1 | ≤ 15 min (GPTQ on 1.7B; `FAST`: 16 calibration sequences) | ≤ 10 GiB (one layer's Hessians in fp32 on GPU, weights streamed) | `lm-c11-qwen3-1p7b-gptq-w4g128`; rows |
| 11.4 | T1 | ≤ 15 min (3 rotation seeds × ppl; W4A4 simulated) | ≤ 10 GiB | rotation seeds, rows |
| 11.5 | T1 | ≤ 15 min for QAT (≈1k steps on Qwen3-0.6B) plus ≤ 5 min inspection | ≤ 14 GiB (0.6B full fine-tune, bf16 params, fp32 Adam state, activation checkpointing) | QAT checkpoint (optional), training log, rows; gpt-oss layer-0 expert slice cache (~0.42 GB, not committed) |
| 11.6 | T1 | ≤ 15 min | ≤ 12 GiB | rows; `tab_lm_comp_all` |

Hardware facts that apply to the whole chapter:

- The build box has RTX 4090 (Ada, 24 GB, `tab_gpu_specs`: 1.0 TB/s, 165 TF bf16 dense with fp32 accumulation, 330 TF fp8) with a ≤ 15 GiB slot.
- The A100 datasheet (https://www.nvidia.com/en-us/data-center/a100/) lists BF16 312 TFLOPS (624 with sparsity), INT8 624 TOPS (1,248 with sparsity), 2,039 GB/s for the 80 GB SXM (1,555 GB/s for the 40 GB PCIe and SXM parts; the 80 GB row is quoted for datasheet comparison only, and any arithmetic about a measured A100 run uses the 40 GB T2 part), and no FP8 or FP4 entry.

| Format | A100 | Ada (build box) | Blackwell |
|---|---|---|---|
| BF16 | native | native | native |
| INT8 W8A8 (integer tensor cores) | native | native | native |
| INT4 weight-only (W4A16: unpack + dequantize in-kernel, BF16 MMA) | runs (software dequant) | runs (software dequant) | runs |
| FP8 W8A8 | no FP8 tensor cores: only weight storage + dequantize | native | native |
| MXFP4 / NVFP4 *arithmetic* | simulated | simulated (a Triton kernel can read MXFP4 *storage* and dequantize into BF16 MMA, see 11.5) | native |
| 2:4 sparse tensor cores | native | to be verified by timing in 11.6 | native |

Each notebook labels every result "native", "real kernel, emulated format" or "simulated (fake quantization)". No speed is ever claimed from a simulated result.

## Optional mini-assignment

**Label:** `sec_lm_a11_mini`

v5 attaches no required assignment to Chapter 11 (A8 covers Chapters 12–13). This optional mini-assignment (label `sec_lm_a11_mini`) is a suggestion to the editor, not part of v5: a mini-assignment "compress Qwen3-1.7B to a budget", T1, with the following checkable targets:

- the packed weight bytes are ≤ a stated budget (computed by `d2l.effective_bits`);
- WikiText-2 perplexity is within a stated band of BF16 with the 95% paired-bootstrap interval reported;
- the quantized layer's output matches the student's own fake-quantized reference within `1e-3` relative error;
- a written analysis answers "which of PTQ, QAT, pruning would you ship for batch 1 on Ada, and for batch 64 on A100?".

Estimated student time: 6–8 hours.

## Teaching sources

All of v5's sources are kept, with what to take from each:

- [MIT 6.5940 Efficient AI](https://hanlab.mit.edu/courses/2026-fall-65940): the lecture sequence quantization → pruning → distillation, and the lab structure (implement k-means and linear quantization from scratch, then inspect). Take the ordering of 11.1 and 11.6 and the framing "memory traffic is the cost".
- Maarten Grootendorst, [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization): the visual vocabulary (absmax vs zero-point, outliers, calibration) for the 11.1 and 11.2 figures. Redraw, do not copy.
- [EMNLP 2025 efficient inference tutorial](https://aclanthology.org/2025.emnlp-tutorials.1/): the survey spine; use as the cross-check that the chapter covers its taxonomy.
- [torchao](https://github.com/pytorch/ao): the production API for 11.1, 11.5, 11.6 (`quantize_`, QAT, float8, sparsity). Documentation: https://docs.pytorch.org/ao/stable/index.html.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): GGUF block formats (11.1) and the k-quants design discussion (PR https://github.com/ggml-org/llama.cpp/pull/1684).

Added (verified 28 Oct 2026):

- Alex Smola, [Efficiency in LLMs](https://alex.smola.org/posts/45-mlss-efficiency/): memory-traffic framing and the weight-compression progression (from the Codex plan, Chapter 12 sources).
- [llm-compressor](https://github.com/vllm-project/llm-compressor): calibration and deployment workflows for supported serving stacks; the production counterpart of 11.2–11.3 (Codex 12.9).
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer): the vendor library for FP8, NVFP4, sparsity and pruning (11.5–11.6).

## Sections

### 11.1 Quantization for inference

- **Label:** `sec_lm_quant_inference` · **Tier:** T1 (no reduced config needed; `FAST=True` uses 40 evaluation windows and 3 timing repeats) · **Budget:** ≤ 15 min wall-clock on the build box, peak ≤ 9 GiB (Qwen3-1.7B BF16 ≈ 3.4 GB of weights; the 8B models are quantized on load from CPU, so GPU peak is the quantized size plus one layer) · **Codex units:** 12.1, 12.2 (and the deployment-report idea of 12.9)
- **Builds on (book):**
  - `sec_hardware`: `eq_hw_decode_bound` (cited, not re-derived), `tab_gpu_specs` (the 4090 row: 1.0 TB/s, 165 TF bf16, 330 TF fp8), `subsec_hw-two-workloads` (prefill compute-bound, decode bandwidth-bound).
  - `sec_numerics`: E4M3/E5M2 layouts (not re-taught).
  - `sec_memory_precision`: "every halving wins twice".
  - `sec_perf_model`: `d2l.Benchmark` for every timing.
  - `sec_compilation`: `torch.compile(mode='reduce-overhead')` for the decode loop.
  - `sec_kv-cache`: KV bytes per token, to show what weight quantization leaves behind.
- **Builds on (this part):**
  - 7.6 (recap in one paragraph, with `:numref:`sec_lm_qlora``): absmax and blockwise quantizers, NF4, the Δ²/12 error. Imports `d2l.quantize_blockwise`, `d2l.quantize_nf4`, `d2l.dequantize`, `d2l.QTensor` (canonical per 01_interfaces.md): `d2l.quantize_blockwise(w, bits=4, block_size=64, ...) -> d2l.QTensor` with fields `codes, scales, zeros, block_size, shape, bits` (`q.scales` replaces the old `absmax`); `d2l.quantize_nf4` returns the same type; `d2l.dequantize(q, dtype=torch.bfloat16)`. Usage: `q = d2l.quantize_blockwise(w, bits=4, block_size=64); w_hat = d2l.dequantize(q)`. Blocks run over the row-major flattened tensor, so a block of size `g` dividing `d_in` equals a group along the input dimension and no extra reshaping is needed (one-line fallback: assert `d_in % block_size == 0` in the notebook).
  - 1.2 (`d2l.load_qwen3`, `d2l.generate`).
  - 1.5 (`d2l.EvalSuite`, `d2l.paired_bootstrap`). Canonical API: `d2l.EvalSuite(['arc_challenge', 'hellaswag', 'mmlu', 'arc_easy', 'piqa', 'sciq'], n=500, seed=0).run(model, tok) -> d2l.EvalResult`; `.scores(task)` is an `np.ndarray` ordered by `item_id`, so two runs pair item by item. `d2l.paired_bootstrap(a, b, n_boot=10000, cluster=None, seed=0) -> d2l.BootstrapResult`; read `r.mean` and `r.ci` (not a 3-tuple). Task ids come from `d2l.EVAL_TASKS`.
  - 5.2 (the decode batch at which a layer turns compute-bound; here extended by the weight format only).
- **Exports** (`#@save`, all new names):
  - `d2l.fake_quant(x, bits, group_size=None, axis=-1, sym=True, clip=1.0)`: quantize then dequantize along `axis` in groups of `group_size` (`None` = whole axis, i.e. per-row/per-token for `axis=-1`). `sym=False` uses scale+zero-point. `clip<1` shrinks the range before rounding. Returns same dtype as `x`. Where a real quantized tensor is wanted (e.g. a 2-D weight with `axis=-1`), use `d2l.quantize_blockwise` (7.6), which returns a `d2l.QTensor`; `fake_quant` is the float round trip only.
  - `d2l.pack_int4(codes)` / `d2l.unpack_int4(packed)`: two unsigned nibbles per `uint8`, exact round trip. `pack_int4(q.codes)` packs the codes of a 7.6 `d2l.QTensor`; the packed result is the payload of a `QTensor` whose `codes` field holds the packed `uint8` when a real kernel consumes it.
  - `d2l.effective_bits(bits, group_size, scale_bits=16, zero_bits=0)`: `bits + (scale_bits+zero_bits)/group_size`.
  - `d2l.replace_linears(model, make, skip=('lm_head',))`: replaces every `nn.Linear` (or the fused linears of a `d2l.GPT`) by `make(name, linear)`, returns the replaced names. Shared by 11.2–11.6.
  - `d2l.W8A8Linear(weight, bias=None, act='token')`: reference int8 linear using `torch._int_mm` with per-token activation scales and per-output-channel weight scales; CPU fallback via int32 `matmul` on `int64` casts.
  - `d2l.window_nll(model, ids, seq_len=2048, batch_size=4)`: mean NLL of non-overlapping windows, one value per window (so perplexity is `exp(mean)` and bootstrap resampling is over windows). Accepts a Transformers model (`.logits`) or a `d2l.GPT` (tensor output).
  - `d2l.calib_tokens(n=128, seq_len=2048, seed=0)`: reads the committed `lm-c11-calib-fineweb-128x2048` array (`d2l.DATA_HUB`), or rebuilds it from streaming FineWeb-Edu `sample-10BT`.
  - `d2l.decode_tps(model, batch=1, ctx=512, steps=64, graph=True)`: tokens/s of single-token decode steps against a static cache of length `ctx`, with CUDA graphs when `graph`; median of repeats, returns `(median, iqr)`.
  - `d2l.compression_row(name, model, tok, **meta)`: runs `window_nll`, `d2l.EvalSuite([...], n=500, seed=0).run(model, tok)` (task scores from `EvalResult.scores(task)`), prefill and decode timing and peak memory, and appends one JSON row to `data/compression_report.jsonl`.
- **Models and data:**
  - `Qwen/Qwen3-1.7B` (Apache-2.0; 28 layers, hidden 2048, MLP 6144, 16 query and 8 KV heads, head_dim 128, vocab 151,936, tied embeddings; 1.72 B unique parameters, of which 0.311 B are the embedding/head).
  - `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` for the iso-memory sweep (all Apache-2.0).
  - Optional pre-quantized references, IDs verified on the Hub: `Qwen/Qwen3-1.7B-FP8` (fine-grained FP8, block 128×128, dynamic activations, Ada-native), `Qwen/Qwen3-1.7B-GPTQ-Int8`, `Qwen/Qwen3-8B-AWQ` and `Qwen/Qwen3-4B-AWQ` (4-bit, group 128, zero point). Each load is wrapped in `try/except` and its row marked `n/a` if the pinned package set cannot run it.
  - GGUF: `Qwen/Qwen3-0.6B-GGUF` (`Qwen3-0.6B-Q8_0.gguf`, ≈ 0.6 GB) for the layout test; file sizes of `Qwen/Qwen3-8B-GGUF` (Q4_K_M, Q5_K_M, Q6_K, Q8_0) read from Hub metadata, no download.
  - Calibration `HuggingFaceFW/fineweb-edu` `sample-10BT` (ODC-By), 128 sequences × 2048 tokens; evaluation `Salesforce/wikitext` `wikitext-2-raw-v1` test split (CC BY-SA 3.0).
- **Packages:** `torch ≥ 2.6`; `torchao` pinned at the version the notebook prints (API names below are checked against the 0.17 documentation; older releases use the function spellings `int4_weight_only(group_size=128)`, `int8_weight_only()`, `int8_dynamic_activation_int8_weight()`, `float8_dynamic_activation_float8_weight()`); `transformers ≥ 4.55`; `gguf`; `datasets`; `safetensors`.

#### Narrative arc

1. **The question.** *Why did my 4-bit model not get faster?* The opening cell loads Qwen3-1.7B in BF16 and prints three numbers: measured batch-1 decode tokens/s, the ceiling `β / bytes_per_token` from `eq_hw_decode_bound` (weights read per step: 1.72 B params × 2 B = 3.44 GB, so about 290 tokens/s at 1.0 TB/s, computed in the cell from the config), and the achieved fraction. It then applies 7.6's 4-bit round trip (`fake_quant(W, 4, 128)`) to every linear and prints perplexity (fine) and tokens/s (unchanged within noise, because the dequantized weights are stored in BF16 again: same bytes). The symptom is that *an accuracy simulation is not a speed-up*; the rest of the section builds the real thing.
2. **What changes from 7.6** (one recap paragraph plus a two-column table `tab_lm_q_vs_qlora`):
   - objective: memory capacity vs memory traffic and arithmetic;
   - weights dequantized to BF16 on the fly vs also quantizing activations so integer tensor cores run;
   - judged by the loss of the fine-tuned adapter vs held-out perplexity and tokens/s;
   - frozen vs final, calibrated weights.
3. **Granularity and effective bits.**
   - Per-tensor, per-channel (one scale per output row), per-group (`g` consecutive input weights), and for activations per-token.
   - Symmetric (scale only) vs asymmetric (scale + zero-point); clipping the range (a closed-form MSE search over `clip` on a Gaussian and on a real Qwen3 weight row).
   - `eq_lm_q_effbits`: `b_eff = b + (b_s + b_z)/g`; `tab_lm_q_effbits` lists the formats of this chapter with layout, bytes per block and `b_eff`: INT8 per-channel; INT4 g128, g64, g32 with FP16 scale (4.125, 4.25, 4.5); with a 16-bit zero point (+0.125 at g128); GGUF Q4_0 (18 B / 32 = 4.5), Q8_0 (34 B / 32 = 8.5), Q4_K (144 B / 256 = 4.5), Q6_K (210 B / 256 = 6.5625); MXFP4 (4 + 8/32 = 4.25); NVFP4 (4 + 8/16 = 4.5, plus one FP32 per tensor). Cell: pack a real weight matrix with `pack_int4` and assert `8·(nbytes(packed)+nbytes(scales))/numel == effective_bits(...)`.
4. **Weight-only vs weight-plus-activation.**
   - Naming: W8A16, W4A16, W8A8 (INT and FP8), W4A8 (one sentence on QServe, arXiv 2405.04532), W4A4 (11.4).
   - The kernels that make each fast: (a) *dequantize inside the GEMM* (tinygemm, Marlin): bytes shrink, arithmetic stays BF16; (b) *integer GEMM with epilogue scaling* (derivation D2); (c) *FP8 GEMM* (Ada, Hopper, Blackwell). Which of these run on A100 vs Ada vs Blackwell: the table in the chapter overview, repeated as `tab_lm_q_hw`.
   - Activations: dynamic per-token scales vs static per-tensor scales from calibration; the observation that a per-input-channel activation scale cannot be applied in an integer epilogue, which is why outlier channels need 11.2.
5. **The roofline for quantized layers** (derivation D1). Cell: `speedup_model(B, fmt)` for BF16, W8A16, W4A16 and W8A8, evaluated with `tab_gpu_specs` values; prints the breakpoints `B*` of each format and the KV-cache crossover (D1 step 7).
6. **Measured layer speed (headline, part A).** `fig_lm_q_layer_speed` for the Qwen3-1.7B MLP shapes (2048→6144 and 6144→2048) with real kernels, against the model of D1.
7. **Measured model quality and speed (headline, part B).** The group-size sweep on Qwen3-1.7B, `tab_lm_q_sweep`.
8. **GGUF-style block formats.** Implement Q8_0 and Q4_0 reference quantizers (10 lines each); check against the bytes of one tensor of `Qwen3-0.6B-Q8_0.gguf` read with `gguf.GGUFReader`; k-quants as a two-level scheme (super-block of 256 with 6-bit sub-scales) (the block and type sizes behind `tab_lm_q_gguf` are read from the `gguf` package's `GGML_QUANT_SIZES` table and asserted equal to the byte counts quoted above, which were written from memory of llama.cpp's layouts), and why `Q4_K_M` is a *mix* (some tensors at Q6_K): its effective bits are file bytes ÷ parameters, read from Hub file sizes for the 8B variants. Local inference trades kernel generality for portability (CPU, Metal), not necessarily peak speed on a datacenter GPU.
9. **Bits per parameter vs parameter count at a fixed memory budget.** The iso-memory frontier `fig_lm_q_pareto`: Qwen3-{0.6B, 1.7B, 4B, 8B} × {BF16, INT8, INT4 g128}, x = weight GiB, y = WikiText-2 perplexity. Question: at 4 GB of weights, is the 1.7B in BF16, the 4B in INT8 or the 8B in INT4 best? Link to Dettmers and Zettlemoyer's k-bit scaling laws (4-bit is almost universally optimal at fixed total bits) and say what the measured points do and do not show (one model family, one metric).

#### Derivations

**D1. Speed of a quantized linear layer (decode, prefill and the crossover).**

*Statement.* Let `W ∈ R^{m×n}`, input `X ∈ R^{B×n}`. A format `f` stores weights at `w_f` bytes per parameter (including scales), moves activations at `a_f` bytes per element, and runs its arithmetic at peak `π_f` (OP/s) on the unit it uses. Then `t_f(B) ≥ max( (w_f·mn + a_f·B(m+n)) / β , 2Bmn / π_f )`.

*Assumptions.* The layer is the only traffic (no KV cache); weights are read once per call; kernels reach the roofline (so the model gives an upper bound on speedup); dequantization is hidden behind memory time (modelled by a measured factor `κ ≥ 1` multiplying the compute term for weight-only formats).

*Steps.*
1. Bytes `= w_f·mn + a_f·B(m+n)`; FLOPs `= 2Bmn` (independent of format).
2. Activation traffic over weight traffic is `(a_f/w_f)·B·(1/m + 1/n)`. For `m = n = 2048`, `a_f = w_f`, and `B = 64` this is 6%, so for `B ≪ min(m, n)` the memory term is `w_f·mn/β`.
3. Ridge batch from equating the two terms: `B*_f = π_f·w_f / (2β)`.
4. Memory-bound speedup against BF16 (`w = 2`): `S = 2/w_f = 16/b_eff`. For W4A16 g128 with FP16 scales, `b_eff = 4.125`, so `S = 3.88`; with a zero point, 3.76.
5. Weight-only formats compute at the BF16 rate (`π_f = π`) so `B*_f = π·w_f/(2β)`: with `π = 165 TF` and `β = 1.0 TB/s` (the `tab_gpu_specs` row) this is 165 for BF16, 82 for W8A16 and about 44 for W4A16 (`w = 0.53`). For `B*_f < B < B*_bf16` the weight-only layer is compute-bound while BF16 is still memory-bound and `S = π/(Bβ)`; for `B > B*_bf16` `S → 1` (and below 1 with dequantization overhead).
6. W8A8 with integer tensor cores has `π_f = ρπ` with `ρ = π_int8/π_bf16` (A100: 624/312 = 2 from the datasheet; Ada: measured in the notebook, see below). Then `B*_f = ρπ/(2β)`, which equals `B*_bf16` when `ρ = 2`: the speedup is `2×` at *every* batch (bytes halve below the ridge, arithmetic doubles above it). In the memory-bound regime it is identical to W8A16: activation quantization adds nothing to decode at small batch, and above `B*_W8A16` (82 here) it is what keeps the speedup from decaying.
7. Whole-model decode. With context `L` the bytes per token are `W_bytes + b·L·k_KV`, where `k_KV = 2·layers·kv_heads·head_dim·bytes` (Qwen3-1.7B, BF16: 2·28·8·128·2 = 114,688 B per token per sequence). Weight quantization shrinks only the first term; the head and embedding (0.311 B parameters, 0.62 GB in BF16) are usually left at 16 bits, so the ceiling with a 4.25-bit body is `(2·1.72)/(0.531·1.41 + 2·0.311) = 3.44/1.37 = 2.5×`, not the 3.8× of the body alone. Past `L ≈ 1.37 GB / k_KV ≈ 12,000` tokens (batch 1, W4 body) the KV cache outweighs the weights: this is the hand-off to 12.4.

*Numerical check (cell).* `speedup_model` evaluated at `B ∈ {1, 16, 44, 82, 165, 512}` for each format; assert (i) `S(1) = 2/w_f` for every format, (ii) W8A8 with `ρ = 2` equals 2 at every `B`, (iii) weight-only `S(B) = 165/B` between `B*_f` and 165 (to relative 1e-12), (iv) the model never exceeds the bytes ratio. Then compare to measurements in the headline (the measured curve may fall below the model, never above, beyond timing noise).

**D2. Integer GEMM with epilogue scaling.**

*Statement.* With `Xq = round(X / s_x)` per row (`s_x ∈ R^B`) and `Wq = round(W / s_w)` per output row (`s_w ∈ R^m`), `Y_fq = (s_x ⊗ s_w) ⊙ (Xq Wqᵀ)`, where the integer product is exact in int32 if `n·127² < 2³¹`, i.e. `n ≤ 133,144`.

*Steps.* (1) The fake-quantized product is `Σ_k (s_x[i] Xq_ik)(s_w[j] Wq_jk)`. (2) Row and output-channel scales do not depend on `k`, so they factor out of the sum. (3) Hence the sum is an integer GEMM followed by one multiply per output element. (4) A per-input-channel scale `c_k` on `X` would sit inside the sum and cannot be factored: it must be folded into `W` beforehand, which is exactly the SmoothQuant move of 11.2.

*Numerical check (cell).* Random `X ∈ R^{64×512}` with one outlier channel, `W ∈ R^{256×512}`, seed 0: `W8A8Linear` output vs `fake_quant` product in float64: relative error ≤ 1e-6. Repeat with a per-channel activation scale applied *inside* the fake-quant: the integer path now disagrees at O(1), demonstrating step 4.

#### From-scratch implementation

Total budget ≈ 140 lines.

| Component | Signature | Lines |
|---|---|---|
| `fake_quant` | above | 25 (grouping via `reshape`, scale from `amax` or from `amax−amin` for asymmetric, clip search helper) |
| `pack_int4`/`unpack_int4`, `effective_bits` | above | 15 |
| `W8A8Linear` | above | 25 |
| `replace_linears` | above | 15 |
| `window_nll`, `calib_tokens` | above | 30 |
| `decode_tps` | above | 30 (static cache from the 1.1/12.1-agnostic HF `StaticCache`, `torch.compile(mode='reduce-overhead')`; fallback eager) |
| Q8_0/Q4_0 reference quantizers | `q8_0(x) -> (d: fp16[n/32], q: int8[n])`, `q4_0` likewise | 20 (not exported) |

Numerical tests (seed 0, CPU unless stated):
1. `fake_quant(W, 4, 64, sym=True)` equals `d2l.dequantize(d2l.quantize_blockwise(W, 4, 64), dtype=W.dtype)` bit for bit (symmetric case) on a `[256, 512]` normal matrix.
2. `unpack_int4(pack_int4(c)) == c` for all 16 values.
3. The packed-bytes identity of step 3 of the narrative to 1e-12.
4. `W8A8Linear` vs `fake_quant` (D2) rel. error ≤ 1e-6 in float64.
5. `window_nll` on a 2-layer random `d2l.GPT` equals a hand-computed cross-entropy over the same hand-sliced window to 1e-6.
6. (CUDA) torchao `Int4WeightOnlyConfig(group_size=128)` linear vs our asymmetric `fake_quant(W, 4, 128, sym=False)` applied to the same bf16 weights: relative output error ≤ 2e-2 of the output RMS (the tolerance is bf16 accumulation plus the kernel's zero-point convention; tighten after measuring).
7. Q8_0 bytes: `q8_0(dequantize_q8_0(tensor)) ==` the original GGUF block bytes (fp16 scale and int8 payload) for one real tensor of `Qwen3-0.6B-Q8_0.gguf` (idempotence of the format).

#### In production

- torchao: `quantize_(model, config)` with `Int8WeightOnlyConfig()`, `Int4WeightOnlyConfig(group_size=g)` (bf16 only, tinygemm kernel; groups 32, 64, 128, 256), `Int8DynamicActivationInt8WeightConfig()`, and on Ada `Float8DynamicActivationFloat8WeightConfig()`; all followed by `torch.compile`. For the 8B models call `quantize_` with `device='cuda'` on a CPU-loaded model so the BF16 weights never live on the GPU together.
- The from-scratch code is the oracle: tests 4 and 6 above, and the model-level check that our fake-quant perplexity agrees with the torchao-kernel perplexity within the paired-bootstrap interval of their difference (the two paths should differ only by BF16 accumulation).
- Hugging Face loading of pre-quantized checkpoints (FP8, GPTQ-Int8, AWQ) as an existence proof; vLLM/Marlin and llm-compressor are named as the serving path (13.x), not run here.

#### Headline experiment

*Hypothesis.* (A) For decode-sized batches a weight-only 4-bit layer is faster than BF16 by a factor between 1 and the bytes ratio `16/b_eff`, falling toward 1 as `B` approaches the ridge batch of D1; W8A8 is flat in `B` and bounded by `ρ`. (B) Perplexity increases with group size and decreases with bits; INT8 is indistinguishable from BF16; end-to-end decode speedup is below the layer speedup because the head, embedding, attention and non-GEMM kernels are unquantized.

*Setup.* Part A: GEMMs with the Qwen3-1.7B MLP shapes, `B ∈ {1, 2, 4, …, 512}`, formats BF16, INT8 weight-only (per-channel), INT4 g128 (tinygemm), INT8 W8A8 (dynamic per-token), FP8 W8A8 (Ada-native; on an A100 this row is skipped and the notebook says why). Part B: Qwen3-1.7B, configs BF16; INT8-WO; INT4-WO with `g ∈ {32, 64, 128, 256}` (RTN, torchao kernels); W8A8-INT; W8A8-FP8; plus the fake-quant counterparts of INT4 for the perplexity cross-check. Evaluation: WikiText-2 test, 2048-token non-overlapping windows; `d2l.EvalSuite(['arc_challenge', 'hellaswag', 'arc_easy', 'piqa'], n=500, seed=0)` (task ids from `d2l.EVAL_TASKS`; 500 examples each); prefill tokens/s at 2048 tokens; decode tokens/s at batch 1 with `ctx=512`. Seeds: no training; timing repeated 7 times with formats interleaved in ABAB order.

*Procedure.* Measure the copy bandwidth `β_meas` (a large device-to-device copy) and `ρ_meas` (8192³ `torch._int_mm` vs BF16 `matmul`) first and use them, not datasheet values, in the model overlay. Then run Part A, then Part B via `d2l.compression_row`.

*Metrics.* Layer: median time and relative IQR over 7 repeats; speedup vs BF16. Model: perplexity with 95% bootstrap interval over windows; paired difference to BF16 via `d2l.paired_bootstrap`; task scores with paired intervals; decode and prefill tokens/s as median and IQR; peak memory; `b_eff`; percent of the decode roofline.

*Plots.*
- `fig_lm_q_layer_speed`: x = `B` (log₂), y = speedup vs BF16 (log); one line per format; dashed model curves from D1 with measured `β` and `ρ`; vertical ticks at each `B*_f`.
- `fig_lm_q_pareto`: x = weight GiB (log), y = perplexity; markers by bit width, color by model size; the iso-memory vertical line at 4 GB.
- `fig_lm_q_decode`: bars of batch-1 decode tokens/s per config with the D1-step-7 ceiling as a marker.

*Expected outcome.* Directions only, until measured. W4A16's layer speedup is largest at `B = 1` and decays after roughly `B*`; the MARLIN paper reports close to the maximum 4× quantization speedup up to batch sizes 16–32 and gradually decreasing gains up to 64–128 on its hardware (arXiv 2408.11743), a useful shape to compare against, not a number to reproduce on Ada. "Give Me BF16 or Give Me Death?" (arXiv 2411.02355) reports FP8 W8A8 as effectively lossless, INT8 W8A8 with 1–3% accuracy degradation, and INT4 weight-only as competitive with 8-bit, on Llama-3.1; Qwen3 is a check of whether that carries over (to be measured). The perplexity gap between `g=32` and `g=256` is to be measured; the paired interval decides whether adjacent group sizes differ.

*Noise check.* Perplexity differences are reported only through the paired bootstrap over the same windows. Timing differences below the larger of the two relative IQRs are reported as ties. Part B's decode ordering is accepted only if it holds in all 7 interleaved repeats.

*Artifacts committed.* `data/compression_report.jsonl` rows for each config; `layer_speed.csv`; the three figures.

#### Figures and tables

- `fig_lm_q_granularity` (T0): one Qwen3 weight row quantized per-tensor, per-channel, g128, g32, drawn as histograms of the error; plus a Qwen3 activation row (per-token vs per-tensor).
- `tab_lm_q_vs_qlora`, `tab_lm_q_effbits`, `tab_lm_q_hw`: as above.
- `fig_lm_q_roofline_layer`: the D1 curves for the four formats at `B ∈ [1, 1024]` (made by `speedup_model`).
- `fig_lm_q_layer_speed`, `fig_lm_q_pareto`, `fig_lm_q_decode`, `tab_lm_q_sweep`: headline.
- `tab_lm_q_gguf`: Q4_0, Q8_0, Q4_K, Q5_K, Q6_K bits per weight from layouts; for the 8B GGUF variants, file GB ÷ parameters.

#### Pitfalls and misconceptions

- *Fake quantization is not faster*; it saves nothing unless the packed tensor is what the kernel reads.
- Forgetting scales in the byte count: g32 INT4 is 4.5 bits, not 4. A "4-bit model" file is rarely 4 bits per parameter (head and embedding stay 16-bit).
- Quantizing the head and embedding: it hurts quality disproportionately and saves less than the body; the standard choice is to skip `lm_head` and embeddings.
- Measuring decode in eager mode: Python and launch overhead hide the memory ratio at batch 1. Use CUDA graphs (`sec_compilation`).
- Comparing perplexities across different tokenizers, window sizes or strides; or reporting perplexity with a different context than the baseline.
- `torch._int_mm` has shape restrictions (to be verified for the pinned version, a minimum row count for `M` and multiples of 8 for `K` and `N`); the decode path pads or falls back, which changes the timing.
- W8A8 does not make decode faster than W8A16 at small batch (D1 step 6); its benefit is prefill and large-batch serving.
- Treating the datasheet ridge as exact: measured `β` and `ρ` are lower.

#### Summary (draft bullets)

- Inference quantization is judged by bytes moved and arithmetic rate, not only bytes stored; `b_eff` must include scales and zero points.
- Weight-only quantization speeds up decode by up to `16/b_eff` until the batch reaches `B* = π·w/(2β)`; W8A8 gives up to `ρ` at every batch and is what helps prefill and large-batch serving.
- A per-input-channel activation scale cannot be applied in an integer epilogue; this is the problem of 11.2.
- Real speedups need kernels that read the packed format; the head, embedding and KV cache set the whole-model ceiling.
- At a fixed memory budget, a larger model in 4 bits often beats a smaller one in 16; the measurements here confirm or contradict this on one family.

#### Exercises

1. *(check)* A model has 8 B parameters in BF16 on a 3.35 TB/s GPU. Predict the batch-1 decode ceiling with BF16, W8A16 and W4A16 g64 (FP16 scales). Which ceiling changes if the KV cache holds 20 GB? *Sketch:* bytes per token 16 GB, 8 GB, 4.25 GB → 209, 419, 788 tokens/s; the KV term adds 20 GB per token only at batch 1 if the whole cache is read; with it the three ceilings fall to 93, 120, 138 tokens/s (all three compress toward the KV-bound).
2. *(check)* Explain why W8A8 with per-tensor activation scales is "free" in the epilogue but W8A8 with per-input-channel activation scales is not. *Sketch:* D2 step 4.
3. *(derive)* Show that for a weight-only format `B*_f = π·w_f/(2β)` and that the speedup between `B*_f` and `B*_bf16` is `π/(Bβ)`. Then find the `B` where W4A16's advantage over W8A8 (`ρ = 2`) vanishes. *Sketch:* equate `π/(Bβ)` with 2 → `B = π/(2β)`; with `π/β = 165`, `B ≈ 82`.
4. *(derive)* Prove the overflow bound `n ≤ 133,144` for int32 accumulation of int8 products and give the bound for symmetric INT4 weights times INT8 activations. *Sketch:* `127·127 = 16,129`, `2³¹/16,129 = 133,144`; with `|w| ≤ 7`, `2³¹/(7·127) ≈ 2.4 M`.
5. *(measure)* Predict, before running, at which batch the INT4 g128 kernel loses to BF16 on your GPU, using your measured `β` and `π`; then run the Part A sweep to `B = 2048` and compare. Report the error of your prediction and explain any discrepancy (dequantization overhead `κ`).
6. *(measure)* Repeat the headline with `lm_head` quantized to INT8 and then INT4. Predict the decode speedup change from D1 step 7 and the perplexity change; measure both with paired intervals.
7. *(open)* Pick a memory budget (for example 6 GB of weights, 8 GB for a local laptop) and find the best Qwen3 checkpoint, format and group size on WikiText-2 and one downstream task. Write a half-page decision memo that states what you did not measure (long context, batch > 1, other tasks).

#### Further reading

- Jacob et al., [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877): the affine scheme and the integer-only matmul with a fixed-point epilogue (D2).
- Dettmers and Zettlemoyer, [The case for 4-bit precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720): bits vs parameters at a fixed total; use for the iso-memory figure.
- Frantar et al., [MARLIN](https://arxiv.org/abs/2408.11743) with code at https://github.com/IST-DASLab/marlin: why a packed 4-bit format needs a specialized kernel and how far batch size can grow.
- Kurtic et al., ["Give Me BF16 or Give Me Death"?](https://arxiv.org/abs/2411.02355): the accuracy–performance trade-offs of FP8, INT8 and INT4 on a model family.
- [torchao](https://github.com/pytorch/ao) and its documentation https://docs.pytorch.org/ao/stable/index.html; [llama.cpp](https://github.com/ggml-org/llama.cpp) and the k-quants PR https://github.com/ggml-org/llama.cpp/pull/1684; [llm-compressor](https://github.com/vllm-project/llm-compressor).
- Teaching: MIT 6.5940 quantization lectures; Grootendorst's visual guide; Smola, [Efficiency in LLMs](https://alex.smola.org/posts/45-mlss-efficiency/).

#### Risks and open decisions

- *torchao API churn* (class configs vs function names; int4 layout changes). Fallback: pin the version, keep the layer-level tests 4 and 6, and gate the model-level rows on `try/except`.
- *`torch._int_mm` or FP8 kernels unavailable on the pinned stack.* Fallback: report fake-quant perplexity and mark speed `n/a`; FP8 is skipped on A100 by design.
- *Decode graph capture fails for quantized tensor subclasses.* Fallback: report the layer microbenchmark and an eager-mode decode with the overhead stated.
- *`EvalSuite` unavailable or slow on the build box.* Task ids are fixed by `d2l.EVAL_TASKS` (resolved). Fallback: perplexity-only rows, with the downstream columns filled when the suite runs.
- *8B on a 15 GiB slot.* BF16 8B (16.4 GB) does not fit a slot; the notebook quantizes on load from CPU and requires ≥ 20 GB of host RAM. On a whole 24 GB GPU it loads in BF16 directly. Fallback: drop the 8B row.
- *Pre-quantized checkpoints need packages (autoawq successors, gptqmodel) that may not install.* They are existence checks only.

### 11.2 Outliers and equivalent rescaling

- **Label:** `sec_lm_outliers` · **Tier:** T1 (`FAST=True`: 16 calibration sequences, 40 evaluation windows) · **Budget:** ≤ 12 min, peak ≤ 9 GiB (one forward pass with hooks on 128 × 2048 tokens, statistics kept as `[d_in]` vectors only) · **Codex units:** 12.3
- **Builds on (book):**
  - `sec_transformer-block`: RMSNorm with learnable `γ` (where scales are absorbed) and the SwiGLU product (why the `up_proj` rows can absorb a `down_proj` input scale and `gate_proj` cannot).
  - `sec_kv-cache`: GQA head layout (a `v_proj` output channel feeds every query head of its group, which constrains the `o_proj` rescaling).
  - `sec_gpt`: the module tree of `d2l.GPT` / the Hugging Face Qwen3 model.
  - `sec_numerics`: not re-taught.
- **Builds on (this part):** 11.1 (`d2l.fake_quant`, `d2l.W8A8Linear`, `d2l.replace_linears`, `d2l.window_nll`, `d2l.calib_tokens`, `d2l.compression_row`, the notation of the overview), 1.5 (`d2l.paired_bootstrap`).
- **Exports** (new names):
  - `d2l.collect_input_stats(model, ids, sites=('qkv','o','gateup','down'))`: forward hooks on the inputs of the linears at each site; returns `{(layer, site): {'absmax': [d_in], 'rms': [d_in], 'mean_abs': [d_in], 'tok_frac': [d_in]}}`, where `tok_frac[j]` is the fraction of tokens with `|x_j| > 6·median_j(|x|)`. Streaming (running max / sum of squares), never stores activations.
  - `d2l.SimQuantLinear(linear, w_bits=None, w_group=None, a_bits=None, a_gran='token', a_scale=None)`: fake-quantizes the weight once at construction and the input at every call. `a_gran ∈ {'token', 'tensor_dynamic', 'tensor_static'}`; `a_scale` is the calibrated static scale. Used again by 11.4.
  - `d2l.smooth_scales(act_absmax, w_absmax, alpha=0.5, eps=1e-5)`: `s_j = a_j^α / w_j^{1−α}`, clamped below by `eps`.
  - `d2l.apply_smoothing(model, scales)`: fuses `1/s` into the producer and `s` into the consumer weight columns at the four sites (table in the narrative); verifies float64 equivalence if a `check` batch is passed.
  - `d2l.awq_search(x, W, bits=4, group_size=128, n_grid=20)`: returns `(s, alpha_best, errs)` with `s_j = (mean|x_j|)^α` normalized to `√(max·min) = 1`, scored by `‖xWᵀ − x(Q(W·diag s)/s)ᵀ‖²` on calibration rows.
  - `d2l.llm_int8_matmul(X, W, threshold=6.0)`: vector-wise INT8 for non-outlier columns plus a BF16 matmul on the outlier columns.
- **Models and data:** `Qwen/Qwen3-1.7B` (main); `Qwen/Qwen3-0.6B` (T0-sized dry runs); optional `Qwen/Qwen3.5-2B` for the open exercise; calibration and evaluation as in 11.1; the `Qwen/Qwen3-8B-AWQ` `config.json` fetched as text (4-bit, group size 128, zero point, version `gemm`, verified) with no weight download.
- **Packages:** as 11.1; optional `llmcompressor` (AWQ/SmoothQuant modifiers) for the production cross-check.

#### Narrative arc

1. **The question.** *Why does 8-bit activation quantization hurt a model that tolerates 8-bit weights?* The opening cell runs Qwen3-1.7B at W8A16 (per-channel INT8 weights), then at W8A8 with a per-tensor static activation scale from calibration absmax (via `SimQuantLinear`), and prints both perplexities next to BF16. The expected symptom is a large gap for per-tensor activations that the W8A16 row does not show (direction to be measured). A second cell prints, for one `down_proj` input, the largest and the median per-channel absmax and their ratio.
2. **Where the outliers live** (stats pass, `fig_lm_out_channels`, `tab_lm_out_sites`).
   - Four sites per block: input of `q/k/v` (post-RMSNorm), input of `o_proj`, input of `gate/up` (post-RMSNorm), input of `down_proj`.
   - For layers 0, 14, 27: sorted per-channel absmax on a log axis, plus `max/median`, kurtosis, and the share of channels above 6× median.
   - Two phenomena, separated by `tok_frac`: *systematic channel outliers* (the same channel is large for most tokens; fixable by per-channel scaling) and *massive activations* (huge values at a few tokens and a few dimensions of the residual stream, described by arXiv 2402.17762; a per-token scale handles them, a per-tensor scale does not). Report which dominates at each site, to be measured.
   - Whether `γ` of the RMSNorm moves outliers: compare the residual stream before the norm and the input after it.
3. **Why a per-tensor scale fails** (T0 synthetic, `fig_lm_out_snr`): `X ∈ R^{2048×512}` Gaussian with 4 channels multiplied by 50, `W` Gaussian; INT8 per-tensor vs per-token activations; relative output error against outlier magnitude. Connect to 11.1's D2: only scales that do not depend on the reduction index `k` factor out of an integer GEMM.
4. **LLM.int8(): decomposition.** Vector-wise scales (per-row `X`, per-column `W`) plus mixed precision: columns of `X` with `|x| > 6` go through a BF16 matmul. Measure the fraction of outlier columns per site at threshold 6 on the stats pass. Compare error against vector-wise INT8 alone on the worst layer. Note what it costs: two kernels and a gather per layer; use as an accuracy baseline, not a speed recipe.
5. **SmoothQuant: move the difficulty into the weights** (derivation D1). Where scales can be absorbed in Qwen3 (`tab_lm_out_fuse`):

   | Site (input of) | Producer that absorbs `1/s` | Consumer that takes `s` | Constraint |
   |---|---|---|---|
   | `q_proj, k_proj, v_proj` | `input_layernorm.weight` | the three weight matrices | one `s` shared by the three |
   | `gate_proj, up_proj` | `post_attention_layernorm.weight` | both | one `s` for both |
   | `o_proj` | `v_proj` output rows | `o_proj` columns | GQA: one `s` per (KV head, dim) shared by the 2 query heads of the group (take the max of their `a_j`) |
   | `down_proj` | `up_proj` output rows (not `gate`, because of the SiLU) | `down_proj` columns | none |

   QK-norm acts after the projections, so it is unaffected by the first row. Three activation granularities (O3 per-tensor static, O2 per-tensor dynamic, O1 per-token dynamic in the SmoothQuant paper's naming).
6. **AWQ: protect the salient weights, weight-only** (derivation D2). Salience is read from activations, not from weights. Implement `awq_search`, apply with `apply_smoothing` (the same fusion sites), and show the error-vs-α curve for one layer (`fig_lm_out_alpha`, U-shaped).
7. **Head-to-head** (headline). RTN, SmoothQuant and AWQ at W8A8 (three activation granularities) and W4 (per-channel and g128); LLM.int8-style as an accuracy reference at W8A8.
8. **Production view.** What an AWQ checkpoint contains (`qweight`, `qzeros`, `scales` from the `Qwen3-8B-AWQ` config and tensor names); `llm-compressor` recipes expressing the same two transforms.

#### Derivations

**D1. The rescaling identity and the balanced scale (SmoothQuant).**

*Statement.* For any `s ∈ R^{d_in}_{>0}`, `X Wᵀ = (X diag(s)⁻¹)(W diag(s))ᵀ`. Define `a_j = max_t |X_tj|`, `w_j = max_i |W_ij|`, and after rescaling `a'_j = a_j/s_j`, `w'_j = w_j s_j`. Then `a'_j w'_j = a_j w_j =: p_j` for every `s`, and `max(a'_j, w'_j) ≥ √p_j` with equality iff `s_j = √(a_j/w_j)`.

*Assumptions.* The step size of per-tensor activation quantization is set by `max_j a'_j`; the weight step of an output row is set by the largest `|W_ij| s_j` along that row; both are proxied by the per-input-channel maxima `a'_j, w'_j`.

*Steps.*
1. `(X diag(s)⁻¹)(W diag(s))ᵀ = X diag(s)⁻¹ diag(s) Wᵀ = X Wᵀ`, since diagonal matrices commute and `s_j > 0`.
2. Per channel, `a'_j w'_j = (a_j/s_j)(w_j s_j) = a_j w_j`: rescaling only redistributes a fixed budget `p_j` between the two operands.
3. For fixed `p_j`, the larger of the two factors is minimized when they are equal, by AM–GM/`max(u,v) ≥ √(uv)`: `s_j* = √(a_j/w_j)`, both equal to `√p_j`.
4. With migration strength `α`, `s_j = a_j^α / w_j^{1−α}` gives `a'_j = p_j^{1−α}` and `w'_j = p_j^{α}`: `α = 0.5` equalizes; `α → 1` makes every activation channel range 1 (flat) and moves the whole budget `p_j` to the weights, and `α → 0` leaves the activations untouched.
5. Consequence: the per-tensor activation range after smoothing is `max_j p_j^{1−α}`, which for outlier channels (`a_j ≫ w_j`) is much smaller than `max_j a_j`, while the weights' per-channel ranges grow only by `p_j^α/w_j`.

*Numerical check (cell).* (i) float64 identity: max relative error ≤ 1e-12. (ii) `a'_j w'_j = a_j w_j` to 1e-12. (iii) `α = 0.5` gives `a'_j = w'_j` to 1e-12. (iv) On the synthetic problem (seed 0, `T=2048`, `d_in=512`, `d_out=256`, 4 channels ×50, `W ~ N(0, 0.05²)`): INT8 per-tensor-activation output relative MSE over `α ∈ {0, .25, .5, .75, 1}` has its minimum at an interior `α`, and `MSE(0.5) < MSE(0)/5`. (A CPU dry run of this exact setup gave a factor of about 40 and an interior minimum; assert the weaker bound.)

**D2. Why scaling salient channels up helps weight-only quantization (AWQ).**

*Statement.* With group-wise symmetric absmax quantization of `W` at `b` bits, group step `Δ = max_{j∈group}|w_j| / (2^{b−1}−1)`, the expected squared output error from channel `j` is `x_j² Δ²/12` (the Δ²/12 of 7.6, with independent uniform errors). Scaling channel `j` by `s_j ≥ 1` before quantization and dividing the activation by `s_j` after changes it to `(x_j/s_j)² Δ'²/12`, with `Δ' = Δ` as long as `s_j |w_j| ≤ max` of the group.

*Steps.*
1. `δy = Σ_j x_j δw_j`, with `δw_j` independent, zero mean, variance `Δ²/12`; so `E δy² = (Δ²/12) Σ_j x_j²`.
2. Scale-then-quantize-then-unscale: `Ŵ_j = Q(w_j s_j)/s_j`, error `δw_j/s_j` in the original units, hence the channel's contribution is `(x_j²/s_j²)(Δ'²/12)`.
3. While `s_j|w_j| ≤ max|w|` of the group, `Δ' = Δ` and the contribution falls as `1/s_j²`; once `s_j|w_j|` exceeds the group maximum, `Δ' = s_j|w_j|/(2^{b−1}−1)` grows linearly in `s_j` and *all other channels in the group* pay `Δ'²/12 · x_k²`.
4. So the total error `E(s) = (Δ'(s)²/12)·Σ_j x_j²/s_j²` has an interior minimum; the best `s_j` increases with `x_j` (salience) but saturates near `max|w| / |w_j|`. A one-parameter family `s_j = (mean|x_j|)^α`, normalized, with a grid search for `α` approximates it; no gradient is needed.

*Numerical check (cell).* One group of 128 weights, two channels with `|x_j|` 30× the rest, seed 0: Monte-Carlo output error (2000 random rows) vs `s ∈ {1, 2, 4, 8, 16}` on the salient channels has an interior minimum and matches the closed form `E(s)` of step 4 within 15% (tolerance to be tightened after measuring). Then the real layer: `awq_search`'s best `α` gives lower layer-output error than `α = 0` on a held-out half of the calibration rows. (The dry run on the synthetic stand-in of the previous check gave an interior minimum at `α = 0.25` and about a 5× error reduction.)

#### From-scratch implementation

Budget ≈ 130 lines.

| Component | Lines |
|---|---|
| `collect_input_stats` (hooks, running max, sum of squares, `tok_frac` needs a first pass for the median: use the median of a 16-sequence subsample) | 30 |
| `SimQuantLinear` (reuses `fake_quant`; static scale from `a_scale`) | 25 |
| `smooth_scales`, `apply_smoothing` for the four sites incl. the GQA group max | 40 |
| `awq_search` (grid, `fake_quant` with `sym=False`, error on calibration rows) | 25 |
| `llm_int8_matmul` | 15 |

Numerical tests:
1. After `apply_smoothing` the Qwen3-0.6B logits equal the original within `atol=1e-3, rtol=1e-4` in float32 on 4 sequences (seed 0) and are bit-near in float64 on a 2-layer random `d2l.GPT` (≤ 1e-10).
2. `SimQuantLinear(a_bits=8, a_gran='token')` reproduces `W8A8Linear` (11.1) within 1e-6 relative.
3. `awq_search` with `α` fixed to 0 returns `s ≡ 1` and the same error as RTN.
4. `llm_int8_matmul` with `threshold=∞` equals vector-wise INT8; with `threshold=0` equals the BF16 matmul (to 1e-3 relative in bf16).
5. `collect_input_stats` absmax equals the absmax from a non-streaming pass on one 2048-token sequence exactly.

#### In production

- `llm-compressor` (`SmoothQuantModifier`, `AWQModifier`, `GPTQModifier` recipes, written against Hugging Face models; output loads in vLLM) and the original repos https://github.com/mit-han-lab/smoothquant and https://github.com/mit-han-lab/llm-awq. The from-scratch `apply_smoothing` is the test oracle: the production-quantized and from-scratch-quantized Qwen3-1.7B should agree in perplexity within the paired interval (optional cell, skipped if `llmcompressor` does not install on the pinned stack).
- Nothing here changes the inference kernels: after fusion the model is an ordinary W8A8 or W4A16 model, which is the whole point of an *equivalent* transformation.

#### Headline experiment

*Hypothesis.* (1) SmoothQuant's benefit is largest for the coarsest activation granularity (per-tensor static) and smallest for per-token dynamic quantization, where the per-token scale already absorbs token-level magnitude; (2) AWQ lowers W4 perplexity relative to RTN, more at per-channel than at g128 (smaller groups already isolate outliers).

*Setup.* Qwen3-1.7B BF16; calibration 128 × 2048 tokens (11.1 artifact); evaluation WikiText-2 test windows plus `EvalSuite` tasks. Configurations: (a) W8A8: weights INT8 per-channel; activations per-tensor static, per-tensor dynamic, per-token dynamic; methods RTN, SmoothQuant `α ∈ {0.5, 0.75}`, LLM.int8-style (reference); (b) W4A16: per-channel and g128; methods RTN, AWQ (grid 20), AWQ + clip search. Fake quantization throughout (simulated; no speed is claimed); the INT8 per-token path is cross-checked once against the real `W8A8Linear`.

*Procedure.* Stats pass once; fuse scales; quantize; evaluate each configuration with `window_nll`; `d2l.compression_row` for the best of each family.

*Metrics.* Perplexity with paired 95% interval against RTN of the same bit setting and against BF16; mean layer-output relative error on held-out calibration rows; task scores with paired intervals; `α` chosen.

*Plots.*
- `fig_lm_out_channels`: sorted channel absmax, 4 panels (sites), 3 layers per panel.
- `fig_lm_out_alpha`: layer-output error and model perplexity vs `α` for SmoothQuant (per-tensor static) and AWQ (g128).
- `tab_lm_out_methods`: rows = (bits, granularity, method), columns = ppl ± CI, Δ vs RTN with paired CI, layer error.

*Expected outcome.* Direction from the toy experiment (11.2 D1 check) and the papers: SmoothQuant reports that per-tensor static W8A8 can be recovered close to FP16 on models with strong outliers (arXiv 2211.10438, tables); AWQ reports gains over RTN at 3–4 bits (arXiv 2306.00978); how the gain depends on group size is hypothesis (2), to be measured. Whether Qwen3 follows either paper is to be measured; its QK-norm and RMSNorm layout may change which sites carry the outliers (this is why the stats pass comes first).

*Noise check.* Every perplexity gap is reported with a paired bootstrap over windows. The calibration draw is a second source of variance: repeat SmoothQuant (α = 0.5) and AWQ with a second 128-sequence draw (seed 1) and report both; claim an ordering only if it holds in both and the paired intervals exclude zero. The `α` grid is searched on 64 calibration sequences and scored on the other 64.

*Artifacts committed.* `lm-c11-act-stats-qwen3-1p7b`; best scales per method (`.npz`, ≈ 1 MB); rows in `data/compression_report.jsonl`.

#### Figures and tables

- `fig_lm_out_snr` (T0 synthetic), `fig_lm_out_channels`, `fig_lm_out_alpha`.
- `tab_lm_out_sites` (outlier statistics per site and layer), `tab_lm_out_fuse` (absorption sites), `tab_lm_out_methods`.
- `fig_lm_out_balance`: for one layer, per-channel `a_j`, `w_j` before and `a'_j`, `w'_j` after smoothing at `α = 0.5`, showing the two curves meeting (D1 step 3).

#### Pitfalls and misconceptions

- *Equivalent in exact arithmetic, not after rounding*: smoothing preserves the function in float, then re-distributes quantization error; always re-check logits with quantization off.
- Applying one `s` to `q/k/v` separately instead of one shared `s` (the shared norm output feeds all three) breaks equivalence.
- GQA: scaling `v_proj` channels per query head instead of per KV group silently changes the model.
- Calibrating on the evaluation text inflates AWQ/SmoothQuant gains; use a different source (FineWeb-Edu vs WikiText-2).
- Reading "outliers" from weights: weight outliers are mild; the problem is activation channels (and in weight-only AWQ, the *salience* of the matching weight channels).
- Using `α = 0.5` as a law: the best `α` varies by model and site; sweep it.
- LLM.int8's mixed-precision decomposition is an accuracy tool; its two-kernel structure is not a speed recipe.

#### Summary (draft bullets)

- Activation quantization is hard because a few channels carry most of the range, and a per-tensor or per-token scale cannot be applied per channel in an integer GEMM.
- `a_j w_j` is invariant under rescaling, so smoothing only redistributes range between activations and weights; the balanced choice is `s_j = √(a_j/w_j)`.
- Scales fuse into the preceding RMSNorm or linear layer, leaving an ordinary model; GQA and SwiGLU dictate where they can go.
- AWQ uses activation statistics to decide which weight channels deserve protection; it needs a grid search, not gradients.
- The gain of either method depends on the quantization granularity and must be measured with paired intervals and a calibration set different from the test text.

#### Exercises

1. *(check)* Which of the four sites in `tab_lm_out_fuse` could absorb a scale into the *previous normalization layer* and which need an adjacent linear layer? Why is scaling `gate_proj` output rows not equivalent for the `down_proj` input scale? *Sketch:* the SiLU is nonlinear, `silu(s·u) ≠ s·silu(u)`; only the linear branch `up` of the product commutes.
2. *(check)* With per-token dynamic INT8 activations, which tokens remain badly quantized when a massive activation appears in one dimension at one token, and which remain fine? *Sketch:* the scale is per row; only that token loses precision in its other channels (error ∝ its max); all other tokens are unaffected.
3. *(derive)* Show that for `α = 0.5` the SmoothQuant range of every channel is `√p_j`. Then prove that for every `s` the product of the per-tensor activation range `max_j a'_j` and the per-tensor weight range `max_j w'_j` is at least `max_j p_j`, and that `α = 0.5` attains `√(max_j p_j)` for both. *Sketch:* let `k = argmax p_j`; `max a' ≥ a'_k`, `max w' ≥ w'_k`, so the product is `≥ a'_k w'_k = p_k`; with `α = 0.5` every channel has `a'_j = w'_j = √p_j ≤ √p_k`. Smoothing cannot reduce the largest product; it only balances it.
4. *(derive)* In D2 take a group with one salient channel (`x_s`) and `d−1` others with equal `x` and equal `|w|` magnitudes `m` except `|w_s| = m/4`. Find the `s_s` that minimizes `E(s)` and the resulting error ratio to `s = 1`. *Sketch:* `Δ' = Δ` up to `s = 4`; the error from the salient channel is `x_s²/s²`; beyond 4 the others pay `(s/4)²`; the optimum is `s = 4` when `x_s²` dominates, giving ratio `(1/16)` on that term.
5. *(measure)* Sweep `α ∈ [0, 1]` in steps of 0.1 for SmoothQuant (per-tensor static) per site type (q/k/v, o, gate/up, down) and plot the best `α` per site. Predict which site wants the largest `α` from `fig_lm_out_channels` first.
6. *(measure)* Vary the AWQ calibration set size over {4, 16, 64, 128, 512} sequences and plot held-out layer error and perplexity; predict where it saturates. (This sets up the calibration discussion of 11.3.)
7. *(open)* Repeat the outlier survey on `Qwen/Qwen3.5-2B` (Gated DeltaNet layers) and on `openai/gpt-oss-20b`'s attention inputs. Do the recurrent or MoE layers have different outlier structure, and does the same fusion table apply?

#### Further reading

- Dettmers et al., [LLM.int8()](https://arxiv.org/abs/2208.07339): emergent outlier features and mixed-precision decomposition; code in [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes).
- Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438), code https://github.com/mit-han-lab/smoothquant: the identity and the choice of `α`; read the O1/O2/O3 table.
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978), code https://github.com/mit-han-lab/llm-awq: salience from activations, the scaling derivation, and the TinyChat kernels.
- Sun et al., [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762): token-specific massive activations, relevant to the per-token vs per-channel distinction.
- Kurtic et al., ["Give Me BF16 or Give Me Death"?](https://arxiv.org/abs/2411.02355): what SmoothQuant-style W8A8 costs in accuracy across a family.
- Teaching: MIT 6.5940 (SmoothQuant and AWQ lectures); [llm-compressor](https://github.com/vllm-project/llm-compressor) recipes.

#### Risks and open decisions

- *Qwen3's outliers may sit mainly in massive activations* rather than channels, making per-token quantization already adequate and SmoothQuant gains small. That is a legitimate result; the section's text is written around "measure the sites first", and the synthetic cell carries the mechanism.
- *Fusing into the Hugging Face module tree differs from `d2l.GPT`* (separate `q/k/v` vs fused `W_qkv`). Fallback: do the surgery on `d2l.GPT` (via `d2l.load_qwen3`) and use the Hugging Face model only for torchao timing; the chapter's tests are written on `d2l.GPT`.
- *`llmcompressor` version churn.* The cross-check is optional.
- *AWQ search cost.* 28 layers × 7 linears × 20 grid points on 128 × 2048 rows; sub-sample 32 sequences for the search (the held-out half uses the rest). If this exceeds the 12 min budget, reduce the grid to 10.

### 11.3 GPTQ and reconstruction-based quantization

- **Label:** `sec_lm_gptq` · **Tier:** T1 (`FAST=True`: 16 calibration sequences, first 8 blocks only for timing, 40 evaluation windows) · **Budget:** ≤ 15 min, peak ≤ 10 GiB (blocks quantized one at a time; block inputs for 128 × 2048 tokens held on the GPU in BF16 ≈ 1.1 GB; the largest Hessian, `down_proj`'s 6144², is 151 MB in fp32) · **Codex units:** 12.4
- **Builds on (book):**
  - `sec_mdl-numerical-stability-conditioning`: `κ(XᵀX) = κ(X)²` (`eq_mdl-opt-kappa-squared`), why `H` is ill-conditioned and what damping buys; backward vs forward error.
  - `subsec_mdl-psd` in `sec_mdl-eigendecompositions`: positive definiteness and the Cholesky factorization (used, not re-proved).
  - `sec_mdl-svd-low-rank`: `H` has rank ≤ number of calibration tokens (a rank-deficient `H` when tokens < `d_in`).
  - `sec_transformer-block`, `sec_gpt`: module order for sequential quantization.
- **Builds on (this part):** 11.1 (`d2l.fake_quant`, `d2l.pack_int4`, `d2l.replace_linears`, `d2l.window_nll`, `d2l.calib_tokens`, `d2l.compression_row`), 11.2 (RTN and AWQ baselines; the calibration-size discussion), 1.5 (`d2l.paired_bootstrap`).
- **Exports** (new names):
  - `d2l.accumulate_hessian(H, n, X)`: running update `H ← n/(n+t)·H + 2/(n+t)·XᵀX` for a batch `X ∈ R^{t×d_in}`; returns `(H, n+t)`; accumulation in float32 from BF16 inputs.
  - `d2l.gptq_quantize(W, H, bits=4, group_size=128, blocksize=128, damp=0.01, act_order=False, sym=True)`: returns `(Q, codes, scales, zeros, info)` where `Q` is the dequantized weight `[d_out, d_in]`, `codes` are integer codes in original column order, and `info` has per-row proxy losses `Σ_q err_q²` and the permutation. `d2l.QTensor(codes, scales, zeros, block_size=group_size, shape=W.shape, bits=bits)` (7.6) wraps the same codes, scales and zeros, so `d2l.dequantize(q)` reproduces `Q`; `info['qtensor']` carries it.
  - `d2l.quantize_model_gptq(model, calib_ids, bits=4, group_size=128, sequential=True, **gptq_kw)`: block-by-block quantization with Hessians collected from the already-quantized prefix of the model ("true sequential" inside a block: `q,k,v` → `o` → `gate,up` → `down`); returns per-linear info for plots.
  - `d2l.obs_step(w, Hinv_F, q, c)`: one OBS update (returns new `w` and the loss increment); pedagogical, used in the derivation checks and exercises.
  - `d2l.brute_force_quantize(w, H, grid)`: exhaustive search over `grid^n` for tiny `n` (≤ 8); the test oracle.
- **Models and data:** `Qwen/Qwen3-1.7B` (main); `Qwen/Qwen3-0.6B` (T0 dry runs and tests); calibration sources for the ablation: FineWeb-Edu (`sample-10BT`), WikiText-2 `train` (`wikitext-2-raw-v1`, CC BY-SA 3.0), and a uniform-random-token control; evaluation on WikiText-2 `test` and on a held-out FineWeb-Edu slice (documents not in the calibration draw). Published GPTQ artifact for reference: `Qwen/Qwen3-1.7B-GPTQ-Int8` (8-bit, group 128, `desc_act=False`, damp 0.01, produced by GPTQModel, `config.json` verified) read only for its `quantization_config`.
- **Packages:** as 11.1; optional `gptqmodel` (https://github.com/ModelCloud/GPTQModel) as the production cross-check; optional IST-DASLab `gptq` reference (https://github.com/IST-DASLab/gptq) as the line-by-line oracle.

#### Narrative arc

1. **The question.** *Nearest rounding is not the best rounding.* Two cells. (a) A 6-weight row with a correlated Hessian: print RTN's output loss `(w−ŵ)ᵀH(w−ŵ)` next to the true minimum over the grid (brute force), and the weights each picks. (b) The real model: Qwen3-1.7B at g128 with RTN: perplexity at 4 and 3 bits and the relative output error of layer 0's `down_proj` at 4, 3, 2 bits (the symptom is the rapid growth below 4 bits; values to be measured).
2. **Quantization as layer-wise reconstruction.** `eq_lm_gptq_obj`: `min_{Ŵ∈grid} ‖X Wᵀ − X Ŵᵀ‖_F² = Σ_rows (w−ŵ)ᵀ H (w−ŵ)`, `H = XᵀX`. Rows decouple, so `H` is shared by all rows of a layer. What `X` is: the layer's inputs on calibration data after the earlier layers have been quantized. Why the proxy loss (local) is not the model loss (global): the second is measured in the headline.
3. **The Hessian is hard** (`fig_lm_gptq_spectrum`, a T0 cell on real activations from a `Qwen3-0.6B` pass). Eigenvalue spectrum of `H` for `q_proj` and `down_proj` input: rank ≤ tokens (a single 2048-token sequence cannot fill a 6144-dimensional `H`), condition number `κ(H) = κ(X)²` with damping 0, 1e-4, 1e-2 (relative to the mean diagonal). Dead columns (zero diagonal). Why Cholesky is done in float32 or float64.
4. **Optimal Brain Surgeon and quantization** (derivation D1): quantize one weight, compensate the rest optimally; the cost; the greedy order of OBQ; GPTQ's observation that one fixed order for all rows costs little and lets all rows share `H⁻¹`.
5. **The Cholesky formulation** (derivation D2): one factorization of `H⁻¹` gives every downdated quantity; the update is a row of `U`. **Lazy block updates**: process 128 columns at a time, accumulate the scaled errors `E`, then update the remaining columns with one matmul; identical numbers, far fewer passes over the weight matrix. Show the pseudo-code and the exact correspondence with the equations.
6. **Groups, act-order and damping in practice.** Group scales are computed when the sweep reaches the group's first column, from the already-compensated weights; `act_order` sweeps columns by decreasing `diag(H)` (large-activation channels first, while compensation capacity is greatest) at the cost of non-contiguous groups unless group parameters are fixed up front (`static_groups`); automatic damping increase if the Cholesky fails (the published checkpoint's config records `damp_percent=0.01`, `damp_auto_increment=0.0025`, `true_sequential=True`). Damping as a ridge toward RTN: `H = I` gives RTN exactly (D2, check (vi)).
7. **How calibration data matters** (headline part 2): number of sequences, domain, and a random-token control; `H`'s effective rank per site vs number of tokens.
8. **GPTQ on Qwen3-1.7B** (headline part 1) and what GPTQ does *not* change: the packed format and kernel are the same as RTN's, so speed (11.1) is unchanged; only the choice of codes differs.
9. **Beyond rounding**: the same machinery prunes (SparseGPT, 11.6) and is combined with rotations (11.4).

#### Derivations

**D1. The OBS update and its cost.**

*Statement.* Let `L(w) = (w−w⁰)ᵀ H (w−w⁰)` with `H ≻ 0`, `w⁰` the original row. Suppose coordinates in `Q` are already fixed and the free set `F` (not yet quantized) holds its conditional optimum, i.e. `∇_F L = 0`. Fixing coordinate `q ∈ F` to the grid value `c` and re-optimizing the other free coordinates changes them by `δ = −(Δ / [H_F⁻¹]_qq) H_F⁻¹ e_q`, with `Δ = w_q − c`, and increases the loss by `Δ² / [H_F⁻¹]_qq`, where `H_F` is the submatrix of `H` on `F`.

*Assumptions.* `H` positive definite (damped); the constraint set is one equality (a scalar coordinate), the grid rounding of `c` is chosen elsewhere (nearest grid point to the current `w_q`).

*Steps.*
1. Because `∇_F L = 0` at the current point, for any `δ` supported on `F`, `L(w+δ) = L(w) + δᵀ H_F δ`: the linear term vanishes.
2. The re-optimization is `min δᵀH_Fδ` subject to `e_qᵀδ = c − w_q = −Δ`.
3. Lagrangian `δᵀH_Fδ + 2λ(e_qᵀδ + Δ)`; stationarity `H_F δ = −λ e_q`, so `δ = −λ H_F⁻¹ e_q`.
4. The constraint gives `−λ [H_F⁻¹]_qq = −Δ`, so `λ = Δ/[H_F⁻¹]_qq`, which yields the stated `δ`.
5. Cost: `δᵀH_Fδ = λ² e_qᵀ H_F⁻¹ e_q = Δ² / [H_F⁻¹]_qq`.
6. Removing `q` from `F` updates the inverse by the Schur complement identity `H_{F∖q}⁻¹ = H_F⁻¹ − H_F⁻¹e_q e_qᵀ H_F⁻¹ / [H_F⁻¹]_qq` (block-inverse formula).
7. Summing over a sweep in a fixed order, the total proxy loss is `Σ_q Δ_q² / [H_{F_q}⁻¹]_qq` exactly (step 1 applies at each stage; no cross terms).

*GPTQ's simplification.* OBQ chooses the next coordinate per row by the smallest cost (an `O(d_out·d_in³)` procedure). GPTQ fixes one order (left to right, or act-order) for all rows, so `H_{F_q}⁻¹` is shared and only one inverse chain is needed. The paper reports that this arbitrary order loses little on large layers; the check below measures it on small ones.

**D2. The Cholesky reformulation.**

*Statement.* Let `H⁻¹ = UᵀU` with `U` upper triangular with positive diagonal (`U = chol(H⁻¹)ᵀ`). Then for a left-to-right sweep, `[H_{F_q}⁻¹]_{qq} = U_qq²` and `[H_{F_q}⁻¹]_{q,j} / [H_{F_q}⁻¹]_{qq} = U_qj / U_qq` for `j > q`, where `F_q = {q, …, d−1}`. The sweep is therefore: for `q = 0 … d−1`, `err_q = (w_q − Q(w_q)) / U_qq`; `w_{q:} ← w_{q:} − err_q · U_{q,q:}` (this sets `w_q` to `Q(w_q)`); loss `+= err_q²`.

*Steps.*
1. Write `H⁻¹ = Σ_k u_k u_kᵀ` where `u_k` is the `k`-th row of `U` (as a column vector). `U` upper triangular means `u_k` is zero in positions `< k`.
2. Claim: `H_{F_q}⁻¹ = Σ_{k≥q} u_k u_kᵀ` (restricted to indices `≥ q`). For `q = 0` this is step 1.
3. Induction step. `H_{F_q}⁻¹ e_q = Σ_{k≥q} u_k (u_k)_q = u_q U_qq` (only `k = q` has a nonzero entry at position `q`), and `[H_{F_q}⁻¹]_qq = U_qq²`. The Schur-complement downdate of D1 step 6 subtracts `(U_qq u_q)(U_qq u_q)ᵀ / U_qq² = u_q u_qᵀ`, leaving `Σ_{k≥q+1} u_k u_kᵀ`.
4. Hence the OBS direction `H_{F_q}⁻¹e_q / [H_{F_q}⁻¹]_qq = u_q U_qq / U_qq² = u_q / U_qq`, and the loss increment is `Δ_q² / U_qq² = err_q²`.
5. Lazy batching: for columns `i … i+B−1` the updates to later columns are `Σ_q err_q U_{q,·}`; accumulate `E = [err_i … err_{i+B−1}]` and apply `W[:, i+B:] −= E · U[i:i+B, i+B:]` once.
6. Damping replaces `H` by `H + λ·mean(diag H)·I` (a ridge penalty toward the RTN solution); all identities hold for the damped matrix. `H = I` gives `U = I`: no compensation (RTN).

*Numerical check (cell, float64, CPU, seed 0).* `n = 6` weights, `N = 64` calibration rows with correlated columns, damp 0.01, 3-bit symmetric grid with step 0.5, `U = cholesky(H⁻¹)ᵀ`: (i) `UᵀU = H⁻¹` (1e-10); (ii) row `q` of `U` times `U_qq` equals the row of the explicitly downdated inverse from D1 step 6 (1e-10); (iii) the first update `w ← w − err_0·U_{0,:}` equals the constrained solution of D1 step 2 solved with `numpy.linalg.solve` (1e-10); (iv) `Σ err_q²` equals `(w−ŵ)ᵀH(w−ŵ)` for the final `ŵ` (1e-10); (v) exhaustive search over all `8⁶` grid points: the GPTQ loss is never below the brute-force minimum, and its mean over 200 random rows is below RTN's mean (a CPU dry run over 20 rows gave GPTQ ≤ RTN in most but not all rows, so do **not** assert it per row); (vi) `H = I` gives GPTQ ≡ RTN exactly; (vii) the lazy-batch version with `B ∈ {1, 2, 3, 6}` matches the unblocked one (1e-10).

#### From-scratch implementation

Budget ≈ 120 lines.

| Component | Contract | Lines |
|---|---|---|
| `accumulate_hessian` | float32 accumulation; scaling as above | 8 |
| `_prep_hessian(H, damp)` | dead columns (`H_jj = 0` → `H_jj = 1`, `w_j = 0`), damping, `U = cholesky(cholesky_inverse(cholesky(H)), upper=True)`; on failure retry with `damp += 0.0025` (up to 5 times) | 15 |
| `gptq_quantize` | the loop of D2 with lazy blocks, group scales taken from the current compensated weights at the first column of each group, optional `act_order` (permute `W`, `H`; un-permute outputs) | 60 |
| `quantize_model_gptq` | block loop with forward hooks, true-sequential order, input propagation through the quantized block; releases Hessians after use | 35 |
| `obs_step`, `brute_force_quantize` | pedagogical, ≤ 15 lines each (not counted in 120) | — |

Numerical tests:
1. The seven checks (i)–(vii) above on the tiny layer.
2. Block-size invariance on a random `[256, 512]` layer in float32: `blocksize ∈ {1, 16, 128, 512}` produce `Q` equal to 1e-5.
3. `act_order=True` then un-permuting equals running with explicitly permuted `W` and `H`: 1e-6.
4. Loss accounting: `info['loss']` summed over rows equals `trace((W−Q) H (W−Q)ᵀ)` to 1e-4 relative (float32).
5. Optional oracle: the IST-DASLab `GPTQ.fasterquant` on the same `(W, H)` with matched conventions (symmetric, per-group min/max), `Q` equal to 1e-4; skipped when the repo is not installed.
6. Determinism: two CPU float64 runs give identical `codes`.

#### In production

- `gptqmodel` (or `llm-compressor`'s `GPTQModifier`) with the same calibration set and `damp_percent=0.01`, `desc_act` on and off; export to a Marlin-compatible 4-bit layout for serving (the kernel is unchanged by GPTQ, 11.1). The from-scratch code is the oracle: at one layer `Q` agrees up to quantizer conventions (test 5); at model level perplexities agree within the paired bootstrap interval (optional cell).
- `Qwen/Qwen3-1.7B-GPTQ-Int8` as the published artifact whose `quantization_config` we read; the chapter's `lm-c11-qwen3-1p7b-gptq-w4g128` stores `qweight` (`pack_int4`, `uint8 [d_out, d_in/2]`), `scales` (`fp16 [d_out, d_in/g]`), `zeros` (`uint8`) per linear.

#### Headline experiment

*Hypothesis.* (1) GPTQ reduces perplexity relative to RTN at 4 and especially at 3 bits at equal `b_eff`, and the per-layer proxy loss falls by a large factor relative to RTN at every site; (2) the gain depends on calibration: it saturates as the number of sequences grows, is largest when the calibration domain resembles the test domain, and a random-token control is clearly worse than text; (3) the local proxy loss predicts the global perplexity change only loosely.

*Setup.* Qwen3-1.7B; RTN, AWQ (11.2), GPTQ, GPTQ + act-order at W4 g128, W3 g128 and W4 per-channel; no activation quantization. Calibration: 128 × 2048 FineWeb-Edu tokens (the 11.1 artifact). Ablation at W4 g128: calibration size `{4, 16, 64, 128, 512}` sequences × domain `{FineWeb-Edu, WikiText-2 train, random tokens}`; damping `{0, 1e-3, 1e-2, 1e-1}` at 16 sequences (to expose rank deficiency). Evaluation: WikiText-2 test and held-out FineWeb-Edu windows; `EvalSuite` tasks for the main configurations. Sensitivity study: quantize one site type at a time (e.g. only `down_proj`) with RTN vs GPTQ and record `Δppl`.

*Procedure.* `quantize_model_gptq` once per configuration; evaluate with `window_nll`; log per-linear proxy loss for RTN and GPTQ (RTN's proxy loss is computed with the same `H`); `d2l.compression_row` for the main rows. Time each quantization run.

*Metrics.* Perplexity with paired 95% intervals vs RTN and vs BF16; per-layer `loss_GPTQ / loss_RTN`; quantization wall-clock; rank and `κ` of `H` per site; task scores.

*Plots.*
- `fig_lm_gptq_layer_loss`: x = layer index, y = `loss_GPTQ/loss_RTN` (log), one line per linear type (7).
- `fig_lm_gptq_calib`: x = calibration sequences (log), y = WikiText-2 perplexity; one line per domain; band = paired 95% interval vs RTN.
- `fig_lm_gptq_spectrum`: `H` eigenvalues (log-log) for two sites, vertical marks at the damping levels.
- `tab_lm_gptq_main`: method × (bits, group) with ppl ± CI, Δ vs RTN, quantization minutes.

*Expected outcome.* The GPTQ paper (arXiv 2210.17323) reports that GPTQ keeps accuracy at 3–4 bits on large OPT and BLOOM models where RTN degrades sharply at 3 bits, and quantizes a 175B model in a few GPU hours; the Optimal Brain Compression paper (arXiv 2208.11580) gives the exact-OBS ancestor. For Qwen3-1.7B the size of the gain, its dependence on group size, and how much act-order adds are to be measured. Direction for the ablation: more calibration helps and saturates; same-domain calibration helps most; random tokens are worst (all to be measured). The sensitivity study tests whether the sites with the largest proxy-loss reduction are the ones whose GPTQ-vs-RTN switch changes perplexity most (to be measured, and reported even if it does not).

*Noise check.* Paired bootstrap over windows for every perplexity difference; two calibration draws (seeds 0 and 1) for the main W4 g128 and W3 g128 rows; GPTQ is deterministic given data, so seeds enter only through the calibration draw. The ablation grid is reported with intervals; adjacent sizes are called different only when intervals exclude zero. Timing is a single run (not a headline claim).

*Artifacts committed.* `lm-c11-qwen3-1p7b-gptq-w4g128`; per-layer loss table (CSV); rows.

#### Figures and tables

- `fig_lm_gptq_toy` (T0): the 6-weight row, RTN vs GPTQ vs brute-force weights on a line plot with the loss ellipse of `H` for two coordinates.
- `fig_lm_gptq_spectrum`, `fig_lm_gptq_layer_loss`, `fig_lm_gptq_calib`, `tab_lm_gptq_main`.
- `tab_lm_gptq_sens`: per-site-type single-site swap, `Δppl` RTN→GPTQ with paired intervals.
- `tab_lm_gptq_defaults`: the production defaults (`damp_percent`, `desc_act`, `static_groups`, `true_sequential`, `group_size`) from the published checkpoint configs.

#### Pitfalls and misconceptions

- *Better local loss ≠ better model.* GPTQ minimizes a per-layer quadratic proxy; keep the whole-model evaluation separate.
- Collecting `H` from the *unquantized* model's activations instead of the progressively quantized one (`sequential=False`) is a legal variant with a different, usually worse, result.
- Too few calibration tokens leave `H` rank-deficient; damping then silently dominates the solution (the sweep makes this visible).
- Calibrating and testing on the same text inflates the gain.
- Computing `H` in BF16 overflows or loses precision; accumulate in float32 and normalize by the token count.
- Act-order with groups breaks the contiguous-group layout of fast kernels unless `static_groups`/`g_idx` is supported.
- GPTQ is not "faster inference"; it chooses better codes for the same format.
- GPTQ's greedy order is not optimal; the brute-force check will show a gap on tiny layers.

#### Summary (draft bullets)

- Quantization as layer-wise reconstruction has a quadratic loss with Hessian `H = XᵀX`, shared by all output rows.
- Fixing one weight and optimally compensating the others (OBS) costs `Δ²/[H_F⁻¹]_qq` and updates the free weights along a column of `H_F⁻¹`.
- GPTQ sweeps columns in a fixed order; a single Cholesky factor of `H⁻¹` supplies every update, and lazy blocks turn them into matmuls.
- Damping makes `H` invertible and pulls the solution toward RTN; calibration size, domain and sequential propagation all change the outcome.
- GPTQ changes which codes are stored, not the kernel; local loss is a proxy to be validated on held-out text with paired intervals.

#### Exercises

1. *(check)* Why do all rows of a layer share one Hessian, and why does GPTQ need only one Cholesky factor per linear layer? *Sketch:* the loss for row `i` is `(w_i−ŵ_i)ᵀXᵀX(w_i−ŵ_i)`; `X` is the same for every row; the fixed sweep order makes `H_{F_q}⁻¹` row-independent.
2. *(check)* A `down_proj` input has 6144 channels; you calibrate on 2 sequences of 2048 tokens. What is the maximum rank of `H`, and what does GPTQ effectively do on the null space? *Sketch:* rank ≤ 4096; directions with no calibration energy are governed by damping alone, so compensation there is arbitrary (ridge).
3. *(derive)* Prove D1 step 6 (the Schur-complement downdate) from the block-inverse formula. *Sketch:* partition `H_F` around index `q`; `(H_{F∖q})⁻¹` equals the Schur complement of `[H_F⁻¹]_qq` in `H_F⁻¹`.
4. *(derive)* Show that with lazy batching of block size `B` the number of times the `d_out × d_in` weight matrix is read and written falls from `d_in` to `d_in/B` while flop count is unchanged, and give the arithmetic intensity of the block update. *Sketch:* the per-column update is a rank-1 update (intensity ≈ 1 flop/byte); the batched update is a `d_out×B` by `B×(d_in−i−B)` matmul with intensity `O(B)`.
5. *(measure)* Plot the proxy-loss ratio `GPTQ/RTN` against the quantity `Σ_j H_jj` (activation energy) per layer; predict the sign of the correlation, then measure. Repeat with `act_order` on.
6. *(measure)* Quantize with `n_cal ∈ {1, 2, 4, 16}` sequences and damping `{1e-3, 1e-2, 1e-1}`; predict where damping starts to dominate from the rank argument of exercise 2, then check with perplexity and with `κ(H)`.
7. *(open)* Mixture-of-experts layers (`Qwen/Qwen3-30B-A3B`, or `openai/gpt-oss-20b`'s experts): each expert sees only a fraction of calibration tokens, so its `H` is far more rank-deficient. Design and test a calibration scheme (more tokens, shared damping, routed-token oversampling), and report the effect on one expert group.

#### Further reading

- Frantar, Ashkboos, Hoefler, Alistarh, [GPTQ](https://arxiv.org/abs/2210.17323); code https://github.com/IST-DASLab/gptq: the algorithm, the three ideas (arbitrary order, lazy batch updates, Cholesky) and the ablations.
- Frantar and Alistarh, [Optimal Brain Compression](https://arxiv.org/abs/2208.11580): the exact OBS-based quantizer/pruner that GPTQ approximates; Hassibi and Stork, "Second order derivatives for network pruning: Optimal Brain Surgeon" (NeurIPS 1992; cited by title, (unverified) landing page).
- Frantar and Alistarh, [SparseGPT](https://arxiv.org/abs/2301.00774): the same Hessian machinery for pruning (11.6).
- [GPTQModel](https://github.com/ModelCloud/GPTQModel) and [llm-compressor](https://github.com/vllm-project/llm-compressor): production implementations with the settings in `tab_lm_gptq_defaults`.
- Teaching: MIT 6.5940 post-training quantization lectures; Smola, [Efficiency in LLMs](https://alex.smola.org/posts/45-mlss-efficiency/).

#### Risks and open decisions

- *Python-loop speed.* The per-column loop is slow in eager PyTorch (6144 iterations for `down_proj`); lazy blocks keep it tolerable. Fallback: `torch.compile` the inner block loop or reduce to `FAST` (8 blocks) and commit the full-model result as an artifact.
- *Hessian memory.* If 10 GiB is exceeded, keep `H` on the CPU for `down_proj` and move it for the Cholesky only.
- *Matching production conventions* (symmetric vs asymmetric, zero-point rounding, group-scale rule) is fiddly; the optional oracle test documents the conventions that must match.
- *Calibration-domain ablation uses WikiText-2 train and test* (same corpus family): report it as an in-domain effect, not as generalization.
- *No published Qwen3 4-bit GPTQ checkpoint is assumed:* only the 8-bit config is read (a 4-bit official GPTQ checkpoint for Qwen3-4B was not found under the ID `Qwen/Qwen3-4B-GPTQ-Int4`, which the Hub returned as unavailable).

### 11.4 Rotations and vector quantization

- **Label:** `sec_lm_rotations` · **Tier:** T1 (`FAST=True`: 40 evaluation windows, one rotation seed) · **Budget:** ≤ 15 min, peak ≤ 10 GiB (the rotated Qwen3-1.7B is an fp32/bf16 copy of the weights plus an untied head: +0.62 GB) · **Codex units:** 12.5
- **Builds on (book):**
  - `sec_mdl-geometry-linear-algebraic-ops`: orthogonal maps preserve norms and inner products; the geometry of a change of basis. Hadamard matrices and incoherence are **not** in the Mathematics part, so this section teaches them from the definition.
  - `sec_mdl-concentration-generalization`: Rademacher variables are sub-Gaussian with proxy 1 (`eq_mdl-subgaussian`), the Chernoff tail, `E[max_j Z_j] ≤ σ√(2 log N)`.
  - `sec_transformer-block`, `sec_kv-cache`: RMSNorm without mean subtraction, the pluggable `GQAAttention` (`W_q, W_k, W_v, W_o`), SwiGLU `ffn` (`W_g` gate, `W_1` up, `W_2` down).
  - `sec_mdl-information_theory`: entropy and the Gaussian rate–distortion function, for the codebook part.
  - `sec_mdl-svd-low-rank`: orthogonal factors of an SVD as the "learned" alternative to random rotations (mentioned, not re-taught).
- **Builds on (this part):** 11.1 (`d2l.fake_quant`, `d2l.window_nll`, `d2l.calib_tokens`, `d2l.compression_row`), 11.2 (`d2l.SimQuantLinear`, `d2l.apply_smoothing` as the comparison baseline), 11.3 (`d2l.gptq_quantize`, optional rotate-then-GPTQ row), 1.2 (`d2l.load_qwen3`), 2.4 (`:numref:`sec_lm_retrieval``: product quantization as the retrieval counterpart of codebooks; concept only). **Confirmed by 1.2:** `d2l.load_qwen3(...)` returns a `d2l.GPT` whose `blks[i]` are `d2l.TransformerBlock`s with `norm1`, `norm2` (names per 1.2's mapping table), `attention` (`W_q, W_k, W_v, W_o`, `q_norm`, `k_norm`) and `ffn` (`W_g, W_1, W_2`), and `token_emb` tied to the head. The accessor `_parts(blk)` centralizes these names.
- **Exports** (new names):
  - `d2l.hadamard(n, normalized=True)`: dense Hadamard matrix for `n = 2^k` (Sylvester) or `n = 12·2^k` (Kronecker with the order-12 Paley matrix); returns `float64` `[n, n]` with `H Hᵀ = n I` (or `I` if normalized).
  - `d2l.fast_hadamard(x, dim=-1)`: orthonormal fast transform along `dim` in `O(n log n)` (butterflies for the `2^k` factor, one small dense matmul for the 12-factor); equals `x @ hadamard(n)` to float tolerance.
  - `d2l.random_hadamard(n, seed)`: returns `(D, apply)` with `D ∈ {±1}^n` and `apply(x) = fast_hadamard(x * D)`, i.e. `R = H D / √n`.
  - `d2l.max_rms(x, dim=-1)`: per-vector `max|x|/RMS(x)`.
  - `d2l.OnlineHadamard(dim)`: module applying `fast_hadamard` on the last axis (the R4 rotation before `down_proj`).
  - `d2l.rotate_gpt(model, R1, R2=None, online_down=True, untie_head=True)`: fuses `γ` into the next linears, then applies the rotations of D2 in place and inserts `OnlineHadamard` before each `W_2`; returns the model. `R1` is a dense orthogonal matrix or a `random_hadamard` pair.
  - `d2l.learn_rotation(X, W, a_bits=4, steps=200, lr=1e-2, seed=0)`: Cayley-parametrized rotation (via `torch.nn.utils.parametrizations.orthogonal(..., orthogonal_map='cayley')`) minimizing the output error of `fake_quant`-ed activations with a straight-through estimator (11.5); the T0 toy of SpinQuant.
  - `d2l.kmeans_vq(W, d, bits_per_weight, iters=25, seed=0)`: vector quantization of `W` in groups of `d` consecutive weights with `2^{d·bits}` centroids; returns `(codebook, codes, mse)`.
- **Models and data:** `Qwen/Qwen3-1.7B` as a `d2l.GPT` (28 layers, hidden 2048 = 2^11, MLP 6144 = 12·2^9, head_dim 128; tied embeddings); `Qwen/Qwen3-0.6B` for exactness checks (hidden 1024, MLP 3072 = 12·2^8); both sizes avoid the awkward Hadamard orders of `Qwen/Qwen3-4B` (hidden 2560 = 20·2^7 needs an order-20 matrix, MLP 9728 = 76·2^7 an order-76 matrix) and are why 4B is not used here; `Qwen/Qwen3-8B` (4096, 12288 = 12·2^10) would work and is a T2-sized extension. Calibration and evaluation as 11.1. For the codebook exercise: one `down_proj` weight of Qwen3-1.7B layer 14, after a random Hadamard rotation.
- **Packages:** as 11.1; optional QuaRot code (https://github.com/spcl/QuaRot) and SpinQuant code (https://github.com/facebookresearch/SpinQuant) for the oracle comparison; `scipy` only for `scipy.linalg.hadamard` in a test (powers of two).

#### Narrative arc

1. **The question.** *Why is W4A4 so much harder than W8A8, and can a change of basis fix it?* The opening cell runs Qwen3-1.7B at W4A16 (RTN, g128) and at W4A4 (RTN weights per-channel; activations per-token) with `SimQuantLinear` and prints perplexity for both next to BF16: the second collapses (direction; magnitude to be measured). A second cell prints `max_rms` for the four activation sites of layer 14 and the same after one random Hadamard rotation: the ratio drops from tens to a few (to be measured).
2. **Quantization error and `max/RMS`.** Recap in one paragraph (7.6: Δ²/12): with absmax scaling the step is `2·max/(2^b−1)`, so a vector with a large `max/RMS` leaves most coordinates in the zero bin. `eq_lm_rot_ratio` defines `ρ(x) = max|x|/RMS(x) ∈ [1, √n]`; a single massive channel is the worst case `ρ = √n`.
3. **Hadamard matrices and the fast transform** (taught from scratch).
   - Sylvester recursion `H_{2n} = [[H_n, H_n],[H_n, −H_n]]`, `H Hᵀ = n I`; orders other than powers of two via Paley (`12` here, for the MLP width 6144); orders that Qwen3-4B would need are not available from simple constructions.
   - The fast Walsh–Hadamard transform: `log₂ n` butterfly stages of adds/subtracts, `n log₂ n` operations; cell: compare with the dense product and time both (CPU), tolerance 1e-5.
   - Plain Hadamard is not enough: for `x = (1, …, 1)` it creates the worst case `ρ = √n`; the random signs `D` are what protect against adversarial inputs (D1 step 1).
4. **Why a random rotation tames outliers** (derivation D1; `fig_lm_rot_ratio`: `ρ` before and after, 1000 draws, `n ∈ {256, 1024, 4096}`).
5. **Computational invariance** (derivation D2): rotate the residual stream and absorb the rotation into weights; the table of what is fused offline and what must run online (`tab_lm_rot_cost`: flops and bytes per token for R1, R2, R4 and, for the KV cache, R3).
6. **Implement and verify on a real model.** `rotate_gpt` on a 2-layer float64 random `d2l.GPT` (exact), then on Qwen3-0.6B in float32 (tolerance), then count the extra parameters of the untied head.
7. **W4A4 before and after rotation** (headline).
8. **Learned rotations** (T0 toy): `learn_rotation` on one layer with a planted outlier structure; compare the learned `R` with 20 random Hadamards by layer-output error (`fig_lm_rot_learned`). SpinQuant reports that some random rotations are much better than others, up to 13 points of zero-shot accuracy on LLaMA-2 7B in its experiments (arXiv 2405.16406), and learns `R1`/`R2` on the Stiefel manifold with Cayley SGD; the model-level version is a T2 extension (exercise 7).
9. **Block-scaled formats change the story.** With group sizes 16–32 a rotation over the full hidden dimension mixes channels across groups; the MR-GPTQ paper (arXiv 2509.23202) argues that NVFP4's small group size neutralizes the usual outlier-mitigation and uses block-wise (micro) Hadamard rotations of the group size instead. One paragraph and a T0 cell comparing full-dimension vs group-size rotations for per-group INT4 on the spike vector; the format itself is 11.5.
10. **Beyond scalar rounding: codebooks and lattices.** After rotation weights look Gaussian and *incoherent*; scalar quantization then leaves a fixed gap to the rate–distortion bound that vectors close (derivation in exercise 4). Cell (`fig_lm_rot_vq`): `kmeans_vq` at 2 bits/weight with `d ∈ {1, 2, 4, 6}` (4, 16, 256, 4096 centroids) on rotated weights; plot SQNR (dB) vs `d` against the Gaussian bound `6.02·R` dB (R = 2) and the scalar Lloyd–Max point computed by the `d = 1` run. Costs: codebook bytes `2^{d·R}·d·2`, lookups per weight. Names: product quantization (independent sub-codebooks, 2.4's retrieval use), additive quantization (AQLM: sum of codewords from several codebooks, learned input-adaptively and fine-tuned jointly per block), lattice codebooks (QuIP#: codewords from the `E_8` lattice, 8-dimensional, with random Hadamard incoherence processing). The exercise is intentionally small; production VQ needs fine-tuning and dedicated kernels, so no speed is claimed.

#### Derivations

**D1. A random rotation brings `max/RMS` down to `O(√log n)`.**

*Statement.* Let `x ∈ R^n` be fixed, `n = 2^k`, `H` the unnormalized Sylvester–Hadamard matrix, `D = diag(ε_1..ε_n)` with iid Rademacher signs, and `y = H D x / √n`. Then `RMS(y) = ‖x‖/√n` exactly, and for every `δ ∈ (0, 1)`, with probability at least `1 − δ`, `ρ(y) = max_i |y_i| / RMS(y) ≤ √(2 ln(2n/δ))`. Without the rotation, `ρ(x)` can be as large as `√n`.

*Assumptions.* `x` is independent of the signs; `n` is a power of two (other orders give the same bound with an `n`-independent constant from the matrix entries being ±1).

*Steps.*
1. `y_i = (1/√n) Σ_j H_ij ε_j x_j`. With `H_ij ∈ {±1}`, this is a Rademacher sum with fixed coefficients `c_j = H_ij x_j/√n`, and `Σ_j c_j² = ‖x‖²/n`.
2. A Rademacher variable is sub-Gaussian with proxy 1 (`eq_mdl-subgaussian`); a weighted sum is sub-Gaussian with proxy `Σ c_j² = ‖x‖²/n`.
3. The Chernoff tail gives `P(|y_i| ≥ t) ≤ 2 exp(−n t² / (2‖x‖²))`.
4. Union bound over the `n` coordinates: `P(max_i |y_i| ≥ t) ≤ 2n exp(−n t²/(2‖x‖²))`. Setting the right-hand side to `δ` gives `t = ‖x‖ √(2 ln(2n/δ) / n)`.
5. `H/√n` is orthogonal, so `‖y‖ = ‖x‖` and `RMS(y) = ‖x‖/√n`. Dividing: `ρ(y) ≤ √(2 ln(2n/δ))` with probability `≥ 1−δ`.
6. Contrast: `ρ(x) = ‖x‖_∞ √n / ‖x‖₂ ≤ √n`, with equality iff `x` has one nonzero entry (one massive channel).
7. For a Haar-random orthogonal matrix the same bound holds up to constants by concentration of a coordinate of a uniform point on the sphere (stated, not proved). A *fixed* Hadamard without signs has no such guarantee: `x = (1, …, 1)` gives `ρ = √n`.
8. Consequence for absmax quantization at `b` bits: noise per coordinate is about `Δ²/12` with `Δ = 2·max/(2^b − 1)`, i.e. a relative MSE `ρ²/(3(2^b−1)²)` **in the regime `Δ ≲` the typical magnitude** (after rotation). Before rotation, with `Δ` much larger than the bulk, the bulk rounds to zero and its relative error is about 1: the formula must not be applied there.

*Numerical check (cell, CPU, seed 0).* (i) For `n ∈ {256, 1024, 4096}`, `x` = Gaussian noise (σ = 0.1) with one entry 50: before-rotation `ρ ≈ √n` (to 5%); after rotation, `ρ ≤ √(2 ln(2n/0.01))` in at least 99% of 1000 sign draws. (ii) `x = (1, …, 1)`: plain `H x/√n` gives `ρ = √n` exactly; with signs, `ρ` below the bound. (iii) Gaussian `x`: `ρ` before and after within 10% (rotation neither helps nor hurts a vector that is already sub-Gaussian). (iv) `n = 2048`, one spike: relative error of 4-bit absmax on the *non-spike coordinates* is `1.0` before rotation (everything rounds to zero) and clearly less than 1 after (a CPU dry run gave about 0.36; to be remeasured); the total relative MSE after rotation matches `ρ²/(12·7²)` for symmetric INT4 (step `max/7`) within 10% (the dry run gave 0.0029 predicted vs 0.0029 measured).

**D2. Computational invariance: what can be rotated without changing the function.**

*Statement.* Let `R ∈ R^{d×d}` be orthogonal and the model use RMSNorm without mean subtraction. Row-vector convention: the residual stream `x ∈ R^{1×d}`. Replace `x` by `x̃ = x R` everywhere in the residual stream and change weights as in the table; the logits are unchanged.

| Weight | Role | Change |
|---|---|---|
| `γ` of `norm1`, `norm2`, final norm | scale after normalization | first fold into the following linears: `W ← W diag(γ)` (columns), set `γ = 1` |
| `W_q, W_k, W_v, W_g, W_1`, head | read the residual | `W ← W R` (input columns) |
| embedding `E`, `W_o`, `W_2` | write the residual | `E ← E R`; `W ← Rᵀ W` (output rows) |
| `W_v` rows / `W_o` columns, per KV head | V and its output projection | `W_v ← R2ᵀ W_v` (per head block), `W_o ← W_o R2` (per head block) |
| `W_2` columns | input = SwiGLU product | `W_2 ← W_2 H` and apply `H` online to the product (`OnlineHadamard`) |

*Steps.*
1. RMSNorm without `γ`: `norm(xR) = xR / rms(xR) = (x/rms(x)) R`, because `‖xR‖ = ‖x‖` for orthogonal `R`. So normalization commutes with `R` once `γ` is folded away (a `γ` that is not scale-free would not commute: this is why it is folded first).
2. A reader `x̃ W'ᵀ = x R (W R)ᵀ = x R Rᵀ Wᵀ = x Wᵀ`.
3. A writer: `(z W_oᵀ) R = z (Rᵀ W_o)ᵀ`, so the contribution added to the stream is the original contribution rotated by `R`; residual addition is linear, so `x̃` stays `x R` through all layers.
4. QK-norm and RoPE act on the outputs of `W_q`, `W_k` and never see the residual basis: unaffected by `R`.
5. R2: attention output per head is `A · v_h` with `A` the attention weights; `A (v_h R2) = (A v_h) R2`, so rotating the V head block by `R2` and un-rotating in `W_o` is exact. GQA: all query heads sharing a KV head see the same rotated V and each takes its own `W_o` block rotated by the same `R2`.
6. R4: `u = silu(g) ⊙ w₁` is a vector in `R^{d_ff}`; `u H · (W_2 H)ᵀ = u W_2ᵀ` with `H` orthonormal Hadamard. `H` cannot be fused into the producer (SiLU is not linear), so it runs online: `O(d_ff log d_ff)` per token.
7. Tied embeddings: `head ← head diag(γ_final) R` differs from `E R` by the folded `γ`, so the head must be untied (a copy). The cost is one extra `vocab × d` matrix.

*Numerical check (cell).* On a 2-layer random `d2l.GPT` in float64 (hidden 64, MLP 192 = 12·16, 4 heads, 2 KV heads), `rotate_gpt` with a random Hadamard `R1`, a random Hadamard `R2`, online R4: `max |logits_before − logits_after| ≤ 1e-10`. Then on Qwen3-0.6B in float32: `atol=1e-3, rtol=1e-4` on logits for 4 sequences (tolerance chosen as an upper bound to be tightened after measuring), and top-1 token agreement on all positions. Finally, with the QK-norm modules deliberately skipped (a deliberate bug), the check fails, confirming that it can fail.

#### From-scratch implementation

Budget ≈ 140 lines.

| Component | Lines |
|---|---|
| `hadamard` (Sylvester + Paley-12 Kronecker), `fast_hadamard`, `random_hadamard`, `max_rms` | 40 |
| `OnlineHadamard` | 8 |
| `rotate_gpt` (fold `γ`, apply table of D2, untie head, insert online R4) | 55 |
| `learn_rotation` (Cayley parametrization + STE loss; the STE is 11.5's `FakeQuantSTE`, imported there, defined here as 3 lines) | 20 |
| `kmeans_vq` (Lloyd iterations with `torch.cdist`, k-means++ init off for speed) | 20 |

Numerical tests:
1. `H Hᵀ = n I` for `n ∈ {2, 64, 12, 24, 3072, 6144}` (1e-9), and `scipy.linalg.hadamard(64)` equals the Sylvester matrix up to row order (the same set of rows).
2. `fast_hadamard(x) = x @ hadamard(n)` (1e-5) for `n ∈ {256, 3072}`; involution: applying it twice returns `x` (1e-6).
3. D1 and D2 numerical checks above.
4. Parameter accounting: untied head adds exactly `vocab·d` parameters; all other parameter counts unchanged.
5. Optional oracle: the QuaRot reference implementation's rotated logits on the same 2-layer model match `rotate_gpt` when given the same `R1` (skipped if the repo is absent).

#### In production

- QuaRot (https://github.com/spcl/QuaRot) fuses R1/R2 offline and implements R3/R4 as fused CUDA/CUTLASS kernels with INT4 GEMMs; SpinQuant (https://github.com/facebookresearch/SpinQuant) adds learned R1/R2. QuIP# (https://github.com/Cornell-RelaxML/quip-sharp) and AQLM (https://github.com/Vahe1994/AQLM) provide VQ codebook kernels and checkpoints. NVIDIA's Model Optimizer and torchao expose rotation/Hadamard options for their low-bit paths (check pinned versions).
- The from-scratch rotated model is the test oracle for any production rotation: the same logits in float, then quantized perplexity within the paired interval.
- No INT4 activation GEMM is run in this chapter on A100 or Ada (the torch eager stack has none; the papers use custom kernels): **all W4A4 numbers here are simulated**, and the notebook states it in the table header. QuaRot and SpinQuant report speed from their kernels, which this notebook does not reproduce.

#### Headline experiment

*Hypothesis.* Rotations leave A8 unchanged and make A4 workable: at W4A4 the rotated model's perplexity is far below the unrotated model's; most of the gain is already in R1 (residual) and R4 (down-projection input), with R2 smaller; the variation across rotation seeds is small compared with the rotated-vs-unrotated gap; weights quantized by GPTQ on the rotated model are better than RTN.

*Setup.* Qwen3-1.7B as `d2l.GPT` in BF16 for evaluation (float32 for the fusion step). Simulated quantization with `SimQuantLinear`: weights W16 or W4 (symmetric per-channel RTN, clip 1.0); activations per-token symmetric `A ∈ {8, 6, 4}` bits, clip 1.0 (and 0.9 for A4 as a knob), applied at all linear inputs except the head; KV cache 16-bit. Rotations: none; SmoothQuant α=0.5 (11.2) as a baseline; R1 only (random Hadamard, seed 0); R1+R2+R4 (random Hadamard, seeds 0, 1, 2); R1 as a dense Haar-random orthogonal matrix with R2/R4 Hadamard (seed 0); R1+R2+R4 then GPTQ W4 (11.3, one seed; optional if the 15-minute budget is exceeded). Evaluation: WikiText-2 test windows, plus `EvalSuite` tasks on the W4A4 rows.

*Procedure.* First verify rotation invariance with quantization off (D2 check) for each seed. Then evaluate the configurations; record `ρ` per site and layer before and after rotation on 16 calibration sequences.

*Metrics.* Perplexity with 95% paired bootstrap interval (over windows) against the unrotated configuration with the same bits; across-seed standard deviation of perplexity; mean per-site `ρ` and the per-layer relative output error of A4 vs A16; task scores.

*Plots.*
- `fig_lm_rot_ratio`: per site (4 panels), `ρ` by layer before/after rotation (log y).
- `fig_lm_rot_abits`: x = activation bits (8, 6, 4), y = perplexity (log), lines: unrotated, SmoothQuant, rotated (mean of seeds with seed-range band), for W16 and W4 (two panels).
- `tab_lm_rot_main`: rows = configuration, columns = ppl ± CI, Δ vs unrotated, task scores, note "simulated".

*Expected outcome.* Direction only until measured. QuaRot reports 4-bit weights, activations and KV cache with at most 0.47 WikiText-2 perplexity loss on LLaMA-2-70B and 6- and 8-bit lossless with RTN (arXiv 2404.00456); SpinQuant reports a 2.9-point average zero-shot gap to full precision on LLaMA-2 7B and substantial variance across random rotations (arXiv 2405.16406). Whether the same magnitudes hold for Qwen3-1.7B, whose activations pass through QK-norm and whose head is tied, is to be measured; a small model at W4A4 may stay well above BF16 even after rotation, and that is a result.

*Noise check.* Three rotation seeds at W4A4 (R1+R2+R4); the gap to the unrotated model is claimed only if it exceeds both the paired interval and the across-seed range. Single seeds elsewhere are labeled single seeds. Perplexity on the same windows for every configuration, so all differences are paired.

*Artifacts committed.* Rotation seeds (the sign vectors, 3 × 2048 bits), per-site `ρ` table, rows in `data/compression_report.jsonl`, the toy learned-rotation result.

#### Figures and tables

- `fig_lm_rot_ratio` (headline stats and the synthetic `n` sweep), `fig_lm_rot_abits`, `fig_lm_rot_learned`, `fig_lm_rot_vq`.
- `tab_lm_rot_cost`: per rotation (R1, R2, R3, R4): fused offline or online, extra flops per token, extra bytes, kernel needed.
- `tab_lm_rot_main`, `tab_lm_rot_hadamard_orders`: Qwen3 widths (1024, 2048, 3072, 4096, 6144, 12288, 2560, 9728) factored as `m·2^k` with the available `m`.

#### Pitfalls and misconceptions

- A rotation is exact only in exact arithmetic and only if every reader and writer of the residual stream is transformed, including the embedding, the head, and the folded `γ`; one forgotten matrix gives plausible but wrong logits. Always run the unquantized invariance check first.
- Folding `γ` is mandatory: RMSNorm with a learned scale does not commute with a dense rotation.
- Tied embeddings must be untied for the head.
- Online rotations are not free: they need a kernel; "free" applies to R1 and R2 only.
- Plain Hadamard without random signs has adversarial inputs (`x = 1`).
- Rotation does not help vectors that are already well spread (Gaussian), and it can slightly hurt a model whose outliers carry semantics that a per-channel format preserved (measure, do not assume).
- With group sizes of 16–32, mixing across groups can reduce the benefit of block scaling; use group-sized rotations there.
- Reporting W4A4 numbers from fake quantization as speedups.

#### Summary (draft bullets)

- Outliers are a basis artifact: `max/RMS` is at most `√n` and attains it for one massive channel; a randomized Hadamard rotation gives `O(√log n)` with high probability.
- Rotations are absorbed into weights through computational invariance (fold `γ`, rotate readers and writers); R1 and R2 are free, R4 needs an online fast transform.
- Hadamard matrices exist for orders `2^k` and `12·2^k`; Qwen3's 1.7B and 8B widths fit, others need other constructions.
- Rotation makes W4A4 feasible in simulation; some random rotations are better than others and learning them helps.
- Vector codebooks close part of the gap between scalar rounding and the rate–distortion bound but cost codebook memory, lookups and kernels.

#### Exercises

1. *(check)* Why must `γ` be folded into the following linear layer before applying `R1`, and which parameters become different (untied) in Qwen3-1.7B afterwards? *Sketch:* D2 steps 1 and 7.
2. *(check)* Which of R1, R2, R4 can be fully fused offline? Which one is needed to quantize the input of `down_proj` to 4 bits and why can it not be fused? *Sketch:* R1 and R2 yes; R4: the SiLU product is not a linear function of any weight, so the rotation must run on the activation.
3. *(derive)* Show that for a fixed Hadamard matrix without random signs, `x = (1, …, 1)` attains `ρ = √n` after the transform, and find a distribution over signs that makes the expected `ρ` small. *Sketch:* `Hx` has the first row of `H` all ones: `(Hx)_1 = n`, others 0; with signs, D1.
4. *(derive)* For a Gaussian source with variance `σ²`, the rate–distortion function gives `D(R) = σ² 2^{−2R}`; a uniform scalar quantizer with entropy coding at high rate has `D = Δ²/12` with entropy `h(X) − log₂ Δ`. Show the gap is `½ log₂(2πe/12) ≈ 0.254` bits (1.53 dB), and explain why vector quantization with a better space-filling lattice can recover part of it (the `E_8` lattice has normalized second moment about 0.0717 against the cube's 1/12, from Conway and Sloane (unverified source)). *Sketch:* `h = ½ log₂(2πeσ²)`; `D = σ²2^{−2R}` vs `Δ²/12 = 2^{2(h−R)}/12 = σ² (2πe/12) 2^{−2R}`; the ratio `2πe/12 = 1.4233` is 1.53 dB.
5. *(measure)* Draw 20 random Hadamard rotations for one layer; plot the layer-output relative error of A4 quantization against the seed; predict the coefficient of variation from D1's spread and compare. Then evaluate the best and worst seed on full-model perplexity (two extra runs) and relate to SpinQuant's finding.
6. *(measure)* Quantize *weights only* at W3 per-channel with and without R1; predict from D1 whether rotation helps weights as much as activations (weights are already roughly Gaussian) and measure.
7. *(open)* Implement SpinQuant-style learning of R1 (Cayley SGD on the Stiefel manifold, STE through `fake_quant`) on 64 calibration sequences for Qwen3-0.6B, and compare against the best of 20 random Hadamards on W4A4 perplexity. Report the compute spent.

#### Further reading

- Ashkboos et al., [QuaRot](https://arxiv.org/abs/2404.00456), code https://github.com/spcl/QuaRot: computational invariance and the four rotations.
- Liu et al., [SpinQuant](https://arxiv.org/abs/2405.16406), code https://github.com/facebookresearch/SpinQuant: learned rotations and the variance of random ones.
- Tseng et al., [QuIP#](https://arxiv.org/abs/2402.04396), code https://github.com/Cornell-RelaxML/quip-sharp; Chee et al., [QuIP](https://arxiv.org/abs/2307.13304): incoherence processing and lattice codebooks.
- Egiazarian et al., [AQLM](https://arxiv.org/abs/2401.06118), code https://github.com/Vahe1994/AQLM: additive multi-codebook quantization.
- Egiazarian et al., [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202): why rotations interact with block formats.
- Teaching: MIT 6.5940 lectures on advanced quantization; Grootendorst's visual guide for the outlier pictures.

#### Risks and open decisions

- *Hugging Face Qwen3 module tree is also needed for torchao timing.* The surgery is written on `d2l.GPT` (names confirmed by 1.2); if a Hugging Face model is used (`q_proj…`, `q_norm`, `k_norm`), the same table applies and the exactness tests are written against both.
- *Qwen3-1.7B may stay far from BF16 at W4A4 after rotation.* Fallback: present W4A8 and W6A6 as the practical points and keep the W4A4 result as the stress test, with the paper's larger-model numbers cited rather than reproduced.
- *Time:* ~17 evaluations of an fp-simulated model; if over budget, reduce the `A ∈ {8, 6}` rows to 80 windows.
- *Hadamard orders:* widths that need other constructions are excluded by model choice; an exercise handles the general case.
- *License:* QuaRot, SpinQuant, QuIP# repos carry their own licenses (check before vendoring; the notebook only calls them optionally).

### 11.5 Quantization-aware training and native low-bit formats

- **Label:** `sec_lm_qat` · **Tier:** T1 (`FAST=True`: 60 training steps, 40 evaluation windows) · **Budget:** QAT headline ≈ 15 min (five to seven runs of a few minutes each on Qwen3-0.6B; per-run wall-clock is printed and `STEPS` is set so one run takes ≈ 2 min; the W3 extension is behind `FULL=True`), gpt-oss inspection ≤ 5 min and ≈ 0.45 GB of range-read downloads · peak ≤ 14 GiB (0.6B parameters: fp32 master weights 2.4 GB + fp32 gradients 2.4 GB + AdamW state 4.8 GB, bf16 autocast, activation checkpointing, batch 8 × 1024 tokens) · **Codex units:** 12.6
- **Builds on (book):**
  - `sec_custom_layer`: the straight-through estimator and the `RoundSTE` cell (`torch.autograd.Function` with identity backward). Recap in one paragraph and **not re-taught**. Note that `RoundSTE` there is a notebook cell, not a `#@save` export, so this section defines the 3-line clipped variant it needs.
  - `sec_numerics`, `sec_hardware` (`tab_gpu_specs` rows for fp8/fp4): the E4M3 and E2M1 layouts and which GPUs have native units; not re-taught.
  - `sec_memory_precision`: fp32 master weights + bf16 autocast for the training run.
- **Builds on (this part):**
  - 4.4: block-scaled formats and the NVFP4 simulation (this section uses and extends the simulator for MXFP4, MXFP8 and NVFP4 weights; `:numref:`sec_lm_low_precision``); low-precision *training* is 4.4's topic, QAT here targets a *deployment* format.
  - 7.4 and 7.6: LoRA (PEFT) and QLoRA as the "frozen quantized weights + adapters" alternative, and LoftQ.
  - 11.1 (`d2l.fake_quant`, `d2l.window_nll`, `d2l.calib_tokens`, `d2l.compression_row`, `d2l.replace_linears`, `d2l.effective_bits`), 11.3 (`d2l.quantize_model_gptq` as the PTQ comparator), 1.2 (`d2l.load_qwen3`), 1.5 (`d2l.paired_bootstrap`, `d2l.EvalSuite`).
- **Exports** (new names):
  - `d2l.FakeQuantSTE`: `torch.autograd.Function`; `forward(x, bits, group_size, clip)` returns the group-wise symmetric absmax fake quantization (scale detached); `backward` passes the gradient where `|x/s| ≤ clip·p` and zero elsewhere (the clipped STE of D1).
  - `d2l.FakeQuantLinear(linear, bits=4, group_size=128, clip=1.0, learn_scale=False, fmt='int')`: replaces the weight by its fake-quantized value in `forward` while keeping the latent fp32 weight as the parameter; `learn_scale=True` adds an LSQ-style learnable per-group scale (D1); `fmt ∈ {'int','mxfp4','nvfp4','mxfp8'}` swaps the quantizer for `quant_mx`.
  - `d2l.quant_mx(x, fmt='mxfp4', block=None, dim=-1)`: simulated block-scaled quantize→dequantize. `mxfp4`: E2M1 elements, block 32, power-of-two (E8M0) scale from `floor(log2 amax) − 2`; `mxfp8`: E4M3 elements, block 32, E8M0 scale; `nvfp4`: E2M1 elements, block 16, E4M3 scale `amax/6` plus a per-tensor float32 scale. Returns same dtype as `x`.
  - `d2l.decode_mxfp4(blocks, scales)`: decodes gpt-oss-style packed tensors (`uint8` `[..., G, 16]` blocks of 32 nibbles, low nibble first; `uint8` scales with bias 127) to `bfloat16`.
  - `d2l.read_safetensors(repo, filename, names, revision)`: reads only the named tensors of a remote `.safetensors` file by HTTP range requests (header first, then byte ranges), returning a dict of tensors; no full-shard download.
  - `d2l.qat_export(model)`: freezes scales from the final latent weights, returns `(codes int8, scales fp16)` per linear and a model whose forward equals the fake-quantized model exactly (the conversion check).
  - `d2l.ternary_absmean(W)`: BitNet b1.58 quantizer `γ = mean|W|`, `Ŵ = γ·clip(round(W/(γ+ε)), −1, 1)`.
- **Models and data:**
  - Training and comparison: `Qwen/Qwen3-0.6B` through `d2l.load_qwen3` (R1; fp32 master copy). PTQ-only measurements also on `Qwen/Qwen3-1.7B`.
  - Training data: FineWeb-Edu `sample-10BT` documents disjoint from the 11.1 calibration draw and the evaluation slice (seeded document-level split); evaluation WikiText-2 test and a held-out FineWeb-Edu slice; `EvalSuite` tasks.
  - Native MXFP4: `openai/gpt-oss-20b` (Apache-2.0, revision pinned): 24 layers, hidden 2880, 32 experts, top-4, `intermediate_size` 2880, `quantization_config = {quant_method: 'mxfp4', modules_to_not_convert: [self_attn, router, embed_tokens, lm_head]}` (verified): only the expert matrices are MXFP4. Layer-0 tensors in shard `model-00000-of-00002.safetensors` (verified from the file header): `gate_up_proj_blocks` `U8 [32, 5760, 90, 16]`, `gate_up_proj_scales` `U8 [32, 5760, 90]`, `down_proj_blocks` `U8 [32, 2880, 90, 16]`, `down_proj_scales` `U8 [32, 2880, 90]`, plus BF16 biases; ≈ 0.42 GB for layer 0's expert tensors. The Transformers MXFP4 page reports 13.8 GB on disk and "within 16 GB" of memory.
- **Packages:** `torch ≥ 2.6` (float8 dtypes for E4M3 rounding on CPU/GPU), `torchao` (QAT reference: the "Fine-tuning with QAT, QLoRA, and float8" page, https://docs.pytorch.org/ao/stable/eager_tutorials/finetuning.html, verified for the 0.17 documentation), `peft`, `safetensors`, `huggingface_hub`, `transformers ≥ 4.55` and `triton ≥ 3.4` + `kernels` (optional end-to-end gpt-oss on Ada).

#### Narrative arc

1. **The question.** *If PTQ at 3–4 bits hurts a small model, why not train it to live on the grid?* Cell 1: Qwen3-0.6B, W4 g128 and W3 g128 RTN perplexity vs BF16 (the gap that QAT should close; values to be measured). Cell 2 (the recap of `sec_custom_layer`): put a plain `round` in the forward pass of one linear layer and show that every gradient to the weight is exactly zero; then swap in the STE and show nonzero gradients and a loss that decreases.
2. **Fake quantization with a straight-through estimator** (derivation D1). The forward pass uses `Ŵ = s·clamp(round(W/s), −n, p)`; the optimizer updates a latent full-precision `W`; scales are recomputed from the latent weights each step (absmax, detached) or learned (LSQ). Practical choices: quantize all block linears, keep embedding, head and norms at 16 bits; fp32 master weights; small learning rate; why weight decay and Adam's state interact with the grid.
3. **QAT vs the alternatives at a matched budget.** A four-row table `tab_lm_qat_alternatives`: PTQ (no gradients), PTQ + adapters on frozen quantized weights (QLoRA-style recovery, 7.6), QAT on full weights, and low-precision *training* from scratch (4.4). What each updates, what it needs (data, memory), what it can fix. Distinguish from LoRA on frozen quantized weights: the latter cannot move the grid, only add a low-rank correction.
4. **Recovery training in practice (headline)**: matched data, token and update budget; a BF16 *control* trained on the same tokens, so the effect of simply seeing FineWeb-Edu is not credited to quantization-awareness; weight displacement diagnostic `‖W_QAT − W₀‖_F/‖W₀‖_F` per layer (ParetoQ reports that at 3 bits and above fine-tuned models stay close to their pretrained distribution, and change drastically at 2 bits or below; measure it here for 4 and 3 bits).
5. **Convert and check**: `qat_export` → codes and scales; assert the converted model's logits equal the fake-quantized model's (rel. 1e-5); then run the packed model through torchao's int4 kernel from 11.1 for speed (QAT changes the weights, not the kernel). This is the Codex plan's "check fake-quantized vs converted outputs before comparing speed".
6. **Microscaling formats** (derivation D2): MXFP8, MXFP4, NVFP4 layouts (element, block, scale, effective bits); `quant_mx`; the MSE-vs-scale-mismatch curve; matched-bits comparison with INT4 (MXFP4 4.25 b vs INT4 g64 FP16-scale 4.25 b; NVFP4 4.5 b vs INT4 g32 4.5 b) as weight-only PTQ on Qwen3-0.6B and 1.7B, simulated. Which hardware runs what natively (overview table): only Blackwell has MXFP4/NVFP4 tensor-core arithmetic; the A100 and the Ada build box simulate it; a software kernel can still read MXFP4 *storage* and dequantize into BF16 MMA, which is how gpt-oss runs on non-Blackwell GPUs (Transformers' Triton kernels; hardware requirement per the Transformers MXFP4 page: compute capability ≥ 7.5 and Triton ≥ 3.4).
7. **A model released natively in MXFP4: `gpt-oss-20b`.** From the config: which modules are quantized. From the file header (no download): tensor names, shapes, dtypes, byte offsets → exact bytes of the MXFP4 experts vs the BF16 rest, and the checkpoint's effective bits per parameter (`4.25` for experts; to be computed). Range-read layer 0's expert tensors, decode with `decode_mxfp4`, and inspect: histogram of the 16 code values, histogram of the E8M0 exponents, fraction of codes at `±6`, weight RMS per expert. Apply `quant_mx('mxfp4')` to the decoded BF16 weights and check *value-level idempotence* (the simulator leaves already-MXFP4 weights unchanged; the scale bytes may differ for blocks whose largest code is below 6; see the check in D2). Then (optional, guarded by free memory ≥ 14 GiB) run end-to-end `generate` on Ada through the Transformers MXFP4 path and report which part is native (storage) and which is emulated (arithmetic). The model card states that the MoE weights were post-trained with MXFP4 quantization and that evaluations use it; the section says what that implies (the released weights are the quantized ones, there is no higher-precision twin to compare with) and what it does not (it does not tell us the recipe beyond the card).
8. **Ternary weights.** The BitNet b1.58 absmean quantizer, `d2l.ternary_absmean`; PTQ of Qwen3-0.6B to ternary (expected catastrophic: measured), the observation that b1.58 models are trained at that precision from the start (arXiv 2402.17764), and ParetoQ's finding about the 2-to-3-bit learning transition (arXiv 2502.02631). No claim about speed (ternary needs dedicated kernels; e.g. the BitNet inference code at https://github.com/microsoft/BitNet).

#### Derivations

**D1. The clipped straight-through estimator and the learnable scale (LSQ).**

*Statement.* For `x̂ = s · clamp(round(x/s), −n, p)` with integer bounds `−n = −2^{b−1}`, `p = 2^{b−1} − 1` and `u = x/s`: the true derivative is `0` almost everywhere; the straight-through estimator replaces `round` by the identity in the backward pass and keeps the clamp's derivative, giving `∂x̂/∂x = 1[−n ≤ u ≤ p]` and, when `s` is learnable, `∂x̂/∂s = round(u) − u` for `−n ≤ u ≤ p`, `−n` for `u < −n`, `p` for `u > p`.

*Steps.*
1. `x̂ = s·c(u)`, `c(u) = clamp(round(u), −n, p)`; away from rounding boundaries `round` is locally constant, so the true gradient of `c` w.r.t. `u` is 0 (this is the zero-gradient cell of `sec_custom_layer`).
2. STE: define the surrogate `c̃(u) = clamp(u, −n, p)` for the backward pass only; its derivative is `1[−n ≤ u ≤ p]`.
3. Weight gradient: `∂x̂/∂x = s · c̃'(u) · (1/s) = 1[−n ≤ u ≤ p]`. Inside the clipping range the gradient flows unchanged; clipped weights receive zero.
4. Scale gradient (the same surrogate applied to `round` only inside the range, product rule): inside, `x̂ = s·round(x/s)`, so `∂x̂/∂s = round(u) + s·(−x/s²)·1 = round(u) − u` (the signed rounding error); outside the range `x̂ = s·p` or `−s·n`, so `∂x̂/∂s = p` or `−n`.
5. With absmax scaling and `clip = 1`, `|u| ≤ p` for every weight (the negative side has one more level), so nothing is clipped and the gradient is the identity; with `clip < 1` or a learned `s` the clipped branch matters (LSQ rescales the scale gradient by `1/√(N_w·p)` for stable step sizes; a one-line implementation note, not derived).
6. The estimator is biased: the forward value is `x̂`, the backward pretends it is `x`. It converges in practice because the latent weight drifts until the grid points it uses are loss-minimal; the diagnostics of the headline (weight displacement, flipped-code fraction per step) show whether codes still change.

*Numerical check (cell, float64, seed 0).* (i) For `x ∈ R^{4096}` normal and `s` chosen so that 3% of entries are clipped: autograd gradient of `Σ x̂ · g` (random `g`) w.r.t. `x` equals `g · 1[...]` exactly. (ii) For the `s`-gradient: finite difference `(x̂(s+ε) − x̂(s−ε))/(2ε)` with `ε = 1e-7` at entries whose `u` is farther than `1e-3` from a rounding boundary equals `round(u) − u` (inside) / `p` / `−n` to 1e-5. (iii) The three cases are all exercised (assert at least one clipped entry on each side). (iv) The plain `round` gives all-zero gradients (the baseline).

**D2. Why power-of-two block scales cost accuracy (MXFP4 vs NVFP4).**

*Statement.* For a block with maximum magnitude `m` and element format E2M1 (grid `{0, .5, 1, 1.5, 2, 3, 4, 6}` and signs), the ideal scale is `s* = m/6` (the block maximum lands on the largest grid point). A format stores a scale `s`; write `κ = s/s*`. MXFP4's E8M0 scale with the reference rule `s = 2^{⌊log₂ m⌋ − 2}` gives `κ = 6/r` with `r = m/s ∈ [4, 8)`, so `κ ∈ (0.75, 1.5]`; NVFP4's E4M3 scale `s = RN_{E4M3}(m/6)` gives `κ ∈ [1 − 2^{−4}, 1 + 2^{−4}]` (3 mantissa bits, relative half-spacing at most `2^{−4}`; a per-tensor float32 scale keeps the E4M3 scale in range), and its block is 16 rather than 32.

*Assumptions.* Round-to-nearest on the grid; elements above `6s` clip to `6s`; the reference conversion rule of the OCP Microscaling spec (to be checked against the spec PDF, https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).

*Steps.*
1. Grid spacing is `0.5` on `[0, 2]`, `1` on `[2, 4]`, `2` on `[4, 6]`; the rounding error of an unclipped element is at most half the local spacing times `s`.
2. If `κ < 1` the block maximum and the largest elements exceed `6s` and clip; the clipping error of the maximum element is `m − 6s = m(1 − κ)`.
3. If `κ > 1` no clipping occurs but all spacings grow by `κ`, so the rounding error grows (and the top grid points are under-used).
4. Hence the per-block MSE is a function `E(κ)`, asymmetric around 1 (sharp for `κ < 1`) and not monotone for `κ > 1` because grid points of the scaled grid can fall near data (`κ = 1.5` aligns the grid `{0, 0.75, 1.5, …}`); a table of `E(κ)` on Gaussian blocks is computed by the cell.
5. MXFP4 draws `κ` from `(0.75, 1.5]` depending on where `log₂ m` falls (about 30% of Gaussian blocks of 32 have `r ≥ 6`, i.e. `κ ≤ 1`, in a CPU dry run; the share depends on the block distribution); NVFP4 draws `κ` within `±6%` of 1. Smaller blocks (16) have smaller `m/RMS` on average, which helps independently of the scale format.
6. Effective bits: MXFP4 `4 + 8/32 = 4.25`; NVFP4 `4 + 8/16 = 4.5` (+ one float32 per tensor).

*Numerical check (cell, seed 0).* 2²⁰ Gaussian entries: (i) the share of blocks with `r ≥ 6` under the MX rule is printed; (ii) `E(κ)` over `κ ∈ {0.5, …, 2}`: its minimum lies in `[0.9, 1.1]`, and `E(0.75)/E(1) ≥ 1.5` (dry run: about 2.3) with block 32; (iii) relative MSE: NVFP4 (4.5 b) below MXFP4 (4.25 b) by at least 20%, MXFP4 within 15% of INT4 g128 (a dry run gave NVFP4 0.0090, MXFP4 0.0132, INT4 g32 0.0094, INT4 g128 0.0138; do not assert these values, assert only the inequalities); (iv) the round-up rule `s = 2^{⌈log₂(m/6)⌉}` (always `κ ≥ 1`) is *not* better on average (dry run: about equal), which shows that clipping is not the whole story.

*Idempotence check on real weights (cell, gpt-oss layer 0).* For every block, a block whose largest decoded code is `c·2^{e₀}` is re-quantized by the reference rule with exponent `e₀ + ⌊log₂ c⌋ − 2`, and `g·2^{e₀} / 2^{e₀+⌊log₂ c⌋−2} = g·2^{2−⌊log₂ c⌋}` lies on the E2M1 grid for every `g ≤ c` on the grid, so `quant_mx(decode(...))` equals `decode(...)` value for value (bytes may differ); assert `torch.equal` on the decoded BF16 values for one expert matrix.

#### From-scratch implementation

Budget ≈ 140 lines.

| Component | Lines |
|---|---|
| `FakeQuantSTE` | 20 |
| `FakeQuantLinear` (latent weight parameter, scale detach or LSQ, `fmt` switch) | 30 |
| `quant_mx` (E2M1 nearest-grid via `torch.bucketize` on midpoints; E4M3 scale via `.to(torch.float8_e4m3fn)`; E8M0 via `2**floor(log2)`; per-tensor scale for NVFP4) | 40 |
| `decode_mxfp4` (16-entry LUT, `torch.ldexp`) | 12 |
| `read_safetensors` (parse the 8-byte header length and JSON, issue `Range` requests with `huggingface_hub`'s HTTP session or `HfFileSystem`) | 25 |
| `qat_export`, `ternary_absmean` | 15 |

Numerical tests:
1. D1 checks (i)–(iv).
2. `FakeQuantLinear(bits=4, group_size=128)` forward equals `d2l.fake_quant` (11.1) on the same weight bit for bit; at `bits=16`-equivalent clip it equals the original linear.
3. `quant_mx('mxfp4')` on the 16 representable values times `2^e` returns them exactly; `quant_mx` output has at most 16 distinct values per block after scaling; compare with a slow Python reference of the OCP rule on 1000 random blocks (exact).
4. `decode_mxfp4`: for a handcrafted block with nibble `0x2 | (0x7 << 4)` and scale byte 127 the output is `[1.0, 6.0]` (low nibble first, LUT `[0, .5, 1, 1.5, 2, 3, 4, 6, −0, …, −6]`); scale 128 doubles it.
5. `read_safetensors` returns the same bytes as `safetensors.safe_open` on a small local file served over `file://` or a local HTTP server (T0).
6. `qat_export` conversion check: logits of the exported model equal the fake-quantized model's to 1e-5 relative (bf16-aware: 2e-2 absolute at the logit scale in bf16).
7. Training smoke test: 20 steps of QAT on a 2-layer random `d2l.GPT` reduce the training loss, and the latent weights change while the codes of at least some groups flip.

#### In production

- torchao QAT: `quantize_(model, IntXQuantizationAwareTrainingConfig(...))`-style prepare/convert flow (class names move across releases; follow the "Fine-tuning with QAT, QLoRA, and float8" page of the pinned documentation, https://docs.pytorch.org/ao/stable/eager_tutorials/finetuning.html), then `convert` to the packed int4 representation that the 11.1 kernels read. The from-scratch `FakeQuantLinear`/`qat_export` is the oracle: same fake-quantized logits (within bf16 tolerance) before convert, and the converted model's logits within the paired interval of perplexity.
- NVIDIA Model Optimizer and Transformer Engine for FP8/NVFP4 QAT on Hopper/Blackwell; Transformers' `MXFP4` and `NVFP4` quantizer pages for loading; `llm-compressor` for QAT-adjacent recipes. All named, not run on A100.
- gpt-oss on Ada: the Transformers path pulls Triton MXFP4 kernels from `kernels-community/triton_kernels`; it stores weights as MXFP4 and dequantizes inside the kernel. It is "native storage, emulated arithmetic" on Ada and A100, and fully native only on Blackwell.

#### Headline experiment

*Hypothesis.* At a matched budget of data, tokens and optimizer steps, QAT on full weights recovers more of the BF16 perplexity than PTQ (RTN or GPTQ) and, at 3 bits more than at 4 bits, more than adapters on frozen quantized weights at equal steps; the BF16 control shows how much of any gain is just domain adaptation. Weights stay close to initialization at 3–4 bits.

*Setup.* Model: Qwen3-0.6B (`d2l.load_qwen3`), fp32 master weights, bf16 autocast, activation checkpointing. Format: INT4 g128 symmetric (main); INT3 g128 (extension). Data: FineWeb-Edu training slice, sequence length 1024, batch 8, `STEPS` set so one run is ≈ 2 min (≈ 8k tokens × `STEPS`; the papers use orders of magnitude more tokens, so this measures a small-budget recovery, not the asymptote). AdamW, linear warmup 20 steps then cosine to 10%, weight decay 0, gradient clip 1.0, seed 0. Runs: (A) BF16 base; (B) BF16 control, trained with the same loop (lr 1e-5); (C) RTN; (D) GPTQ (11.3, 128 calibration sequences); (E) PTQ(GPTQ) + LoRA r=32 on all linears (PEFT), lr ∈ {1e-4, 5e-4}; (F) QAT full weights (STE, absmax scale), lr ∈ {5e-6, 2e-5}. The learning rates are chosen on a held-out FineWeb-Edu validation slice (not on WikiText-2), the same two-point grid size for E and F. Seeds: seed 0 for the grid; seed 1 repeated for the best E and best F. Evaluation: WikiText-2 test and held-out FineWeb-Edu windows, `EvalSuite` tasks.

*Procedure.* Evaluate A, C, D; train B, E, F; evaluate each; for F log per-layer displacement `‖W − W₀‖_F/‖W₀‖_F` and the fraction of weights whose code changed from the initial RTN codes; export F with `qat_export`, verify conversion, and time the packed model with 11.1's torchao int4 path for one row. With `FULL=True`, repeat C, D, E, F at INT3 g128 with the learning rates picked at INT4.

*Metrics.* Perplexity with paired 95% bootstrap intervals (over windows) between each method and RTN, and between F and E; task scores; training loss curves; displacement and flipped-code fraction; wall-clock and peak memory of E vs F (the cost side of the comparison); effective bits (identical for C–F by construction: 4.125).

*Plots.*
- `fig_lm_qat_curves`: x = training step, y = held-out FineWeb-Edu loss (every 25 steps); series B, E, F with the horizontal levels of A, C, D.
- `fig_lm_qat_bars`: WikiText-2 perplexity bars with paired-interval error bars for A–F at INT4 (and INT3 under `FULL`).
- `fig_lm_qat_disp`: per-layer relative weight displacement, one line per linear type.
- `tab_lm_qat_main`: method × (ppl ± CI, Δ vs RTN, task score, minutes, peak GiB, trainable parameters).

*Expected outcome.* Directions only. LLM-QAT (arXiv 2305.17888) reports large improvements over training-free methods at low bit-widths on LLaMA 7B–30B; ParetoQ (arXiv 2502.02631) compares QAT across 1–4 bits and reports that weights stay near the pretrained distribution at 3 bits and above. Whether a ≈ 2-minute budget on Qwen3-0.6B reproduces an advantage over GPTQ at INT4 is to be measured, and the answer "QAT does not beat GPTQ at 4 bits within this budget" is an acceptable result if the intervals say so. The control B is expected to move held-out FineWeb-Edu loss down and may move WikiText-2 perplexity either way; this is why the comparison is to B as well as to A.

*Noise check.* Paired bootstrap over windows for each perplexity difference; a second training seed for the best E and F; the ordering of E and F is claimed only if both seeds and the paired interval agree. Learning-rate choice is made on validation, not on the reported test set.

*Artifacts committed.* Training logs (JSON lines, loss and eval), the best QAT checkpoint codes/scales (optional, ≈ 0.4 GB, not committed if > 100 MB), `data/compression_report.jsonl` rows, the gpt-oss tensor-inspection table (`lm-c11-gptqoss-l0-stats.json`, a few KB).

#### Figures and tables

- `fig_lm_qat_ste` (T0): one quantizer's staircase, the true derivative (0), STE derivative, and the LSQ scale derivative, for `b = 3` over `u ∈ [−6, 6]`.
- `fig_lm_qat_curves`, `fig_lm_qat_bars`, `fig_lm_qat_disp`, `tab_lm_qat_alternatives`, `tab_lm_qat_main`.
- `fig_lm_mx_mismatch`: `E(κ)` for block 32 and 16 with the MXFP4 and NVFP4 `κ` ranges shaded.
- `tab_lm_mx_formats`: element format, block, scale format, effective bits, native hardware: MXFP8, MXFP4, NVFP4, INT4 g64/g32 for comparison.
- `fig_lm_mx_gptoss`: code histogram, exponent histogram of layer-0 experts; `tab_lm_mx_gptoss`: per-tensor bytes (blocks, scales, biases) and the checkpoint's effective bits.
- `tab_lm_mx_ptq`: PTQ perplexity of MXFP4, NVFP4, INT4 g64/g32 on Qwen3-0.6B and 1.7B (weights only, simulated).

#### Pitfalls and misconceptions

- Comparing QAT to PTQ without a BF16 control trained on the same tokens credits domain adaptation to quantization-awareness.
- Choosing the learning rate on the test set; using different data orders for compared methods.
- QAT and LoRA-on-quantized-weights are different: the adapter cannot change the grid assignment of the base weights.
- Fake-quantized perplexity is not a deployed result: convert and check the packed model, then time it (11.1 kernels).
- Treating "MXFP4" as one number: block size and scale format determine the accuracy; MXFP4 and NVFP4 differ in both.
- Calling the Ada or A100 run "native FP4": arithmetic is emulated; at best storage is native.
- gpt-oss's MXFP4 weights have no BF16 twin; comparing a re-quantized copy to them measures the simulator, not the training recipe.
- Ternary PTQ of a model trained at 16 bits is not BitNet; those models are trained with the quantizer in the loop.
- LSQ and absmax-detached scale are different estimators; say which one a number comes from.

#### Summary (draft bullets)

- The derivative of rounding is zero; the straight-through estimator pretends it is one inside the clipping range, which is enough to train through the grid.
- QAT keeps a latent full-precision weight and fake-quantizes it in the forward pass; its cost is a training run, its output is a model for the same packed kernel as PTQ.
- A fair QAT-vs-PTQ comparison matches tokens and steps and includes a BF16 control and a frozen-weights-plus-adapters baseline.
- Block-scaled 4-bit formats trade element precision for per-block scales; the scale format (power of two vs E4M3) and block size set the error.
- Native MXFP4 releases such as gpt-oss store only the expert matrices in 4 bits; the arithmetic is native only on Blackwell and emulated elsewhere.

#### Exercises

1. *(check)* In D1, why is the weight gradient identity when `clip = 1` with absmax scaling, and what changes with a learned scale? *Sketch:* no weight exceeds `p·s`; with a learned `s`, entries outside `[−n, p]·s` are clipped and get zero weight-gradient but nonzero scale-gradient.
2. *(check)* A model is released "in MXFP4". Which tensors are 4-bit, what else must you download in 16 bits, and what is the effective bits per parameter of the whole file? *Sketch:* expert matrices only; attention, router, embeddings, head, norms, biases in BF16; compute from the header (experts at 4.25 b).
3. *(derive)* Show the loss of the MX rule is a function of `κ = 6/r` and compute the clip fraction of the block maximum when `r` is uniform on `[4, 8)` in log scale. *Sketch:* `P(r ≥ 6) = log₂(8/6) = 0.415`; for Gaussian blocks of 32 the empirical share is lower (the block maximum is not log-uniform): the gap is the point of the cell.
4. *(derive)* Derive the optimal scale of a ternary quantizer `Ŵ = γ·t`, `t ∈ {−1, 0, 1}`, for a Gaussian weight under MSE, as a joint choice of threshold `τ` and `γ`, and compare with the absmean rule `γ = mean|W|`. *Sketch:* for a given threshold `τ`, `γ* = E[|w| | |w| > τ]`, and MSE-optimality of the threshold requires `τ = γ/2`; the fixed point is `τ ≈ 0.61σ`, `γ ≈ 1.22σ`, with MSE ≈ 0.19σ² (a CPU check on 10⁶ Gaussian samples). The absmean rule has `γ = mean|W| ≈ 0.80σ` and threshold `0.40σ`, MSE ≈ 0.26σ²: a different operating point, chosen for training behaviour rather than for MSE.
5. *(measure)* Plot the fraction of weights whose INT4 code changed (vs the initial RTN codes) as a function of training step for QAT at two learning rates; predict whether the curve saturates and measure. Relate to the displacement figure.
6. *(measure)* Replace the STE quantizer by `fmt='mxfp4'` and run 100 steps of recovery on Qwen3-0.6B starting from the MXFP4 PTQ model; predict how much of the PTQ gap closes (relative to INT4 g64 at 4.25 bits) and measure with paired intervals.
7. *(open)* Distill instead of fitting next-token loss: QAT with a KL loss against the frozen BF16 model on the same tokens (9.1) and, separately, on text generated by the BF16 model itself (the data-free recipe of LLM-QAT). Which helps more at INT3? Report with intervals.

#### Further reading

- Bengio, Léonard, Courville, [Estimating or Propagating Gradients Through Stochastic Neurons](https://arxiv.org/abs/1308.3432): the straight-through estimator.
- Liu et al., [LLM-QAT](https://arxiv.org/abs/2305.17888); Liu et al., [ParetoQ](https://arxiv.org/abs/2502.02631): QAT recipes and the 2-to-3-bit transition.
- Rouhani et al., [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537): the MX formats; the OCP specification PDF linked in D2; NVIDIA, [NVFP4 trains with precision of 16-bit](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/) and [Pretraining LLMs with NVFP4](https://arxiv.org/abs/2509.25149).
- OpenAI, [gpt-oss model card](https://arxiv.org/abs/2508.10925) and the [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) repository; Transformers' [MXFP4 integration](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/mxfp4.py) as the reference decoder.
- Ma et al., [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764); [BitNet inference code](https://github.com/microsoft/BitNet).
- Code and docs: [torchao](https://github.com/pytorch/ao) QAT, [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine).
- Teaching: MIT 6.5940 (quantization-aware training lecture and lab).

#### Risks and open decisions

- *Training budget.* ≈ 2 minutes per run is a small budget; the section says so and reports an intervalled comparison, not a general verdict. Fallback: if five runs exceed 15 minutes, drop the learning-rate grid to one point per method and say so.
- *Memory for the 0.6B full fine-tune.* If the 14 GiB peak is exceeded, use an 8-bit AdamW (7.6) or a smaller batch with gradient accumulation; keep the matched-budget rule.
- *torchao QAT API drift.* The from-scratch path carries the headline; the torchao path is a conversion cross-check, gated by `try/except`.
- *gpt-oss end-to-end on Ada* needs about 13–14 GiB plus KV cache and Triton ≥ 3.4: marginal in a 15 GiB slot, comfortable on a whole 24 GB GPU. The tensor-level inspection (≈ 0.45 GB of range reads) is the primary deliverable; the end-to-end cell is optional, guarded by `torch.cuda.mem_get_info()`.
- *`read_safetensors` and Hub rate limits or Xet-backed storage* may reject range requests on some files; fallback: `huggingface_hub.hf_hub_download` of the shard to a scratch directory (about 5 GB for shard 0; the notebook prints the size and asks for confirmation).
- *OCP MX conversion rule* quoted from memory of the spec; the check against the PDF is a listed task for the author. If the spec differs (e.g. in rounding or the treatment of block maxima), only the `quant_mx('mxfp4')` scale rule and D2 step 5 change.

### 11.6 Pruning and lossless compression

- **Label:** `sec_lm_pruning` · **Tier:** T1 (`FAST=True`: 40 evaluation windows, 16 calibration sequences, 100 recovery steps) · **Budget:** ≤ 15 min (pruning passes ≈ 6 min, one structured recovery ≈ 6 min, kernel timing and coding ≈ 3 min; `STEPS` set from measured seconds per step), peak ≤ 12 GiB (a 1.7B BF16 teacher 3.4 GB + a pruned student ≈ 2.8 GB + LoRA/activations; the DF11 comparison loads 4B BF16 at 8 GB and then its DF11 twin, one at a time) · **Codex units:** 12.7, 12.8 (and the deployment-report idea of 12.9)
- **Builds on (book):**
  - `sec_numerics`, `sec_memory_precision`: BF16 layout 1+8+7 (used, not re-taught).
  - `sec_hardware`: tensor-core throughput tables (dense numbers; 2:4 is the sparse doubling cited from the A100 datasheet).
  - `sec_mdl-svd-low-rank`: Eckart–Young (`eq_mdl-eckart-young`) for the low-rank factorization baseline.
  - `sec_mdl-information_theory`: entropy, prefix codes and the Huffman bound `H ≤ L < H + 1`.
  - `sec_mdl-numerical-stability-conditioning`: via 11.3's damping.
- **Builds on (this part):** 11.1 (`d2l.window_nll`, `d2l.calib_tokens`, `d2l.decode_tps`, `d2l.compression_row`, `d2l.replace_linears`, `d2l.effective_bits`), 11.2 (`d2l.collect_input_stats` for the activation norms Wanda needs), 11.3 (`d2l.accumulate_hessian`; the OBS derivation `D1` and the Cholesky sweep `D2` of `:numref:`sec_lm_gptq``, **not** re-derived), 9.1 (`:numref:`sec_lm_distillation``: the distillation loss; here `d2l.kd_loss(student_logits, teacher_logits, mask, kind='fkl', alpha=0.5, kappa=1.0, chunk=4096)` from 9.1, imported, not redefined; `kappa` is the temperature), 7.4/7.5 (PEFT LoRA for recovery), 1.5 (`d2l.paired_bootstrap`, `d2l.EvalSuite`).
- **Exports** (new names):
  - `d2l.prune_masks(score, pattern='unstructured', sparsity=0.5)`: boolean keep-mask `[d_out, d_in]` from a score matrix of the same shape; `pattern='unstructured'` keeps the top `(1−sparsity)` fraction **per output row** (Wanda's comparison group), `'2:4'` (or any `'n:m'`) keeps the top `n` of every `m` consecutive input entries per row.
  - `d2l.wanda_prune(W, act_norm, pattern='unstructured', sparsity=0.5)`: score `|W|·act_norm[None, :]` with `act_norm_j = ‖X_j‖₂` over calibration tokens; returns `(W_pruned, mask)`; no weight update.
  - `d2l.sparsegpt_prune(W, H, pattern='unstructured', sparsity=0.5, blocksize=128, damp=0.01)`: OBS-based pruning with the Cholesky sweep of 11.3; returns `(W_pruned, mask)` with masked entries set exactly to zero after the sweep.
  - `d2l.to_semi_structured(model, names=None)`: converts the selected linears (BF16/FP16, CUDA, 2:4-conformant weights) to `torch.sparse.SparseSemiStructuredTensor` with the available backend; returns the list of converted names and the backend used.
  - `d2l.block_influence(model, ids)`: per-block `1 − cos(input, output)` of the residual stream averaged over calibration tokens (a depth-importance score).
  - `d2l.prune_mlp_width(model, importance, keep)` / `d2l.prune_depth(model, drop)`: structured surgery for `d2l.GPT`/Hugging Face Qwen3: remove MLP intermediate neurons (rows of gate and up, columns of down) or whole blocks; returns a smaller dense model and its config.
  - `d2l.bf16_fields(w)`: sign, exponent, mantissa fields of a BF16 tensor as `uint8` arrays; `d2l.entropy_bits(counts)`.
  - `d2l.df11_encode(w, block=256)` / `d2l.df11_decode(enc)`: exact exponent coding. `enc` holds the raw `uint8` sign+mantissa bytes, a length-limited canonical Huffman table for the 8-bit exponents, the packed exponent bitstream and per-block bit offsets; decoding is vectorized across blocks (a `block`-step loop of table lookups). Bit-exact by construction and by test.
- **Models and data:** `Qwen/Qwen3-1.7B` (pruning, kernels, exponent statistics); `Qwen/Qwen3-4B` and `DFloat11/Qwen3-4B-DF11` (existence verified on the Hub; production DF11 comparison); calibration and evaluation as 11.1.
- **Packages:** `torch ≥ 2.6` (`torch.sparse.to_sparse_semi_structured`; CUDA, compute capability ≥ 8.0 per PyTorch's semi-structured sparsity documentation, https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html), `peft`, `dfloat11` (https://github.com/LeanModels/DFloat11, CUDA build; optional), `numpy`; optional NVIDIA Model Optimizer for the pruning-and-distillation reference (https://github.com/NVIDIA/Model-Optimizer).

#### Narrative arc

1. **The question.** *Half of the weights are zero; why is nothing faster?* Cell 1: Wanda at 50% unstructured on one MLP matrix of Qwen3-1.7B, then time the dense BF16 matmul with the pruned matrix at `B = 1` and `B = 2048` (identical to the unpruned matrix to timing noise), and print the sparsity, the nonzero count, and the bytes (unchanged). Cell 2: prune the same matrix 2:4 and convert with `to_semi_structured`; print the timings at the same batches. The symptom: parameter removal and realized acceleration are different things (to be measured).
2. **What "pruning" can mean** (`tab_lm_prune_taxonomy`; the key table of the section). Four granularities and the question each answers:

   | Pattern | Stored | Decode (bandwidth-bound) | Prefill (compute-bound) | Needs |
   |---|---|---|---|---|
   | Unstructured 50% | dense unless sparse format | no change on dense kernels | none | a sparse format and kernel to benefit |
   | 2:4 semi-structured | 0.5625 of BF16 bytes (D2) | up to 1.78× (bytes) | up to 2× on sparse tensor cores | Ampere-or-later sparse MMA, cuSPARSELt/CUTLASS |
   | Structured width (neurons, heads) | proportional | proportional (dense, smaller GEMMs) | proportional | nothing special |
   | Structured depth (whole blocks) | proportional | proportional (fewer layers, also fewer KV bytes) | proportional | nothing special |
   | Lossless (exponent coding) | about 0.7 of BF16 bytes | up to 1/0.7 if decompression is hidden | none | a decode kernel |

   Distinguish from the low-rank alternative (one paragraph and `fig_lm_prune_svd`: truncated SVD of the MLP matrices at matched parameters, with the singular spectrum showing why it fares poorly without recovery; to be measured).
3. **Which weights to remove: magnitude, Wanda, SparseGPT** (derivation D1). Magnitude pruning per layer vs per output row; why activation norms matter (the same outlier channels as 11.2); Wanda's score and its OBS interpretation; SparseGPT as GPTQ with the grid `{0}` for the masked entries (one sentence per component of 11.3 reused: Hessian, damping, Cholesky, lazy blocks) plus the new part, adaptive mask selection within each block of columns. Cell: the three methods on a small layer, `fig_lm_prune_toy`: proxy loss `trace((W−Ŵ)H(W−Ŵ)ᵀ)`.
4. **Unstructured and 2:4 pruning of Qwen3-1.7B** (headline, part A). Sparsity sweep for magnitude and Wanda (30–70%); 50% unstructured and 2:4 for all three methods; `fig_lm_prune_sweep`.
5. **Does 2:4 run faster?** (derivation D2; headline, part B). Layer microbenchmarks at the MLP shapes for `B ∈ {1, 16, 256, 2048}`, then whole-model prefill and decode with the MLPs converted. Report which parts are native on the A100 (sparse tensor cores, datasheet), which on Ada (verified by timing in this cell; not assumed), and that the 2:4 benefit is limited to GEMMs of a minimum size.
6. **Structured pruning and distillation recovery** (headline, part C). Width: MLP intermediate neurons by an activation-based importance (mean |SwiGLU product| over calibration tokens); depth: drop blocks of lowest block influence. Matched body-parameter reduction (about 18%: MLP 6144 → 4608 vs 5 of 28 blocks dropped). No-recovery perplexity; then recovery by distillation from the original (forward KL on logits plus next-token loss) with LoRA (r=64) on the pruned student at a fixed token budget; Minitron's recipe in miniature. Dense kernels need nothing special: measure real decode and prefill speed.
7. **Lossless compression of BF16 weights** (derivation D3). Fields of BF16; the exponent histogram of real weights; entropy, Huffman code lengths and the size ratio `(8 + L)/16`; the encoder/decoder, bit-exact; the comparison of format-dependent empirical entropy to a "universal bound" (it is not one: it depends on the representation and on the model). Cell: the entropy of RTN INT4 codes of a layer, to show lossless coding on top of lossy codes (to be measured). Production: `DFloat11/Qwen3-4B-DF11` vs `Qwen/Qwen3-4B`: identical greedy outputs, memory, load time, prefill and decode speed; lossless means identical weights, and identical outputs also need an identical kernel and numerical path (the notebook checks and reports it).
8. **Putting the chapter together.** Load `data/compression_report.jsonl`; `fig_lm_comp_frontier` (weight GiB vs perplexity with paired CI bars, marker = method, fill = native / emulated / simulated); `tab_lm_comp_all`; `tab_lm_comp_decision`: for four deployment targets (batch 1 on a 24 GB RTX 4090; batch 64 serving on an A100; CPU/laptop; Blackwell with native FP4), which of PTQ weight-only, W8A8/FP8, GPTQ/AWQ, rotations, QAT, native MXFP4, pruning, lossless coding the chapter's evidence supports, and why. Everything in the decision table cites a row of the report or a derivation; a claim without a measured row is marked "not measured here".

#### Derivations

**D1. Wanda is OBS under a diagonal Hessian; SparseGPT is GPTQ with the grid `{0}`.**

*Statement.* For one output row `w` with reconstruction loss `(w−ŵ)ᵀH(w−ŵ)`, removing weight `j` (setting it to 0) and optimally compensating the others costs `w_j² / [H⁻¹]_jj` (11.3 D1 with `c = 0`). If `H` is replaced by its diagonal, `[H⁻¹]_jj = 1/H_jj = 1/‖X_j‖²`, and the cost is `w_j² ‖X_j‖²`, so ranking weights by `|w_j|·‖X_j‖₂` ranks them by OBS saliency without compensation: this is Wanda's score. SparseGPT runs the 11.3 sweep with `Q(w_q) ∈ {0, w_q}` (zero for masked entries), so its update and loss are those of D2 in 11.3 with `err_q = w_q/U_qq` for masked `q`.

*Steps.*
1. 11.3 D1: fixing `w_q` to `c = 0` increases the loss by `Δ²/[H_F⁻¹]_qq` with `Δ = w_q`, the update to the rest is `−(w_q/[H_F⁻¹]_qq) H_F⁻¹ e_q`.
2. First step (`F` = all coordinates): `cost_j = w_j²/[H⁻¹]_jj`. Without any compensation, zeroing `w_j` alone costs exactly `w_j² H_jj = w_j²‖X_j‖²` for any `H` (the quadratic form of `−w_j e_j`); this is Wanda's score as an exact statement about one weight, and it equals the OBS cost when `H` is diagonal.
3. With `H = diag(‖X_j‖²)`: `H⁻¹ = diag(1/‖X_j‖²)` and `cost_j = w_j²‖X_j‖²` and the compensation vector `H⁻¹e_j` is supported on `j` alone: no other weight moves, matching Wanda's "no weight update".
4. Rows are independent, so the choice of which `k` weights to remove can be made per row (Wanda's per-output comparison group); a layer-wise threshold mixes rows with different scales.
5. SparseGPT: within each block of `B` columns choose the mask by the smallest `w_j²/U_jj²` (the value of `[H_F⁻¹]_jj` in the sweep is `U_jj²`, 11.3 D2 step 3), then sweep the block applying `err_q = (w_q − 0·1[masked])/U_qq` and `w_{q:} −= err_q U_{q,q:}`; masked entries end at exactly zero up to round-off, which the code sets explicitly.
6. n:m patterns constrain the mask only: within each group of `m` columns keep the `n` best by the same score (Wanda) or by the block-wise OBS score (SparseGPT).

*Numerical check (cell, float64, seed 0).* (i) With a diagonal `H`, OBS saliency `w_j²/[H⁻¹]_jj` equals `(|w_j|·‖X_j‖)²` to 1e-12, and the pruned set is identical to Wanda's. (ii) On random layers (`m=32`, `n=64`, 50%, correlated inputs with log-normal column scales, 10 seeds) the proxy loss ordering magnitude > Wanda holds for every seed, and the mean ordering magnitude > Wanda > SparseGPT holds (a CPU dry run gave about 1170 > 268 > 240 and SparseGPT < Wanda in 8 of 10 seeds, so assert the means and the first inequality, not SparseGPT < Wanda per seed). (iii) Exact sparsity of the SparseGPT output equals the requested one only after the explicit zeroing (float round-off leaves values near `1e-18`: the test asserts `(Ŵ == 0).mean() == sparsity` after `sparsegpt_prune`). (iv) `sparsegpt_prune` with `sparsity=0` returns `W` unchanged.

**D2. What 2:4 can and cannot buy.**

*Statement.* A 2:4 sparse BF16 matrix stores, per group of 4 weights, 2 values (32 bits) and 4 bits of metadata (2 bits of position per kept value), i.e. 9 bits per original weight, `9/16 = 0.5625` of the dense bytes; INT8 values give `(16 + 4)/32 = 0.625`. The decode speed bound for a bandwidth-bound layer is therefore `≤ 1/0.5625 = 1.78×`, and the compute bound is `≤ 2×`. The information-theoretic metadata minimum is `log₂ C(4,2) = log₂ 6 = 2.585` bits per group, about 35% below the 4 bits the hardware format stores.

*Assumptions.* Hardware metadata of 2 bits per kept element (Mishra et al., arXiv 2104.08378); the sparse tensor core processes a 2:4 tile in the time of a dense tile of half the arithmetic (the paper: twice the math throughput); the A100 datasheet lists BF16 312 TFLOPS dense and 624 with sparsity, INT8 624 and 1,248 TOPS.

*Steps.*
1. Bytes per weight: `(2·16 + 2·2)/4 = 9` bits for BF16.
2. Bandwidth-bound time ratio `= 9/16`, speedup `≤ 16/9 = 1.78`.
3. Compute-bound ratio: peak 2× from the datasheet pair (312 → 624).
4. Roofline with the sparse peak: the ridge batch of a 2:4 BF16 layer is `B* = π_sparse · w / (2β)` with `w = 9/8` bytes per weight and `π_sparse = 2π`, i.e. `B*_{2:4} = 2π·(9/8)/(2β) = 1.125·π/β`, about 1.125 times the dense ridge: 2:4 is a bandwidth win at decode and a compute win only above the dense ridge batch.
5. Realized speed is lower: kernels need minimum matrix sizes and alignment (`K`, `N` multiples of 8 or 16), batch rows ≥ a library minimum, and quality loss from 2:4 is larger than from unstructured 50% at the same sparsity (measured in the headline).
6. Unstructured 50% on a dense kernel: bytes and FLOPs unchanged, speedup exactly 1 in the model.

*Numerical check (cell).* Pack a random 2:4 matrix with `to_semi_structured`; assert (i) its compressed `nbytes` (values + metadata, as exposed by the tensor) equals `0.5625 ×` the dense bytes within 1% (to be verified for the backend in use; if the backend pads metadata, report the ratio instead); (ii) `sparse_linear(x) == dense_linear(x · mask)` within BF16 tolerance for the same masked weights; (iii) the formulas of steps 1–4 evaluated for the datasheet numbers (printed).

**D3. How small can BF16 weights get without loss? Entropy of the exponent field and Huffman length.**

*Statement.* BF16 stores sign (1), exponent (8), mantissa (7). If the exponent field `E` of the weights of a tensor has empirical distribution `p`, an optimal prefix code has average length `L` with `H(p) ≤ L < H(p)+1`, so the lossless size is `(1 + 7 + L)/16` of the BF16 size, with sign and mantissa stored raw. For `w ~ N(0, σ²)` the exponent field concentrates on about 10 adjacent values of `⌊log₂|w|⌋` around `log₂ σ`, whose entropy is about 2.5 bits.

*Steps.*
1. `|w| = 1.m × 2^{E−127}` with `m` the 7-bit mantissa; for a Gaussian, `P(E = e) = P(2^{e−127} ≤ |w| < 2^{e−126})`, a unimodal distribution whose mass sits on a few exponents.
2. The mantissa of a smooth density is close to uniform over a binade (the density varies slowly within `[2^e, 2^{e+1})` away from the tails), so it has close to 7 bits of entropy and is left raw; the sign is a fair coin.
3. Huffman on the exponent histogram achieves `L < H + 1`; a canonical code is described by the 256 code lengths; with a maximum length `L_max` the decoder table has `2^{L_max}` entries (hierarchical LUTs for long codes, as in DFloat11's GPU kernel).
4. Total bits per weight `8 + L ≈ 10.5–11` (hence "DFloat11"); the saving is `1 − (8+L)/16 ≈ 30%` when `L ≈ 2.6–3`, matching the paper's headline of a 30% size reduction (arXiv 2504.11651).
5. The bound is for *this* representation (a BF16 tensor with its empirical exponent histogram), not a property of the model; a model stored in FP8 has a different (smaller) redundancy, and INT4 codes of a Gaussian weight have entropy below 4 bits for a different reason.
6. Per-block bit offsets let a decoder work on blocks independently (parallel decoding); the cost is the offset table, `32 bits / block`.

*Numerical check (cell, CPU, seed 0).* (i) For `10⁷` samples of `N(0, 0.02²)` rounded to BF16: exponent entropy within `[2.3, 2.8]` bits (a CPU dry run gave 2.55) and the Huffman average length within `[H, H+1)` (dry run 2.59, ratio 0.662). (ii) On a real Qwen3-1.7B tensor: the same numbers (to be measured), and the encode→decode round trip `torch.equal` on `4·10⁶` weights. (iii) `df11_encode` output size equals `8·n + Σ_e count_e·len_e` bits plus the offset table, within 0.1%.

#### From-scratch implementation

Budget ≈ 150 lines.

| Component | Lines |
|---|---|
| `prune_masks` (per-row top-k, n:m by reshape and `topk`), `wanda_prune` | 20 |
| `sparsegpt_prune` (adaptive block masks + the 11.3 sweep with grid `{0}`) | 40 |
| `to_semi_structured` | 15 |
| `block_influence`, `prune_mlp_width`, `prune_depth` (config update included) | 35 |
| recovery loop with PEFT, calling `d2l.kd_loss` (9.1) | 15 (loop outside the budget) |
| `bf16_fields`, `entropy_bits`, canonical Huffman lengths (package-merge-free: `heapq` then length-limit by flattening), `df11_encode`, `df11_decode` | 50 |

Numerical tests:
1. D1 checks (i)–(iv).
2. `prune_masks('2:4')`: every group of 4 consecutive input entries in every row has exactly 2 kept; `'unstructured'` keeps exactly `⌈(1−s)·n⌉` per row.
3. `prune_mlp_width` with `keep=1.0` returns bit-identical logits; with `keep<1` the model runs and its parameter count equals `L·3·d·d_ff'` for the MLPs plus the unchanged attention and embedding terms; `prune_depth(drop=[])` is the identity.
4. Structured pruning preserves the function exactly when the removed neurons are zero (zero out neurons first, prune, compare logits: 1e-5).
5. `df11_decode(df11_encode(w)) == w` bitwise for `n ∈ {1, 255, 256, 257, 10⁶}` including subnormals, `±0`, and `inf`/`nan` patterns (as raw bits).
6. D2 and D3 checks.

#### In production

- NVIDIA Model Optimizer and the Minitron code (https://github.com/NVlabs/Minitron; NVIDIA's blog "Pruning and Distilling LLMs Using NVIDIA TensorRT Model Optimizer") for width/depth pruning with distillation at scale; Wanda's reference code (https://github.com/locuslab/wanda); SparseGPT's reference code (linked from the paper); `torch.sparse` semi-structured and torchao's sparsity utilities for 2:4 inference; `DFloat11` (https://github.com/LeanModels/DFloat11) for GPU decoding of the exponent code, ZipNN (arXiv 2411.05239) for the file-level variant.
- The from-scratch scores and masks are the oracle: production Wanda masks must equal `wanda_prune`'s on the same activation norms (test on one layer), and the production DF11 checkpoint decoded on CPU through `df11_decode`'s table construction must reproduce the BF16 tensor exactly for one weight matrix (optional cell, needs the DF11 table format; otherwise the whole-model greedy-output equality test is the check).

#### Headline experiment

*Hypothesis.* (A) At 50% sparsity, Wanda keeps perplexity far closer to dense than magnitude pruning, SparseGPT is at least as good as Wanda on average, and 2:4 costs more perplexity than unstructured 50% at equal sparsity. (B) Unstructured 50% gives no speed-up on dense kernels; 2:4 gives a speed-up at large batch or prefill but little or none at batch 1 decode in this stack. (C) At matched removed parameters, structured width and depth pruning without recovery are worse than unstructured pruning at the same parameter count, and distillation recovery closes a large part of the gap; the pruned dense models are faster in proportion to the parameters removed. (D) Exponent coding is exactly lossless and reduces weight bytes by roughly 30% (as reported for DFloat11).

*Setup.* Qwen3-1.7B BF16; calibration 128 × 2048 FineWeb-Edu tokens (16 under `FAST`); activation norms from `collect_input_stats` (11.2) and Hessians from `accumulate_hessian` (11.3). Part A: magnitude and Wanda at sparsity {30, 40, 50, 60, 70}%; magnitude, Wanda and SparseGPT at 50% unstructured and at 2:4 (28 blocks, true-sequential for SparseGPT). Part B: MLP linears (gate, up, down) converted for 2:4; layer timing at `B ∈ {1, 16, 256, 2048}` (BF16 dense, dense with 50% zeros, 2:4 sparse) and whole-model prefill (2048 tokens) and decode (batch 1) with `d2l.decode_tps`. Part C: width (keep 4608 of 6144 MLP neurons, importance = mean |SwiGLU product|) and depth (drop 5 blocks of lowest block influence), evaluated without recovery; recovery with LoRA r=64 on all linears of the student (PEFT), loss = `d2l.kd_loss(student_logits, teacher_logits, mask, kind='fkl', alpha=0.5, kappa=1.0)` to the original model on the logits + 0.1× next-token CE (the CE weight is a stated choice; the exercises vary it), `STEPS` steps of 8 × 1024 tokens, AdamW lr 2e-4, cosine, seed 0, both configurations if time permits (`FULL`), else width only. Part D: exponent statistics for every tensor type of Qwen3-1.7B; `df11_encode`/`decode` on one 4M-weight slice; whole-model ratio from histograms; production comparison of `Qwen/Qwen3-4B` vs `DFloat11/Qwen3-4B-DF11` with 8 prompts × 64 greedy tokens.

*Procedure.* Verify all tests; run A; run B; run C; run D; then build the chapter table (`tab_lm_comp_all`) from the JSONL file. All speed numbers state "native" (dense/BF16, 2:4 on the sparse tensor cores where the backend uses them) or "emulated" (none here); no simulation is involved in this section's speeds.

*Metrics.* Perplexity with paired 95% intervals (against dense and between methods); task scores; sparsity achieved; bytes (dense, 2:4 compressed, structured, DF11); layer and model speed with IQR; for C, tokens and wall-clock of recovery; for D, exact equality of all weights and of the 8 × 64 generated tokens, load time, peak memory.

*Plots.*
- `fig_lm_prune_sweep`: x = sparsity (%), y = WikiText-2 perplexity (log); lines magnitude and Wanda, markers for SparseGPT and for 2:4 at 50%; band = paired interval to dense.
- `fig_lm_prune_speed`: x = `B`, y = time speedup vs dense BF16 (log); lines dense-with-zeros, 2:4; model curves from D2.
- `fig_lm_prune_recovery`: x = recovery step, y = held-out loss; series width and depth; horizontal levels for the teacher and for the no-recovery students.
- `fig_lm_prune_exp`: exponent histograms per tensor type with entropy annotated; `tab_lm_prune_df11`: bits per weight per tensor type and whole-model ratio.

*Expected outcome.* Wanda reports large gains over magnitude pruning on LLaMA and performance competitive with SparseGPT without weight updates (arXiv 2306.11695); SparseGPT reports 50–60% unstructured sparsity at small perplexity cost on OPT-175B and BLOOM-176B and extension to 2:4 and 4:8 (arXiv 2301.00774); Mishra et al. report 2× math throughput for 2:4 sparse tensor cores (arXiv 2104.08378); Minitron reports compressing models by 2–4× with under 3% of the original training data, up to 40× fewer tokens per derived model, using width/depth pruning plus distillation retraining (arXiv 2407.14679); DFloat11 reports 30% lossless size reduction (arXiv 2504.11651). For Qwen3-1.7B, whose parameters per layer are smaller than those models', the magnitudes of all of these are to be measured; the recovery budget here is a small fraction of the papers'.

*Noise check.* Paired bootstrap over windows for perplexity differences; calibration draw repeated (seed 1) for Wanda and SparseGPT at 50%; the 2:4-vs-unstructured gap and the Wanda-vs-SparseGPT ordering are claimed only if the paired interval excludes zero in both draws. Speed: 7 interleaved repeats, ties reported when intervals overlap. Recovery: one seed; the comparison of width vs depth is therefore labeled "one run" and supported by the paired interval over windows only.

*Artifacts committed.* Rows in `data/compression_report.jsonl` (the chapter's table source); `lm-c11-prune-masks-qwen3-1p7b-wanda50` (bit-packed masks, ≈ 0.2 GB if kept; otherwise regenerated); the exponent-statistics JSON; speed CSVs; the final `tab_lm_comp_all` and `fig_lm_comp_frontier`.

#### Figures and tables

- `tab_lm_prune_taxonomy`, `fig_lm_prune_toy` (T0), `fig_lm_prune_svd` (singular spectra of an MLP matrix and perplexity of truncated SVD at matched parameters), `fig_lm_prune_sweep`, `fig_lm_prune_speed`, `fig_lm_prune_recovery`, `fig_lm_prune_exp`, `tab_lm_prune_df11`.
- Chapter wrap-up: `fig_lm_comp_frontier`, `tab_lm_comp_all`, `tab_lm_comp_decision`.

#### Pitfalls and misconceptions

- *Sparsity is not speed.* A zero in a dense matrix costs the same bytes and flops; only structured removal, sparse tensor cores or a sparse format change the cost.
- Pruning per layer instead of per output row hurts: layers' rows have different magnitude scales (11.3: rows are independent).
- 2:4 needs 2:4-conformant weights *per group along the input dimension of each row*; masks from unstructured methods do not convert.
- Counting only values: 2:4 also stores metadata; the realized byte ratio is 0.5625, not 0.5.
- Mask round-off: after SparseGPT's updates, "pruned" weights may be `1e-18`, not zero; zero them explicitly.
- Comparing structured pruning with recovery to unstructured pruning without it; or distilling for different token budgets.
- Lossless is about the weights; bit-identical outputs additionally require the same kernels and numerics (verified in the DF11 comparison, not assumed).
- DF11 decompression has a cost; at batch 1 on a bandwidth-bound GPU it can pay off only if hidden behind the memory reads (measure).
- Perplexity can hide a pruned model's loss of long-tail knowledge; the task rows are part of the report.

#### Summary (draft bullets)

- Pruning is removal at four granularities; only structured removal and hardware-supported patterns change dense-kernel cost.
- Wanda is OBS saliency under a diagonal Hessian with no weight update; SparseGPT is the GPTQ sweep with a zero grid, with mask selection per block.
- 2:4 stores 9 bits per BF16 weight (values plus metadata), bounding decode speedup by 1.78× and prefill by 2× on sparse tensor cores.
- Width and depth pruning produce ordinary smaller dense models that need distillation to recover; they are the only pruning variants that speed everything up without special kernels.
- BF16 exponents have about 2.5–3 bits of entropy, so Huffman coding saves about 30% exactly; the chapter's report puts every method on one table of quality, bytes and measured speed.

#### Exercises

1. *(check)* A 2:4-pruned model is stored as dense BF16 with zeros. What are its bytes, decode speed and prefill speed relative to the dense model, and what changes if it is stored in the `SparseSemiStructuredTensor` format? *Sketch:* none, none, none for the dense storage; 0.5625 of the bytes and up to 1.78× / 2× for the sparse format.
2. *(check)* Why does Wanda compare weights within each output row rather than across the whole layer? *Sketch:* rows contribute independently to the output error and have different scales; a global threshold removes entire rows' worth of small-scale weights disproportionately.
3. *(derive)* Prove that `[H⁻¹]_jj ≥ 1/H_jj` for positive definite `H`, with equality iff `e_j` is an eigenvector of `H`. Conclude that optimal compensation lowers the cost of removing weight `j` from `w_j² H_jj` (Wanda's score, the exact cost of zeroing it with no update) to `w_j²/[H⁻¹]_jj`, and explain why Wanda's ranking can differ from OBS's ranking when `H` is far from diagonal. *Sketch:* Cauchy–Schwarz, `1 = (e_jᵀH^{1/2}H^{−1/2}e_j)² ≤ (e_jᵀHe_j)(e_jᵀH⁻¹e_j)`, with equality iff `H^{1/2}e_j ∥ H^{−1/2}e_j`; the factor `H_jj[H⁻¹]_jj ≥ 1` is large for weights whose input channel is strongly correlated with others, which compensation can exploit and Wanda cannot.
4. *(derive)* Compute the exact fraction of BF16 bytes saved by exponent Huffman coding if the exponent entropy is `H` and the code is optimal, and show that "saving 30%" corresponds to `L ≈ 3.2` bits. *Sketch:* `(8 + L)/16 = 0.70 → L = 3.2`; DFloat11's ≈ 11 bits per weight corresponds to `L ≈ 3`.
5. *(measure)* Plot Wanda perplexity against sparsity per layer type (prune only `q_proj`, only `down_proj`, …) at 50%; predict the most sensitive type from the activation-norm statistics of 11.2, then measure.
6. *(measure)* Time the 2:4 MLP conversion at `B ∈ {1, …, 4096}` on your GPU and locate the batch where the sparse kernel first beats dense; compare with D2's prediction. Repeat with INT8 weights if the backend supports them.
7. *(open)* Combine methods: 2:4 pruning then 4-bit GPTQ, or width pruning then AWQ. Build one combined checkpoint under a stated byte budget and show, with measured speed and paired perplexity intervals, whether the savings multiply; explain what is wrong with multiplying published speedups.

#### Further reading

- Sun et al., [Wanda](https://arxiv.org/abs/2306.11695), code https://github.com/locuslab/wanda; Frantar and Alistarh, [SparseGPT](https://arxiv.org/abs/2301.00774): the two one-shot pruners.
- Mishra et al., [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378): the 2:4 pattern, metadata and the retraining workflow. PyTorch's [semi-structured sparsity tutorial](https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html). NVIDIA [A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/).
- Sreenivas et al., [Compact Language Models via Pruning and Knowledge Distillation (Minitron)](https://arxiv.org/abs/2407.14679), code https://github.com/NVlabs/Minitron; Gromov et al., [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887) for depth pruning.
- Zhang et al., [DFloat11](https://arxiv.org/abs/2504.11651), code https://github.com/LeanModels/DFloat11; Hershcovitch et al., [ZipNN](https://arxiv.org/abs/2411.05239).
- Teaching: MIT 6.5940 pruning and sparsity lectures and labs; Smola, [Efficiency in LLMs](https://alex.smola.org/posts/45-mlss-efficiency/) for the lossless-vs-lossy comparison; [llm-compressor](https://github.com/vllm-project/llm-compressor) for sparsity-plus-quantization recipes.

#### Risks and open decisions

- *Semi-structured sparse kernels on Ada.* PyTorch documents compute capability ≥ 8.0, which includes Ada, but whether the CUTLASS or cuSPARSELt backend is faster than dense for these shapes there is to be measured. Fallback: report the A100 datasheet numbers as the claim, the layer timing as the measurement, and the Ada result as "no benefit" if that is what is found.
- *No A100 in the T1 build.* The A100-native 2:4 speedups are therefore not reproduced by this notebook; the claim rests on the datasheet and the paper. If an A100 runner is available (one A100 40 GB, the T2 resource class), the timing cell runs there unchanged.
- *DF11 installation* (CUDA-specific wheels) may fail. Fallback: from-scratch encode/decode, entropy tables and whole-model size ratio only; the production greedy-output equality cell is skipped and marked.
- *Recovery quality with LoRA* is capacity-limited compared with Minitron's full distillation; the T2 variant (full-weight distillation with 8-bit Adam) is noted as the extension. If the width/depth comparison is within noise, it is reported as a tie.
- *No assignment is attached by v5;* the optional mini-assignment in the chapter overview uses `tab_lm_comp_all` as the autograder source of truth.
