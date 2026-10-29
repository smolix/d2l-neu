# Chapter 12: Efficient Inference — Blueprint

**Label:** `chap_lm_inference` · **Arc:** VI. Deploying (11 Compression, **12 Efficient Inference**, 13 Serving Systems) · **Sections:** 12.1 `sec_lm_fast_decoder`, 12.2 `sec_lm_speculative`, 12.3 `sec_lm_drafters`, 12.4 `sec_lm_kv_compression` · **Tiers:** T1 (12.1, 12.2, 12.4), T2 with a T1 reduction (12.3)

## Chapter overview

### Draft introduction (≈170 words)

A trained model answers one request in two phases. Prefill reads the prompt in one compute-bound pass. Decode then emits one token per pass, and each pass must stream every weight, and the request's whole KV cache, through the memory system. `sec_hardware` showed that this makes decode a bandwidth problem, and `sec_kv-cache` showed where the cache comes from. This chapter asks how close one GPU can get to that bound for one request or a handful of requests, using only techniques that leave the pretrained weights as they are or change them after training. We build a decoder that reaches the bound (static caches, CUDA graphs, int4 weights). We then beat the bound's per-token form by verifying several tokens in one weight read: speculative decoding, whose exactness we prove, and the drafters (separate models, EAGLE-style heads, multi-token-prediction heads) that make it pay. Finally we shrink the KV cache after training and measure what it costs on tasks that a needle-in-a-haystack test does not cover. Scheduling many requests is Chapter 13, and architectures that need pretraining are Chapter 14.

### Learning objectives

After the chapter a reader can:
1. Decompose a decode step into kernel time, CPU launch time and wasted bytes, and bring Qwen3-0.6B (R1) from eager mode to within a stated fraction of the `eq_hw_decode_bound` ceiling, attributing every rung of the ladder to one cause.
2. Write the bytes ledger of a model (weights by format, KV per token, tied-head effects) and predict, before measuring, the speedup of int8/int4 weights, including why quantizing only the transformer blocks caps the gain.
3. Prove that speculative sampling returns exactly the target distribution, derive the expected number of tokens per target call, and use a wall-clock model (draft cost, batch size, context length) to predict where speculation wins and where it loses.
4. Train a feature-level drafter (EAGLE-style) and Medusa heads for R1, run the Qwen3.5 MTP head, verify draft trees, and compare acceptance lengths across drafters under identical sampling settings.
5. Quantize and evict the KV cache of a trained model (per-channel keys, per-token values, pre-RoPE keys, H2O, SnapKV), account for every byte, and evaluate with RULER-style and QA tasks with paired-bootstrap intervals, including a second-question test that exposes query-dependent eviction.

### Prerequisites

Book labels:
- `sec_gpt`, `sec_transformer-block`: the `d2l.GPT` structure (`token_emb`, `blks[i].norm1/attention/norm2/ffn`, `norm`) that `d2l.FastGenerator` wraps.
- `sec_kv-cache`: cache bytes formula `eq_kv-cache-bytes`; `forward_cached`/`generate_cached` (the dynamic `torch.cat` cache, our baseline rung); `GQAAttention`; prefill vs decode arithmetic intensity; the low-rank/MLA sketch, sliding windows and sinks (all **not** re-taught; 12.4 reuses sinks as a baseline eviction policy and cites the section).
- `sec_decoding`: temperature, top-k, top-p, min-p. 12.2 needs the *processed* distribution, not just a sample.
- `sec_hardware`: `eq_hw_decode_bound`, `subsec_hw-two-workloads`, `tab_gpu_specs` (RTX 4090 1.0 TB/s, 165 TF bf16; ridge ~165 FLOP/B), `tab_rules_of_thumb` (kernel launch 5–15 µs).
- `sec_perf_model`: `d2l.Benchmark`, correct GPU timing, profiler reading.
- `sec_compilation`: `torch.compile`, `mode='reduce-overhead'`, CUDA-graph rigidity (fixed shapes and addresses), graph breaks. 12.1 uses these and does not re-explain capture; it adds what a *decoder* needs (static state, bucketed shapes).
- `sec_attention-at-scale`: online softmax/FlashAttention; sliding windows. 12.1 adds only the decode-shaped (query length 1) consequence: split-KV.
- `sec_numerics`, `sec_memory_precision`: formats, bytes per element.
- `sec_mdl-statistics`: bootstrap (via 1.5).

Earlier sections of this part (all via reserved names):
- 1.1: `d2l.load_hf`, `d2l.config_memory` (cross-check of the ledger).
- 1.2: `d2l.load_qwen3` → R1 as `d2l.GPT`; `d2l.generate` (sampler conventions); logit-parity tolerance for BF16 established there.
- 1.5: `d2l.EvalSuite`, `d2l.paired_bootstrap`.
- 2.4–2.5: `d2l.book_corpus`, `d2l.book_qa` (haystack text and QA set for 12.4).
- 4.2: R2 (optional second target for 12.3 exercises).
- 5.1–5.2: chip specs and the decode step-time model `t(b, L)`.
- 7.6: `d2l.quantize_blockwise`, `d2l.dequantize` (int4 weight quantizers).
- 11.1: group-size/perplexity trade-off of weight quantization (12.1 reuses the conclusion, not the sweep).
- 14.8 (forward pointer): MTP as a pretraining objective.

### Section dependency graph

```
sec_lm_fast_decoder (12.1) ── owns d2l.FastGenerator, StaticKVCache, KVPolicy, WeightOnlyLinear
   ├──> sec_lm_speculative (12.2): verify/rollback hooks, Speculator protocol, exactness, wall-clock model
   │        └──> sec_lm_drafters (12.3): Medusa / EAGLE / MTP speculators, tree verification (needs 12.2's accept rule)
   └──> sec_lm_kv_compression (12.4): KVPolicy + cache backends (quantized, evicting)
12.2 and 12.4 share the draft-with-small-KV idea (the MagicDec fix); 12.4's policies plug into 12.2's draft model.
Forward: Chapter 13 reuses FastGenerator (13.2 slots and chunked prefill; 13.3 swaps the KV backend for paged blocks; A8 uses 12.2's Speculator).
```

### Running examples and shared artifacts

Used:
- **R1** = Qwen3-0.6B in `d2l.GPT` (28 layers, hidden 1024, 16 query heads, 8 KV heads, head_dim 128, FFN 3072, vocab 151,936, tied embeddings, RoPE θ = 10⁶; checked against `config.json` at revision `c1899de289a04d12100db370d81485cdf75e47ca`). 596,049,920 parameters when the tied matrix is counted once; the Hub API reports 751,632,384 because the safetensors file stores `lm_head.weight` separately (the difference is exactly the embedding, 155,582,464). KV per token in bf16: 2·28·8·128·2 = 114,688 B (112 KiB).
- **Qwen3-4B** as the 12.2 target (36 layers, hidden 2560, 32 query heads, 8 KV heads, head_dim 128, FFN 9728; 4,022,468,096 parameters, matching the safetensors total; revision `1cfa9a7208912126459214e8b04321603b3df60c`). KV per token in bf16: 147,456 B (144 KiB). Note the ratio of KV per token draft/target = 0.778, which matters in 12.2.
- **Qwen3.5-2B** (hybrid, Apache-2.0, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`) for the MTP head in 12.3: `mtp_num_hidden_layers = 1`, 15 `mtp.*` tensors in its safetensors index.
- **R3** (the book as a corpus) as RULER haystack and QA source in 12.4.

Produced (committed under `DATA_HUB` keys with a `lm-inference_` prefix; formats in each section):
- `d2l.FastGenerator` and its backends (12.1), the speed ladder table `tab_lm_fast_ladder` (12.1).
- Draft-regeneration corpus `lm-inference-regen-r1` (12.3), trained `MedusaHeads`, `EagleDrafter` checkpoints for R1 (12.3).
- RULER-style task instances and the full sweep table (12.4).

### Compute plan

| § | Tier | Hardware assumption | Wall-clock budget | GPU-hours | Committed artifacts |
|---|---|---|---|---:|---|
| 12.1 | T1 | one 4090 slot ≤ 15 GiB, R1 (≈ 1.2 GB bf16) | ≤ ~12 min; peak ≈ 3 GiB (weights + 8×2048 static cache = 1.8 GiB) | < 0.3 | ladder table (CSV), profiler summary (JSON), CUDA-graph capture log |
| 12.2 | T1 | Qwen3-4B bf16 (8.05 GB) + Qwen3-0.6B bf16 (1.19 GB) + caches | ≤ ~15 min; peak ≲ 13 GiB | < 0.3 | acceptance traces per task (NPZ), speedup grid (CSV) |
| 12.3 | T2 (reduced T1 config) | 1× A100 40 GB (PCIe), peak ≲ 38 GiB usable; T1 reduction trains on ~10% of tokens | T2 ≤ ~3 h incl. data regeneration (budget; the 40 GB PCIe part streams at 1,555 GB/s against 2,039 GB/s for the 80 GB SXM, so bandwidth-bound regeneration takes 2,039/1,555 = 1.31× longer per step: to be re-measured and tightened after the 12.1 batch-throughput measurement); T1 ≤ ~15 min training from the committed regenerated slice | ≈ 3 (T2) | regenerated corpus shard, Medusa and EAGLE checkpoints (~20 M params each), per-position acceptance tables |
| 12.4 | T1 | R1, contexts ≤ 8k, batch ≤ 4 (fallback `Qwen/Qwen3-4B`: 12.0 GiB of weights plus KV at 8k × 4, so peak ≈ 13–14 GiB, to be measured) | ≤ ~15 min for the reduced grid; the extended grid is precomputed | < 0.3 (live) | extended sweep table (Parquet), RULER-style instances (JSONL, seeded) |

Hardware facts that constrain the plans (conventions): A100 has no FP8 tensor cores, so every FP8 variant in this chapter is skipped (KV FP8 exercises run on the Ada slot). Marlin-style int4 kernels target Ampere and Ada, so both tiers can run them (Marlin paper; check the installed wheel). Datasheet numbers used in predictions: A100 40 GB PCIe 1,555 GB/s and 312 TF bf16 dense, 250 W (the T2 part; the datasheet's 80 GB SXM row, 2,039 GB/s, is quoted only for the PyTorch blog's A100-80GB numbers); RTX 4090 1.0 TB/s and 165 TF bf16 dense (`tab_gpu_specs`; NVIDIA A100 datasheet and RTX 4090 product page, both URLs verified).

## Assignment note: A8 is Chapter 13's

v5 attaches **A8 (inference engine: paged KV, continuous batching, speculative decoding)** to Chapter 13 (:numref:`sec_lm_a8`), and it is blueprinted there. Chapter 12 owes it the following stable hooks; A8's starter code should import exactly these and nothing deeper.

| A8 need | Provided by | Contract A8 can rely on |
|---|---|---|
| Single-GPU forward over R1 with slot-based KV | 12.1 `FastGenerator.forward/prefill/decode_step`, `alloc/free` | per-slot `pos`; active slots are the prefix `0..n-1`; fixed-shape graph buckets |
| Swap the KV backend for paged blocks | 12.1 `KVCache` protocol (`append`, `view`, `bytes`) | a paged backend implements the same three methods; `FastGenerator` does not touch cache tensors directly |
| Chunked prefill mixed with decode | 12.1 `forward(tokens[B,T], ...)` with per-row `pos` | the same primitive handles T = 1 (decode), T = γ+1 (verify) and T = chunk (prefill) |
| Speculative decoding in the scheduler | 12.2 `Speculator` protocol and `d2l.sd_accept` | `propose/commit` interface; exactness test `d2l.spec_greedy_check` |
| Batch-invariant greedy outputs (A8 checks 4 and 8) | 12.1 and 12.2 pitfalls | equal token ids against the fp32 oracle hold with TF32 off and a fixed reduction order, with near-ties (small top-2 logit gap) excluded and counted as in A8 check 4(ii); in bf16 only up to near-ties (the section measures the divergence rate); bitwise-equal logits against an engine's own batch-1 run is 13.2's `det_mode` |

12.1–12.2 therefore include a short "Chapter 13 hand-off" block in their exports (listed below). The A8 author decides the throughput and P99 targets; Chapter 12 supplies the single-stream numbers (tokens/s at b = 1, the ladder) that serve as the baseline row of the A8 report.

## Teaching sources

Kept from v5 (annotated with what to take from each):
- PyTorch, [Accelerating Generative AI II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/): the template for 12.1's ladder. Take the sequence of steps (eager 25.5 → compile 107.0 → int8 157.4 → int4+GPTQ 202.1 tok/s for Llama-7B on an A100-80GB, batch 1) and the 72% memory-bandwidth-utilization argument, which is the same accounting as `eq_hw_decode_bound`. Also its speculative section (2× for CodeLlama-34B + 7B, about 1.3× for Llama-7B + TinyLlama), which 12.2 reproduces qualitatively.
- Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html): the compute / bandwidth / overhead taxonomy that 12.1 applies to decode; already the framing of `subsec_perf-regimes`.
- Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/): survey map for the whole chapter (quantization, speculative decoding, KV tricks).
- [ICML 2024 tutorial on efficient LLM serving](https://icml.cc/virtual/2024/tutorial/35229): algorithm-to-system framing; use for 12.2's "why speculation interacts with batching".
- [EMNLP 2025 efficient inference tutorial](https://aclanthology.org/2025.emnlp-tutorials.1/): taxonomy of draft mechanisms and KV methods for 12.3–12.4.

Added (verified, HTTP 200):
- Alex Smola, [Efficiency in LLMs](https://alex.smola.org/posts/45-mlss-efficiency/) (Codex chapter reference): the weight-versus-KV accounting thread and the maximal-coupling picture of speculative sampling used in 12.2's proof.
- PyTorch, [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/): the capture/replay mechanics and memory-pool sharing that 12.1 relies on.
- Hugging Face, [Assisted Generation](https://huggingface.co/blog/assisted-generation): the library-side view of draft-and-verify (12.2 "In production").

## Sections

### 12.1 A fast decoder on one GPU

- **Label:** `sec_lm_fast_decoder` · **Tier:** T1 (no T2 part) · **Budget:** ≤ ~12 min wall-clock and ≈ 3 GiB peak on one 4090 slot at T1; the same notebook on an A100 40 GB PCIe only changes the bound line (1,555 GB/s, so ≈ 1,300 tok/s for R1's 1.19 GB of bf16 weights, KV ignored) · **Codex units:** 13.1 (reference generator), 13.3 (compilation and overhead); 13.2's IO-aware attention is *not* repeated (owned by `sec_attention-at-scale`)
- **Builds on (book):**
  - `sec_kv-cache`: `eq_kv-cache-bytes`; the cached forward and its `torch.cat` cache (rung 1 of the ladder and the correctness oracle for the static cache); prefill/decode intensity. Not re-taught: why caching works, GQA.
  - `sec_hardware`: `eq_hw_decode_bound` (the line every rung is plotted against), `tab_rules_of_thumb` (launch cost).
  - `sec_compilation`: `torch.compile(mode='reduce-overhead')`, capture/replay rigidity, graph breaks. Not re-taught: what a CUDA graph is. This section adds the decoder-specific obligations: static state, bucketed shapes, host-sync-free loops.
  - `sec_perf_model`: `d2l.Benchmark`, GPU timing, profiler tables.
  - `sec_attention-at-scale`: online softmax. Only the decode-shaped consequence (query length 1, parallelism over the KV axis) is added.
  - `sec_gpt`/`sec_transformer-block`: module layout of `d2l.GPT`.
- **Builds on (this part):**
  - 1.2: `d2l.load_qwen3` returns R1. module names are those of `01_interfaces.md` section 1: `model.token_emb`; `blks[i].attention` with `W_q, W_k, W_v, W_o, q_norm, k_norm` and `head_dim, num_heads, num_kv_heads`; `blks[i].ffn` with SwiGLU `W_g, W_1, W_2` (`W_2(silu(W_g x) * W_1 x)`; there are no `W_gate/W_up/W_down`); norm names (`blks[i].norm1/norm2`, final `norm`) follow 1.2's mapping table. R1 is a `d2l.Qwen3GPT`: each `blks[i].attention` is a `d2l.Qwen3Attention` with the attribute `rope_theta` (1e6), and its RoPE uses the book's interleaved layout because 1.2 permutes the `q_proj`, `k_proj`, `q_norm` and `k_norm` rows at load (`map_qwen3_state_dict(rope_layout='permute')`). 12.1 rotates interleaved pairs at the cache offset, as `Qwen3Attention.forward_step` does; no adapter is needed.
  - 5.1–5.2: `d2l.ServingModel(cfg, chip, calib=None, dtype_bytes=2, kv_bytes=2, mem_util=0.9, kv_bytes_override=None).decode_step(b, L)` is the decode step-time model `eq_lm_hw_decode_step` (5.2's `t(b, L) = max(2·N_act·b/P, (W + b·L·k_KV)/β)`); there is no `d2l.decode_step_time`, and 12.1 does not define one. Weight-only int4 is expressed through `dtype_bytes=0.5` (plus the 4/G-byte scale-and-zero overhead, 4.25 bits at G = 128, folded in via `calib` or by comparing with `decode_ledger`); the measured per-rung numbers use `calib`. `d2l.chip_specs()` returns a DataFrame with columns `chip, mem_bw_tbs` (TB/s; multiply by 1,000 for the GB/s used in the bound formulas), `p16_tflops`, `mem_gb`, and so on, as in `01_interfaces.md`.
  - 7.6: `d2l.quantize_blockwise(w, bits=4, block_size=64, ...) -> d2l.QTensor` with fields `codes` (uint8, unpacked 0..2^bits−1), `scales` (`[out, in/block]`), `zeros` (`None` if symmetric), `block_size`, `shape`, `bits`; `d2l.dequantize(q, dtype=torch.bfloat16)`. The default block size is 64, so 12.1 passes `block_size=128` explicitly wherever it uses groups of 128. 12.1 owns only the *packing* into the kernel layout.
  - 1.1: `d2l.config_memory` (cross-check of the ledger within 1%).
- **Exports (`#@save`):**
  - `d2l.FastGenerator`, `d2l.StaticKVCache`, `d2l.KVPolicy` (below). `d2l.KVCache` is the documented protocol (an abstract base class).
  - `d2l.rope_tables(head_dim, max_len, theta, layout, dtype, device) -> (cos, sin)` with shapes `[max_len, head_dim/2]`; one place where the Qwen3 split-halves layout is encoded.
  - `d2l.WeightOnlyLinear(q: QTensor | tensor, bits, kernel='ref'|'tinygemm'|'marlin')` and `d2l.quantize_model_(gen_weights, bits, group_size, quantize_head)`.
  - `d2l.decode_ledger(config, *, weights='bf16', group_size=128, quantize_head=False, kv_len=0, batch=1, kv_bits=16) -> dict` with keys `weight_bytes, kv_bytes, bound_tok_s(bandwidth)`; the single source of the bytes arithmetic used in 12.1–12.4.
  - `d2l.bench_decode(gen, prompt_len, new_tokens, batch, repeats=7) -> dict(median_tok_s, min, max, step_ms, gpu_busy_frac)`.
- **Models and data:** `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca` (Apache-2.0; 1.19 GB bf16 as computed from the config). Prompts: 16 prompts from R3 headings (`d2l.book_corpus`) so no external data is needed; decode length fixed at 256 new tokens with `min_new_tokens=256` so runs are comparable. Weights are loaded via 1.2, not re-downloaded.
- **Packages:** `torch>=2.5` (`enable_gqa` in SDPA, stable `torch.cuda.graph`, `torch.compile(fullgraph=True)` on a stateless step); `torchao` at the version pinned by 11.1 for `int4` tinygemm (**check at authoring time** whether the Ada/A100 int4 packed layout API is `Int4WeightOnlyConfig` or its successor); optional `flash-attn` (for `flash_attn_with_kvcache` as an oracle), optional `flashinfer`; optional `triton` for the exercise kernel.

#### The `d2l.FastGenerator` API (normative; the author implements exactly this)

```python
class KVCache:                                   # protocol, d2l.KVCache
    stores_pre_rope: bool = False                # True: keys are stored before RoPE, generator rotates after view()
    def append(self, layer, k, v, rows, cols): ...   # k, v: [B, n_kv, T, D]; rows: LongTensor[B] slot ids; cols: LongTensor[B, T] columns
    def view(self, layer, upto, nrows):          # -> (K, V): [nrows, n_kv, upto, D], compute dtype, slots 0..nrows-1 (a *view* for the static backend)
    def copy_slot(self, src, dst): ...           # one slot's KV, all layers (used by free() compaction and prefix reuse in 13.3)
    def copy_cols(self, rows, src_cols, dst_cols): ...   # in-slot column moves (tree commit in 12.3, eviction compaction in 12.4)
    def reset(self, rows): ...
    def nbytes(self, kv_len=None) -> int         # allocated bytes, or bytes read by one decode step at kv_len columns

class StaticKVCache(KVCache):                    # default backend
    # k, v: one tensor each, shape [layers, max_batch, n_kv, max_len, head_dim], bf16, allocated once.
    # Contiguous (n_kv, max_len, D) slabs per (layer, slot): a slice [:, :, :upto] is a view and is bmm-friendly.

class KVPolicy:                                  # d2l.KVPolicy: eviction/quantization hook; the default is a no-op full cache
    needs_probs = False                          # True: decode attention also returns softmax probabilities to observe()
    prefill_probs = None                         # None | ('window', w) | ('colsum',): during prefill, additionally compute softmax probs of the last w query rows (SnapKV),
                                                 # or their column sums over all query rows, in query blocks (H2O); delivered to observe() with phase 'prefill_window' | 'prefill_colsum'
    def make_cache(self, gen) -> KVCache: ...    # default: StaticKVCache(gen's geometry)
    def observe(self, layer, probs, rows, phase): ...   # probs: [B, n_kv, g, T, S] float32 (g = query heads per KV head; for 'prefill_colsum' [B, n_kv, g, S]); phase 'decode' | 'prefill_window' | 'prefill_colsum'
    def compact(self, gen, rows): ...            # between steps, outside any graph; may shrink gen.kv_len[rows] and rewrite columns in place

class FastGenerator:
    def __init__(self, model, *, max_batch=8, max_len=2048, dtype=torch.bfloat16, device='cuda',
                 mode='graph',                   # 'eager' | 'compile' | 'graph' | 'compile+graph'
                 batch_buckets=(1, 2, 4, 8),     # graphs are captured for these batch sizes (<= max_batch)
                 len_buckets=None,               # attended-length buckets, e.g. (256, 512, 1024, 2048); None: one bucket = max_len
                 fuse=True,                      # fuse `W_q/W_k/W_v` into one QKV weight and `W_g/W_1` into one gate/up weight (copies; `model` is untouched and stays the oracle)
                 weights='bf16',                 # 'bf16' | 'int8' | 'int4'
                 group_size=128, quantize_head=False, int4_kernel='tinygemm',
                 attn='einsum',                  # 'einsum' (grouped bmm, any KV layout) | 'sdpa' | 'flash_kvcache'
                 kv_policy=None,                 # KVPolicy; None: StaticKVCache
                 hidden_layers=(),               # block indices whose outputs forward() can return (EAGLE-3 features, 12.3)
                 seed=0)
    # ---- state: device tensors, never read on the host inside a loop ----
    pos: LongTensor      # [max_batch] RoPE position of the next token per slot
    kv_len: LongTensor   # [max_batch] valid cache columns per slot; equals pos unless a policy evicted (12.4)
    n_active: int        # active slots are always the prefix 0..n_active-1
    # ---- slots (Chapter 13 admits and retires requests with these) ----
    def alloc(self, n=1) -> list[int]
    def free(self, slots) -> None        # swap-with-last compaction (copy_slot) keeps the active prefix dense
    def reset(self) -> None
    # ---- the primitive: append T new tokens per row at per-row offsets, attend over the cache ----
    def forward(self, tokens, rows=None, *, logits_at='all', tree=None, return_hidden=False) -> Tensor
        # tokens: LongTensor[B, T]; rows: slots 0..B-1 if None. Writes KV at columns kv_len..kv_len+T-1 (tentative: pos/kv_len unchanged),
        # position ids pos+t (pos+depth for a tree), mask j <= kv_len_b + t (tree: ancestor mask among the new T columns).
        # logits_at='all' -> [B, T, V] float32; 'last' -> [B, V]; return_hidden=True also returns hidden [n_hidden, B, T, H] (pre-final-norm block outputs for `hidden_layers`, plus the last block's output).
    def advance(self, rows, n) -> None   # pos += n, kv_len += n   (commit)
    def rollback(self, rows, n) -> None  # pos -= n, kv_len -= n; raises if a policy has compacted these rows
    # ---- the three named uses of forward ----
    def prefill(self, prompts, rows=None, *, return_hidden=False) -> Tensor    # list[list[int]] right-padded; causal; logits of each last real token [B, V]; sets pos = kv_len = len; chunked if prefill_chunk
    def decode_step(self, tokens, rows=None) -> Tensor                         # [B] -> [B, V]; forward(T=1) + advance(1); graph replay
    def verify(self, tokens, rows=None, *, tree=None, return_hidden=False)    # forward(T=gamma+1 or tree size) with logits_at='all'; does NOT advance
    def commit(self, rows, n_accepted, path=None) -> None                      # advance by n_accepted[b]; with a tree, move accepted-path columns to contiguous columns (copy_cols)
    # ---- sampling hooks ----
    def processed_probs(self, logits, *, temperature=1.0, top_k=None, top_p=None, min_p=None) -> Tensor  # the distribution that is actually sampled (sec_decoding); temperature 0 -> one-hot argmax
    def sample(self, probs, generator=None) -> LongTensor                       # inverse-CDF / Gumbel-max on device
    def generate(self, prompts, max_new_tokens, *, sampling=None, eos_id=None, speculator=None,
                 sync_every=16, return_stats=False) -> list[list[int]]
        # sampling: dict with the keys of d2l.generate (1.2); sampler: callable(logits, state)->tokens may replace it.
        # The loop keeps tokens on device and checks EOS every `sync_every` steps (one host sync per window).
        # speculator: a 12.2 Speculator; with None this is plain decoding.
    def warmup(self, batches=None, Ts=(1,), lens=None) -> dict   # capture graphs; returns capture seconds and graph-pool bytes
    stats: dict          # target_calls, tokens_out, graph_hits, graph_misses, host_syncs
```

Design rules the author must implement and test (each is one line of the "Design notes" table in the notebook):
1. **One primitive.** `prefill`, `decode_step`, `verify` and chunked prefill are `forward` with different `T` and `logits_at`. Prefill with an empty cache and `T = len` uses SDPA with `is_causal=True` and `enable_gqa=True` (flash backend; no mask) and computes the LM head only on the last real row (saves 2·T·V·H FLOPs; the head is 155.6 M parameters); T > 1 with a non-empty cache (chunk, verify) uses the explicit masked path.
2. **Decode attention is explicit and GQA-grouped:** `q: [B, n_kv, g, D]`, `scores = q @ K.transpose(-1, -2)` over a *view* of the cache, mask, `softmax` in fp32, `@ V`. No `repeat_interleave` of K/V (that would double the bytes read). The profiler must show no `copy_`/`cat` kernel in the attention path (test).
3. **Cache writes** are `index_put_` into the static tensor at `(rows, :, cols)`; no allocation inside a step; every tensor a graph touches is allocated before capture.
4. **Active slots are a prefix**, so a graph for batch bucket `Bb` reads `cache[:, :Bb]`, a view. Padding rows inside a bucket carry `pos = 0` and a dummy token; their (wasted) bytes are counted in the ledger.
5. **Length buckets.** With `len_buckets` set, the step picks the smallest bucket `Lb >= max(kv_len) + T` using one host read per `sync_every` window (the bucket is chosen for the next window with slack = `sync_every`), slices `K[:, :, :Lb]`, and masks to `kv_len`. Without buckets the step reads `max_len` columns every time (the symptom the section measures).
6. **Graphs:** keyed by `(Bb, T, Lb, phase)`; captured lazily or in `warmup`; one shared memory pool (`torch.cuda.graph(g, pool=...)`); static input buffers `tokens, pos, kv_len` and static output buffers `logits` (and `hidden`); replay = three `copy_` calls + `g.replay()`. Sampling runs outside the graph by default (arbitrary processors, graph-safe RNG not required); `fuse_sampling` is out of scope.
7. **Weights** are fused and, if requested, quantized at construction; `generate` never touches `model`. The reference path `kernel='ref'` dequantizes then multiplies (an oracle, deliberately slow because it moves 4× the bytes); `'tinygemm'` calls the fused torchao int4 kernel; `'marlin'` (optional) uses the Marlin kernel through the installed package.
8. **Numerics tests** (below) gate every mode.

**Hooks other sections attach to (frozen for 12.2–12.4 and Chapter 13):** `forward/verify/commit/rollback` (draft–verify, 12.2–12.3); `hidden_layers` + `return_hidden` (EAGLE-3 and MTP features, 12.3); `KVCache.stores_pre_rope`, `KVPolicy.make_cache/observe/compact`, `kv_len` vs `pos` (12.4); `KVCache` protocol and `alloc/free` (13.2–13.3); `processed_probs/sample` (12.2 needs `p` and `q` after identical processing).

#### Narrative arc

1. **The question.** Opening cell: load R1, run `generate_cached` (the `sec_kv-cache` loop, dynamic `torch.cat` cache) for 256 new tokens at batch 1, print tokens/s next to the ceiling from `d2l.decode_ledger(config)` (≈ 840 tok/s at 1.0 TB/s with 1.19 GB of bf16 weights; KV ignored). Print the ratio. Then `torch.profiler` over ten steps and print two numbers: kernel launches per step and the fraction of wall time during which the GPU ran a kernel (`gpu_busy_frac`). Hypothesis shown to the reader: the gap is neither compute nor bandwidth; it is CPU launch overhead (hardware.md puts a launch at 5–15 µs). The measured values are "to be measured"; the direction is: tokens/s far below the ceiling, `gpu_busy_frac` well below 1 (the PyTorch blog reports 25.5 tok/s for an eager Llama-7B on an A100, whose bound from 2 TB/s and ~14 GB of fp16 weights is roughly 140 tok/s).
2. **Where the step goes: a bytes ledger and a kernel census.** 
   - `decode_ledger` table for R1: embedding/LM head 311,164,928 B (tied, read once as the LM head), blocks 880,932,864 B, norms; KV per token 114,688 B; table with rows for bf16, int8 blocks, int4 blocks, int4 everything, and KV at L = 256 / 2,048 / 8,192 (`tab_lm_fast_ledger`).
   - Kernel census by operator name from the profiler, grouped into the layer's pieces (norm, QKV, QK-norm, RoPE, cache write, attention, output projection, MLP, residual). This is the to-do list for fusion.
   - Cross-check: ledger total vs `d2l.config_memory` within 1%.
3. **A static cache and a position-indexed forward.**
   - Layout of `StaticKVCache` and why a slot slab is contiguous; positions as device tensors; the mask `j <= kv_len + t`.
   - Correctness first (cell): fp32, TF32 off, `FastGenerator(mode='eager')` logits against `model(X)[:, -1]` of the full forward at ten prompt lengths, including length 1 and a prompt that fills the cache. Then against `forward_cached` (the book's dynamic cache).
   - Padded prefill and per-row positions in a batch of 4 prompts of lengths (5, 17, 64, 200): per-row logits equal the single-prompt logits.
4. **CUDA graphs for decode.**
   - What a decoder adds to `sec_compilation`: all state is static; positions are device tensors, so one graph serves every step; the host loop has no `.item()`.
   - Manual capture (`torch.cuda.graph`) vs `torch.compile(mode='reduce-overhead')`: show both, explain why the manual path is the one used by the class (control over buckets and the pool; no re-capture surprises), and print capture time and pool bytes.
   - Sampling outside the graph; `sync_every` host syncs; measure the cost of syncing every step (a rung-sized effect on a small model).
   - Equality test: graph replay vs the same eager function, bf16: bitwise-equal logits (`torch.equal`) for `mode='graph'` against `mode='eager'` with identical kernels; `torch.compile` may fuse and is checked by tolerance instead.
5. **Wasted bytes: reading `max_len` columns.**
   - Time one decode step against the true context length `L` at `max_len = 8192` with and without `len_buckets`; the unbucketed curve is flat (reads the whole cache), the bucketed curve follows `W + L·k_KV`. Overlay the ledger prediction (`tab_lm_fast_ledger`). This foreshadows paged KV in 13.3 (reads proportional to used blocks).
   - Decode-shaped attention: at b = 1 the grid is `n_kv × 1` = 8 thread-block rows, which cannot fill 128 SMs; split-KV (FlashDecoding) parallelizes over the KV axis. Compare `attn='einsum'`, `'sdpa'` and `'flash_kvcache'` (and FlashInfer if installed) at L = 2k, 8k, 32k with b = 1 and b = 8; keep the table as `tab_lm_fast_attn`. Direction expected: einsum is competitive at short L and falls behind at long L and small b (to be measured).
6. **Fewer bytes: weight-only int8 and int4.**
   - `d2l.quantize_blockwise(w, bits=4, block_size=128)` output (`QTensor`; `block_size` passed explicitly, since the default is 64) → packed layout; reference `kernel='ref'` (dequantize then matmul) is correct but moves the dequantized bf16 weights through memory (slower than bf16; show it); fused `tinygemm` (and optional Marlin) read 4.25 bits/weight (group 128, bf16 scale and zero).
   - The tied-head Amdahl effect (derivation D3): quantizing the blocks only caps the speedup at 2.19×; `quantize_head=True` lifts the cap to 3.76×; measure the perplexity cost of quantizing the head on a held-out slice of R3 (`d2l.score_choices`-style token NLL from 1.4), reported in `tab_lm_fast_quality`.
   - Batch size: at b = 1…8 the fused kernel stays bandwidth-bound; at b ≳ ridge the dequantize-in-register cost shows (preview of 11.1's W4A16 batch behaviour; one sentence plus the cell).
7. **The ladder and the bound (headline experiment).** Ladder with one change per rung, the `eq_hw_decode_bound` line recomputed per rung from its own byte ledger, percent of bound printed.
8. **From one stream to many.** One short subsection and one cell: `generate` on a batch of 8 prompts. Throughput in tokens/s vs batch size against `d2l.ServingModel(...).decode_step(b, L)` (5.2, `eq_lm_hw_decode_step`); shows the weights-shared-across-streams effect and ends by pointing at Chapter 13 for scheduling.

#### Derivations

**D1. Bytes per decode step and the static-cache tax (`eq_lm_fast_bytes`).**
- Statement: for a decode step at batch `b` with true context `L`, bandwidth-bound step time is `t = (W + b·L·k_KV)/β` where `W` is the sum over tensors actually read once; a static cache read in full costs `t_static = (W + b·L_max·k_KV)/β`, so the tax is `b·(L_max − L)·k_KV/β`.
- Assumptions: bandwidth-bound regime (b below the ridge batch `b* = P·W/(2·N_act·β)`), activations negligible, one read of every weight per step.
- Steps: (1) enumerate tensors read: for R1, 28 blocks × 15,730,944 params, the tied matrix read as the LM head (embedding lookup reads one row only), final norm. (2) Write `W(format)` for bf16, int8 (1 B + negligible per-channel scale), int4 group `G` (0.5 + 4/G bytes per weight with bf16 scale and zero, i.e. 4.25 bits at G = 128). (3) Write `k_KV = 2·n_layers·n_kv·d_head·bytes_kv`. (4) Substitute into `eq_hw_decode_bound`. (5) State the effective bound `tokens/s ≤ b·β/(W + b·L·k_KV)`.
- Check cell: print the ledger for R1 at L ∈ {256, 2048, 8192} and the four weight formats and compare with `d2l.config_memory`; assert agreement within 1%. Computed values from the config and β = 1,008 GB/s: bf16 ceiling 825/706/473 tok/s at L = 256/2,048/8,192; int4-blocks 1,754/1,292/679; int4-everything 2,913/1,828/802 (the notebook prints them; they are arithmetic, not measurements).

**D2. CPU launch overhead versus kernel time, and why graphs remove it (`eq_lm_fast_overhead`).**
- Statement: with `n` kernels per step, GPU durations `g_1..g_n` in launch order and a CPU that enqueues kernel `i` at time `i·c` (`c` = per-op dispatch cost), the step ends at `T = max_i ( i·c + Σ_{j ≥ i} g_j )`. Hence `T ≥ max(Σ g, n·c + g_n)`, with equality at the extremes; replay of a captured graph replaces `c` by a much smaller per-node cost `c_g` paid on the GPU side.
- Assumptions: single stream, no host syncs in the step, enqueueing never blocks (queue not full).
- Steps: (1) kernel `i` starts at `s_i = max(i·c, s_{i−1} + g_{i−1})`. (2) Unroll the recurrence: `s_i = max_{k ≤ i}( k·c + Σ_{k ≤ j < i} g_j )`. (3) Finish time `T = s_n + g_n`; collect terms to get the stated max. (4) Two regimes: `c·n < Σ g` (GPU-bound: CPU is hidden, graphs do not help) and `c·n > Σ g` (CPU-bound: `T ≈ n·c`, graphs help by `n·(c − c_g)`). (5) Predict the eager step time from the profiler's `g_i` and a measured `c`.
- Check cell: (a) measure `c` by timing `n` launches of `x.add_(1)` on a one-element tensor (CPU-bound by construction) and dividing; (b) feed the profiler's `g_i` for one R1 step into the formula; (c) compare predicted and measured eager step time (target: within 15%; the section prints the error and does not tune `c` to hit it); (d) repeat with `mode='graph'` and report the implied `c_g`.

**D3. Amdahl's law for bytes: int4 on blocks only, and with KV (`eq_lm_fast_amdahl`).**
- Statement: if a fraction `f` of the bytes read per step is compressed by factor `r`, bandwidth-bound speedup is `1/((1 − f) + f/r)`; for R1, the tied LM head is 311.2 MB of 1,192.1 MB (26%) and KV adds `L·112 KiB`.
- Steps: (1) per-weight bytes for int4 group `G`: `0.5 + 4/G` (bf16 scale and zero), `r_w = 2/(0.5 + 4/G)` = 3.76 at G = 128. (2) With the head left in bf16 the speedup at `L = 0` is `1192.1/(234.0 + 311.2 + norms)` = 2.19; int8 blocks give 1.59. (3) Add KV: at `L = 2,048` KV is 234.9 MB, about as large as the entire int4 block weight (234.0 MB). With everything int4 the ceiling falls from 3.76× at `L = 0` to `(1192.1 + 234.9)/(316.7 + 234.9)` = 2.59× at `L = 2,048` and `(1192.1 + 939.5)/(316.7 + 939.5)` = 1.70× at `L = 8,192`. (4) Conclude: past int4, the KV cache is the next target, which is 12.4's question.
- Check cell: compute the three ratios from `decode_ledger`, then measure the *achieved* speedup of the int4 rungs and print achieved/predicted.

#### From-scratch implementation

Components (line budget ≤ ~150 lines of class code, excluding the numbered tests; the kernel-level Triton exercise is outside the budget):

| Component | Signature | Lines |
|---|---|---:|
| `rope_tables` + `apply_rope(x, cos, sin, pos)` | gathers `cos[pos[:, None] + arange(T)]`, split-halves layout | ~12 |
| `StaticKVCache` | `append/view/copy_slot/copy_cols/reset/nbytes` | ~30 |
| `FastGenerator._forward_impl(tokens, pos, kv_len, Lb)` | pure function of static buffers: embed, blocks, final norm, head | ~45 |
| graph registry | `_get_graph(key)`, capture, replay, static buffers | ~25 |
| `prefill`, `decode_step`, `verify`, `generate` | thin wrappers | ~30 |
| `WeightOnlyLinear` (ref path + tinygemm packing) | packs `QTensor` to the kernel layout; `forward(x)` | ~25 |

Numerical tests (all must pass before any timing):
1. fp32, TF32 off (`torch.backends.cuda.matmul.allow_tf32 = False`), `mode='eager'`: `max|Δlogits| / max|logits| < 1e-5` against `model(X)` at prompt lengths {1, 2, 17, 255, 1024}, seed 0. Hold the tolerance as a constant in the cell; if it fails, bisect layer by layer using 1.2's matcher.
2. bf16, `mode='graph'` vs `mode='eager'`: `torch.equal(logits)` for decode steps 0..63.
3. bf16 vs the fp32 reference: record `err_bf16 = max|Δ|/max|logits|` over 50 steps; every later bf16 mode (compile, int8, int4) is reported as `err / err_bf16` with the pass rule `top-1 agreement ≥ 98%` over 2,000 positions (the same positions for all modes). The 98% line is a screening threshold, not a result; int4 may fail it, and the table says so.
4. Greedy end-to-end: fp32 eager `FastGenerator` vs `generate_cached` on 20 prompts × 128 tokens: identical token ids (exact). In bf16 report the fraction identical and the median index of first divergence; this number is reused by the A8 note.
5. Batch consistency: a prompt run alone and as row 3 of a batch of 8: fp32 logits within 1e-5 relative; bf16 may differ (a different GEMM tile), and the section says that is expected (pointer to 13.2).
6. `alloc/free`: free slot 1 of 4 with distinct prompts; slots 0, 2, 3 continue to produce their reference tokens; slot copied from the last active slot is bit-identical.
7. Static-cache overflow: `forward` at `kv_len + T > max_len` raises (checked on the host at `generate`/`prefill` time, not inside a graph).

#### In production

- **gpt-fast** is the reference design (static cache, `torch.compile`, int8/int4, speculative); the section diffs its per-token design choices against ours in a table (same `index_copy_` cache write; its decode mask over `max_seq_length`; int4 through `_weight_int4pack_mm`). The from-scratch `FastGenerator` logits are the oracle for any variant a reader tries.
- **Transformers** `generate(cache_implementation='static')` + `torch.compile`: run it on R1 as a rung (`tab_lm_fast_ladder`, "HF static + compile") to show that the library path reaches the same region; version pinned by 1.1.
- **torchao** int4 weight-only quantization (`quantize_`), compared with our `WeightOnlyLinear('tinygemm')` on logits (tolerance as in test 3) and speed.
- **Marlin** (optional): load an R1 GPTQ-format int4 checkpoint if one is available, or repack our `QTensor` into the Marlin layout; report whether it beats `tinygemm` at b = 1 and b = 8 (Marlin targets batch sizes up to ~16–32 with near-ideal speedup per its paper).
- **FlashInfer / `flash_attn_with_kvcache`** for attention (oracle for our decode attention: max abs diff < the bf16 rounding of the output).
- **vLLM/SGLang** as the ceiling that Chapter 13 compares against; one sentence here, no run.

#### Headline experiment

- **Hypothesis.** Each rung of the ladder removes a different, identifiable cost; after CUDA graphs R1 decodes at a substantial fraction of the bandwidth bound of its own byte ledger, and int4 raises tokens/s by less than the byte ratio because the tied head stays bf16 and because other costs do not shrink.
- **Setup.** R1; RTX 4090 slot (T1); batch 1; prompt 128 tokens, 256 new tokens, greedy; `max_len = 2048`; 16 prompts; each configuration timed 7 times after warmup, order randomized per repetition (interleaved, not blocked, to average out clock/thermal drift on the shared slot); bf16 except where stated.
- **Rungs** (each reports tokens/s median and min–max over the 7 repetitions, bytes per step from the ledger, percent of that rung's bound, `gpu_busy_frac`): (0) HF `generate`, eager, dynamic cache; (1) book `generate_cached`; (2) `FastGenerator` eager, static cache; (3) + fused QKV/gate-up; (4) + `torch.compile` (no graph); (5) + manual CUDA graph; (6) + length buckets; (7) + int8 blocks; (8) + int4 blocks (head bf16); (9) + int4 head; (10) HF static cache + compile (library reference); (11) A100 column only if run at T2 (bound recomputed at 1,555 GB/s, the A100 40 GB PCIe row).
- **Metrics.** tokens/s; ms/step; % of bound; `gpu_busy_frac`; kernels per step; top-1 agreement with the fp32 reference (3.); NLL on a fixed 64k-token R3 slice for rungs 7–9.
- **Plot (`fig_lm_fast_ladder`).** Horizontal bars (x: tokens/s, y: rung), a diamond marker per rung at its own bandwidth bound (from the ledger), a thin error bar for min–max; annotate the cause each rung removes in the y label. A second panel (`fig_lm_fast_timeline`): two profiler timelines (eager vs graph) rendered as GPU-busy/idle strips for 3 steps.
- **Expected outcome.** Direction: rung 0–2 are launch-bound; the largest single jump is the graph (rung 5); bucketing helps only when `L` is small relative to `max_len`; int8 and int4 move tokens/s by less than 1.59× and 2.19× respectively unless the head is quantized. The PyTorch blog reports, on an A100 with Llama-7B, 25.5 → 107 tok/s from compilation (72% of bandwidth bound), 157.4 with int8 and 202.1 with int4 + GPTQ; magnitudes for R1 are to be measured and will differ because R1 has 28 small layers (more kernels per byte).
- **Noise check.** Report the min–max spread; do not claim an ordering between two rungs whose intervals overlap (explicitly flagged in the table with a "≈" marker computed from the intervals). The 4090 is shared by T1 slots: record `nvidia-smi` clocks and utilization before each repetition and discard a repetition if another process used the GPU (rule written in the cell).
- **T1 reduction.** None needed (this section is T1). The A100 column is optional and omitted from the committed artifact unless the build host has one.
- **Artifacts committed.** `lm-inference-fast-ladder.csv` (rung, tok/s per repetition, bytes per step, kernels per step), `lm-inference-fast-profile.json` (kernel census), `capture_log.txt`.

#### Figures and tables

- `fig_lm_fast_ladder`: the ladder, as above. Made in the headline cell.
- `fig_lm_fast_timeline`: GPU-busy strips, eager vs graph (profiler traces parsed to intervals).
- `fig_lm_fast_static_tax`: decode ms/step vs true context `L` at `max_len = 8192`, unbucketed vs bucketed vs ledger prediction.
- `tab_lm_fast_ledger`: bytes ledger for R1 (four weight formats × three `L`) with the ceiling in tokens/s.
- `tab_lm_fast_attn`: decode attention variants × `L` × `b` (ms).
- `tab_lm_fast_quality`: NLL and top-1 agreement for bf16/int8/int4/int4-head.
- `tab_lm_fast_design`: the eight design rules mapped to the symptom each prevents (one line each).

#### Pitfalls and misconceptions

- "Graphs make the kernels faster." They remove gaps between kernels; if `gpu_busy_frac` was already ~1, nothing changes (D2's GPU-bound regime).
- A graph that captured a Python integer (the position, the length) replays the stale value forever; positions must be device tensors, and a test advances the position across 3 replays and checks the cache column written each time.
- Timing without `torch.cuda.synchronize()`, or including capture/compile time in tokens/s; the table separates first-request latency (compile + capture) from steady state and reports both.
- `contiguous()` forgotten on a sliced cache view silently copies the cache every step (the `sec_kv-cache` leak again, now as bandwidth rather than memory); the profiler check in design rule 2 catches it.
- Quantizing `lm_head` of a tied model quantizes the embedding as well unless the tie is broken; `quantize_head=True` must untie explicitly and the test checks the embedding lookup is still bf16 or knowingly quantized.
- Reporting a speedup on int4 without a quality number.
- Comparing eager HF `generate` (which samples with processors, EOS checks and host syncs) with a bare graph loop as if they did the same work: the table's first rung lists what each loop does per token.

#### Summary (draft bullets)

- Decode time is the larger of a bandwidth term set by the bytes in the ledger and a launch term set by the number of kernels; eager decode of a small model is dominated by the second.
- A static cache with position tensors turns a decode step into a fixed-shape function, which a CUDA graph can replay without Python; length buckets stop the step from reading unused cache columns.
- One primitive, `forward(tokens, pos)`, serves prefill, decode and verification; this is what lets later sections add speculation and cache policies without new model code.
- Weight-only int4 cuts the bytes read, but the speedup is limited by whatever stays in bf16 (the tied LM head, then the KV cache): Amdahl's law applied to bytes.
- Every rung is measured against its own byte-ledger bound, so the remaining gap is always attributable.

#### Exercises

1. **(check) Why the head dominates.** R1's blocks hold 440.5 M parameters and the tied matrix 155.6 M. Explain why the tied matrix contributes 311 MB per decoded token even though the embedding lookup reads one row, and say what happens to the per-token bytes of Qwen3-4B (blocks 3.63 B, tied matrix 0.39 B). *Sketch:* the LM head multiplies by the full matrix each step; for 4B the head is 9.7% of weights so int4 on the blocks only gives 8.045 GB / (1.930 + 0.778) GB ≈ 2.97×, against 2.19× for R1 where the head is 26% of the bytes; quantizing everything gives 3.76× for both.
2. **(check) Graph or no graph?** Given a profiler summary with `gpu_busy_frac = 0.97`, `n = 900` kernels/step and `c = 8 µs`, decide whether a CUDA graph helps, citing D2. *Sketch:* `n·c = 7.2 ms`; if `Σg` already ≥ 7.2 ms the step is GPU-bound and graphs bring little; compute `Σg` from `gpu_busy_frac × T`.
3. **(derive) The staircase.** With length buckets `L_1 < L_2 < …` the step reads `L_b` columns where `L_b` is the smallest bucket ≥ `L`. Derive the expected wasted fraction of KV bytes for uniformly distributed `L ∈ [0, L_max]` with geometric buckets of ratio 2, and the bucket ratio that minimizes `waste + capture_cost` if each graph costs `κ` bytes of pool. *Sketch:* for ratio `ρ` and `L` uniform on `[0, L_max]`, `E[read] = L_max·ρ/(ρ+1)` and `E[L] = L_max/2`, so the wasted fraction of KV bytes is `(ρ−1)/(2ρ)`: 25% for `ρ = 2` against 50% with a single bucket. With `K = log_ρ(L_max/L_min)` graphs, minimize `(ρ−1)/(2ρ)·KV_bytes_cost + κ·K` over `ρ`; the notebook checks the 25% by simulation.
4. **(derive) When is quantization worth the dequantization compute?** For W4A16 with fused dequantization, show that decode at batch `b` stays bandwidth-bound while `b·2·N/P` + dequantization FLOPs per weight `δ·N` ≤ `W_int4/β`; solve for the maximum `b`, with `δ` ≈ 2 FLOPs per weight (shift, scale-subtract-multiply), on the 4090. *Sketch:* time_compute = `(2b + δ)N/P` vs time_mem = `0.53125 N/β`; ratio gives `b_max = 0.53125·P/(2β) − δ/2`; evaluate with datasheet numbers (note CUDA-core, not tensor-core, rate for `δ`).
5. **(measure) Bucket granularity.** Sweep `len_buckets` from one bucket to 32 and plot decode ms/step vs context for a request that grows from 128 to 2,048 tokens; predict the curve first from exercise 3. Record graph-pool memory as the price.
6. **(measure) Sync frequency.** Sweep `sync_every ∈ {1, 4, 16, 64}`; plot tokens/s. Predict from `c_sync` measured as the cost of one `.item()` after a graph replay; explain why the effect is larger for R1 than for an 8B model.
7. **(open) A fused int4 kernel.** Write a Triton GEMV for `M ∈ {1, …, 16}` that reads packed int4 codes and per-group scale/zero and accumulates in fp32; match `tinygemm` logits and report the fraction of its speed at b = 1 and b = 8. Discuss what limits you (unpack ALU work, tile shape, scale loads).

#### Further reading

- Papers: [FlashInfer](https://arxiv.org/abs/2501.01005) (v5: decode-shaped attention kernels, split-KV, block-sparse KV); [MARLIN](https://arxiv.org/abs/2408.11743) (new; why a 4-bit kernel can stay near-ideal up to batch 16–32 and how the layout is made for it); [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (cited in 5.2; the analytical roofline for decode).
- Tutorials and blogs: [GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/) (teaching source); [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html); [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).
- Code: [gpt-fast](https://github.com/pytorch-labs/gpt-fast) (v5; redirects to `meta-pytorch/gpt-fast`); [Marlin](https://github.com/IST-DASLab/marlin) (v5); [torchao](https://github.com/pytorch/ao) (new, verified); [flash-attention](https://github.com/Dao-AILab/flash-attention) (new, verified; `flash_attn_with_kvcache`).
- Datasheets: NVIDIA [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) and [RTX 4090 page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/) for β and P.

#### Risks and open decisions

- **R1's RoPE attributes** are fixed by 1.2's `Qwen3Attention` contract (`rope_theta`; interleaved layout after the load-time permutation), so no `_adapt` shim is needed. The remaining risk is numerical: 12.1's cached path must match `Qwen3GPT.forward` to 1.2's tolerance, which the 12.1 equality test checks.
- **torchao int4 layout drift.** If `_weight_int4pack_mm` or the config API changed in the pinned torchao, fall back to `kernel='ref'` for correctness and a Marlin repack for speed; if neither runs on Ada, report int8 only and keep int4 as a bytes-ledger prediction, clearly labelled.
- **Shared T1 slot noise.** Speed numbers on a shared 4090 may vary; the interleaved protocol and the min–max reporting are the mitigation; if the spread exceeds the gap between adjacent rungs, merge those rungs in the narrative.
- **`flash_attn` wheel availability** on the build image: the `'flash_kvcache'` row is optional and silently omitted if the import fails.
- **Graph capture of the LM head on `logits_at='all'`** at large `T` (verify) can exceed the pool; capture only the `T` values 12.2–12.3 request.
- **Compile times** of 1–3 minutes for `torch.compile` may break the 12-minute budget: use cached inductor artifacts via `DATA_HUB` or drop rung 4 to the committed table if the build host cold-starts (the table row is then marked "from committed artifact").

### 12.2 Speculative decoding

- **Label:** `sec_lm_speculative` · **Tier:** T1 · **Budget:** ≤ ~15 min wall-clock; peak ≲ 13 GiB (Qwen3-4B bf16 8.05 GB + Qwen3-0.6B bf16 1.19 GB + static caches for batch ≤ 32 at ≤ 512 context; the long-context arm uses batch 2) · **Codex units:** 13.7 (exact speculative decoding)
- **Builds on (book):**
  - `sec_kv-cache`: explicitly defers speculative decoding here; its exercise "Shared weight reads" (m drafted tokens cost no more than one while `m` is below the ridge) is the premise of the whole section, so we start from it and do not repeat it. Not re-taught: prefill vs decode, cache bytes.
  - `sec_hardware`: `eq_hw_decode_bound`, ridge points in `tab_gpu_specs`.
  - `sec_decoding`: the temperature/top-k/top-p/min-p dials. The new requirement here is that the sampler be a *function from logits to a distribution* (`processed_probs`), because acceptance compares two such distributions.
  - `sec_mdl-divergences-distances`: total variation distance. The acceptance probability equals `1 − TV(p, q)`; we use that fact and do not derive TV properties.
  - `sec_mdl-bayesian-computation` (importance/rejection sampling background): one-sentence link only.
  - `sec_compilation`: graphs, used through 12.1.
- **Builds on (this part):**
  - 12.1: `d2l.FastGenerator` (`forward/verify/commit/rollback`, `processed_probs`, `sample`, slots), `d2l.decode_ledger`, `d2l.bench_decode`.
  - 5.2: `d2l.ServingModel(...).decode_step(b, L)` (`eq_lm_hw_decode_step`), used as in 12.1; `d2l.chip_specs()` columns `p16_tflops`, `mem_bw_tbs`.
  - 1.2: `d2l.load_qwen3(model_id)`, both sizes; 1.5: `d2l.paired_bootstrap` (returns a `BootstrapResult`; read `.mean`, `.ci`) for intervals over prompts.
  - 11.1/7.6 quantizers only through `FastGenerator(weights='int4')` (the self-draft in the exactness test).
- **Exports (`#@save`):**
  - `d2l.sd_accept(p, q, draft_tokens, generator=None) -> (n_accepted, next_token)`: vectorized acceptance and residual sampling (below).
  - `d2l.Speculator` (base class) and `d2l.DraftModelSpeculator(draft_gen, sampling)`.
  - `d2l.speculative_generate(target, speculator, prompts, max_new_tokens, gamma=4, sampling=None, eos_id=None, return_stats=True) -> (list[list[int]], SpecStats)`; `FastGenerator.generate(..., speculator=...)` delegates to it.
  - `d2l.spec_speedup_model(alpha, gamma, *, b, L, target, draft, kappa=1, t_accept=0.0) -> float`; `target`/`draft` are small dicts with `n_act, weight_bytes, kv_bytes_per_token` (from `decode_ledger`) or measured step-time tables.
  - `d2l.spec_greedy_check(target, speculator, prompts, n_tokens)`: asserts greedy equality with plain decoding; the standard for Chapter 13 and A8 (run in fp32 with TF32 off, or under 13.2's `det_mode`; A8 check 8 requires one of the two for the verify pass).
  - `d2l.SpecStats` (dataclass): `target_calls, draft_calls, tokens_out, accepted_hist[γ+1], alpha_by_position[γ], t_draft, t_verify, t_accept, t_other`.
  - **Chapter 13 hand-off:** `Speculator.propose/commit` is what 13.2's scheduler calls per iteration for the rows that opted in; the scheduler owns slot admission, the speculator owns draft state keyed by slot.
- **`Speculator` protocol (normative):**
  ```python
  class Speculator:                                   # d2l.Speculator
      def reset(self, rows): ...                       # forget draft state for these slots
      def prefill(self, prompts, rows, target_out): ...  # draft ingests the prompt (target_out holds the target's prefill hidden states for feature drafters)
      def propose(self, last_tokens, rows, gamma): ... # -> (tokens [B, γ] LongTensor, q [B, γ, V] float32 | None)   None means deterministic (one-hot) drafts
      def commit(self, rows, n_accepted, next_token, target_out=None): ...  # draft state advances to the accepted prefix; target_out carries verified hidden states
      tree = None                                      # a DraftTree for tree drafters (12.3); None: a chain
  ```
- **Models and data:**
  - Target `Qwen/Qwen3-4B` (rev `1cfa9a7208912126459214e8b04321603b3df60c`, Apache-2.0), draft `Qwen/Qwen3-0.6B` (rev `c1899de289a04d12100db370d81485cdf75e47ca`); same tokenizer and chat template; non-thinking mode (`enable_thinking=False`) for the main arms, thinking mode for one exercise.
  - Prompts: 48 each from `HuggingFaceH4/ultrachat_200k` (`test_sft`, first user turn; MIT), `openai/openai_humaneval` (`test`, prompt field; MIT), `openai/gsm8k` (`main`, `test`, question; MIT). Seeded subsample (seed 0). `max_new_tokens = 128`.
  - Sampling arms: greedy; "Qwen non-thinking" (temperature 0.7, top-p 0.8, top-k 20, as given on the Qwen3 model card, **confirm at authoring time**); pure temperature 1.0.
- **Packages:** `transformers` (pinned by 1.1) for the `assistant_model` oracle; `numpy`; `scipy.stats.chi2` for the exactness test (or an inline survival function).

#### Narrative arc

1. **The question.** Opening cell: Qwen3-4B decodes at ≈ (bound) tokens/s at batch 1 on the 4090 (bf16 bound from the ledger: 1.008e12/8.05e9 ≈ 125 tok/s; measured: to be measured). Print `t_target(1) = W/β` and, beside it, the measured time of one *verification* forward of 5 tokens on the same graph machinery (`verify` with T = 5). Printed symptom: the 5-token pass costs close to one single-token pass (direction; to be measured), so five tokens "per weight read" are almost free *if we knew which tokens*. That is the whole idea.
2. **Draft, then verify.** One schematic figure (`fig_lm_spec_round`): a round = γ draft steps + one target pass over `[y, x_1..x_γ]` + acceptance; what is in each cache after the round.
3. **The acceptance rule (D1) with a toy.** Categorical `p, q` over 6 symbols; code the rule in 10 lines; histogram of 10⁶ outputs vs `p` (`fig_lm_spec_toy`), with the accepted fraction against `Σ min(p, q)`.
4. **Why the output is exactly `p` (D2).** The proof, then a *sequence-level* toy (a 3-symbol bigram "model" pair with an exact joint over 3 tokens): empirical frequencies of all 27 sequences from the full speculative loop vs the exact product `p₁p₂p₃`, χ² test. This is the test that catches off-by-one errors in the bonus token.
5. **How many tokens per target call (D3).** The `(1 − α^{γ+1})/(1 − α)` formula; Monte Carlo check; estimating `α_i` per position from traces; the single-α formula vs measured.
6. **The loop on real models.** `DraftModelSpeculator` with Qwen3-0.6B for Qwen3-4B: show the round in code; the `SpecStats` counters; correctness tests (greedy equality in the fp32 self-draft setting; HF `assistant_model` oracle).
7. **Acceptance depends on the task and the sampler.** α_i by position for chat, code, math under the three sampling arms; τ with bootstrap intervals over prompts.
8. **A wall-clock model with draft cost and batch size (D4)**, then validation: microbenchmark `t_target(b, T)` for `T ∈ {1, γ+1}` and `t_draft(b)` in graph and eager modes, feed the model, compare predicted and measured end-to-end speedup on cells that were not used to fit anything.
9. **Where speculation loses (headline).** Four regimes, each measured and predicted: (i) the draft runs eagerly (draft cost `c` near the target's own launch-bound cost), (ii) large batch with verify compute-bound, (iii) long context with a full-KV draft, (iv) a draft degraded until `α` falls to `c` (weights perturbed by Gaussian noise of growing relative norm), which tests the `α > c` rule directly. Each ends with the one-line fix (CUDA-graph the draft; lower γ or stop at `b_c`; cap the draft's context; a better-aligned drafter, 12.3).
10. **Hand-off.** What 12.3 changes (the draft is cheaper and better aligned) and what Chapter 13 changes (batch composition varies per iteration, so γ and the on/off switch become scheduler decisions).

#### Derivations

All four carry a numbered proof and a check cell; the checks are CPU-only and take seconds. Notation: `p` target distribution, `q` draft distribution over a vocabulary of size `V`, both *after* the same processors (temperature/top-k/top-p/min-p), `β = Σ_x min(p(x), q(x))`.

**D1. The acceptance rule and the residual distribution (`eq_lm_spec_accept`, `eq_lm_spec_resid`).**
- Statement: draw `x ~ q`; accept with probability `min(1, p(x)/q(x))`; if rejected, draw from `r(x) = max(p(x) − q(x), 0)/(1 − β)`. The probability of acceptance is `β = 1 − TV(p, q)`, and `β` is the largest P(X = Y) achievable by any coupling of `p` and `q` (maximal coupling).
- Assumptions: `q(x) > 0` wherever the draft can emit `x` (we sample `x` from `q`, so division is safe); `1 − β > 0` when a rejection can occur; deterministic drafts are `q = δ_x`.
- Steps: (1) `P(accept) = Σ_x q(x) min(1, p(x)/q(x)) = Σ_x min(q(x), p(x)) = β`. (2) The identity `Σ_x max(p − q, 0) = Σ_x (p − min(p, q)) = 1 − β`, so `r` is a distribution. (3) Coupling bound: for any joint law of `(X, Y)` with marginals `p, q`, `P(X = Y) = Σ_x P(X = Y = x) ≤ Σ_x min(p(x), q(x)) = β`; the rule attains it, so no exact single-draft rule accepts more often. (4) Deterministic drafts: `q = δ_{x_0}` gives `β = p(x_0)`, acceptance probability `p(x_0)`, residual `p` with `x_0` removed and renormalized; greedy target (`p = δ_{argmax}`) gives "accept iff the draft token is the argmax".
- Check cell: `p, q` ~ Dirichlet(1) on `V = 6` (seed 0), `N = 2·10⁶` draws: accepted fraction within 5 standard errors of `β`; residual sampling frequency of rejected draws within 5σ of `r`.

**D2. The output distribution is exactly the target's (`eq_lm_spec_exact`).**
- Statement: for one position, the emitted token `Z` satisfies `P(Z = x) = p(x)` for every `x`. For a block of `γ` drafts plus the bonus, the emitted tokens `(Z_1, …, Z_{n+1})` have the law of `n + 1` consecutive samples of the target (`P(Z_{1:k}) = Π p_i`), for every stopping rule that is a function of the acceptance outcomes.
- Assumptions: `p_i` is the target's processed next-token distribution given the *accepted* prefix; `q_i` is the draft's distribution given the same prefix plus its own earlier drafts (which are accepted prefixes when position `i` is evaluated); draws use fresh randomness; the target logits in the verify pass equal those a plain decode would produce (exact arithmetic; see the pitfall for floating point).
- Steps: (1) One position: `P(Z = x) = P(accept ∧ X = x) + P(reject) · r(x) = q(x) min(1, p(x)/q(x)) + (1 − β) · max(p(x) − q(x), 0)/(1 − β) = min(p(x), q(x)) + max(p(x) − q(x), 0) = p(x)`. (2) Block, by induction on the position index: position 1 is step (1) with `p = p_1`. Given that the first `i − 1` emitted tokens equal the accepted drafts `x_1..x_{i−1}` (event `A_{i−1}`), position `i` is evaluated with the correct prefix, and by step (1) its emitted token has law `p_i(· | prefix)` whether it comes from acceptance or from the residual; if the residual is drawn, the block stops. (3) Chain rule: `P(Z_1 = z_1, …, Z_k = z_k) = Π_{i ≤ k} p_i(z_i | z_{<i})`, because each factor was obtained conditionally on the event that the previous tokens were emitted. (4) Bonus: if all `γ` drafts are accepted, the emitted token `Z_{γ+1}` is drawn from `p_{γ+1}(· | x_{1..γ})`, which is already a plain target draw, so the law is again `Π p_i`. (5) Independence from the draft: the result holds for any `q_i`; a poor `q` only lowers `β`.
- Check cell: (a) the 6-symbol one-position test; (b) the sequence-level test: bigram tables `P, Q` over 3 symbols, γ = 2, speculative loop for `N = 3·10⁵` blocks producing 3-token continuations from a fixed start; compare the 27 empirical frequencies with the exact product; pass if the χ² statistic's p-value ≥ 0.01 for seed 0 and for ten other seeds none is below 0.001 (the cell prints the p-values; a failing implementation gives p ≈ 0).

**D3. Expected tokens per target call (`eq_lm_spec_tokens`).**
- Statement: if position `i`'s acceptance probability given that positions `< i` were accepted is `α_i`, then the number `N` of tokens emitted per round (accepted drafts plus one) has `E[N] = Σ_{i=0}^{γ} Π_{j=1}^{i} α_j`. With a constant `α`, `E[N] = (1 − α^{γ+1})/(1 − α)`; `α = E_prefix[β]` averaged over prefixes.
- Steps: (1) `N = 1 + A`, `A` = number of consecutive accepted drafts, truncated at `γ`. (2) `P(A ≥ i) = Π_{j ≤ i} α_j` for `i ≤ γ`. (3) `E[A] = Σ_{i=1}^{γ} P(A ≥ i)`. (4) Constant `α`: geometric sum. (5) Limits: `E[N] → 1/(1 − α)` as `γ → ∞`; `E[N] = γ + 1` at `α = 1`; `E[N] = 1` at `α = 0` (the bonus token makes speculation never worse than one token per call). (6) Estimating `α` from traces: `α_i^ = #{rounds with A ≥ i}/#{rounds with A ≥ i−1}`. Note the single-α formula is a model: `α_i` typically decreases with `i`, and the section prints its error against the exact formula.
- Check cell: Monte Carlo with `p, q` fixed as in D1 and the same context at every position (so the single-α model is exact): mean `N` within 5 standard errors of the formula for γ ∈ {1, 3, 5}; then on real traces, the exact `Σ Π α_j` reproduces the measured mean `N` to numerical precision (an identity), and the single-α formula's error is printed.

**D4. Wall-clock speedup including draft cost, batch size and context (`eq_lm_spec_speedup`).**
- Statement: with `t_t(b, T)` the target forward time for `b` sequences and `T` new tokens each, `t_d(b)` a draft step, `κ` extra draft steps per round (1 for the catch-up step of the simple implementation, ≈ 0 if the first step of the next round is fused) and `t_a` the acceptance cost,
  `S = E[N] · t_t(b, 1) / ( (γ + κ) · t_d(b) + t_t(b, γ+1) + t_a )`.
  With the roofline of 5.2 (`eq_lm_hw_decode_step`; `t_t(b, T) = max(2·N_t·b·T/P, (W_t + b·L·k_t)/β)` and the analogous `t_d`), four consequences follow: (a) bandwidth-bound idealization: `S = E[N]/(1 + (γ+κ)c)` with `c = t_d/t_t`; `S > 1` iff `Σ_{i=1}^{γ} α^i > (γ + κ)·c`, so for `κ = 0` some `γ` gives `S > 1` iff `α > c`. (b) The marginal-γ rule: `S(γ+1) > S(γ)` iff `α^{γ+1}(1 + γc) > c · (1 − α^{γ+1})/(1 − α)`. (c) Compute-bound verify: the verification pass leaves the bandwidth regime when `b·(γ+1)·2·N_t/P > (W_t + b·L·k_t)/β`, i.e. for `L = 0` at `b > b_c(γ) = b*/(γ+1)` with `b* = P·W_t/(2·N_t·β)`; beyond it `S → E[N]/((γ+1) + γ·N_d/N_t) < 1`: speculation always loses at large batch when the cache is small. (d) Long context (MagicDec): with KV, verification is compute-bound only if `b·(a(γ+1) − L·k_t/β) > W_t/β` with `a = 2·N_t/P`; if `a(γ+1) ≤ L·k_t/β`, i.e. `L ≥ L_c(γ) = (γ+1)·2·N_t·β/(P·k_t)`, the target never becomes compute-bound at any batch, so speculation keeps winning at large batch, *provided the draft is cheap*; a draft with a full KV cache has `c → k_d/k_t = 0.778` as `L → ∞` (Qwen3-0.6B vs Qwen3-4B), which destroys the gain; a draft with a capped context keeps `c` small.
- Assumptions: single GPU, graph-captured steps, no CPU overhead (otherwise add `δ`'s measured per step), draft and target not run concurrently, acceptance independent of `b`.
- Steps: (1) Time per round = `(γ+κ)·t_d + t_t(b, γ+1) + t_a`. (2) Tokens per round per sequence = `E[N]` (D3); sequences progress independently via per-slot `pos`, so batch throughput ratio equals the per-sequence ratio. (3) Plain decoding time per token = `t_t(b, 1)`. (4) Ratio gives `S`. (5) (a): with `t_t(b, γ+1) = t_t(b, 1)` and `t_d = c·t_t`, `S = f_γ/(1 + (γ+κ)c)` where `f_γ = Σ_{i≤γ} α^i`, so `S > 1` iff `Σ_{i=1}^{γ} α^i > (γ+κ)c`. For `κ = 0` the left side is at most `γα`, so `S > 1` requires `α > c`; and `γ = 1` gives `S = (1 + α)/(1 + c) > 1` iff `α > c`, so the condition is necessary and sufficient. (6) (b): `f_{γ+1} = f_γ + α^{γ+1}`; cross-multiply. (7) (c)–(d): equate the compute and memory terms of `t_t` and solve for `b`; numbers: on the RTX 4090 with datasheet peak, Qwen3-4B bf16 has `b* = 163.7`, so `b_c = 32.7` for γ = 4 and 18.2 for γ = 8; with int4 blocks (head bf16) `b* = 55.1`, `b_c(4) = 11.0`; `L_c(γ) = 333·(γ+1)` tokens (about 1.7 k at γ = 4), all computed from `config.json` and `tab_gpu_specs` with *peak* `P` (achieved `P` is lower, so real thresholds are lower). Weight quantization of the target therefore moves speculation's break-even batch down.
- Check cell: (i) a brute-force grid over `(α, γ, c)` confirms the sign of `S − 1` against `α > c` for `κ = 0` (no counterexample on a 100×10×100 grid); (ii) the marginal-γ rule predicts the argmax of `S(γ)` on 20 random `(α, c)` pairs; (iii) `spec_speedup_model` evaluated on the Qwen3-4B/0.6B ledger prints `b_c(γ)` and `L_c(γ)` and the regime map `fig_lm_spec_regimes` (axes `b` and `L`, colour `S`, contour `S = 1`), before any model is run.

#### From-scratch implementation

Budget ≤ ~140 lines.

| Component | Signature and contract | Lines |
|---|---|---:|
| `sd_accept` | `p [B,γ+1,V]`, `q [B,γ,V]` or `None`, `x [B,γ]` → `n_acc [B]`, `z [B]`. `r_i = p_i(x_i)/q_i(x_i)` gathered; `acc_i = u_i < min(1, r_i)` with `u ~ U(0,1)`; `n = cumprod(acc).sum(1)`; residual `max(p_n − q_n, 0)` normalized (fallback to `p_n` if its sum is < 1e-20); bonus row `p_γ` when `n = γ`; the draw `z` by inverse CDF on device. No host sync. | ~28 |
| `DraftModelSpeculator` | holds a `FastGenerator` for the draft; `propose` loops `γ` times with `decode_step`, `processed_probs`, `sample`; `commit` rewinds the draft's `pos` to the accepted prefix (`rollback`) and runs the catch-up step for rows with `n = γ` | ~35 |
| `speculative_generate` | the round: propose → `target.verify([y, x_1..x_γ])` → `processed_probs` of all rows → `sd_accept` → `target.commit(n+1)` → `speculator.commit` → write emitted tokens into an output buffer on device; EOS/limit handled per row every `sync_every` rounds | ~50 |
| `spec_speedup_model` | the formula of D4 using `d2l.ServingModel(...).decode_step(b, L)` (5.2, `eq_lm_hw_decode_step`) or measured tables | ~20 |

Numerical tests: (1) D1/D2 checks above (the proofs' executable form); (2) `sd_accept` with `q = p` accepts everything (`n = γ` always) and the bonus equals a draw from `p_γ` (frequency test); (3) `q` one-hot at the argmax of `p` accepts everything and emits the argmax; (4) `q` with disjoint support from `p`: always rejects and emits `p` (frequency test); (5) **greedy exactness on real weights**: target R1 in fp32 with TF32 off, draft = `FastGenerator(R1, weights='int4')`; `spec_greedy_check` on 20 prompts × 128 tokens must return *identical token ids* to plain greedy decoding (exact equality; if it fails, report the first differing position and the logit margin there, which should be below fp32 noise); (6) the same in bf16 with the 4B/0.6B pair: report the fraction of prompts identical and the median position of first divergence together with the target's top-2 logit margin there (near-ties; Chapter 13's batch-invariant kernels would remove the issue); (7) the `Speculator` state invariant: after every round, `draft.pos[row] == target.pos[row] − 1` for the accepted prefix (assert in debug mode).

#### In production

- **Hugging Face assisted generation** (`model.generate(assistant_model=draft)`; `do_sample` and `num_assistant_tokens`): run the same pair and prompts; report acceptance statistics where exposed and compare greedy output with ours in the fp32 R1 setting (oracle). The blog states the library's motivation and its limits.
- **vLLM / SGLang** speculative configurations: draft-model support has varied across vLLM versions (ngram, EAGLE/EAGLE-3 and MTP methods exist; **check the installed version for draft-model support**); read acceptance from their metrics rather than re-implementing. Present as a table of `method`, `parameters`, `what it measures`; no run required at T1.
- **gpt-fast** ships a speculative loop on the same static-cache design; the PyTorch blog reports roughly 2× for CodeLlama-34B with a 7B draft on code and about 1.3× for Llama-7B with TinyLlama, which our task table should echo in *direction* (code easier than chat).
- The from-scratch `sd_accept` is the oracle for any kernel-fused implementation (a Triton or CUDA rejection-sampling kernel); its tests are reused unchanged.

#### Headline experiment

- **Hypothesis.** (H1) Speculative sampling with Qwen3-0.6B drafting for Qwen3-4B yields `E[N] > 1` and a wall-clock speedup at batch 1 that is larger for code than for chat and smaller under temperature sampling than under greedy; (H2) the D4 model, fed with *measured* step times and the *measured* `E[N]`, predicts end-to-end speedup within 10% on held-out cells; (H3) speculation loses in each of the four regimes of step 9, in the direction predicted.
- **Setup.** Tier T1. Target and draft in bf16, both `mode='graph'` with `max_batch = 32`, `max_len = 1024` (long-context arm: `max_len = 4608`, batch 2). γ ∈ {1, 2, 3, 4, 6, 8}; batches {1, 2, 4, 8, 16, 32}; prompts as in "Models and data"; 3 seeds for sampling arms (greedy runs once). Timing: 5 interleaved repetitions per cell; the speedup is `tokens/s(spec) / tokens/s(target-only)` for the *same* prompts and batch.
- **Procedure.** (1) Accept statistics per (task, sampler) at γ = 8 and b = 1 (α_i for i ≤ 8; every smaller γ is a prefix of these traces, so one run serves the whole γ sweep; the trace is committed). (2) Microbenchmark `t_t(b, T)`, `T ∈ {1, 2, …, 9}`, and `t_d(b)` for graph and eager draft; `d2l.Benchmark`. (3) Model prediction grid. (4) End-to-end measurement on a subgrid of 12 cells never used in (2). (5) The four loss regimes: (i) draft `mode='eager'` at b = 1; (ii) b = 32 with γ = 8 (past `b_c`); (iii) context 4,096 at b = 2 with a full-KV draft; (iv) the degraded draft: relative noise `ε ∈ {0, 0.02, 0.05, 0.1, 0.2}` added to every draft weight, which sweeps `α` downward at constant `c`; plot `S` against measured `α` and mark `α = c`; each regime is shown against its fix.
- **Metrics.** `τ = E[N]` (tokens per target call), `α_i`, `S` (wall-clock), predicted `S^`, relative error `|S^ − S|/S`, `c = t_d/t_t` measured, draft/verify/accept time shares.
- **Plots.** `fig_lm_spec_alpha`: α_i vs position, one line per (task, sampler), intervals from the prompt bootstrap. `fig_lm_spec_grid`: heat maps of measured and predicted `S` over `(b, γ)` with the `S = 1` contour in both; third panel: predicted−measured scatter with the diagonal. `fig_lm_spec_losses`: bar chart of `S` for the baseline and each loss regime and its fix, with a dashed line at 1. `fig_lm_spec_regimes` (from D4's check, now overlaid with the measured cells as markers).
- **Expected outcome.** Directions: τ > 1 on every task; code ≥ chat; greedy ≥ sampling; `S(γ)` rises then falls with an interior optimum that the marginal-γ rule predicts; `S < 1` in regimes (i)–(iii), and `S` crossing 1 near `α = c` in (iv). Published magnitudes for calibration only: Leviathan et al. report 2–3× on T5-XXL, Chen et al. 2–2.5× for Chinchilla 70B with a 4B draft, the PyTorch blog 2× (CodeLlama-34B + 7B, code) and about 1.3× (Llama-7B + TinyLlama); our magnitudes are to be measured, and a 0.6B/4B pair is a *small* ratio (`c` near 0.15 with graphs), so we expect smaller gains than 2×.
- **Noise check.** Intervals on τ and `α_i` by bootstrap over prompts (1.5's `d2l.paired_bootstrap`, reading `.mean` and `.ci` of the `BootstrapResult`, for paired comparisons between samplers on the same prompts); timing from 5 interleaved repetitions with min–max shown; a "loss" is only claimed if the whole interval of `S` is below 1; the model-vs-measurement error is reported with its own repetition spread, and H2 is "supported" only if the median error is below 10% with the spread below the error.
- **T1 reduction.** None needed; if the build slot is busy, drop γ ∈ {1, 3} and batches {2, 16} (the committed full grid supplies them).
- **Artifacts committed.** `lm-inference-spec-traces.npz` (per prompt: accepted count per round, per task and sampler), `lm-inference-spec-steptimes.csv` (microbenchmarks), `lm-inference-spec-grid.csv` (measured and predicted `S`).

#### Figures and tables

- `fig_lm_spec_round`: schematic of one round, cache contents before and after.
- `fig_lm_spec_toy`: empirical vs exact `p` (D1/D2 toy) and the 27-sequence joint.
- `fig_lm_spec_alpha`, `fig_lm_spec_grid`, `fig_lm_spec_losses`, `fig_lm_spec_regimes` as above.
- `tab_lm_spec_tasks`: per (task, sampler): τ, α_1, α_4, S at γ = 4, with intervals.
- `tab_lm_spec_model`: the four consequences (a)–(d) of D4, each with its threshold for the Qwen3-4B/0.6B pair on the 4090 and on an A100 40 GB PCIe (1,555 GB/s, 312 TF from the datasheet).

#### Pitfalls and misconceptions

- "Speculative decoding changes the output." It does not, in distribution (D2); it does not in greedy decoding either except through floating-point noise between a T-token and a 1-token pass; those near-tie flips are real and measured here, and they matter for A8's batch-invariance checks (4 and 8).
- Applying the acceptance rule to *unprocessed* softmax distributions while the plain sampler uses top-p: the output is exact for the wrong target. Both `p` and `q` must come from `processed_probs` with the same settings (test 2 and a deliberate-mismatch demonstration).
- Forgetting the bonus token (emitting only the accepted prefix) or emitting the residual and the bonus: the sequence-level toy catches both.
- Counting tokens per *draft* call instead of per *target* call; `τ` is per target call, wall-clock `S` also includes draft and accept time.
- Measuring speedup against an eager baseline (flattering) instead of the graph-captured target-only decoder of 12.1.
- Believing acceptance is a model property: it depends on task, sampler, prompt template and draft size (`fig_lm_spec_alpha`).
- "More γ is better": `S(γ)` has an interior maximum (marginal-γ rule); rejected drafts also cost compute that competes at large batch.
- The catch-up draft step (`κ = 1`) is a real cost in the simple batched implementation.
- Recurrent/hybrid targets (Qwen3.5) cannot roll back a rejected suffix by decrementing a position, because recurrent state has already absorbed it; 12.3 returns to this for the MTP experiment, and 13.3 for state caching.

#### Summary (draft bullets)

- Verifying `γ` tokens costs about one weight read while the pass stays bandwidth-bound, so a cheap guess of `γ` tokens can be checked almost for free.
- Accept `x ~ q` with probability `min(1, p/q)`, else resample from the normalized `max(p − q, 0)`; the emitted token has law `p` exactly, and the acceptance probability `1 − TV(p, q)` is the best any exact rule can do.
- A round yields `(1 − α^{γ+1})/(1 − α)` tokens in expectation for constant acceptance `α`; in general `Σ Π α_j`.
- Wall-clock gain is that number divided by the cost of `γ + κ` draft steps and one wider target pass; some `γ` helps iff `α` exceeds the draft/target cost ratio `c` in the bandwidth-bound idealization.
- Speculation loses when the draft is slow (eager, or full-KV at long context), when the batch makes verification compute-bound, and when acceptance is low; the roofline model predicts where.

#### Exercises

1. **(check) Required acceptance.** With `c = 0.15`, `κ = 0`, γ = 4, what constant `α` gives `S = 2`? Is `α = 0.6` enough for any γ? *Sketch:* `E[N] = 2(1 + 4·0.15) = 3.2`, and solving `(1 − α⁵)/(1 − α) = 3.2` gives `α ≈ 0.775`. For `α = 0.6 > c` some γ gives `S > 1`, but `S(γ) = f_γ/(1 + 0.15γ)` peaks near 1.5 (γ = 2 or 3; its limit as γ → ∞ is 0), so `S = 2` is out of reach.
2. **(check) One residual.** For `p = (0.5, 0.3, 0.2)`, `q = (0.2, 0.5, 0.3)`: compute `β`, the residual, and verify `P(Z = x) = p(x)` by hand. *Sketch:* `β = 0.2 + 0.3 + 0.2 = 0.7`; residual `(1, 0, 0)`; `P(Z) = (0.2 + 0.3, 0.3, 0.2) = p`.
3. **(derive) Optimal γ.** Prove `S(γ + 1) > S(γ)` iff `α^{γ+1}(1 + γc) > c(1 − α^{γ+1})/(1 − α)` (`κ = 0`), and show that `S` is unimodal in γ. *Sketch:* `f_{γ+1} = f_γ + α^{γ+1}`; cross-multiply; the left side decreases geometrically while the right side increases in γ, so the sign changes at most once.
4. **(derive) Multiple drafts for one position.** The target distribution is `p`; the draft proposes two *independent* tokens `x^{(1)}, x^{(2)} ~ q`. Show that trying them in sequence with `p ← p`, then `p ← norm(max(p − q, 0))` after a rejection, and finally sampling from the twice-reduced residual yields exactly `p`. *Sketch:* apply D2 step 1 recursively: output law `= min(p, q) + (1 − β)·p'` with `p' = norm(max(p − q, 0))`, and the second trial outputs `p'` by the same argument. (This is the tree-verification rule of 12.3.)
5. **(measure) Temperature sweep.** For the code task, sweep temperature ∈ {0, 0.3, 0.7, 1.0, 1.3}. Predict whether `α_1` increases or decreases with temperature before running; plot `S` against temperature. Explain with `TV(p, q)` for softmax(z/T) pairs.
6. **(measure) Draft size and weight format.** Replace the bf16 draft by the int4 draft (`FastGenerator(weights='int4')`) and by the target's own first 12 layers (if you built it from `d2l.GPT` by truncation, with the final norm and head kept); plot `S` vs `c` and mark `α = c`.
7. **(open) Copy-heavy drafts.** In the RAG setting of 2.5 (answers quote the retrieved passages) an n-gram draft that copies from the prompt has `c ≈ 0`. Build it as a `Speculator` over R3 questions, measure `α` and `S`, and find the retrieved-context length at which it loses to the 0.6B draft.

#### Further reading

- Papers (v5): [Leviathan et al.](https://arxiv.org/abs/2211.17192) (algorithm and proof); [Chen et al.](https://arxiv.org/abs/2302.01318) (independent formulation, distributed setting); [MagicDec](https://arxiv.org/abs/2408.11049) (long context and large batch; the sparse-KV draft); [SpecDecode-Bench](https://specdecode-bench.github.io/) (workload-sensitive evaluation; take its list of conditions under which gains vanish).
- Tutorials: [Assisted Generation](https://huggingface.co/blog/assisted-generation) (new); [GPT, Fast, step 3](https://pytorch.org/blog/accelerating-generative-ai-2/) (v5 teaching source); Smola's [Efficiency in LLMs](https://alex.smola.org/posts/45-mlss-efficiency/) for the maximal-coupling picture.
- Code: [gpt-fast](https://github.com/pytorch-labs/gpt-fast) (`generate.py`, speculative path); [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) (new, verified; follows a redirect).

#### Risks and open decisions

- **Draft model for R1.** The Qwen3 family has no model smaller than 0.6B, so R1 itself cannot be a *target* with a standard separate draft; 12.2 pairs Qwen3-0.6B (draft) with Qwen3-4B (target), whose tokenizers are identical (same `tokenizer.json`, `vocab.json`, `merges.txt` and chat template at the pinned revisions, vocabulary 151,936), and R1 is that draft; 12.3's *separate-draft arm* is where the 103 M model for R1 is distilled, because no smaller Qwen3 sibling exists. A Qwen3.5 pair (0.8B draft, 9B target; tokenizers shared within the family, not with Qwen3) stays an exercise in 12.3, because verifying and rolling back the recurrent state is the new difficulty.
- **Memory at 4B + 0.6B + caches on a 15 GiB slot.** Peak ≈ 11–13 GiB at batch 32/context 512; if the slot is tighter, halve `max_batch` and drop the b = 32 cell (the compute-bound regime then rests on γ = 8 at b = 16 plus the model's prediction, stated as a prediction).
- **Reproducibility of acceptance under bf16 batch effects.** Acceptance counts depend on logits that vary slightly with batch size; the traces are committed and the figure analyses them, so rebuilds do not change plots; only the live timing rows can differ.
- **Qwen sampler settings** quoted from memory of the model card: confirm and cite at authoring time; if they differ, use the card's values.
- **vLLM draft-model support** may be absent in the pinned version: the "In production" table says so and the oracle falls back to HF assisted generation.

### 12.3 Drafters and multi-token prediction

- **Label:** `sec_lm_drafters` · **Tier:** T2 (reduced T1 config: ~10% of the training tokens from a committed regenerated shard, 16 prompts per task, one epoch; `config.TIER = 't1'`) · **Budget:** T2 ≤ ~3 h on one A100 40 GB (PCIe), of which data regeneration is expected to dominate (budget; the part's 1,555 GB/s is 1.31× slower per bandwidth-bound step than the 80 GB SXM's 2,039 GB/s, so re-measure and tighten after 12.1's batch-throughput measurement), peak ≲ 38 GiB usable (design target ≲ 30 GiB, to be measured; the global batch is run as micro-batches, see Setup); T1 ≤ ~15 min (training ≈ 8 min), peak ≲ 12 GiB. Part B (the Qwen3.5 MTP head) is T1 in both tiers: Qwen3.5-2B bf16 ≈ 4.5 GB · **Codex units:** 2.6 (auxiliary prediction), 13.8 (modern drafting mechanisms)
- **Builds on (book):** `sec_kv-cache` (cache rollback ideas only through 12.1), `sec_transformer-block` (the decoder block that the drafter reuses: RMSNorm, GQA, SwiGLU, QK-norm), `sec_decoding`, `sec_adamw` and `sec_scheduler` (the drafter is trained with AdamW and a warmup + cosine schedule; no recap), `chap_modern_rnn` (why a Gated-DeltaNet layer's state cannot be truncated; one sentence).
- **Builds on (this part):**
  - 12.1: `FastGenerator` with `hidden_layers=(2, 14, 25)`, `forward(return_hidden=True)`, `verify(tree=...)`, `commit(path=...)`, `KVCache.copy_cols`.
  - 12.2: `Speculator` protocol, `d2l.sd_accept`, `d2l.speculative_generate`, `d2l.spec_speedup_model`, `SpecStats`; the exactness proof (D2 there) is reused for trees.
  - 1.1: Qwen3.5-2B through Transformers; its configuration's `mtp_num_hidden_layers` and the 15 `mtp.*` tensors (1.1's annotated config table points here).
  - 1.2: R1; 1.5: `d2l.paired_bootstrap`; 14.8 (forward pointer): MTP as a pretraining objective; 14.6 (forward pointer): Qwen3.5's hybrid stack.
  - **assumed:** nothing beyond the 12.1/12.2 surface.
- **Exports (`#@save`):**
  - `d2l.DraftTree` (dataclass: `tokens [B,T]`, `parents [T]`, `depth [T]`, `mask [T,T] bool` (ancestor-or-self), `rank [T]`; node 0 is the committed token `y` (the root, `parents[0] = −1`, depth 0), so `T = 1 + N` for a budget of `N` draft nodes; `N` always means draft nodes and `T` the number of tokens in the verify pass), `d2l.build_static_tree(acc, n_nodes)` (from a measured accuracy table `acc[d, r]`), `d2l.expand_dynamic_tree(drafter_step, root_feat, depth, topk, n_nodes)` (EAGLE-2-style confidence-driven expansion).
  - `d2l.tree_accept_greedy(tree, target_argmax [B,N]) -> (path_idx [B, d+1], n_accepted [B], next_token [B])` and `d2l.tree_accept_sampling(tree, p_nodes, q_nodes, generator=None)` (multi-draft rule, D2).
  - `d2l.MedusaHeads(hidden, vocab, n_heads=4, lm_head=None)` and `d2l.MedusaSpeculator`.
  - `d2l.EagleDrafter(target_cfg, draft_vocab=32000, feature_layers=(2, 14, 25), n_unroll=3)` and `d2l.EagleSpeculator`; `d2l.build_draft_vocab(token_counts, size)` returning `(draft_to_target, target_to_draft)` index tables.
  - `d2l.NGramSpeculator(n=3, max_draft=8)` (prompt lookup baseline, no training).
  - `d2l.MTPHead.from_qwen35(model_id, revision)` and `d2l.MTPSpeculator` (HF target; teacher-forced evaluation plus rollout).
  - `d2l.regenerate_corpus(gen, prompts, n_tokens, sampling, shard_dir)` and `d2l.train_drafter(drafter, target, shard_dir, steps, lr, ...)`.
  - Data artifacts (`DATA_HUB`): `lm-inference-regen-r1-2m` (token ids, uint32 NPY shards, ~2 M tokens, committed for T1), `lm-inference-medusa-r1.pt`, `lm-inference-eagle-r1.pt`, `lm-inference-draft100m-r1.pt`, `lm-inference-draft-vocab-r1.npy`.
- **Models and data:**
  - Target R1 (`Qwen/Qwen3-0.6B`, rev `c1899de289a04d12100db370d81485cdf75e47ca`); Qwen3.5-2B (`Qwen/Qwen3.5-2B`, rev `15852e8c16360a2fea060d615a32b45270f8a8fc`, Apache-2.0) for Part B; optional reference drafter `AngelSlim/Qwen3-4B_eagle3` (rev `fd331e59626c8e95c392381a16ee59d518727fbb`; 218 M parameters in safetensors; config: one Llama-style layer, hidden 2560, 32 heads, 8 KV heads, FFN 9728, `draft_vocab_size` 32000; **licence:** the Hub carries no licence tag, but the repo's `License_AngelSlim_model_and_dataset.txt` states that AngelSlim's released models and datasets are Apache 2.0 (verified 29 Oct 2026; it lists third-party components separately, ShareGPT-GPT4 data under CC BY 4.0 and the Qwen base models under Apache 2.0); recorded here, and the drafter stays an optional run-time download, never committed).
  - Training prompts: `HuggingFaceH4/ultrachat_200k` `train_sft` (MIT), first-turn user messages, 80k seeded sample (T2), 9k (T1 shard). Responses are *regenerated by R1 itself* (non-thinking, temperature 0.7, top-p 0.8, top-k 20, 256 new tokens), so the drafter is trained on the target's own distribution.
  - Evaluation prompts: the 48-per-task sets of 12.2 (ultrachat `test_sft`, HumanEval, GSM8K), disjoint from training by construction.
- **Packages:** as 12.1/12.2; `safetensors` (reads the `mtp.*` tensors directly); `transformers` at the version that loads Qwen3.5 (the config says 4.57.0.dev0 or later); optional `vllm` (nightly, per the Qwen3.5 model card) and `sglang` for the MTP oracle; optional `specforge` as the training reference.

#### Narrative arc

1. **The question.** Opening cell: in 12.2 the draft was a 0.6B model whose cost ratio `c` was ≈ 0.15 at best. For R1 (0.6B, the very model that was 12.2's draft) no smaller sibling exists, so the separate-draft arm below is a 103 M model distilled here. Print R1's own byte ledger split into blocks (881 MB) and the tied head (311 MB); the 311 MB is what *any* drafter that reads a full-vocabulary head pays. Then print the measured `α_1` of a *trained-on-nothing* baseline: the n-gram prompt-lookup drafter on chat vs on an R3-quoting QA prompt. The symptom: free drafts help only when text repeats; learned drafters must supply the rest.
2. **Three ways to draft, one verifier.** Taxonomy table `tab_lm_draft_taxonomy`: separate model; heads on the target's top feature (Medusa); a small autoregressive head on target features plus the sampled token (EAGLE); heads trained jointly with the model (DeepSeek-V3/Qwen3.5 MTP); copy from the context (n-gram). Columns: what it reads, extra parameters, extra weight bytes per drafted token, needs training?, trained with or after the target, depth supported.
3. **Medusa heads.** The architecture (residual block per head, frozen shared LM head), why a head at position `t` predicting `t+k+1` is a harder task for larger `k` (the accuracy ladder), training on target-regenerated text with the loss weights `λ_k = 0.8^k`; per-head top-k accuracies `acc[k, r]`; the independence assumption between heads and its cost.
4. **Tree verification (D1, D2).** From a chain to a tree: the ancestor mask and position ids; `verify(tree)` through the 12.1 primitive (the graph is keyed by `N`, the topology is data); greedy acceptance walk; `commit(path)` copies accepted KV columns; tree-building from `acc` (static) and from drafter confidence (dynamic); the sampling rule for multiple children (D2).
5. **EAGLE: drafting in feature space.** Why a draft that sees the target's *hidden states* is better aligned than a separate model, and why it must also see the *sampled token* (the feature at position `t` is uncertain about which token was sampled); the EAGLE-3 changes: multi-layer feature fusion and direct token prediction; training-time test (the drafter sees its own predicted features during training so that depth > 1 matches inference); the reduced draft vocabulary and why (cost ledger, D3). Code: `EagleDrafter`, 60 lines.
6. **Training a drafter for R1 (T2).** Regeneration with `FastGenerator` (reports tokens/s, reusing 12.1's batch throughput); online features from the frozen target; the training loop; loss curves; held-out top-1 accuracy per unroll depth. T1 uses the committed shard.
7. **Drafters compared on one target (headline).** n-gram, separate 100 M draft model (distilled), Medusa chain and tree, EAGLE chain and tree, at greedy and at temperature 0.7; `τ`, `α_i`, `S` at batch 1; then batch dependence via D3's tree-size rule.
8. **Multi-token prediction heads that ship with a model (Part B).** DeepSeek-V3's MTP module and what Qwen3.5 ships (`mtp_num_hidden_layers = 1`; 15 tensors, 60.8 M parameters, shape table from the safetensors header); how Transformers and vLLM expose it (below); implement the head in ~45 lines and measure per-depth acceptance on Qwen3.5-2B (teacher-forced, greedy), validate against vLLM's reported acceptance if available. Include the note on pretraining vs deployment: the head was trained jointly (14.8), and here it is only *used*.
9. **Hybrid targets cannot roll back.** One short subsection: Qwen3.5's Gated-DeltaNet layers fold every processed token into a fixed-size state; a rejected suffix cannot be un-applied by decrementing `pos`. Options (snapshot state per drafted position and keep the accepted one; recompute the accepted prefix) with their bytes; points to 13.3 (state caching). This is why Part B is measured teacher-forced, not as an end-to-end loop.
10. **Where each drafter wins.** Summary figure: `S` vs batch size for each drafter at its best tree; the crossover where the cheapest drafter wins at large batch.

#### Derivations

**D1. Expected accepted length of a draft tree, and the optimal tree (`eq_lm_tree_length`, `eq_lm_tree_opt`).**
- Statement: under greedy verification, a tree of draft nodes `V` (root = the committed token, not counted) yields `E[A] = Σ_{v ∈ V} P(path(v) matches the target's greedy continuation)`, where `A` is the number of accepted draft tokens per round. If the node probabilities `π_v` are known, the `N`-node tree maximizing `E[A]` consists of the `N` nodes with largest `π_v`; and when `π` is non-increasing from parent to child this set is automatically a tree. With independent per-head accuracies, `π_v = Π_{u ∈ path(v)} acc[depth(u), rank(u)]`.
- Assumptions: greedy target (the matching path is unique); distinct candidate tokens among siblings; for the product form, independence across depths.
- Steps: (1) The target's greedy continuation is one path `y_1, y_2, …`. Node `v` at depth `d` is *matched* iff its tokens equal `y_1..y_d`. (2) Matched nodes form a chain from the root (a matched node's parent is matched) and contain at most one node per depth (siblings have distinct tokens). (3) Therefore `A` = number of matched nodes. (4) Linearity of expectation: `E[A] = Σ_v π_v`. (5) A budget of `N` nodes maximizes a sum of non-negative terms by taking the largest `π_v`, subject to closure under parents; since `π_child = π_parent · acc ≤ π_parent`, the top-`N` set is closed. (6) Complexity: a priority queue over the frontier, `O(N log N)`.
- Check cell: (a) enumerate every parent-closed subset with `N ≤ 6` nodes of the `3 × 3` candidate lattice (3 heads, top-3 each) for a random accuracy table (seed 0) and confirm the greedy tree attains the maximum `E[A]`; (b) simulate `10⁵` rounds from the independent model and compare the mean number of matched nodes with `Σ π_v` (within 5 standard errors); (c) on real R1 data, compute `Σ π_v` from the measured `acc` table and compare with the measured mean accepted length of the same tree; the *gap* is the cost of the independence assumption, and it is printed.

**D2. Multi-draft sampling is exact (`eq_lm_tree_sampling`).**
- Statement: let the target distribution at a node be `p`, and let the draft propose `k` i.i.d. children `x_1, …, x_k ~ q`. Process them in order: at step `i` accept `x_i` with probability `min(1, p_i(x_i)/q(x_i))` with `p_1 = p`; on rejection set `p_{i+1} = norm(max(p_i − q, 0))`; if all `k` are rejected, sample from `p_{k+1}`. The emitted token has law exactly `p`.
- Assumptions: i.i.d. children from the same `q`; independent uniforms; for a tree, apply the rule at each node of the accepted path (the children of different nodes are drawn from the drafter's distribution at that node).
- Steps: (1) `k = 1` is D2 of 12.2. (2) Induction on the number of remaining children: given target `p_i`, one trial emits `x ~ q` with acceptance `min(1, p_i/q)`; the output law is `min(p_i, q) + (1 − β_i) · P(rest emits ·)`, with `β_i = Σ min(p_i, q)`. (3) By the induction hypothesis, the remaining trials emit a sample with law `p_{i+1}` (the residual target), so the law is `min(p_i, q) + (1 − β_i) p_{i+1} = min(p_i, q) + max(p_i − q, 0) = p_i`. (4) Base: `p_{k+1}` is sampled directly. (5) For the path through the tree, the chain rule of 12.2 D2 applies node by node. (6) Remark: drafting *without* replacement (Sequoia) changes `q` between trials and needs a different residual update; we implement i.i.d. children (SpecInfer-style) and point to Sequoia.
- Check cell: categorical toy (`V = 6`, `k ∈ {1, 2, 3}`, `10⁶` draws each, seed 0): empirical law within 5σ of `p` per symbol; the probability that at least one of the `k` trials is accepted equals `1 − Π_i (1 − β_i)` with `β_i = Σ min(p_i, q)`, verified empirically.

**D3. Drafter cost, the LM-head effect, and hardware-aware tree size (`eq_lm_draft_cost`, `eq_lm_tree_size`).**
- Statement: a drafter's cost ratio is `c ≈ bytes_read(drafter step)/bytes_read(target step)` in the bandwidth-bound regime. For R1 (bf16): blocks 880.9 MB, tied head 311.2 MB, total 1,192.1 MB. (i) EAGLE-style drafter with a 32,000-token head: 55.8 M parameters = 111.7 MB per step, `c = 0.094`; with the full 151,936-token head (frozen target head): `111.7 − 65.5 + 311.2 = 357.4 MB`, `c = 0.30`. (ii) A separate 103 M-parameter draft (8 blocks, width 512, tied head): 206 MB, `c = 0.17`; of which the tied head is 155.6 MB (75%). (iii) Medusa-4 with a frozen shared head: 8.4 MB of blocks plus one read of the target head (311.2 MB) per round, `c_heads = 0.27`; with four *separate* full-vocab heads: `4 · 311.2 + 8.4 = 1,253 MB`, `c = 1.05`. (iv) Tree size: adding the `N`-th node (path probability `π_(N)`, the `N`-th largest) raises `E[A]` by `π_(N)` and raises verify time by `Δt_t(b, N) = t_t(b, N+1) − t_t(b, N)`, which is ≈ 0 while `b·(N+1)` is below the ridge batch `b*` and `2·N_t·b/P` per node beyond it; so grow the tree while `π_(N) · t_t(b, 1) > Δt_t(b, N)`: the optimal `N` falls with batch size and is `≈ b*/b` (in nodes) where the ridge is crossed.
- Steps: (1) Bytes of a drafter step: parameters read once per step (embedding lookups read rows, not tables). (2) Substitute: the numbers above are arithmetic on the shapes. (3) The tree rule: `S(N) = (1 + E[A](N)) · t_t(b, 1)/(t_draft + t_t(b, N+1) + t_a)`; compare `S(N)` and `S(N−1)`. (4) Use the measured `t_t(b, T)` table of 12.2 (`T` up to 64) for the bandwidth/compute transition.
- Check cell: print the ledger table for the four drafter designs from their configs (assert agreement with `sum(p.numel())·2` of the instantiated modules within 1%); then for batch ∈ {1, 4, 16, 32} compute `argmax_N S(N)` from the measured `t_t(b, N+1)` and the measured `π_(N)`; plot `N*(b)`.

#### From-scratch implementation

Budget ≤ ~150 lines of drafter and tree code (the training loop and the regeneration driver are library-style helpers and are counted separately, ~60 lines).

| Component | Contract | Lines |
|---|---|---:|
| `MedusaHeads` | `K` residual blocks `h + silu(W_k h)`, shared frozen `lm_head`; `forward(h [B,H]) -> logits [B,K,V]` | ~15 |
| `DraftTree` builders | `build_static_tree` (heap over frontier), `expand_dynamic_tree` (top-k by cumulative log-prob per depth, global top-`N` re-rank) | ~35 |
| `tree_accept_greedy` | walks children, picks the child equal to the parent's argmax, returns path indices and bonus | ~20 |
| `tree_accept_sampling` | D2's rule over children of each visited node | ~25 |
| `EagleDrafter` | `fc: 3H→H`, one decoder layer whose attention reads `[norm(emb), norm(feat)]` (2H), final norm, head `H→32,000`; `forward(feats, tokens, kv)` with its own KV cache; `d2t` mapping | ~60 |
| `MTPHead` | `fc: 2H→H` on `[RMSNorm(emb), RMSNorm(hidden)]`, one full-attention layer with gated query and QK-norm, final norm, shared head | ~45 |

Numerical tests: (1) D1/D2 checks (enumeration, simulation, χ²); (2) chain consistency: a degenerate tree that is a chain behaves identically to 12.2's `verify` (same logits, same commit); (3) mask test: for the tree with parents `[−1, 0, 0, 1, 1, 2]` (node 0 is the committed token), `FastGenerator.verify(tree)` logits at each node equal the logits of running that node's root-to-node path as a plain sequence (fp32, TF32 off, tolerance 1e-5 relative): **the key tree test**; (4) commit test: after `commit(path)` the cache columns equal those of a plain decode of the accepted tokens; (5) `MTPHead` shape test against the safetensors header (exact shapes and parameter count 60,828,160); (6) `EagleDrafter` overfit test: on one 64-token sequence it must reach 100% top-1 on its own training tokens in ≤ 300 steps (catches mask and shift bugs); (7) draft-vocab round trip: `target_to_draft[draft_to_target[i]] = i`; coverage of the 32k most frequent training tokens is printed (fraction of next-token mass inside the draft vocabulary: the ceiling on `α_1`).

#### In production

- **EAGLE** repository and **SpecForge** (SGLang's training framework) as the reference for training-time test and data regeneration; **AngelSlim**'s released `Qwen3-4B_eagle3` as a trained reference: (a) load its `config.json` and weights, print the parameter table, and compare with our `EagleDrafter` (documenting what differs: layer count, vocabulary map, which layers feed `fc`); (b) optional: run it through vLLM or SGLang's EAGLE-3 mode on the 12.2 prompt sets for the 4B target and compare its acceptance length with the 0.6B draft of 12.2 on identical prompts.
- **vLLM** exposes MTP as `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` (Qwen3.5 model card, vLLM recipe page) and **SGLang** as `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`; both read the `mtp.*` weights from the main checkpoint. Read per-position acceptance from vLLM's spec-decode metrics (**check the metric names of the installed version**).
- **Transformers**: the Qwen3.5 classes load the base model; whether the `mtp.*` tensors are loaded is decided by `from_pretrained(..., output_loading_info=True)`: the notebook prints `unexpected_keys` and states what it finds (expected: the 15 `mtp.*` names are reported unexpected and ignored; if they are loaded, use the library module as the oracle for `MTPHead`).
- The from-scratch `MTPHead`'s depth-1 logits are the oracle for the vLLM path: compare top-1 on 1,000 positions.

#### Headline experiment

- **Hypothesis.** (H1) On R1 as target, feature-level drafters (EAGLE-style) reach a larger mean accepted length `τ` than a separate 100 M draft distilled on the same tokens, and Medusa heads sit between n-gram and EAGLE; (H2) trees raise `τ` over chains at batch 1, and the benefit in *wall-clock* shrinks as batch grows, as D3 predicts; (H3) the Qwen3.5 MTP head's acceptance falls with depth, with the first position much higher than the rest.
- **Setup.** *Part A (T2):* target R1 bf16 on one A100 40 GB (PCIe); `FastGenerator(mode='graph', hidden_layers=(2, 14, 25))`; regeneration of 20 M tokens (T2; T1 uses the committed 2 M); training: AdamW (β = 0.9, 0.95), peak LR 3e-4, warmup 50 steps, cosine to 10%, global batch 128 sequences × 512 tokens run as 8 micro-batches of 16 sequences (8,192 tokens; full-vocabulary bf16 logits for the online KL are 8,192 × 151,936 × 2 B = 2.5 GB per micro-batch, where one 65,536-token batch would need 19.9 GB), 3 epochs (T2) / 1 epoch (T1), bf16 autocast, target frozen with online features; EAGLE `n_unroll ∈ {1, 3}` (ablation), Medusa `K = 4`, separate draft by token-level distillation (KL to R1's temperature-1 next-token distribution, computed online over the full vocabulary). Evaluation: 48 prompts × 3 tasks (T2) / 16 × 3 (T1), greedy and temperature 0.7 (top-p 0.8, top-k 20), `max_new_tokens = 128`, γ = 4 chains and trees with `N ∈ {8, 16, 32, 64}`, batch 1 for `τ` and `S`, batch ∈ {1, 4, 16, 32} for the D3 sweep. *Part B (T1):* Qwen3.5-2B HF target, 48 prompts × 3 tasks (16 at T1), greedy continuations of 128 tokens, teacher-forced draft depths 1–3.
- **Procedure.** (1) Calibrate `acc[k, r]` for Medusa and EAGLE depth ranks on a 2k-position held-out slice of the training distribution; (2) train; (3) log held-out top-1 per depth every 100 steps; (4) evaluate `τ` and `S`; (5) tree-size sweep; (6) D1(c) independence gap; (7) Part B.
- **Metrics.** Mean accepted length `τ` (tokens per target call, including the bonus), `α_i`, `S` vs target-only graph decoding, drafter step time and its measured `c`, training tokens, final held-out top-1 per depth, and for Part B the acceptance of depth 1, 2, 3 (greedy).
- **Plots.** `fig_lm_draft_tau`: grouped bars of `τ` by drafter × task × sampler with bootstrap intervals. `fig_lm_draft_train`: held-out per-depth top-1 against training tokens (log-x). `fig_lm_draft_tree`: `τ` and `S` against tree size `N` for Medusa and EAGLE at b = 1 and b = 16. `fig_lm_draft_batch`: best-tree `S` against batch for each drafter. `fig_lm_draft_mtp`: Qwen3.5 MTP acceptance by depth and task.
- **Expected outcome (direction; magnitudes to be measured).** `τ`: n-gram < Medusa < separate draft ≲ EAGLE; trees > chains; training-token scaling is still steep at 20 M tokens (EAGLE-3 reports gains from far more data and from regenerating responses with the target; we are data-starved by design). Published calibration: the Medusa paper reports speedups for Vicuna models of roughly 2.2–3.6× with its heads and tree (Medusa-1/-2); EAGLE-3 reports up to about 6.5× over vanilla decoding on its benchmarks; DeepSeek-V3's report gives an 85–90% acceptance rate for its second-token MTP prediction and about 1.8× TPS. These numbers are *quoted for orientation only* and must be re-checked against each paper's tables at authoring time; none is a prediction for R1, whose per-step cost is dominated by the 311 MB tied head.
- **Noise check.** Bootstrap over prompts for `τ`; paired differences between drafters on identical prompts via `d2l.paired_bootstrap`; two training seeds for EAGLE (T2) and the claim `EAGLE > separate` is made only if the paired 95% interval of `τ` excludes 0 under both seeds; timing as in 12.2 (5 interleaved repetitions, min–max).
- **T1 reduction.** Config flag `TIER='t1'`: 2 M committed tokens, one epoch, 16 prompts per task, `n_unroll = 3` only, trees with `N ∈ {8, 16}`; the notebook then *analyzes the committed T2 tables* for the full comparison and prints both side by side.
- **Artifacts committed (T2).** Regenerated shard (token ids), the three checkpoints (≈ 56 M, ≈ 4 M, ≈ 103 M parameters), training logs (CSV), `acc` tables, per-prompt acceptance traces (NPZ), `lm-inference-mtp-q35-traces.npz`.

#### Figures and tables

- `tab_lm_draft_taxonomy`; `tab_lm_draft_cost` (D3 ledger for the four designs, with `c`); `tab_lm_draft_mtp_shapes` (the 15 `mtp.*` tensors with shapes and parameter counts, from the safetensors header); `tab_lm_draft_results` (`τ`, `α_1`, `S` per drafter/task/sampler with intervals).
- `fig_lm_draft_tree_mask`: a 6-node tree, its ancestor mask and position ids (built by `DraftTree`); `fig_lm_draft_eagle`: block diagram of the EAGLE-style drafter and of the training-time-test unroll; `fig_lm_draft_tau`, `fig_lm_draft_train`, `fig_lm_draft_tree`, `fig_lm_draft_batch`, `fig_lm_draft_mtp` as above.

#### Pitfalls and misconceptions

- Evaluating a drafter on text it was trained on (or on teacher-forced *human* text instead of the target's own generations): acceptance is measured on the target's greedy or sampled continuations of held-out prompts.
- Training the drafter on human text instead of target-regenerated text, which mismatches the inference distribution (the reason EAGLE-3 regenerates).
- Letting the draft use features of *rejected* tokens: after a rejection the drafter must restart from verified target features (tested through the commit invariant).
- A tree mask or position-id off-by-one: a tree whose nodes silently attend to siblings (test 3 catches this).
- Counting a drafter with full-vocab LM head as "tiny": D3 shows the head dominates for small targets.
- Comparing `τ` across drafters that use different tree budgets: always report `N` and the verify cost.
- MTP heads in hybrid models: no rollback of recurrent state (step 9); teacher-forced acceptance is not an end-to-end speedup.
- Using the wrong target hidden state for the MTP head (before or after the final norm). The head normalizes its inputs, but the final norm's elementwise weight still changes the direction; the notebook tests both and shows the gap in depth-1 acceptance (the correct choice is the one consistent with vLLM).
- Quoting published speedups without the benchmark, hardware and batch size.

#### Summary (draft bullets)

- Drafters differ in what they read and how much they cost per drafted token; for small targets the shared vocabulary head dominates that cost, which is why EAGLE-3 trims the draft vocabulary.
- Medusa predicts several future tokens from the last hidden state with independent heads; EAGLE drafts autoregressively in feature space and sees the sampled token, which aligns it with the target and lets it use its own features during training (training-time test).
- A draft tree is verified in one pass with an ancestor mask; under greedy verification the expected accepted length is the sum of node match probabilities, so the best `N`-node tree takes the `N` most probable nodes; multi-child sampling stays exact with the recursive residual rule.
- The right tree size shrinks with batch size, because extra nodes cost compute once verification leaves the bandwidth regime.
- MTP heads shipped with a model (DeepSeek-V3, Qwen3.5) are drafters trained jointly with the base; using them needs the head, the weight tying, and, for hybrid models, a state-rollback plan.

#### Exercises

1. **(check) Drafter ledger.** Compute the bytes per step and `c` for an EAGLE-style drafter on R1 with vocabularies of 32,000 and 151,936, and for the same drafter on Qwen3-4B (hidden 2560, head 388.9 M parameters). *Sketch:* R1: 111.7 MB vs 357.4 MB against 1,192 MB; 4B: the head is 9.7% of target bytes, so the full-vocabulary head costs 0.78 GB against 8.05 GB (`c` from the head alone ≈ 0.10, plus the drafter's own layer): vocabulary trimming matters less for large targets.
2. **(check) Tree mask.** For `parents = [−1, 0, 0, 1, 1, 2]` (node 0 is the committed token, `T = 6`): write the ancestor-or-self mask, the position offsets, and the set of new tokens that node 4 attends to. *Sketch:* depths `[0, 1, 1, 2, 2, 2]`; position ids are `pos + depth`; node 4's ancestors are 1 and 0, so it attends to `{0, 1, 4}` among the new tokens, plus every cached prefix column; siblings 3 and 5 and the other branch (2) are masked.
3. **(derive) The independence gap.** Show that `E[A] = Σ_v π_v` holds without independence if `π_v` is the true probability that the path to `v` matches, and that top-`N` by `π_v` is optimal; then give a two-head example where independence-based `π` ranks two nodes in the wrong order. *Sketch:* steps (1)–(4) of D1 never use independence. Example: depth-1 candidates `a` (`P(match) = 0.5`) and `b` (0.3); a depth-2 candidate `c` with `P(c matches | a matches) = 0.1` and `P(c matches | b matches) = 0.9`, so its marginal accuracy is `0.5·0.1 + 0.3·0.9 = 0.32`. Independence gives `π(a,c) = 0.5·0.32 = 0.16` and `π(b,c) = 0.3·0.32 = 0.096`, ranking `(a,c)` first; the true values are `0.05` and `0.27`, the opposite order.
4. **(derive) Untied Medusa heads.** Show that `K` Medusa heads each with its own full-vocabulary matrix cost `K` times the head bytes, hence `c ≥ K·(311.2/1,192.1) = 0.26K` for R1, and find the vocabulary size `V'` at which `K = 4` heads cost the same as a single EAGLE step. *Sketch:* each head row costs `2·H = 2,048` bytes, so `4·V'·2,048 B = 111.7 MB` gives `V' ≈ 13.6 k`; a full 151,936-token vocabulary is eleven times larger.
5. **(measure) Data scaling.** Train the EAGLE-style drafter on 0.5 M, 2 M, 8 M and 20 M tokens (T1: only the first two) and plot held-out top-1 and `τ` against tokens on a log axis; predict the 20 M value from the first two points by a log-linear fit before running, and report the prediction error.
6. **(measure) Unroll depth.** Compare `n_unroll ∈ {1, 2, 3}` at depth-3 acceptance; predict the sign before running. Explain with the train/inference mismatch of feature inputs.
7. **(open) A drafter for Qwen3.5.** Train an EAGLE-style drafter for Qwen3.5-0.8B using Transformers features (no rollback, so evaluate teacher-forced), or implement the state-snapshot variant of step 9 for its Gated-DeltaNet layers and measure the bytes it adds per drafted token. Or, as a separate-draft version, pair Qwen3.5-0.8B (draft) with Qwen3.5-9B (target) in 12.2's `DraftModelSpeculator`: both models share one tokenizer (identical `tokenizer.json`, Apache-2.0 weights; Qwen3 and Qwen3.5 cannot be mixed), the 9B target takes a whole 24 GB GPU (17.9 GB of text-only bf16 weights), and the new work is rolling back the recurrent state of *both* models after a rejection; report the bytes of state snapshotted per round and the acceptance length against the 12.2 Qwen3-0.6B/4B pair.

#### Further reading

- Papers (v5): [Medusa](https://arxiv.org/abs/2401.10774) (heads, tree attention, typical acceptance); [EAGLE-3](https://arxiv.org/abs/2503.01840) (multi-layer features, training-time test); [Multi-token prediction](https://arxiv.org/abs/2404.19737) (the objective; deployed as self-speculation); [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (the MTP module and its reported acceptance). New and verified: [EAGLE](https://arxiv.org/abs/2401.15077) (feature-level drafting; the feature-uncertainty argument); [EAGLE-2](https://arxiv.org/abs/2406.16858) (dynamic draft trees); [SpecInfer](https://arxiv.org/abs/2305.09781) (tree verification with multi-step speculative sampling); [Sequoia](https://arxiv.org/abs/2402.12374) (hardware-aware tree construction and sampling without replacement).
- Tutorials: the Qwen3.5 [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) (new, verified; how MTP is switched on in practice); [EMNLP 2025 tutorial](https://aclanthology.org/2025.emnlp-tutorials.1/) for the taxonomy.
- Code: [EAGLE](https://github.com/SafeAILab/EAGLE) (v5); [SpecForge](https://github.com/sgl-project/SpecForge) (new, verified; training EAGLE-3 drafters); [prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) (new, verified; the n-gram baseline).

#### Risks and open decisions

- **EAGLE-3 details** (loss definition, attention-mask layout of the training-time test, the `fc` input layers) are specified here from the paper's abstract and the released config; the author must read §3 of the paper and the EAGLE-3 trainer in the EAGLE repo and align the notebook; where the paper and our simplification differ the notebook says so (no claim of reproducing the paper's numbers).
- **Time budget of regeneration.** If 20 M regenerated tokens exceed the T2 budget at the measured 12.1 batch throughput, reduce to the token count that fits and move the data-scaling exercise to the committed logs.
- **Which hidden state feeds the Qwen3.5 MTP head** (pre- or post-final-norm) is not documented in the model card: determined empirically against vLLM acceptance; if vLLM cannot be installed on the build host, the depth-1 acceptance with the better choice is reported with that caveat.
- **Transformers support for Qwen3.5** (version pins) and nightly vLLM: the notebook has a no-vLLM path (teacher-forced measurement only) and states the missing validation explicitly.
- **AngelSlim licence** (no Hub tag; Apache 2.0 per the repo's licence file, recorded in Models and data) and weight-format drift (`pytorch_model.bin` for the 1.7B head, safetensors for 4B): download at run time, never redistribute; if the port into `EagleDrafter` takes too long, the reference arm uses vLLM/SGLang only.
- **No smaller Qwen3 sibling** for R1: the separate-draft arm uses a 103 M model distilled in the same session, so it is a *trained-here* baseline, not a released one; say so wherever it is compared with released EAGLE-3 heads.

### 12.4 Compressing the KV cache at inference time

- **Label:** `sec_lm_kv_compression` · **Tier:** T1 (the extended grid is precomputed offline on any host and committed; the notebook re-runs only a reduced grid live) · **Budget:** reduced grid ≤ ~15 min on one 4090 slot (sized by the formula in the headline experiment; to be measured), peak ≲ 6 GiB (R1 1.2 GB, prefill of 8k tokens, snapshot of one prefill's KV 0.94 GB, explicit-attention blocks 0.27 GB) · **Codex units:** 13.5 (quantizing the KV cache), 13.6 (reading or retaining less state)
- **Builds on (book):**
  - `sec_kv-cache`: `eq_kv-cache-bytes`; sliding windows and attention sinks (`subsec_kv-cache-sinks`; reused as the baseline eviction policy, not re-taught); the cache-reduction comparison table (this section adds two post-hoc rows: quantized KV, evicted KV); GQA; the low-rank/MLA sketch (a *training-time* design; this section is about methods applicable to any checkpoint, MLA proper is 14.3).
  - `sec_positional-information`: the RoPE rotation of channel pairs (the premise of pre-RoPE quantization).
  - `sec_attention-at-scale`: online softmax; FlashAttention never materializes attention probabilities, which is why H2O/SnapKV need an explicit path.
  - `sec_numerics`, `sec_memory_precision`: formats; the uniform-quantization error `Δ²/12` (owned by 7.6; used, not re-derived beyond the key-specific consequence).
  - `sec_mdl-statistics`: bootstrap, via 1.5.
- **Builds on (this part):**
  - 12.1: `FastGenerator`, the `KVCache` protocol (`append/view/copy_cols/nbytes`, `stores_pre_rope`), `KVPolicy` (`needs_probs`, `prefill_probs`, `observe`, `compact`), `kv_len` vs `pos`, `d2l.decode_ledger(..., kv_bits=)`, `d2l.bench_decode`.
  - 12.2: the long-context regime (a draft whose KV is capped); the policies here are what 12.2's exercise "cap the draft's context" calls for.
  - 7.6/11.1: group-wise absmax/min–max quantization vocabulary and the *effective bits including scales* convention.
  - 1.5: `d2l.EvalSuite([...task ids], n=..., seed=...).run(model, tok)` returning a `d2l.EvalResult` (per-instance scores via `.scores()`), `d2l.paired_bootstrap(scores_a, scores_b, n_boot=10000, cluster=None) -> BootstrapResult` (read `.mean` and `.ci`); 2.4–2.5: `d2l.book_corpus` (haystack), `d2l.book_qa`.
  - 5.2: the decode step-time model (bytes of KV in `t(b, L)`); 5.3/13.3 are forward pointers (offload and paging).
- **Exports (`#@save`):**
  - `d2l.quantize_groupwise(x, bits, dim, group) -> (codes uint8, scale, zero)` and `d2l.dequantize_groupwise(codes, scale, zero, dim, group, dtype)`; asymmetric min–max, `scale = (max − min)/(2^bits − 1)`, `zero = min`; `d2l.pack_bits(codes, bits)` / `d2l.unpack_bits(packed, bits, n)` (2, 3→ stored as 4, 4, 8 bits; 3-bit is reported with its true 3-bit cost and stored in 4-bit lanes, flagged in the byte ledger).
  - `d2l.QuantKVCache(num_layers, max_batch, n_kv, max_len, head_dim, *, bits_k=2, bits_v=2, k_scheme='channel', v_scheme='token', group=32, residual=128, pre_rope=False, sink_fp=0, dtype=torch.bfloat16, device='cuda')`: a `KVCache` backend (KIVI-style when `k_scheme='channel'`, naive per-token when `'token'`); stores packed codes and bf16 scale/zero, keeps the latest `residual` tokens in bf16, and dequantizes in `view`.
  - `d2l.SinkWindowPolicy(sinks=4, window)`, `d2l.H2OPolicy(budget, recent, slack=16)`, `d2l.SnapKVPolicy(budget, window=32, pool=7)`, `d2l.RandomEvictPolicy(budget, recent)` (control); all share the `KVPolicy` interface and keep **true positions** in the keys (post-RoPE storage; eviction never renumbers).
  - `d2l.ruler_task(name, length, n, seed, tokenizer, corpus=None) -> list[dict(prompt_ids, answers, meta)]` for names `'s_niah'`, `'mk_niah'`, `'vt'`, `'cwe'`, `'squad_qa'`, `'hotpot_qa'`, `'book_qa'`, `'multi_q'`; `d2l.ruler_score(name, outputs, answers) -> np.ndarray[n]` (string-match recall as in RULER; QA contain-match).
  - `d2l.kv_sweep(gen, instances, configs, max_new_tokens, seed) -> DataFrame`: prefills each instance once, snapshots the prefill KV, then branches every config (restore snapshot → compress → decode); columns `task, length, instance, config, score, kv_bytes, kv_frac`.
  - `d2l.attention_capture(attn_probs, kept_idx, future_probs) -> float`: attention mass captured by a kept set (D3).
- **Models and data:**
  - R1 (`Qwen/Qwen3-0.6B`, rev `c1899de289a04d12100db370d81485cdf75e47ca`, native context 32,768 with 40,960 positions in `config.json`); fallback flag `MODEL='Qwen/Qwen3-4B'` (rev `1cfa9a7208912126459214e8b04321603b3df60c`, Apache-2.0; 4,022,468,096 parameters = 8.05 GB bf16; 36 layers, 8 KV heads, head_dim 128, so the KV cache is `2·36·8·128·2` = 147,456 B per token, against R1's 114,688; the model 12.2 already loads) if R1 is at the accuracy floor on the retrieval tasks at 4k. The choice rests on the Qwen3 technical report (arXiv 2505.09388, appendix Table 23), which publishes RULER only from 4B up: non-thinking Qwen3-4B scores 95.1 at 4K and 93.6 at 8K (average 85.2 over 4K–128K; thinking mode 92.7 and 88.7), measured with YaRN factor 4; it publishes no RULER score for 0.6B or 1.7B. Fit on a 15 GiB slot at 8k × batch 4: weights 7.49 GiB + KV `4·8,192·147,456 B` = 4.83 GB = 4.50 GiB = 12.0 GiB, plus the 1.1 GiB snapshot of one 8k prefill (`8,192·147,456 B` = 1.21 GB); prefill must compute logits at the last position only (full-position bf16 logits for 8k tokens are 2.3 GiB); if the measured peak exceeds 15 GiB, run the 8k cells at batch 2.
  - Haystack text: R3 chunks (`d2l.book_corpus`), tokenized with R1's tokenizer; lengths are token counts (target ± 2%).
  - QA: `rajpurkar/squad_v2` (`validation`, answerable questions only; CC BY-SA 4.0) and `hotpotqa/hotpot_qa` (`distractor`, `validation`; CC BY-SA 4.0): instances are **regenerated from ids and seeds at build time, never committed as text** (share-alike content); `d2l.book_qa` for the book questions.
  - Synthetic tasks follow RULER's templates (single needle, multi-key needle, variable tracking, common-words extraction), re-implemented from the paper and the RULER repository's task descriptions with R3 as haystack (the original uses essays); no RULER data is copied.
- **Packages:** as 12.1; `pandas`; optional `kvpress` (NVIDIA; the production oracle for eviction), optional `transformers` quantized-cache backends, optional `triton` for the packed dequantization exercise.

#### Narrative arc

1. **The question.** Opening cell: R1's KV bytes against its weights as context grows: `d2l.decode_ledger` at `L = 1k … 32k` shows the cache overtaking the 1.19 GB of weights at `L ≈ 10.4k` tokens (112 KiB/token) and, with int4 weights from 12.1, at `L ≈ 4.8k` (545 MB). Second printed symptom: the *predicted* bandwidth-bound speed-up of an ideal 5× smaller cache at 8k context: 1.54× with bf16 weights and 2.02× with int4 blocks (arithmetic from the ledger, no measurement). That is the prize and the question: what does the compression cost in accuracy?
2. **Two levers, applied after training.** Table `tab_lm_kvc_levers`: shrink *bits per element* (quantize) vs shrink *entries* (evict); what each saves in stored bytes and in bytes *read* per step (eviction saves both; quantization saves both only with a fused kernel); what each needs (nothing from the model; only a policy and a cache backend).
3. **Where the keys' error comes from (D1).** Look at real R1 keys before pre-RoPE rotation: per-channel ranges across a 32-token group (`fig_lm_kvc_channels`: heat map of channel range, layer 3 and layer 14, showing a few channels with outsized magnitude); then the error formula: per-token vs per-channel quantization of keys, predicted vs measured score error.
4. **Per-channel keys, per-token values: the KIVI cache.** `QuantKVCache` in ~70 lines: groups along tokens for keys (channel-wise scale per 32-token chunk), groups along channels for values (per token), the full-precision residual window, flush rule, packed codes; byte accounting that includes scales, zeros, residual buffer and padding (`eq_lm_kvc_bytes`); the 2×2 grid {key scheme} × {value scheme} at 2 and 4 bits on attention-output error for 200 sampled (layer, head) slices; `tab_lm_kvc_error`.
5. **Quantizing before RoPE (D2).** Why post-rotation keys lose the per-channel structure; the range formula and its knee at `G·θ_i ≈ π` (pair index ≈ 11 of 64 for `G = 32`, base 10⁶); measured range ratio post/pre per pair (`fig_lm_kvc_rope_range`); the price: apply RoPE to the whole dequantized key block each step (the cache backend sets `stores_pre_rope = True`; the generator rotates after `view`); measure the added bytes/FLOPs.
6. **Reading less: evicting entries (D3).** The error bound `‖o − ô‖ ≤ 2·m·max‖v‖`; why keeping the largest-mass entries minimizes `m`; the persistence hypothesis behind H2O (accumulated attention predicts future attention) and the observation-window hypothesis behind SnapKV (the last queries preview what the answer needs); `SinkWindowPolicy` as the baseline; implementation notes: per-KV-head selection from group-aggregated scores (`g = 2` query heads per KV head for R1), selected columns kept in original order with original RoPE positions, `kv_len` vs `pos`.
7. **Measuring capture before measuring accuracy.** Offline on 50 prompts: for each policy and budget, the fraction of the *future* queries' attention mass that the kept set captures per layer (D3) against an oracle that knows the future (`fig_lm_kvc_mass`); and against a *different* question over the same context (the quantity that decides multi-question failure).
8. **Evaluation beyond a needle.** Task list and what each stresses: `s_niah` (one fact), `mk_niah` (distractor facts), `vt` (multi-hop), `cwe` (aggregation over the whole context, which eviction should hurt most), `squad_qa`/`hotpot_qa` (natural questions, one and two hops), `book_qa` (domain text), `multi_q` (two questions over one compressed context). Pre-registered rules: tasks whose uncompressed accuracy is below 0.3 are excluded from the comparison and listed; the order of questions (question last, as in RULER) is fixed except in `multi_q`.
9. **The sweep (headline).** Bit width × scheme and budget × policy over the tasks; the frontier plot; the "needle vs everything else" ranking flip.
10. **What the numbers license.** A closing table of rules of thumb with the evidence cell for each, and the pointer to what needs training (14.3–14.5) and to paging and offload (13.3, 5.3).

#### Derivations

**D1. Why keys want per-channel scales (`eq_lm_kvc_keyerr`).**
- Statement: let `k ∈ R^D` be a key quantized with `b` bits using uniform min–max steps. For a query `q`, the score error `e = q·(k̂ − k)` has `Var(e) = Σ_d q_d² Δ_d²/12`, where `Δ_d` is the step for channel `d`. With one range per token (all `D` channels share a step set by the largest channel range `r_max` within the token) `Δ_d = r_max/(2^b − 1)`; with one range per channel over a group of `G` tokens `Δ_d = r_d/(2^b − 1)`, `r_d` being the channel's own range over the group. Hence `Var_token/Var_channel = r_max² Σ_d q_d² / Σ_d q_d² r_d² ≥ 1`, with equality iff all channel ranges are equal.
- Assumptions: high-resolution quantization (error uniform on `±Δ/2`, independent across channels), a query independent of the rounding error; the formula is an approximation at 2 bits and the check shows by how much.
- Steps: (1) Uniform error of variance `Δ²/12` per coordinate (the 7.6 result). (2) Linearity: `e = Σ_d q_d ε_d`, `Var(e) = Σ q_d² Δ_d²/12` by independence. (3) Substitute the two step rules. (4) `r_d ≤ r_max` gives the inequality. (5) Example: `D = 128`, four channels with range 20× the others (`R = 20`, `r = 1`), isotropic `q`: ratio `= R²·D/(m R² + (D − m) r²) = 51,200/1,724 ≈ 29.7`. (6) Why keys and not values: keys have a few channels with consistent large magnitude (cheap for min–max per channel, because a *consistent offset has a small range over tokens*, and expensive for per-token because the token's range spans the outlier); values have no such structure, and their error enters through attention weights `a_t²` per token, which per-token scaling confines to the token itself.
- Check cell: (a) synthetic keys with the outlier structure above: predicted variance ratio vs measured (10⁵ samples; within 15% at 4 bits; the 2-bit departure is printed); (b) R1 keys from layers 3, 14, 25 on 20 prompts: ratio predicted from the measured channel ranges vs measured ratio of mean-squared score errors at 2, 3, 4 bits; (c) values: the same four-scheme grid, reported as measurement (no formula claim).

**D2. Pre-RoPE quantization (`eq_lm_kvc_rope`).**
- Statement: RoPE rotates each channel pair `(a, b)` by angle `φ_t = t·θ_i`, `θ_i = ϑ^{−2i/D}` (`ϑ = 10⁶` for Qwen3; Qwen3 pairs channel `i` with channel `i + D/2`). For a pair whose pre-rotation vector over a group of `G` tokens is `m + η_t` with mean `m` and spread `‖η‖ ≤ s`, the per-channel range after rotation satisfies `r_post ≈ ‖m‖·min(2, G·θ_i) + O(s)` whereas `r_pre = O(s)`. Hence quantizing pre-RoPE keys removes the penalty `‖m‖·min(2, Gθ_i)` per channel, which is non-negligible exactly for the pairs with `G·θ_i ≳ ε` and saturates at `2‖m‖` for `G·θ_i ≥ π`.
- Assumptions: tokens in a group are consecutive; `m` roughly constant over the group (the consistent-offset structure of D1); small spread.
- Steps: (1) Post-rotation channel `a'_t = ‖(m + η_t)‖ cos(φ_t + ψ_t)` where `ψ_t` is the phase of the pre-rotation vector. (2) For `‖η‖ ≪ ‖m‖`, `a'_t ≈ ‖m‖ cos(φ_0 + tθ_i + ψ) + O(s)`. (3) The range over `t ∈ [0, G)` of `cos` over an angular span `Gθ_i` is at most `min(2, Gθ_i)`. (4) Pre-rotation range is the spread, `O(s)`. (5) Knee: `G·θ_i = π` gives `i* = (D/2)·ln(G/π)/ln ϑ`; for `G = 32`, `ϑ = 10⁶`, `D = 128` this is `i* ≈ 10.8`, i.e. `θ_i < π/32 ≈ 0.098` for `i ≥ 11` (of 64 pairs). The first ≈ 11 high-frequency pairs rotate through a full half-turn inside a group and lose most; the remaining 53 rotate slowly within a group and gain little. (6) Cost: pre-RoPE storage requires rotating all cached keys at each decode step, `L·D/2` sincos multiply-adds per KV head per layer, plus the materialization of the rotated block unless fused (KVQuant fuses it).
- Check cell: measure `r_post/r_pre` per pair index on R1 keys (layers 3, 14, 25; groups of 32) and overlay the predicted knee; assert the measured ratio is ≥ 1 for ≥ 90% of pairs and ≈ 1 for `i > 40` (tolerance 10%); then measure the score error of 2-bit KIVI-style keys with and without pre-RoPE.

**D3. Eviction error and attention-mass capture (`eq_lm_kvc_evict`).**
- Statement: for one query with attention weights `a_t` over cached entries, evict a set `E` with dropped mass `m = Σ_{t ∈ E} a_t` and renormalize the softmax over the kept set `K`. The attention output changes by `‖o − ô‖ ≤ m · max_{t ∈ E} ‖v_t − ô‖ ≤ 2·m·max_t ‖v_t‖`. Keeping the `B` entries of largest `a_t` minimizes `m`. Eviction by an *estimate* of future `a_t` (accumulated past attention, or an observation window) is optimal only to the extent that the estimate ranks future mass correctly.
- Assumptions: single head, a single query (multi-query: average of bounds); exact arithmetic.
- Steps: (1) `o = Σ_t a_t v_t`, `ô = Σ_{t ∈ K} a_t v_t/(1 − m)`. (2) `o = (1 − m)·ô + Σ_{t ∈ E} a_t v_t`, so `o − ô = Σ_{t ∈ E} a_t (v_t − ô)`. (3) Triangle inequality gives `‖o − ô‖ ≤ m·max_E ‖v_t − ô‖`. (4) `ô` lies in the convex hull of `{v_t}`, so `‖v_t − ô‖ ≤ 2 max‖v‖`. (5) For a fixed budget `B`, `m = 1 − Σ_{t ∈ K} a_t` is minimized by the top-`B` weights. (6) Define the *capture* `c = 1 − m` and measure it for each policy.
- Check cell: (a) random `a, v` (seed 0): the bound holds in `10⁴` trials and the top-`B` set attains the smallest `m` among 500 random subsets; (b) R1 attention from 50 prompts: capture curves for oracle-future, H2O, SnapKV, window-with-sinks, random, versus budget (`fig_lm_kvc_mass`); (c) *different-question capture*: the same prompts with a second question appended; capture of the second question's mass by sets chosen with the first question's window (the quantity that predicts the multi-question failure).

**D4. Byte accounting and the bandwidth-bound prediction (`eq_lm_kvc_bytes`, `eq_lm_kvc_speedup`).**
- Statement: with `b` bits, group size `G`, bf16 scale and zero per group, a quantized element costs `b + 32/G` bits; the KIVI cache of `n` tokens with a bf16 residual window of `R` tokens stores `(n − R)·k_KV·(b + 32/G)/16 + R·k_KV` bytes (`k_KV` = bf16 bytes per token). Eviction to budget `B` stores `B·k_KV` plus per-head score accumulators. In the bandwidth-bound regime the predicted speed-up of decode is `(W + b·L·k_KV)/(W + b·L·k_KV·ρ)` with `ρ` the stored/bf16 ratio.
- Steps: (1) Per-group metadata: two bf16 numbers per group. (2) Residual window is bf16: constant cost per sequence. (3) Substitute into 12.1's ledger. (4) R1 at `L = 8,192`, bf16 KV 939.5 MB: KIVI 2-bit (G = 32, R = 128) 188.1 MB (4.99×), KIVI 4-bit 303.7 MB (3.09×); predicted decode speed-ups at b = 1 (bf16 weights / int4 blocks / int4 everything): 1.54 / 2.02 / 2.49 for 2-bit, 1.43 / 1.75 / 2.02 for 4-bit, and 1.63 / 2.89 for 8× eviction with bf16 weights / int4-everything weights (the 2.89 is `(316.7 + 939.5)/(316.7 + 117.4)`).
- Check cell: build a `QuantKVCache` for 8,192 tokens and compare `torch.cuda.memory_allocated` growth with the formula (within 2% after accounting for the 4-bit lanes used for 3-bit); print the table above from `decode_ledger`; for **eviction only**, measure real decode ms/step with length buckets against the prediction (eviction reduces columns read, so a measured speed-up is legitimate); for quantization, *no speed is claimed* (the dequantizing `view` is not a fused kernel), only the prediction.

#### From-scratch implementation

Budget ≤ ~150 lines: the quantizer and cache (~70), the three policies (~55), the capture and sweep helpers (~25). The task generators (~110 lines) are data code and are listed separately.

| Component | Contract | Lines |
|---|---|---:|
| `quantize_groupwise` / `dequantize_groupwise` / `pack_bits` | min–max, groups along `dim`; round-half-even; `codes ∈ [0, 2^b − 1]`; scale/zero bf16 | ~25 |
| `QuantKVCache` | `append` (writes to the residual buffer; flushes the oldest group of `G` when the residual exceeds `R`), `view(layer, upto)` (dequantize flushed groups, concatenate residual), `nbytes` | ~45 |
| `H2OPolicy` | accumulates `observe` probabilities (summed over the `g` query heads of a KV head) into a score buffer; `compact` every `slack` steps keeps `recent` newest + top scores up to `budget` | ~25 |
| `SnapKVPolicy` | at prefill: window probs → sum over window and group → 1-D max-pool (`pool`) → top `budget − window` per KV head, plus the window | ~20 |
| `SinkWindowPolicy` | keep first `sinks` and last `window` | ~10 |
| `attention_capture`, `kv_sweep` | as in Exports | ~25 |

Numerical tests: (1) `QuantKVCache` with `bits=16` behaves as the static cache (logits equal within test-1 tolerance of 12.1); (2) round-trip: dequantized values within `Δ/2` of the originals for all groups (exact bound check), for 2, 4, 8 bits; (3) packing: `unpack_bits(pack_bits(c)) == c`; (4) residual rule: after appending `n` tokens one at a time, the number of flushed tokens is `G·⌊(n − R)/G⌋` (bounded below by 0) and the cache `view` has length `n`; (5) pre-RoPE equivalence: with `bits=16` and `pre_rope=True` the logits equal those of `pre_rope=False` (the rotation moved, nothing else); (6) eviction identity: a policy with `budget ≥ L` is a no-op (logits equal); (7) position test: after eviction, the decoded token's RoPE position is `pos`, not `kv_len` (a deliberately wrong version must fail on S-NIAH at 4k); (8) SnapKV selection with `pool = 1, window = L` equals top-`B` of column sums (hand-checkable on a 6-token example); (9) `kv_sweep` determinism: two runs with the same seed give identical scores.

#### In production

- **kvpress** (NVIDIA): run its `SnapKVPress` and `StreamingLLMPress` on the same prompts with matching settings (window 32, pool 7 for SnapKV) and report the Jaccard overlap of kept indices per layer against `SnapKVPolicy` (design rule: median ≥ 0.9; if lower, document the difference, which is likely the pooling or the per-head aggregation).
- **KIVI** reference code (CUDA kernels for packed 2-bit): not run at T1 if it pins an old Transformers; read its cache layout and compare with `QuantKVCache` (tile and residual conventions); the KIVI paper's reported memory and throughput gains are the claims a fused kernel substantiates, which a fake-quant simulation cannot.
- **Transformers** `QuantizedCache` (quanto/HQQ backends; per-token by default) as the naive-per-token reference: its attention-output error should match our `k_scheme='token'` arm.
- **vLLM** `--kv-cache-dtype fp8` and **llama.cpp** quantized KV flags (`--cache-type-k`, `--cache-type-v`; check names in the installed version) as the production forms of "value-agnostic KV quantization"; on A100 FP8 KV has no tensor-core path, so FP8 KV runs on the Ada slot only.
- **FlashInfer / flash-attn** support paged and quantized KV in kernels (read-only pointer).

#### Headline experiment

- **Hypothesis.** (H1) At 2 bits, per-channel keys + per-token values (KIVI) lose much less accuracy than per-token for both, and pre-RoPE keys help further at 2 bits; (H2) eviction methods that choose entries with the question in view (SnapKV, H2O) match full-cache accuracy on single-question retrieval at small budgets but collapse on `multi_q` and on aggregation (`cwe`), whereas quantization degrades smoothly; (H3) ranking the methods by `s_niah` alone gives a different order from ranking by the mean over all tasks.
- **Setup.** Tier T1. R1 bf16; contexts 4,096 tokens (reduced grid) with the 8,192 and 2,048 results from the committed extended grid; 30 instances per task and length (reduced) / 100 (extended); tasks `s_niah` (depths 0.1…0.9 cycled), `mk_niah`, `vt`, `cwe`, `squad_qa`, `hotpot_qa`, `multi_q` (the first question's answer is generated, then a second, different question over the same context is appended with chunked prefill and answered); greedy decoding, `max_new_tokens = 32` (CWE 64); seeds fix instance construction and nothing else is random. Configs (reduced): full bf16; 8-bit (naive per-token, G = 128); for each of 4 bits and 2 bits: naive per-token/per-token, KIVI (channel keys, token values), KIVI + pre-RoPE keys; eviction budgets `1/2, 1/4, 1/8, 1/16` of the context for sink+window, H2O, SnapKV, random (control); 4 sinks and window 32 for SnapKV/H2O recent. The extended grid adds 3-bit, budgets `1/32`, G ∈ {32, 64, 128}, sink-in-bf16, and contexts 2k and 8k.
- **Procedure.** `d2l.kv_sweep` prefill-once-branch-many; scores per instance; relative accuracy `acc(config)/acc(full)` per task with a paired cluster bootstrap over instances (`d2l.paired_bootstrap`, 10,000 resamples); KV bytes from `nbytes` (including scales, zeros, residual, score buffers); for eviction configs, measured decode ms/step at 8k with length buckets; for quantization, predicted speed only. Cost sizing: the reduced grid needs `n_instances × n_tasks` prefills and `× n_configs` short decodes; the notebook prints the count and the measured seconds per prefill and per branch decode at start and aborts with a message if the projected time exceeds the 15-minute budget (then falls back to 20 instances).
- **Metrics.** Task score (string-match recall or contain-match), relative accuracy, KV bytes fraction, capture (D3), attention-output relative error (offline), decode ms/step (eviction only).
- **Plots.** `fig_lm_kvc_frontier`: three panels (retrieval: `s_niah`, `mk_niah`, `vt`; aggregation and QA: `cwe`, `squad_qa`, `hotpot_qa`; `multi_q`), x = KV bytes fraction (log), y = relative accuracy, one line per method with bootstrap bands, horizontal line at 1; `fig_lm_kvc_rank`: method rank by `s_niah` alone vs by the mean over tasks (slope chart); `fig_lm_kvc_channels`, `fig_lm_kvc_rope_range`, `fig_lm_kvc_mass` as above.
- **Expected outcome.** Directions: KIVI ≥ naive at 2 bits by a clear margin; pre-RoPE ≥ post-RoPE at 2 bits; `s_niah` nearly unaffected by H2O/SnapKV down to small budgets while `multi_q` and `cwe` degrade; the rank flip of H3. Published context for calibration (re-read the abstracts at authoring time before quoting): the KIVI paper reports about 2.6× lower peak memory (weights included) at 2 bits and the possibility of larger batches and higher throughput with its fused kernels; RULER reports that models with near-perfect needle retrieval still degrade on its harder tasks as length grows; the KV-compression benchmark (2407.01527) and SCBench (2412.10319) report that methods tuned for single-question evaluation lose accuracy under multi-query or multi-turn use of the same context. Magnitudes for R1 are to be measured, and a 0.6B model may sit near the floor on `vt`/`cwe`/`hotpot_qa`, in which case those tasks are reported but excluded from the comparison by the pre-registered rule.
- **Noise check.** Every headline claim is a *paired* relative-accuracy difference with a 95% cluster-bootstrap interval; a difference is not stated unless the interval excludes 0; instance counts are printed; the extended grid (100 instances) is used for any claim the reduced grid cannot resolve; one repetition with a different instance seed reproduces the ordering of the main methods (checked on the extended grid).
- **T1 reduction.** The notebook *is* T1; the extended grid is precomputed on any host (or an A100 for speed) and committed; the live run covers the reduced grid and re-verifies two rows of the committed table (the full-cache row and one KIVI row) within bootstrap error as a consistency check.
- **Artifacts committed.** `lm-inference-kvc-sweep.parquet` (extended grid, all columns above), `lm-inference-kvc-instances.json` (ids and seeds only), `lm-inference-kvc-capture.npz` (capture curves), `lm-inference-kvc-ranges.npz` (channel ranges per layer).

#### Figures and tables

- `fig_lm_kvc_channels` (channel-range heat map), `fig_lm_kvc_rope_range` (range ratio post/pre vs pair index with the predicted knee), `fig_lm_kvc_mass` (capture vs budget, three panels: same question, different question, per-layer), `fig_lm_kvc_frontier`, `fig_lm_kvc_rank`.
- `tab_lm_kvc_levers` (quantize vs evict), `tab_lm_kvc_error` (2×2 key/value scheme grid at 2 and 4 bits: relative attention-output error), `tab_lm_kvc_bytes` (D4: bytes and predicted speed-ups), `tab_lm_kvc_tasks` (what each task stresses, baseline accuracy, included or excluded), `tab_lm_kvc_rules` (the closing rules of thumb with their evidence cells).

#### Pitfalls and misconceptions

- Needle-in-a-haystack is a retrieval probe with one fact and one question; it rewards methods that keep whatever the question asks for and says nothing about aggregation, multi-hop reasoning, or a second question.
- Reporting bits per element without scales, zero points and the full-precision residual window; at 2 bits and `G = 32` the metadata adds 1 bit per element (50% overhead).
- Fake quantization measured for speed: the simulated dequantizing `view` is slower than bf16; only a fused kernel can realize the bandwidth saving, and the notebook labels quantization speed as a prediction.
- Evicting entries and renumbering RoPE positions: keys keep their original rotations; the new token's position is `pos`, not `kv_len`.
- Selecting per query head under GQA: the cache is per KV head, so scores must be aggregated over the `g` query heads that share it.
- Question-aware compression (SnapKV's window) applied before a *different* question arrives; the `multi_q` task exists to show this.
- Putting the question before the context for eviction methods (it then sits outside the observation window or in the evicted region); the task fixes the RULER order and varies it only in one labelled ablation.
- Judging 2-bit quantization on perplexity alone, which hides retrieval failures, and judging retrieval alone, which hides drift in generation.
- Comparing a method at its best budget with another at a different budget: always compare at equal KV bytes (the frontier plot's x axis).
- Confusing MLA or a GQA conversion (training-time changes, 14.3) with these post-hoc methods.

#### Summary (draft bullets)

- Once weights are quantized, the KV cache is the next cost in decode: at 8k tokens R1's cache (0.94 GB) is larger than its int4 weights (0.55 GB).
- Keys have a few consistently large channels, so per-channel scales (over a group of tokens) quantize them far more accurately than per-token scales; values are quantized per token. Quantizing before RoPE keeps the per-channel structure that rotation destroys, at the cost of rotating the cache at read time.
- Eviction changes the attention output by at most twice the dropped mass times the largest value norm, so a good policy is one whose retained set captures the future attention mass; policies built from past or windowed attention estimate that mass, and fail when the next question differs.
- Honest byte accounting includes scales, zeros, the residual window and score buffers; honest speed claims need fused kernels.
- Evaluation must include aggregation, multi-hop and a second question over the same context, with paired intervals; a needle-only ranking of methods can be wrong.

#### Exercises

1. **(check) Bytes.** R1, batch 4, 8,192 tokens each. Compute the KV bytes in bf16 and for KIVI 2-bit (`G = 32`, `R = 128`), including metadata and residual. *Sketch:* bf16: `4 · 8,192 · 114,688 B = 3.76 GB`; KIVI-2: per sequence `8,064 · 114,688 · 3/16 + 128 · 114,688 = 173.4 + 14.7 = 188.1 MB`, so `752 MB` for four sequences, a 5.0× reduction (effective 3 bits, not 2).
2. **(check) GQA selection.** R1 has 16 query heads and 8 KV heads. Why must H2O/SnapKV scores be aggregated over the two query heads of a group, and what breaks if each query head selects its own `B` entries? *Sketch:* the cache stores one K/V per KV head; two query heads sharing it would need different subsets, so the union is stored (up to `2B` entries) or one head reads entries chosen for another; aggregation (sum or max) gives one set per KV head.
3. **(derive) Tightness of the eviction bound.** Show an instance where `‖o − ô‖` equals `m · max_E ‖v_t − ô‖`, and one where the bound `2·m·max‖v‖` is attained. *Sketch:* take `E` a single entry with `v_E` far from the rest and equal norms; `o − ô = a_E (v_E − ô)` exactly; with `ô = −v_E`-direction vectors of equal norm the factor 2 is attained.
4. **(derive) Bandwidth-bound attention under quantized KV.** For GQA with `g` query heads per KV head, decode attention performs about `2g` FLOPs per K element and `2g` per V element. Show the arithmetic intensity at 16 bits is `g` FLOP/B, find it at 3 effective bits with `δ = 2` FLOPs of dequantization per element, and conclude whether CUDA-core compute becomes the bottleneck on the 4090 (ridge ≈ 82 FLOP/B at fp32 CUDA-core rate). *Sketch:* `(2g + δ)/(3/8 B) = 6/0.375 = 16 FLOP/B` for `g = 2`, far below the ridge: still bandwidth-bound; the limit is kernel quality, not arithmetic.
5. **(measure) Sinks in full precision.** Keep the first 4 tokens in bf16 in the 2-bit cache (`sink_fp = 4`); predict the effect on `s_niah` and `multi_q` before running; report the paired difference with its interval.
6. **(measure) SnapKV window and pooling.** Sweep window ∈ {8, 32, 128} and pool ∈ {1, 7, 15} at budget 1/8; predict which setting helps `multi_q` and why (a larger window sees more of the question's keywords; pooling keeps neighbours of important tokens); report `fig`-ready tables with intervals.
7. **(open) Break your method.** Design a task on which a query-aware eviction method with a large budget still fails while a 4-bit quantized cache succeeds (or the reverse), implement it with `ruler_task`'s interface, and show the frontier. Alternatively, compose quantization and eviction (SnapKV 1/4 + KIVI-4) and test whether the accuracy losses add.

#### Further reading

- Papers (v5): [KIVI](https://arxiv.org/abs/2402.02750) (per-channel keys, per-token values, residual window); [KVQuant](https://arxiv.org/abs/2401.18079) (pre-RoPE keys, non-uniform datatypes, dense-and-sparse outliers, sink handling); [H2O](https://arxiv.org/abs/2306.14048) (accumulated-attention eviction, the persistence-of-importance observation); [SnapKV](https://arxiv.org/abs/2404.14469) (observation-window selection with pooling). New and verified: [RULER](https://arxiv.org/abs/2404.06654) (task taxonomy beyond needles); [StreamingLLM](https://arxiv.org/abs/2309.17453) (sinks and rolling windows; already the book's baseline); [KV Cache Compression, But What Must We Give in Return?](https://arxiv.org/abs/2407.01527) (a benchmark of long-context-capable approaches); [SCBench](https://arxiv.org/abs/2412.10319) (KV-cache-centric evaluation across shared-context requests); [TurboQuant](https://arxiv.org/abs/2504.19874) (advanced: rotation-based online vector quantization; its claims are conditional on its stated assumptions, per the Codex plan).
- Tutorials and benchmarks: the [RULER repository](https://github.com/NVIDIA/RULER) (task generators and configs); the [EMNLP 2025 tutorial](https://aclanthology.org/2025.emnlp-tutorials.1/) (v5 teaching source) for the taxonomy of KV methods.
- Code: [kvpress](https://github.com/NVIDIA/kvpress) (new, verified; a library of KV-cache "presses"); [KVQuant code](https://github.com/SqueezeAILab/KVQuant) (new, verified); [KIVI code](https://github.com/jy-yuan/KIVI), [H2O code](https://github.com/FMInference/H2O) and [SnapKV code](https://github.com/FasterDecoding/SnapKV) (all returned HTTP 200 on 28 Oct 2026).

#### Risks and open decisions

- **R1 may sit at the accuracy floor** on `vt`, `cwe`, `hotpot_qa` at 4k. The pre-registered rule excludes tasks with baseline accuracy < 0.3; the fallback `MODEL='Qwen/Qwen3-4B'` fits the slot at 8k × batch 4 (12.0 GiB before the snapshot, arithmetic in Models and data) and keeps the code unchanged; the editor should decide whether the committed extended grid should use 4B for the headline figure. Its prefill is roughly 6.7× R1's parameters (4.02 B against 0.60 B), so the notebook's printed cost projection will likely fall back to 20 instances (to be measured); every 4B-specific number in the narrative (the cache-overtakes-weights length, the predicted speed-ups, exercise 1's bytes) is recomputed from `d2l.decode_ledger` for the active `MODEL`, never hard-coded.
- **Time budget.** Prefill with H2O's column sums (explicit attention in query blocks) is much slower than flash prefill; the sweep prefills each instance once and branches configs, and the notebook prints a projected cost; fall back to 20 instances or two lengths if over budget.
- **`QuantKVCache` speed** is deliberately not part of any claim. If a packed kernel (Triton exercise or KIVI's CUDA) is available, add a measured decode row and label it separately.
- **Licences.** SQuAD v2 and HotpotQA are CC BY-SA 4.0: text is regenerated at build time from ids and seeds, never committed. AngelSlim-style licence caution does not apply here (no third-party weights).
- **EOS and instruction format.** R1 is the post-trained hybrid-thinking model; use the chat template with thinking disabled, and a plain-text completion format as fallback, because RULER's templates are designed for base-model-style prompting; the notebook states which was used and checks that the full-cache baseline is not hurt by the template.
- **Interaction with 12.2.** Using `SinkWindowPolicy` for the long-context draft of 12.2 is left as that section's exercise; if the editor prefers it as a measured arm in 12.2, 12.4's policy must be exported earlier (a dependency inversion to decide).
