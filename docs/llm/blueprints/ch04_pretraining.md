# Chapter 4: Pretraining — Blueprint

**Label:** `chap_lm_pretraining` · **Arc:** II. Building a base model (3 Data, 4 Pretraining) · **Sections:** 4.1–4.7 (4.7 optional ◇) · **Tiers:** T0–T2 live; the R2 production run is a one-off T3 job whose artifacts T1/T2 notebooks analyze.

## Chapter overview

### Draft introduction (about 170 words)

Chapter 3 ended with a corpus manifest. This chapter turns it into a base model and, more importantly, into evidence about how a base model is made. A pretraining run is a long computation that can fail in ways a loss curve hides: a data path that silently repeats or leaks tokens across documents, an attention logit that grows for ten thousand steps before it blows up, a precision format that is fine at step 1 and wrong at step 100,000. We build the data path first (4.1), so that every batch is a pure function of the step and a run can be interrupted without a trace. We then train the book's reference model R2 (4.2), read its logs, and compare its trajectory with a fully open one. Sections 4.3–4.6 take one pathology or extension each: instabilities, FP8 and simulated FP4, continued pretraining, and context extension. Section 4.7 closes with a budget question that the earlier chapters of the book leave open: what to train when inference is part of the cost.

### Learning objectives (measurable)

After the chapter the reader can:

1. Build a packed-sequence stream whose batch at step *s* is a pure function of `(shards, seed, s)`; show that (a) the next 50 batches after an interrupt are bit-identical, and (b) per-document logits under packing with document masks match unpacked logits to ≤ 1e-5 (fp32, CPU).
2. Train a ~150M-parameter decoder with WSD, AdamW and z-loss; compute its MFU from first principles to within 5% of a profiler-based FLOP count; and name from the logs which of loss, gradient norm, update/weight RMS or a downstream metric moved first at each phase.
3. Reproduce attention-logit growth and a loss spike in a small model, apply QK-norm, z-loss, softcapping and QK-clip one at a time, and report each variant's stable learning-rate range with a stated criterion and a seed-noise check.
4. Run FP8 training on an Ada GPU, measure loss gap, throughput and overflow statistics against BF16, simulate block-scaled FP4 and predict its error from block size and in-block dynamic range.
5. Continue pretraining a released base model on a domain corpus, with and without replay, and plot the forgetting-versus-domain-gain frontier.
6. Extend a 4K model to 16K with PI, NTK-aware scaling and YaRN, and report effective context by task, not by perplexity alone.
7. (◇) Compute the compute-optimal model size when lifetime inference tokens enter the objective, and state where the answer is sensitive to the assumed demand.

### Prerequisites

Book labels (what is used, what must not be re-taught):

| Label | Used | Not re-taught |
|---|---|---|
| `sec_gpt` | `d2l.GPT` (tied head, `blks`, `attn_factory` seam via `blk.attention = ...`), `d2l.GPT.generate`, the BPE pipeline and the GPT-2 loading (the R2 tokenizer itself is 3.6's) | GPT assembly, sampling |
| `sec_transformer-block` | `d2l.TransformerBlock`, `d2l.FeedForward` (SwiGLU width `round(8d/3)`), QK-norm (`subsec_qk-norm`), the signal-propagation view | normalization placement, RMSNorm |
| `sec_kv-cache` | `d2l.GQAAttention` (`W_q, W_k, W_v, W_o`, `rope=True`, `enable_gqa`), GQA ratios | GQA derivation, cache sizing |
| `sec_adamw` | decay/no-decay parameter groups (`subsec_adamw-what-not-to-decay`), optimizer-state bytes | AdamW derivation, decoupling |
| `sec_scheduler` | WSD, branching off the plateau (`subsec_scheduler-branching`), linear-to-zero frontier | schedule derivations; note `WSDScheduler` there is not `#@save`d, so 4.2 defines `d2l.wsd_lr` |
| `sec_muon` | `torch.optim.Muon` and the `MultiOptimizer` split, RMS matching, and the MuonClip subsection with `d2l.qk_clip_` (4.3 imports it) | Newton–Schulz, norm-based derivation, the QK-clip rescaling and its guarantee |
| `sec_batch_size` | critical batch size, batch ramps | gradient-noise scale |
| `sec_scaling` | muP coordinate check, transfer | muP rules |
| `sec_practice` | clipping, EMA, the "Additional Stability Methods" paragraph (z-loss, QK-norm, QK-clip, PaLM rewind-and-skip) | 4.3 *implements and measures* what that paragraph only names |
| `sec_scaling-laws` | `eq_six_nd`, parameter census, Chinchilla law `eq_chinchilla_law`/`eq_chinchilla_optimum`, the two published fits in its exercises (Hoffmann; Besiroglu replication) | the fit, the 6ND derivation |
| `sec_memory_precision` | autocast, GradScaler, tf32 caveat, `d2l.Benchmark`, gradient accumulation (`subsec_mp-accumulation`) | mixed precision, accumulation equivalence for equal-size microbatches |
| `sec_numerics`, `sec_hardware` | float format tables, `tab_gpu_specs` (RTX 4090 and H100 columns), fp8 E4M3/E5M2 `finfo` | format layouts, loss scaling |
| `sec_perf_model` | `d2l.Benchmark`, roofline, profiler FLOP counting | roofline derivation. Note: MFU is only an exercise in `sec_fast_transformer`; 4.2 defines it. |
| `sec_init_param` | residual-branch scaling, zero-init, variance across depth | initialization theory |
| `sec_positional-information` | RoPE (`eq_rope-def`), `subsec_pos-context-extension` (names PI and YaRN, no derivation), ALiBi/NoPE | RoPE derivation. 4.6 derives the per-frequency scale factors the book only names. |
| `sec_read_write`, `sec_repro` | safetensors, run checkpoints, seeds, determinism flags | seeds, determinism |

Earlier sections of this part:

- **1.1–1.2:** `d2l.load_hf(model_id, revision=...)` (OLMo 2 revisions in 4.2; Qwen3-0.6B-Base in 4.5), `d2l.generate` (returns `list[d2l.Generation]`; use `[g.text for g in ...]` wherever strings are needed).
- **1.4–1.5:** `d2l.score_choices`, `d2l.EvalSuite`, `d2l.paired_bootstrap`. Every downstream number in this chapter goes through them.
- **2.4:** `d2l.book_corpus()` (R3) as the held-out technical domain in 4.2 and the domain corpus in 4.5.
- **3.x (soft):** the tokenizer and data-mixture sections inform choices. **3.6 (interface adopted from the editor's note):** the R2 tokenizer (byte-level BPE, vocabulary 32,768, Qwen3-style single-digit split regex, EOS id 0, no BOS, 32 special-token ids 0–31: four named, 28 reserved), exported as `d2l.r2_tokenizer()`; `d2l.write_shards` and `d2l.TokenShards` (shard format below); `d2l.bits_per_byte` (1.4). DATA_HUB keys from Chapter 3: `lm-r2-tokenizer`, `lm-fwedu-val`, `lm-fwedu-shards-100m`, `lm-fwedu-manifest-4b`.
- **6.1 (assumed):** `d2l.launch(script, nodes=1, gpus_per_node=8, **env)` to run the R2 job at T3.

### Section dependency graph

```
4.1 data path ──► 4.2 R2 training ──┬─► 4.3 instabilities (uses 4.1 resume+skip, 4.2 pretrain loop)
   │  (PackedStream,                ├─► 4.4 low precision (uses 4.2 loop, T1 preset)
   │   PackedGPT)                   ├─► 4.5 continued pretraining (uses 4.1 data, 4.2 evals/ckpts)
   │                                ├─► 4.6 long context (uses 4.1 PackedAttention rope hook, 4.2 T1/R2 ckpts)
   └──────────────────────────────► └─► 4.7 budgets ◇ (uses 4.2 measured FLOPs/MFU, sec_scaling-laws)
```

Courses: "Large Language Models" takes 4.1–4.3 and 4.6; "Systems for Large Models" takes 4.1 and 4.4; "Model Architecture" takes 4.2–4.3 (v5).

### Running examples and shared artifacts

**Produced here.** R2 (the reference model) and its logs; the T1 preset checkpoint `r2t1`; the library below. Tokenizer and token shards come from Chapter 3.

**Consumed later (v5).** Chapter 6 shards R2 under FSDP/TP/PP; Chapter 7 fine-tunes it and Qwen3-0.6B; Chapter 11 quantizes it; Chapter 12 serves it with `d2l.FastGenerator`; 8.x and 9.x use R2 only as a cheap policy for pipeline tests.

**Library (`#@save`)** — reserved names (owned here): `d2l.PackedStream` (4.1), `d2l.load_reference_model` (4.2). Additional names, each listed under its section's Exports:

| Section | Names |
|---|---|
| 4.1 | `d2l.PackedAttention`, `d2l.PackedGPT`, `d2l.PackCtx`, `d2l.token_weighted_loss`, `d2l.chunked_lm_loss`, `d2l.doc_ids_from_eos` |
| 4.2 | `d2l.PretrainConfig`, `d2l.r2_config`, `d2l.wsd_lr`, `d2l.param_groups`, `d2l.pretrain`, `d2l.attn_logit_stats`, `d2l.flops_per_token`, `d2l.mfu` |
| 4.3 | `d2l.softcap`, `d2l.make_bad_batch`, `d2l.lr_sensitivity` (`d2l.qk_clip_` is imported from `sec_muon`) |
| 4.4 | `d2l.fp8_quant`, `d2l.Fp8Linear`, `d2l.fake_quant_fp4`, `d2l.block_nmse` |
| 4.5 | `d2l.mix_streams`, `d2l.forgetting_gain` |
| 4.6 | `d2l.rope_inv_freq`, `d2l.yarn_mscale`, `d2l.make_recall_task`, `d2l.long_context_gain` |
| 4.7 | none (cells only) |

**Data artifacts (`DATA_HUB`)**; the full list with sizes is in 4.1 and 4.2 and summarized in the compute plan.

### Compute plan

GPU-hours are *projections from FLOP counts at an assumed MFU band of 35–45% (A100) or 30–50% (Ada)* and are labeled "to be measured" in each section; they exist so budgets can be checked, not as results. Peak numbers: A100 40 GB PCIe (the T2 part; about 38 GiB usable), 312 TFLOP/s BF16 dense, 1,555 GB/s HBM2 and a 250 W power cap, joined to a partner by an NVLink bridge in a T2 pair ([NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf); its 80 GB SXM column prints 2,039 GB/s and 400 W). The PCIe part's lower power cap may hold sustained MFU below the SXM band, so the 35–45% band is an assumption to re-measure; RTX 4090, 165 TFLOP/s BF16 dense (fp32 accumulate), from `tab_gpu_specs` in `sec_hardware`.

| § | Tier (reduced T1) | Live budget | Committed artifacts |
|---|---|---|---|
| 4.1 | T1 | about 15 min, under 6 GiB | `lm-pdp-throughput` (consumes Chapter 3's `lm-fwedu-shards-100m`, `lm-fwedu-val`) |
| 4.2 | T2 (T1 preset: 25.2M parameters, 100M tokens, about 4–6 min of compute on one Ada GPU plus evals) | T2 live run is the 1.05B-token short schedule, about 3.1–4.0 h on one A100 40 GB; the committed R2 is the 3.15B-token schedule from a one-off T3 job that runs once, about 1.3–1.7 h on 8×A100 40 GB (the same schedule on one A100 would take 9.4–12.1 h) | R2 checkpoints, training log, eval JSON, OLMo 2 trajectory JSON (see 4.2) |
| 4.3 | T1 | about 15 min, under 8 GiB | `lm-inst-grid` (all runs), spike-and-skip logs |
| 4.4 | T1 on Ada (FP8 real); simulation on any device | about 15 min, under 14 GiB | `lm-fp8-gemm-sweep`, `lm-lowp-runs` |
| 4.5 | T2 (T1: R2 checkpoints, 12M tokens per arm) | 3 arms × 100M tokens, about 3–4 h on one A100 40 GB | `lm-cpt-runs` (logs and per-item eval scores; the three arm checkpoints, about 1.2 GB each in bf16, are not committed) |
| 4.6 | T2 (T1: `r2t1`, 1K→4K) | 4 arms × 100M tokens at 16K, about 3.5–4.5 h on one A100 40 GB | `lm-lc-evals`, the PI/YaRN checkpoints of R2 at 16K for Chapter 12 (about 0.3 GB each, two of them) |
| 4.7 ◇ | T0 | minutes | none |

## Assignment A4: pretrain a small model with packing, document masks and WSD

**Label:** `sec_lm_a4`

**Goal.** The student builds the 4.1 data path and the 4.2 training loop at the T1 preset, trains on a fixed 100M-token FineWeb-Edu slice, and is graded on structural tests and on a validation-loss band.

**Starter code.** `a4_starter.py`, generated from the notebooks, contains the model (`d2l.PackedGPT` at the T1 preset), `d2l.r2_tokenizer()`, `d2l.TokenShards` (Chapter 3's reader), and the following stubs. Each stub has a docstring that states its contract and the `#@save` solution name in the chapter.

- `pack_chunks(tokens, seq_len)` returns `(input_ids, labels, doc_ids, pos_ids)` with the 4.1 conventions.
- `doc_mask(doc_ids)` returns the boolean attention mask.
- `token_weighted_loss(logits, labels)` returns `(sum_loss, n_valid)`.
- `wsd_lr(step, total, warmup, decay, peak, final=0.0)`.
- `param_groups(model, wd)` returns the decay/no-decay split.
- `Stream.state_dict()` and `Stream.load_state_dict(s)` for the cursor.

**Data.** Chapter 3's `DATA_HUB['lm-fwedu-shards-100m']` (about 100M R2-tokenizer tokens; shards in the `d2l-corpus/1` format of 3.6) and `DATA_HUB['lm-fwedu-val']` (the validation shard; token count and size as in its manifest, to be read at authoring); tokenizer `DATA_HUB['lm-r2-tokenizer']`. The `a4_reference.json` band is computed on `lm-fwedu-val` truncated to its first 5M tokens.

**Tasks.**

1. *Packing.* Implement concat-chunk packing with EOS-terminated documents; compute doc ids and per-document position ids; mask the target at every document boundary.
2. *Masked attention.* Make attention block-diagonal causal (the starter provides the SDPA mask path; FlexAttention is an optional extension).
3. *Loss.* Token-weighted reduction; verify against one full-batch pass for accumulation factors 1, 2, 4, 8.
4. *Cursor.* Stateless addressing: batch *s* is a function of `(seed, s)`; implement `state_dict` and `load_state_dict`; support world sizes 1, 2 and 4 with identical global batches.
5. *Schedule and optimizer.* WSD (warmup 5%, decay 15%, linear to 0; the unit test in check 7 uses a different fixture) and AdamW with two parameter groups.
6. *Train.* T1 preset, 100M tokens, global batch 65,536 tokens, peak learning rate chosen by the student from {3e-4, 6e-4, 1.2e-3} using the first 20M tokens only, one run per value.
7. *Ablate.* Re-run the chosen configuration without document masks and without per-document positions (3 seeds each, 25M tokens each).

**Autograder checks** (exact tolerances; numerics on CPU in fp32 unless stated):

| # | Check | Tolerance |
|---|---|---|
| 1 | `pack_chunks` on three hand-built fixtures (docs of lengths [5,3,9,2], [20], [1,1,1,30]; `seq_len=8`) | exact integer equality of all four arrays |
| 2 | Packed logits of each document vs the same document run alone (seed-0 model, d=64, L=2, learned positions *and* RoPE) | max abs difference ≤ 1e-5 |
| 3 | Same test with the mask removed must **fail**: the check asserts the difference exceeds 1e-3 (guards against a mask that is silently all-ones) | difference ≥ 1e-3 |
| 4 | `token_weighted_loss` accumulated over k microbatches vs one pass, on a fixture with unequal valid counts | abs diff ≤ 1e-6; the naive mean-of-means must differ by ≥ 1e-3 |
| 5 | Resume: run 37 steps, save state, build a new stream, load, take 20 batches vs the uninterrupted stream | exact equality of `input_ids`, `labels`, `doc_ids` |
| 6 | World-size elasticity: union over ranks of the step-*s* batches for world 1, 2, 4 | exactly the same multiset of sequences |
| 7 | `wsd_lr` at steps {0, 1, 99, 100, 1000, 8499, 8500, 9999} for the fixture `total=10000, warmup=100, decay=1500, peak=1e-3, final=0` (the warmup is linear to `peak` at step `warmup-1`; the decay is linear from step `total-decay`) | abs diff ≤ 1e-9 |
| 8 | Param groups: decayed set = 2-D non-embedding weights | exact name-set equality |
| 9 | Final validation loss after exactly 100M tokens, student's run | inside the band `[c − h, c + h]`; `c` is the median of three staff-solution seeds and `h = max(3 s, h_min)` with `s` the sample standard deviation of those seeds. **To be calibrated by the authors** and stored in `a4_reference.json`; `h_min` is a protocol parameter set after the first calibration |
| 10 | Throughput sanity: useful tokens/s of the student's packed pipeline | ≥ 0.9 × the staff pipeline on the same GPU (guards against Python-loop packing) |

**Written analysis** (about 600 words, with figures from the student's runs):

1. On the 100M-token slice, what fraction of tokens belongs to a document that is cut by a sequence boundary under concat-chunk packing at `T = 1024`? Predict it from the length histogram with the 4.1 formula, then measure it.
2. State the paired difference (mean and 95% interval) between masked and unmasked training at 25M tokens over three seeds. Say whether the data support a conclusion, and why per-document position resets cannot change RoPE logits within a document.
3. A teammate averages per-microbatch mean losses over 8 microbatches. Construct, from your fixture, the exact bias this introduces.
4. List every piece of state beyond the data cursor that must be restored for a *numerically* identical resume, and name one that cannot be restored bit-exactly on your GPU.
5. From your logs: at which step does the loss curve separate from the constant-LR extrapolation, and how many tokens before the end of the run is that?

**T1 path.** The assignment *is* the T1 path (T1 preset, 25.2M parameters). Time on one Ada GPU: about 10 min for each 100M-token run (to be measured), three LR probes at 20M tokens, six ablation runs at 25M tokens: about 45 GPU-min in total. **T2 full (optional).** Run the 152M R2 architecture for 1.05B tokens and compare with the committed R2 log at the same token count (validation loss within the staff band). **Estimated student hours:** 12–16.

## Teaching sources

All of v5's are kept, annotated with what to take from each.

- [Stanford CS336](https://cs336.stanford.edu/), assignment 1 ([code](https://github.com/stanford-cs336/assignment1-basics)): the component-by-component test discipline (our autograder checks 1–8 follow it); take the trainer-state checkpoint contract.
- Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt): the incremental style, the 0.5M-token batch with gradient accumulation, GPT-3-style hyperparameters, and the HellaSwag-during-training habit that 4.2 generalizes.
- Karpathy's [nanochat](https://github.com/karpathy/nanochat): a current end-to-end small pipeline; take its pretraining stage as a cross-check of R2's token budget logic.
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook): operational practice (ablation discipline, throughput debugging, what to log); take its monitoring checklist for 4.2.
- [SmolLM3 blog](https://huggingface.co/blog/smollm3): a documented multi-stage recipe (long-context extension, mid-training); take it as the worked example for 4.5 and 4.6.
- [OLMo code](https://github.com/allenai/OLMo) (and [OLMo-core](https://github.com/allenai/OLMo-core)): state/cursor handling, checkpointing and the instrumentation that 4.2 and 4.3 imitate.
- Added: [TorchTitan](https://github.com/pytorch/torchtitan) as the production system 4.2 and 4.4 cross-check against; [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) as the public record of what moves a small-model loss (cited in `sec_muon`).

## Sections

### 4.1 The pretraining data path

- **Label:** `sec_lm_data_path` · **Tier:** T1 (no reduction needed; every run is the T1 preset, flag `--tokens`) · **Budget:** about 15 min wall-clock on one Ada GPU and under 10 GiB peak (to be measured); T0 for all structural tests (CPU, SDPA backend) · **Codex units:** 1.8, 5.1, 8.4.
- **Builds on (book):**
  - `sec_gpt`: `d2l.GPT` (tied head, the `blk.attention = ...` swap idiom of `subsec_kv-cache-gqa-quality`).
  - `sec_kv-cache`: `d2l.GQAAttention`, the base of 1.2's `d2l.Qwen3Attention`; `PackedAttention` subclasses the latter and keeps its parameter names, so checkpoints load into either.
  - `sec_adamw` (`subsec_adamw-what-not-to-decay`): the decay/no-decay split, reused verbatim in `param_groups`.
  - `sec_repro`, `sec_read_write`: seeds, determinism flags, safetensors, the run-checkpoint recipe. Not re-taught: only the *data* part of resume is built here.
  - `sec_memory_precision` (`subsec_mp-accumulation`): accumulation equals the full batch **for equal-size microbatches**. 4.1 derives what changes when valid-token counts differ.
- **Builds on (this part):** 3.6 and Chapter 3 (interface adopted from the editor's note): `d2l.r2_tokenizer()`, `d2l.TokenShards` (with `.locate(i)` returning a resume cursor), `d2l.write_shards`; `d2l.bits_per_byte` from 1.4 where BPB is needed; from 1.2, `d2l.Qwen3Attention` and `d2l.Qwen3GPT` (the bases of `PackedAttention` and `PackedGPT`) and `d2l.load_qwen3` (4.5). No other `d2l` names imported besides the book's.
- **Exports:**
  - `d2l.doc_ids_from_eos(tokens, eos_id=0) -> (doc_ids, pos_ids)`: int32 tensors shaped like `tokens` (EOS id 0 for the R2 tokenizer; the helper is the reference implementation and the oracle for the offsets-based path). `doc_ids[t]` is the number of EOS tokens at indices `< t` (the EOS belongs to the document it ends). In the stream itself, boundaries come from the shard's `.idx` document offsets, and the helper is used in tests (3.6 confirms that `d2l.write_shards` appends the EOS to every document; the test checks this). `pos_ids[t]` is `t` minus the index of the first token of that document within the window (0 at the first token of a fragment).
  - `d2l.PackCtx(doc_ids, pos_ids)`: a `NamedTuple` carried through blocks in the `valid_lens` slot of `d2l.TransformerBlock.forward` (the seam the book built for this).
  - `d2l.PackedAttention(d2l.Qwen3Attention)`: `__init__(num_hiddens, num_heads, num_kv_heads, *, head_dim=None, qk_norm=False, rope_theta=10000.0, norm_eps=1e-6, bias=False, rope=True, rope_inv_freq=None, rope_mscale=1.0, softcap=None, backend='auto')`; the arguments through `rope` go to `Qwen3Attention` unchanged (`head_dim` defaults to `num_hiddens // num_heads`; Qwen3-0.6B has 128 against 64, used in 4.5); `forward(q, k, v, ctx=None)`. Backends: `'sdpa'` (explicit boolean block-diagonal causal mask; runs on CPU), `'flex'` (FlexAttention `create_block_mask` with a `mask_mod` on `doc_ids`; GPU), `'math'` (explicit scores; needed for soft-capping and for logit probes in 4.3). With `ctx=None` it is exactly `d2l.Qwen3Attention`. `rope_inv_freq` (the per-pair angular frequencies `θ_m`) and `rope_mscale` (a factor multiplying both `cos` and `sin`, so logits scale by its square) are the hooks 4.6 uses; a given `rope_inv_freq` overrides the frequencies derived from `rope_theta`. `qk_norm=True` gives the inherited per-head `q_norm`, `k_norm` (RMSNorm of size `head_dim`).
  - `d2l.PackedGPT(d2l.Qwen3GPT)`: `__init__(vocab_size, num_hiddens, num_heads, num_kv_heads, num_blks, max_len, pos='rope', qk_norm=False, rope_theta=1e4, backend='auto', **qwen3_kwargs)`: calls `d2l.Qwen3GPT.__init__` (R2's RoPE base 10,000 replaces Qwen3's 1e6; `norm_eps` stays 1e-6; `head_dim`, `ffn_hidden` and `tie_embeddings` pass through `qwen3_kwargs`), then swaps every `blk.attention` for a `PackedAttention` of the same shape (cheap at these sizes). `pos='learned'` adds a `pos_emb` table as the book's `d2l.GPT` does and builds attention with `rope=False`; only the 4.1 position-reset check uses it. `forward(X, ctx=None, return_hidden=False)` returns logits (or the final normed hidden state, for the chunked loss). The classmethod `PackedGPT.adopt(model, backend='auto')` takes a `d2l.Qwen3GPT` (R1 from `d2l.load_qwen3`, used in 4.5), builds a `PackedGPT` of the same shape on the meta device from the attributes 1.2 guarantees (`token_emb`, `len(blks)`, each attention's `head_dim, num_heads, num_kv_heads, rope_theta` and `q_norm` presence, `ffn.W_g.out_features`, `norm.eps`, `lm_head` presence), and loads the weights with `load_state_dict(model.state_dict(), assign=True)`: the parameter names are identical, so nothing is copied. Initialization: `N(0, 0.02)` for all linear layers and embeddings; `W_o` and `W_2` (residual-writing matrices) scaled by `1/sqrt(2 L)` (the GPT-2 convention; `sec_init_param` for why).
  - `d2l.token_weighted_loss(logits, labels, ignore_index=-100) -> (sum_nll, n_valid)`; `d2l.chunked_lm_loss(hidden, weight, labels, chunk=8192, z_loss=0.0, ignore_index=-100) -> (sum_loss, n_valid)` (canonical signature per `01_interfaces.md`; also imported by 7.1 and 8.x, so it must not change): the same quantity computed by slices of `chunk` positions under `torch.utils.checkpoint`, so the `[tokens, vocab]` logit tensor is never alive in full. The `z_loss` argument adds `z_loss * logsumexp(logits)^2` per valid token (consumed in 4.2 and 4.3).
  - `d2l.PackedStream(shards, seq_len, global_batch, seed=0, rank=0, world=1, epoch=0, skip=())` where `shards` is a `d2l.TokenShards` (Chapter 3's reader; 4.1 defines no shard format of its own):
    - `.batch(step) -> dict(input_ids int64[B,T], labels int64[B,T] with -100 on masked targets, doc_ids int32[B,T], pos_ids int32[B,T])` with `B = global_batch // world`; a pure function of `(manifest, seed, epoch, step, skip)`.
    - `__iter__`, `.state_dict()`, `.load_state_dict(s)`, `.skip(step)` (marks a physical step for dropping, used by 4.3), `.physical_index(step)`.
  - Data artifacts: none new. 4.1 consumes Chapter 3's `lm-fwedu-shards-100m` (T1), `lm-fwedu-val` and `lm-fwedu-manifest-4b` (about 8 GB for up to 4.0e9 tokens; the R2 run uses its first 3,145,728,000 tokens). Each is a set of shards in the `d2l-corpus/1` layout: a flat `uint16` `.bin` (the 32,768-token vocabulary fits), a `uint32` `.idx` of document offsets, `meta.parquet` per-document metadata (including the source URL host used for 4.2's domains), and `manifest.json`. `d2l.TokenShards.locate(i)` maps a global token index to a resume cursor (shard, byte offset, document index); `PackedStream` uses it for random access. A small `lm-pdp-throughput` JSON (the measured table of the headline experiment) is committed.
- **Models and data:** `HuggingFaceFW/fineweb-edu`, config `sample-10BT` (ODC-By 1.0), as prepared by Chapter 3: the dataset revision and the parquet files used are recorded in the Chapter 3 manifests (the blueprint's reference revision is `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; T1 uses the 100M-token shard set, R2 the 4B manifest, validation `lm-fwedu-val`). Tokenizer: the R2 tokenizer of 3.6 (vocabulary 32,768, EOS id 0, no BOS; the embedding is not padded: 32,768 is a multiple of 128). The tokenization pipeline (`d2l.write_shards`) belongs to 3.6; 4.1 shows only how to *read* it.
- **Packages:** `torch>=2.5` (FlexAttention; `enable_gqa` in SDPA needs 2.5), `pyarrow` (for `meta.parquet`), `numpy`; optional `datasets>=3.0` and `torchdata` for the production comparison.

#### Narrative arc

1. **The question.** The opening cell trains nothing. It takes two documents `A` (200 tokens) and `B` (300 tokens), packs them into one 512-token sequence the way `nanoGPT` does (concatenate, attend causally across the boundary), and prints the max absolute difference between the logits of `B` inside the pack and `B` alone: a number far from zero (to be measured). It then simulates a crash at step 1,000 of a shuffled-stream run, restarts it with a fresh `random.Random(seed)` and prints the fraction of the first 100 post-restart batches that were already seen before the crash (a large fraction if the shuffle is re-seeded; to be measured). Two silent failures, one number each.
2. **A step is a pure function: the contract.** One cell states the design rule: `batch(step)` depends only on `(manifest, seed, epoch, step, skip)`. Consequences: resume needs only an integer, the world size can change at resume, and a bad batch can be named and removed (4.3). Compare with the three alternatives in a table, `tab_lm_pdp_exactness`: stateful iterators (resume by replaying), HF `IterableDataset.state_dict()` (shard + example offset), torchdata `StatefulDataLoader` (worker state). Define three exactness levels: **L1** identical batches (this section guarantees it); **L2** identical loss trajectory up to floating-point nondeterminism (atomics, fused kernels); **L3** bit-identical (needs deterministic algorithms, slower, `sec_repro`). State that dataset reproducibility (L1) is distinct from deterministic numerical training (L2/L3).
3. **Pre-tokenized sharded streams (Chapter 3's format, read here).** Open `lm-fwedu-shards-100m` with `d2l.TokenShards`; print its manifest (schema `d2l-corpus/1`, tokenizer, EOS id, dtype, shard hashes, source files, dataset revision); show the flat `uint16` token array, the `uint32` document offsets and one row of `meta.parquet`; print the document-length histogram and the `dump` composition per shard if `meta.parquet` carries it (FineWeb dumps are time-ordered; a shard that is mostly one dump is a data-order bug that sequence-level shuffling fixes). Use `locate(i)` to turn a token index into a cursor and verify it against a linear scan on a small shard. Say why shard sizes that are powers of two make window addressing simple, and what the stream does when they are not (windows may cross a shard boundary through two reads; the last partial window of the corpus is dropped). Dtype rule: `uint16` holds vocabularies up to 65,535; Qwen3's 151,936 needs `uint32` (4.5; 3.6 confirms `write_shards` uses `uint32` when the vocabulary exceeds 65,535).
4. **Sequence packing.**
   - Concat-chunk: the token stream is cut into windows of `T+1` tokens with one-token overlap (inputs `x[0:T]`, labels `x[1:T+1]`); the last window of the last shard is dropped.
   - Conventions (`tab_lm_pdp_conventions`): each document is `tokens + [EOS]`; the EOS is part of its document and is a *predicted* token (the model learns to end); the first token of a document has no predicted label from the previous document (masked, `-100`); `doc_ids` increments after each EOS; `pos_ids` restart at 0 at each document start, and at 0 at the start of a window for a document continued from the previous window (a *fragment*).
   - Fragmentation and truncation: show, from the length histogram, how many tokens sit in fragments. Introduce best-fit packing (Ding et al.) as the alternative that avoids most cuts at the price of padding; measure both on the shard (analysis cell only; R2 uses concat-chunk because it keeps `batch(step)` stateless).
   - Figure `fig_lm_pdp_truncation`.
5. **Document masks and positions.**
   - Mask: `mask[q, kv] = (q >= kv) & (doc[q] == doc[kv])`. Show it as an image for a 32-token toy (`fig_lm_pdp_masks`).
   - **Predict before running (a cell with a markdown question):** with RoPE, does resetting positions change the logits within a document? Answer after the check: no (RoPE scores depend only on relative offsets), up to floating-point error from large absolute angles; with `pos='learned'` it does. So per-document positions are a correctness requirement for absolute schemes and a numerical nicety for RoPE.
   - Backends: SDPA with an explicit mask (reference, any device), FlexAttention block masks (`create_block_mask` once per micro-batch, shared by all layers; `torch.compile`d `flex_attention`; fully masked blocks are skipped, so cost follows the sum of squared document lengths rather than `T^2`). Print the block-sparsity of the mask on real batches.
   - Leak diagnostic: for a model trained *without* the mask (the headline experiment), the fraction of attention probability mass that falls on other-document keys, per layer (`fig_lm_pdp_leak`).
6. **Token-weighted loss.** The loss is a sum over valid targets divided by the **global** number of valid targets. Derivation below. Show the DDP form: each rank computes `sum_nll / N_global * world` (DDP then averages), with `N_global` from an `all_reduce` of `n_valid` *before* the backward pass. Show `chunked_lm_loss` and its memory (the `[tokens, 32,768]` fp32 logits of a 16k-token microbatch is 2.1 GB; chunking removes it).
7. **Resumable cursors and the production alternatives.**
   - Stateless addressing: `perm = default_rng([seed, epoch]).permutation(n_seq)`; step `s` takes `perm[s*G:(s+1)*G]`; rank `r` takes `idx[r::world]`; random access through `TokenShards` (`locate` for the first token of a window, two reads when the window or its one-token overlap crosses a shard edge).
   - `state_dict() = {version, seed, epoch, step, skip, layout_hash}`; `load_state_dict` raises on a layout-hash mismatch (a silent shard swap is the worst failure).
   - What resume also needs: model, optimizer, scheduler step, RNG states, scaler/EMA state (`sec_read_write`, `sec_repro`); only the data side is built here.
   - Production alternatives, shown side by side in one cell each: `datasets` streaming with `IterableDataset.state_dict()`; `torchdata.stateful_dataloader.StatefulDataLoader` around a map-style dataset; both reproduce L1 only if the shuffle buffer and shard order are part of the state.
8. **Input starvation.** Measure batches/s of `PackedStream` (memmap, uncompressed) against the training step rate of the T1 preset and against a zstd-compressed shard variant (if Chapter 3 ships one; otherwise compress one shard locally). The data rate needed is tiny (tokens/s × 2 bytes: at 100k tokens/s, 0.2 MB/s), so the bottleneck, if any, is the Python work per batch (the `doc_ids_from_eos` cumsum and the mask build). Report where it lands (to be measured) and the fix (vectorize; overlap in a background thread).

#### Derivations

**D1. Packed attention equals unpacked attention per document.**
*Statement.* Let a window contain documents `D_1, ..., D_m` with token ranges `R_k`. Let `f` be a `GPT` with RoPE, RMSNorm, per-token FFN and tied head, run on the window with the block-diagonal causal mask and any `pos_ids`. For every `k` and `t in R_k`, the logit vector at `t` equals the logit vector at the corresponding position when `D_k` is run alone with a plain causal mask.
*Assumptions.* RoPE only (no absolute table); no cross-token operation other than attention; exact arithmetic.
*Steps.*
1. Embedding, RMSNorm, FFN, residual add and the output head act per token: if the input hidden states of `R_k` agree with the stand-alone run at layer `l`, so do these operations' outputs.
2. The attention score between `t, u in R_k` is `q_t^T R_{u-t} k_u` where `R` is the RoPE rotation (`eq_rope-def`). It depends on `u - t` only, so it is unchanged by any shift of `pos_ids` that is constant on `R_k`, and by the absolute offset of `R_k` in the window.
3. The mask removes every key outside `R_k` for queries in `R_k`; the softmax denominator therefore sums over exactly the keys of the stand-alone run.
4. Hence the attention output for `t in R_k` equals the stand-alone output, so layer `l+1` inputs agree. Induction on `l` from the embedding.
5. With a learned absolute table, step 2 fails unless `pos_ids` resets to 0 at each document start; with resets the table index equals the stand-alone index and the claim holds again.
*Check cell.* Seed-0 `PackedGPT` (d=64, L=2, H=4, 2 documents of 5 and 9 tokens, fp32): max abs logit difference ≤ 1e-5 for the masked pack, with and without resets (RoPE); ≥ 1e-3 for the unmasked pack; with `pos='learned'`, ≤ 1e-5 only with resets. The same check is repeated with the `'flex'` backend on GPU in bf16 against the `'sdpa'` backend (max abs difference ≤ 2e-2 on logits of std about 1, a tolerance to be tightened after the first run).

**D2. How many tokens are cut by a window boundary.**
*Statement.* Document lengths `ℓ` are i.i.d. with mean `ℓ̄`; the stream is cut every `T` tokens. A document of length `ℓ` is split with probability `min(1, (ℓ-1)/T)` when its start is uniform modulo `T`, so the token-weighted fraction in split documents is `f_cut(T) = E[ℓ · min(1, (ℓ-1)/T)] / ℓ̄`.
*Steps.*
1. A document occupies `ℓ` consecutive stream positions; it is split iff a cut lies strictly inside, i.e. one of `ℓ-1` interior gaps is at a multiple of `T`.
2. For a start uniform over residues mod `T`, the number of cuts inside is `floor`-or-`ceil` of `(ℓ-1)/T`; for `ℓ-1 < T` it is Bernoulli with mean `(ℓ-1)/T`; for `ℓ-1 ≥ T` it is at least 1.
3. Weighting by `ℓ/ℓ̄` (a token is in a length-`ℓ` document with probability `ℓ f(ℓ)/ℓ̄`) gives the formula; evaluate on the empirical histogram.
4. The number of *predicted-label losses* per window: every fragment start loses one label, so valid targets per window `= T - (number of document starts inside the window)`; the expected loss of labels is `T/ℓ̄ + 1` per window at most.
*Check cell.* Compute `f_cut` from the histogram of the 5M-token validation shard for `T ∈ {256, 512, 1024, 2048, 4096, 8192}` and compare with the measured fraction from actually packing; agreement within 2% absolute (statistical error of the empirical fraction on 5M tokens). Then run best-fit packing on the same documents and report its split fraction (only documents longer than `T` are split) and its padding fraction (to be measured).

**D3. Token-weighted reduction and the bias of mean-of-means.**
*Statement.* For microbatches `k = 1..K` with `n_k` valid targets and mean loss `ℓ̄_k`, the full-batch mean is `L = Σ n_k ℓ̄_k / N` with `N = Σ n_k`. The naive average `L̃ = (1/K) Σ ℓ̄_k` satisfies `L̃ - L = -Cov_k(n_k/n̄, ℓ̄_k)` where `n̄ = N/K` and the covariance is over the uniform distribution on `k`.
*Steps.*
1. `L = Σ w_k ℓ̄_k` with `w_k = n_k/N = (1/K)(n_k/n̄)`.
2. `L̃ - L = Σ (1/K - w_k) ℓ̄_k = -(1/K) Σ (n_k/n̄ - 1) ℓ̄_k`.
3. `Σ_k (n_k/n̄ - 1) = 0`, so subtracting `ℓ̄ · 0` turns the sum into `K` times the covariance.
4. The gradients obey the same identity with `ℓ̄_k` replaced by `∇ℓ̄_k`.
5. DDP: with per-rank loss `S_r / N_global · world`, the mean over ranks of the gradients is `∇(Σ_r S_r / N_global)`, which is the global token-weighted gradient; without the all-reduce of `n_valid` the per-rank denominator is local and the identity of steps 1–4 applies across ranks.
*Check cell.* Fixture of 8 microbatches with valid counts `[1024, 900, 512, 1024, 300, 1000, 700, 1024]` and loss drawn from a seeded normal; compare `token_weighted_loss` accumulated to a single pass (≤ 1e-6, fp32) and the identity of step 2 to ≤ 1e-6; show that the naive value differs by the predicted covariance.

#### From-scratch implementation

About 150 lines in all (the stream is the largest part):

| Component | Signature | Lines |
|---|---|---|
| `doc_ids_from_eos` | cumsum of `(x == eos)` shifted by one; `pos = arange - max.accumulate(where(start, arange, 0))` | 12 |
| `PackedAttention` | `__init__`, `forward(q, k, v, ctx)`; `_mask_sdpa`, `_flex` | 55 |
| `PackedGPT` | init scaling and `forward(X, ctx, return_hidden)` | 30 |
| `token_weighted_loss`, `chunked_lm_loss` | as above | 22 |
| `PackedStream` | `batch`, `state_dict`, `load_state_dict`, `skip`, `physical_index` | 40 |

Numerical tests (seed 0, CPU, fp32 unless stated):

1. `doc_ids_from_eos` on `[5, 6, EOS, 7, EOS, EOS, 8]`: `doc_ids = [0,0,0,1,1,2,3]`, `pos_ids = [0,1,2,0,1,0,0]`.
2. D1 check (≤ 1e-5 / ≥ 1e-3, see above).
3. `PackedAttention` with `ctx=None` vs `d2l.GQAAttention` with the same weights: exact equality.
4. SDPA vs flex vs math backends: ≤ 1e-5 fp32 on CPU for sdpa/math; flex vs sdpa ≤ 2e-2 in bf16 on GPU.
5. Stream: `batch(s)` repeated twice is identical; resume at step 37 gives 50 identical batches; world sizes 1, 2, 4 produce the same multiset of sequence indices at every step; changing `seed` changes the batch; `skip=(3,)` makes logical step 3 equal physical step 4.
6. `chunked_lm_loss` equals `token_weighted_loss(F.linear(hidden, W), labels)` to 1e-6 (value and the gradient with respect to `hidden`).
7. Layout hash: editing one byte of a shard changes the SHA and `load_state_dict` raises.

#### In production

- **FlexAttention** (PyTorch ≥ 2.5, `torch.nn.attention.flex_attention`): `create_block_mask`, `and_masks`, score mods. Here it is the fast path; `'sdpa'` with the explicit mask is the test oracle for it. Mention the other production route, `flash_attn_varlen_func` with `cu_seqlens` (no padding, no mask), and that TorchTitan's data loader and trainer provide document masking and resumable streaming (state of the dataset iterator in the checkpoint). Verify the exact TorchTitan option names at authoring time (unverified here).
- **Streaming.** HF `datasets` iterable datasets (`IterableDataset.state_dict()`/`load_state_dict()` are documented on the streaming page) and `torchdata.stateful_dataloader.StatefulDataLoader` (the data repo's README lists it as a drop-in for `DataLoader`). The cell compares their resume behavior with `PackedStream` on the first 1,000 steps: expected, per their documentation, resumption from shard and example offsets, which matches L1 provided the shuffle seed is fixed (to be measured).
- **Oracle role.** The from-scratch `PackedStream` and the D1 equivalence test are the oracles any production loader must pass; Chapter 6 reuses them when the stream is partitioned across ranks.

#### Headline experiment

- **Hypothesis.** Training with block-diagonal document masks changes the validation loss of the T1 preset by less than seed noise at 25M tokens and `T = 1024`, even though an unmasked model measurably places attention mass on other documents. (A two-sided claim: the notebook does not presume a direction; Zhao et al. and the Llama 3 report motivate masking, mostly at longer contexts.)
- **Setup.** `PackedGPT` T1 preset (d=384, L=8, H=6, H_kv=2, head dim 64, RoPE base 10,000, SwiGLU width 1,024, tied embeddings, QK-norm off, 25.2M parameters: 12.6M embedding plus 12.6M blocks), `T = 1024`, global batch 32 sequences (32,768 tokens), 763 steps (25M tokens), AdamW (0.9, 0.95), weight decay 0.1 on matrices, peak LR 1e-3, WSD (warmup 40, decay last 15%), bf16 autocast, `chunked_lm_loss`, seeds 0–2, tier T1.
- **Procedure.** Six runs: {masked, unmasked} × 3 seeds, same data order across the pair for a given seed (the stream seed equals the model seed). Evaluate on `lm-fwedu-val` with the *masked* loss in both cases (so the comparison is on the same targets). For the unmasked models compute the attention-leak diagnostic on 16 validation windows.
- **Metrics.** Validation loss at the end; per-seed paired difference `Δ = L_unmasked - L_masked`, its mean and a bootstrap or t interval over the 3 pairs (`d2l.paired_bootstrap` is for items, not seeds; use the t interval with 2 degrees of freedom and say how weak that is); leak fraction per layer; useful tokens/s and peak memory for four pipelines (padded-unpacked, packed without mask, packed + SDPA mask, packed + flex mask) at `T ∈ {1024, 4096}`.
- **Plots.** (i) `fig_lm_pdp_leak`: x layer index, y fraction of attention mass on other-document keys, one line per seed, plus a horizontal line at the fraction expected from a uniform attention over the window (the average share of other-document keys). (ii) `fig_lm_pdp_throughput`: grouped bars of useful tokens/s with peak-memory annotations.
- **Expected outcome.** The leak is positive in every layer (direction); a validation-loss difference of unresolved sign is the likely result at this scale and budget (to be measured), and the notebook must say "no conclusion" if the interval contains 0. For throughput, packing removes the padding of the unpacked run; the flex path should beat the explicit-mask SDPA path once documents are shorter than `T` (direction; the size is to be measured).
- **Noise check.** Three seeds with paired data order, t interval, and a stated inability to resolve differences below the seed spread.
- **T1 reduction.** None needed. For T2+ there are no artifacts beyond the token shards (listed under Exports).

#### Figures and tables

- `fig_lm_pdp_masks`: a 32×32 mask image for three documents (lengths 12, 8, 12), made by `PackedAttention._mask_sdpa`.
- `fig_lm_pdp_truncation`: x window length `T` (log scale), y fraction of tokens in split documents; series: measured concat-chunk, formula D2, best-fit.
- `fig_lm_pdp_leak`, `fig_lm_pdp_throughput`: see above.
- `tab_lm_pdp_conventions`: rows EOS, first-token label, `doc_ids`, `pos_ids`, fragment, final window, dtype, with the rule for each.
- `tab_lm_pdp_exactness`: rows = scheme (stateful iterator replay, HF iterable `state_dict`, `StatefulDataLoader`, `PackedStream`), columns = L1/L2/L3, elastic world size, ability to drop one batch, state size in bytes.

#### Pitfalls and misconceptions

- Seeding a shuffle again at restart replays old data; a fresh `DataLoader` with `num_workers > 0` forks the same RNG into every worker and duplicates batches.
- Literal `<|endoftext|>` in a web page becomes a real EOS and silently splits a document: `disallowed_special='all'`.
- Masking the attention but not the loss: the first token of a document is then predicted from the previous document's EOS, with a label the model cannot know.
- "Per-document positions are essential" is true for learned tables and false for RoPE logits; the check cell is there to make this concrete.
- Mean of microbatch means (D3); DDP without the global `n_valid`.
- `uint16` silently wraps for vocabularies above 65,536 (not an issue for R2's 32,768; it is for Qwen3 in 4.5).
- Recompilation: FlexAttention block masks of changing shape retrace under `torch.compile`; fix shapes (fixed `B`, `T`).
- Dataset reproducibility is not run reproducibility: identical batches do not give identical weights (atomics, non-deterministic kernels).

#### Summary (draft bullets)

- Make the data path a pure function of the step: resume needs one integer, a bad batch can be named and skipped, and the world size can change.
- Pack by concatenation with EOS-terminated documents; mask targets and attention at document boundaries; document-level equivalence with unpacked inference holds to 1e-5, and holds for RoPE even without position resets.
- Reduce the loss by the global count of valid tokens; averaging microbatch means is biased by the covariance of valid-token share and loss.
- Best-fit packing trades split documents for padding; concat-chunk keeps addressing stateless, and the fraction of split tokens is computable from the length histogram.
- At T1 scale the loss effect of masking is small and may not be resolvable; the attention leak is measurable.

#### Exercises

1. **(check)** A window of `T = 8` holds two complete documents of lengths 5 and 3, each ending in EOS: `x = [a1 a2 a3 a4 EOS b1 b2 EOS]`; the next window begins with `c1`. Write `labels`, `doc_ids`, `pos_ids` and the number of valid targets. *Sketch:* `labels = [a2 a3 a4 EOS -100 b2 EOS -100]` (position 4 would predict `b1` from the previous document's EOS; position 7 would predict `c1`, the first token of a new document); `doc_ids = [0,0,0,0,0,1,1,1]`; `pos_ids = [0,1,2,3,4,0,1,2]`; 6 valid targets, i.e. `T` minus the number of document starts that a label would have to cross.
2. **(check)** Why does `batch(step)` use `perm[s*G:(s+1)*G]` and `idx[r::world]` instead of giving each rank its own shuffled shard list? *Sketch:* the union over ranks must not depend on the world size, otherwise the resumed run changes the global batch; a rank-local shard assignment makes the set of sequences in step `s` depend on `world` and prevents elastic resume.
3. **(derive)** Extend D2 to the case where documents longer than `T` are dropped instead of split. Give the fraction of tokens lost and the new valid-target density. *Sketch:* tokens lost `= E[ℓ 1{ℓ > T}]/ℓ̄`; the remaining documents are packed without splitting only if best-fit, otherwise the cut probability applies to the survivors.
4. **(derive)** Show that the gradient of the token-weighted loss with respect to the logit at a valid position `t` is `(softmax_t - onehot(y_t))/N`, and that masked positions receive exactly zero gradient. Why does the `1/N` (not `1/n_k`) matter under accumulation? *Sketch:* differentiate the sum; accumulation sums the microbatch gradients, each already divided by the global `N`; dividing by `n_k` would reweight microbatches by `N/(K n_k)`.
5. **(measure)** Predict, then measure, the fraction of other-document attention mass for the unmasked T1 model at `T = 256` and `T = 4096` (train each for 10M tokens). Which layer leaks most, and does the leak track the fraction of the window outside the current document? *Expected:* a longer window gives a larger fraction outside the current document (direction); the layer profile is to be measured.
6. **(measure)** Replace the in-RAM `perm` by a hash-based shuffle (a Feistel permutation) so that no `O(n_seq)` array is needed. Verify bit-identical resume and compare the time to compute step-`s` indices for `n_seq = 768,000` against the permutation array. State what you gave up (the exact uniformity of the permutation). 
7. **(open)** Design a data path for a mixture of five sources with fixed sampling weights and resumable, deterministic mixing across ranks (the 3.4 mixture and 4.5 replay both need it). Specify the state, show that the realized proportions converge to the weights, and decide how the cursor survives a change of weights at resume (the situation of a curriculum).

#### Further reading

- Papers: Ding et al., [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830) (best-fit packing; the truncation measurements D2 predicts); Zhao et al., [Analysing the Impact of Sequence Composition on Language Model Pre-Training](https://arxiv.org/abs/2402.13991) (effect of cross-document attention, motivates the headline experiment); [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (document masking in 8K-token sequences, stated in the pretraining section); [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) (FineWeb-Edu, what the shards contain).
- v5's sources: [FlexAttention](https://pytorch.org/blog/flexattention/) (block masks, document masking example; API: [flex_attention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)); [datasets streaming](https://huggingface.co/docs/datasets/stream); [torchdata](https://github.com/pytorch/data).
- Tutorials: Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) (the shard-and-cursor data loader this section hardens); CS336 lecture notes on data loading.
- Code: [OLMo](https://github.com/allenai/OLMo) data module (deterministic global-index shuffling), [TorchTitan](https://github.com/pytorch/torchtitan) (checkpointable datasets).

#### Risks and open decisions

- *FlexAttention API drift or an Ada-specific compile issue:* the `'sdpa'` backend is the default for tests; the flex path is behind `backend='flex'` and the headline experiment's throughput table reports "flex unavailable" without failing.
- *Chapter 3's shard interface* (`write_shards`, `TokenShards`, `locate`) is confirmed by 3.6 (EOS appended per document; `uint32` above 65,536); if `.idx` offsets exclude the EOS, only the boundary derivation changes (see `doc_ids_from_eos`).
- *Hosting* is Chapter 3's (`lm-fwedu-shards-100m`, `lm-fwedu-val`). Fallback if the 100M set is too large for the autograder: a 25M-token slice, which makes the A4 band wider.
- *FineWeb-Edu license* (ODC-By 1.0) requires attribution; the manifest carries it. CommonCrawl terms of use also apply to redistribution of derived tokens: Chapter 3 decided it (2026-10-29): FineWeb-Edu-derived data ships with attribution, and the shards this chapter consumes carry it.
- *The EOS-as-predicted-token convention* means R2 emits EOS at document ends, which later chapters rely on; flag if Chapter 7 expects a different end token.

### 4.2 Training the reference model

- **Label:** `sec_lm_reference_model` · **Tier:** T2 (reduced T1 config: the T1 preset, `d2l.r2_config('t1')`, 25.2M parameters, 100M tokens, same code) · **Budget:**
  - T1 preset, one Ada GPU: about 4–6 min of compute at 30–50% MFU (3.8–6.4 min from FLOPs, to be measured), plus about 4 min for evals; peak memory to be measured (expected under 10 GiB with `chunked_lm_loss`).
  - T2 live (`r2_short`: same architecture, 2,000 steps, 1.05B tokens): 3.1–4.0 h on one A100 40 GB at 45–35% MFU (the 312 TFLOP/s peak is the same for the PCIe and SXM parts); memory to be measured, design target ≲ 30 GiB against about 38 GiB usable at micro-batch 8 × 4,096 tokens (2.2 GB of fp32 weights, gradients and AdamW moments for 138.4M parameters; the rest is activations). Fallback if the measured peak exceeds 38 GiB: micro-batch 4 with accumulation 32 on one GPU (accumulation 4 per GPU in the T3 job): the same global batch of 128 sequences and the same loss (numerical test 3 below).
  - The committed R2 (6,000 steps, 3.15B tokens): one T3 job on 8×A100 40 GB (micro-batch 8 × 4,096 tokens per GPU, accumulation 2; it fits the 40 GB cards by the budget above), run once, 1.3–1.7 h at 45–35% MFU with 90% data-parallel efficiency assumed (the same schedule on one A100 takes 9.4–12.1 h). The assumed efficiency is conservative even without NVLink: the ring moves `2(n−1)/n · 2P` = 485 MB per GPU per step for bf16 gradients, and at a quarter of the datasheet's printed 64 GB/s PCIe Gen4 rate (16 GB/s) that is 30 ms against about 0.8 s of compute per step at MFU 0.40 (about 4%; about 8% if the gradients are reduced in fp32). The job logs `nvidia-smi topo -m` and the measured efficiency. These are projections, not results.
  - Codex units: 5.1, 5.9 (logging half), 16.2 (model-card analysis).
- **Builds on (book):**
  - `sec_scheduler`: WSD and `subsec_scheduler-branching` (R2's stable-phase checkpoint is a branch point). `WSDScheduler` there is not exported, hence `d2l.wsd_lr` below.
  - `sec_practice`: clipping (`Clipped`), EMA, and the recipe table `tab_practice_recipes` (R2's row is added to it as an exercise).
  - `sec_adamw`: parameter groups; `sec_muon`: AdamW remains the default, Muon is an exercise.
  - `sec_scaling-laws`: `eq_six_nd`, parameter census, the profiler FLOP check. The FLOP formula is extended with the attention-score term and GQA, not re-derived.
  - `sec_perf_model`, `sec_hardware`: peak FLOP/s, roofline, `d2l.Benchmark`. MFU itself is defined here (the book has it only as an exercise in `sec_fast_transformer`).
  - `sec_init_param`, `sec_memory_precision`, `sec_numerics`: initialization, autocast, bf16.
- **Builds on (this part):**
  - 4.1: `d2l.PackedStream`, `d2l.PackedGPT`, `d2l.PackedAttention`, `d2l.chunked_lm_loss`, `d2l.token_weighted_loss`.
  - 1.1: `d2l.load_hf(model_id, revision=...)` for OLMo 2 revisions.
  - 1.4–1.5: `d2l.score_choices`, `d2l.EvalSuite`, `d2l.paired_bootstrap`.
  - 2.4: `d2l.book_corpus()` as one held-out domain.
  - 6.1 (assumed): `d2l.launch(script, nodes=1, gpus_per_node=8, **env)`, used once to run the T3 job.
  - 1.5: `d2l.EvalSuite(['hellaswag','arc_easy','piqa','sciq','winogrande'], n=..., seed=0).run(model, tok)` returns a `d2l.EvalResult` (`sciq` is `allenai/sciq`, CC-BY-NC-3.0, checked 2026-10-29: the suite downloads it at run time and this chapter never commits its item text, only per-item 0/1 scores); `.scores(task)` gives per-item 0/1 arrays ordered by `item_id`, which feed `d2l.paired_bootstrap(scores_a, scores_b)`; that returns a `d2l.BootstrapResult` (read `.mean` and `.ci`, not a 3-tuple). Tasks are scored with `d2l.score_choices(..., norm='token')`.
  - `assumed:` `d2l.generate` (returns `list[d2l.Generation]`; use `.text`) and `d2l.score_choices` accept any callable `model(X) -> logits` plus a tokenizer exposing `encode`, `decode`, `eos_token_id`; `d2l.r2_tokenizer()` (3.6, vocabulary 32,768, EOS id 0) is assumed to provide it. `d2l.bits_per_byte` is imported from 1.4, not redefined here.
- **Exports:**
  - `d2l.PretrainConfig`: a dataclass with `arch` (`d, L, H, H_kv, f, vocab, T, rope_base, qk_norm`), `steps, micro_batch, accum, peak_lr, warmup, decay, final_lr, betas, eps, wd, clip, z_loss, dtype, compile, ddp, log_every, probe_every, val_every, ckpt_steps, out_dir, seed`.
  - `d2l.r2_config(preset='r2')`: returns the config of the `r2`, `r2_short` or `t1` preset (values in `tab_lm_r2_spec`).
  - `d2l.wsd_lr(step, total, warmup, decay, peak, final=0.0) -> float`: linear warmup to `peak` at step `warmup - 1`, constant, linear decay over the last `decay` steps to `final`.
  - `d2l.param_groups(model, wd) -> list[dict]`: decay set is the 2-D weights that are not embeddings; no decay for embeddings, norm gains, QK-norm gains.
  - `d2l.pretrain(model, stream, cfg, val_fn=None, hooks=(), resume=None) -> list[dict]` (about 100 lines, see below): returns the per-step log; writes checkpoints at `cfg.ckpt_steps`.
  - `d2l.attn_logit_stats(model, batch) -> dict`: per layer and head, `max_logit` and `entropy` of the attention distribution on one probe batch, computed by an explicit `q k^T / sqrt(d_h)` outside the fused kernel (4.3 reuses it).
  - `d2l.flops_per_token(arch, T=None, causal_half=False) -> float` and `d2l.mfu(tokens_per_s, arch, peak_flops, n_gpus=1) -> float`. `arch` may be a `PretrainConfig.arch`, a Hugging Face config, or Chapter 6's `d2l.ModelShape`; attributes are read by HF names (`num_hidden_layers, hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, vocab_size`), falling back to the short names (`L, d, H, H_kv, f, vocab`). Returns *training* FLOPs per token (forward + backward), `6 N_mm + 12 L d T` in the PaLM convention of 4.2 D1, with the attention term halved when `causal_half=True` and omitted when `T=None`; divide by 3 for forward-only (inference) FLOPs. 6.2 and 7.8 import these functions (6.2 does not define its own `mfu` or `model_flops_per_token`).
  - `d2l.load_reference_model(step=None, dtype=torch.bfloat16, device='cpu', preset='r2') -> d2l.PackedGPT` (eval mode; `step=None` is the final step 6000; it downloads `DATA_HUB['lm-r2-config']` and `DATA_HUB['lm-r2-s{step}']` and loads them into a `PackedGPT` built from the config; allowed steps: 500, 1000, 2000, 3000, 4000, 4500, 4800, 5100, 5400, 5700, 6000; the T1 preset is `d2l.load_reference_model(step=1526, preset='t1')`, key `lm-r2t1-s1526`); the tokenizer is `d2l.r2_tokenizer()` from 3.6.
  - **Data artifacts (`DATA_HUB`)**, `uint16`/safetensors unless stated:

| Key | Contents | Size |
|---|---|---|
| `lm-r2-config` | the `PretrainConfig` of the R2 run, dataset revision, manifest `layout_hash`, git hash of the training script | 2 KB |
| `lm-r2-s{500,1000,2000,3000,4000,4500,4800,5100,5400,5700,6000}` | bf16 weights, `PackedGPT.state_dict()` (138,442,752 parameters × 2 bytes) | 276.9 MB each, 3.05 GB in all |
| `lm-r2-s5100-state` | fp32 weights + AdamW `m`, `v` (12 bytes per parameter), scheduler step, stream state, RNG states: the **branch point** of the stable phase | 1.66 GB |
| `lm-r2-train-log` | `jsonl.gz`: one record per step (loss, lr, grad norm, tokens/s, MFU), update/weight RMS every 50 steps, attention stats every 250, val every 250 | under 5 MB |
| `lm-r2-evals` | per-item 0/1 scores and per-domain NLL for each committed checkpoint on the five tasks (scores and item ids only; no item text, so the non-commercial SciQ is fetched at run time) | about 1–2 MB |
| `lm-r2t1-s1526`, `lm-r2t1-log` | the T1-preset checkpoint (25,173,376 parameters, bf16) and its log; needed by 4.6 and 4.4 | 50.3 MB |
| `lm-olmo2-1b-trajectory.json` | per-item scores and per-domain bits-per-byte for the OLMo 2 revisions below | 1–2 MB |
| `lm-fwedu-val-domains` | `jsonl.gz` with `text`, `domain`, `url_host` for documents totaling about 10M R2-tokenizer tokens (the file stores text, so BPB is tokenizer-independent), 2M per domain, from parquet file 012 | about 16 MB |

  - These about 4.8 GB live in a d2l-owned Hugging Face repository; each `DATA_HUB` entry is `('https://huggingface.co/<repo>/resolve/<commit>/<file>', sha1)` with a pinned commit, never `main` (`01_interfaces.md` §3); the repository also holds the R2 tokenizer (3.6). The R2 run is a one-off T3 job: its outputs are uploaded once, and readers never retrain it. A **core subset** of steps {1000, 3000, 5100, 6000}, `lm-r2-s5100-state`, the logs and the T1 files (about 2.8 GB) supports every downstream chapter; the other seven steps feed only the trajectory and averaging analyses of this section.
- **Models and data:**
  - Train/val data and tokenizer as in 4.1 (FineWeb-Edu `sample-10BT`, ODC-By 1.0; R2 tokenizer of 3.6). The train stream for R2 is Chapter 3's `lm-fwedu-manifest-4b` (about 8 GB for up to 4.0e9 tokens; 3,145,728,000 consumed, so about 79% of it); validation is `lm-fwedu-val`; the tokenizer is `lm-r2-tokenizer`.
  - Domains of `lm-fwedu-val-domains`: `wikipedia` (host ends with `wikipedia.org`), `edu`, `gov`, `org` (other `.org`), `other`; plus a sixth, `book`, built from `d2l.book_corpus()` with held-out chapters. If a bucket has fewer than 2M tokens in file 012, fill from file 011.
  - OLMo 2 (Apache-2.0): `allenai/OLMo-2-0425-1B` (1,484,916,736 parameters; vocabulary 100,352; context 4,096; fp32 safetensors, about 6 GB per revision, so budget about 48 GB for eight). Revisions on the Hub that exist (checked 2026-10-28): `stage1-step300-tokens1B`, `stage1-step10000-tokens21B`, `stage1-step100000-tokens210B`, `stage1-step500000-tokens1049B`, `stage1-step1000000-tokens2098B`, `stage1-step1500000-tokens3146B`, `stage1-step1907359-tokens4001B`, and `main` (after the stage-2 mid-training mix). `allenai/OLMo-2-1124-7B` (965 revisions) is the optional extension (bf16 weights 14.6 GB: too much for a 15 GiB slot once activations are added, but within a whole 24 GB GPU, so readers can run it; the build runs it as a T2 job).
  - Evaluation tasks: HellaSwag, ARC-Easy, PIQA, SciQ (`allenai/sciq`, CC-BY-NC-3.0: downloaded at run time only, per-item scores may be committed, item text may not), Winogrande (a 150M model is near chance on ARC-Challenge, OpenBookQA and MMLU, so they carry no signal at R2's scale; the notebook prints the chance level next to each task).
- **Packages:** `torch>=2.5`, `safetensors`, `transformers>=4.48` (OLMo 2), `torchao` not needed here.

#### Narrative arc

1. **The question.** Opening cell: the one-line config census of R2 (parameters 138.4M, embedding share 18.2%, 1.51 GFLOP per token, 3.15B tokens = 22.7 tokens per parameter) and a projection table (hours on 1 A100 and 8 A100 at MFU 35–45%) next to the A100 datasheet peak. Then three questions the logs must answer *before* the loss does: Is the run healthy? Is it efficient? Is it learning what we care about? The measured symptom is a pair of curves from the committed log: a smooth validation loss and a HellaSwag accuracy that is flat near chance for the first thousand steps (shape to be measured).
2. **The R2 specification and why each number.** `tab_lm_r2_spec` (below) with one line of justification per row, citing the book for each choice: width/depth (a ~140M GPT-2-small-to-medium shape; the 32,768-token vocabulary of 3.6 puts 18% of the parameters in the embedding, against 25% with GPT-2's), GQA 3:1 (cache size for Chapter 12, `sec_kv-cache`), SwiGLU/RMSNorm/QK-norm (`sec_transformer-block`), RoPE base 10,000 (a *small* base on purpose: 4.6 extends it), tied embeddings, context 4,096 (so that 4.6's 4K→16K extension is exactly the book's `subsec_pos-context-extension` setting), peak LR 6e-4 at 0.5M-token batches (the GPT-3 small-model setting of Brown et al., Table 2.1, as in build-nanogpt; the T1 LR probe in an exercise checks it), z-loss 1e-4 (PaLM), `1/sqrt(2L)` residual-branch init, token budget of 22.7 tokens per parameter (27.8 per non-embedding parameter; the Chinchilla ratio `sec_scaling-laws` quotes, with its caveats).
3. **Initialization and parameter groups; the first-step checks.** Cells: initial loss equals `ln V + d σ² / 2 = ln 32,768 + 0.154 = 10.40 + 0.15 = 10.55` within 0.1; gradient norm at step 0; logit RMS; update/weight RMS at step 1. These are the "is it wired right" checks before spending hours.
4. **The training loop.** Walk through `d2l.pretrain` in about 100 lines: WSD, accumulation with the global valid-token count (4.1 D3), clipping, bf16 autocast over fp32 master weights, `torch.compile`, checkpoints, resume from `(model, optimizer, step, stream state, RNG)`. A resume test: run 40 steps, save at 20, resume and compare steps 21–40 of loss to ≤ 1e-3 (bf16; nondeterminism noted) and the *batches* exactly.
5. **What to log, and what each curve is for.** One table `tab_lm_r2_logs`, columns: quantity, how computed, cost, which failure it catches first.
   - Loss (train, per-domain val), learning rate, gradient norm before clipping, fraction of clipped steps.
   - Update/weight RMS per module type (D3 explains the expected level and reads the gradient signal-to-noise from it).
   - Max attention logit and entropy per layer (`attn_logit_stats`), RMS of final hidden state and max output logit (4.3's early warnings).
   - Tokens/s and MFU (D2), data-wait time per step (4.1's starvation).
   - Downstream evals on checkpoints.
6. **MFU and where the missing FLOPs go.** D2 plus a breakdown cell using `torch.profiler` on one step of the T1 preset (Ada) and, in the committed log, of R2 (A100): time share of GEMMs, attention, norms/elementwise, optimizer, loss, data wait. Use `sec_perf_model`'s regimes to say which share is memory-bound.
7. **Reading the R2 run.** On the committed log: the loss curve with its three phases; the drop at the start of decay (`sec_scheduler`); per-domain curves (Wikipedia and `.edu` fall faster than `other`, if the log says so: to be measured); the update RMS by module; the effect of WSD's decay on downstream metrics.
8. **When do downstream metrics move relative to loss?** Plot accuracy against validation loss and against tokens. Define the comparison on bits per byte (`sec_language-model`) so R2 and OLMo 2 share an axis. Discuss: loss is smooth, task accuracy is a nonlinear, noisy function of loss (`sec_scaling-laws` cites the emergent-abilities discussion); the *item-level bootstrap interval* decides whether two checkpoints differ.
9. **Choosing and averaging checkpoints.** Candidates at the end of training: final (step 6000), stable-phase step 5100, uniform average of {4500, 4800, 5100} (LAWA/early weight averaging), the decayed branch. Compare on held-out val loss and the suite with paired bootstraps. Then the **selection-bias cell**: choose the best of the 11 checkpoints on one half of the items, report on the other half; show the winner's-curse gap. Say why a checkpoint chosen on the downstream suite cannot also be reported on it.
10. **Reading a fully open model's trajectory (OLMo 2 1B).** Load the eight revisions; score the same five tasks and the `lm-fwedu-val-domains` BPB (`d2l.load_hf` with `revision`). Figures: capability against BPB for R2 and OLMo 2 on one axis; OLMo 2's curve over tokens (log axis) with the stage-1/stage-2 boundary marked. Questions: which tasks saturate first, where the stage-2 mid-training (Dolmino mix) moves the curve, what the offset between R2 and OLMo 2 at equal BPB says about data and architecture (not a clean ablation: tokenizer, data mix, size and context all differ; say so). The default path loads `lm-olmo2-1b-trajectory.json`; two revisions are re-scored live as a spot check (to be measured: about 10 min).
11. **In production.** Re-express R2 as a TorchTitan job and compare the first 200 steps of loss; list what changes (FSDP2 sharding, `torch.compile` per block, async checkpointing, float8 option of 4.4). Name OLMo-core's trainer as the reference for checkpoint/restore across hardware failures (Chapter 6.8).

#### Derivations

**D1. Exact parameter and FLOP count of R2.**
*Statement.* For width `d`, `L` blocks, `H` heads of size `d_h = d/H`, `H_kv` key-value heads, SwiGLU width `f`, vocabulary `V`, context `T`, QK-norm on, no biases, tied head: parameters `P = L (2 d² + 2 d H_kv d_h + 3 d f + 2 d + 2 d_h) + V d + d`, and training FLOPs per token (PaLM convention) `F = 6 (L (2 d² + 2 d H_kv d_h + 3 d f) + V d) + 12 L d T`.
*Steps.*
1. Attention: `W_q`, `W_o` are `d × d`; `W_k`, `W_v` are `d × H_kv d_h`.
2. FFN: three `d × f` matrices; norms: two RMSNorm gains per block of size `d`, two QK-norm gains of size `d_h`.
3. Embedding `V d` is counted once for storage; the tied head matmul contributes `2 V d` forward FLOPs per token, hence the `6 V d` term; the lookup itself costs none.
4. Matrix multiplies cost 2 FLOPs per weight forward and 4 backward (`eq_six_nd`); attention scores cost `4 T d` per layer per token forward over the full `T × T` (the PaLM count; a causal kernel does half), times 3 for forward plus backward: `12 L d T`.
5. GQA changes only `P`, not `F`'s score term (scores are per query head).
*Check cells.* With the R2 config, `P = 138,442,752` equals `sum(p.numel())` exactly; `F = 1,509,949,440` FLOP/token at `T = 4096` (1,170,210,816 if causal-half is counted). Compare with `torch.utils.flop_counter.FlopCounterMode` on one forward+backward of 2 sequences: within 3% of either count (the cell prints both; which one the installed counter matches is to be measured) and with the profiler count of `sec_scaling-laws` (which omits fused-attention FLOPs).

**D2. MFU and the time projection.**
*Statement.* `MFU = (tokens/s · F) / (n_GPU · P_peak)`; the training time for `D` tokens is `D F / (n_GPU P_peak · MFU · η)` where `η` is the data-parallel efficiency (1 for one GPU).
*Steps.* (1) The numerator is the model FLOPs actually required (recomputation not counted); (2) `P_peak = 312e12` for A100 BF16 dense (datasheet) and `165e12` for RTX 4090 (`tab_gpu_specs`); (3) substitute `D = 3,145,728,000`, `F = 1.5099e9`: `C = 4.750e18` FLOP, giving 9.4 h (MFU 0.45) to 12.1 h (0.35) on one A100 and 1.3–1.7 h on 8 with `η = 0.9`; (4) the T1 preset: `C = 1.89e16` per 100M tokens, 3.8–6.4 min at MFU 0.5–0.3 on the 4090.
*Check cell.* Recompute the projection table from `PretrainConfig`; after a short measured run, solve for the MFU actually achieved and print the ratio to the profiler-based count (to be measured).

**D3. What update/weight RMS says about gradient signal.**
*Statement.* For AdamW with `β_1`, `β_2`, in steady state, a parameter whose gradient is `g_t = μ + σ ξ_t` (`ξ` i.i.d., unit variance) has an update `u = m̂/(√v̂ + ε)` with `E u² ≈ (μ² + κ σ²)/(μ² + σ²)`, `κ = (1-β_1)/(1+β_1)`. For pure noise `RMS(u) = √κ = 0.229` at `β_1 = 0.9`; for pure signal `RMS(u) = 1`.
*Steps.* (1) `m_t = (1-β_1) Σ β_1^k g_{t-k}` has mean `μ` and variance `κ σ²` (sum of squares of geometric weights: `(1-β_1)²/(1-β_1²)`); (2) `v_t` has mean `μ² + σ²` and is concentrated for `β_2 = 0.95`; (3) divide, drop `ε`; (4) the per-step relative update of a weight of RMS `w` is `lr · RMS(u) / w`: for `lr = 6e-4`, `w = 0.02`, noise-dominated, `6.9e-3`; (5) solving for the signal fraction `ρ = μ²/(μ²+σ²)` from a measured `RMS(u)`: `ρ = (RMS(u)² - κ)/(1 - κ)`.
*Check cell.* Simulate `g_t` for `ρ ∈ {0, 0.1, 0.5, 0.9}` over 2e6 steps; measured `RMS(u)` within 3% of the formula; recover `ρ` from the measured `RMS(u)` within 0.03 absolute. A quick check (done while writing this blueprint) gives 0.226 against 0.229 at `ρ = 0`.

#### From-scratch implementation

About 150 lines in all, on top of 4.1:

| Component | Contract | Lines |
|---|---|---|
| `wsd_lr`, `param_groups` | as exported | 18 |
| `flops_per_token`, `mfu` | D1, D2 | 14 |
| `attn_logit_stats` | explicit scores on one batch (no fused kernel), per-head max and entropy | 22 |
| `pretrain` | loop with accumulation by global valid tokens, clip, z-loss through `chunked_lm_loss`, update-RMS hook (snapshot of parameters every `log_every` steps), checkpoint and resume, optional `torch.compile` and DDP | 100 |

Numerical tests: (1) `wsd_lr` table at the fixture of A4 check 7; (2) `param_groups` decayed-parameter count for R2: all 2-D matrices of the blocks, `18 × (2·768² + 2·768·256 + 3·768·2048) = 18 × 6,291,456 = 113,246,208` parameters; (3) `pretrain` on a 2-layer toy for 30 steps with accumulation 1 vs 4 (same global batch): losses equal to 1e-5 in fp32; (4) resume test above; (5) `attn_logit_stats` max logit equals the max of the explicit softmax inputs of one layer computed independently; (6) initial loss and `RMS(u)` checks of narrative step 3.

#### In production

- **TorchTitan** (maintained PyTorch training system): reproduce the R2 architecture as a model config, run 200 steps on the same shards, compare loss to the from-scratch loop within bf16 noise (tolerance to be set from the seed-to-seed spread of 3 short runs). Use its FSDP2/compile/async-checkpoint paths as the Chapter 6 bridge. Verify configuration names against the current release when authoring (unverified here).
- **OLMo-core / OLMo trainer**: the instrumentation (per-module update norms, per-layer attention stats) and restart logic that R2's logging imitates.
- **Evaluation:** the Chapter 1 suite is the oracle; cross-check HellaSwag on `r2-s6000` with `lm-evaluation-harness` (acc_norm) within the paired-bootstrap interval, which validates `d2l.score_choices` on a model that was not loaded through `transformers`.

#### Headline experiment

- **Hypothesis.** (H1) R2's validation loss falls smoothly during the stable phase and drops by a visible margin when the LR decay starts. (H2) Downstream accuracy is a monotone function of validation BPB across R2's checkpoints and OLMo 2 1B's, with the pre-saturation region of each task near chance; the two models lie on offset curves rather than one curve (direction; the offset is to be measured).
- **Setup.** `r2_config('r2')`: PackedGPT 138.4M parameters, T = 4,096, batch 524,288 tokens, 6,000 steps (3,145,728,000 tokens), AdamW (0.9, 0.95), ε 1e-8, weight decay 0.1 on matrices, clip 1.0, z-loss 1e-4, peak LR 6e-4, WSD with warmup 300, stable to step 5,100, linear decay to 0 over steps 5,100–6,000, bf16 autocast with fp32 master weights, `torch.compile`; seed 0. Tier: the production run is T3 (8×A100 40 GB, DDP, micro-batch 8, accumulation 2) and the notebook analyzes its artifacts; the T2 live run is `r2_short`; the T1 live run is `t1`.
- **Procedure.** (1) Launch the run with `d2l.launch`, log as in `tab_lm_r2_logs`; (2) score the 11 committed checkpoints on the five tasks with `d2l.EvalSuite([...five ids...], n=None, seed=0).run(model, tok)` (full item sets; the run itself uses 1,000-item subsets every 500 steps for the live curve); (3) compute per-domain NLL and BPB of all R2 and OLMo 2 checkpoints on `lm-fwedu-val-domains`; (4) for adjacent checkpoints compute paired bootstrap intervals (`d2l.paired_bootstrap(...)`, reading `.mean` and `.ci` of the `BootstrapResult`) for each task, with `EvalResult.scores(task)` as inputs; (5) the selection-bias cell.
- **Metrics.** Validation loss and BPB per domain; per-task accuracy with 95% bootstrap intervals; gradient norm, update RMS by module, MFU, tokens/s.
- **Plots.** (i) `fig_lm_r2_loss`: x tokens (linear), left y train loss (smoothed, raw faint) and validation loss, shaded decay phase, per-domain validation as thin lines. (ii) `fig_lm_r2_monitors`: four stacked panels over steps: gradient norm; update RMS for `attn.qkv`, `attn.o`, `ffn`, `emb`; max attention logit; MFU. (iii) `fig_lm_r2_capability`: x validation BPB (reversed), y accuracy, one panel per task, R2 points with intervals and OLMo 2 points with intervals, chance line. (iv) `fig_lm_r2_olmo_tokens`: OLMo 2 1B BPB and accuracy against tokens, log x, stage boundary.
- **Expected outcome.** H1: `sec_scheduler` documents the abrupt end-of-run decrease under WSD; its size at 150M is to be measured. H2: Wen et al. (cited in `sec_scheduler`) and the OLMo 2 report support smooth loss with later task gains; no number is claimed before the measurement. OLMo 2 1B has two orders of magnitude more tokens and 10× the parameters, so its points sit at lower BPB and higher accuracy (direction; the overlap range is narrow and the notebook states it).
- **Noise check.** Item-level bootstrap intervals on every accuracy; the T1 preset trained with 3 seeds gives the seed spread of validation loss and the downstream accuracies at that scale (3 × about 8 min, run once and committed in `lm-r2t1-log`); differences between adjacent R2 checkpoints smaller than that spread are marked "not resolved".
- **T1 reduction.** `t1` preset (about 6 min), plus a 2-revision OLMo spot check.
- **Artifacts committed for T2+.** All of the R2 table above. T3 record: one run (`r2`, seed 0, 6,000 steps) on 8×A100 40 GB, run once; it commits the `lm-r2-config`, the 11 checkpoints, the `lm-r2-s5100-state` branch point, the training log and the evals of the table, plus the node's `nvidia-smi topo -m` and the measured data-parallel efficiency. Nothing was cut: the production run has no arms and no extra seeds, and the seed spread comes from the T1 preset (`lm-r2t1-log`). It fits 8× 40 GB.

#### Figures and tables

- `tab_lm_r2_spec`: architecture, parameters, tokens, optimizer, schedule, batch, hardware projections, checkpoint list.
- `tab_lm_r2_logs`: the monitored quantities (narrative step 5).
- `tab_lm_r2_projection`: hours at MFU 0.30–0.50 on 1 and 8 A100, cell from D2.
- `fig_lm_r2_loss`, `fig_lm_r2_monitors`, `fig_lm_r2_capability`, `fig_lm_r2_olmo_tokens`: as above.
- `fig_lm_r2_averaging`: validation loss and mean task accuracy of {5100, average(4500, 4800, 5100), 6000} with intervals.

#### Pitfalls and misconceptions

- Reporting MFU with a different FLOP convention than the hardware peak's (causal half vs full, recompute counted or not) changes it by 10–30%; state the convention.
- Comparing losses across tokenizers: use BPB.
- Evaluating a mid-schedule checkpoint of a cosine run as if it were a finished model; WSD's stable-phase checkpoints are also not finished models (they are branch points).
- Checkpoint selection on the evaluation set (the selection-bias cell).
- Near-chance accuracies at small scale carry no information; plotting them with large bars is noise-reading.
- Treating OLMo 2 1B vs R2 as an ablation of anything: five things differ at once.
- A "resumed" run whose scheduler step restarted at 0 (restoring the optimizer but not the step).

#### Summary (draft bullets)

- R2 is a 138.4M-parameter, 18-layer, GQA, QK-norm decoder trained on 3.15B FineWeb-Edu tokens with WSD; its parameter and FLOP counts are exact functions of the config.
- MFU is tokens/s times FLOPs per token over peak; the FLOP convention must be stated.
- Update/weight RMS reads the gradient signal-to-noise per module: `RMS(u) = √κ ≈ 0.23` for pure noise at `β_1 = 0.9`.
- Loss is smooth and downstream accuracy is a noisy, thresholded function of it; use bits per byte to compare across tokenizers and item-level bootstrap to decide whether two checkpoints differ.
- WSD's decay beats averaging stable checkpoints on held-out loss in the direction the schedule section predicts; selection on the evaluation items biases the winner.
- OLMo 2's public revisions let one read a real trajectory, including the jump from mid-training.

#### Exercises

1. **(check)** Why is `V d` counted once in `P` but contributes `6 V d` to `F`? Where would an untied head change each? *Sketch:* storage counts the shared matrix once, but both the lookup (free) and the head matmul (2 FLOPs per weight forward) use it; untying adds another `V d` to `P` and nothing to `F`.
2. **(check)** The log shows update RMS of 0.9 for `ffn` weights at step 3,000 and 0.25 for `attn.qkv`. What does D3 say about their gradient signal? *Sketch:* `ρ = (0.81 - 0.0526)/0.947 = 0.80` versus `ρ = (0.0625 - 0.0526)/0.947 = 0.01`: the FFN gradients are mostly consistent across steps, the QKV gradients noise-dominated; both are readings of the optimizer's view, not statements about importance.
3. **(derive)** Derive the data-parallel efficiency `η` of D2 for `n` GPUs when a ring all-reduce of the bf16 gradients (`2 P` bytes) overlaps a fraction `φ` of the backward pass (`sec_multi_gpu`). Evaluate it with the measured per-step compute time of the T1 preset and an effective bus bandwidth `B_eff` that you measure on a T2 pair (two A100 40 GB over an NVLink bridge); without a pair, use the datasheet's printed NVLink bridge and PCIe Gen4 rates as two assumed values and label them so (the build box runs no multi-GPU experiment). *Sketch:* the ring moves `2 (n-1)/n` times the buffer through each link, so `t_a = 2 (n-1)/n · 2P / B_eff`; the exposed communication is `(1-φ) t_a`, hence `η = t_c / (t_c + (1-φ) t_a)`; the exercise is to see that `η` falls with `n` only through the factor `(n-1)/n` and that `B_eff`, not `n`, is what the box changes.
4. **(derive)** For WSD with linear decay to zero over the last `δ` fraction of `N` steps, compute the learning-rate-weighted token count `Σ lr_t` and the fraction of it spent in the decay phase. Use it to argue how much "learning" a branch decay of 900 steps gets relative to the 5,100-step stable phase. *Sketch:* `Σ lr = peak (N(1-δ) + δ N/2)` plus warmup; the decay phase holds `δ/(2-δ)` of the total (about 8% for δ = 0.15), yet the loss improvement is large.
5. **(measure)** Predict, then measure with the T1 preset, how validation loss at 100M tokens changes for peak LR in {3e-4, 6e-4, 1.2e-3, 2.4e-3} (one seed each, plus 3 seeds at the best two). Where does the update RMS curve change shape? Report the interval for the best-vs-second-best difference.
6. **(measure)** Compare four ways to finish the T1 run from its step-1,297 stable-phase checkpoint: decay over 229 steps, decay over 57 steps, EMA of the last 100 steps with no decay, average of 4 checkpoints. Predict the order; measure validation loss with 3 seeds.
7. **(open)** Add a Muon variant (`torch.optim.Muon` for hidden matrices, AdamW for the rest, as in `sec_muon`) to `d2l.pretrain`, tune it on the T1 preset under a stated protocol, and decide whether the evidence supports adopting it for R2. State the per-optimizer tuning budget (`sec_muon` on matched tuning).

#### Further reading

- Papers: v5's [OLMo 2](https://arxiv.org/abs/2501.00656) (stability recipe, checkpoints, the mid-training stage; its [checkpoints](https://huggingface.co/allenai/OLMo-2-1124-7B)); [Olmo 3](https://arxiv.org/abs/2512.13961) (the current open recipe and trajectory); [Dolma](https://arxiv.org/abs/2402.00159) (the data behind OLMo); [OLMES](https://arxiv.org/abs/2406.08446) (how to score small-model evaluations; the task-selection argument); [SmolLM2](https://arxiv.org/abs/2502.02737) (small-model data-centric recipe); [PaLM](https://arxiv.org/abs/2204.02311) (the MFU definition and the FLOP convention used here; z-loss); [Early Weight Averaging meets High Learning Rates](https://arxiv.org/abs/2306.03241) and [Latest Weight Averaging](https://arxiv.org/abs/2209.14981) (checkpoint averaging, 4.2 narrative step 9); [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) (the data).
- Tutorials: [build-nanogpt](https://github.com/karpathy/build-nanogpt) (v5; the 0.5M-token batch recipe and HellaSwag-in-the-loop); the [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (monitoring and ablation practice); CS336 scaling lectures.
- Code: [OLMo](https://github.com/allenai/OLMo) and [OLMo-core](https://github.com/allenai/OLMo-core); [TorchTitan](https://github.com/pytorch/torchtitan); [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (cross-check).

#### Risks and open decisions

- *Wall-clock exceeds v5's "a few hours" for a single A100.* The full 3.15B-token R2 is 9.4–12.1 h projected on one A100 40 GB. Mitigation: the committed R2 comes from a T3 job; the T2 notebook runs the 1.05B-token `r2_short` (3.1–4.0 h) and analyzes the committed full-run artifacts. Fallback if T3 is unavailable: train `r2_short` alone and commit it as R2 (fewer tokens, about 6.9 tokens per parameter; downstream chapters still work). Open question for the editor.
- *Hosting 4.8 GB.* Decided: a d2l-owned Hugging Face repository holds all of it. The core subset (2.8 GB) is the fallback only for a mirror with a size limit; the averaging and trajectory analyses then use four checkpoints.
- *Chapter 1 harness* (`d2l.EvalSuite`, `d2l.EvalResult`) is canonical per `01_interfaces.md`; the only residual risk is the small-model task ids, with a 20-line adapter over `d2l.score_choices` as fallback.
- *OLMo 2 downloads (about 48 GB fp32 for eight revisions).* Fallback: the committed `lm-olmo2-1b-trajectory.json` and live evaluation of two revisions; if the Hub rate-limits, use three revisions.
- *Near-chance downstream tasks at 150M.* If the five tasks are all within chance bands for the early checkpoints, the "when do metrics move" figure is plotted on loss only, and the notebook says so; do not invent a signal.
- *Tokenizer.* If 3.6 prefers a different tokenizer for R2, the R2 spec's vocabulary, embedding share and token count in tokens change; the architecture does not.

### 4.3 Training instabilities

- **Label:** `sec_lm_instabilities` · **Tier:** T1 (no reduction needed; a `--quick` flag runs variants V0, V1, V3 at three learning rates) · **Budget:** about 10–15 min on one Ada GPU for 45 runs of 400 steps (about 0.27 PFLOP each at 8,192 tokens per step: 8–15 s per run, to be measured), peak memory under 8 GiB (the `'math'` attention backend at `T = 256` materializes `4 × 256 × 256` scores per layer, negligible) · **Codex units:** 5.9.
- **Builds on (book):**
  - `sec_practice`, "Additional Stability Methods": z-loss, QK-norm, QK-clip and the PaLM rewind-and-skip practice are *named* there; this section implements each and measures it. Not re-taught: clipping itself (`Clipped`) and the OPT logbook discussion.
  - `sec_transformer-block`, `subsec_qk-norm` and its exercise on QK-norm: the book's exercise runs Wortsman et al.'s protocol on `CharLM`; here the model is R2-shaped and the protocol adds z-loss, softcap and QK-clip. The `QKNormAttention` of that exercise is superseded by `PackedAttention(qk_norm=True)`.
  - `sec_attention-scoring-functions` (cited there): softmax saturation and its Jacobian. `sec_muon`: RMS matching and the MuonClip subsection, which owns `d2l.qk_clip_` (decided 2026-10-29, README item 19; a book-side edit): 4.3 imports it and measures it as variant V5, and does not re-derive or re-implement it. `sec_numerics`: bf16 spacing.
- **Builds on (this part):** 4.1 (`PackedStream` resume and `skip`; `PackedAttention` with `qk_norm`, `softcap` and the `'math'` backend; `chunked_lm_loss(z_loss=...)`); 4.2 (`d2l.pretrain` and its hooks, `d2l.attn_logit_stats`, `d2l.wsd_lr`, `d2l.r2_config`, the T1 token shards).
- **Exports:**
  - `d2l.softcap(x, c) -> Tensor`: `c * tanh(x / c)`.
  - Not exported here: `d2l.qk_clip_(model, stats, tau, alpha=0.5) -> dict`, owned by `sec_muon` (MuonClip subsection). 4.3 relies on this contract: in place, after an optimizer step; for every head whose `stats[layer]['max_logit'][head] > tau` (`S` is that value), multiplies the head's rows of `W_q` by `(tau/S)^alpha` and of `W_k` by `(tau/S)^(1-alpha)`; returns the per-layer scale factors applied; a no-op with a warning, returning `{}`, when `qk_norm=True` (D1 step 7). `stats` is the output of `d2l.attn_logit_stats` (4.2). The testbed has `H = H_kv`, so the per-head contract is all that 4.3 exercises.
  - `d2l.make_bad_batch(batch, kind='random', vocab=50257, seed=0) -> dict`: replaces `input_ids` by uniform random tokens (`'random'`) or by one repeated token (`'repeat'`), keeping labels consistent with the new tokens.
  - `d2l.lr_sensitivity(final_losses, l0) -> float`: the Wortsman et al. summary `mean over learning rates of (min(loss, l0) - best loss)`, with `l0 = ln V` the loss at initialization.
  - Data artifacts: `lm-inst-grid` (JSON: for every `(variant, lr, seed)` the loss trace every 10 steps, max attention logit per layer every 10 steps, max output logit, mean `log Z`, update RMS by module, and the final loss; about 3 MB) and `lm-inst-spike` (loss and gradient-norm traces of the injected-spike runs and replays; under 1 MB). Both are produced by the notebook's own T1 run; committing them lets the analysis cells run on a reference machine's numbers.
- **Models and data:** the instability testbed is a `PackedGPT` with `d = 256`, `L = 6`, `H = H_kv = 4` (head dimension 64), SwiGLU width 683 (`round(8d/3)`), tied embeddings, `T = 256`, vocabulary 32,768 (8.4M embedding plus 4.7M block parameters, 13.1M in all), RoPE base 10,000. Data: the 4.1 `lm-fwedu-shards-100m` shards (first 3.3M tokens suffice: 400 steps × 8,192 tokens). No external model.
- **Packages:** `torch>=2.5`; nothing else.

#### Narrative arc

1. **The question.** Opening cell: two runs of the same model, seed and data differing only in peak learning rate (3e-3 and 3e-2, no QK-norm, no z-loss, bf16 autocast, AdamW (0.9, 0.95), no weight decay, 400 steps). Print the final loss, the maximum attention logit `max |q·k|/√d_h` over layers and heads at steps 100/200/400, and the maximum output logit. The measured symptom is a pair of rows in which the loss of the high-LR run has left the smooth curve while the logit columns have grown by orders of magnitude (to be measured; fallback in Risks if no divergence is seen).
2. **Two places where scale escapes: attention logits and output logits.**
   - Attention logits: `s_{ij} = q_i·k_j/√d_h` with `q = W_q x`, `k = W_k x`; saturating softmax means near-zero Jacobian, collapsing attention entropy (cite Zhai et al., Wortsman et al.). Show per-layer max logit and mean entropy of the baseline over steps.
   - Output logits: cross-entropy is invariant to adding a constant to all logits; the model can drift `log Z` upward at no loss cost; in bf16 large logits lose precision (D2). Show max `|z|` and mean `log Z` over steps.
   - A small table `tab_lm_inst_failures` listing symptom → first monitor that moves → remedy (QK-norm, QK-clip, softcap for attention logits; z-loss, final softcap, fp32 head for output logits).
3. **The remedies, one at a time** (each is a few lines of code on top of `PackedAttention`/`chunked_lm_loss`):
   - QK-norm (`subsec_qk-norm`): per-head RMSNorm of `q` and `k`; what it bounds (D1, step 6).
   - z-loss (PaLM): add `λ (log Z)²` per token, `λ = 1e-4`.
   - Soft-capping (Gemma 2): `c tanh(s/c)` on attention logits (`c = 50`) and on final logits (`c = 30`), values as reported in the Gemma 2 report (verify when authoring); needs the `'math'` backend (or FlexAttention `score_mod`).
   - QK-clip (MuonClip, Kimi K2; `d2l.qk_clip_` from `sec_muon`, used as given): after each step, per head, shrink `W_q`, `W_k` if the max logit exceeds `τ`. Choose `τ` so that the baseline at 1e-2 clips on a fraction of steps (set `τ = 30` after looking at the baseline's logit scale at the first run; the K2 report uses 100 for a far larger model). QK-clip cannot be combined with QK-norm: with QK-norm the scale of `W_q`, `W_k` is normalized away (D1, step 7).
4. **Mapping the stable learning-rate range.** The grid, the variants and the stability criterion (below). Plot final loss against LR per variant; read the *width* of the stable band in half-decades and the Wortsman sensitivity; overlay the logit traces. Discuss which remedy widens the band on the high side and which only delays divergence.
5. **A loss spike, on purpose.** At the highest LR that is still stable for the full recipe (V3), at step 300 substitute a degenerate batch (`make_bad_batch`). Show the spike in loss and gradient norm, and what clipping at norm 1.0 does to it. Then the *diagnosis protocol* (PaLM): from the checkpoint at step 250 replay (a) the same batches (spike recurs: batch-caused), (b) the same batches with batch 300 skipped (no spike), and (c) from the checkpoint at step 50 replay with the bad batch (does the spike recur from an earlier state? The PaLM report found that the same batches from a different state did not recur; to be measured here). Table `tab_lm_inst_diagnosis` maps outcomes to causes (bad data; unstable state; hardware/numerics).
6. **Restart and skip, exactly.** Use `PackedStream.skip(300)` and `load_state_dict` to restart from step 250 with the bad batch removed; check that the restarted trajectory equals a *reference run in which the bad batch was never present* (the stream with `skip` set from the start): losses within bf16 nondeterminism (tolerance below) and batches exactly. This is the payoff of 4.1's pure-function addressing.
7. **What the field does at scale.** One paragraph (no new experiments): QK-norm in current open models (the book's `subsec_qk-norm` list), z-loss in PaLM and OLMo-style recipes, MuonClip in Kimi K2 (which reports no loss spikes in a 15.5T-token run, `sec_muon`), Gemma 2's soft-capping; and that recovery is still manual at scale (`sec_practice`: OPT logbook, PaLM rewind).

#### Derivations

**D1 (v5's derivation). Attention-logit growth follows from the growth of `W_Q`, `W_K`; clipping their product bounds it.**
*Setting.* One head with `W_Q, W_K ∈ R^{d × d_h}`; pre-norm input `x_i = γ ⊙ x̂_i` with `‖x̂_i‖_2 = √d` (RMSNorm without gain; the gain `γ` is bounded by `g = ‖γ‖_∞`). Logit `s_{ij} = x_i^T W_Q W_K^T x_j / √d_h`.
*Steps.*
1. **Bound.** `|s_{ij}| ≤ ‖x_i‖ ‖W_Q W_K^T‖_2 ‖x_j‖ / √d_h ≤ g² d σ_max(W_Q) σ_max(W_K) / √d_h`, using submultiplicativity of the spectral norm. Hence the logits on *every* input are bounded by the product of the two spectral norms; if the product grows, the bound grows.
2. **Why the factors grow together (gradient flow).** For `M = W_Q W_K^T` and loss gradient `G = ∂L/∂M`: `dW_Q/dt = -G W_K`, `dW_K/dt = -G^T W_Q`. Then `d(W_Q^T W_Q)/dt = -W_K^T G^T W_Q - W_Q^T G W_K = d(W_K^T W_K)/dt`, so `W_Q^T W_Q - W_K^T W_K` is constant: the two factors cannot grow apart (a conservation law of the bilinear form; it holds without weight decay and for any `G`).
3. **Positive feedback.** `dM/dt = -G W_K W_K^T - W_Q W_Q^T G`: the effective step on the logit matrix `M` is preconditioned by the Gram matrices of the factors. The larger the factors, the larger the logit movement per unit gradient, so growth accelerates (a rich-get-richer dynamics); weight decay or a clip is what breaks it.
4. **With Adam.** The update has entries of size `≈ η` per step independent of gradient scale (D3 of 4.2: `RMS(u) ≥ 0.23`); a persistent signal moves `W_Q`, `W_K` coherently, and the change of `M` per step is `ΔW_Q W_K^T + W_Q ΔW_K^T`, again scaling with the other factor's norm. This is the heuristic behind the observed logit growth at high LR (Wortsman et al.); it is not a proof for Adam.
5. **QK-clip (derived in `sec_muon`, MuonClip subsection).** The rescaling `W_Q ← (τ/S)^α W_Q`, `W_K ← (τ/S)^{1-α} W_K` for a measured batch maximum `S > τ`, and its guarantee that every logit on that batch scales by exactly `τ/S` (so the new batch maximum is `τ`), are derived there and not repeated. This section needs only the consequence for step 1: the product's norm drops by `τ/S`, so the bound drops by the same factor for all inputs; with `α = 1/2` the shrinkage is split evenly and the conserved difference of step 2 stays near zero.
6. **QK-norm.** Replace `q, k` by `q̂ = g_q ⊙ q/rms(q)`, `k̂` likewise: `‖q̂‖_2 ≤ √d_h ‖g_q‖_∞`, so `|s| ≤ √d_h ‖g_q‖_∞ ‖g_k‖_∞`, independent of `W_Q`, `W_K` and of the input. The bound moves to the learned gains, which are two vectors of length `d_h` per layer.
7. **Why the two do not combine.** Under QK-norm the scale of `W_Q`, `W_K` does not enter `s`; clipping the weights leaves the logits unchanged, so QK-clip would have to act on `g_q`, `g_k`.
*Check cell.* (a) Random weights and inputs: the bound of step 1 holds for 1e5 samples, and the ratio of the actual maximum to the bound is printed. (b) Gradient-flow conservation: a tiny bilinear model `L = ½‖X W_Q W_K^T X^T - T‖²` trained by small-step gradient descent (lr 1e-4, 2e4 steps, seed 0): the maximum drift of `W_Q^T W_Q - W_K^T W_K` is under 1e-2 relative to its initial norm and shrinks linearly with the step size (a quick check while writing measured 1.3e-3 at lr 1e-4). (c) QK-clip (a contract test of the imported `d2l.qk_clip_`, which 4.3's V5 depends on): after the clip the batch max logit equals `τ` (relative 1e-5), an immediate second clip is a no-op, and heads below `τ` are untouched bit-for-bit. (d) QK-norm: the empirical max logit never exceeds `√d_h ‖g_q‖_∞‖g_k‖_∞` over random inputs scaled by 1e3.

**D2. Output logits: shift invariance, the z-loss gradient and bf16 rounding.**
*Statement.* (i) `CE(z + c·1, y) = CE(z, y)`, so the cross-entropy gradient has no component along `1`; (ii) with `L_z = λ (log Z)²`, `Z = Σ_j e^{z_j}`, `∂L_z/∂z_j = 2 λ (log Z) softmax_j`, whose component along `1` is `2 λ log Z`; (iii) rounding a logit of magnitude in `[2^k, 2^{k+1})` to bf16 has absolute error up to `2^{k-8}`.
*Steps.* (1) `log Z(z + c1) = log Z(z) + c`, so `CE = log Z - z_y` is unchanged. (2) Differentiate `CE`: `softmax - onehot`, whose coordinates sum to 0, i.e. no force along `1`. (3) z-loss differentiates to the stated vector (since `∂ log Z/∂z_j = softmax_j`), summing to `2λ log Z`: it acts only along the direction CE ignores and pulls `log Z` to 0. (4) bf16 has 8 significant bits, so the spacing in `[2^k, 2^{k+1})` is `2^{k-7}` and round-to-nearest error is at most `2^{k-8}`; for `|z| ∈ [64, 128)` this is 0.25 nats, so differences between logits of a high-confidence token and its competitors are resolved only to that precision. (5) Hence unchecked drift of `log Z` costs precision in exactly the quantity the loss reads.
*Check cell.* (1) Shift invariance to 1e-6 in fp32; (2) the gradient of `L_z` against autograd to 1e-6; (3) the maximum rounding error of `z.bfloat16()` for `z ∈ U[64, 128)` is 0.25 (exactly, by construction) and for `U[8, 16)` is 0.03125; (4) the cross-entropy computed from bf16-rounded logits with a common offset of 100 added differs from the fp32 value by an amount the cell prints (to be measured; expected to be of order the rounding error).

#### From-scratch implementation

About 65 lines (reuse of 4.1/4.2 is heavy; `qk_clip_` is imported from `sec_muon`, so it costs no lines here):

| Component | Contract | Lines |
|---|---|---|
| `softcap` | `c·tanh(x/c)` | 3 |
| `make_bad_batch` | degenerate batches | 12 |
| `lr_sensitivity` | Wortsman et al. summary | 8 |
| `run_variant(variant, lr, seed)` (cell code, not exported) | builds the model (`PackedGPT`, flags per variant), runs `d2l.pretrain` for 400 steps with a hook that logs the stats every 10 steps and applies `qk_clip_` where required | 40 |

Numerical tests: (1) `softcap(x, c)` within `|x|/c · 1e-3` relative of `x` for `|x| ≤ 0.05 c`, absolute bound `< c`; (2) D1 checks (a)–(d); (3) D2 checks (1)–(3); (4) restart equivalence: the skip-restart run and the reference run with `skip` set from the start have batches that are exactly equal and loss traces whose maximum difference over steps 251–400 is within the *measured* difference between two identical re-runs of the reference (the nondeterminism floor, measured first and reported: the criterion is "within 2× the floor"); (5) with `qk_norm=True`, the imported `qk_clip_` returns an empty dict and warns (the guard that V1 and V3 rely on).

#### In production

- QK-norm ships in Qwen3 (R1, 1.2), OLMo 2 and Gemma 3; the from-scratch `PackedAttention` is the oracle for the library kernels. `torch.nn.attention.flex_attention` expresses soft-capping as a `score_mod` and keeps the fused path (the FlexAttention blog lists tanh soft-capping among its examples; verify when authoring), so Gemma-style capping does not require the `'math'` backend at scale.
- z-loss and per-step grad-norm/skip logic exist in OLMo's trainer configuration (name of the option to be verified at authoring); `torch.nn.utils.clip_grad_norm_` has an `error_if_nonfinite` option for the skip-on-NaN pattern.
- MuonClip: the Kimi K2 report is the reference; `torch.optim.Muon` (core since 2.9, `sec_muon`) plus `d2l.qk_clip_` (both in `sec_muon`'s MuonClip subsection) is a faithful miniature, which exercise 6 runs on this section's testbed.

#### Headline experiment

- **Hypothesis.** (H1) The baseline's stable band is narrower than the band of the full recipe (QK-norm + z-loss) on the high-LR side, and the baseline's divergence is preceded by growth of the max attention logit. (H2) QK-norm alone and z-loss alone each widen the band less than both together, because they address different escapes (attention logits and output logits). (H3) Soft-capping and QK-clip bound the attention logits by construction; whether that widens the band depends on whether attention logits were the binding failure (to be measured). Wortsman et al. report that both QK-norm and z-loss reduce LR sensitivity of small models and that the instabilities appear in them at high LR; no number is predicted here.
- **Setup.** Testbed above; variants `tab_lm_inst_variants`: V0 baseline; V1 +QK-norm; V2 +z-loss (1e-4); V3 +QK-norm +z-loss; V4 +soft-cap (attention 50, final 30); V5 +QK-clip (`τ = 30`, no QK-norm). AdamW (0.9, 0.95), ε 1e-8, no weight decay, clip off (clipping would hide the instability; its effect is exercise 5), 400 steps at constant LR after 20 warmup steps, batch 32 × 256 tokens, bf16 autocast, LR grid `{1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`, same data order and init across variants within a seed. Seeds 0 and 1 for V0, V1, V3; seed 0 for V2, V4, V5. 45 runs. Tier T1.
- **Procedure.** Run the grid; log every 10 steps; compute for each run the finite flag, the final loss (mean of the last 20 steps), the max attention logit trajectory, and `log Z`. Stability criterion (a *protocol parameter*): a run is stable iff all losses are finite and the final loss is at most `ℓ_ref + δ` with `ℓ_ref` the best final loss of V0 over the grid and `δ = 0.3`; the notebook checks that `δ` exceeds three times the seed-0 versus seed-1 difference on the shared cells, and widens it otherwise. Stable range: the largest `[lr_lo, lr_hi]` of grid points that are stable, in half-decades. Then the spike protocol (narrative step 5) for V3 at its highest stable LR.
- **Metrics.** Stable range and `lr_sensitivity` per variant; max attention logit at step 100 and at the end; max `|z|`; time to first non-finite loss; for the spike: loss and grad-norm traces, the outcome of the three replays.
- **Plots.** `fig_lm_inst_lr_sweep` (x log LR, y final loss clipped at `ln V`, one line per variant, seed-1 as hollow markers, shaded band for `ℓ_ref + δ`); `fig_lm_inst_logit_growth` (x step, y max attention logit on a log axis, V0/V1/V4/V5 at the LR where V0 diverges); `fig_lm_inst_spike` (loss and grad norm over steps 200–400 for the spiked run and the three replays); `fig_lm_inst_protocol` (decision tree of the diagnosis protocol).
- **Expected outcome.** As in the hypotheses (directions). "To be measured" for all magnitudes. If V0 does not diverge anywhere in the grid, the result is "the grid is below the instability threshold at this scale"; the fallback ladder is in Risks.
- **Noise check.** Two seeds for the three headline variants over the whole grid; the criterion's `δ` calibrated against the seed difference as described; the spike replays repeated three times to measure the nondeterminism floor.
- **T1 reduction.** None needed (this *is* T1). For T2+ none.

#### Figures and tables

- `tab_lm_inst_failures` (symptom → monitor → remedy), `tab_lm_inst_variants` (V0–V5 flags), `tab_lm_inst_diagnosis` (replay outcomes → cause), `tab_lm_inst_stable` (stable range and sensitivity per variant, from the cell that aggregates `lm-inst-grid`).
- `fig_lm_inst_lr_sweep`, `fig_lm_inst_logit_growth`, `fig_lm_inst_spike`, `fig_lm_inst_protocol`, and `fig_lm_inst_entropy` (attention entropy per layer over steps for V0 vs V1 at the divergent LR).

#### Pitfalls and misconceptions

- "z-loss and QK-norm are alternatives": they bound different quantities (D2 vs D1).
- Gradient clipping can mask the growth that precedes a spike; run the diagnostic with and without it.
- QK-clip with QK-norm is a no-op (D1 step 7).
- Looking only at the loss: the attention-logit and `log Z` monitors move earlier.
- Measuring the max logit with the fused kernel (it does not return scores): the explicit probe is mandatory and costs a forward pass.
- Reporting an LR range from one seed; ranges near the edge flip with the seed.
- A spike that disappears on replay from the same checkpoint is not "fixed": distinguish batch-caused from state-caused by replaying from an *earlier* state too.
- Skipping a batch changes the data order: the skipped run is a different run, and its stream state must record the skip (4.1).

#### Summary (draft bullets)

- Two escapes of scale cause most small-scale instabilities: attention logits (bounded by the product of the spectral norms of `W_Q`, `W_K`) and output logits (the `log Z` drift that cross-entropy ignores).
- Gradient flow conserves `W_Q^T W_Q - W_K^T W_K` and preconditions the logit dynamics by the factors' Gram matrices, which is the feedback behind runaway growth.
- QK-norm bounds the logits by the learned gains; QK-clip bounds them on the observed batch by rescaling weights; soft-capping bounds them by construction; z-loss pins `log Z`.
- Report a variant's stability as a range over a grid with a stated criterion and a seed check, not as one LR.
- Spike diagnosis is a replay protocol: same batch from recent and from earlier state, then skip; exact data addressing makes the skip reproducible.

#### Exercises

1. **(check)** A model has `‖W_Q‖_2 = 4`, `‖W_K‖_2 = 5`, `d = 256`, `d_h = 64`, `g = 1`. What is the worst-case logit bound of D1, step 1? Is it likely to be attained? *Sketch:* `256 · 20/8 = 640`; attained only if `x_i` and `x_j` align with the top singular directions of both factors, which is not the case for typical inputs, so the bound is a certificate, not a prediction.
2. **(check)** Why can't z-loss be replaced by clipping the logits to `[-30, 30]` after the head? *Sketch:* clipping kills the gradient for clipped logits (they stop learning) and does not remove the shift freedom; z-loss keeps all gradients and removes only the shift.
3. **(derive)** Show that for `α ∈ [0, 1]` the QK-clip rescaling changes `W_Q^T W_Q - W_K^T W_K` by `((τ/S)^{2α} - 1) W_Q^T W_Q - ((τ/S)^{2(1-α)} - 1) W_K^T W_K`, and that this vanishes in trace-sense iff `α = 1/2` when the factors are balanced. *Sketch:* expand the Gram matrices; with `‖W_Q‖_F = ‖W_K‖_F`, take the trace.
4. **(derive)** With one outlier input direction scaled by `ρ`, show that the logit scale grows like `ρ²` for an unnormalized head and is capped for QK-norm; relate to the outliers of Chapter 11. *Sketch:* `s` is bilinear in the two projections of the same direction, giving `ρ²`; QK-norm divides each projection by its RMS first.
5. **(measure)** Repeat the V0 and V3 sweeps with gradient clipping at 1.0. Predict whether the stable band widens for V0, and whether the max-logit growth is slowed or only the loss spikes are hidden. Report with the seed check.
6. **(measure)** Run `sec_muon`'s MuonClip on this section's testbed (4.3-specific extension; do not re-implement it): `torch.optim.Muon` for hidden matrices, AdamW for the rest (the `MultiOptimizer` of `sec_muon`), plus `d2l.qk_clip_` with `τ ∈ {10, 30, 100}`. Does Muon alone show larger max logits than AdamW at matched LR scales (the K2 report motivates the question; no direction is claimed)? Use two seeds.
7. **(open)** Design an automatic spike handler: detect (a rule on loss and grad norm), rewind to a checkpoint, skip a window of batches, and resume, with a logbook entry per intervention. Test it on injected spikes of three kinds (bad batch, LR too high, a NaN in one layer's weights) and state what fraction it handles without manual input.

#### Further reading

- Papers: v5's [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322) (the protocol of the headline experiment); [PaLM](https://arxiv.org/abs/2204.02311) (z-loss, rewind and skip); [Kimi K2](https://arxiv.org/abs/2507.20534) (MuonClip, QK-clip); [Muon is Scalable](https://arxiv.org/abs/2502.16982); added: [Stabilizing Transformer Training by Preventing Attention Entropy Collapse](https://arxiv.org/abs/2303.06296) (entropy collapse, the reparametrization), [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871) (the update-RMS mechanism of D1 step 4), [Gemma 2](https://arxiv.org/abs/2408.00118) (soft-capping), [Methods of improving LLM training stability](https://arxiv.org/abs/2410.16682) (a menu of fixes with measurements), [OLMo 2](https://arxiv.org/abs/2501.00656) (QK-norm and the embedding-decay instability of `subsec_adamw-what-not-to-decay`).
- Tutorials: the [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) sections on monitoring and debugging; the OPT logbook (cited in `sec_practice`).
- Code: [OLMo](https://github.com/allenai/OLMo) trainer (z-loss, skip logic); [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (Muon in a small-model setting; QK-clip itself is not in that repo).

#### Risks and open decisions

- *No instability appears at the chosen scale.* Fallback ladder, each rung documented in the notebook: (1) extend the grid to 3e-1; (2) set `β_2 = 0.999` (Molybog et al.'s regime) and remove warmup; (3) lengthen to 800 steps; (4) enlarge to `d = 512`. If none diverges, report the stable ranges as "all stable" for V0 and rely on the injected spike for the diagnosis protocol. Run this ladder once during authoring and commit the settings that work.
- *Nondeterminism of the replays.* `torch.backends.cudnn.deterministic` and atomics-free kernels are not guaranteed on Ada; the floor is measured and reported (`sec_repro`).
- *The `'math'` backend is slow if the grid is enlarged;* soft-capping through FlexAttention `score_mod` is the fallback.
- *Threshold choice `τ` and the stability criterion are protocol parameters*, not results; they are stated in the notebook and checked against seed noise.
- *`qk_clip_` ownership (decided 2026-10-29, README item 19):* `d2l.qk_clip_` lives in `sec_muon`'s MuonClip subsection (a book-side edit made separately). Until that edit lands, 4.3's V5 and exercise 6 need a local copy of the contract in the Exports note; delete it when `sec_muon` exports the name.

### 4.4 Low-precision training

- **Label:** `sec_lm_low_precision` · **Tier:** T1 on Ada (native FP8); every numerical-format experiment is also runnable on CPU or A100 through the *simulated* path, but only the Ada run measures speed · **Budget:** about 15 min on one Ada GPU (six arms of 20M tokens on a 75.5M-parameter model at about 0.53 GFLOP per token: 1.06e16 FLOP per arm, 2–4 min each at 30–50% of the 165 TFLOP/s BF16 peak, to be measured), peak memory under 14 GiB; the block-error derivation cells are T0 (seconds on CPU) · **Codex units:** 4.5 (format simulators), 5.6.
- **Where each experiment runs (stated once, repeated in the notebook as `tab_lm_lowp_where`):**

| Experiment | Device | Native or simulated |
|---|---|---|
| GEMM microbenchmark, BF16 vs FP8 (`torch._scaled_mm`, torchao) | Ada only | **native** FP8 tensor cores |
| FP8 training, torchao `tensorwise` | Ada only | native |
| FP8 training, torchao `rowwise` | Ada if the installed `torch._scaled_mm` supports row-wise scales on sm89, otherwise reported as unsupported and replaced by the simulated row-wise arm | native or simulated (checked at runtime, printed) |
| From-scratch `Fp8Linear`, fake-quant path (exact FP8 rounding via the `torch.float8_e4m3fn`/`e5m2` casts, BF16 matmul); delayed and current scaling; blockwise 1×128 / 128×128 scaling | any device (CPU, Ada, A100) | **simulated**; gives the numerics, never a speedup. A100 has no FP8 tensor cores, so on A100 only this path exists |
| BF16 weights vs FP32 master weights; stochastic rounding; 8-bit optimizer state | any | native dtypes |
| MXFP4 and NVFP4 (E2M1 values, 32- or 16-element blocks) | any | **simulated** by quantize-dequantize; neither A100 nor Ada has FP4 tensor cores; no speed or memory saving is measured or claimed |

- **Builds on (book):**
  - `sec_numerics`: FP8 E4M3/E5M2 `finfo`, "Below 16 Bits" (explicitly defers the FP8 training recipe to here), loss scaling (not re-taught).
  - `sec_hardware`: `tab_gpu_specs` (RTX 4090: 165 TFLOP/s BF16 dense, 330 TFLOP/s FP8 dense, 1.0 TB/s; H100 and B200 columns) and `fig_float_formats`.
  - `sec_memory_precision`: autocast, FP32 master weights, the bytes-per-parameter anatomy (`fig_memory_anatomy`).
  - `sec_fast_transformer`, "Further Optimization Techniques": states that FP8 tensor cores exist on the Ada GPUs and that the training recipe is separate; 4.4 delivers it (the contract table's "FP8 training recipe").
  - `sec_perf_model`: roofline and regimes (why quantization kernels are bandwidth-bound), `d2l.Benchmark`.
  - `sec_custom_layer`: custom `autograd.Function` (the from-scratch `Fp8Linear`). `sec_adamw`: optimizer-state bytes.
- **Builds on (this part):** 4.1/4.2 (`PackedGPT`, `d2l.pretrain` (the model is converted before the call: torchao's `convert_to_float8_training` or a loop that swaps `nn.Linear` for `d2l.Fp8Linear`), `d2l.chunked_lm_loss`, the token shards and `lm-fwedu-val`); 7.6 and Chapter 11 own the weight-only quantizers and may reuse `d2l.fake_quant_fp4` and `d2l.block_nmse` (not required; 4.4 owns the *training-time* simulators).
- **Exports:**
  - `d2l.fp8_quant(x, fmt='e4m3', granularity='tensor', scale_mode='current', state=None) -> (q, scale)`: `q` is a `torch.float8_*` tensor, `scale` the float32 multiplier such that `q.float() * scale ≈ x`; `granularity` in `{'tensor', 'row', 'block1x128', 'block128x128'}`; `scale_mode='delayed'` takes the scale from `state`, an amax history of length 16 that the call updates.
  - `d2l.Fp8Linear(nn.Linear)`: `__init__(in_features, out_features, bias=False, granularity='tensor', scale_mode='current', native=False)`; forward GEMM with `x`, `w` in E4M3, backward `dgrad`/`wgrad` with `dy` in E5M2 (`native=True` calls `torch._scaled_mm`, `native=False` quantizes then dequantizes into a BF16 matmul); `.stats()` returns saturation and underflow fractions and relative quantization error per operand since the last call.
  - `d2l.fake_quant_fp4(x, block=16, scale='e4m3', rounding='rtn', dim=-1, tensor_scale=True, generator=None) -> Tensor`: quantize-dequantize to the E2M1 grid `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` with per-block scales of kind `'fp32'`, `'e4m3'` (NVFP4) or `'pow2'` (MXFP4, E8M0); `rounding='sr'` is stochastic rounding.
  - `d2l.block_nmse(x, quantizer, block) -> float`: `‖Q(x) - x‖² / ‖x‖²` over blocks.
  - Data artifacts: `lm-fp8-gemm-sweep` (JSON: shapes, bf16 and fp8 times and TFLOP/s on the reference Ada box, about 20 KB) and `lm-lowp-runs` (JSON: per-arm loss traces, tokens/s, peak memory, operand statistics; about 1 MB).
- **Models and data:** preset `fp8` of `r2_config`: `PackedGPT` `d = 768`, `L = 8`, `H = 12`, `H_kv = 4`, SwiGLU width 2,048, `T = 1,024`, vocabulary 32,768, tied embeddings, QK-norm on; 75,511,552 parameters (`8 × 6,293,120 + 25,165,824 + 768`; the census cell prints and asserts it; all linear dimensions are multiples of 16, as `torch._scaled_mm` requires). Data: the 4.1 train shards (first 20M tokens) and `lm-fwedu-val`.
- **Packages:** `torch>=2.5` with FP8 dtypes; `torchao` (the `torchao.float8` training API: `convert_to_float8_training`, `Float8LinearConfig.from_recipe_name` with recipes `tensorwise`, `rowwise`, `rowwise_with_gw_hp`; documented on the [torchao training page](https://docs.pytorch.org/ao/main/workflows/training.html)); optional `transformer_engine` (not required; not on the build box by default).

#### Narrative arc

1. **The question.** Opening cell: a single square GEMM `16,384 × n × n` for `n ∈ {512, 1024, 2048, 4096}` in BF16 and in FP8 (GEMM only, `torch._scaled_mm`) on the Ada GPU, printed as TFLOP/s next to `tab_gpu_specs`' 165 and 330. The measured symptom is the FP8-to-BF16 throughput ratio versus `n`, set against the datasheet ratio of 2 (to be measured). Then the same shapes *including* the casts, where the ratio should fall at small `n` (D2).
2. **What FP8 training is: three GEMMs per linear layer.** `Y = X W^T`, `dX = dY W`, `dW = dY^T X`; which operand is cast to which format and with what scale (`tab_lm_lowp_recipes`):
   - Micikevicius et al.: E4M3 for forward operands, E5M2 for gradients (range over precision for `dy`);
   - DeepSeek-V3: E4M3 everywhere with fine-grained scaling (1×128 tiles for activations, 128×128 blocks for weights) and higher-precision accumulation promotion;
   - torchao `tensorwise` (one scale per tensor), `rowwise`, `rowwise_with_gw_hp` (weight gradient GEMM kept in higher precision).
   - What stays in higher precision in every recipe: embedding, head, normalization, attention scores/softmax, master weights, optimizer state.
3. **Scaling granularity and the scale's timing.** Tensorwise, rowwise, blockwise; *current* (dynamic) scaling takes `amax` of the tensor about to be cast (one extra memory pass), *delayed* scaling takes the max of an `amax` history from earlier steps (no extra pass; fails when `amax` jumps). Implement `fp8_quant` with all of these; plot the saturation and underflow fractions of the delayed scheme when the input scale jumps by 4× (a synthetic test) and during real training.
4. **From-scratch `Fp8Linear`.** Autograd function; simulated path; `native=True` path; the test that compares the two on Ada: same inputs, relative Frobenius difference of outputs (tolerance in "From-scratch").
5. **Accumulation, master weights and optimizer state.**
   - Accumulation precision: FP8 products accumulate in FP32 in the tensor core; DeepSeek-V3 reports that limited accumulation precision on Hopper motivated promoting partial sums every 128 elements (cited, not reproduced).
   - Bytes per parameter by recipe (`tab_lm_lowp_bytes`): FP32 master + FP32 `m`, `v` + FP32 grad (16 B); BF16 weights with FP32 master copy; BF16 moments (DeepSeek-V3's choice, per its report); 8-bit moments (`torchao.optim.AdamW8bit`).
   - **Why FP32 master weights matter in the decay phase:** D3 shows that BF16 weights without stochastic rounding discard updates whose relative size is below `2^-9 … 2^-8`, which happens exactly when WSD's LR approaches zero.
6. **Block-scaled 4-bit formats, simulated.**
   - E2M1 has 8 non-negative values; a per-block scale carries the range (`fig_float_formats` in `sec_hardware` has the layout). MXFP4: 32-element blocks, `E8M0` (power-of-two) scale; NVFP4: 16-element blocks, `E4M3` scale plus a per-tensor FP32 scale.
   - The NVFP4 training recipe of NVIDIA's pretraining paper ([2509.25149](https://arxiv.org/abs/2509.25149)), as this blueprint understands it (the author re-checks each element against the paper): 2-D (16×16) scaling for weights, random Hadamard transform on the inputs of the weight-gradient GEMM, stochastic rounding for gradients, round-to-nearest for forward operands, a minority of layers left in higher precision. Each element is implemented as a switch of `fake_quant_fp4` and the RHT cell, with the paper's exact list of kept layers copied from its text at authoring time.
   - Compute the error of each variant on activations and gradients captured from the training run (not on Gaussians only).
7. **Block-scaling error: theory then measurement** (D1): NMSE versus block size for Gaussian and heavy-tailed blocks; versus an outlier ratio `ρ`; versus scale format (fp32, E4M3, power of two).
8. **The experiment.** BF16, FP8 tensorwise (native), FP8 rowwise (native or simulated), from-scratch delayed-scaling FP8 (simulated), NVFP4-simulated on the FFN GEMMs only; throughput, loss, operand statistics. A paragraph on *why the end-to-end speedup is bounded* (D2, Amdahl).
9. **What happens at scale (no new experiment).** Fine-grained FP8 in DeepSeek-V3; the late-training instability of FP8 attributed to SwiGLU outlier amplification and the Smooth-SwiGLU fix (Fishman et al., [2409.12517](https://arxiv.org/abs/2409.12517); verify the claim wording when authoring); NVFP4's 12B-parameter, 10T-token run (the paper's abstract reports parity with an FP8 baseline).

#### Derivations

**D1 (v5's derivation). Quantization error of block scaling as a function of block size and in-block dynamic range.**
*Setting.* A block `x ∈ R^B`; a grid `G` of non-negative representable values (E2M1: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`) with maximum `Q = 6`, smallest normal `z_min = 1`, subnormal spacing `Δ_s = 0.5`, `m = 1` mantissa bit; real-valued scale `s = amax/Q`; round-to-nearest `R`; `q_i = s R(x_i/s)`.
*Steps.*
1. **Error model.** Condition on `z_i = x_i/s`. Treating the rounding error as uniform over the grid cell containing `z_i`, `E e_i² ≈ s² Δ(z_i)²/12` with `Δ(z)` the spacing of that cell (0.5 on `[0, 2]`, 1 on `[2, 4]`, 2 on `[4, 6]` for E2M1).
2. **Two regimes.** For `|z| ≥ z_min` the spacing is between `2^{-m-1}|z|` and `2^{-m}|z|`, so the error is *relative*: NMSE contribution between `2^{-2m}/48` and `2^{-2m}/12` of `x_i²` (0.005 to 0.021 for `m = 1`). For `|z| < z_min` the spacing is `Δ_s` and the error variance `s² Δ_s²/12` is *absolute*, independent of `x_i`: elements smaller than `amax/ (Q/z_min)` (here `amax/6`) are in this regime.
3. **Block NMSE.** `NMSE ≈ [Σ_{rel} κ_i 2^{-2m} x_i²/12 + n_abs s² Δ_s²/12] / Σ x_i²`, with `κ_i ∈ (1/4, 1]` and `n_abs` the number of absolute-regime elements. If every element is in the relative regime, NMSE is a constant of the format, independent of `B` and of the dynamic range.
4. **Dynamic range.** A block with one outlier `ρσ` and `B-1` elements `N(0, σ²)` has `s = ρσ/Q`; for `ρ` large enough that every other element is absolute-regime, `NMSE ≈ ρ² Δ_s² / (12 Q²)`, which is `ρ²/1728` for E2M1 — quadratic in the in-block dynamic range — until elements fall below `Δ_s/2` in grid units and round to zero (NMSE saturates at 1, for `ρ ≳ 24`).
5. **Block size.** For Gaussian blocks `amax ≈ σ √(2 ln 2B)` grows with `B`, so `s` grows and a larger share of elements moves toward the coarse low cells; and the block maximum is represented exactly, which matters at small `B`. The model of step 1 over-predicts the error at small `B` for that reason.
6. **Scale format.** An `E4M3` scale has relative spacing at most `2^{-3}`; rounding `s` to it changes `amax/s` by at most 6%; a power-of-two scale wastes up to a factor 2 of the grid's range (`s` rounded up) or clips the largest elements (`s` rounded down, as in the OCP MX convention where `amax/s ∈ [4, 8)` for E2M1), so its error is larger by up to a factor of the order of 2 to 4 in the absolute regime.
*Check cell.* (a) Gaussian blocks, `B ∈ {8, 16, 32, 128, 1024}`, 2e4 blocks each, fp32 scale: measure NMSE and compare with the step-1 model evaluated with the exact grid spacing; the ratio model/measured lies in `[1.0, 1.6]` and falls to within 1.1 for `B ≥ 128` (checked while writing: 1.58 at `B = 8`, 1.31 at 16, 1.18 at 32, 1.08 at 128, 1.05 at 1024). (b) The outlier sweep `ρ ∈ {4, 8, 16, 32, 64}` at `B = 16`: measured NMSE on the non-outlier elements within 10% of `ρ²/1728` at `ρ = 8, 16` (0.0375 against 0.0370 and 0.147 against 0.148 when checked), and saturating toward 1 at `ρ = 32, 64`. (c) The scale-format factorial `{B = 16, 32} × {fp32, E4M3, pow2 floor, pow2 ceil}` on Gaussian and Student-`t(3)` data: direction expected (and seen in a quick CPU check while writing): smaller blocks lower, E4M3 within a few percent of fp32 scales, power-of-two scales 1.3–2.5× worse; the notebook reports the numbers it measures and the same numbers for activations and gradients captured from the training run.

**D2. Where FP8 GEMMs win: a cost model with the casts, and Amdahl.**
*Statement.* For an `n × n × n` GEMM with BF16 peak `P`, FP8 peak `rP` (`r = 2` in `tab_gpu_specs` for the 4090), efficiency `η` for both, and cast kernels that read BF16 (2 B/element) and write FP8 (1 B/element) at bandwidth `W`: FP8 beats BF16 iff `(2 n³/(η P)) (1 - 1/r) > c · 6 n²/W`, where `6 n²` bytes is the traffic of casting the two operands and `c ∈ [0, 1]` is the fraction of it not hidden by fusion. Hence `n* = 3 c η P r / ((r - 1) W)`; with `P = 165e12`, `W = 1.0e12`, `r = 2`, `c = 1`, `η = 1`: `n* = 990`.
*Steps.* (1) BF16 time `t_b = 2 n³/(η P)`; (2) FP8 time `t_8 = 2 n³/(η r P) + t_cast`; (3) casts: reading `2n²` bytes of an operand and writing `n²` bytes, per operand, plus an `amax` pass for current scaling (another read of `2n²` bytes); (4) set `t_8 < t_b` and solve for `n`; (5) **Amdahl** for a whole model: with fraction `φ` of the step time in the cast-able GEMMs, the end-to-end speedup is at most `1/(1 - φ + φ/r)`; for the `fp8` preset, FLOP-weighted `φ ≈ 0.57` (block GEMMs 302M of 528M FLOP per token; the head and the attention scores stay in BF16), so `≤ 1.40` at `r = 2`.
*Check cell.* Evaluate `n*` and the Amdahl bound from the preset and `tab_gpu_specs`; compare `n*` with the measured crossover of the sweep; print both. (Expected: the measured crossover is at larger `n` than the model because `η < 1` and the casts are not free; to be measured. If FP8 never wins on this part with fp32 accumulation, the result stands as measured and the notebook says so.)

**D3. BF16 weights lose small updates.**
*Statement.* For `w` in `[2^k, 2^{k+1})`, BF16 has spacing `2^{k-7}`; round-to-nearest changes `w` by an update `δ w` only if `|δ| ≳ 2^{-9}` to `2^{-8}` (relative half-spacing, depending on where `w` sits in its binade); stochastic rounding keeps the expected change equal to `δ w` at the price of variance.
*Steps.* (1) Spacing `2^{k-7}`; relative spacing `2^{-7}` at the bottom to `2^{-8}` at the top of the binade; (2) RTN rounds an update smaller than half a spacing back to `w`; (3) for AdamW the relative update is `lr · RMS(u)/|w|` (4.2 D3): with `RMS(u) = 0.23` (noise-dominated), `|w| = 0.02`: it falls below `2^{-9}` when `lr < 1.7e-4`, i.e. below 28% of R2's peak 6e-4, which is reached about 70% of the way through the linear decay; (4) stochastic rounding is unbiased: `E[round(w + Δ)] = w + Δ`.
*Check cell.* `w ~ N(0, 0.02²)` stored in BF16; relative updates `δ ∈ {5e-4, 1e-3, 2e-3, 4e-3, 8e-3}`; print the fraction of weights that change under RTN (a step-like function of `δ` with the transition between `2^{-9}` and `2^{-8}`) and the mean relative change under stochastic rounding (within 5% of `δ` over 1e6 weights). Then a 200-step toy run with a decaying LR comparing FP32 master with BF16-RTN weights (loss gap grows as the LR falls; to be measured).

#### From-scratch implementation

About 140 lines:

| Component | Contract | Lines |
|---|---|---|
| `fp8_quant` | amax, scale (current or delayed history), cast to `float8_*`, saturating | 30 |
| `Fp8Linear` | `autograd.Function` with the three GEMMs; simulated and native paths; `.stats()` | 55 |
| `fake_quant_fp4` | block reshape, scale kinds, grid rounding by `bucketize` on midpoints, stochastic rounding, optional per-tensor scale | 35 |
| `block_nmse` and the RHT helper | | 20 |

Numerical tests: (1) `fp8_quant` followed by dequantization has relative error at most `2^{-4}` per element for elements in the normal range (E4M3: 3 mantissa bits, half-spacing `2^{-4}` relative) and never exceeds 448 in magnitude after scaling; (2) `Fp8Linear(native=False)` with BF16 inputs against an FP32 `F.linear`: relative Frobenius error of the output between `1e-2` and `1e-1`, dominated by input quantization (checked against the analytic NMSE of the two operand casts: within a factor 2); (3) on Ada: native vs simulated output relative difference at most `5e-3` (BF16 output rounding is `2^{-8} ≈ 3.9e-3`; any larger difference is a bug in the scale handling); (4) gradients: `dX`, `dW` of `Fp8Linear` against FP32 autograd within `1e-1` relative Frobenius (E5M2 has 2 mantissa bits); (5) `fake_quant_fp4` outputs lie exactly on the grid times the block scale (max deviation 0), `SR` is unbiased (mean over 1e5 draws within 1% of the input) and `RTN` is idempotent; (6) the RHT is orthogonal (`H H^T = I` to 1e-6) and leaves a matrix product unchanged when applied to both operands along the reduction dimension.

#### In production

- **torchao `float8`** (`convert_to_float8_training(model, config, module_filter_fn)` plus `torch.compile`): used for the native arms; the filter excludes the head and any layer whose dimensions are not multiples of 16. The torchao README reports 1.43–1.51× faster pre-training of Llama-3.1 70B and 405B with float8 on H100-class clusters (Nov 2024 entry), and its docs list per-recipe throughput tables. These are H100 numbers: the notebook reproduces the *method*, not the number.
- **NVIDIA Transformer Engine** (`te.Linear`, `fp8_autocast` with delayed or current-scaling recipes; MXFP8 and NVFP4 on Blackwell): the production counterpart of `Fp8Linear`; use it as the oracle on Hopper or Blackwell, not available on the build box (unverified here).
- **TorchTitan** exposes the same torchao float8 recipes in its trainer; DeepSeek-V3's kernels (DeepGEMM) are Hopper-only.
- Role of the from-scratch code: the simulated path is the numerical oracle for every recipe on any hardware; the native Ada run validates it.

#### Headline experiment

- **Hypothesis.** (H1) FP8 tensorwise training of the `fp8` preset for 20M tokens ends within the seed-to-seed spread of BF16's validation loss. (H2) The end-to-end tokens/s gain of FP8 on this GPU is below the Amdahl bound of D2 (about 1.40) and may be below 1. (H3) Delayed scaling shows measurably more saturation/underflow than current scaling around the LR warmup and decay, with a loss effect that is small at this scale. (H4) NVFP4-simulated FFN GEMMs show a visible loss gap to BF16 at this scale; the NVFP4 paper reports parity at 12B parameters and 10T tokens, so the gap here is not evidence against the recipe at scale (direction; magnitudes to be measured). The DeepSeek-V3 report states a relative loss error for its fine-grained FP8 recipe below a quarter of a percent in its ablations (verify against its Section 3.3 when authoring); that number is for a different model and recipe.
- **Setup.** Preset `fp8`, 20M tokens (1,221 steps × 16 sequences × 1,024), AdamW (0.9, 0.95) with weight decay 0.1 on matrices, clip 1.0, peak LR 1e-3, WSD (warmup 60, decay last 15%), z-loss 1e-4, FP32 master weights, `torch.compile` in all arms. Arms: A1 BF16 (seed 0); A2 BF16 (seed 1, noise reference); A3 torchao FP8 tensorwise (native); A4 torchao FP8 rowwise (native, else simulated and labeled); A5 from-scratch FP8 delayed scaling (simulated, with statistics); A6 NVFP4-simulated FFN GEMMs (RTN forward, SR on gradients, 16×16 weight blocks, RHT on wgrad inputs; attention and head in BF16). Same initialization and data order (stream seed 0) for A1, A3–A6. Tier T1.
- **Procedure.** Train each arm; every step log loss; every 50 steps log operand statistics (A5, A6); measure tokens/s as the median of 200 steps after compilation warm-up with `torch.cuda.synchronize` timing (the `d2l.Benchmark` convention); record `torch.cuda.max_memory_allocated`. Evaluate on `lm-fwedu-val` at the end. Run the GEMM sweep first and the arms after, in separate processes, so that compile caches do not interact.
- **Metrics.** Final validation loss; the mean over the last 200 steps of the paired training-loss difference to A1; A1–A2 difference as the noise reference; tokens/s ratio to A1; peak memory; per-operand saturation fraction (`|x/s| > max` before clamping), underflow fraction (`|x/s|` below half the smallest subnormal), and relative quantization error; the Amdahl bound.
- **Plots.** `fig_lm_lowp_gemm` (TFLOP/s versus `n`, three series plus datasheet lines at 165 and 330 and the model of D2); `fig_lm_lowp_loss` (training-loss difference to A1 versus step for each arm, with the shaded A1–A2 difference); `fig_lm_lowp_stats` (saturation and underflow fractions for each operand role in A5 versus step, with delayed and current scaling overlaid); `fig_lm_lowp_blockerr` (D1(a)–(c)).
- **Expected outcome.** As in the hypotheses (directions). Throughput and loss magnitudes: to be measured. If FP8 is not faster on the build box at any `n`, that is the result and D2's `r = 2` assumption is then questioned in the text.
- **Noise check.** A1 vs A2 sets the loss noise; throughput is the median of 200 steps with its interquartile range; a difference smaller than either is reported as "not resolved". No conclusion about FP8 quality from a single 20M-token run beyond "within noise at this scale".
- **T1 reduction.** None needed. **Not on A100.** On A100 the notebook runs only the simulated arms and the T0 derivation cells, clearly labeled.

#### Figures and tables

- `tab_lm_lowp_where` (above), `tab_lm_lowp_formats` (E4M3, E5M2, E2M1: max, smallest normal, smallest subnormal, relative spacing, from `torch.finfo` and the grid), `tab_lm_lowp_recipes` (per recipe: formats and granularity of each of the six GEMM operands, rounding, what is kept in higher precision), `tab_lm_lowp_bytes` (bytes per parameter by recipe), `tab_lm_lowp_results` (arms × final loss, paired difference, tokens/s ratio, memory).
- `fig_lm_lowp_gemm`, `fig_lm_lowp_loss`, `fig_lm_lowp_stats`, `fig_lm_lowp_blockerr`, `fig_lm_lowp_rtn` (D3: fraction of weights that change under RTN and mean change under SR versus relative update).

#### Pitfalls and misconceptions

- Comparing FP8 speed against an FP32 or TF32 baseline (`sec_memory_precision`'s warning): the baseline is BF16 with `torch.compile`.
- Reading datasheet ratios as speedups: GeForce parts run reduced-rate tensor math for some accumulation modes (`sec_hardware`'s note), and cast kernels are bandwidth-bound; hence the sweep.
- Timing without synchronization or before compilation.
- FP8 *storage* is not FP8 *training*: the weights' master copy, the optimizer state and the accumulations stay high precision.
- Delayed scaling silently clips after an activation jump; always log saturation.
- Calling a fake-quant run "FP4 training": it measures numerics only; no speed, no memory saving.
- Using `E5M2` for forward operands or `E4M3` for gradients without checking the range: `dy` spans more orders of magnitude than activations.
- Applying QK-clip-style weight rescaling or per-tensor scales to tensors with a few huge outliers (D1 step 4): the scale format is not the fix, the block size or an outlier transform is.

#### Summary (draft bullets)

- FP8 training casts the operands of three GEMMs per linear layer, E4M3 forward and E5M2 for gradients in the classic recipe; scale granularity (tensor, row, block) and timing (current, delayed) set the error and the overflow risk.
- Block-scaled error has a relative part set by the mantissa and an absolute part quadratic in the in-block dynamic range (`ρ²/1728` for E2M1); smaller blocks and an `E4M3` scale lower it; power-of-two scales raise it.
- Whether FP8 is faster depends on shape and bandwidth: casts make it lose below a crossover size, and Amdahl caps the end-to-end gain by the share of FLOPs in the converted GEMMs.
- FP32 master weights matter most late, when learning rates fall below the BF16 rounding threshold.
- MXFP4/NVFP4 are simulated here; numerics are measurable on any hardware, speed only on Blackwell.

#### Exercises

1. **(check)** Which of the six GEMM operands of one linear layer step use E5M2 in the classic recipe, and why? *Sketch:* `dy` in `dX = dY W` and in `dW = dY^T X`; gradients span a wider dynamic range than weights or activations, so the two extra exponent bits are worth more than the extra mantissa bit.
2. **(check)** A delayed-scaling tensor has an `amax` history `[1, 1, 1, 1]` and the current `amax` is 6. What happens after the cast to E4M3 with `s = 1/448 · 1`? *Sketch:* the scale was computed for `amax = 1`, so `6/s` is 2,688, which saturates at 448: the largest elements clip by a factor of 6; that is overflow, and the history catches up only after the step.
3. **(derive)** Extend D1 step 4 to `k` outliers of magnitude `ρσ` in a block of `B`: the NMSE of the remaining elements and the condition for saturation. *Sketch:* the scale is unchanged (set by `amax`); the non-outlier error is still `ρ²/1728` per element, and the NMSE over the whole block is diluted by the outliers' energy `k ρ² σ²` in the denominator, i.e. smaller; the *non-outlier* NMSE is the damaging quantity.
4. **(derive)** Derive the crossover `n*` for a *forward-only* inference GEMM where the weight is pre-quantized (no weight cast per call) and only `x` is cast. How does `n*` change, and why is FP8 inference attractive earlier than FP8 training? *Sketch:* the cast term drops from `3·2n²` to `3n²` bytes (activations only), halving `n*`; for `M ≫ n` (large batch) the weight cast is amortized.
5. **(measure)** Predict, then measure, the crossover `n*` for `M = 4,096` and `M = 65,536` rows on the Ada GPU for the native tensorwise path, with the casts included. Report as a plot and compare with D2's model.
6. **(measure)** Sweep NVFP4-simulated block size `B ∈ {8, 16, 32, 64}` and scale format `{E4M3, pow2}` on the FFN GEMMs of the `fp8` preset for 5M tokens, two seeds. Which factor moves the loss, and does the NMSE of D1 predict the order?
7. **(open)** Reproduce a miniature of the FP8 instability reported for long runs (SwiGLU outlier growth): track the maximum FFN-intermediate activation over a long single-seed run of the `fp8` preset with weight decay off, and decide whether the T1 budget can show anything. State what you would need to run it honestly (tokens, scale).

#### Further reading

- Papers: v5's [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (E4M3/E5M2, the first training recipe); [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (fine-grained scaling, promoted accumulation, what stays in BF16, BF16 optimizer moments); [Scaling FP8 training to trillion-token LLMs](https://arxiv.org/abs/2409.12517) (late-training instability and Smooth-SwiGLU); [Pretraining LLMs with NVFP4](https://arxiv.org/abs/2509.25149); [Microscaling Data Formats](https://arxiv.org/abs/2310.10537) (MX blocks, E8M0 scales); added: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (loss scaling, master weights) and [Training LLMs with MXFP4](https://arxiv.org/abs/2502.20586) (MXFP4 training with stochastic rounding and Hadamard transforms).
- Tutorials: the [torchao training docs](https://docs.pytorch.org/ao/main/workflows/training.html) (float8 recipes, benchmarks); the Smol Training Playbook's low-precision notes where present (verify).
- Code: [torchao](https://github.com/pytorch/ao) (`torchao.float8`); [Transformer Engine](https://github.com/NVIDIA/TransformerEngine); [TorchTitan](https://github.com/pytorch/torchtitan) float8 integration.

#### Risks and open decisions

- *FP8 is not faster on the build box* (cast overhead; GeForce fp32-accumulate rates). The notebook reports the measurement and keeps the numerics story. Fallback for a more convincing throughput section: run the GEMM sweep and the throughput arm on any H100 available, as an appendix (no H100 is in the compute tiers).
- *torchao row-wise FP8 may not support sm89,* and API names change between releases: the arm is guarded and falls back to the simulated row-wise `Fp8Linear`; pin the torchao version in the notebook environment. Verify the three recipe names against the installed release.
- *Compile time and recompilation* can dominate a 2–4 min run: measure after warm-up and cache compiled graphs per process.
- *NVFP4 recipe details* (which layers are kept in higher precision, the RHT size) must be copied from the paper at authoring time; the blueprint does not fix them. Marked for the author.
- *Open question for the editor:* whether the weight-only quantizers (7.6, Chapter 11) should import `fake_quant_fp4` from here or define their own; the registry reserves 7.6's names, not this one.

### 4.5 Continued pretraining and mid-training

- **Label:** `sec_lm_continued_pretraining` · **Tier:** T2 (reduced T1 config: the base is R2 instead of Qwen3-0.6B-Base, 12M tokens per arm, 3 arms) · **Budget:**
  - T2: four arms (A, B, C and the noise arm A′) of 80M tokens each on Qwen3-0.6B-Base at `T = 2,048`: 4.985 GFLOP per token (6 × 596M matmul parameters + the attention term `12 L (H d_h) T` with `H d_h = 2,048`) gives 3.99e17 FLOP per arm, 0.8–1.0 h at 45–35% of the 312 TFLOP/s A100 peak, 3.2–4.1 h for four arms, plus about 40 min of suite evaluations (to be measured); peak memory to be measured, design target ≲ 30 GiB against about 38 GiB usable on the 40 GB card, at micro-batch 8 × 2,048 with `chunked_lm_loss` (596,049,920 parameters × 16 bytes of fp32 weights, gradients and two AdamW moments is 8.9 GiB; the rest is activations; the full-vocabulary logits of 16k tokens in fp32 would be 9.9 GB without chunking). Fallback if the measured peak exceeds 38 GiB: micro-batch 4 with accumulation 32 (same global batch).
  - T1: R2 base, 12M tokens per arm at `T = 1,024` (1.00 GFLOP per token, 1.2e16 FLOP per arm, about 2.5–4 min each at 50–30% of the 4090's 165 TFLOP/s), peak under 10 GiB.
  - Codex units: 5.8.
- **Builds on (book):**
  - `sec_scheduler`: warmup, cosine and WSD; `subsec_scheduler-branching` for the stable-phase branch. The noise-ball argument of `sec_sgd` is used in D2.
  - `sec_adamw`: weight decay choices, optimizer state memory for a 0.6B model (about 9.6 GB in fp32).
  - `sec_practice`: EMA and averaging (optional).
  - `sec_scaling-laws`: the data-repetition remark (Muennighoff et al.) when R3 is repeated.
  - `sec_mdl-statistics`: the cluster (document-level) bootstrap used for loss differences.
  - Not re-taught: optimizers, schedules, the Chinchilla fit.
- **Builds on (this part):**
  - 4.1: `d2l.PackedStream` with `eos_id=151643` and `uint32` shards, `d2l.PackedGPT.adopt`, `d2l.chunked_lm_loss`.
  - 4.2: `d2l.pretrain`, `d2l.wsd_lr`, `d2l.param_groups`, the R2 checkpoints (`d2l.load_reference_model(step)`), the suite and the checkpoint-comparison cells.
  - 1.1–1.5: `d2l.load_hf`, `d2l.EvalSuite`, `d2l.score_choices`, `d2l.paired_bootstrap`.
  - 2.4: `d2l.book_corpus(chunking='sections')` (R3).
  - `d2l.load_qwen3(model_id='Qwen/Qwen3-0.6B-Base')` (1.2) returns a `d2l.Qwen3GPT` whose attention modules are `d2l.Qwen3Attention` with `W_q, W_k, W_v, W_o, q_norm, k_norm` and the attributes `head_dim, num_heads, num_kv_heads, rope_theta` (1.2's contract), which is what `PackedGPT.adopt` reads. If the loaded model fails `adopt`'s name check (a 1.2 regression), the fallback is to train the Hugging Face `Qwen3ForCausalLM` directly with the same loop, packed by `PackedStream` and passing `position_ids` (document masks then need Flash-Attention 2's packed-sequence path; without it the arms train without document masks and the notebook says so).
- **Exports:**
  - `d2l.mix_streams(streams, weights, seed=0) -> MixStream`: a deterministic mixture of `PackedStream`s. In step `s` the counts `(n_k(s))_k ~ Multinomial(B, w)` are drawn from `default_rng([seed, s])` (they sum to `B`), the sequences come from stream `k`'s own cursor at `c_k(s) = Σ_{s'<s} n_k(s')` (a cumulative-sum table computed once from the seed and the weights); `batch(step)` is a pure function and the state is the step (answers 4.1 exercise 7 for fixed weights).
  - `d2l.forgetting_gain(base, run, ...) -> dict`: per-checkpoint domain gain `L_d(base) - L_d(run)` and forgetting `L_g(run) - L_g(base)` with document-level bootstrap intervals.
  - Data artifacts: `lm-pes2o-12m` (12M R2-tokenizer tokens of peS2o full texts, Chapter 3 shard format, `uint16`, about 24 MB; T1 domain data), `lm-fwedu-replay-12m` (about 24 MB; T1 replay data from FineWeb-Edu parquet file 005, document-disjoint from R2's train and val files), `lm-pes2o-val` (2M tokens, 4 MB), `lm-cpt-runs` (JSON: logs, per-document losses for the domain and general validation sets, per-item suite scores for every arm, scores only with no benchmark item text (SciQ is CC-BY-NC-3.0 and is fetched at run time); about 5 MB). The T2 token shards (Qwen tokenization, `uint32`, about 80M × 4 bytes per arm) are regenerated by `data/tokenize_cpt.py`, not hosted.
- **Models and data:**
  - `Qwen/Qwen3-0.6B-Base` (Apache-2.0; 596,049,920 parameters; 28 layers, `d = 1024`, 16 query heads and 8 key-value heads of dimension 128, SwiGLU width 3,072, vocabulary 151,936, tied embeddings, RoPE base 1,000,000, context 32,768; `torch_dtype` bf16), pinned by revision hash at authoring time. Qwen3-0.6B-Base tokenization: EOS `<|endoftext|>` = 151,643, so shards are `uint32`.
  - Technical corpus: `allenai/peS2o`, version `v2`, `s2orc` (full-text open-access papers; ODC-By, verified 2026-10-29 from the dataset's metadata; files `data/v2/train-0000x-of-00020.json.gz`, about 1.6 GB each; a `validation` split exists). peS2o derives from S2ORC, whose README notes that the original S2ORC was released under a non-commercial licence and that the current S2ORC is ODC-By 1.0 (<https://raw.githubusercontent.com/allenai/s2orc/master/README.md>); `lm-pes2o-12m` and `lm-pes2o-val` are hosted with ODC-By attribution to peS2o and S2ORC, and the dataset card records that history. Use the first file for training (about 90M tokens are drawn from it), and the validation split for 2M held-out tokens. No field-of-study metadata is available in the dump, so "technical" means scientific full text; say so in the notebook.
  - R3: `d2l.book_corpus()`; train/held-out split by section so that held-out sections are never seen (every fifth section, deterministic).
  - General proxy and replay: FineWeb-Edu `sample-10BT`, parquet file 005 for replay, file 013 for general validation (as in 4.1/4.2).
  - Evaluation tasks (Chapter 1 suite): HellaSwag, ARC-Easy, ARC-Challenge, PIQA, Winogrande.
- **Packages:** `torch>=2.5`, `transformers>=4.51` (Qwen3), `safetensors`.

#### Narrative arc

1. **The question.** Opening cell: Qwen3-0.6B-Base's loss on three held-out sets: held-out R3 sections, peS2o validation, FineWeb-Edu validation, each in nats per token and in bits per byte (tokens are Qwen tokens; the three numbers are the baseline to beat and not to forget). Then a one-line experiment on 3M tokens of peS2o at LR 1e-4 that *raises* FineWeb-Edu loss visibly (the measured symptom of forgetting; to be measured).
2. **Why continue instead of restart.** Cost arithmetic with `sec_scaling-laws`: 80M tokens is a few millionths of the base model's pretraining budget (Qwen3 reports tens of trillions of tokens; no exact figure is quoted here). When it works (new domain, new language, new length), and what it is *not* (instruction tuning, Chapter 7).
3. **What changes between stage 1 and a continued run.** Four dials: data distribution, learning-rate path, replay share, optimizer state (restarted: first and second moments cold). A short table `tab_lm_cpt_dials`.
4. **Re-warming and re-decaying.**
   - Start from a model whose LR ended near zero; raising the LR to `η` creates a transient loss bump in the *old* distribution (D2: the noise-ball height is proportional to `η`), which re-decay heals.
   - WSD changes the picture: a stable-phase branch point needs no re-warm. The T1 arm C uses R2's step-5100 checkpoint to show this.
   - The notebook's schedule: linear warmup (10 steps at T2, 9 at T1), cosine to 10% of peak; schedule held fixed across arms so that only peak LR varies.
5. **Replay.** Mixing a fraction `r` of general data (`d2l.mix_streams`); why it helps (D1, the quadratic picture); the accounting question: compare arms at equal total tokens or at equal *domain* tokens? Arm B uses `r = 0.25` at equal total tokens and the notebook also reports the equal-domain-token comparison from the logged trajectory (arm A at 60M domain tokens against arm B at 80M total).
6. **Measuring forgetting against gain.** The three evaluation sets and the suite; domain gain `ΔL_d` and forgetting `ΔL_g` defined relative to the base; document-level paired bootstrap; the frontier plot. Warn about the single-seed limitation and the noise arm.
7. **Mid-training stages in released models.** Table `tab_lm_cpt_stages` (qualitative columns: stage, what changes in data, LR state, purpose) for OLMo 2 (stage 2 on the Dolmino mix with LR decayed to zero, model soups), SmolLM3 (three pretraining stages, then long-context and reasoning mid-training), Llama 3 (annealing on high-quality data, long-context stages) and Qwen3 (general, reasoning-heavy, then long-context stages). Use the OLMo 2 1B `stage1` versus `main` revisions from 4.2's trajectory to *measure* what the final stage does to each task (no new evaluation cost: reuse `lm-olmo2-1b-trajectory.json`). Point to 4.6 for the long-context stage and to Chapter 7 for what follows.
8. **In production.** What a continued run looks like in TorchTitan or OLMo-core (load an HF checkpoint, reset the optimizer, change the data mix, new LR schedule), and the pieces this section's loop shares.

#### Derivations

**D1. Forgetting and gain in a quadratic model; why replay can beat early stopping.**
*Setting.* General loss `L_g(θ) = ½ (θ-θ_g)^T H_g (θ-θ_g)` with minimizer `θ_g` (the base model), domain loss `L_d(θ) = ½ (θ-θ_d)^T H_d (θ-θ_d)`, `H_g, H_d` positive definite. Define gain `G(θ) = L_d(θ_g) - L_d(θ)` and forgetting `F(θ) = L_g(θ) - L_g(θ_g)`. Continued training with replay fraction `r` minimizes `L_r = (1-r) L_d + r L_g`.
*Steps.*
1. **Replay solution.** `θ_r = ((1-r) H_d + r H_g)^{-1} ((1-r) H_d θ_d + r H_g θ_g)`.
2. **One dimension** (`H_g = h_g`, `H_d = h_d`, `Δ = θ_d - θ_g`). Let `u = r h_g / ((1-r) h_d + r h_g) ∈ [0, 1]`. Then `θ_r - θ_g = (1-u) Δ`, `F = ½ h_g Δ² (1-u)²`, `G = ½ h_d Δ² (1-u²)`.
3. **The frontier.** Normalizing by the maxima `F_max = ½ h_g Δ²`, `G_max = ½ h_d Δ²`, `G̃ = 2 √F̃ - F̃`, independent of `h_g`, `h_d`, `Δ` and `r`: concave, with infinite slope at `F̃ = 0` (`dG̃/dF̃ = 1/√F̃ - 1`). The first bit of domain gain costs forgetting only *quadratically*, because `L_g` is flat at its minimum.
4. **Early stopping, one dimension.** Gradient flow on `L_d` from `θ_g`: `θ(t) - θ_g = (1 - e^{-h_d t}) Δ`, i.e. `u = e^{-h_d t}`: the same frontier. In one dimension replay and early stopping select different points of one curve.
5. **Two or more dimensions.** Take `H_g = diag(10, 0.1)`, `H_d = I`, `θ_g = 0`, `θ_d = (1, 1)`. Early stopping moves both coordinates at the same rate and pays `½ · 10 x²` for coordinate 1. Replay anchors coordinate 1 (`u_1 = 10 r/((1-r) + 10 r)` large) while coordinate 2 (`u_2 = 0.1 r/((1-r) + 0.1 r)` small) moves freely. At equal forgetting replay has more gain.
6. **Reading.** Replay helps to the extent that the domain gradient has components along directions where the general loss is sensitive *and* components along directions where it is not; it buys the second without paying for the first. This is a picture, not a theorem about transformers; the experiment measures whether the frontier of arm B lies above that of the lower-LR arm C.
*Check cell.* (a) 1-D: the replay curve equals `G̃ = 2√F̃ - F̃` to 1e-9 for three values of `(h_g, h_d, Δ)`; the early-stopping path equals it too. (b) 2-D example: at `F ∈ {0.05, 0.2, 0.5, 1.0}` replay's gain exceeds the path's (checked while writing: 0.54 vs 0.19, 0.66 vs 0.36, 0.76 vs 0.53, 0.84 vs 0.69 in units of the base model's domain loss excess); assert replay ≥ path for all `F` on a grid.

**D2. Why re-warming raises the old loss: the noise-ball height scales with the learning rate.**
*Statement.* For SGD on `L = ½ h θ²` with per-step gradient `h θ + ξ`, `Var ξ = σ²/b` (batch `b`), the stationary excess loss is `E[L] = η σ² / (2 b (2 - η h)) ≈ η σ² / (4 b)` for `η h ≪ 1`, and the relaxation rate of the transient is `η h` per step.
*Steps.* (1) `θ_{t+1} = (1 - η h) θ_t - η ξ_t`; (2) stationarity `E θ² = η² (σ²/b)/(1 - (1-η h)²) = η σ²/(b h (2 - η h))`; (3) multiply by `h/2`; (4) the transient after a change from `η_0` to `η` relaxes with rate `η h`, so the excess loss at the end of a warmup of `w` steps is `≈ η σ²/(4 b) (1 - e^{-η h w})`-shaped (plus the initial excess), proportional to the peak LR; (5) re-decay shrinks `η` and the excess with it, at the rate set by the schedule (the WSD "river valley" picture of `sec_scheduler` is the high-dimensional analogue).
*Check cell.* Simulate 1-D SGD for `η ∈ {0.01, 0.05}`, `σ = 1`, `b = 1`, `h = 1` over 4e5 steps: the measured stationary mean of `½ h θ²` within 3% of the formula (checked while writing: 0.00248 against 0.00251 and 0.01284 against 0.01282). Then the same measurement on the T1 arms' general loss: the bump after warmup divided by peak LR across arms C vs A (to be measured; the formula predicts only the *direction* of proportionality, since the real noise scale `tr(HΣ)` is unknown).

#### From-scratch implementation

About 80 lines on top of 4.1/4.2:

| Component | Contract | Lines |
|---|---|---|
| `mix_streams` | seeded multinomial counts per step, cumulative cursors, pure `batch(step)` | 30 |
| `forgetting_gain` | per-document losses on the base and on each checkpoint, cluster bootstrap | 25 |
| tokenization script `data/tokenize_cpt.py` | peS2o and FineWeb-Edu to Qwen- or R2-tokenized shards through `d2l.write_shards` (dtype by vocabulary) | (script, not counted) |
| arm driver (cell code) | config for A/B/C/A′, `d2l.pretrain` with the schedule above, checkpoint at 25/50/75/100% of the run for the trajectory | 25 |

Numerical tests: (1) `mix_streams` over 10,000 steps with `B = 128`: realized fractions within three binomial standard errors `3 √(w (1-w)/(B · steps))` of the weights; counts sum to `B` each step; resume at an arbitrary step gives identical batches; (2) after `PackedGPT.adopt` on Qwen3-0.6B-Base, logits on 4 fixed sequences equal the `d2l.load_qwen3`/Transformers logits within the BF16 tolerance of A1 (`sec_gpt`'s loading check; tolerance as in 1.2); (3) the per-document losses on the *base* model reproduce the opening cell's three numbers exactly; (4) the cluster bootstrap on a synthetic two-model comparison with a known shift of 0.05 nats recovers it with an interval that contains 0.05 and excludes 0.

#### In production

- **TorchTitan:** load an HF Qwen3 checkpoint (its conversion tooling), set a new LR schedule and data mixture, and resume with a fresh optimizer state; the loop of this section is the oracle for the first 50 steps (loss within bf16 noise).
- **OLMo-core / OLMo:** public recipes for annealing and mid-training stages (data mixes, LR to zero); **Hugging Face + Flash-Attention 2 packed sequences** as the lightweight route (position-id based packing); **LLaMA-Factory/axolotl**-style continued-pretraining recipes exist but hide the choices this section exposes (names not verified here).
- The suite and the bootstrap of Chapter 1 are the oracle for "did the base model survive".

#### Headline experiment

- **Hypothesis.** (H1) Without replay, domain loss falls and general loss rises, and a 10× larger peak LR gives a larger gain *and* more forgetting (arms A vs C). (H2) Replay at 25% reduces forgetting at a small cost in domain gain relative to A, and the (forgetting, gain) point of arm B lies above the line through the base and arm C's point (D1's two-dimensional picture; to be measured, and not guaranteed). (H3) The general loss shows a transient bump after re-warming that heals during the decay (D2). Direction from the sources: Gupta et al. find that re-warming and re-decaying with a larger LR adapts more at the price of more forgetting; Ibrahim et al. report that re-warming, re-decaying and a small replay share recover most of the performance of retraining at scale; Gururangan et al. find domain-adaptive pretraining helps domain tasks. No magnitudes are claimed.
- **Setup (T2).** Qwen3-0.6B-Base, `T = 2,048`, global batch 262,144 tokens (128 sequences; micro-batch 8, accumulation 16), 305 steps (79,953,920 tokens), AdamW (0.9, 0.95), weight decay 0.1 on matrices, clip 1.0, bf16 autocast with fp32 master weights, schedule: warmup 10 steps, cosine to 10% of peak. Arms: A peak LR 1e-4, no replay; B peak LR 1e-4, replay 25% (so 60M domain tokens); C peak LR 1e-5, no replay; A′ = A with another data-order seed (noise arm). Domain stream: 90% peS2o, 10% R3 train sections (R3 repeated about 4 times; say so). Same initialization (the base) and, for A, B, C, the same data order of the domain portion.
- **Setup (T1).** R2 base; `T = 1,024`, batch 65,536 tokens, 183 steps (12M tokens); arms: A from step 6000 (decayed), peak LR 6e-4 (full re-warm to R2's peak), no replay; B the same with 25% replay; C from step 5100 (plateau), constant LR 6e-4 then linear decay over the last 15%, no replay (no re-warm). Domain data `lm-pes2o-12m`; general validation `lm-fwedu-val`; replay `lm-fwedu-replay-12m`.
- **Procedure.** Evaluate the base; train each arm, saving the weights at 25/50/75/100% of the steps; evaluate each saved checkpoint on the three loss sets (per-document losses stored) and the suite (final checkpoints only at T2; all at T1).
- **Metrics.** `ΔL_d`, `ΔL_g` (nats per token on the same tokenization) with document-level cluster bootstrap intervals (paired: same documents across arms); per-task accuracy change with paired item bootstrap (`d2l.paired_bootstrap`); the general-loss bump after warmup and its LR proportionality.
- **Plots.** `fig_lm_cpt_frontier` (x `ΔL_g`, y `ΔL_d`, trajectories of each arm with the 25/50/75/100% points, final points with intervals, the line from the base through arm C); `fig_lm_cpt_curves` (two panels: domain and general validation loss versus step for all arms, with the LR schedule on a twin axis); `fig_lm_cpt_suite` (change in accuracy per task and arm with intervals); `fig_lm_cpt_quad` (D1: the 1-D frontier and the 2-D replay-vs-path curves).
- **Expected outcome.** As in the hypotheses (directions). Magnitudes and the sign of the replay-versus-lower-LR comparison: to be measured.
- **Noise check.** Arm A′ at T2 and three seeds of arm A at T1 (3 × about 4 min) give the seed spread of `ΔL_d`, `ΔL_g` and the suite; document-level bootstrap intervals cover evaluation-set noise; an arm difference smaller than the seed spread is reported as "not resolved".
- **T1 reduction.** The R2-based version above, same code (`--base r2` flag).
- **Artifacts committed.** `lm-cpt-runs`.

#### Figures and tables

- `tab_lm_cpt_dials` (data, LR path, replay, optimizer state: what each does and its cost), `tab_lm_cpt_stages` (mid-training stages of OLMo 2, SmolLM3, Llama 3, Qwen3; qualitative columns, each row tied to its report), `tab_lm_cpt_results` (arms × `ΔL_d`, `ΔL_g`, suite change, with intervals).
- `fig_lm_cpt_frontier`, `fig_lm_cpt_curves`, `fig_lm_cpt_suite`, `fig_lm_cpt_quad`, `fig_lm_cpt_stage2` (OLMo 2 1B: accuracy and BPB around the stage-1 to stage-2 boundary, from 4.2's artifact).

#### Pitfalls and misconceptions

- Evaluating domain gain and forgetting on different tokenizations or different documents across arms.
- Counting replay tokens as domain tokens (the equal-total versus equal-domain-token comparison).
- Re-using the base model's final-LR optimizer state, or none: moments are cold after a restart, so the first steps are effectively larger; warmup covers this.
- Restoring a *decayed* checkpoint and continuing at the old peak LR without re-warm: an immediate loss spike.
- Reading a single-seed difference below the seed spread as a result.
- Calling the continued model "better" because the domain loss fell; report the general loss and the suite alongside.
- Repeated R3 data (4 passes) overfits a small corpus; report held-out R3 sections separately from peS2o.
- Forgetting a different tokenizer's `uint16` limit: Qwen needs `uint32` shards.

#### Summary (draft bullets)

- Continued pretraining is a short, low-cost move along a trade-off between domain gain and forgetting; the base model's general loss is flat at its minimum, so small moves cost little.
- Re-warming raises the learning rate and therefore the noise-ball height on the old distribution; re-decaying heals it; WSD plateau checkpoints avoid the re-warm.
- Replay anchors the directions the general loss is sensitive to and can beat simply lowering the learning rate when those directions differ from the domain's; this is measurable on the (forgetting, gain) plane.
- Report both losses, the suite and noise-aware intervals, at equal total and at equal domain tokens.
- Mid-training stages in released models are the same dials at larger scale: data upsampled toward quality, LR decayed to zero, then later stages for length and reasoning.

#### Exercises

1. **(check)** In D1 step 3, why is the frontier concave and why does that matter for the practice of stopping early? *Sketch:* `dG̃/dF̃ = 1/√F̃ - 1` is decreasing, so the marginal gain per unit of forgetting falls: the first steps are cheap, the last expensive; stopping early captures most of the gain at little forgetting.
2. **(check)** A continued run restores R2's step-6000 checkpoint (LR ended at 0) with fresh AdamW state and a constant LR of 6e-4 from step 0, without warmup. What do you expect in the first 20 steps and why? *Sketch:* cold second-moment estimates make the first Adam steps large (`m̂/√v̂` close to ±1 for every coordinate), at a high LR; D2 adds the noise-ball bump; expect a loss spike, healed only if the LR is lowered.
3. **(derive)** Extend D1 to a mixture with *two* domains and one general loss, and show the replay solution is again a `H`-weighted average. What changes in the frontier picture? *Sketch:* `θ = (Σ_i w_i H_i)^{-1} Σ_i w_i H_i θ_i`; the frontier becomes a surface in `(F, G_1, G_2)`; replay moves along the weights simplex.
4. **(derive)** For D2, compute the excess general loss at the end of a linear warmup from `η_0 = 0` to `η` over `w` steps, in the 1-D model, and show it is `O(η²)` when `η h w ≪ 1` and `O(η)` when `η h w ≫ 1`. *Sketch:* integrate the variance recursion: `v_{t+1} = (1-η_t h)² v_t + η_t² σ²/b`; for short warmups `v ≈ Σ η_t² σ²/b ∝ η² w`, for long ones it reaches the stationary `η σ²/(b h (2 - η h))`.
5. **(measure)** At T1, sweep the replay share `r ∈ {0, 0.1, 0.25, 0.5}` at equal total tokens (R2 step 6000, peak 6e-4, one seed, plus 3 seeds at `r = 0` and `0.25`). Predict the forgetting curve's shape from D1, then plot forgetting and gain against `r` with intervals.
6. **(measure)** Branch from R2's step 5100 versus restore step 6000 for the same continued run (peak 6e-4, 12M tokens, 3 seeds). Predict which has less bump and less final forgetting; report the paired differences.
7. **(open)** Design the data mix for a mid-training stage that adds long-context capability to R2 without losing short-context quality, using the three dials of this section; state how you would decide the long-document share (link to 4.6) and the replay share, and how you would detect that you broke something.

#### Further reading

- Papers: v5's [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964) (domain-adaptive pretraining, the transfer gains), [Simple and Scalable Strategies to Continually Pre-train LLMs](https://arxiv.org/abs/2403.08763) (re-warming, re-decaying, replay at scale), [OLMo 2](https://arxiv.org/abs/2501.00656) (the Dolmino mid-training stage, soups); added: [Continual Pre-Training of LLMs: How to (re)warm your model?](https://arxiv.org/abs/2308.04014) (the learning-rate re-warming study this section's H3 follows); [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (repeating a small domain corpus; cited in 4.7); [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (annealing and long-context stages).
- Tutorials: the [SmolLM3 blog](https://huggingface.co/blog/smollm3) (a documented multi-stage recipe, mid-training included); the Smol Training Playbook's sections on data mixing across stages; Chapter 3's mixture section (3.4) for how stage mixes are chosen.
- Code: [OLMo](https://github.com/allenai/OLMo) and [OLMo-core](https://github.com/allenai/OLMo-core) (stage configs); [TorchTitan](https://github.com/pytorch/torchtitan); [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (the base checkpoint).

#### Risks and open decisions

- *Wall-clock.* Four 80M-token arms on Qwen3-0.6B-Base are 3.2–4.1 h of training plus evaluation on one A100 40 GB (projections). Fallback: drop A′ (noise from the T1 seeds only) or cut each arm to 40M tokens. The T1 path is always available.
- *`d2l.load_qwen3` internals*: fixed by 1.2's `Qwen3GPT` contract, and `adopt` checks the names. Fallback for a 1.2 regression: the Hugging Face model with the same loop, or R2 for the whole section.
- *peS2o has no field metadata and is not code or math:* "technical" is scientific prose. A code/math corpus with a permissive licence would broaden the study; Chapter 3's FineMath (ODC-By) and permissively licensed code pools are the candidates, and any other corpus follows the part's licence rule (permissive licences ship with attribution, the rest is downloaded at run time).
- *R3's size.* If R3 has fewer than about 2M tokens, repetition above 4 passes is not used: the R3 share falls below 10% and the remainder goes to peS2o.
- *License.* peS2o (ODC-By; derived from S2ORC, which was non-commercial in its first release, and is ODC-By 1.0 now per its README) and FineWeb-Edu (ODC-By) shards are hosted with attribution in the manifest and the dataset card; Qwen3 (Apache-2.0) and OLMo 2 (Apache-2.0) are unrestricted.
- *Hosting `uint32` shards* is avoided by regenerating them; the T1 slices are hosted.

### 4.6 Long-context training

- **Label:** `sec_lm_long_context` · **Tier:** T2 (reduced T1 config: the T1-preset checkpoint `r2t1`, 1K → 4K, same code, flag `--base r2t1`) · **Budget:**
  - T2: four trained arms of 100M tokens at `T = 16,384` from R2 step 6000 (trained at 4,096). FLOPs per token at 16K are 3.55 G (`6 × 138.4M + 12 L d T`, PaLM convention; 2.35× the 1.51 G at 4K), so 3.55e17 FLOP per arm: 0.7–0.9 h at 45–35% of the 312 TFLOP/s A100 peak (long-sequence MFU is likely lower, to be measured), 2.8–3.6 h for four arms, plus about 30 min of evaluation. Global batch 262,144 tokens (16 sequences), micro-batch 2, 381 steps. Peak memory to be measured (fused causal attention is `O(T)`); design target ≲ 30 GiB against about 38 GiB usable on the 40 GB card, at 32,768 tokens per micro-batch, the same as 4.2's. Fallback if the measured peak exceeds 38 GiB: micro-batch 1 (one 16,384-token sequence) with accumulation 16 (same global batch of 16 sequences).
  - T1: `r2t1` (25.2M parameters, trained at 1,024), extension to `T = 4,096` (`s = 4`, the same factor as 4K → 16K), 16M tokens per arm (244 steps of 65,536 tokens; 0.30 GFLOP per token, 4.8e15 FLOP per arm, about 2 min each); eight runs (PI, YaRN × 3 seeds, NTK-aware, YaRN-progressive × 1 seed) plus evaluation: about 15–20 min (to be measured); `--quick` uses one seed.
  - Codex units: 2.4, 5.8, 15.4.
- **Builds on (book):**
  - `sec_positional-information`: `eq_rope-def`, `subsec_pos-context-extension` (names PI and YaRN, reports that rescaling alone raises perplexity and a brief fine-tune recovers it, no frequency-wise derivation), the extrapolation experiment and ALiBi/NoPE. 4.6 supplies what that section leaves out: the per-frequency scale factors, NTK-aware and YaRN in code, and the training recipe.
  - `sec_attention-at-scale`: online softmax and fused kernels (why 16K training is feasible); `sec_kv-cache`: KV bytes per token (R2's GQA gives 18,432 B/token, 302 MB at 16K).
  - `sec_scaling-laws` (`eq_six_nd` and its attention-term remark) for D2. Not re-taught: RoPE, PI as an idea, FlashAttention.
- **Builds on (this part):** 4.1 (`PackedStream` for 16K sequences with `shards` of the long-data mix, `PackedAttention(rope_inv_freq=..., rope_mscale=...)`); 4.2 (`d2l.pretrain`, `d2l.load_reference_model(step=6000)`, `d2l.r2_tokenizer`, `lm-r2t1-s1526`); 4.5 (`d2l.mix_streams` for the long/short mixture); 1.4–1.5 (`d2l.score_choices`, `d2l.paired_bootstrap`); 1.1 (`d2l.load_hf` for the RULER cross-check). 12.x and Chapter 6.5 (context parallelism) reuse the 16K checkpoints.
- **Exports:**
  - `d2l.rope_inv_freq(d_head, base=10000.0, scheme='none', s=1.0, L=4096, alpha=1.0, beta=32.0) -> Tensor[d_head/2]`: `scheme ∈ {'none', 'pi', 'ntk', 'yarn'}`; returns the angular frequencies `θ_m` used as `angle = pos · θ_m` (float32).
  - `d2l.yarn_mscale(s) -> float`: `0.1 ln(s) + 1` (the YaRN paper's fit, Eq. 15); passed as `rope_mscale`.
  - `d2l.make_recall_task(tokenizer, length, depth, n_pairs=None, n_candidates=8, n_items=200, seed=0) -> dict`: key–value recall items; `depth ∈ [0, 1]` is the relative position of the queried pair; returns prompts and candidate values for `d2l.score_choices`.
  - `d2l.long_context_gain(model, windows, short_len, long_len, min_gap) -> dict`: per-distance-bucket mean of `NLL_short - NLL_long` on repeated-rare-word tokens (D2 of the experiment protocol below).
  - Data artifacts: `lm-r2-16k-pi`, `lm-r2-16k-yarn` (bf16 weights of the two best T2 arms by the effective-context criterion, 276.9 MB each: Chapter 12 uses one of them); `lm-pg19-test` (64 held-out PG-19 test books truncated to 16,384 tokens, `uint16`, 2 MB); `lm-pg19-30m` (T1 long training data, 60 MB); both PG-19 shards are hosted with Apache-2.0 attribution to `deepmind/pg19` (public-domain books); `lm-lc-evals` (JSON: per-window and per-item scores of every arm, about 3 MB).
- **Models and data:**
  - R2 (4.2) and `r2t1`. Tokenizer: the R2 tokenizer (3.6).
  - Long data: PG-19 (`deepmind/pg19`, Apache-2.0; the parquet conversion `emozilla/pg19` is the practical source; public-domain books, about 28,000 train books, long documents of tens of thousands of tokens) as the natural source of long-range dependencies; the peS2o `s2orc` full-text papers with at least 8K tokens as a second long source (ODC-By; documents longer than 16K tokens are rare, so used as a minority). Short data: FineWeb-Edu replay. Mixture of each batch: 50% long (PG-19 and peS2o), 50% short documents packed to `T` with document masks (4.1); the fraction is a protocol parameter (exercise 7).
  - RULER cross-check: `Qwen/Qwen3-1.7B` (native context 32,768; Apache-2.0) with the [RULER](https://github.com/NVIDIA/RULER) single-needle tasks at 4K/8K/16K/32K; KV cache at 32K is 3.8 GB (28 layers × 8 KV heads × 128 × 2 × 2 bytes per token = 114,688 B), which fits a T1 slot.
- **Packages:** `torch>=2.5`; `transformers>=4.50` for the `rope_utils` oracle; `ruler` dependencies only for the optional cross-check.

#### Narrative arc

1. **The question.** Opening cell: R2 at 4K context evaluated on 16K-token PG-19 windows; print the negative log-likelihood by position bucket (512, 1K, 2K, 4K, 8K, 16K). The measured symptom is a flat curve up to 4K and a jump after it (to be measured). A second row: the same model with RoPE frequencies rescaled by PI at inference only (no training): the curve past 4K improves, the curve before 4K worsens (to be measured; the book's `subsec_pos-context-extension` reports the second effect for the character model).
2. **What a longer context is made of.** Three separable things: a positional mechanism that does not break beyond the trained range (this section's first half), training on sequences that need the extra range (second half), and an evaluation that tells accepted length from used length (third). Accepting a longer tensor is not evidence of using distant information.
3. **Frequencies and wavelengths** (D1). Plot each RoPE pair's wavelength against the 4K training window for R2; classify pairs by rotations `r_m = L/λ_m`: 23 of 32 pairs complete at least one rotation within 4K, 9 do not.
4. **The four schemes in code.** `rope_inv_freq` with `'none'`, `'pi'`, `'ntk'`, `'yarn'`; the per-pair scale factor plot (`fig_lm_lc_scales`) and the phase-overhang plot (`fig_lm_lc_overhang`); YaRN's attention temperature through `rope_mscale`. Verify against `transformers`' `ROPE_INIT_FUNCTIONS` (the library knows the types `default`, `linear`, `dynamic` (NTK-aware), `yarn`, `longrope`, `llama3`).
5. **Zero-shot rescaling.** Apply each scheme to R2 at inference without training and evaluate at 16K. No gradient steps: cheap, immediate, and it separates "the positional mechanism alone" from "the mechanism plus training".
6. **Training for length.**
   - Cost: D2 (attention FLOPs share rises from 45% at 4K to 77% at 16K), hence *progressive* schedules (spend tokens at the cheaper lengths first).
   - Data: documents with real long-range dependencies versus concatenated short documents (Fu et al.: upsample long documents of a domain, do not concatenate unrelated ones; Gao et al. (ProLong): keep a short-data share, evaluate beyond perplexity). How to *check* that a data source has long dependencies: the long-context gain statistic of the protocol below, computed on the data itself.
   - Recipe: switch `rope_inv_freq`, reuse `d2l.pretrain` with `T = 16,384`, restart WSD (warmup 20 steps, constant, last 15% linear to 0 at peak LR 1e-4 for 381 steps; a protocol choice, compared with the continuing-pretraining dials of 4.5).
7. **Measuring effective context.** Three families, each at lengths `{4K, 8K, 12K, 16K}` (T1: `{1K, 2K, 3K, 4K}`):
   - **E1** perplexity by position bucket on held-out PG-19 (the cheap diagnostic, and the one that misleads: the LongPPL study shows perplexity can look fine while tasks that need distant information fail);
   - **E2** synthetic recall: key–value lines, query one key, choose among 8 candidate values by likelihood (chance 12.5%), grid over length and depth (relative position of the queried pair);
   - **E3** natural long-range dependence: on PG-19 test windows, tokens that are repeat occurrences of rare words whose previous occurrence is at least `min_gap` tokens earlier; compare their NLL with the full 16K context against the NLL with only the last 4K tokens (a sliding window): the *long-context gain* by distance bucket.
   - Effective context per family: the largest length at which the lower 95% bootstrap bound of the metric clears a stated threshold (recall accuracy at least 0.5; gain positive). The thresholds are protocol parameters, reported with the result.
8. **Production-grade evaluation.** What RULER and HELMET measure and why a 152M base model cannot take them (instruction following; document QA), so the notebook runs RULER single-needle tasks on Qwen3-1.7B at 4K–32K as a demonstration of the tasks and of RULER's notion of effective length, and states HELMET's application-oriented tasks (and the finding that synthetic scores predict them imperfectly) without running them.
9. **The cost of the longer context after training:** KV cache of R2 at 16K (302 MB per sequence) and what GQA bought (`sec_kv-cache`): a pointer to Chapter 12.4, and to context parallelism (6.5) for lengths that do not fit one device.

#### Derivations

**D1 (v5's derivation). Which RoPE frequencies interpolation compresses, and why YaRN treats high and low frequencies differently.**
*Setting.* Head dimension `d_h` (R2: 64), base `B` (10,000), pair `m ∈ {0, …, d_h/2 - 1}` with frequency `θ_m = B^{-2m/d_h}`, wavelength `λ_m = 2π/θ_m`, training length `L` (4,096), target `L' = sL` (16,384, `s = 4`), rotations in the training window `r_m = L/λ_m`. A method replaces `θ_m` by `θ'_m` (position scaling `pos → pos/s` is the same as `θ'_m = θ_m/s`).
*Steps.*
1. **Which phases were seen.** Pair `m` rotates by `θ_m · pos`. If `r_m ≥ 1` the window covers a full circle, so every *phase* of that pair was seen in training; extending the length produces no new phase for it individually. If `r_m < 1` only the fraction `r_m` of the circle was seen; positions beyond `L` produce phases never seen. Extrapolation is therefore out-of-distribution mainly for the low-frequency pairs.
2. **What interpolation removes and what it costs.** PI (`θ'_m = θ_m/s`) maps the new range back onto the old one for every pair, so no pair sees an unseen phase. But the *phase step between adjacent tokens* shrinks from `θ_m` to `θ_m/s` for every pair; for the fastest pair (`θ_0 = 1` rad per token) this is 1 rad → 0.25 rad at `s = 4`. Local order is read from the high-frequency pairs, so their resolution is exactly what PI degrades; the book's character-model run (rescaling alone increases perplexity) is this effect.
3. **NTK-aware.** Choose a new base `B'` so that the highest frequency is unchanged and the lowest is divided by exactly `s`: `θ'_{last} = θ_{last}/s`, i.e. `B'^{-(d_h-2)/d_h} = s^{-1} B^{-(d_h-2)/d_h}`, so `B' = B · s^{d_h/(d_h - 2)}`. Then `θ'_m = θ_m · s^{-2m/(d_h-2)}`: a geometric ramp of per-pair scale factors `s_m = s^{2m/(d_h-2)}`, from 1 at `m = 0` to `s` at the last pair.
4. **Where NTK-aware falls short.** The ramp is fixed by the *index*, not by `r_m`. For R2 (`L = 4,096`, `s = 4`) the 9 pairs with `r_m < 1` are `m = 23, …, 31`; pair `m = 23` has `s_m = 4^{46/62} = 2.80 < 4`, so its phase range at 16K is `4/2.80 = 1.43×` the trained range: a 43% extrapolation. Eight of the nine low-frequency pairs are under-interpolated; only the last is scaled by exactly `s`. (At T1, `L = 1,024`: 13 of 14, worst overhang 79%.)
5. **YaRN (NTK-by-parts).** Define the ramp `γ(r) = clip((r - α)/(β - α), 0, 1)` with `α = 1`, `β = 32` (the values the paper reports for Llama), and `θ'_m = (1 - γ(r_m)) θ_m/s + γ(r_m) θ_m`. Pairs with `r_m < α` get exactly PI (no unseen phase); pairs with `r_m > β` are untouched (high-frequency resolution kept); pairs in between are blended. For R2 at `L = 4,096`: 9 pairs fully interpolated, 12 blended (`m = 11..22`), 11 untouched (`m = 0..10`), independent of `s`. (At `L = 1,024`: 14, 12, 6.)
6. **Attention temperature.** After interpolation, attention entropy changes; YaRN scales `q` and `k` rotations by `m_s = 0.1 ln s + 1` so the logits scale by `m_s^2 = 1/t`. For `s = 4`: `m_s = 1.1386`, logits ×1.2965. This is an empirical fit on Llama (the paper's Eq. 15); the notebook reports whether it helps R2 and says it is not derived.
7. **Summary of per-pair scale factors.** `s_m = s` (PI), `s^{2m/(d_h-2)}` (NTK-aware), `1/((1-γ_m)/s + γ_m)` (YaRN): `fig_lm_lc_scales`; phase overhang `θ'_m L'/(θ_m L) - 1` on pairs with `r_m < 1`: zero for PI and YaRN, positive for NTK-aware: `fig_lm_lc_overhang`.
*Check cell.* (a) `s = 1` returns the base frequencies for every scheme (≤ 1e-7); (b) PI by position scaling and by `θ/s` agree exactly in the rotated queries; (c) NTK-aware: first pair unchanged, last pair `θ_last/4` (relative 1e-6), `B' = 10,000 · 4^{64/62}`; (d) YaRN counts for `(d_h, B, L, α, β) = (64, 10,000, 4,096, 1, 32)` are 9/12/11 and for `L = 1,024` are 14/12/6; (e) `yarn_mscale(4) = 1.1386` and `PackedAttention` with `rope_mscale` produces logits equal to `m_s²` times the unscaled logits (1e-5); (f) the NTK-aware under-interpolation numbers above (8 of 9 pairs, worst overhang 0.43 at `m = 23`); (g) against `transformers` `ROPE_INIT_FUNCTIONS['yarn']` with `truncate=False`: max difference in `inv_freq` below 1e-6 (with the library default `truncate=True` the library rounds the ramp's correction range to integer pair indices, so differences at the ramp edges are expected and printed; verify for the installed version).

**D2. Cost of length: the attention share of training FLOPs.**
*Statement.* Per-token training FLOPs are `F(T) = 6 N_mm + 12 L d T` (PaLM convention; `N_mm` the matrix-multiply parameters including the tied head), so the attention share is `a(T) = 12 L d T/(6 N_mm + 12 L d T)`.
*Steps.* (1) The first term is the context-free cost of `eq_six_nd`; (2) the second is the score and value-mixing matmuls, `4 T d` per layer forward, thrice for forward plus backward; (3) for R2: `6 N_mm = 0.830 G`, `12 L d = 165,888`, so `F(4,096) = 1.510 G`, `F(8,192) = 2.189 G`, `F(16,384) = 3.548 G`, and `a = 0.45, 0.62, 0.77`; (4) at a fixed FLOP budget, training at 16K sees `F(4K)/F(16K) = 0.43` as many tokens as at 4K; (5) a progressive schedule with `x` tokens at 8K and `y` at 16K costs `x F(8K) + y F(16K)` against `(x+y) F(16K)`, saving `x (F(16K) - F(8K))`, e.g. 40M of 100M tokens at 8K saves 0.54e17 FLOP of 3.55e17, 15%.
*Check cell.* Evaluate `F` from `d2l.flops_per_token` and from `FlopCounterMode` for `T ∈ {4K, 8K, 16K}` on the T1 preset (within the 3% of 4.2 D1), and print the savings table for three progressive splits.

#### From-scratch implementation

About 90 lines:

| Component | Contract | Lines |
|---|---|---|
| `rope_inv_freq`, `yarn_mscale` | D1 formulas | 30 |
| `make_recall_task` | unique multi-token keys and values from a vocabulary of common single-token words; filler lines; query line; 8 candidates; depth-controlled placement | 35 |
| `long_context_gain` | repeated-rare-word detection (first occurrence at least `min_gap` tokens earlier), full-context and sliding-window NLL | 30 |
| arm driver (cell code) | per-arm config and a loop over `d2l.pretrain` | 15 |

Numerical tests: (1) D1 checks (a)–(g); (2) `make_recall_task`: every item's gold value appears exactly once after the queried key in the prompt, candidates are distinct, the prompt length equals the target within a tolerance of one line (assert), depth places the queried pair within one pair of the requested relative position; (3) a hand-built "oracle copier" (a function that returns the token after the last occurrence of the key) scores 100% on every length, which validates the task independently of any model; (4) `long_context_gain` on a synthetic sequence that repeats a random 50-token block at distance 6,000: gain is large and positive for a model that has an induction circuit and ≈ 0 for a unigram model; (5) per-position NLL buckets average to the overall NLL (weighted) to 1e-6.

#### In production

- **Transformers `rope_parameters`/`rope_scaling` types** (`linear` = PI, `dynamic` = NTK-aware computed at run time, `yarn`, `longrope`, `llama3`): the oracle for `rope_inv_freq` (D1(g)). Llama 3.1's piecewise-by-wavelength scheme is the `llama3` type: the same idea as YaRN's ramp with different constants (Meta's report documents the long-context stage). The ProLong recipe ([2410.02660](https://arxiv.org/abs/2410.02660) and its [code](https://github.com/princeton-nlp/ProLong)) is the production recipe whose data and length-schedule choices this section imitates at small scale.
- **vLLM/SGLang** apply the same scaling types at serving time (Chapter 13); training and serving must use the same scaling or the model sees a different positional code than it was trained for (a pitfall below).
- **RULER** ([code](https://github.com/NVIDIA/RULER)) and **HELMET** ([code](https://github.com/princeton-nlp/HELMET)) are the evaluation oracles; the notebook's synthetic recall is the from-scratch counterpart of RULER's needle tasks.

#### Headline experiment

- **Hypothesis.** (H1) Zero-shot, YaRN-style scaling degrades R2's short-context perplexity less and its 16K perplexity less than PI does, with NTK-aware in between or comparable (the YaRN paper reports lower perplexity for NTK-by-parts than for PI and NTK-aware without fine-tuning, on Llama models; direction to be checked here). (H2) After 100M tokens of training at 16K, the gaps in perplexity between the schemes shrink (the PI and YaRN papers both fine-tune to recover; the YaRN paper reports that YaRN needs about ten times fewer tokens than PI to reach comparable results; the notebook measures this at its scale with a training-curve comparison, not a claim). (H3) Perplexity at 16K recovers earlier than recall accuracy and long-context gain do, so perplexity alone overstates the effective context. (H4) The progressive schedule matches or beats one-shot extension at equal tokens (no direction asserted; D2 says it is cheaper).
- **Setup (T2).** Base R2 step 6000 (4K). Zero-shot arms Z-none, Z-PI, Z-NTK, Z-YaRN (evaluation only; `s = 4`). Trained arms, each 100M tokens at 16K, global batch 262,144 tokens, peak LR 1e-4 with warmup 20 steps, constant, final 15% linear to 0, AdamW (0.9, 0.95), weight decay 0.1, clip 1.0, bf16 autocast, fp32 master weights, `PackedStream` with document masks, 50% long / 50% short mixture: T-PI, T-NTK, T-YaRN (`α = 1`, `β = 32`, `rope_mscale = yarn_mscale(4)`), T-YaRN-prog (40M tokens at 8K with `s = 2`, then 60M at 16K with `s = 4`). Seed 0 for all; data order identical across arms. Tier T2.
- **Setup (T1).** Base `r2t1`; `s = 4` from 1,024 to 4,096; 16M tokens per arm at `T = 4,096` (progressive: 6.4M at 2K, 9.6M at 4K), batch 65,536 tokens; data `lm-pg19-30m` and `lm-fwedu-replay-12m`; the same arms and evaluation lengths `{1K, 2K, 3K, 4K}`; seeds 0–2 for T-PI and T-YaRN, seed 0 otherwise.
- **Procedure.** Evaluate the four zero-shot arms and the base; train the four arms; evaluate each on E1, E2, E3 at the lengths above; compute per-family effective context with bootstrap intervals (windows for E1/E3, items for E2).
- **Metrics.** NLL by position bucket; recall accuracy by length and depth (200 items per cell; the interval is binomial or bootstrap); long-context gain by distance bucket; the effective context per family; training loss curves at 16K by arm (to compare sample efficiency); tokens to reach the zero-shot-YaRN 16K NLL.
- **Plots.** `fig_lm_lc_ppl_position` (x position bucket, log; y NLL; one line per arm; vertical line at the trained length 4K); `fig_lm_lc_recall` (per arm a length-by-depth heatmap of accuracy, with the chance level 12.5% in the colour bar's reference and a column of effective-context markers); `fig_lm_lc_gain` (long-context gain versus distance bucket, per arm, with intervals); `fig_lm_lc_scales`, `fig_lm_lc_overhang` (D1); `fig_lm_lc_train` (training loss at 16K versus tokens, per arm).
- **Expected outcome.** As in the hypotheses (directions). "To be measured" for all magnitudes. The notebook must report if all trained arms are indistinguishable.
- **Noise check.** T1: three seeds for PI and YaRN give the seed spread of 16K NLL and of recall accuracy (the paired difference PI minus YaRN with a t interval on 2 degrees of freedom, with the caveat that this is weak); T2: single seed, so the T2 comparison between trained arms is reported with bootstrap intervals for evaluation noise only and labeled "single run"; differences below the T1 seed spread are not interpreted.
- **T1 reduction.** As above (`--base r2t1`). **Artifacts committed (T2).** `lm-r2-16k-pi`, `lm-r2-16k-yarn`, `lm-lc-evals`.

#### Figures and tables

- `tab_lm_lc_schemes` (scheme → per-pair rule → hyperparameters → what it preserves), `tab_lm_lc_flops` (D2: `F(T)`, attention share, tokens per FLOP budget), `tab_lm_lc_effective` (arms × families: effective context with intervals), `tab_lm_lc_recipes` (documented long-context recipes in Llama 3, SmolLM3, ProLong: base-frequency change versus scaling, length stages, data share; qualitative columns, each tied to its report).
- `fig_lm_lc_ppl_position`, `fig_lm_lc_recall`, `fig_lm_lc_gain`, `fig_lm_lc_scales`, `fig_lm_lc_overhang`, `fig_lm_lc_train`, `fig_lm_lc_wavelength` (R2's 32 wavelengths against the 4K window and the 16K target, coloured by class).

#### Pitfalls and misconceptions

- Perplexity at length `T` is not use of length `T`; a model can ignore distant tokens and still look fine (E1 versus E2/E3).
- Concatenated unrelated documents have no long-range dependency; training on them teaches nothing about long context while costing 2.3× the FLOPs per token.
- Serving with a different scaling than training (PI at training and none at serving, or the reverse) is a silent quality loss; store the scaling in the checkpoint config.
- YaRN's `α`, `β` were tuned on Llama (head dimension 128); R2 has head dimension 64 with only 32 pairs, so the ramp spans fewer pairs; treat them as hyperparameters (exercise 5).
- Applying the attention temperature twice (once as `rope_mscale`, once as a softmax scale).
- A recall task whose keys repeat or whose answers occur twice has more than one correct answer: the gold-occurrence test of the from-scratch section.
- Evaluating "effective context" with a fixed threshold copied from another model or task.
- Raising the RoPE base without training is not extension: the zero-shot arms show what the positional mechanism alone does.

#### Summary (draft bullets)

- Low-frequency RoPE pairs (wavelength beyond the training window) are the ones whose phases are unseen at longer lengths; high-frequency pairs carry local order.
- PI compresses every pair by `s`; NTK-aware compresses by a geometric ramp fixed by the pair index and leaves most low-frequency pairs partly extrapolated; YaRN chooses the factor per pair from the number of rotations in the training window, plus an attention temperature.
- Longer training costs more per token because attention's share of FLOPs rises from 45% at 4K to 77% at 16K; progressive schedules spend tokens where they are cheaper.
- Long-range-dependent data (books, long papers) is what teaches the use of length; concatenated short documents do not.
- Effective context must be measured by tasks and by dependence on distant tokens; perplexity at length is a necessary, not a sufficient, check.

#### Exercises

1. **(check)** For R2, which pairs have `r_m < 1` and how many rotations does pair `m = 10` make in the 4K window? *Sketch:* `λ_m = 2π · 10^{m/8}` (since `B^{2m/d_h} = 10,000^{m/32} = 10^{m/8}`); `m = 10`: `λ = 2π · 17.8 = 112`; `r = 4,096/112 = 36.6` rotations (above `β = 32`, so YaRN leaves it untouched); pairs `m ≥ 23` have `r < 1`.
2. **(check)** With `s = 4`, what are the PI, NTK-aware and YaRN scale factors for `m = 0` and `m = 31` (`d_h = 64`, `L = 4,096`)? *Sketch:* PI: 4 and 4; NTK-aware: `1` and `4^{62/62} = 4`; YaRN: `m = 0` has `γ = 1` so factor 1; `m = 31` has `r < 1`, `γ = 0`, factor 4.
3. **(derive)** Show that PI's per-token phase step on pair `m` is `θ_m/s` and derive the smallest offset difference the pair can resolve with phase precision `ε` radians. Use it to say which pairs lose resolving power at `s = 4` for `ε = 0.05`. *Sketch:* offsets `δ` and `δ+1` differ in phase by `θ_m/s`; resolvable iff `θ_m/s > ε`; with `θ_m = 10^{-m/8}` and `s = 4` the condition is `θ_m > 0.2`, i.e. `m < 8 log10 5 = 5.6`, so only pairs `m ≤ 5` keep adjacent-token resolution; unscaled, `θ_m > 0.05` gives `m < 10.4`, i.e. `m ≤ 10`.
4. **(derive)** Derive the progressive-schedule saving of D2 for a general split and find the split fractions that minimize FLOPs subject to a minimum number of tokens at the target length. *Sketch:* with `y ≥ y_min` at 16K and the rest at 8K, FLOPs are `x F(8K) + y_min F(16K)`; the minimum is at `y = y_min`; the quality constraint, not FLOPs, decides `y_min`.
5. **(measure)** Sweep YaRN's `β ∈ {8, 16, 32, 64}` (with `α = 1`) at T1 for one seed each, then at the best two with three seeds. Predict from `d_h = 64` whether the paper's `β = 32` is best; report intervals.
6. **(measure)** Evaluate Z-PI, Z-NTK, Z-YaRN and the trained arms at 32K (`s = 8` from 4K). Predict where each scheme fails first (E1, E2 or E3) and measure.
7. **(open)** Vary the long-data share (0, 25, 50, 75, 100%) at T1 and decide the share that keeps short-context NLL within the seed spread while maximizing 4K recall accuracy; state your criterion before running, and discuss whether the answer would transfer to a 7B model.

#### Further reading

- Papers: v5's [YaRN](https://arxiv.org/abs/2309.00071) (the derivation's source; NTK-by-parts, temperature), [How to Train Long-Context Language Models (Effectively)](https://arxiv.org/abs/2410.02660) (ProLong: data mix, length schedule, evaluation), [RULER](https://arxiv.org/abs/2404.06654) (synthetic tasks, effective length), [HELMET](https://arxiv.org/abs/2410.02694) (application-oriented evaluation); added: [Position Interpolation](https://arxiv.org/abs/2306.15595) (PI and its fine-tuning recipe), [Scaling Laws of RoPE-based Extrapolation](https://arxiv.org/abs/2310.05209) (base versus length), [Data Engineering for Scaling Language Models to 128K Context](https://arxiv.org/abs/2402.10171) (upsampling long documents; length-wise data), [Effective Long-Context Scaling of Foundation Models](https://arxiv.org/abs/2309.16039) (the base-frequency change and continued pretraining at length), [What is Wrong with Perplexity for Long-context Language Modeling?](https://arxiv.org/abs/2410.23771) (why E1 misleads), [Lost in the Middle](https://arxiv.org/abs/2307.03172) (position sensitivity of use), [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).
- Tutorials: the [SmolLM3 blog](https://huggingface.co/blog/smollm3) (a documented long-context stage); the Hugging Face [rotary embeddings utilities page](https://huggingface.co/docs/transformers/main/en/internal/rope_utils) (the scaling types in code).
- Code: [YaRN reference implementation](https://github.com/jquesnelle/yarn); [ProLong](https://github.com/princeton-nlp/ProLong), [RULER](https://github.com/NVIDIA/RULER), [HELMET](https://github.com/princeton-nlp/HELMET).

#### Risks and open decisions

- *A 152M base model may show no usable recall at 16K even after extension,* because induction circuits trained on 4K contexts may not scale at this size and token budget. Fallback ladder: (1) evaluate recall at `{4K, 6K, 8K}` instead; (2) increase training tokens at 8K; (3) report E1 and E3 only and say recall was at chance. The T1 version has a clear a-priori task (copying by induction); verify with the oracle test and a trained-from-scratch 4K model first.
- *Training time at 16K is the largest cost here.* Fallback: 50M tokens per arm (1.5–1.9 h for four arms).
- *`rope_utils` differences (`truncate`)* may make the HF oracle disagree at the ramp edges; the check prints them and passes with `truncate=False`.
- *PG-19 license and provenance (decided 2026-10-29):* `deepmind/pg19` lists Apache-2.0 and its books are public domain, so `lm-pg19-test` and `lm-pg19-30m` are hosted with attribution. `emozilla/pg19` (the parquet conversion used as the practical reader) carries no licence field; the shard card cites `deepmind/pg19` as the source.
- *RULER needs a vLLM or HF generation setup and a long runtime at 32K;* the cross-check is optional and may use only 4K/8K/16K.
- *YaRN's attention factor* is an empirical Llama fit; if it hurts R2, the notebook reports `rope_mscale = 1` as well.

### 4.7 ◇ Budgets beyond Chinchilla

- **Label:** `sec_lm_budgets` · **Tier:** T0 (CPU; seconds to minutes; no reduced configuration needed) · **Budget:** under 5 min on a laptop CPU, no GPU; the only measured inputs are committed JSON tables · **Codex units:** 5.5.
- **Builds on (book):**
  - `sec_scaling-laws`: `eq_six_nd`, the Chinchilla law `eq_chinchilla_law`, the allocation `eq_chinchilla_optimum`, the two published fits quoted in its exercises (Hoffmann et al.: `E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28`; the Besiroglu et al. refit: `E = 1.82, A = 482.0, B = 2085.4, α = 0.348, β = 0.366`), the data-repetition remark (Muennighoff et al.), and its exercise "Training for inference" (`C_life = 6ND + 2ND_inf`; Sardana et al.), which this section turns into a calculator. Not re-taught: the fit, the 6ND derivation, the compute-optimal allocation derivation.
  - `sec_kv-cache`: decode costs about `2N` FLOPs per token and is memory-bound; KV bytes per token. `sec_perf_model` and `sec_hardware`: roofline, `tab_gpu_specs`, bandwidth ladders.
- **Builds on (this part):** 4.2 (`d2l.flops_per_token`, `d2l.mfu`, R2's measured training tokens/s and MFU from its log); 12.1 (assumed, optional: `d2l.FastGenerator`, whose measured decode rates fill `decode-rates`; if 12.1 is not yet authored the table is measured with Hugging Face `generate` and a static cache, at authoring); 14.1 (optional: `d2l.config_stats`, parameters and cache bytes from a config).
- **Exports:** none (cells only). **Data artifact:** `DATA_HUB['lm-decode-rates']` (JSON: decode tokens/s and GPU memory for R2, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` at batch 1, 8, 32, 128 and context 1,024, measured on one Ada GPU at authoring; about 5 KB; to be measured).
- **Models and data:** no training. Reported token budgets used in the "how overtrained are released models" table (cited, to be re-checked when authoring): Llama 3 8B about 15T tokens (Llama 3 report), OLMo 2 1B and 7B 4T (model cards), SmolLM2 1.7B about 11T (SmolLM2 paper), Qwen3 about 36T (Qwen3 report), R2 3.15B (4.2).
- **Packages:** `numpy`, `scipy` (root finding), `matplotlib`.

#### Narrative arc

1. **The question.** Opening cell: R2 has 138.4M parameters and saw 3.15B tokens (22.7 per parameter). A model like it will serve far more tokens than it was trained on. Print the training-versus-inference identity `6 N D_tr = 2 N D_inf` at `D_inf = 3 D_tr`: after serving three times the training tokens, inference FLOPs exceed training FLOPs for the same `N`, so a training-only optimum is the wrong objective for a deployed model. The measured symptom is a second row from the committed `decode-rates` table: GPU-seconds per generated token against GPU-seconds per trained token, whose ratio is much greater than the FLOP ratio of 1/3 because decode is memory-bound (to be measured).
2. **Lifetime cost in FLOPs and in GPU-seconds.** `C_life = N (6 D_tr + 2 D_inf)` (the book's exercise); then the measured version `T_life = 6 N D_tr/(MFU_tr · P) + D_inf / r_dec(N, b)` with `r_dec` the measured decode rate (tokens/s per GPU at batch `b`). Define `κ(N, b)`, the cost per inference FLOP relative to the cost per training FLOP; the FLOP version is `κ = 1`.
3. **The compute-optimal model for a target loss, with inference in the objective** (D1). The optimality condition, its limits, and the picture: the optimum moves to smaller `N` and more training tokens per parameter as `D_inf` grows. A table of optimal `(N, D_tr, D_tr/N)` for demand scenarios, under both published fits.
4. **How decode speed enters: a roofline model** (D2), validated against `decode-rates`; the batch dependence; why large `N` pays more than 2N FLOPs per token at small batch.
5. **Overtraining small models.** Where released models sit relative to 20 tokens per parameter (`tab_lm_bud_released`), with the calculator's prediction for plausible demand; the notion of an overtraining multiple `D_tr/(20 N)`; the empirical finding that loss and downstream error stay predictable under overtraining (Gadre et al.); a caution that the Chinchilla-form law is fitted near its own data regime.
6. **Repeated data under a token shortage** (D3). The effective-token model; what it says about doubling epochs on a fixed corpus; how it combines with D1 when `D_tr` is capped by the corpus.
7. **Sensitivity, not a point estimate.** The calculator reports bands over the two fits, over demand in a factor-of-ten range, over `r_dec` ±30%, and over MFU; a region of demand where the preferred model changes is shown as a boundary; where the band is wider than the difference between candidates, the notebook says "not decided".

#### Derivations

**D1 (v5's derivation). Compute-optimal size when inference tokens enter the objective.**
*Statement.* Minimize `C(N) = N (6 D + 2 D_inf)` over `N` subject to `E + A N^{-α} + B D^{-β} = ℓ`, for a target loss `ℓ > E` and fixed `D_inf`. Let `a = A N^{-α}` and `b = B D^{-β}` (so `a + b = ℓ - E`). The optimum satisfies
`α a = β b (1 + D_inf/(3 D))`  (eq. `eq_lm_bud_foc`),
which reduces to `eq_chinchilla_optimum` for `D_inf = 0`, and is unique.
*Assumptions.* The Chinchilla form holds with the chosen constants; `N` is the parameter count that costs `2N` per generated token; prompt and decode tokens are lumped in `D_inf`; `κ = 1`.
*Steps.*
1. Feasibility needs `ℓ - E - A N^{-α} > 0`, i.e. `N > N_min = (A/(ℓ - E))^{1/α}`. On the constraint, `D(N) = (B/(ℓ - E - A N^{-α}))^{1/β}`, finite and decreasing for `N > N_min`, and `D → ∞` as `N → N_min`.
2. `C(N) = N (6 D(N) + 2 D_inf)`; `dC/dN = 6 D + 2 D_inf + 6 N D'(N) = 0`.
3. Differentiate `D`: with `ε(N) = ℓ - E - A N^{-α}`, `D' = -(1/β) D ε'/ε`, `ε' = α A N^{-α-1}`, so `N D' = -(α/β) D a/ε`, and on the constraint `ε = b`.
4. Substitute: `6 D + 2 D_inf - (6 α/β) D a/b = 0`, i.e. `(α/β)(a/b) = 1 + D_inf/(3 D)`, which is the stated condition.
5. For `D_inf = 0` this is `α a = β b`, the book's `eq_chinchilla_optimum` (which was derived at fixed compute instead of fixed loss: the two coincide at the optimum because the same tangency condition holds).
6. *Uniqueness.* Define `F(N) = α a - β b (1 + D_inf/(3 D))`. As `N` grows, `a` falls, `b = ε` rises, `D` falls so `D_inf/(3D)` rises: `F` is strictly decreasing; `F → α a(N_min) > 0` as `N → N_min` and `F < 0` for large `N`. One root.
7. *Direction.* Raise `D_inf`: for the same `N` the bracket grows, `F` falls, so the root moves to smaller `N`; by the constraint `D` rises. The optimum moves to smaller `N` and more tokens per parameter as demand grows.
8. *Limit.* As `D_inf/D → ∞`, the condition forces `b → 0`, `a → ℓ - E`, i.e. `N → N_min` and `D → ∞`: a model as small as the target loss allows, trained on as many tokens as it takes.
9. *Measured cost.* Replace `2 N D_inf` by `D_inf` times the measured GPU-seconds per generated token (converted to FLOP-equivalents through `κ(N)`); since `κ` depends on `N` through the roofline of D2 there is no closed form, so the notebook minimizes `T_life(N)` by a scan over a grid of `N`, with `D(N)` from the constraint. With a constant `κ` the condition is exactly `α a = β b (1 + κ D_inf/(3 D))` (exercise 3).
*Check cell.* (a) Solve `F(N) = 0` by bisection on `(N_min, 1e6 N_min)` for `ℓ ∈ {2.0, 2.2}` and `D_inf ∈ {0, 1e11, 1e12, 1e13}` under the Hoffmann constants; at `D_inf = 0` the root equals the closed-form `N* = G (C/6)^{β/(α+β)}`, `D* = (C/6)/N*`, `G = (αA/(βB))^{1/(α+β)}`, with `C = 6 N D` evaluated at the root, to relative 1e-8. (b) `D_inf` increases ⇒ `N` decreases and `D/N` increases, monotonically, in all cases. (c) Checked while writing this blueprint (Hoffmann constants): at `ℓ = 2.0`, `N` falls from 15.3B (`D_inf = 0`) to 11.5B (1e12) and 6.5B (1e13), while `D/N` rises from 79 to 143 and 600; at `ℓ = 2.2`, `N` falls from 3.5B to 1.7B and 0.97B, `D/N` from 58 to 296 and 1,855. These are properties of that fit, not predictions about real models; the notebook repeats the table under the Besiroglu constants and prints the differences, which the book's exercise on the precision of the fitted ratio makes the point of.
(d) The Llama-3-8B scenario of the book's exercise: predicted loss of `N = 8e9`, `D = 1.5e13` under the Besiroglu fit, the compute-optimal model of equal predicted loss, the ratio of training costs and the break-even `D_inf` where the lifetime costs are equal; the cell reproduces the exercise's answer so the calculator is checked against the book.

**D2. Decode rate from a roofline, with the KV cache.**
*Statement.* At batch `b`, context `c`, weight bytes per parameter `w`, KV bytes per token `k` (R2: 18,432 B/token from `sec_kv-cache`), memory bandwidth `W` and sustained compute `P_eff`, `r_dec(N, b) ≈ b / max(t_mem, t_cmp)` with `t_mem = (N w + b c k)/W` and `t_cmp = 2 N b / P_eff` per step.
*Steps.* (1) A decode step reads all weights once and each sequence's KV cache once: `N w + b c k` bytes; (2) it performs `2 N` FLOPs per sequence: `2 N b`; (3) the step time is the larger of the memory and compute times (`sec_perf_model`'s regimes); (4) at `b = 1` the step is memory-bound and `r_dec = W/(N w + c k)`: inversely proportional to `N`; (5) the batch at which the two times cross is `b* ≈ P_eff w/(2 W)` ignoring KV, i.e. the arithmetic intensity at the ridge of `tab_gpu_specs` divided by `2/w`; for bf16 (`w = 2`) it equals the ridge in FLOP/B; (6) `κ(N, b) = (b-dependent GPU-seconds per generated token × MFU_tr P) / 2N`: large at `b = 1`, tending to `MFU_tr P/P_eff` for large `b`.
*Check cell.* Predict `r_dec` for the four models at batch `{1, 8, 32, 128}` from the datasheet numbers of the 4090 (1.0 TB/s, 165 TFLOP/s bf16, from `tab_gpu_specs`) with `P_eff` = 0.5 of peak, compare with `decode-rates`; the A5-style tolerance of the part (15%) is the target at `b = 1`, and the notebook reports the larger deviations at large batch honestly (to be measured).

**D3. Effective tokens under repetition.**
*Statement.* In the Muennighoff et al. model, `U` unique tokens seen `R + 1` times (`R` repeats) are worth `D' = U + U R*_D (1 - e^{-R/R*_D})` effective unique tokens, with a fitted constant `R*_D` (the paper reports its fit; the check below uses `R*_D = 15` as an illustration and the author replaces it with the paper's value). The loss law is evaluated at `D'`.
*Steps.* (1) The marginal value of the `(R+1)`-th pass is `dD'/dR = U e^{-R/R*_D}`: the first repeats are almost as good as new data and the value decays exponentially with scale `R*_D`; (2) as `R → ∞`, `D' → U (1 + R*_D)`: infinite repetition is worth at most `1 + R*_D` times the unique data; (3) the compute spent is `6 N (R + 1) U`, not `6 N D'`; (4) combine with D1 by replacing `D` in the constraint with `D'(U, R)` and minimizing over `(N, R)` at fixed `U`.
*Check cell.* With `R*_D = 15`: `D'/ (U (R+1))` is 1.00, 0.98, 0.93, 0.82, 0.65, 0.37 for 1, 2, 4, 8, 16, 40 epochs (checked while writing), consistent in shape with the paper's finding that up to about four epochs cost little. Verify numerically that `dD'/dR` matches the finite difference.

#### From-scratch implementation

About 70 lines in cells (no exports):

| Component | Contract | Lines |
|---|---|---|
| `solve_optimum(ell, D_inf, consts, kappa=None)` | bisection on `F(N)` (FLOP version) or a grid scan of `T_life` (measured version) | 25 |
| `lifetime_cost(N, D_tr, D_inf, rates)` | FLOPs and GPU-seconds | 15 |
| `decode_roofline(N, b, c, W, P)` | D2 | 10 |
| `effective_tokens(U, R, R_star)` | D3 | 6 |
| sensitivity driver | grid over fits × demand × `r_dec` scale | 15 |

Numerical tests: (1) D1 checks (a), (b) and (d); (2) `lifetime_cost` at `D_inf = 0` equals `6 N D` (exact); (3) `decode_roofline` equals `W/(N w + c k)` at `b = 1` when memory-bound (exact); (4) `effective_tokens` at `R = 0` equals `U`, and its derivative matches a finite difference (1e-6); (5) monotonicity of the optimum in `D_inf` (strict).

#### In production

- No library is required: the calculator *is* the tool. Real planning uses internal scaling-law fits and measured serving costs per hardware class; the structure here (a fitted loss law, a lifetime cost with measured decode rates, and sensitivity bands) is the same.
- The measured decode rates should come from the serving stack that will deploy the model (`d2l.FastGenerator`, vLLM/SGLang of Chapter 13): the choice of stack changes `κ` more than the choice between close fits. Point to 12.1 and 13.1 for the stack and to 14.9 for the design-under-a-budget capstone.

#### Headline experiment

- **Hypothesis.** (H1) Under both published fits, the lifetime-optimal tokens-per-parameter ratio exceeds the training-optimal ratio and grows with assumed demand (this follows from D1; the notebook confirms and measures the size). (H2) Replacing the FLOP cost of inference with measured GPU-seconds shifts the optimum further toward smaller models because decode at small batch is memory-bound (to be measured; the size depends on the batch). (H3) The two fits disagree on the *level* of the optimal ratio by more than the effect of a factor-of-ten change in demand over part of the range, so the notebook cannot name one ratio (direction; to be measured; the book's exercise on fitted ratios is the motivation).
- **Setup.** Candidate models: R2 (138.4M, 3.15B tokens) and three hypothetical sizes at equal predicted loss from D1; decode-rate table for R2 and Qwen3-0.6B/1.7B/4B at four batch sizes (committed); target losses as a parameter; demand `D_inf ∈ {1e10, 1e11, 1e12, 1e13}` tokens; fits: Hoffmann and Besiroglu constants; `MFU_tr` 0.35–0.45 (the A100 band of 4.2); `r_dec` multiplier 0.7–1.3. Tier T0.
- **Procedure.** For each scenario solve D1 (FLOP version) and the measured version; tabulate `(N*, D*, D*/N*, lifetime cost)`; compute the demand at which the lifetime cost of the Chinchilla-optimal model of equal loss equals that of the overtrained smaller model (the break-even); show the sensitivity bands.
- **Metrics.** `N*`, `D*/N*`, break-even `D_inf`, lifetime cost ratio, each with its band over fits and rates.
- **Plots.** `fig_lm_bud_foc` (x `N`, y lifetime cost at fixed target loss, one curve per `D_inf`, optimum marked, Hoffmann constants); `fig_lm_bud_ratio` (x demand `D_inf`, y optimal `D*/N*`, bands over fits and decode rates, a horizontal line at 20 and points for released models and R2 from `tab_lm_bud_released`); `fig_lm_bud_decode` (D2: measured against predicted decode tokens/s versus batch, per model); `fig_lm_bud_repeat` (D3: effective-token fraction versus epochs for three values of `R*_D`).
- **Expected outcome.** As in the hypotheses (directions). No optimum ratio is claimed: the result is a band, and the notebook prints the demand range over which the preferred model is the same under all scenarios and the range over which it is not.
- **Noise check.** Not statistical: the uncertainty is in the fit constants and the assumptions, shown by the two published fits, the demand range, the decode-rate multiplier and the MFU band; the measured decode rates are medians of repeated timings with their spread (the `d2l.Benchmark` convention, `sec_perf_model`). A comparison that flips inside the band is reported as undecided.
- **T1 reduction.** Not needed (T0). If `decode-rates` has to be re-measured, it is a T1 job of a few minutes per model.

#### Figures and tables

- `tab_lm_bud_released` (model, `N`, `D`, tokens per parameter, overtraining multiple `D/(20N)`; sources per row), `tab_lm_bud_scenarios` (D1 results per fit and demand), `tab_lm_bud_decode` (measured decode rates).
- `fig_lm_bud_foc`, `fig_lm_bud_ratio`, `fig_lm_bud_decode`, `fig_lm_bud_repeat`.

#### Pitfalls and misconceptions

- Counting only parameters as a proxy for serving cost: decode speed depends on weights *and* KV bytes and the batch (D2), and the 2N-FLOPs assumption is a lower bound on cost, not the cost.
- Taking the compute-optimal ratio of one fit as a constant of nature (the two fits disagree; the exercise of `sec_scaling-laws`).
- Extrapolating a law fitted at one tokenizer and data mixture to another: absolute losses do not transfer, only the structure does.
- Forgetting that overtraining multiplies *training* cost relative to the Chinchilla model of equal size, so it pays only above the break-even demand.
- Ignoring that a smaller model may need a longer context or more reasoning tokens to match quality in deployment (Chapters 9 and 12), which raises `D_inf`.
- Treating repetition as free: D3's marginal value decays and the compute is paid per pass.

#### Summary (draft bullets)

- Training-optimal and deployment-optimal models differ: with `D_inf` in the objective the optimum satisfies `α a = β b (1 + D_inf/(3D))`, moving to smaller `N` and more tokens per parameter as demand grows.
- The optimum is unique for a given target loss and fit; as demand grows without bound it tends to the smallest feasible model on unboundedly many tokens.
- Decode cost per FLOP is far above training cost per FLOP at small batch because decode is memory-bound; measured rates, not FLOPs, should enter a real budget.
- Under data scarcity, extra epochs are worth less each time with an exponential decay; the effective-token model gives the budget.
- The answer is a band across fits, demand and serving efficiency: report the region where the choice is robust and the region where it is not.

#### Exercises

1. **(check)** At what `D_inf/D_tr` do inference and training FLOPs coincide for the same `N`? Why is the break-even in dollars different? *Sketch:* `6 N D = 2 N D_inf` gives `D_inf = 3 D`; decode runs at a lower utilization than training (memory-bound), so a GPU-second buys fewer decode FLOPs than training FLOPs and the break-even is reached at smaller `D_inf`.
2. **(check)** Why must `N > N_min` for a target loss `ℓ`, and what does `N_min` mean physically? *Sketch:* even with infinite data the loss floor of an `N`-parameter model is `E + A N^{-α}`; below `N_min` the floor exceeds the target.
3. **(derive)** Redo D1 with a per-token inference cost `κ 2 N` (constant `κ`): show the condition becomes `α a = β b (1 + κ D_inf/(3 D))`. *Sketch:* replace `2 D_inf` by `2 κ D_inf` in step 2; the rest is unchanged.
4. **(derive)** With D3, minimize `C = 6 N (R + 1) U` plus `2 N D_inf` over `(N, R)` at fixed `U` and fixed target loss; show that the optimal `R` is set by equating the marginal loss reduction per unit compute of an extra pass with that of a larger model, and give the first-order condition. *Sketch:* Lagrangian with the constraint `A N^{-α} + B D'(U, R)^{-β} = ℓ - E`; derivative in `R` brings the factor `dD'/dR = U e^{-R/R*_D}`.
5. **(measure)** Measure decode tokens/s for R2 and Qwen3-0.6B at batch 1, 8, 32 on the Ada GPU and compare with D2. Predict where the model fails (large batch, KV at long context) before measuring.
6. **(measure)** Replace the Hoffmann constants by your own fit to R2's validation-loss trajectory and the T1 preset's (two points are not a fit: say what you would need) or, better, fit `A, B, α, β` to the 4.2 per-domain losses at the checkpoints with `E` fixed to one of the published values; report the uncertainty and its effect on `D*/N*`.
7. **(open)** A product expects `D_inf = 5e12` tokens over its lifetime and needs P95 latency below a bound. Specify the model family (sizes you would train), the decode stack and batch, the training tokens, and the sensitivity analysis that would make you change your mind; state which two numbers you would measure first.

#### Further reading

- Papers: v5's [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) (the inference-aware objective, D1) and [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (D3); added: [Language models scale reliably with over-training and on downstream tasks](https://arxiv.org/abs/2403.08540) (loss and downstream error remain predictable for overtrained models); the Chinchilla paper ([2203.15556](https://arxiv.org/abs/2203.15556), cited in `sec_scaling-laws`) and the replication cited there (Besiroglu et al.).
- Tutorials: the Smola efficiency tutorial ([alex.smola.org/posts/45-mlss-efficiency](https://alex.smola.org/posts/45-mlss-efficiency/), cited in the Codex plan) for deployment cost framing; Stanford CS336's scaling-laws lectures.
- Code: [CS336 assignment 3](https://github.com/stanford-cs336/assignment3-scaling) (a constrained-budget scaling exercise); the notebook's own calculator.

#### Risks and open decisions

- *The fit constants are published fits, not fits to this book's models;* the section must not present `D*/N*` as a property of real models. Fallback: present only the structure and the sensitivity band.
- *`decode-rates` depends on the serving stack available when authoring* (12.1 may not be ready): fall back to Hugging Face `generate` with a static cache and say so; re-measure when `d2l.FastGenerator` lands.
- *Reported token budgets of released models* are taken from model cards and reports and must be re-checked against them at authoring; entries that cannot be verified are dropped from `tab_lm_bud_released`.
- *The Muennighoff `R*_D` value* is not fixed in this blueprint; the author reads it from the paper. If it cannot be read unambiguously, D3 uses the shape only.
- *Optional (◇):* if the part is cut to the core, 4.7 can be replaced by a two-paragraph pointer to `sec_scaling-laws`' exercise.
