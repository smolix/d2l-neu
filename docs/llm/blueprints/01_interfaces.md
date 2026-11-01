# Cross-Chapter Interfaces

The 14 chapter blueprints were drafted in parallel, so some chapters assumed names or signatures that differ from what the owning chapter exports. This file settles every such case.

It takes precedence over any `assumed:` line in a chapter blueprint. Notebook authors read it together with `00_conventions.md`.

Rules:
- Each name has exactly one owner, which defines it with `#@save`. Every other section imports it.
- A section may wrap a shared name in a local helper, but must not redefine it.
- If a consumer needs different behaviour, it uses a new name listed in its own Exports.

## 1. Shared API

| Name | Owner | Canonical signature and return | Main consumers |
|---|---|---|---|
| `d2l.load_hf` | 1.1 | `(model_id, revision=None, dtype=torch.bfloat16, device_map='auto') -> (model, tokenizer)` | all |
| `d2l.config_params` | 1.1 | `(config, include_vision=False, include_mtp=False, by_group=False) -> int \| dict[str, int]`. Dense and hybrid configs only; MoE raises `NotImplementedError('use d2l.config_stats (14.1)')`. With `by_group=True`, returns per-group counts that sum to the total. | 14.1 |
| `d2l.config_memory` | 1.1 | `(config, seq_len, batch_size, dtype_bytes=2, *, state_bytes=4, ...) -> dict` with keys `params, weights, kv_cache, recurrent_state, activations, logits, total` | 5.x, 6.x, 7.1, 14.x |
| `d2l.load_qwen3` | 1.2 | `(model_id='Qwen/Qwen3-0.6B', revision=None, *, dtype=torch.bfloat16, device=None) -> d2l.Qwen3GPT` (built on the meta device, weights assigned) | 4, 11, 12, 13 |
| `d2l.Qwen3Attention(d2l.GQAAttention)` | 1.2 | `(num_hiddens, num_heads, num_kv_heads, *, head_dim=None, qk_norm=False, rope_theta=10000.0, norm_eps=1e-6, bias=False, rope=True, causal=True)`; adds `forward_step(X, cache)`. The book's class is unchanged. | 4.1 (`PackedAttention`), 12, 13, 14 |
| `d2l.SwiGLU(d2l.FeedForward)` | 1.2 | `(num_hiddens, hidden, bias=False)`: SwiGLU with an explicit width, names `W_g, W_1, W_2` | 14.6 |
| `d2l.Qwen3GPT(d2l.GPT)` | 1.2 | `(vocab_size, num_hiddens, num_heads, num_kv_heads, num_blks, *, head_dim=None, qk_norm=True, rope_theta=1e6, ffn_hidden=None, norm_eps=1e-6, tie_embeddings=True, max_len=40960, bias=False)`; overrides `forward`, `forward_cached`; defines `generate_cached`; inherits `generate`. `d2l.qwen3_gpt_kwargs(cfg)` maps an HF config to these arguments. | 4.1 (`PackedGPT`), 6, 11, 12, 13, 14 |
| `d2l.Qwen3GPT` module names (R1) | 1.2 | `model.token_emb`; `model.blks[i].attention` with `W_q, W_k, W_v, W_o, q_norm, k_norm, head_dim, num_heads, num_kv_heads`; `model.blks[i].ffn` with SwiGLU `W_g, W_1, W_2` (`W_2(silu(W_g x) * W_1 x)`). Each attention also has `rope_theta`; RoPE is interleaved (1.2 permutes the HF rows at load). Norm names follow 1.2's mapping table. There are no `W_gate/W_up/W_down`. | 4.1, 6.3, 11.x, 12.x, 13.x, 14.x |
| `d2l.generate` | 1.2 | `(model, tokenizer, prompts, max_new_tokens=None, *, strategy='greedy', temperature=1.0, k=None, p=None, min_p=None, seed=0, eos_token_ids=None, stop=None, chat=False, enable_thinking=None, return_logprobs=False, batch_size=8) -> list[d2l.Generation]`. `Generation` has fields `text, token_ids, logprobs, finish_reason`. Consumers that need strings use `[g.text for g in ...]`. The book's old `generate(step_fn, prefix, num_tokens)` of `sec_decoding` is renamed `d2l.generate_ids` (a book-side edit, see README); there is no dispatch. | 2, 3, 4, 8, 9, 10 |
| `d2l.score_choices` | 1.4 | `(model, tokenizer, prompts, choices, norm='token')` | 2, 4, 7, 9 |
| `d2l.pass_at_k` | 1.4 | `(n, c, k) -> float` (unbiased) | 7, 8, 9, 10 |
| `d2l.extract_answer` | 1.4 | `(text, kind='gsm8k', strict=False) -> str \| None`, with `kind` in `{'gsm8k', 'boxed'}` | 8.5, 9.x |
| `d2l.bits_per_byte` | 1.4 | `(model, tokenizer, texts, ctx_len=2048, stride=None, batch_size=4) -> float` | 3.4, 4.x |
| `d2l.EvalSuite` | 1.5 | `EvalSuite(tasks, name='suite', batch_size=8, *, n=None, seed=0)`. `tasks` is a list of `d2l.EvalTask` objects or of task ids from `d2l.EVAL_TASKS`; `n` overrides each task's subsample size. `.run(model, tokenizer, *, seed=0, cache_dir=None, rerun=False) -> d2l.EvalResult`. | all |
| `d2l.EvalResult` | 1.5 | `.per_item` (DataFrame: `task, item_id, cluster, score`), `.scores(task=None)` (`np.ndarray` ordered by `item_id`, or a dict over tasks when `task` is None), `.summary()`, `.macro()`, `.compare(other) -> dict[str, BootstrapResult]`, `.save()`, `.load()` | all |
| `d2l.EVAL_TASKS` | 1.5 | Built-in task ids: `arc_challenge, hellaswag, mmlu, mmlu_pro, gsm8k, humaneval`, plus zero-shot small-model tasks `arc_easy, piqa, sciq, winogrande` | 4.2, 11, 14 |
| `d2l.paired_bootstrap` | 1.5 | `(scores_a, scores_b, n_boot=10000, cluster=None, *, seed=0, alpha=0.05) -> d2l.BootstrapResult`. `BootstrapResult` is a NamedTuple with `mean, ci=(lo, hi), se, p_two_sided, n, n_clusters, n_boot`. Consumers read `r.mean` and `r.ci`; it is not a 3-tuple. | all |
| `d2l.judge_pairwise` | 1.5 | `(judge, tokenizer, question, answer_a, answer_b, *, rubric=..., batch_size=8) -> d2l.JudgeResult` (or a list), with fields `pref_a, delta, position_bias, consistent, verdict, logits` | 2.5, 7, 8 |
| `d2l.ChatAPI` | 1.7 | `ChatAPI(base_url, model, api_key=None, *, cache=None, mode='replay', price=None, cost_cap_usd=None, timeout=120.0, max_retries=6, default_params=None, http_client=None)`; `.chat(messages, *, replica=0, stream=False, **params) -> d2l.ChatResult`, `async .achat(...)`, `.models()`, `.spent_usd`, `.manifest()`. `mode='replay'` raises `d2l.ReplayMiss` on a cache miss; a live call past `cost_cap_usd` raises `d2l.CostCapExceeded` | 1.5 (exercise), 2.5, 7.8 |
| `d2l.chat_batch` | 1.7 | `async (api, requests, *, concurrency=8) -> list[d2l.ChatResult \| Exception]`, with full-jitter backoff on 429 and 5xx | 1.5 (exercise), 2.5, 7.8 |
| `d2l.api_cost` | 1.7 | `(usage, price: d2l.PricePoint) -> float` (USD; `PricePoint(input, output, cached_input=None, per=1_000_000)`) | 7.8 |
| `d2l.embed_texts` | 2.3 | Text embedder used by 2.4–2.5. 3.2 keeps its own `d2l.embed_docs` under a different name on purpose. | 2.x |
| `d2l.book_corpus` | 2.4 | `(chunking='sections') -> d2l.BookCorpus`. Chunks have `id, doc_id, text, heading_path, char_start, char_end, n_tokens`. | 2, 4, 7, 10, 12, 13 |
| `d2l.BookRetriever` | 2.4 | `.search(query, k, mode=...) -> list[d2l.Hit]`. `Hit` is a NamedTuple with `chunk_id, score, rank, text, heading_path, scores`; it is not a dict. | 2.5, 7.9, 10.1 |
| `d2l.book_qa` | 2.5 | `(split='dev', answerable=None, types=None, n=None, seed=0) -> d2l.BookQA`. Splits are `train/dev/test`. Items have `qid, split, type, question, reference, key_facts, answerable, gold_spans, ...` (schema in 2.5). There is no `gold_ids` field; derive gold chunks with `d2l.gold_chunk_ids(item, corpus)`. | 2.3, 7.9, 10.4 |
| `d2l.rag_judge` | 2.5 | `(judge, tokenizer, item, result, context_texts) -> {'correct', 'faithful', 'supported'}`. Scoring answers to book QA goes through this. There is no `book_qa_score`. | 7.9 |
| `d2l.near_dedup` | 3.3 | `(texts, threshold=0.8, ngram=5, num_perm=128, seed=0) -> list[int]` (indices to keep). Convenience wrapper over `shingles`, `minhash_signature`, `lsh_candidates` and `union_find_clusters`. | 7.3 |
| `d2l.r2_tokenizer` | 3.6 | Byte-level BPE, vocabulary 32,768, EOS id 0, no BOS. Ids 0–31 are specials: `<|endoftext|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, then `<|reserved_4|>` … `<|reserved_31|>`; merges start at id 32 | 4.x |
| `d2l.write_shards`, `d2l.TokenShards` | 3.6 | Flat token `.bin` (`uint16` when V ≤ 65,536, i.e. every id ≤ 65,535; `uint32` otherwise) + `uint32` `.idx` + `meta.parquet` + `manifest.json` (`d2l-corpus/1`). EOS is appended per document. `.locate(i)` returns the resume cursor. | 4.1, 4.5 |
| `d2l.PackedStream` | 4.1 | Reads `TokenShards`. `state_dict()` stores the global consumed count and reloads under a different world size. | 4.x, 5.3, 6.x, 14.2 |
| `d2l.chunked_lm_loss` | 4.1 | `(hidden, weight, labels, chunk=8192, z_loss=0.0, ignore_index=-100) -> (sum_loss, n_valid)` | 7.1, 7.x, 8.x |
| `d2l.flops_per_token`, `d2l.mfu` | 4.2 | `flops_per_token(arch, T=None, causal_half=False) -> float`; `mfu(tokens_per_s, arch, peak_flops, n_gpus=1) -> float`. `arch` may be a `PretrainConfig.arch`, a Hugging Face config or a `d2l.ModelShape` (6.2); attributes are read by HF names (`num_hidden_layers, hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, vocab_size`) with the `PretrainConfig.arch` short names as fallback. Returns *training* FLOPs per token (forward + backward), `6 N_mm + 12 L d T` in the PaLM convention of 4.2 D1, with the attention term halved when `causal_half=True` and omitted when `T=None`; divide by 3 for forward-only (inference) FLOPs. | 5.2, 6.x, 7.8 |
| `d2l.load_reference_model` | 4.2 | `(step=None, dtype=..., device=..., preset='r2') -> d2l.PackedGPT` (model only; the tokenizer is `d2l.r2_tokenizer()`, 3.6); keys `lm-r2-config`, `lm-r2-s{step}` | 7, 11, 14 |
| `d2l.chip_specs` | 5.1 | DataFrame columns: `chip, vendor, kind, year, mem_type, mem_gb, mem_bw_tbs, sram_mb, p16_tflops, p8_tflops, p4_tflops, tdp_w, scaleup_gbs, scaleup_note, scaleup_domain, scaleout_gbs, sparsity_note, source_url, retrieved, notes`. Consumers use these names, not `bf16_tflops`, `hbm_gbps`, `peak_bf16_tflops`, `mem_bw_gbs` or `hbm_tbps`. | 5–8, 12–14 |
| `d2l.ServingModel` | 5.2 | `(cfg, chip, calib=None, dtype_bytes=2, kv_bytes=2, mem_util=0.9, kv_bytes_override=None)`, with `.decode_step(b, L)`, `.prefill_time(n, b=1)`, `.max_batch(L)`, `.b_star(L)`, `.L_crit()`, `.frontier(...)`. This is the decode step-time model `eq_lm_hw_decode_step`; there is no `d2l.decode_step_time`. | 12, 13, 14 |
| α–β tools | 5.3–5.5 | `d2l.fit_alpha_beta` (5.3); `d2l.parse_nccl_tests`, `d2l.nccl_busbw_factor`, `d2l.coll_time` (5.4); `d2l.ring_allreduce_time`, `d2l.tree_allreduce_time`, `d2l.hier_allreduce_time`, `d2l.a2a_inter_node_time` (5.5); `d2l.axis_ratio` (5.6) | 6.x, 13.4 |
| `d2l.launch` | 6.1 | `(script, nodes=1, gpus_per_node=8, **env) -> d2l.LaunchResult` | 4, 5, 7, 8, 9, 13 |
| `d2l.fsdp_wrap`, `d2l.fsdp_train_step` | 6.2 | FSDP2 wrapping and one training step | 7.8, 8.6 |
| `d2l.a2a_bytes` | 6.6 | All-to-all bytes per MoE layer | 13.4, 14.2 |
| `d2l.failure_rate` | 6.8 | `(gpus, source='llama3') -> failures per hour`; `source` is `'llama3'` (default, 3.79 per 1,000 node-days), `'rsc1'` (6.50), `'rsc2'` (2.34), or a float in failures per 1,000 node-days | 5.6 |
| `d2l.sft_train` | 7.2 | `(model, batches, optimizer, scheduler=None, loss_fn=None, grad_accum=1, max_grad_norm=None, log_every=10, callback=None)`, where `loss_fn(model, batch, num_items) -> Tensor`. There is no `SFTTrainer` class. Helpers: `render_chat`, `sft_collate`, `sft_loss`, `completion_logprobs`. | 8, 9 |
| `d2l.LoRALinear` | 7.4 | Parameters `lora_A [r, d_in]`, `lora_B [d_out, r]`, `scaling`, with aliases `A`, `B`, `scale`; `base`; `enabled` (bool); `merge()`, `unmerge()` | 8, 9, 10, 13 |
| `d2l.add_lora` = `d2l.apply_lora` | 7.4 | `(model, target_modules='all', r=8, alpha=16.0, dropout=0.0, rslora=False, exclude=('lm_head',), adapter_dtype=torch.float32) -> list[str]` (replaced module names; modifies in place) | 8, 9, 10 |
| `d2l.lora_disabled` | 7.4 | Context manager that turns every adapter off, giving the reference policy without a second copy of the model. There is no `adapters_disabled`. | 8.x |
| `d2l.quantize_blockwise`, `d2l.quantize_nf4`, `d2l.dequantize`, `d2l.QTensor` | 7.6 | `quantize_blockwise(w, bits=4, block_size=64, ...) -> QTensor` with fields `codes, scales, zeros, block_size, shape, bits`; `quantize_nf4` returns the same type; `dequantize(q, dtype=torch.bfloat16)`. It is not a `(codes, absmax)` tuple. | 11, 12 |
| `d2l.RewardModel` | 8.1 | `(backbone, hidden_size)`; `forward(input_ids, attention_mask) -> (B,)` reward at the last non-pad token | 8.x, 9.4 (pattern) |
| `d2l.token_logprobs` | 8.1 | `(model, input_ids, attention_mask, response_mask=None, temperature=1.0) -> (B, T-1)` | 8.x, 9.2 |
| `d2l.group_advantages` | 8.5 | `(rewards, std=True, loo=False, eps=1e-8)`; rewards have shape `(G, K)` | 9, 10.4 |
| `d2l.verify_math` | 8.5 | `(text, gold) -> float` in `{0.0, 1.0}`. It uses `d2l.extract_answer(text, kind='boxed')` (1.4), then 8.5's `d2l.canonical_answer`. There is no `math_reward`. | 9.x |
| `d2l.sample_responses` | 8.5 | `(engine, prompts, K, max_new_tokens, temperature=1.0, top_p=1.0, seed=0) -> {ids, resp_mask, logp_sampler, text, finished}`. There is no `RolloutEngine`. | 9.x, 10.4 |
| `d2l.grpo_train` | 8.5 | `(policy, tasks, cfg, reward_fn=None) -> list[dict]`. `cfg` fields: `K, prompts_per_step, steps, aggregation, std_norm, loo, eps_low, eps_high, beta, kl_estimator, dynamic_sampling, overlong, num_minibatches, engine, seed`. `reward_fn(texts, items) -> list[float]` is batch-level and defaults to `verify_math(text, item['answer'])` per item. A task item is a dict with `prompt` (chat messages or a string) and `answer` (str); `ArithmeticTasks` (8.5) and `make_problems` (9.1) both emit these fields. There are no `GRPOTrainer` or `GRPOConfig` classes. | 9.x |
| `d2l.sync_weights` | 8.6 | `(trainer, engine, mode) -> (seconds, bytes)` | 9.x |
| `d2l.kd_loss` | 9.1 | `(student_logits, teacher_logits, mask, kind='fkl', alpha=0.5, kappa=1.0, chunk=4096)` | 9.x, 11.6 |
| `d2l.pareto_front` | 9.7 | `(points, x='tokens', y='acc') -> DataFrame` | 10.5 |
| `d2l.FastGenerator` | 12.1 | The normative API in 12.1: `alloc/free/reset`, `forward/advance/rollback`, `prefill/decode_step/verify/commit`, `generate(prompts, max_new_tokens, *, sampling, eos_id, speculator, sync_every, return_stats)`, `warmup`, `stats`; per-slot `pos` and `kv_len`; `KVCache`, `KVPolicy`. It has no `generate_batch` or `capture`. | 4.7, 12.x, 13.x |
| `d2l.Speculator`, `d2l.speculative_generate` | 12.2 | Draft/verify interface | 12.3, 13 (A8) |
| `d2l.config_stats` | 14.1 | `(config) -> d2l.ConfigStats` (parameters, active parameters, FLOPs per token, cache bytes per token). It calls `config_params(by_group=True)` for dense parts and feeds `ServingModel(kv_bytes_override=...)`. | 14.x |

**Names added during the patch round.** Each one is owned by the listed section; none duplicates a name above.

| Name | Owner | Contract |
|---|---|---|
| `d2l.bpb_from_nll` | 3.4 | `(sum_nll_nats, n_bytes) -> float`; 3.4's helper next to 1.4's `bits_per_byte` |
| `d2l.fit_alpha_beta` | 5.3 | `(sizes, times, weights=None) -> (alpha, beta, cov)`; weighted least squares on `t = α + s/β` |
| `d2l.canonical_answer` | 8.5 | `(s) -> str \| None`; fractions, `\frac`, units and thousands separators |
| `d2l.grpo_train_with_reset` | 9.3 | `(policy, tasks, cfg, reset_every, reward_fn=None)`; runs `grpo_train` in segments with a reference reset |
| `d2l.group_advantages_by_id` | 10.4 | Scatters rewards keyed by group id into `(G, K)` and calls `group_advantages` |
| `d2l.agent_pareto` | 10.5 | Boolean-mask variant of `pareto_front` for cost/accuracy arrays |
| `d2l.KVManager` | 13.2 | Admission and accounting protocol over 12.1's `KVCache`; `SlotKV` (13.2) and `PagedKV` (13.3) implement it |
| `d2l.ServeStepModel`, `d2l.MoEServeStepModel` | 13.1, 13.4 | Fitted step-time models (`P_eff`, `β_eff`, `t_ovh`), as distinct from the analytic `ServingModel` (5.2) |

## 2. Duplicates resolved

| Name | Kept in | Removed from | Consumer change |
|---|---|---|---|
| `bits_per_byte` | 1.4 | 3.4 | 3.4 imports it. If 3.4 needs bits per byte from a summed NLL, it adds `d2l.bpb_from_nll(sum_nll_nats, n_bytes)`. |
| `extract_answer` | 1.4 | 8.5 | 8.5 renames its canonicalizer to `d2l.canonical_answer(s)`; `verify_math` composes the two. |
| `chunked_lm_loss` | 4.1 | 7.1 | 7.1 recaps it in one paragraph and imports it. |
| `mfu`, `flops_per_token` | 4.2 | 6.2 (`mfu`, `model_flops_per_token`) | 6.2 calls 4.2's functions with a `ModelShape`. |
| `apply_lora`, `adapters_disabled` | 7.4 (`apply_lora`, `lora_disabled`) | 8.1 | 8.1 imports them; call sites use 7.4's signature. |
| `token_logprobs` | 8.1 | 9.2 | 9.2 imports it. |
| `group_advantages` | 8.5 | 10.4 | 10.4 imports it; a variant keyed by group id would be named `group_advantages_by_id`. |
| `pareto_front` | 9.7 | 10.5 | 10.5 imports it; a 2-D cost-accuracy variant would be named `agent_pareto`. |
| `logit_kd_loss` | 9.1 (`kd_loss`) | 11.6 | 11.6 uses `d2l.kd_loss(..., kind='fkl')`. |
| `SwiGLUFFN` | `d2l.SwiGLU` (1.2) | 14.6 | 14.6 uses `d2l.SwiGLU(d, hidden)`. |
| `eq_lm_fsdp_mem`, `eq_lm_fsdp_comm`, `tab_lm_fsdp_comm` | 6.2 | 7.8 | 7.8 renames its own to `eq_lm_ft_fsdp_mem`, `eq_lm_ft_fsdp_comm`, `tab_lm_ft_fsdp_comm`, and cites 6.2's. |

## 3. Artifacts

DATA_HUB keys are `lm-<chapter-or-topic>-<name>` in hyphenated lower case, following the book's `gpt2-weights` style. All blueprints were normalized to this on 2026-10-28; legacy keys such as `ptb` are unchanged.

**Hosting (decided 2026-10-28).** Small artifacts live under `d2l.DATA_URL + 'lm/'`. The large R2 artifacts live in d2l-owned Hugging Face repositories: a model repository for the tokenizer (3.6), the checkpoints and the training state (4.2), and a dataset repository for the FineWeb-Edu shards, validation set and manifest (3.6), whose card carries the ODC-By 1.0 attribution. Their `DATA_HUB` entries are `('https://huggingface.co/[datasets/]<repo>/resolve/<commit>/<file>', sha1)` with a pinned commit hash, never `main`; the book's `gpt2-weights` entry already points at the Hub this way, minus the pin. The R2 run is a one-off T3 job whose outputs are uploaded once.

Cross-chapter artifacts:

| Key | Producer | Consumers | Content |
|---|---|---|---|
| `lm-book-r3-v1` | 2.4 | 2.x, 4, 7, 10, 12, 13 | R3 corpus (cleaned book text and chunks) |
| `lm-r2-tokenizer`, `lm-fwedu-val`, `lm-fwedu-shards-100m`, `lm-fwedu-manifest-4b` | 3.6 | 4.x, 6.x | R2 tokenizer and FineWeb-Edu shards |
| `lm-r2-config`, `lm-r2-s{500,…,6000}`, `lm-r2-s5100-state`, `lm-r2t1-s1526` | 4.2 | 7, 11, 14 | R2 checkpoints (about 4.8 GB in total) |
| `lm-hw-a100-microbench`, `lm-hw-a100-workloads`, `lm-hw-node-storage`, `lm-hw-node-p2p`, `lm-hw-t2pair-p2p`, `lm-hw-t2pair-net`, `lm-hw-cluster-nccl`, `lm-hw-prices`, `lm-hw-systems` | 5.x | 6.x, 7.8, 13.4, A5 | Microbenchmark logs. 5.3 commits tier bandwidths and checkpoint-stall records in `lm-hw-node-storage`. 5.4 and 5.5 commit fitted α–β summaries (`alpha_beta.json`) in `lm-hw-node-p2p` and `lm-hw-cluster-nccl`. There are no separate `lm-ckpt-io`, `lm-tier-bandwidth` or `lm-link-alpha-beta` keys. |
| `lm-mfu-table` | 6.7 | 5.2, 5.6 | Measured MFU per parallel configuration |
| `lm-sft-qwen3-0p6b` (on `lm-sft-mix-v0`); `lm-sft-qwen3-0p6b-v1` | 7.2; 7.3 | 8.x (A7), 9 | SFT checkpoint, chat template and mask idiom |
| `lm-lora-r1-demo` | 7.4 | 13.5 | r=8 LoRA adapter for Qwen3-0.6B |
| `lm-grpo-qwen3-0p6b`, `lm-grpo-qwen3-1p7b-base` | 8.5 | 9.3, 9.7 | GRPO arm A, seed 0: π_SFT + LoRA (T1/T2), and the R1-Zero-style replicate on Qwen3-1.7B-Base (T2) |

## 4. Labels

- Section labels: the registry in `00_conventions.md`.
- Subsection labels: `subsec_lm_<slug>`, unique across the part (checked).
- Assignment labels:

  | Assignment | Label | Section |
  |---|---|---|
  | A1 | `sec_lm_a1` | Chapter 1 |
  | A2 | `sec_lm_a2` | Chapter 2 |
  | A3 | `sec_lm_a3` | Chapter 3 |
  | A4 | `sec_lm_a4` | Chapter 4 |
  | A5 | `sec_lm_a5` | Chapter 5 |
  | A6 | `sec_lm_a6` | Chapter 6 |
  | A7 part 1 (SFT) | `sec_lm_a7_sft` | Chapter 7 |
  | A7 | `sec_lm_a7` | Chapter 8 |
  | A8 | `sec_lm_a8` | Chapter 13; Chapter 12 supplies its hooks |
  | Capstone | `sec_lm_capstone` | Chapter 14 |
  | Distillation mini-assignment | `sec_lm_a9_mini` | Chapter 9 |
  | Compression mini-assignment | `sec_lm_a11_mini` | Chapter 11 |

- Figure, table and equation labels carry a section-specific prefix, e.g. `fig_lm_hw_*`, `eq_lm_ft_*`.

## 5. Patch list (applied 2026-10-28)

Each chapter blueprint was patched to conform to sections 1–4 above. The owners' additions:
- 1.5 adds `EvalSuite(..., n, seed)` with string task ids, `EvalResult.scores()`, and the four zero-shot tasks.
- 1.1 adds `config_params(by_group=...)`.
- 1.2 adds `Qwen3Attention`, `SwiGLU`, `Qwen3GPT` (subclasses; the book's classes are unchanged) and defines `generate_cached` on `Qwen3GPT`.
- 3.6 reserves 32 special-token ids.
- 3.3 adds `near_dedup`.
- 3.6 states EOS-per-document and the `uint32` rule.
- 5.3–5.5 add the α–β and tier summaries.
- 8.5 adds `reward_fn`, `canonical_answer` and the two GRPO checkpoints.

The consumers' changes are listed in the "Consumer change" column of section 2 and implied by the signatures in section 1.
