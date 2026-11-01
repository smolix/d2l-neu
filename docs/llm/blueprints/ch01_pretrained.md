# Chapter 1: Working with Pretrained Language Models — Blueprint

**Label:** `chap_lm_pretrained` · **Arc:** I. Using and measuring · **Sections:** 1.1 `sec_lm_hf_models`, 1.2 `sec_lm_d2l_checkpoint`, 1.3 `sec_lm_prompting`, 1.4 `sec_lm_scoring`, 1.5 `sec_lm_eval_suites`, 1.6 `sec_lm_interpretability`, 1.7 `sec_lm_api` · **Tiers:** T1 throughout (T0 fallbacks noted per section); 1.7 is T0 by default (it replays committed API responses) with optional live calls and a T1 local arm · **Assignment:** A1

## Chapter overview

### Draft introduction (for the chapter's index page)

The Language Models part starts from released checkpoints, because every later chapter needs a capable model to adapt, compress, serve or measure. This chapter shows what such a model is and how to work with it honestly. We first use a current model the way practitioners do, through Hugging Face Transformers, and read its configuration file until every field maps to a section of this book. We then rebuild a released model (Qwen3-0.6B) as a small diff against the book's own `d2l.GPT` (three subclasses made through the hooks `d2l.TransformerBlock` already provides), so that every tensor is visible, and prove the port correct against Transformers, layer by layer. With a trusted model in hand we ask what its behavior depends on: the prompt format, the chat template and the demonstrations. We then ask how to score it, and the same checkpoints rank differently under different, equally defensible conventions. We build the evaluation harness that every later notebook imports, with paired comparisons and error bars, so that a difference is reported only if it survives a bootstrap. The chapter then opens the model: logit and tuned lenses, probes, activation patching and sparse autoencoders, each with an intervention or a control that can falsify the claim being made. A last section reaches the models too large for any tier here through an OpenAI-compatible API, and treats the client as a measuring instrument: latency, cost, rate limits and a response cache that lets the book replay what it measured.

### Learning objectives

After this chapter a reader can:

1. Find, load and pin a released model by commit hash with Transformers, and read its `config.json` and safetensors index well enough to reconcile its parameter count to the byte (1.1).
2. Predict, from a configuration alone, the weight, KV-cache, recurrent-state and activation memory of a dense or hybrid decoder, and check the prediction against measured allocator and cache sizes (1.1).
3. Port a released checkpoint as a diff against `d2l.GPT` (`d2l.Qwen3GPT`) by mapping weight names, layouts and RoPE conventions, and show that logits agree with the reference to a stated tolerance in FP32 and a noise-floor-relative tolerance in BF16; localize a seeded porting bug to its first diverging sublayer (1.2).
4. Explain, with a rendered token sequence, what a chat template and a thinking switch add to the input, and measure how number, order, format and label correctness of demonstrations change few-shot accuracy for base and post-trained checkpoints (1.3).
5. Implement log-likelihood scoring under three normalizations, the unbiased pass@k estimator and an execution-free answer extractor, and show a ranking flip between conventions (1.4).
6. Report a model comparison with a paired, optionally clustered bootstrap interval, and judge pairs with an LLM judge in both orders so that position bias is measured rather than assumed (1.5).
7. Test a mechanistic claim about a model by intervening on activations (patching, steering, SAE-feature ablation) and state what a probe or lens does and does not show (1.6).
8. Call a model that cannot run locally through the OpenAI-compatible chat protocol with one client for hosted and local endpoints; measure its latency, throughput under concurrency and cost from the client; make the run replayable through a request-hash cache; and say what the API removes (prompt log-probabilities, weights) and which terms of use apply (1.7).

### Prerequisites

| Book label | What this chapter uses |
|---|---|
| `sec_text-sequence` | Byte-level BPE, special tokens: read in 1.1 through Qwen's tokenizer, never re-derived |
| `sec_language-model` | Perplexity, bits per byte: the denominators discussed in 1.4 |
| `sec_decoding` | `d2l.sample_next`, `d2l.generate_ids(step_fn, ...)` (the sec_decoding loop, renamed in the book from `d2l.generate`): the sampling semantics that 1.1 maps onto Transformers' arguments, that 1.2 reuses and that 1.7 meets as request fields |
| `sec_gpt`, `sec_transformer-block` | `d2l.GPT`, `d2l.TransformerBlock` and its `attn_factory`/`ffn_factory` hooks, `d2l.FeedForward`, loading GPT-2, RMSNorm, SwiGLU, QK-norm (exercise there) |
| `sec_kv-cache` | `d2l.GQAAttention`, `forward_cached`, the KV-cache byte formula, prefill vs decode |
| `sec_positional-information` | RoPE (interleaved implementation) |
| `sec_scaling-laws` | `tab_modern-recipe` row for Qwen3 |
| `sec_hardware`, `sec_perf_model` | Decode bandwidth bound; `d2l.Benchmark` |
| `sec_read_write`, `sec_repro` | safetensors, seeds (1.7: what a seed promises when the sampler is a provider's) |
| `sec_deltanet`, `sec_hybrids` | Gated DeltaNet recurrent state, hybrid stacks: read in 1.1 as configuration fields only |
| `sec_what-attention-computes` | Residual stream, induction heads, the evidence-level ladder `tab_induction-evidence` |
| `sec_mdl-statistics` | Bootstrap, percentile intervals, paired sign-flip test |
| `sec_mdl-svd-low-rank` | Low rank: used in 1.6 only for the PCA view of steering directions |
| `sec_regularized` | Bradley–Terry model, used in 1.5 for rankings from pairwise judgments |
| `sec_softmax` | Calibration (1.3 contextual calibration, 1.4 calibration of choice probabilities) |

### Section dependency graph

```
1.1 HF models ──► 1.2 d2l checkpoint (R1) ──► 1.6 Interpretability
   │                    │
   └──► 1.3 Prompting ◄─┘ (HF model of 1.1 or d2l.generate of 1.2)
              │
              ▼
        1.4 Scoring ──► 1.5 Eval suites, judges, error bars ──► every later chapter (d2l.EvalSuite)
                                   │
                                   └──► 1.7 Models behind an API (d2l.ChatAPI) ──► any section that calls a hosted model
```

1.1 and 1.2 are a unit (A1 parts 1–2). 1.3 can be taught before 1.2. 1.4 → 1.5 is strict. 1.6 needs 1.2 only. 1.7 needs 1.4 (`extract_answer`, the GSM8K problem ids) and 1.5 (`paired_bootstrap`), uses 1.3's `fewshot_prompt`, and is independent of 1.6; it can be taught right after 1.5. Its local arm needs a running `transformers serve`, not 1.2.

### Running examples and shared artifacts

**Used:** R1 is produced here (1.2). Released checkpoints (all Apache-2.0 unless stated): `Qwen/Qwen3.5-2B` (primary in 1.1; `-0.8B` as the T0/T1 fallback, `-4B` as the option), `Qwen/Qwen3.5-2B-Base`, `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-1.7B-Base`, and the Qwen-Scope SAEs for Qwen3-1.7B-Base (1.6; the card says `license: other` (`qwen`) and adds a use-restriction Caution while the repo's LICENSE file is Apache-2.0; downloaded at run time, never committed; see 1.6). Reached over an API, not loaded (1.7): `Qwen/Qwen3.5-397B-A17B` (Apache-2.0) through the Hugging Face router, and for one optional exercise `openai/gpt-oss-120b` (Apache-2.0); `Qwen/Qwen3.5-9B` also serves as the local endpoint of 1.7.

**Produced (and imported by later chapters):**

| Artifact | Section | Consumers |
|---|---|---|
| `d2l.load_hf`, `d2l.config_memory`, `d2l.HF_REVISIONS` | 1.1 | every notebook that loads a checkpoint |
| `d2l.Qwen3Attention`, `d2l.SwiGLU`, `d2l.Qwen3GPT` (subclasses of the book's `GQAAttention`, `FeedForward`, `GPT`), `d2l.load_qwen3`, `d2l.generate` | 1.2 | R1 users: 4.2, 7.x, 8.x, 9.x, 11.x, 12.1 |
| `d2l.score_choices`, `d2l.pass_at_k` | 1.4 | 1.5, 4.x (eval during training), 7.x–9.x |
| `d2l.EvalSuite`, `d2l.paired_bootstrap`, `d2l.judge_pairwise` | 1.5 | every notebook that reports a metric difference |
| `d2l.ChatAPI`, `d2l.ChatResult`, `d2l.chat_batch`, `d2l.backoff_delay`, `d2l.api_cost`, `d2l.PricePoint`, `d2l.price_from_listing`, `d2l.ReplayMiss`, `d2l.CostCapExceeded` | 1.7 | every section that uses a model too large to run locally (`00_conventions.md`, "Models"); 3.5, 7.3 and 9.1 add the terms-of-use check of 1.7 before an API model generates training data |
| `DATA_HUB` keys listed in each section (suite manifests, sample pools, tuned-lens translators, small SAE, API response caches `lm-api-*`) | 1.3–1.7 | committed so that analysis cells run offline |

### Compute plan

Hardware: T1, one 24 GB Ada GPU (RTX 4090 class, the readers' floor); budgets are for a 15 GiB build-box slot unless a row says the notebook claims a whole GPU (the build box is 4× RTX 4090 without NVLink: no multi-GPU run in this chapter). Every wall-clock figure below is "to be measured on the build box" and the author records it in the section's header cell; the table gives only the dominant memory term, which is derived.

| § | Tier | Models (BF16 weights) | Dominant GPU memory | GPU time | Committed artifacts |
|---|---|---|---|---|---|
| 1.1 | T1 (T0 with Qwen3.5-0.8B for all non-timing cells) | Qwen3.5-2B (4.55 GB file), 2B-Base, Qwen3-1.7B | weights 4.55 GB + KV and state (derived in 1.1) | to be measured | `data/lm_revisions.json`, config-reading table |
| 1.2 | T1 | Qwen3-0.6B twice (HF and `d2l.GPT`), FP32 gate: 2 × 2.4 GB | ~5 GB in FP32 gate | to be measured | `qwen3_0.6b_logit_reference.pt` (logits of 8 fixed prompts, FP32) |
| 1.3 | T1 | Qwen3-1.7B-Base and Qwen3-1.7B (3.4 GB each, sequentially) | ~4 GB | to be measured | per-condition prediction tables (`lm-fewshot-grid.csv`) |
| 1.4 | T1 | Qwen3-0.6B-Base, Qwen3-1.7B-Base, Qwen3-1.7B | ~4 GB | to be measured | cached sample pools for pass@k (`lm-gsm8k-samples.jsonl`) |
| 1.5 | T1 | the 1.4 models plus judge Qwen3.5-9B (17.9 GB, loaded after the answerers are freed) | ~20 GB (a whole 24 GB GPU) | to be measured | judgments (`lm-judge-pairs.jsonl`) |
| 1.6 | T1 | Qwen3-0.6B (R1), Qwen3-1.7B-Base + one 537 MB SAE layer | ~4.5 GB | to be measured | tuned-lens translators, probe weights, small SAE |
| 1.7 | T0 (replay); T1 for the live local arm | hosted `Qwen3.5-397B-A17B` over an API (no local memory); local `Qwen3.5-9B` BF16 behind `transformers serve` | none at T0; ~20 GB live (a whole 24 GB GPU) | T0 ~2 min; live ~15–25 min, to be measured | four response caches `lm-api-*`, listing snapshot, probe matrix |

Tier counts: 1.1–1.6 are T1; 1.7 is T0 by default (replay) with a T1 live arm and no T2+ run. Live API calls need a token held outside the repository and a stated cost cap; replay needs neither.

---

## Assignment A1: Run, port, and measure

**Label:** `sec_lm_a1`

**Title.** "Qwen3.5 through Transformers; Qwen3 into `d2l.GPT`; an evaluation harness with error bars."

**Tier.** T1 (one 24 GB GPU; the reference solution fits a 15 GiB build slot). Every model in the assignment has 0.8B parameters or fewer except one optional run. A CPU-only path exists for Tasks 1 and 2 with a 3-prompt, 32-token test set (it exercises the same code, minus timing).

**Estimated student time.** 12–16 hours (Task 1: 2 h, Task 2: 5 h, Task 3: 4 h, Task 4: 2 h, report: 2 h).

### Starter code (provided as a package `a1/`)

- `a1/hf_tools.py`: `safetensors_header(model_id, revision)` reading headers by HTTP range request (the instructor's version of the notebook's helper), `cache_nbytes(cache)`.
- `a1/port.py`: stubs `map_qwen3_state_dict(hf_state, cfg) -> dict`, `first_divergence(ref_hidden, our_hidden, rtol) -> (layer, sublayer)`, and the `Qwen3Attention`/`SwiGLU`/`Qwen3GPT` signatures from 1.2 (docstrings only).
- `a1/harness.py`: stubs `score_choices`, `pass_at_k`, `paired_bootstrap`, `EvalSuite` with the exact signatures of 1.4 and 1.5.
- `a1/fixtures/`: `qwen3_0.6b_logit_reference.pt` (FP32 reference logits for the eight fixed prompts), `lm_eval_reference_arc.json` (lm-evaluation-harness choices for 200 ARC-Challenge items under `acc` and `acc_norm`), three corrupted state dicts (hidden; used only by the autograder).
- `a1/autograder.py`: runs the checks below on a student submission and prints pass/fail with the measured quantity.

### Tasks and deliverables

**Task 1 (1.1): Pin, load, account.** Deliverable `task1.py` plus table.
1. Load `Qwen/Qwen3.5-0.8B` pinned to the SHA in `a1/revisions.json` with `d2l.load_hf`; print config-field to section map for the 12 fields of the assignment sheet.
2. Implement `config_memory(config, seq_len, batch_size)` for the model's `text_config` (student version; instructor reference is `d2l.config_memory`).
3. Reconcile the safetensors header total to the sum of the three buckets (language model, vision tower, MTP).
4. Measure KV-plus-state bytes for contexts {128, 1024, 4096} with batch 1 and compare.

**Task 2 (1.2): Port.** Deliverable `port.py` plus three-bug localization table.
1. Implement `map_qwen3_state_dict` for Qwen3-0.6B (tied embedding; RoPE half-to-interleaved row permutation of `q_proj`, `k_proj`, `q_norm`, `k_norm`).
2. Match logits on the eight fixed prompts in FP32 and BF16.
3. Implement `first_divergence`; run it on the three hidden corrupted state dicts and report the layer and sublayer for each.
4. Greedy-decode 64 tokens for 8 prompts with `d2l.generate` on the port and with Transformers; report the first index of divergence per prompt.

**Task 3 (1.4–1.5): Harness.** Deliverable `harness.py` plus unit tests.
1. `pass_at_k(n, c, k)` (vectorized over NumPy arrays of `n`, `c`).
2. `score_choices(..., norm in {'sum','token','char'})` and the lm-evaluation-harness agreement check.
3. `paired_bootstrap(scores_a, scores_b, n_boot, cluster)` with item-level and cluster-level resampling.
4. `EvalSuite` running one multiple-choice task (ARC-Challenge, 1172 test items) and returning per-item scores.

**Task 4 (1.5): Reproduce a published number.** Run Qwen3-0.6B-Base on MMLU (5-shot, 14,042 test questions, `cais/mmlu` subset `all`, split `test`; few-shot demonstrations from the `dev` split of the same subject) with your harness and report the score with a 95% bootstrap interval. The Qwen3 technical report lists 52.81 for this model and setting (Table 8 of arXiv 2505.09388, which states 5-shot for MMLU).

### Autograder checks (exact tolerances)

| # | Check | Tolerance |
|---|---|---|
| 1 | `sum(header tensor bytes)` equals `config_params`-derived weights for the three buckets | exact (integers) |
| 2 | KV plus recurrent-state bytes from `config_memory` vs the bytes of tensors in the live cache, contexts {128, 1024, 4096} | exact, if the student set the state dtype per the cache; otherwise relative difference ≤ 1e-6 after the dtype is fixed |
| 3 | FP32 logits of the port vs `a1/fixtures` reference, max absolute difference over all positions and vocabulary entries | ≤ 1e-3 (provisional; the author of the autograder replaces it by 3 × the measured CPU-vs-GPU FP32 difference of Transformers on the same prompts if that is larger) |
| 4 | BF16: mean absolute logit error of the port vs FP32 reference | ≤ 1.5 × the same quantity for Transformers in BF16 (the noise-floor rule of 1.2) |
| 5 | `first_divergence` on the three hidden corrupted state dicts | the exact (layer, sublayer) for 3 of 3 |
| 6 | Greedy equality: port in FP32 vs Transformers FP32, 64 tokens, 8 prompts | identical token ids on at least 7 of 8 prompts (a tie at a near-equal top-2 is permitted on one) |
| 7 | `pass_at_k` on cases n=5,c=2,k=1 → 0.4; n=10,c=0 → 0; n=10,c=10,k=3 → 1; n=4,c=2,k=3 → 1; n=200,c=37,k=10 against `1 - comb(163,10)/comb(200,10)` | absolute error ≤ 1e-12 |
| 8 | `score_choices` vs lm-evaluation-harness choices on the 200 ARC items, norms `sum` and `char` | at most 2 of 200 items differ per norm (provisional: BF16 near-ties) |
| 9 | `paired_bootstrap`: 95% interval of the mean paired difference on a seeded synthetic set (n=500) vs the paired-t interval | each endpoint within 5% of the interval's width |
| 10 | `paired_bootstrap` with `cluster=`: synthetic set of 50 clusters of 10 items with intra-cluster correlation 1/3, so the design effect is 1 + (10 - 1)/3 = 4 and the standard error is inflated by 2 by construction | clustered interval width / iid interval width within 10% of 2.0 |
| 11 | MMLU: the published value (52.81) lies inside the student's 95% interval, or inside the interval of the lm-evaluation-harness reference run supplied in `a1/fixtures/` | pass if either holds |

### Written analysis (two pages)

1. Which one of your three bugs was the hardest to see in BF16 and why? Use the layer-error curves.
2. The post-trained Qwen3-0.6B file has 751,632,384 tensor elements, the Base file 596,049,920. Explain the difference and say what `model.num_parameters()` returns for each.
3. For Qwen3.5-0.8B, at what context length does the hybrid's KV-plus-state cache equal that of a counterfactual all-attention stack of the same width? Show the algebra.
4. Your MMLU score differs from the published 52.81 by Δ. Is Δ inside your interval? List three protocol differences (prompt, answer extraction, normalization) that could explain a gap, and measure one of them.
5. You compare two scoring conventions on ARC-Challenge. Report the paired difference with interval, and say whether a ranking flip between two checkpoints, if you found one, survives the paired bootstrap.

### T1 path

Everything above fits a 15 GiB build-box slot, and so a 24 GB card with room to spare: Qwen3.5-0.8B (1.75 GB), Qwen3-0.6B twice in FP32 (2 × 2.4 GB), MMLU on Qwen3-0.6B-Base. MMLU may be run on a fixed 20% subset (2,808 questions) during development; the final report uses the full test split. The 2B/4B Qwen3.5 variants and GSM8K 4-shot (published 59.59 for Qwen3-0.6B-Base, same table) are stretch goals.

---

## Teaching sources

All of v5's sources are kept; the annotation says what to take from each.

- [CS336](https://cs336.stanford.edu/) (evaluation lectures): the course's treatment of likelihood vs generative evaluation, and its habit of reporting what was held fixed; the model for 1.4's "convention table".
- [HF LLM Course](https://huggingface.co/learn/llm-course/chapter1/1), chapter 2: the Transformers API tour; assign before 1.1 for readers new to Auto classes. 1.1 goes further on revisions, configuration and memory.
- [NeurIPS 2024 tutorial on evaluating LLMs](https://neurips.cc/virtual/2024/tutorial/99524) and [ICML 2024 tutorial on LM evaluation](https://icml.cc/virtual/2024/tutorial/35227): the overall evaluation landscape; pair with 1.5's suite taxonomy.
- Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/): the practitioner argument for a small development suite built from real failures; 1.5's "development vs reporting".
- Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): batch-dependent kernels as the source of run-to-run drift; the basis of 1.4's "batch-dependent nondeterminism" subsection and of the batch-size sweep there.
- [ARENA](https://github.com/ARENA-education/ARENA_materials) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens): the interpretability curriculum and library that 1.6 follows in spirit (logit lens, patching, SAEs); 1.6 implements the operations with plain hooks so that the code is not tied to a library that lacks Qwen3.5 support.

New (verified 28 Oct 2026, listed per section below): the Transformers Qwen3.5 and Qwen3 model files, the Hub download guide, the MMLU/Open-LLM-Leaderboard post on format sensitivity, and the interpretability papers cited in 1.6. New (verified 29 Oct 2026, 1.7): the Hugging Face Inference Providers documentation (router, chat-completion specification, pricing), the Transformers and vLLM serving documentation, the AWS note on exponential backoff and jitter, and two papers on auditing which model an API serves.

## Sections

### 1.1 Pretrained models with Hugging Face

- **Label:** `sec_lm_hf_models` · **Tier:** T1 (reduced T0 config: `Qwen/Qwen3.5-0.8B` on CPU for every cell except the timing and memory-measurement cells; flag `MODEL = '0.8B'`) · **Budget:** T1 about 15 min wall-clock, peak GPU memory about 4.55 GB weights plus the derived cache (see the derivation; at 16k context the cache is under 0.3 GiB, so about 6 GiB peak with the logits workspace); T0 about 10 min · **Codex units:** the opening walkthrough (Codex lines 34–41; outside the numbered units)
- **Builds on (book):**
  - `sec_text-sequence`: byte-level BPE and special tokens. Reused as vocabulary: the notebook prints Qwen3.5's 248,320-entry vocabulary vs Qwen3's 151,936 and tokens per character for English, code and Chinese. The BPE algorithm is not re-taught.
  - `sec_decoding`: `d2l.sample_next(logits, strategy, temperature, k, p, min_p, rng)`. Reused as the semantic reference for Transformers' sampling arguments (table `tab_lm_hf_sampling_map`). Not re-taught.
  - `sec_read_write`: safetensors as a file format. Reused: the notebook reads the header without loading tensors. The format itself is not re-taught.
  - `sec_hardware`, `sec_perf_model`: bandwidth bound for decode, `d2l.Benchmark`. Reused: the decode ceiling is computed with the formula of `subsec_kv-cache-prefill-decode`.
  - `sec_kv-cache`: eq. `eq_kv-cache-bytes` is recapped in one paragraph and extended, not re-derived.
  - `sec_deltanet`, `sec_hybrids`: Gated DeltaNet state and the hybrid design rules are read as configuration fields; the recurrence is not re-taught.
- **Builds on (this part):** none. Forward pointers: 1.2 (R1 from Qwen3), 12.3 (MTP head), 13.1 (serving), 14.4 (gated attention, partial RoPE), 14.6 (hybrids at scale).
- **Exports (`#@save`):**
  - `d2l.HF_REVISIONS: dict[str, str]`: model id to 40-character commit SHA, loaded from `data/lm_revisions.json` (committed, table in "Models and data").
  - `d2l.load_hf(model_id, revision=None, dtype=torch.bfloat16, device_map='auto') -> (model, tokenizer)`. Contract: `revision=None` means `d2l.HF_REVISIONS[model_id]` if present, otherwise resolve the current head SHA through `HfApi().model_info(model_id).sha`, print it with the line `unpinned: add to data/lm_revisions.json`, and continue. Always passes `revision` to both model and tokenizer, `trust_remote_code=False`, `use_safetensors=True`. Uses `AutoModelForCausalLM`. `dtype` accepts a `torch.dtype` or `'auto'`. `device_map='auto'` with no GPU resolves to CPU. After loading, asserts that `model.hf_device_map`, if present, maps all modules to one device and warns otherwise (silent CPU offload is a performance trap). Returns the model in eval mode. Edge cases: gated repo without token raises with the repo URL; a revision that does not exist raises `RevisionNotFoundError` unchanged.
  - `d2l.config_params(config, include_vision=False, include_mtp=False, by_group=False) -> int | dict[str, int]`: parameter count of a dense decoder from its configuration, with the hybrid layer types below; with `by_group=True` it returns a dict of per-group counts that sum exactly to the total, with the groups `embed` (token embedding), `attention` (full-attention blocks, including `q_norm`/`k_norm`), `linear_attention` (Gated DeltaNet blocks; 0 for a dense config), `mlp`, `norm` (block norms and the final norm), `lm_head` (0 when tied), `vision` and `mtp` (each 0 unless its flag is set); 14.1's `d2l.config_stats` calls it for the dense parts; `config` is a `PretrainedConfig` or a plain dict; if it has `text_config`, that is used. Mixture-of-experts configs (any of `num_experts`, `n_routed_experts`) raise `NotImplementedError('use d2l.config_stats (14.1)')`.
  - `d2l.config_memory(config, seq_len, batch_size, dtype_bytes=2, *, state_bytes=4, conv_bytes=None, include_vision=False, include_mtp=False, logits_to_keep=1) -> dict[str, int]`. Keys, all in bytes except `params`: `params` (count), `weights`, `kv_cache`, `recurrent_state`, `activations`, `logits`, `total` (the sum of the five byte terms). `dtype_bytes` is the bytes per weight and per cached key/value element and per activation element; `state_bytes` is the dtype of the recurrent matrix state (4: the config's `mamba_ssm_dtype` is `float32`); `conv_bytes` defaults to `dtype_bytes`. Layer kinds are read from `layer_types` (`'full_attention'`, `'sliding_attention'` with `sliding_window`, `'linear_attention'`); a config without `layer_types` is all full attention. Sliding layers cache `min(seq_len, window)` positions. `head_dim` defaults to `hidden_size // num_attention_heads`; `num_key_value_heads` defaults to `num_attention_heads`; `attn_output_gate` doubles the query projection. `seq_len` greater than `max_position_embeddings` emits a warning and still computes. `batch_size` scales every term except `weights`. `logits_to_keep=None` counts all `seq_len` positions in the logits term.
  - `d2l.safetensors_header(model_id, revision=None) -> dict[str, tuple[tuple[int, ...], str]]`: tensor name to (shape, dtype string) for every shard, reading only the JSON header of each file by two HTTP range requests (8-byte length, then the header). No tensor data is downloaded.
- **Models and data:**

  | Model | Size (safetensors total) | Revision (resolved 28 Oct 2026) | License |
  |---|---|---|---|
  | `Qwen/Qwen3.5-2B` | 2,274,069,824 params, BF16 | `15852e8c16360a2fea060d615a32b45270f8a8fc` | Apache-2.0 |
  | `Qwen/Qwen3.5-2B-Base` | 2,274,069,824 | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | Apache-2.0 |
  | `Qwen/Qwen3.5-0.8B` | 873,438,784 | `2fc06364715b967f1860aea9cf38778875588b17` | Apache-2.0 |
  | `Qwen/Qwen3.5-4B` (option) | 4,659,865,088 | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Apache-2.0 |
  | `Qwen/Qwen3-1.7B` (thinking default, comparison) | 2,031,739,904 | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0 |

  The author re-resolves these on the authoring date and commits `data/lm_revisions.json`; the SHAs above are the values at blueprint time. No datasets are needed; the five prompts of the chat comparison are committed in the notebook (two knowledge questions, one arithmetic word problem, one code request, one instruction with a length constraint).
- **Packages:** `transformers >= 5.2.0` (the first tag checked in which `models/qwen3_5` exists; `v5.0.0` lacks it; PyPI's latest is 5.21.0 on 28 Oct 2026; the author records the installed version in the notebook header), `huggingface_hub`, `safetensors`, `accelerate`, `torch >= 2.5`. Optional: `flash-linear-attention` and `causal-conv1d` for the fast Gated DeltaNet path; the notebook prints which callable is bound (`chunk_gated_delta_rule` fast vs `torch_chunk_gated_delta_rule` fallback) and runs either way.

#### Narrative arc

1. **The question.** Opening cell: `model, tok = d2l.load_hf('Qwen/Qwen3.5-2B')`, then print `model.num_parameters()`, the safetensors total (2,274,069,824) and the model card's "2B". The printed symptom: three numbers that disagree, and a chat reply. "What is in this checkpoint, what does it cost to run, and what pins the answer to a version?"
2. **The Hub as a package registry** (`subsec_lm_hub`).
   - A model repo is a git repo: `config.json`, `generation_config.json` (absent for Qwen3.5-2B; present for Qwen3-0.6B with `temperature 0.6, top_k 20, top_p 0.95`), `tokenizer.json`, `chat_template.jinja`, shards `model.safetensors-0000i-of-0000n.safetensors` and `model.safetensors.index.json` (`weight_map`: tensor name to shard). Cell: `HfApi().model_info(id, files_metadata=True)` printing file sizes; `d2l.safetensors_header` reading 632 tensor shapes of Qwen3.5-2B with two small requests.
   - Revisions: branch names move, commit SHAs do not. Cell: load `Qwen/Qwen3-0.6B` at its pinned SHA, show `model.config._commit_hash`, and show that the tokenizer and the chat template come from the same commit. `tab_lm_revisions` lists the pinned SHAs. Rule: pin model and tokenizer together; record the `transformers` version too, because modeling code lives in the library and not in the repo.
   - Licenses and gating: read the card's `license`, `license_link`; note which repos are gated (`gated: auto` e.g. the GPQA dataset) and which have a non-Apache license (the Qwen-Scope SAE repos are tagged `license: other` / `qwen`, add a use-restriction "Caution" to the card, and carry an Apache-2.0 text only in the LICENSE file, so the card and the file disagree; 1.6). `trust_remote_code=True` executes Python from the repo at load time: never for this book; all models used here have in-library code. This absorbs "Pin the Identity" and "Trust and Licenses" from the retiring Tools appendix.
3. **Auto classes and what `from_pretrained` does** (`subsec_lm_auto`).
   - `AutoConfig` reads `model_type` (`qwen3_5`), the class registry maps it, weights are materialized on the target device in the requested `dtype` (`dtype=` in Transformers v5; `torch_dtype=` is the deprecated spelling), tied weights are tied after loading. `AutoModelForCausalLM` maps `qwen3_5` to `Qwen3_5ForCausalLM`, the text-only class: the repo's architecture is `Qwen3_5ForConditionalGeneration`, so the vision tower (`model.visual.*`) and the MTP head (`mtp.*`) are in the file but not in the loaded model. The cell prints the load report and the class name; the expected parameter buckets are derived below and the cell asserts them.
   - `AutoTokenizer`: no BOS token (`add_bos_token: false`, `bos_token: null` for Qwen3), `pad_token`, `eos_token`; the cell encodes an English sentence, a Python line and a Chinese sentence and prints tokens per character for both vocabularies (Qwen3: 151,936 rows; Qwen3.5: 248,320; numbers to be measured). This is the only tokenizer content here.
   - `device_map` and `dtype`: what `'auto'` does with one GPU, with no GPU, and when the model does not fit (offload to CPU, with a slowdown that the cell measures once on the 4B model only if memory forces it; otherwise skipped).
4. **`generate` and its arguments, mapped to `sec_decoding`** (`subsec_lm_generate`).
   - Table `tab_lm_hf_sampling_map`: HF argument to book function: `do_sample=False` to `strategy='greedy'`; `temperature` to `temperature`; `top_k` to `k`; `top_p` to `p` (HF drops tokens whose cumulative probability is at most `1 - p`, the book keeps the smallest head with mass at least `p`: identical except on exact ties); `min_p` to `min_p`; `num_beams` to `d2l.beam_search`; `repetition_penalty` has no book counterpart; presence penalty is recommended by the Qwen3.5 card for server frameworks and is not a `generate` argument: the author verifies against the installed version and states the result.
   - Cell: greedy decoding of 32 tokens with `model.generate` equals the loop `d2l.generate_ids(step_fn, prefix, 32)` (the loop of `sec_decoding`, renamed `generate_ids` in the book) where `step_fn(ids)` runs one forward on the HF model (it will be slow; that is the point, quadratic recompute vs `past_key_values`); token ids identical. The default `generation_config` silently turns on sampling for Qwen3-0.6B: the cell shows that omitting `do_sample=False` yields different outputs on two calls, and that `model.generation_config` explains it.
5. **Base, post-trained and thinking variants; chat templates** (`subsec_lm_variants`).
   - Render `tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` for Qwen3.5-2B: the default ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n` (non-thinking), with `enable_thinking=True` it ends with `<think>\n`. For Qwen3-1.7B the default is the opposite (thinking on unless `enable_thinking=False`). Both rendered strings are printed with special tokens visible, plus their token ids. (Chat templates as tokenized input, with roles and system prompts, are the subject of 1.3; here only the switch is shown.)
   - Same five prompts to `Qwen3.5-2B-Base` (raw text, no template) and `Qwen3.5-2B` (chat template): the base model continues, the post-trained model answers. No scoring: a qualitative table `tab_lm_base_vs_post`.
   - Thinking on vs off: tokens generated, wall time, and fraction of generations that hit `max_new_tokens=1024` without `</think>`, on the five prompts at the card's thinking-mode settings (`temperature 1.0, top_p 0.95, top_k 20`) and at greedy. The Qwen3.5-2B card warns that this size "is more prone to entering thinking loops"; the table reports what happens (to be measured; five prompts give a rate that is not a measurement of loop frequency, and the cell says so).
6. **Reading a hybrid configuration** (`subsec_lm_config`).
   - Table `tab_lm_qwen35_config`, fields of Qwen3.5-2B `text_config`, each with value, meaning and the section that teaches it (full table below under "Figures and tables").
   - The layer layout: `layer_types` is `3 x linear_attention, 1 x full_attention` repeated 6 times; `full_attention_interval: 4`. Cell prints the 24-element list compactly and counts 18 and 6. The model card's "6 x (3 x (Gated DeltaNet then FFN) then 1 x (Gated Attention then FFN))" is the same statement.
   - Side by side with the dense Qwen3-0.6B `config.json` (28 layers, 16/8 heads, `head_dim 128`, `intermediate_size 3072`, `rope_theta 1e6`, `tie_word_embeddings true`): note `head_dim` is not `hidden_size / heads` (1024/16 = 64 vs 128): the reason 1.2 needs a `head_dim` argument in `d2l.Qwen3Attention`.
   - Reconcile the parameter count with the safetensors header (derivation D1.1a). Buckets: language model 1,881,825,088 (embedding 508,559,360 plus 18 linear layers at 58,814,624, plus 6 full layers at 52,433,408, plus final norm 2,048), vision tower 331,416,576, MTP 60,828,160; sum 2,274,069,824, equal to the Hub's `safetensors.total`. Then the dense contrast: Qwen3-0.6B-Base file 596,049,920 vs post-trained file 751,632,384, whose extra 155,582,464 is a stored `lm_head.weight` that duplicates the tied embedding (`311` tensors = 28 x 11 + 3).
7. **Memory and speed** (`subsec_lm_memory_speed`).
   - Derivation D1.1b below. Cell prints `d2l.config_memory` for Qwen3.5-2B at batch 1 and contexts {1k, 4k, 16k, 64k, 262,144}; KV for 262,144 tokens is 3.0 GiB; recurrent state is a constant 18.6 MiB per sequence.
   - Measured side: after prefill, walk the live cache object, sum `tensor.nbytes` over key/value tensors and over recurrent and conv states, and compare exactly; then `torch.cuda.memory_allocated` growth (includes allocator rounding and transient workspace) and `max_memory_allocated` for the prefill (includes the logits).
   - Speed: prefill tokens/s and decode tokens/s with `d2l.Benchmark`, against the decode ceiling `BW / bytes_read_per_step` where `BW` is measured with a 1 GiB device-to-device copy (as `sec_hardware` does) and also quoted from the GPU's datasheet (the author cites the build box GPU's datasheet number; not stated here). Bytes read per decode step: the language-model weights (the tied embedding is read in full as the output head), plus the KV read for the 6 attention layers, plus the read and write of 18 recurrent states.
8. **What Transformers is for, and where it stops** (`subsec_lm_limits`).
   - One paragraph each: eager-mode Python decode loop overheads (the measured decode rate is below the ceiling; 12.1 shows CUDA graphs), `generate`'s static vs dynamic caches (link to `cache_implementation` and the docs page on cache strategies), high-throughput serving (vLLM, SGLang: 13.1, 13.2), and the book's own code (1.2) as the path when every tensor must be exposed (1.6) or modified (7.x, 11.x, 14.x).

#### Derivations

**D1.1a: Parameter count from a configuration** (`eq_lm_hf_params`).
- *Statement.* For a decoder with `L_f` full-attention layers and `L_l` linear-attention layers, all with SwiGLU MLPs and tied embeddings, the language-model parameter count is `P = V d + L_f (A_f + M + 2d) + L_l (A_l + M + 2d) + d`, with the terms below.
- *Assumptions.* RMSNorm has `d` weights and no bias; no projection has a bias (`attention_bias: false`); `lm_head` is tied to the embedding (counted once); MTP and vision excluded.
- *Steps.*
  1. SwiGLU: `M = 3 d f` (gate, up, down).
  2. Full attention with an output gate: `A_f = d (2 H d_h) + 2 d (H_kv d_h) + (H d_h) d + 2 d_h`; the `2 H d_h` is the query projection that also emits the gate; the last term is the `q_norm` and `k_norm` weights. Without the gate the first term is `d H d_h`.
  3. Gated DeltaNet: with key width `K = H_k d_k`, value width `U = H_v d_v` and conv channels `C = 2K + U`: `A_l = d C + d U + 2 d H_v + C k_conv + 2 H_v + d_v + U d`, i.e. fused q/k/v input projection, `z` gate projection, the two per-head scalar projections `a` and `b`, the depthwise convolution (`k_conv = 4`), `A_log` and `dt_bias` per value head, the gated norm weight over `d_v`, and the output projection.
  4. Substitute Qwen3.5-2B: `d = 2048, f = 6144, H = 8, H_kv = 2, d_h = 256, H_k = H_v = 16, d_k = d_v = 128, k_conv = 4, V = 248,320`: `A_f + M + 2d = 52,433,408`, `A_l + M + 2d = 58,814,624`, `P = 508,559,360 + 6 x 52,433,408 + 18 x 58,814,624 + 2,048 = 1,881,825,088`.
  5. Add the vision tower (331,416,576, from the header: 24 blocks of width 1024 plus patch embedding, position table and merger) and MTP (one full-attention block 52,433,408, `fc` 8,388,608, three norms 6,144): 60,828,160.
- *Check cell.* `d2l.config_params` for the three settings equals 1,881,825,088, 2,213,241,664 (language plus vision) and 2,274,069,824 (all), and the last equals the `safetensors.total` from the Hub API; for Qwen3-0.6B it returns 596,049,920 and 751,632,384 with `tie=False` semantics (head counted separately).

**D1.1b: Memory of a forward pass** (`eq_lm_hf_kv`, `eq_lm_hf_state`, `eq_lm_hf_peak`).
- *Statement.* For batch `B` and context `n`, in bytes: weights `W = b_w P`; KV `= 2 B n L_f H_kv d_h b_kv` (sliding layers: `n -> min(n, w)`); recurrent state `S = B L_l (H_v d_k d_v s + C (k_conv - 1) b_c)` independent of `n`; transient activations for a prefill of `n` tokens (inference, fused attention, no stored graph) `Act = B n b (2d + 3f + (H + 2H_kv) d_h)` as an upper envelope for one layer's live tensors plus the residual, and the logits term `B n_out V b`.
- *Assumptions.* Cache stores post-RoPE keys; fused SDPA never materializes the `n x n` scores (eager attention would add `B H n^2 b`); the allocator rounds to blocks (so measured exceeds formula slightly).
- *Steps.*
  1. KV: each full-attention layer stores K and V of shape `(B, H_kv, n, d_h)`; 6 layers x 2 x 2 heads x 256 x 2 bytes = 12,288 B per token per sequence (a dense all-attention 24-layer stack of the same shape would store 49,152 B).
  2. State: the matrix state per value head is `d_k x d_v`; per layer `H_v d_k d_v = 262,144` elements at 4 bytes = 1,048,576 B; conv state `C (k_conv - 1) = 6144 x 3` elements at 2 bytes = 36,864 B; 18 layers: 19,537,920 B = 18.63 MiB per sequence.
  3. Crossover: hybrid total `12,288 n + 19,537,920` equals the dense-equivalent `49,152 n` at `n* = sigma / (2 H_kv d_h b) = 1,085,440 / 2,048 = 530` tokens, where `sigma` is the state bytes per linear layer (exercise D1).
  4. Peak at long prefill: the logits for all positions, `n V b = 4096 x 248,320 x 2 B = 2.03 GB`, exceed the cache; `logits_to_keep=1` (what `generate` does at prefill) removes it.
- *Check cell.* Predicted KV and state bytes equal the live cache's `nbytes` exactly (assert, by contexts {128, 1024, 4096}); predicted `Act + logits` vs `max_memory_allocated` is reported as a ratio, not asserted.

#### From-scratch implementation

Components (about 110 lines in all): `safetensors_header` (25 lines), `config_params` (35), `config_memory` (35), cache walker `cache_nbytes(cache)` (10, local to the notebook).

Numerical tests:

| Test | Reference | Tolerance |
|---|---|---|
| `config_params(qwen3_5_2b_cfg, True, True)` | Hub `safetensors.total` 2,274,069,824 | exact |
| `config_params(qwen3_0.6b_cfg)` | 596,049,920 (Base file total) | exact |
| `sum(config_params(cfg, by_group=True).values())` for Qwen3-0.6B and for Qwen3.5-2B (with `include_vision=True, include_mtp=True`) | `config_params(cfg, ...)` with `by_group=False` | exact (integers) |
| `config_memory(...)['kv_cache']`, `['recurrent_state']` at n in {128, 1024, 4096}, B=1 | `cache_nbytes` of the live Transformers cache after one prefill | exact; if the live state dtype is not FP32, the state term uses the observed dtype and the notebook reports the discrepancy (see risks) |
| `safetensors_header` vs `safe_open(...).get_slice` shapes after download | same names and shapes | exact |
| MoE config | raises `NotImplementedError` | n/a |

#### In production

- `AutoModelForCausalLM.from_pretrained`, `huggingface_hub` (`snapshot_download`, `HfApi`, `get_safetensors_metadata` as the library counterpart of the from-scratch header reader), `generate` with `GenerationConfig`, `tok.apply_chat_template`. The from-scratch header reader and `config_memory` are the test oracles for "is the model I am about to load the model I think it is, and will it fit": the check runs before a `from_pretrained` of a multi-gigabyte repo. Mention `transformers serve` and the hub's `hf` CLI only as pointers.
- Versions: record `transformers.__version__`, `huggingface_hub.__version__`, `torch.__version__`; state that Qwen3.5 modeling code is in `transformers` (file `models/qwen3_5/modeling_qwen3_5.py`) and that outputs depend on it as well as on the weights.

#### Headline experiment

- **Hypothesis.** Memory is predictable from the configuration: measured cache bytes of Qwen3.5-2B equal the derived `12,288 n + 19.54 MB` (exactly, if the state dtype is FP32), and decode speed follows the bandwidth ceiling up to a constant overhead factor that is independent of context length while the cache is small relative to the weights.
- **Setup.** `Qwen/Qwen3.5-2B` at its pinned SHA, BF16, one Ada GPU, batch 1, contexts `n` in {256, 1024, 4096, 16384} built from a fixed text (the first `n` tokens of a committed passage, repeated if needed; no sampling), `max_new_tokens=64` greedy decode for the speed series. `d2l.Benchmark(warmup=3, repeats=10)` for speed. Counterfactual series computed, not measured: the same shape with all 24 layers attention.
- **Procedure.** For each `n`: prefill, record `cache_nbytes` and `memory_allocated` growth; time 64 decode steps; compute the formula values and the ceiling.
- **Metrics.** Cache MiB vs `n`; decode tokens/s vs `n`; ratio measured/ceiling.
- **Plot (`fig_lm_hf_memory`).** Two panels. Left: x = context (log2, 256 to 262,144, measured up to 16,384 and formula beyond, dashed), y = cache MiB (log), series: measured hybrid (points), formula hybrid (line), formula dense counterfactual (line); a vertical marker at `n* = 530`. Right: x = context, y = decode tokens/s; series: measured median with interquartile band, bandwidth ceiling (line).
- **Expected outcome.** Cache points on the formula line (exact for the cache walker; the allocator series sits a little above it from block rounding). Decode below the ceiling at every context; the gap's size is "to be measured" and the notebook states in a sentence what part of it the Python launch overhead explains by counting kernel launches with the profiler (`torch.profiler`, per the source list) in one decode step.
- **Noise check.** Timing: 10 repeats, median and interquartile range; claims are made only where bands do not overlap. Memory is deterministic and needs no seeds. The five-prompt thinking comparison in item 5 is explicitly labeled anecdotal.
- **T1 reduction.** None needed beyond the contexts above (the 4B option is the only larger run: the notebook prints the derived total first and refuses if it exceeds the free memory (`torch.cuda.mem_get_info()`), which in a 15 GiB build slot at 16k context is itself the demonstration; on a whole 24 GB GPU the run goes ahead).
- **Artifacts committed.** `data/lm_revisions.json`; `lm-qwen35-2b-header.json` (names, shapes, dtypes of the 632 tensors, about 100 KB) so that the reconciliation runs offline; the measured table behind `fig_lm_hf_memory`.

#### Figures and tables

- `tab_lm_revisions`: pinned SHAs and licenses (above).
- `tab_lm_qwen35_config`: one row per field of Qwen3.5-2B (`text_config` unless noted): value; meaning; taught in.
  - `hidden_size 2048`; residual width; `sec_transformer-block`.
  - `num_hidden_layers 24`, `layer_types`, `full_attention_interval 4`; 18 linear plus 6 full; `sec_hybrids`, 14.6.
  - `num_attention_heads 8`, `num_key_value_heads 2`, `head_dim 256`; GQA 4:1; `sec_kv-cache`.
  - `attn_output_gate true`; query projection also emits an elementwise output gate; 14.4.
  - `rope_parameters {rope_theta 1e7, partial_rotary_factor 0.25, mrope_section [11,11,10], mrope_interleaved true}`; rotary on 64 of 256 dims; multimodal sections unused for text; `sec_positional-information`, 14.4.
  - `linear_num_key_heads 16`, `linear_num_value_heads 16`, `linear_key_head_dim 128`, `linear_value_head_dim 128`, `linear_conv_kernel_dim 4`, `mamba_ssm_dtype float32`; Gated DeltaNet geometry and state precision; `sec_deltanet`.
  - `intermediate_size 6144`, `hidden_act silu`; SwiGLU; `sec_transformer-block`.
  - `mtp_num_hidden_layers 1`, `mtp_use_dedicated_embeddings false`; multi-token-prediction head, in the file, not loaded by `Qwen3_5ForCausalLM`; 12.3.
  - `vocab_size 248320`, `tie_word_embeddings true`; `sec_text-sequence`; tied head.
  - `max_position_embeddings 262144`; native context; 4.6.
  - `rms_norm_eps 1e-6`, `eos_token_id 248044`, `attention_bias false`; details that 1.2 must match for a port.
  - top level `vision_config` (depth 24, width 1024, patch 16), `image_token_id`, `video_token_id`; the multimodal tower (later part).
- `tab_lm_hf_sampling_map`: HF argument vs `d2l.sample_next` (above).
- `tab_lm_base_vs_post`: five prompts, base continuation vs post-trained reply, qualitative.
- `fig_lm_hf_memory`: specified above.
- `tab_lm_param_buckets`: the five buckets of D1.1a with counts.

#### Pitfalls and misconceptions

- "A 2B model has 2B parameters": the label is the language model; vision and MTP add 392M here.
- Tied weights appearing twice in a file (Qwen3-0.6B post-trained) inflate tensor-element counts; `model.parameters()` deduplicates, `state_dict()` does not.
- Passing `revision` to the model but not the tokenizer; floating `main`; forgetting the library version.
- `generation_config.json` silently sets sampling; Qwen3.5-2B has none, so defaults are greedy unless the caller sets them: both directions bite.
- `device_map='auto'` offloading to CPU silently when memory is short.
- Reading thinking behavior off five prompts.
- `torch_dtype` vs `dtype`; BF16 on hardware without BF16 support falls back slowly.
- Assuming the MTP weights are used by `generate`: they are not (12.3 uses them).

#### Summary (draft bullets)

- A Hub model is a pinned git commit plus library code; reproducibility requires the revision of the weights, the tokenizer and chat template, and the version of `transformers`.
- `config.json` plus the safetensors header determine the parameter count exactly; for Qwen3.5-2B the 2.27B file is a 1.88B language model, a 0.33B vision tower and a 0.06B MTP head.
- `generate`'s arguments are the dials of `sec_decoding`, with default values hidden in `generation_config.json`.
- A hybrid cache is a per-token KV term over the attention layers plus a constant recurrent state; for Qwen3.5-2B 12 KiB per token plus 18.6 MiB, smaller than an all-attention stack beyond 530 tokens.
- Decode speed is set by bytes moved per token; eager Transformers sits below that ceiling.

#### Exercises

1. **(check) Where do the 2.27B parameters go?** Name the three buckets and say which two a text-only `AutoModelForCausalLM` load does not instantiate. *Sketch:* language model 1.88B instantiated; vision tower 0.33B and MTP 0.06B are in the file, not in `Qwen3_5ForCausalLM`; `model.num_parameters()` therefore returns 1,881,825,088 (to be confirmed by the load report).
2. **(check) Reproducibility of a chat reply.** List every artifact whose revision must be fixed for the same greedy reply tomorrow, and one that is not in the repo. *Sketch:* weights, tokenizer files, `chat_template.jinja`, `generation_config.json`, all in one commit; not in the repo: `transformers` modeling code, kernels (`fla`), `torch` version, and the hardware's kernel choices.
3. **(derive) The crossover.** For a hybrid with `L_l` linear layers of state `sigma` bytes each and `L_f` attention layers, show that its KV-plus-state cache is below that of an all-attention stack with `L_f + L_l` layers if and only if `n > sigma / (2 H_kv d_h b)`, and evaluate for Qwen3.5-2B. *Sketch:* `2 L_f H_kv d_h b n + L_l sigma < 2 (L_f + L_l) H_kv d_h b n` gives `L_l sigma < 2 L_l H_kv d_h b n`; with `sigma = 1,085,440` and `2 H_kv d_h b = 2,048` the answer is 530.
4. **(derive) Memory-bound batch.** At decode, weights are read once per step for the whole batch while each sequence adds its own cache read. Derive the batch size `B*(n)` at which cache traffic equals weight traffic for Qwen3.5-2B and for its dense counterfactual, and interpret. *Sketch:* weight traffic `W_r` (language model bytes), cache traffic per sequence `12,288 n + 2 x 19,537,920` (state is read and written); `B* = W_r / (12,288 n + 39,075,840)`; the dense counterfactual replaces the denominator by `49,152 n`; at large `n` the hybrid supports about 4x larger batches before cache traffic dominates.
5. **(measure) Qwen3.5-4B at 32k.** Before running, predict with `d2l.config_memory` the total for batch 4 at 32,768 tokens in BF16 and whether it fits a 15 GiB build slot and a 24 GB card; then try it. Report prediction, measurement and the largest term you did not model.
6. **(measure) Thinking loops.** Run 50 prompts (author supplies a fixed list in `data/`) with thinking on at `max_new_tokens=1024` under the card's recommended settings and under greedy; report the fraction that never close `</think>` with a bootstrap interval, and explain which setting you would ship.
7. **(open) A model passport.** Write a tool that takes a Hub id and prints, from `config.json` and the safetensors header only, parameters by bucket, weight/KV/state memory for a chosen workload, license and gating, and a list of config fields it does not understand. Run it on five recent models of different families and report what breaks.

#### Further reading

- Papers: [Transformers: State-of-the-Art Natural Language Processing](https://arxiv.org/abs/1910.03771) (the library's design); [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (Table 1 for the dense architectures, the base for 1.2).
- Documentation (v5): [Transformers docs](https://huggingface.co/docs/transformers/index); [quick tour](https://huggingface.co/docs/transformers/quicktour); [generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies); [model cards](https://huggingface.co/docs/hub/model-cards); [safetensors](https://huggingface.co/docs/safetensors/index); [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). New: [Qwen3.5 model doc](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5) (classes `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`); [Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download) (`revision`, `snapshot_download`).
- Course: [HF LLM Course, chapter 2](https://huggingface.co/learn/llm-course/chapter2/1).
- Model cards: [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B); [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B); [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).
- Code: [`modeling_qwen3_5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py): read `Qwen3_5GatedDeltaNet` (fused projections, convolution, torch fallback vs fast kernels) and `Qwen3_5Attention`; the file is the oracle for every row of `tab_lm_qwen35_config`.

#### Risks and open decisions

- **Transformers version drift.** Qwen3.5 modeling code is new; class names and cache classes may change between 5.x releases. Fallback: pin the exact version in the notebook environment and run the cache-walker test in CI; the cache-walker cell is written against `cache.get_recurrent_state(layer_idx, ...)` and `get_conv_state(...)` as in the source read for this blueprint, with a generic fallback that sums all tensors found by `named_buffers`-style traversal.
- **State dtype in the live cache.** The config says `mamba_ssm_dtype: float32`; whether Transformers allocates the recurrent state in FP32 or in the model dtype must be read off the cache. The formula takes `state_bytes` as a parameter and the notebook reports the observed value; the claimed numbers 19,537,920 B and `n* = 530` hold for FP32 state plus BF16 conv state.
- **Kernels.** `fla` and `causal-conv1d` may not install on the build box; the PyTorch fallback is slower, especially at long prefill. Fallback: report the path in use and restrict the long-context speed series to the contexts that finish within budget.
- **Load report.** Whether Transformers prints the dropped `mtp.*` and `model.visual.*` keys at load, and the exact class returned by `AutoModelForCausalLM`, are read from the source and not run in this blueprint; the notebook prints both and the narrative text must follow what it prints.
- **Thinking mode at 2B.** Looping may swamp the thinking comparison. Fallback: use Qwen3.5-4B for item 5 or cap at 512 tokens and report truncation as the result.

### 1.2 From `d2l.GPT` to a released checkpoint

- **Label:** `sec_lm_d2l_checkpoint` · **Tier:** T1 (reduced T0 config: `Qwen/Qwen3-0.6B` in FP32 on CPU with 3 prompts of 32 tokens; flag `DEVICE='cpu'`) · **Budget:** T1 about 10 min wall-clock; peak GPU memory about 5 GB (two 0.6B models in FP32: 2 x 2.4 GB plus activations); BF16 cells about 2.5 GB · **Codex units:** the opening walkthrough, Codex 2.1 (Reading a model checkpoint: configuration fields, state-dict keys, tied parameters, "replace one projection with an equivalent implementation and verify logits within a documented numerical tolerance")
- **Builds on (book):**
  - `sec_gpt`: `d2l.GPT` (subclassed by `Qwen3GPT`, not edited), its constructor flags, the GPT-2 loading cell (name mapping, `Conv1D` transpose, loss and greedy checks). This section is the second instance of that pattern and recaps it in one paragraph.
  - `sec_transformer-block`: `d2l.TransformerBlock`, `d2l.FeedForward` (SwiGLU; subclassed by `SwiGLU`), RMSNorm (`eq_rmsnorm`), the QK-norm subsection and its exercise (its `attn_factory` and `ffn_factory` hooks are how `Qwen3GPT` is assembled; this section is where QK-norm becomes a flag of `Qwen3Attention`).
  - `sec_kv-cache`: `d2l.GQAAttention` (subclassed by `Qwen3Attention`, not edited; `W_q`, `W_k`, `W_v`, `enable_gqa`, interleaved `_rope`), `forward_cached`, `generate_cached`, eq. `eq_kv-cache-bytes`. Not re-derived.
  - `sec_scaling-laws`: `tab_modern-recipe` Qwen3 row and the cell that expresses it as a constructor call; 1.2 shows what that row left out (head dimension, FFN width, theta, eps, layout).
  - `sec_positional-information`: RoPE. The rotation is reused unchanged; only the layout convention and the base differ.
  - `sec_decoding`: `d2l.sample_next`, and the loop `d2l.generate_ids(step_fn, ...)` (the book's `generate`, renamed in the book so that `d2l.generate` is free for 1.2's batched API; see Risks).
  - `sec_read_write`: safetensors loading.
- **Builds on (this part):** 1.1: `d2l.load_hf`, `d2l.HF_REVISIONS`, `d2l.safetensors_header`, `d2l.config_params` (reused for the shape-based parameter count), the Qwen3 configuration table row.
- **Exports (`#@save`):**
  - Three subclasses of book classes, defined in 1.2's notebook with `#@save`. The book's `GQAAttention`, `FeedForward`, `TransformerBlock` and `GPT` are not edited. The names are fixed because other chapters import them. Each subclass calls `nn.Module.__init__(self)` and builds its layers once, at the final width, instead of calling the parent constructor; the default-width layers of the parent are never allocated (a comment in the cell says why). About 80 lines in all.

    | Class | Signature and behavior | Why Qwen3 needs it |
    |---|---|---|
    | `d2l.Qwen3Attention(d2l.GQAAttention)` | `__init__(num_hiddens, num_heads, num_kv_heads, *, head_dim=None, qk_norm=False, rope_theta=10000.0, norm_eps=1e-6, bias=False, rope=True, causal=True)`. `head_dim=None` keeps `num_hiddens // num_heads`; otherwise `W_q: d -> H*head_dim`, `W_k, W_v: d -> H_kv*head_dim`, `W_o: H*head_dim -> d`. The four `nn.Linear` layers are built directly. `qk_norm=True` adds `q_norm`, `k_norm = nn.RMSNorm(head_dim, eps=norm_eps)`, applied per head after the reshape and before RoPE. A `_rope(x, offset=0)` override uses `rope_theta`. `forward(queries, keys, values, valid_lens=None)` is overridden so the QK-norm can be inserted (`valid_lens` must be `None`: causal self-attention, no padding masks, as before). `forward_step(X, cache)` is new: cached attention for new tokens with `H_kv` heads cached, `enable_gqa=True`, RoPE at the cache offset. Names: `W_q, W_k, W_v, W_o, q_norm, k_norm`; attributes `head_dim, num_heads, num_kv_heads, rope_theta`. | Qwen3-0.6B: `head_dim 128` with `hidden 1024`, 16 query heads (2048 query width, wider than the residual); QK-norm; `rope_theta 1e6`. The book's `GQAAttention` fixes the head width at `d/H`, has no norm, and its `_rope` has base 1e4 and no offset; the book's cached `forward_step` exists only for `GPT.CausalAttention` (fused `W_qkv`) |
    | `d2l.SwiGLU(d2l.FeedForward)` | `__init__(num_hiddens, hidden, bias=False)`: `nn.Module.__init__`, `self.act = 'swiglu'` (the inherited `forward` branches on it), then `W_g, W_1: d -> hidden` and `W_2: hidden -> d`. About 6 lines. The inherited `forward` is `W_2(silu(W_g x) * W_1 x)`; module names `W_g, W_1, W_2` are the book's. | Qwen3 `intermediate_size` is 3072 for `d = 1024` (3d), not `round(8d/3)` = 2731, which the book's `FeedForward` fixes |
    | `d2l.Qwen3GPT(d2l.GPT)` | `__init__(vocab_size, num_hiddens, num_heads, num_kv_heads, num_blks, *, head_dim=None, qk_norm=True, rope_theta=1e6, ffn_hidden=None, norm_eps=1e-6, tie_embeddings=True, max_len=40960, bias=False)`. It does not call `d2l.GPT.__init__`, which would first allocate fused-`W_qkv` blocks (gigabytes at 8B). It calls `nn.Module.__init__`, sets `pos='rope'` and `max_len`, builds `token_emb` and the blocks directly with `d2l.TransformerBlock(..., attn_factory=lambda: d2l.Qwen3Attention(...), ffn_factory=lambda: d2l.SwiGLU(num_hiddens, ffn_hidden or round(8 * num_hiddens / 3), bias))`, sets `eps = norm_eps` on each block's `norm1` and `norm2` and builds the final `norm = nn.RMSNorm(num_hiddens, eps=norm_eps)`. `tie_embeddings=False` adds `lm_head = nn.Linear(d, V, bias=False)`. Overrides: `forward` (head is `lm_head` if untied) and `forward_cached` (calls `blk.attention.forward_step`, uses `lm_head` if untied; `@torch.no_grad()` like the book's). Inherited unchanged from `d2l.GPT`: `generate` (sec_gpt; it calls only `self(X)`, `self.max_len`, `self.parameters()`). Defined on `Qwen3GPT`: `generate_cached`, the book's loop from `sec_kv-cache` in about 12 lines, because that cell has no `#@save` (see the risk). `isinstance(m, d2l.GPT)` is true. | untied Qwen3-8B; `eps 1e-6`; blocks assembled through the hooks of `sec_transformer-block` (`attn_factory`, `ffn_factory`) instead of the fused-QKV default |

    Construction is direct (not on the meta device) inside `Qwen3GPT`: no wasted allocation happens, but the layers are still randomly initialized, so `load_qwen3` constructs the model under `torch.device('meta')` and loads with `assign=True`; 8B is then built without allocating a single weight until the state dict is assigned. `Qwen3Attention` and `Qwen3GPT` read nothing from the parents' constructors except attribute names (`act`, `pos`, `max_len`, `blks`, `token_emb`, `norm`).

    Features the subclass inherits from `d2l.GPT` and the blocks, and uses as is: RMSNorm pre-norm blocks, the SwiGLU forward (`W_2(silu(W_g x) * W_1 x)`), the interleaved RoPE rotation formula, bias-free linears, the tied embedding via `F.linear(norm(H), token_emb.weight)` when `tie_embeddings=True`, and the sampling loops `generate` and `generate_cached`. Not supported and not added: sliding windows, YaRN, per-sequence attention masks (batched generation uses length buckets instead).
  - `d2l.qwen3_rope_permutation(head_dim) -> torch.LongTensor`: `idx` of length `head_dim` with `idx[2i] = i`, `idx[2i+1] = i + head_dim//2`; row `j` of the permuted matrix is row `idx[j]` of the original.
  - `d2l.qwen3_gpt_kwargs(cfg) -> dict`: maps a Qwen3 `config.json` (dict or `PretrainedConfig`) to the `d2l.Qwen3GPT` constructor arguments (`vocab_size, num_hiddens, num_heads, num_kv_heads, num_blks, head_dim, ffn_hidden, norm_eps, rope_theta, tie_embeddings, max_len`; `qk_norm=True`, `bias=False`); raises on `use_sliding_window`, non-null `rope_scaling` or `attention_bias`.
  - `d2l.map_qwen3_state_dict(hf_state, cfg, rope_layout='permute') -> dict[str, Tensor]`: the name and layout map of the table below; returns a state dict for `d2l.Qwen3GPT`; raises `KeyError` on any HF tensor not consumed and on any `Qwen3GPT` key not produced (strict accounting). `rope_layout='none'` skips the permutation (used for bug injection).
  - `d2l.load_qwen3(model_id='Qwen/Qwen3-0.6B', revision=None, *, dtype=torch.bfloat16, device=None) -> d2l.Qwen3GPT`: resolves the revision as 1.1 does, downloads the safetensors file(s) (index plus shards for 4B and 8B) with `huggingface_hub.hf_hub_download`, builds a `d2l.Qwen3GPT` from the config through `qwen3_gpt_kwargs` under `torch.device('meta')` (no default-width or random-initialized weights are ever allocated, which matters at 8B), loads with `load_state_dict(..., assign=True)` (`model_type` must be `qwen3`; `qwen3_moe` raises `NotImplementedError('14.2')`, `qwen3_5` raises `NotImplementedError('14.6')`), through `map_qwen3_state_dict`, then moves to `device` (default `d2l.try_gpu()`), sets `eval()`, and attaches `model.hf_config` and `model.revision`. The tokenizer is obtained separately with `AutoTokenizer.from_pretrained(model_id, revision=model.revision)` (no BOS is added by this tokenizer). Edge cases: the post-trained 0.6B and 1.7B files contain an explicit `lm_head.weight`; the mapper asserts it equals `model.embed_tokens.weight` when `tie_word_embeddings` is true and drops it; for 8B it is kept in `lm_head`. The result is a `d2l.GPT` (`isinstance` holds), so every consumer of R1 that accepts a `d2l.GPT` accepts it.
  - `d2l.sample_logits(logits, strategy='greedy', temperature=1.0, k=None, p=None, min_p=None, generator=None) -> LongTensor` and `d2l.truncated_probs(logits, temperature=1.0, k=None, p=None, min_p=None) -> Tensor`: batched torch versions of `d2l.sample_next` (same semantics: top-p keeps the smallest head with mass at least `p`; ties broken by token index), `logits` shaped `(B, V)`; `truncated_probs` returns the post-truncation renormalized distribution so that the semantics can be tested exactly.
  - Module names of the port (other chapters rely on them): `model.token_emb`; `model.blks[i].attention.{W_q, W_k, W_v, W_o, q_norm, k_norm}` (plus attributes `head_dim, num_heads, num_kv_heads, rope_theta`); `model.blks[i].ffn.{W_g, W_1, W_2}` with `W_2(silu(W_g x) * W_1 x)`; `model.norm`; `model.lm_head` only when untied. There are no `W_gate/W_up/W_down`.
  - `d2l.Generation` (a `typing.NamedTuple`): `text: str`, `token_ids: list[int]`, `logprobs: list[float] | None`, `finish_reason: str` (`'eos'`, `'length'`, `'stop'`).
  - `d2l.generate(model, tokenizer, prompts, max_new_tokens=None, **sampling) -> list[d2l.Generation]`. Signature in full: `generate(model, tokenizer, prompts, max_new_tokens=None, *, strategy='greedy', temperature=1.0, k=None, p=None, min_p=None, seed=0, eos_token_ids=None, stop=None, chat=False, enable_thinking=None, return_logprobs=False, batch_size=8)`.
    - `prompts`: `str` or `list[str]` (with `chat=True` each string is a single user message rendered by `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=enable_thinking)`; a list of message dicts is used as is). Returns one `Generation` per prompt, in order.
    - `model` may be a `d2l.GPT` (including `d2l.Qwen3GPT`; uses the cached path through `forward_cached`) or a Transformers `PreTrainedModel` (own loop with `past_key_values`, `use_cache=True`, left padding with `attention_mask`, `logits_to_keep=1` at prefill), or any `nn.Module` with the contract `model(input_ids) -> logits` (full recompute, no cache). Batching: `d2l.GPT` and generic modules batch only prompts of equal token length (the function buckets by length), because neither takes an attention mask; Transformers models batch with padding.
    - Randomness: a `torch.Generator` seeded by `seed` per call; results are reproducible for a fixed batch composition (batch-dependent kernels can change results across batch sizes, see 1.4).
    - Stopping: `eos_token_ids` defaults to the tokenizer's `eos_token_id` plus the model's `generation_config.eos_token_id` list if present (Qwen3 post-trained stops on `<|im_end|>` 151645 and `<|endoftext|>` 151643); `stop` is a list of strings checked on the decoded suffix.
    - Name: the `sec_decoding` loop `generate(step_fn, prefix, num_tokens, eos_id=None, **strategy)` is renamed `d2l.generate_ids` in the book (a book-side rename, see Risks), so `d2l.generate` is only the batched API of this section and has no dispatch and no legacy path. Calls such as `d2l.generate_ids(step_fn, prefix, 25, strategy='sample', ...)` in `chapter_recurrent-modern/mamba.md` use the new name and their outputs are unchanged (test).
  - `d2l.first_divergence(ref_hiddens, our_hiddens, rtol=...) -> tuple[int, str] | None` and `d2l.collect_hiddens(model, input_ids, *, kind)` (hook-based capture for `kind in {'hf', 'd2l'}` of, per layer, `attn_out`, `resid_mid`, `mlp_out`, `resid_post`).
- **Models and data:**

  | Model | Params | Revision (28 Oct 2026) | Notes |
  |---|---|---|---|
  | `Qwen/Qwen3-0.6B` | file 751,632,384 (with duplicate head), model 596,049,920 | `c1899de289a04d12100db370d81485cdf75e47ca` | R1; Apache-2.0; `generation_config`: `temperature 0.6, top_k 20, top_p 0.95` |
  | `Qwen/Qwen3-0.6B-Base` | 596,049,920 | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | no `lm_head` in file |
  | `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` | 1,720,574,976; 4,022,468,096; 8,190,735,360 | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; `1cfa9a7208912126459214e8b04321603b3df60c`; `b968826d9c46dd6066d109eabc6255188de91218` | the Qwen3 dense family loaded by the same function (8B untied) |

  Eight fixed prompts committed in the notebook (two English prose, two Python, one Chinese, one arithmetic, one multilingual snippet, one repeated-token string), each tokenized to exactly 64 tokens by truncation: they exercise the embedding rows of different scripts and the induction pattern.
- **Packages:** `torch >= 2.5` (`enable_gqa` in `scaled_dot_product_attention`), `transformers >= 5.2.0`, `safetensors`, `huggingface_hub`. The reference model runs in Transformers with `attn_implementation='eager'` for the layer-by-layer match (so that the reference does not use a different fused kernel from the port), and with the default for the timing comparison.

#### Narrative arc

1. **The question.** Opening cell: build `d2l.GPT` from the `tab_modern-recipe` Qwen3 row (the constructor call of `sec_scaling-laws`), load Qwen3-0.6B's embedding and norms into it by hand, run one prompt and print the logits' top-5 tokens next to Transformers': the top-5 lists disagree and the maximum absolute logit error is order-of-magnitude larger than any rounding explanation. Symptom printed: `max |logit diff| = ...` (to be measured; large). "The recipe row names the ingredients. It does not say what the weights need."
2. **What the recipe table leaves out: a port is a diff** (`subsec_lm_recipe_gaps`).
   - Teaching point stated first: porting a released model is computing a diff against a model you already have. Qwen3 is the book's `d2l.GPT` with a short list of specific changes: separate `W_q, W_k, W_v` and a GQA layout instead of the fused `W_qkv` of `d2l.GPT.CausalAttention`; a head dimension that is not `d/H`; QK-norm; RoPE base 1e6; FFN width `3d` instead of `round(8d/3)`; an explicit RMSNorm epsilon; and, for 8B, an untied head (the book's `GPT` always ties). `sec_transformer-block` already provides the seams (`attn_factory`, `ffn_factory`), so each change lands in one small subclass and the book's classes stay untouched: `Qwen3Attention`, `SwiGLU`, `Qwen3GPT`.
   - Table `tab_lm_qwen3_gaps` (below): constructor argument vs Qwen3-0.6B's `config.json` vs what the book's `d2l.GPT` does, one row per change in the diff. Then one cell per subclass, each ending with the shape check `Qwen3GPT(**d2l.qwen3_gpt_kwargs(cfg))` and `sum(p.numel())`. The cell for `Qwen3GPT` also builds the 8B configuration on the meta device and prints its parameter count without allocating weights.
   - Cell: the strict-generalization check (test table): with the Qwen3 features switched off (`num_kv_heads=num_heads, head_dim=None, qk_norm=False, rope_theta=1e4, norm_eps=None`) a `Qwen3GPT` initialized from a small `d2l.GPT`'s weights reproduces that model's logits. The diff is empty when the changes are off, so the subclass generalizes the book's model rather than replacing it.
   - Features the subclass inherits unchanged from `d2l.GPT`: pre-norm RMSNorm blocks (`d2l.TransformerBlock`), the SwiGLU forward, the interleaved RoPE rotation, bias-free linears, the token-embedding tie `F.linear(norm(H), token_emb.weight)`, and `generate` (`generate_cached` is re-stated on the subclass, see the risk).
   - A one-paragraph recap of `sec_gpt`'s GPT-2 loading. Then a one-line contrast: GPT-2 needed a transpose; Qwen3 needs a name map, a layout permutation for four tensors, a head dimension and eps.
   - Cell: the parameter count reconstructed from shapes (derivation D1.2a) before any weight is loaded: 596,049,920.
3. **Name and layout map** (`subsec_lm_name_map`).
   - Table `tab_lm_qwen3_map`. With `i` the layer index (`P` = RoPE row permutation of D1.2b; `-` = no change):

     | Transformers tensor (shape) | `d2l.Qwen3GPT` tensor | Transform |
     |---|---|---|
     | `model.embed_tokens.weight` (151936, 1024) | `token_emb.weight` | - |
     | `lm_head.weight` (post-trained files only) | tied; asserted equal, dropped | - |
     | `model.layers.i.input_layernorm.weight` (1024) | `blks[i].norm1.weight` | - |
     | `model.layers.i.self_attn.q_proj.weight` (2048, 1024) | `blks[i].attention.W_q.weight` | rows of each 128-row head block permuted by `P` |
     | `...k_proj.weight` (1024, 1024) | `...W_k.weight` | same, per KV head |
     | `...v_proj.weight` (1024, 1024) | `...W_v.weight` | - |
     | `...o_proj.weight` (1024, 2048) | `...W_o.weight` | - |
     | `...q_norm.weight`, `...k_norm.weight` (128) | `...q_norm.weight`, `...k_norm.weight` | `P` applied to the vector |
     | `model.layers.i.post_attention_layernorm.weight` | `blks[i].norm2.weight` | - |
     | `...mlp.gate_proj.weight` (3072, 1024) | `blks[i].ffn.W_g.weight` | - |
     | `...mlp.up_proj.weight` | `blks[i].ffn.W_1.weight` | - |
     | `...mlp.down_proj.weight` (1024, 3072) | `blks[i].ffn.W_2.weight` | - |
     | `model.norm.weight` | `norm.weight` | - |

     311 tensors in the Qwen3-0.6B file: 28 x 11 per-layer tensors plus embedding, head and final norm.
   - Query-to-KV head grouping: Transformers repeats each KV head `H / H_kv` times consecutively; SDPA's `enable_gqa` does the same, so query head `h` reads KV head `h // (H / H_kv)`: no reordering is needed (assert by a one-layer test: one-hot `k_proj` rows).
   - Why `v_proj`, `o_proj` and the MLP need no permutation: derivation D1.2b.
   - Cell: the strict-accounting error. Remove one HF tensor from the dict and show the `KeyError` message that names it.
4. **Matching, layer by layer** (`subsec_lm_layerwise`).
   - Tolerance protocol (derivation D1.2c), then cells in order: FP32 logits; per-layer relative error of `resid_post` (hooks, not `output_hidden_states`, because the last entry of that tuple is after the final norm in some versions: the cell asserts `norm(h_L) @ E.T == logits`); BF16 logits vs the noise floor.
   - Bug injection: five seeded bugs, each a one-line change, each run through `first_divergence`: (B1) `rope_layout='none'`, (B2) `rope_theta=10000`, (B3) swap `W_g` and `W_1`, (B4) no QK-norm, (B5) `norm_eps=None` (PyTorch default `finfo(dtype).eps`: 1.19e-7 in FP32 but 7.8e-3 in BF16, because `torch.finfo(torch.bfloat16).eps` is 2^-7). Table `tab_lm_port_bugs`: bug, first diverging (layer, sublayer), relative error there, whether the final top-1 token changes on the 8 prompts. The table is to be measured; directions: B1, B2, B4 diverge in layer 0 attention output, B3 in layer 0 MLP output; B5 is invisible in FP32 at `rtol 1e-3`... and visible in BF16 only through the floor test: the point of the row is that a tolerance too loose hides a real bug.
   - Cell: parameter-count check, `sum(p.numel())` of the loaded `d2l.Qwen3GPT` equals 596,049,920.
5. **Generate with the book's sampler** (`subsec_lm_generate_d2l`).
   - `d2l.generate(model, tok, prompts, 64, chat=True, enable_thinking=False)` on the port (a `d2l.Qwen3GPT`, so the cached path through its `forward_cached`); the same call on the Transformers model; greedy outputs compared token by token (FP32: identical on all 8 prompts expected; BF16: report the first divergence index per prompt: divergences happen at near-ties of the top-2 logits, and the cell prints the logit gap at the divergence step, to be measured).
   - Sampled outputs: `strategy='sample', temperature=0.6, k=20, p=0.95` (the model's own `generation_config`) with seeds 0–2; the sampler is the book's (test of `truncated_probs` vs `sample_next` in the from-scratch section).
   - Throughput table `tab_lm_generate_paths`: naive recompute (`generate` of `sec_gpt`, inherited by `Qwen3GPT`), cached (`generate_cached`, re-stated in 1.2), `d2l.generate` on the port, `model.generate` of Transformers; tokens per second at batch 1 and 8, 128-token prompts, 64 new tokens (to be measured; direction: cached beats naive by a growing factor, Transformers is the same order as the cached port).
6. **What the port is for** (`subsec_lm_port_use`): R1 is now an object whose every tensor is visible. Two examples run in the notebook: (a) print the Frobenius norms of `W_q` and `q_norm.weight` per layer (a quantity inaccessible through `generate`); (b) a hook that zeros one attention head's output, with the logit change on a prompt: the first intervention, previewing 1.6.

#### Derivations

**D1.2a: Parameter count from shapes** (`eq_lm_qwen3_params`).
- *Statement.* For Qwen3 dense with width `d`, `H` query heads, `H_kv` KV heads, head dimension `d_h`, FFN width `f`, `L` layers, vocabulary `V`, tied head: `P = V d + L [ d H d_h + 2 d H_kv d_h + H d_h d + 2 d_h + 3 d f + 2 d ] + d`; untied adds `V d`.
- *Steps.* (1) Embedding `V d`. (2) Per block: `q_proj` `d x H d_h`, `k_proj` and `v_proj` `d x H_kv d_h` each, `o_proj` `H d_h x d`, `q_norm`+`k_norm` `2 d_h`, three MLP matrices `3 d f`, two block norms `2d`. (3) Final norm `d`. (4) Substitute 0.6B (`d 1024, H 16, H_kv 8, d_h 128, f 3072, L 28, V 151936`): block `15,730,944`, total `596,049,920`. Same formula reproduces the Hub totals 1,720,574,976 (1.7B), 4,022,468,096 (4B), and 8,190,735,360 (8B, untied).
- *Check cell.* Formula vs `safetensors_header`-derived total vs `sum(p.numel())` of `d2l.Qwen3GPT`, for 0.6B, 1.7B-Base, 4B, 8B (header-only for the last three: no weights downloaded).

**D1.2b: RoPE layout equivalence** (`eq_lm_rope_layout`).
- *Statement.* Let `R_pos` rotate coordinate pairs `(2i, 2i+1)` by angle `pos * w_i` (the book's interleaved layout) and `R'_pos` rotate pairs `(i, i + d_h/2)` by the same angles (Transformers' split-half layout, "rotate_half"). With the permutation matrix `P` defined by `(P x)_{2i} = x_i`, `(P x)_{2i+1} = x_{i + d_h/2}`, `R_pos P = P R'_pos`. Hence if `W_q^{ours} = P W_q^{HF}` (rows permuted) the query vectors satisfy `q^{ours} = P q^{HF}`, and the same for keys, so `<R_a q^{ours}, R_b k^{ours}> = <R'_a q^{HF}, R'_b k^{HF}>`.
- *Assumptions.* The same angles `w_i = theta^(-2i/d_h)` for index `i` in both layouts (true for Qwen3); `P` orthogonal.
- *Steps.* (1) Write `R'_pos` in coordinates `(x_i, x_{i+d_h/2})` as a 2x2 rotation by `pos * w_i`. (2) `P` maps that coordinate pair to positions `(2i, 2i+1)`, so `P R'_pos P^T` is block-diagonal with the same 2x2 rotations, which is `R_pos`. (3) The inner product is preserved by orthogonal `P`: `<P u, P v> = <u, v>`. (4) QK-norm: RMSNorm is `x / rms(x) * gamma` with `rms` a symmetric function, so `P(RMSNorm_gamma(x)) = RMSNorm_{P gamma}(P x)`: the norm weight is permuted with the same `P`. (5) `v_proj`, `o_proj` and the MLP are untouched because the value path has no RoPE.
- *Check cell (CPU, no weights).* Random `q, k` of dimension 128, `theta 1e6`, positions 0–63: scores from the HF-layout rotation of `(q, k)` equal scores from the interleaved rotation of `(Pq, Pk)` to `1e-6`; a per-head RMSNorm with random `gamma` commutes with `P` as stated; omitting the `gamma` permutation breaks the equality by an amount the cell prints.

**D1.2c: A tolerance for BF16** (`eq_lm_bf16_floor`).
- *Statement.* Two correct implementations in BF16 differ because rounding order differs, so the right yardstick is the distance between two independent correct BF16 evaluations, not a number chosen in advance.
- *Assumptions.* The FP32 Transformers run is the reference; the Transformers BF16 run is an independent correct evaluation.
- *Steps.* (1) Let `e_hf = mean |Lhf16 - Lhf32|` over positions and vocabulary (rounding error of one correct BF16 stack). (2) Let `e_our = mean |Lour16 - Lhf32|`. (3) Accept the port if `e_our <= 1.5 e_hf`, and if the mean KL(softmax(`Lhf32`) || softmax(`Lour16`)) is at most 1.5 times the same for HF BF16. The constant 1.5 is a convention, stated as such. (4) In FP32, with TF32 disabled, accept `max |Lour32 - Lhf32| <= 1e-3` (provisional; the author measures the CPU-vs-GPU FP32 difference of Transformers itself on the same prompts and uses 3 times that if larger). (5) A relative effect of a wrong epsilon: `1/sqrt(m + e)` vs `1/sqrt(m + e')` for mean square `m`: relative change about `(e - e') / (2m)`, which for `e = 7.8e-3`, `e' = 1e-6`, `m = 1` is 0.39%, comparable to BF16's own `2^-8 = 0.39%` relative rounding: the bug sits at the noise floor in a single layer and is visible only because it accumulates over 57 norms.
- *Check cell.* Compute `e_hf`, `e_our`, KL ratios on the 8 prompts; assert the acceptance rule; print the B5 bug's `e_our / e_hf`.

#### From-scratch implementation

Components (about 270 lines of notebook code, all in 1.2's notebook; no book-side edits to the book's classes; the only book-side change is the `generate_ids` rename):
- `Qwen3Attention` (about 45 lines: constructor 14, `_rope` with theta and offset 8, `_qkv` helper 8, `forward` 6, `forward_step` 14), `SwiGLU` (6), `Qwen3GPT` (about 30: constructor 16, `forward` 4, `forward_cached` 10): about 80 lines in all.
- Equivalence-test helper `copy_gpt_into_qwen3(gpt, qwen3)` (splits `W_qkv`; 15 lines).
- `qwen3_gpt_kwargs(cfg)`: 10 lines; maps `hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, num_hidden_layers, rms_norm_eps, rope_theta, tie_word_embeddings, max_position_embeddings, vocab_size` to the `Qwen3GPT` constructor of Exports.
- `qwen3_rope_permutation`, `map_qwen3_state_dict`: 45 lines.
- `load_qwen3`: 25 lines (sharded index handling 8 lines).
- `collect_hiddens`, `first_divergence`: 35 lines.
- `truncated_probs`, `sample_logits`, `generate` (cached path and equal-length bucketing): 60 lines (the notebook shows the sampler and the loop; the Transformers path is 15 of those lines).
Numerical tests:

| Test | Reference | Tolerance | Seed |
|---|---|---|---|
| FP32 logits of the port, 8 prompts x 64 tokens | Transformers FP32 (eager attention) | `max abs <= 1e-3` (provisional, D1.2c) | none (deterministic; TF32 off) |
| BF16 logits | Transformers BF16 and FP32 | D1.2c rule with factor 1.5 | none |
| `truncated_probs` vs `sample_next`-implied distribution (5-token toy and 20 random 1000-token logit vectors; `k`, `p`, `min_p`, temperature grid) | NumPy reimplementation of `d2l.sample_next`'s truncation | `atol 1e-6` | 0 |
| `sample_logits` empirical frequencies, 200,000 draws from the 5-token toy | `truncated_probs` | total variation at most `3 / sqrt(200000)` ~ 0.0067 (a 3-sigma-style bound for 5 outcomes; the notebook prints the observed value) | 0 |
| Greedy `d2l.generate` vs `model.generate(do_sample=False)` (Transformers), 64 tokens, 8 prompts, FP32 | token ids | identical, with at most one tie-excused prompt | none |
| Book-side rename `d2l.generate` to `d2l.generate_ids`: `d2l.generate_ids(step_fn, prefix, 25, strategy='sample', ...)` in the Mamba cells of `chapter_recurrent-modern/mamba.md` | the Mamba cell outputs with seed fixed, before the rename | identical | existing |
| Strict generalization: `Qwen3GPT(num_kv_heads=num_heads, head_dim=None, qk_norm=False, rope_theta=1e4, norm_eps=None, tie_embeddings=True)` (the `norm_eps=None` keeps PyTorch's dtype-dependent default, as the book's norms do), with weights copied from a small default `d2l.GPT` (2 blocks, `d` 64, 4 heads, vocabulary 100; `W_qkv` split into `W_q, W_k, W_v` by `chunk(3, 0)`; all else by name), on 4 random sequences of 32 tokens | logits of that `d2l.GPT` | `max abs <= 1e-5` in FP32 (same arithmetic; the only difference is three small matmuls instead of one); and `forward_cached` agrees with `forward` of the same model to `1e-5` | torch seed 0 |

#### In production

- Transformers' `Qwen3ForCausalLM` (`modeling_qwen3.py`) is the reference and the thing the port is tested against; the port is the oracle for the other direction (kernel authors, 12.1, 14.x variants). `from_pretrained(..., attn_implementation='sdpa')` and `'eager'` are both run once to show the reference itself has a noise floor. Mention that vLLM and SGLang have their own Qwen3 implementations tested the same way (pointer to 13.1). The notebook ends with a short list of what the port deliberately does not do (sliding-window and YaRN configurations, batched left padding, FlashAttention-3), each tied to the section that handles it.

#### Headline experiment

- **Hypothesis.** A correct port is indistinguishable from the reference up to the reference's own rounding noise at every layer, and each of five common porting bugs is detectable by the layer-wise curve at the layer and sublayer where it acts, including one (wrong epsilon) that a final-logit check alone would not flag.
- **Setup.** Qwen3-0.6B (pinned), 8 prompts x 64 tokens, FP32 (TF32 off) and BF16, one Ada GPU, eager attention in the reference. No training, no sampling.
- **Procedure.** For the correct port and each bug B1–B5: capture `attn_out`, `resid_mid`, `mlp_out`, `resid_post` per layer from both models; compute relative Frobenius error per layer and sublayer; run `first_divergence` with `rtol` set to 10 times the correct port's maximum FP32 per-layer error (the rule: a divergence is a layer whose error exceeds ten times the clean-port error at that layer); record final-logit metrics of D1.2c.
- **Metrics.** Per-layer relative error; first diverging (layer, sublayer); `e_our / e_hf`; top-1 agreement over the 8 x 64 positions.
- **Plot (`fig_lm_port_layer_error`).** x = layer 0–27; y = relative error of `resid_post` (log scale); series: clean port FP32, clean port BF16, HF BF16 vs HF FP32 (the floor, dashed), B1, B2, B3, B4, B5 (all FP32 to localize; B5 also in BF16). Median over prompts as the line, min-max over the 8 prompts as the band.
- **Expected outcome (directions).** Clean FP32 curve is near FP32 rounding and nearly flat in depth; clean BF16 sits at the HF-BF16 floor and rises slowly with depth; B1–B4 are orders of magnitude above the floor from layer 0 (B1, B2, B4 at attention, B3 at the MLP); B5 is within a small factor of the floor in each layer and separates only through the accumulated curve or the KL ratio. All magnitudes are to be measured.
- **Noise check.** The computation is deterministic with TF32 off; run twice and assert bitwise-identical captures. The 8 prompts supply the spread shown in the bands; the claim "B5 is distinguishable" is made only if its KL ratio exceeds the maximum ratio among the three clean-port reruns with different batch compositions.
- **T1 reduction.** The experiment is already T1 (0.6B, 8 prompts); the T0 fallback uses 3 prompts of 32 tokens.
- **Artifacts committed.** `qwen3_0.6b_logit_reference.pt` (FP32 reference logits for the 8 prompts: 8 x 64 x 151,936 x 4 B = 311 MB; stored as top-100 per position plus full logits for the last position only, about 2 MB, with `torch.save`; the autograder compares on those entries), the layer-error table behind `fig_lm_port_layer_error` as CSV.

#### Figures and tables

- `tab_lm_qwen3_gaps` (made by the shape-check cell): rows: head dimension (config 128; book `d2l.GPT`/`GQAAttention`: `d / H` = 64), query width (2048 vs 1024), FFN width (3072 vs `round(8d/3)` = 2731), RoPE base (1e6 vs 1e4), RoPE layout (split-half vs interleaved), QK-norm (yes vs none), norm epsilon (1e-6 vs `finfo` default), tied head (yes for 0.6B/1.7B/4B, no for 8B vs always tied), attention layout (separate `W_q, W_k, W_v` vs the fused `W_qkv` of `d2l.GPT.CausalAttention`).
- `tab_lm_qwen3_map`: above.
- `tab_lm_port_bugs`: five bugs, measured table, made by the bug-injection cell.
- `fig_lm_port_layer_error`: above.
- `tab_lm_generate_paths`: throughput of the four generation paths, made by `d2l.Benchmark`.

#### Pitfalls and misconceptions

- A shape check does not validate a port (`sec_gpt` made the same point): a transposed or mis-permuted square matrix still multiplies; only logits and layer-wise errors validate.
- `output_hidden_states` conventions differ across versions (the last entry may be post-norm): use hooks.
- Comparing FP32 port to BF16 reference and blaming the port.
- Eager vs SDPA attention in the reference changes rounding; fix and state it.
- The RoPE permutation applies to `q_norm` and `k_norm` weights as well as to the projections.
- PyTorch's `nn.RMSNorm` epsilon default depends on dtype.
- Counting parameters from the file: the post-trained 0.6B and 1.7B files store a duplicate head.
- Hooking after the final norm.
- Qwen3's tokenizer adds no BOS; adding one shifts every logit.

#### Summary (draft bullets)

- A recipe row names the ingredients; a released checkpoint additionally fixes head dimension, FFN width, RoPE base and layout, norm epsilon and tying.
- A port is a diff: Qwen3 is the book's `d2l.GPT` with specific changes (separate `W_q, W_k, W_v` instead of fused `W_qkv`, GQA, head dimension, QK-norm, RoPE base, FFN width, epsilon, optional untied head), made through the `attn_factory` and `ffn_factory` hooks of `sec_transformer-block`, in three subclasses (`Qwen3Attention`, `SwiGLU`, `Qwen3GPT`) with the book's classes untouched. With the changes switched off, the subclass reproduces `d2l.GPT` exactly.
- Name maps are tested by strict accounting, and layouts by an algebraic identity (RoPE permutation) plus a numerical check.
- Tolerances should be set relative to the reference's own rounding noise: the distance between BF16 and FP32 evaluations of the reference.
- Layer-wise hooks localize a bug to a layer and sublayer; a final-logit check alone can miss a small one.

#### Exercises

1. **(check) Why only four tensors are permuted.** Which tensors of a Qwen3 layer carry the RoPE layout and why not `v_proj`? *Sketch:* `q_proj`, `k_proj`, `q_norm`, `k_norm`, because RoPE acts on queries and keys, and QK-norm weights sit on the rotated coordinates; values are not rotated.
2. **(check) Tied head.** The post-trained Qwen3-0.6B file has 751,632,384 tensor elements and the model 596,049,920 parameters. Give the difference and say what the loader must assert before dropping `lm_head.weight`. *Sketch:* 155,582,464 = 151,936 x 1,024; assert `torch.equal(lm_head, embed_tokens)`.
3. **(derive) Both layouts agree.** Prove `R_pos P = P R'_pos` for `d_h = 4` by writing both rotations as 4x4 matrices, and show what goes wrong if only `q_proj` rows are permuted. *Sketch:* the cross terms in `<R_a q, R_b k>` pair coordinate `i` of one vector with coordinate `i + d_h/2` of the other and produce a score that depends on `a - b` incorrectly.
4. **(derive) Wrong epsilon in BF16.** For RMSNorm inputs with mean square `m`, show the relative output error from using `eps = 2^-7` instead of `1e-6` is about `eps / (2m)`, and estimate how it compounds over `2L + 1` norms if the per-norm errors are independent and have random sign vs same sign. *Sketch:* first-order expansion of `(m + eps)^(-1/2)`; independent: `sqrt(2L+1)` growth, same sign: linear in `2L+1`.
5. **(measure) B1 and the head dimension.** Predict the layer-0 relative error of B1 (no permutation) before running; then run it with the head dimension 128 and with a toy `d_h = 8` and explain why the error scales as it does.
6. **(measure) Cache vs recompute.** Using `d2l.generate` with the cached and non-cached paths, measure tokens per second for 16 to 1024 new tokens; fit the exponent of the cost in the number of new tokens for each; compare with `sec_kv-cache`'s analysis. Report intervals over 5 repeats.
7. **(open) Port a new family.** Choose a released dense model that is not Qwen3 (for instance a Llama-3.2 or an OLMo 2 checkpoint), list the gaps between its configuration and `d2l.GPT` in the format of `tab_lm_qwen3_gaps`, add what is missing, and validate with the protocol of D1.2c.

#### Further reading

- Papers: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (architecture paragraph: GQA, RoPE base, QK-norm replacing QKV bias).
- Documentation: [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B); [safetensors docs](https://huggingface.co/docs/safetensors/index).
- Code: [Raschka, LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch) (his Qwen3 from-scratch notebook is the closest sibling: same checkpoint, same permutation question); [Raschka's from-scratch materials](https://www.sebastianraschka.com/from-scratch/); new: [`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) (the reference being matched).
- Background: [HF LLM Course, chapter 2](https://huggingface.co/learn/llm-course/chapter2/1) for the Auto-class machinery the port bypasses.

#### Risks and open decisions

- **Subclasses, not book edits (design, not fallback).** `d2l.Qwen3Attention`, `d2l.SwiGLU` and `d2l.Qwen3GPT` are defined in 1.2's notebook and subclass `d2l.GQAAttention`, `d2l.FeedForward` and `d2l.GPT`; no book class changes. Cost: about 80 lines of notebook code. Risk: the subclasses skip their parents' `__init__` (to avoid allocating default-width layers), so they depend on the parents' attribute names (`act` for `FeedForward.forward`; `pos`, `max_len`, `blks`, `token_emb`, `norm` for `GPT`); if the book renames one, the equivalence test (GPT-to-`Qwen3GPT` logits) fails, which is its purpose.
- **`generate_cached` is not saved by the book.** In `sec_kv-cache` the cells that define `d2l.GPT.forward_cached` and `generate_cached` (and the cached `forward_step` and `rope`) carry no `#@save`; only `GQAAttention` does. `Qwen3GPT` therefore defines its own `forward_cached` and inherits the sec_gpt `generate` (it calls only `self(X)`), and defines `generate_cached` as well (12 lines, the book's loop; the throughput table needs it), so this part does not depend on a book edit. If the book later marks that cell `#@save`, the override can be dropped.
- **`d2l.generate_ids` rename (book-side).** `sec_decoding`'s `generate(step_fn, prefix, num_tokens, eos_id=None, **strategy)` is renamed `d2l.generate_ids` in the book: `chapter_recurrent-neural-networks/decoding.md` (definition and 2 calls), `chapter_recurrent-modern/mamba.md` (2 calls) and `docs/rnn-ssm-modernization-apis.md`. 1.2's `d2l.generate` is only the batched API and has no dispatch. Until the rename lands, the Mamba and 1.1 cells call a name that does not exist yet; the test row checks the Mamba outputs after the rename.
- **FP32 tolerance.** `1e-3` is provisional; if the measured Transformers CPU-vs-GPU difference is larger, the autograder uses the larger.
- **Memory on the build box.** Two FP32 0.6B models need about 5 GB; if the build box shares memory between two slots, hold the HF model on CPU for the FP32 gate and move hooks' outputs only.
- **4B and 8B.** Loading 8B in BF16 needs 16.4 GB: more than a 15 GiB build slot, within a 24 GB card. The build exercises the function on 4B (8 GB) and on 8B by header and mapping dry run (names and shapes, no weights); an optional cell, guarded by `torch.cuda.mem_get_info()`, loads 8B on a whole 24 GB GPU and checks its logits, the only real-weights test of `tie_embeddings=False` (0.6B, 1.7B and 4B are tied).

### 1.3 Prompting and in-context learning

- **Label:** `sec_lm_prompting` · **Tier:** T1 (reduced T0 config: `Qwen/Qwen3-0.6B-Base` and `Qwen/Qwen3-0.6B` on 50 test items, 2 seeds, CPU is feasible for the format and template cells only; flag `SMALL=True`) · **Budget:** the default grid is sized to fit about 15 min on the build box per model pair after the author measures tokens per second; the committed grid artifact lets the figure cells run in seconds (`RERUN=False`); peak GPU memory about 4 GB (one 1.7B model in BF16 at a time, prompts of at most about 1,100 tokens) · **Codex units:** new (no Codex unit)
- **Builds on (book):**
  - `sec_what-attention-computes`: induction heads and the repeated-sequence task (`subsec_two-block-induction`, "Induction Heads in Larger Models", `tab_induction-evidence`). Reused: the repeated-random-token probe as a behavioral signature of in-context copying; the evidence ladder for what a behavioral result can claim. Not re-taught: the circuit.
  - `sec_decoding`: sampling dials; here scoring is by restricted next-token logits, so only greedy-equivalent decisions are used.
  - `sec_softmax`: calibration vocabulary, used in D1.3.
  - `sec_mdl-statistics`: bootstrap over items (a 6-line local version is used; the shared `d2l.paired_bootstrap` arrives in 1.5 and the text says so).
- **Builds on (this part):** 1.1 (`d2l.load_hf`, `apply_chat_template`, the `enable_thinking` switch, Qwen3 default thinking on, Qwen3.5 default off), 1.2 (`d2l.generate` for the free-generation spot checks).
- **Exports (`#@save`):**
  - `d2l.fewshot_prompt(demos, query, template='plain', *, label_map=None, system=None) -> str | list[dict]`: `demos` is a list of `(text, label)`; `template` is `'plain'` (`"Article: {x}\nTopic: {y}\n\n"` repeated, query last with the label left open), `'qa'` (`"Q: ... \nA: ..."`), `'chat_multi'` (a list of messages: user = text, assistant = label, repeated), `'chat_single'` (one user message that contains all demonstrations and the query); `label_map` maps dataset labels to the label words shown (natural words or abstract symbols); string templates return `str`, chat templates return the message list for `tokenizer.apply_chat_template`.
  - `d2l.label_logprobs(model, tokenizer, prompts, label_words, *, batch_size=8, chat=False, enable_thinking=False) -> torch.Tensor`: `(len(prompts), len(label_words))` log-probabilities of each label word's first token as the next token, renormalized over the label set (a log-softmax restricted to those token ids). Asserts that the first tokens of the label words are distinct (otherwise raises with the colliding words) and that each label word is a single token when preceded by the format's separator (the leading-space convention is handled by encoding `separator + word` and taking the last token of the difference). Works for a Transformers model (left padding with `attention_mask`, only the last-position logits are read through `logits_to_keep=1`) and for `d2l.GPT` with equal-length buckets.
  - `d2l.calibrate_labels(logprobs, content_free_logprobs) -> torch.Tensor`: contextual calibration of D1.3: subtracts the mean of the content-free log-probabilities (taken over the supplied content-free inputs) and renormalizes.
- **Models and data:**
  - Models: `Qwen/Qwen3-1.7B-Base` (`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`) and `Qwen/Qwen3-1.7B` (`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`), BF16, same size and family so that "base vs instruct" is the only difference; T0 pair Qwen3-0.6B-Base/-0.6B. Optional comparison: `Qwen/Qwen3.5-2B-Base` vs `Qwen/Qwen3.5-2B` (hybrid; an exercise).
  - Data: `fancyzhx/ag_news` (4 classes: World, Sports, Business, Sci/Tech; train 120,000 for demonstrations, test 7,600 for evaluation; the Hub card lists the license as unknown: the notebook downloads it at run time and the committed artifacts contain only item indices and predictions, never article text). A fixed stratified test subset of 200 items (50 per class, seed 0) is the default; demonstrations are drawn from `train` by seed, never from the test subset. Content-free inputs for calibration: `"N/A"`, `""`, `"[MASK]"`.
- **Packages:** `transformers >= 5.2.0`, `scikit-learn` (logistic regression for the probe baseline), `datasets`.

#### Narrative arc

1. **The question.** Opening cell: classify one AG News headline with Qwen3-1.7B-Base zero-shot (restricted label logits) and with 8 demonstrations in two orders that differ only by swapping two demonstrations; print both predicted labels and the label probabilities. The symptom: the prediction flips, or its confidence moves by a visible amount, under a change that carries no information. "How much of what we call the model's ability is the prompt?"
2. **Chat templates as tokenized input** (`subsec_lm_templates`).
   - Render and tokenize a three-message conversation with Qwen3-1.7B: system, user, assistant. Print the string with `<|im_start|>`, `<|im_end|>` visible and the token ids with a color-coded role per token (a `d2l` print helper in the cell). Show where `add_generation_prompt=True` stops (`<|im_start|>assistant\n`), what `enable_thinking=False` appends (`<think>\n\n</think>\n\n`), and what the Qwen3.5 template does by default (closes the think block unless `enable_thinking=True`; the contrast with Qwen3 from 1.1).
   - What a template is not: the model sees only token ids. Wrong-template experiment: score the same zero-shot item for the instruct model with the template, as plain text without it, and with the base model's template (identical string): the three label distributions printed; no accuracy claim yet.
   - System prompts: a one-line system instruction naming the label set vs none; effect on zero-shot accuracy for the instruct model (measured in the grid, E3).
3. **Zero-shot and few-shot formats** (`subsec_lm_formats`).
   - The four formats of `d2l.fewshot_prompt` (`plain`, `qa`, `chat_multi`, `chat_single`), printed for k=2. Label words: natural (`World`, `Sports`, `Business`, `Tech`) vs abstract (`foo`, `bar`, `baz`, `qux`).
   - How the answer is read: restricted next-token log-probabilities over the four label first tokens (1.4 will compare this with other conventions and with generation). Accuracy is the argmax.
4. **The grid** (`subsec_lm_grid`). One experiment family, run on both checkpoints with the same test items and the same demonstration draws so that every comparison is paired:
   - E1 number: k in {0, 1, 2, 4, 8, 16}, `plain` format, natural labels, classes balanced for k >= 4, 3 demonstration seeds, fixed order.
   - E2 order: k=8, one demonstration set, 12 random permutations; report the range and standard deviation of accuracy across permutations and compare them with the across-seed standard deviation of E1.
   - E3 format: the four formats at k=8, natural vs abstract labels, 3 seeds; plus the system-prompt condition at k=0 for the instruct model.
   - E4 label correctness: fraction rho in {0, 0.25, 0.5, 0.75, 1} of demonstration labels replaced by a uniformly random wrong label, k=8, 3 seeds, natural and abstract labels. (Min et al. and Wei et al. disagree in emphasis; this is the experiment that shows how they can both hold.)
   - E5 calibration: contextual calibration (D1.3) applied at k in {0, 1, 4}.
   - E6 weights vs context: logistic regression on the base model's last-layer mean-pooled features (standardized, L2, `C=1`) of the same demonstrations used for each k (no prompt, no template), tested on the same items.
   - A "predict first" cell: the reader enters predictions for E2 (range wider or narrower than the E1 across-seed spread?), E4 (accuracy drop at rho=1 for natural and abstract labels, base vs instruct) and E6 (the k at which the probe overtakes ICL, if ever); the figure cells overlay them.
5. **The mechanism's behavioral signature** (`subsec_lm_induction`). Repeated random tokens: 64 random token ids repeated once; per-position loss of Qwen3-1.7B-Base on the first vs the second copy. This reproduces the lab of `sec_what-attention-computes` on a released model: the second-copy loss collapses (direction taken from the book's lab and the induction-heads paper; magnitude to be measured). Paragraph on the evidence ladder: this is behavioral evidence of copying, not a component-level claim; the component-level test is an exercise in 1.6's tools.
6. **In-context learning vs weight updates** (`subsec_lm_icl_vs_weights`): E6 result and a paragraph that separates three things: conditioning on demonstrations (no weights change, cost grows with context), fitting a head (E6), and fine-tuning (7.2). Sample-efficiency curve; the notebook does not claim ICL "is" gradient descent (pointer to the literature in the further reading; not taught).

#### Derivations

**D1.3: Contextual calibration as a prior correction** (`eq_lm_ctx_calibration`).
- *Statement.* Suppose the model's label posterior factorizes as `p_M(y | x) proportional to q(x | y) pi(y)` with `pi` an implicit label prior that depends on the prompt (majority label, recency, label-word frequency) and `q` the evidence term. If a content-free input `x0` carries no evidence (`q(x0 | y)` equal for all `y`), then `p_M(y | x0) = pi(y)` and the corrected posterior `p_cal(y | x) proportional to p_M(y | x) / p_M(y | x0)` equals the posterior under a uniform prior. For one-hot class weights the corrected argmax is the maximum-likelihood label.
- *Assumptions.* The prompt fixes `pi`; `x0` is uninformative for all labels; a single `x0` or the mean of several.
- *Steps.* (1) Write `log p_M(y|x) = log q(x|y) + log pi(y) - log Z(x)`. (2) For `x0`: `log p_M(y|x0) = log c + log pi(y) - log Z(x0)` (with `q(x0|y) = c`). (3) Subtract: `log p_M(y|x) - log p_M(y|x0) = log q(x|y) + const(x, x0)`, constant in `y`. (4) Renormalize over `y`. (5) With several content-free inputs, replace `log p_M(y|x0)` by their mean; the estimate's variance falls as `1/m`.
- *Check cell (CPU).* Simulate a 4-class Gaussian classifier with known `q` and a biased prior `pi = (0.7, 0.1, 0.1, 0.1)`: uncalibrated accuracy vs calibrated accuracy vs accuracy of the uniform-prior Bayes rule on 20,000 draws; calibrated and uniform-prior accuracies must agree to within Monte Carlo error (`3 / sqrt(20000)`), the uncalibrated one lower. A second cell applies `d2l.calibrate_labels` to the real model at k=0.

#### From-scratch implementation

Components (about 90 lines): `fewshot_prompt` (30), `label_logprobs` (35, including the tokenization of label words with separator), `calibrate_labels` (8), a local `boot_ci(correct, n_boot=2000, seed=0)` (8), the grid runner (10, behind `RERUN`).

Numerical tests:

| Test | Reference | Tolerance | Seed |
|---|---|---|---|
| `label_logprobs` equals the full-vocabulary `log_softmax` at the label token ids, renormalized over the four ids | direct computation on 4 prompts | `1e-5` (FP32) | none |
| Batched (left-padded) vs one-at-a-time `label_logprobs` | itself | `1e-4` in FP32; in BF16 report the maximum (batch-dependent kernels, 1.4) | none |
| `fewshot_prompt` token count for `plain` at k=0 equals the hand-counted token ids of the committed fixture | fixture | exact | none |
| Label-collision assertion fires for labels `Tech`, `Technology` | n/a | raises | none |
| D1.3 simulation | uniform-prior Bayes | `3 / sqrt(20000)` | 0 |

#### In production

- `tokenizer.apply_chat_template` is the production path and is used in every chat cell; `lm-evaluation-harness` and `lighteval` implement few-shot prompting with `fewshot_as_multiturn` options (pointer, not run here; 1.4 uses the harness as a test oracle). Transformers' `generate` with `max_new_tokens=1`, `output_scores=True` is the alternative to the restricted-logit reader and is run on four prompts to show agreement.

#### Headline experiment

- **Hypothesis.** For a 1.7B model, (a) few-shot accuracy rises with k for both checkpoints, more for the base model at small k; (b) the spread of accuracy over demonstration order at fixed k is not small compared with the spread over demonstration sets; (c) correctness of demonstration labels matters little for natural label words and more for abstract label words (the pattern reported by Min et al. and by Wei et al.); (d) format changes (template, system prompt, chat vs plain) move the instruct model more than the base model, and in the base model the chat templates are a mismatch. Each of (a)–(d) is a prediction to be confirmed or refuted by the grid; magnitudes are to be measured.
- **Setup.** The two checkpoints above, BF16, one Ada GPU, 200 test items (50 per class), 3 demonstration seeds, grid E1–E6 with the sizes listed. Scoring by restricted label logits. Seeds: demonstration draw seeds 0, 1, 2; permutation seeds 100–111.
- **Procedure.** Run the grid, store per-item correctness for every condition (`lm-fewshot-grid.csv`: model, experiment, condition, seed, item id, correct). All plots and statistics read the CSV.
- **Metrics.** Accuracy per condition; paired difference against the reference condition of its panel (k=0 for E1, rho=0 for E4, `plain` natural for E3) with a 95% paired bootstrap interval over test items, resampling items within a seed and then averaging over seeds; across-seed standard deviation; permutation range.
- **Plots.**
  - `fig_lm_icl_k`: x = k (0, 1, 2, 4, 8, 16), y = accuracy; series: base, instruct, E6 probe (base features); line = mean over seeds, band = min to max over the 3 seeds; item-bootstrap interval of the k=16 point as an error bar.
  - `fig_lm_icl_order`: strip plot of 12 permutation accuracies at k=8 for each model next to the 3 seed-level accuracies; horizontal line at the mean.
  - `fig_lm_icl_labels`: x = fraction of wrong labels rho, y = accuracy; four lines (base/instruct x natural/abstract).
  - `tab_lm_prompt_formats`: accuracy by format and label type, with paired interval against `plain` natural.
- **Expected outcome and sources.** Directions (a)–(d) above; the literature: accuracy grows with demonstrations (GPT-3, arXiv 2005.14165); permutation spread can be very large for small models (arXiv 2104.08786); label correctness mattering little (arXiv 2202.12837) vs larger models overriding priors with flipped labels and learning abstract labels (arXiv 2303.03846); prompt-format sensitivity (arXiv 2310.11324). Magnitudes for these 1.7B checkpoints: to be measured.
- **Noise check.** Every claimed effect needs (i) a paired-bootstrap interval over items that excludes 0 and (ii) an effect larger than the across-seed standard deviation of its condition; effects failing either are reported as "not distinguishable at this sample size". 200 items give a binomial interval of about plus or minus 7 points at 50% accuracy by the normal approximation `1.96 sqrt(0.25/200)`: panels do not interpret differences below that unless paired.
- **T1 reduction.** Already T1; `SMALL=True` runs the 0.6B pair on 50 items with 2 seeds (interpretation limited to format and template cells).
- **Artifacts committed.** `lm-fewshot-grid.csv` (per item; about 100k rows at default size; item ids and booleans only), the permutation list, the demonstration index lists per seed.

#### Figures and tables

- `fig_lm_icl_k`, `fig_lm_icl_order`, `fig_lm_icl_labels`, `tab_lm_prompt_formats` as above.
- `fig_lm_chat_tokens`: the rendered conversation with role-colored tokens (made by the template cell).
- `tab_lm_templates`: how Qwen3 and Qwen3.5 templates differ in what they add before the answer (default thinking state, generation prefix).
- `fig_lm_repeat_loss`: per-position loss over the 128-token sequence, first and second copy, Qwen3-1.7B-Base.

#### Pitfalls and misconceptions

- "The model learned the task from the demonstrations": with natural labels much of the benefit is format and label-space information (Min et al.), not input-label mapping.
- Treating one demonstration order or one seed as "the" few-shot score.
- Leading-space tokenization of label words (`" Sports"` vs `"Sports"`): the first-token reader breaks silently when the format changes.
- Chat-templating a base model, or omitting the template for an instruct model, and attributing the change to "ability".
- Putting test items or near-duplicates in the demonstration pool.
- Reading restricted-logit accuracy as the accuracy the model would have if it generated freely (1.4).
- Few-shot as free: cost grows with prompt length (E6 compares at equal labeled data, not at equal cost).

#### Summary (draft bullets)

- The model sees token ids; a chat template is a deterministic rendering that decides what precedes the answer, and templates differ across families and versions.
- Few-shot accuracy depends on number, order, format and label space of demonstrations; the dependence is measured against the right noise: across seeds, across orders, and across test items.
- Natural labels let wrong demonstrations hurt little; abstract labels expose how much is learned from the input-label mapping.
- Contextual calibration removes the prompt-induced label prior and is exact under its assumption.
- In-context learning conditions without changing weights; it is a different object from fitting a head or fine-tuning, with different costs.

#### Exercises

1. **(check) What the generation prefix contains.** For Qwen3-1.7B with `enable_thinking=False`, write the exact last 12 tokens (as strings) of the prompt for a one-turn conversation. *Sketch:* `...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` with each special token one id.
2. **(check) Which labels can the first-token reader score?** Give a label set of four words that is invalid for `label_logprobs` and explain how to repair it. *Sketch:* any two sharing a first token (`Tech`/`Technology`); repair by using letters or by scoring the whole label (1.4).
3. **(derive) Calibration is exact under its assumption.** Prove D1.3 and show what is left of the correction if `x0` carries some evidence `q(x0|y) = c(y)`. *Sketch:* the subtraction removes `log c(y)` together with the prior, a bias of `log c(y)` in the corrected posterior.
4. **(derive) How large an item sample?** To distinguish two conditions whose true accuracies differ by 3 points with a paired design whose per-item disagreement rate is 15%, how many test items give a 95% interval half-width of 3 points? *Sketch:* the paired difference has variance `d / n` where `d` is the disagreement rate (for small true gap); `1.96 sqrt(0.15 / n) = 0.03` gives `n` about 640.
5. **(measure) Permutation spread vs scale.** Predict whether E2's permutation spread is larger for the 0.6B or the 1.7B pair of base models, then run both and report the ranges with bootstrap intervals over permutations.
6. **(measure) Hybrid vs dense.** Repeat E1 with Qwen3.5-2B-Base vs Qwen3-1.7B-Base at matched k; report the paired differences with intervals. Does the hybrid's recurrent state change the dependence on k?
7. **(open) A format audit.** Choose a task from the 1.4 suite and a second post-trained model. Design and run a prompt-format audit (at least 6 formats), report accuracy with paired intervals, and write a recommendation of the format to use and the one to avoid, with evidence.

#### Further reading

- Papers: [GPT-3](https://arxiv.org/abs/2005.14165) (few-shot as a capability); [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837) (what demonstrations contribute); [Induction heads](https://arxiv.org/abs/2209.11895) (mechanistic account of in-context copying); new: [Calibrate Before Use](https://arxiv.org/abs/2102.09690) (D1.3); [Fantastically Ordered Prompts](https://arxiv.org/abs/2104.08786) (order); [Larger language models do in-context learning differently](https://arxiv.org/abs/2303.03846) (flipped and abstract labels); [Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design](https://arxiv.org/abs/2310.11324) (format).
- Documentation: [chat templating](https://huggingface.co/docs/transformers/chat_templating) (Transformers).
- Code and course: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (few-shot construction options); HF LLM Course chapter 2 for tokenizers and templates.

#### Risks and open decisions

- **Dataset license.** AG News's Hub card lists the license as unknown; if the editor prefers a clearly licensed set, swap to `stanfordnlp/sst2` (also unknown) or to ARC-style multiple choice (`allenai/ai2_arc`, CC BY-SA 4.0, used in 1.4) and keep the same code. Artifacts contain no article text either way.
- **Runtime.** Grid size may exceed 15 minutes per pair; reductions in order: seeds 3 to 2, items 200 to 150, E3 abstract-label cell dropped. The committed CSV makes the figures independent of this.
- **Qwen3.5 template defaults** (non-thinking when `enable_thinking` is undefined) are read from the chat template file of the pinned revision; the notebook prints the rendered tail so the text cannot drift from the file.
- **Restricted-logit reader.** If a label word is not a single token after the separator in some format, that format is dropped from E3 and the reason printed (not silently).

### 1.4 Scoring language models

- **Label:** `sec_lm_scoring` · **Tier:** T1 (reduced T0 config: Qwen3-0.6B-Base, 100 ARC items, 20 GSM8K problems with n=4; flag `SMALL=True`) · **Budget:** T1 about 15 min wall-clock with committed sample pools (the scoring cells take a few minutes; the generation cells run only with `RERUN=True`, about 10 min, to be measured); peak GPU memory about 4 GB (one 1.7B model at a time) · **Codex units:** Codex 15.2 (task scoring mechanics); the perplexity and bits-per-byte material of Codex 15.1 is folded in as the likelihood subsection because v5 lists `sec_language-model` as this section's prerequisite
- **Builds on (book):**
  - `sec_language-model`: perplexity and bits per byte. Reused as definitions; this section adds the tokenizer-invariance argument (D1.4c) and a windowed evaluator.
  - `sec_softmax`: calibration, reused for the reliability diagram of choice probabilities.
  - `sec_decoding`: sampling for the pass@k pools; `d2l.sample_logits` and `d2l.generate` from 1.2 are the samplers.
  - `sec_mdl-statistics`: bootstrap (local 8-line version; the shared function is 1.5's).
- **Builds on (this part):** 1.1 (`d2l.load_hf`), 1.2 (`d2l.generate`, `d2l.Generation`, `d2l.sample_logits`), 1.3 (`d2l.label_logprobs`, a special case of the scoring here; `d2l.fewshot_prompt`).
- **Exports (`#@save`):**
  - `d2l.score_choices(model, tokenizer, prompts, choices, norm='token', *, batch_size=8, return_components=False) -> torch.Tensor | d2l.ScoreComponents`. `prompts`: `list[str]` of length `N`; `choices`: `list[list[str]]` (`N` lists, possibly of different lengths) or one `list[str]` shared by all prompts. Returns a float tensor `(N, C_max)` of normalized scores, with `-inf` in padded columns; the prediction is `argmax(dim=-1)` (first index on ties). Contract per norm, with `ll` the sum over the continuation tokens of the log-probability of the token given the prompt and the preceding continuation tokens (log-softmax computed in FP32 whatever the model dtype):
    - `'sum'`: `ll` (the harness's `acc` convention);
    - `'token'`: `ll / n_tokens` (mean log-probability per continuation token);
    - `'char'`: `ll / len(choice)` in Unicode characters (the harness's `acc_norm`; the author confirms the length definition against the pinned harness version);
    - `'byte'`: `ll / len(choice.encode('utf-8'))`.
    Tokenization convention (the harness's): encode `prompt + choice` and `prompt` separately, take the continuation as the suffix of the first beyond the length of the second; if `prompt` ends with whitespace, move that whitespace to the front of the choice; no BOS is added (Qwen3 has none; for a tokenizer that has `bos_token`, `add_special_tokens=False` is used and the first-token problem is documented in the docstring). A choice whose continuation is empty after tokenization raises `ValueError`. A prompt longer than the model's context is truncated from the left with a warning that names the item. Batching: right-padded with `attention_mask`; the function sorts by length, restores the original order, and reports in a `warnings.warn` if the batch size can change the result (always true in BF16; see D1.4 and the nondeterminism subsection).
  - `d2l.ScoreComponents` (`typing.NamedTuple`): `ll`, `n_tokens`, `n_chars`, `n_bytes`, each `(N, C_max)`; one forward pass yields every convention (`'sum'/'token'/'char'/'byte'` are elementwise functions of these), so conventions are compared without re-running the model.
  - `d2l.format_mc(question, options, fmt='cloze', *, subject=None, shots=()) -> tuple[str, list[str]]`: returns `(prompt, choices)`. `fmt='cloze'`: prompt `"Question: {q}\nAnswer:"`, choices `[" " + option for option in options]`. `fmt='letter'`: the question, the options on lines `A. ...`, `B. ...`, then `"Answer:"`; choices `[" A", " B", " C", " D"]`. `shots` is a list of solved examples in the same format. This pair of formats follows Robinson and Wingate (arXiv 2210.12353).
  - `d2l.pass_at_k(n, c, k)`: unbiased estimator of D1.4a. Accepts Python ints or integer `numpy` arrays (broadcast), returns `float` or `float64` array. Numerically stable product form `1 - prod_{i=n-c+1}^{n} (1 - k/i)`. Raises `ValueError` if `c < 0`, `c > n`, `k < 1` or `k > n`. If `n - c < k` returns 1.0 exactly; if `c == 0` returns 0.0 exactly.
  - `d2l.pass_at_k_variance(n, p, k) -> float`: exact variance of the estimator when each sample succeeds independently with probability `p` (the binomial sum of D1.4a).
  - `d2l.extract_answer(text, kind='gsm8k') -> str | None`: `kind='gsm8k'`: the number after `####` if present, else after `"the answer is"` (case-insensitive), else the last number in the text; commas and trailing periods stripped; returns the normalized string or `None`. `kind='boxed'`: the content of the last `\boxed{...}` (brace-matched). `strict=True` disables the last-number fallback.
  - `d2l.run_python_tests(program, test_code, timeout=5.0) -> bool`: runs `program + "\n" + test_code` in a fresh `python -I` subprocess in a temporary directory with a wall-clock timeout, `resource` limits for memory and CPU time on POSIX, stdin closed. It is not a security boundary (documented in the docstring and in the pitfalls): run model-written code only in a container or VM.
  - `d2l.bits_per_byte(model, tokenizer, texts, ctx_len=2048, stride=None, batch_size=4) -> float`: windowed evaluation of a causal LM on raw texts: sliding windows of `ctx_len` tokens with step `stride` (default `ctx_len // 2`); every token is scored exactly once, with at least `ctx_len - stride` tokens of context except in the first window; the sum of negative log-likelihoods in nats divided by `ln(2) * total_utf8_bytes(texts)`. Also exposes `d2l.perplexity` (token-level, same windows) as a second return of `return_tokens=True`. Edge cases: a text shorter than `ctx_len` is a single window; no BOS; documents are scored independently (no cross-document context).
- **Models and data:**
  - Models: `Qwen/Qwen3-0.6B-Base` (`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`), `Qwen/Qwen3-1.7B-Base` (`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`), `Qwen/Qwen3-1.7B` (`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, post-trained, `enable_thinking=False`), `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`). Four checkpoints give six pairs.
  - Data (all on the Hub as of 28 Oct 2026): `allenai/ai2_arc` config `ARC-Challenge`, test, 1,172 items (CC BY-SA 4.0); `Rowan/hellaswag` validation, fixed seeded subset of 2,000 of 10,042 (license per the dataset card: the author records it); `openai/gsm8k` config `main`, test, a fixed seeded 100 of 1,319 problems, 4-shot with the first four `train` problems as demonstrations (MIT); `openai/openai_humaneval`, 40 seeded problems of 164 (MIT); `Salesforce/wikitext` config `wikitext-2-raw-v1`, test, 1.3M characters (CC BY-SA 3.0 and GFDL). `lm_eval_reference_arc.json`: per-item loglikelihoods and `acc`, `acc_norm` from `lm-evaluation-harness` for Qwen3-0.6B-Base on the first 200 ARC-Challenge test items, produced once by the author at a pinned harness version and committed.
- **Packages:** `lm-eval` (pinned; used only to make the fixture and in the "In production" cell), `datasets`, `transformers >= 5.2.0`.

#### Narrative arc

1. **The question.** Opening cell: two checkpoints, one task (ARC-Challenge, 1,172 items), one number each under the convention a reader would pick without thinking (`cloze`, `'sum'`), then under two more; the printed table has the ranking of the two checkpoints change between rows. The measured symptom: "model A beats model B" is a statement about a convention. (The direction of the flip is a hypothesis; the cell prints whichever result occurs, and "no flip found" is a valid and reported outcome.)
2. **Scoring by likelihood** (`subsec_lm_loglik`).
   - What a score is: the log-probability a model assigns to a continuation given a prompt; tokenization boundary convention (the harness's), BOS, trailing whitespace, FP32 log-softmax.
   - Cloze vs letter formats: `format_mc` printed for one ARC item in both formats; in the letter format the model must map the content to a symbol, an ability that small base models lack and post-trained models often have (the finding of arXiv 2210.12353, direction only).
   - Length normalization: `'sum'`, `'token'`, `'char'`, `'byte'`, one forward pass through `ScoreComponents`. Derivation D1.4b says when conventions can disagree and what each assumes.
   - A pair of toy items where normalization flips the decision, printed with token counts.
3. **Ranking flips, measured** (`subsec_lm_flips`).
   - Grid: 4 checkpoints x conventions {cloze-sum, cloze-token, cloze-char, letter} x tasks {ARC-Challenge, HellaSwag-2000}. Table `tab_lm_conventions` with accuracy and a bootstrap 95% interval over items per cell.
   - For every pair of checkpoints, the paired difference per convention with a paired bootstrap interval over items (local 8-line `boot_diff`; 1.5 generalizes). A ranking flip is reported only if the paired interval excludes 0 with opposite signs under two conventions.
   - Agreement with the harness: on the 200-item fixture, `score_choices(..., 'sum')` and `'char'` predictions equal the harness's `acc` and `acc_norm` choices on all but a stated number of items; the cell lists the disagreeing items with their score margins (expected to be near-ties in BF16).
4. **Are the probabilities calibrated?** (`subsec_lm_calibration`) Softmax over the normalized scores as confidence; reliability diagram with 10 equal-mass bins and expected calibration error with a bootstrap interval, per convention. (`sec_softmax` defines the diagnostics; the point here is that confidence depends on the convention too.)
5. **Likelihood of text and tokenizers** (`subsec_lm_bpb`).
   - Perplexity per token vs bits per byte on wikitext-2 test for Qwen3-0.6B-Base (151,936-entry vocabulary) and Qwen3.5-0.8B-Base (248,320-entry vocabulary): token perplexities are not comparable across tokenizers; bits per byte are (derivation D1.4c). The cell prints tokens per byte for both tokenizers and checks the identity `bpb = (T/B) log2 PPL_tok`.
   - Window policy: with `stride < ctx_len` vs `stride = ctx_len` (no overlap), the bits-per-byte difference (to be measured; direction: lower with more context).
6. **Generative answers and answer extraction** (`subsec_lm_generative`).
   - GSM8K, 4-shot chain of thought on `Qwen3-1.7B-Base` and `Qwen3-1.7B` (non-thinking), n=8 samples per problem at temperature 0.6, top-p 0.95 (the model's own `generation_config` values), `max_new_tokens=256`; generations produced by `d2l.generate` (seeded) or loaded from the committed pool.
   - Three extractors (strict `####`, "answer is", last number) change accuracy: the table shows the same generations under each (this is a measurement of the scorer, not the model).
   - Execution-based check: HumanEval-40, n=10 samples per problem at temperature 0.8, `run_python_tests` with a 5 s timeout; pass rates per problem.
   - pass@k from the pools for k in {1, 2, 4, 8} (GSM8K) and {1, 5, 10} (HumanEval), with problem-level bootstrap intervals; `maj@8` for GSM8K computed in one line and labeled as a different quantity (aggregation across samples, not an estimate of pass@k; 9.5 develops it).
7. **Nondeterminism** (`subsec_lm_nondeterminism`).
   - Same prompt, same weights, different batch composition: the logits of a fixed 128-token prompt computed alone and as a member of batches of 2, 4, 8, 16 (neighbors chosen from other prompts, so padding differs too): maximum absolute logit difference in FP32 and BF16, and the fraction of 32 prompts whose 128-token greedy continuation changes between batch size 1 and 16 (and the median first-divergence index). Then the controls: FP32; batch size fixed; `torch.use_deterministic_algorithms(True)` (fixes run-to-run reproducibility at a fixed batch, not batch invariance: the cell shows both); a note on batch-invariant kernels (the Thinking Machines post).
   - Consequence for scoring: the item-level predictions near ties depend on the batch size; the harness's fixed `batch_size` belongs in the report.

#### Derivations

**D1.4a: The unbiased pass@k estimator and its variance** (`eq_lm_passk`, `eq_lm_passk_var`).
- *Statement.* A problem is solved by a single sample with probability `p`. The target is `pass@k = 1 - (1-p)^k`, the probability that at least one of `k` independent samples is correct. Given `n >= k` samples with `c` correct, `U(n, c, k) = 1 - C(n-c, k) / C(n, k)` is an unbiased estimator of `pass@k` (`C` the binomial coefficient, `C(m, k) = 0` for `m < k`). Its variance is `Var(U) = sum_{c=0}^{n} Bin(c; n, p) U(n,c,k)^2 - (1 - (1-p)^k)^2`. Special cases: `k = 1`: `U = c/n` and `Var = p(1-p)/n`; `k = n`: `U = 1[c >= 1]` and `Var = (1-p)^n (1 - (1-p)^n)`.
- *Assumptions.* The `n` samples are independent and identically distributed given the prompt; the verifier is deterministic.
- *Steps.* (1) Let `C ~ Binomial(n, p)`. Given `C = c`, the `n` outcomes are a uniformly random arrangement of `c` ones. (2) Draw a uniformly random size-`k` subset of the `n` samples; it contains no correct sample with probability `C(n-c, k) / C(n, k)`. So `U` is the conditional probability, over the subset, of at least one success. (3) Unconditionally, a size-`k` subset of iid samples is itself `k` iid samples, so `P(no success) = (1-p)^k`. (4) By the tower property, `E[U] = E[ P(subset has a success | C) ] = 1 - (1-p)^k`. (5) The variance is the second moment of `U(C)` over the binomial pmf minus the squared mean. (6) The common "plug-in" `1 - (1 - c/n)^k` is biased: its expectation differs because `(1 - C/n)^k` is a convex function of `C`. (7) Numerically stable form: `C(n-c, k)/C(n, k) = prod_{i=n-c+1}^{n} (1 - k/i)`, which never forms large binomials.
- *Check cell (CPU).* For `n = 20`, `p = 0.3`, `k = 5`: the exact binomial sum gives `E[U] = 0.83193` equal to `1 - 0.7^5 = 0.83193`, standard deviation `0.1344` (variance `0.018063`), while the plug-in has expectation `0.79583` (bias `-0.0361`). Monte Carlo with 200,000 draws (seed 0) must match both exact values to within `3 sigma / sqrt(200000)`. A second cell checks `pass_at_k(200, 37, 10) = 1 - C(163,10)/C(200,10) = 0.87737` and `k = 1` equals `c/n`.

**D1.4b: Why conventions are different decision rules** (`eq_lm_norm_flip`).
- *Statement.* For choices `a, b` with continuation lengths `l_a, l_b` (tokens) and mean log-probabilities per token `m_a, m_b < 0`, `'sum'` prefers `a` iff `l_a m_a > l_b m_b` and `'token'` prefers `a` iff `m_a > m_b`. They disagree exactly when `(m_a - m_b)` and `(l_a m_a - l_b m_b)` have opposite signs. If `l_a > l_b`, disagreement with `a` preferred by the mean means `|m_a| < |m_b| < (l_a / l_b) |m_a|`: the longer choice is slightly less surprising per token but costs more in total. `'char'` and `'byte'` replace token counts by character or byte counts and agree with `'token'` only if tokens-per-character is equal across choices.
- *Assumptions.* Scores are exact log-probabilities of full continuations, independent of tokenization boundaries.
- *Steps.* (1) Define `S_x = l_x m_x`. (2) `S_a > S_b` and `m_a > m_b` agree unless the sign of `S_a - S_b = l_a m_a - l_b m_b` differs from that of `m_a - m_b`. (3) With `l_a > l_b` and `m_a > m_b` but `S_a < S_b`: `l_a m_a < l_b m_b` gives `|m_a| l_a > |m_b| l_b`, i.e. `l_a / l_b > |m_b| / |m_a| > 1`. (4) The disagreement region is therefore a wedge between `|m_b|/|m_a|` and `l_a/l_b`; its probability mass grows with the spread of choice lengths (this is why HellaSwag, whose endings vary in length, is sensitive and a task with single-word answers is not). (5) What each rule assumes: `'sum'` ranks by total probability of the string (a prior that favors short strings); `'token'`/`'char'`/`'byte'` rank by rate, which removes the length prior but not the prior over which strings are frequent (the reason unconditional or PMI normalization exists; exercise).
- *Check cell (CPU).* Draw lengths from a shifted geometric distribution and per-token rates from a normal; estimate the disagreement probability of `'sum'` vs `'token'` as the length spread grows; compare with the wedge formula's indicator on the same draws (identical by construction: the cell asserts equality of the two computations).

**D1.4c: Bits per byte does not depend on the tokenizer** (`eq_lm_bpb`).
- *Statement.* For a text of `B` UTF-8 bytes encoded canonically by a tokenizer into `T` tokens with total negative log-likelihood `NLL` nats under a model, `bpb = NLL / (B ln 2)` and `PPL_tok = exp(NLL / T)`, hence `bpb = (T / B) log2 PPL_tok`.
- *Assumptions.* Each model is evaluated on the canonical tokenization of its own tokenizer; `bpb` compares models only to the degree that they are evaluated on the same bytes; it is not invariant to marginalizing over alternative segmentations (the model assigns probability to one segmentation) nor to preprocessing.
- *Steps.* (1) `NLL` is the sum over tokens. (2) Dividing by `T` gives the mean per token whose exponential is `PPL_tok`. (3) Dividing by `B ln 2` instead normalizes by the text's size in the unit common to all tokenizers. (4) Substitute: `bpb = T log2(PPL_tok) / B`. (5) Consequence: a tokenizer with fewer tokens per byte (larger vocabulary) has a larger `PPL_tok` for the same `bpb`.
- *Check cell.* On wikitext-2 test: compute `T/B` for both tokenizers, the two per-token perplexities and the two bpb values; assert the identity to `1e-9` (relative) for each model; report that the per-token perplexities order the models differently from bpb if they do (to be measured; the identity is exact either way).

#### From-scratch implementation

Components (about 140 lines): `score_choices` with `ScoreComponents` (55), `format_mc` (15), `pass_at_k` and `pass_at_k_variance` (15), `extract_answer` (20), `run_python_tests` (15), `bits_per_byte` (20).

Numerical tests:

| Test | Reference | Tolerance | Seed |
|---|---|---|---|
| `score_choices` FP32, 8 ARC items: `'sum'` scores vs hand loop over `log_softmax` of the full sequence | direct computation | `1e-4` | none |
| `'sum'` and `'char'` predictions on the 200-item fixture | harness `acc`/`acc_norm` choices | at most 2 of 200 items differ per norm (provisional) | none |
| `score_choices` with `norm='token'` on a batch with different choice lengths vs one item at a time | itself | `1e-4` in FP32 | none |
| `pass_at_k` cases of the assignment table | closed forms | `1e-12` | none |
| Estimator mean and variance by simulation | exact binomial sums | `3 sigma` | 0 |
| `extract_answer` on 12 fixture strings (with and without `####`, commas, negative numbers, trailing periods, no number) | expected strings | exact | none |
| `run_python_tests` on 4 fixtures: correct, wrong, infinite loop, syntax error | `True, False, False, False` | exact; the loop returns within `timeout + 1` s | none |
| `bits_per_byte` for a text shorter than `ctx_len` vs the naive full-sequence computation | itself | `1e-6` | none |
| `bpb = (T/B) log2 PPL` identity | formula | `1e-9` relative | none |

#### In production

- `lm-evaluation-harness` (`HFLM` model wrapper, `simple_evaluate`, tasks `arc_challenge`, `hellaswag`, `gsm8k`; `acc`, `acc_norm`, `exact_match`; `--log_samples`) is the production tool and the test oracle for the from-scratch scorer (the 200-item fixture); `lighteval` is the alternative. HumanEval's reference implementation (`openai/human-eval`) defines the sandbox protocol; its README states that it executes untrusted model-generated code and ships with the execution call disabled until the user enables it: the notebook cites that warning next to `run_python_tests`. Transformers' perplexity guide defines the sliding-window procedure that `bits_per_byte` follows.

#### Headline experiment

- **Hypothesis.** The ranking of at least one pair among four checkpoints on a multiple-choice task depends on the scoring convention: in particular the post-trained model gains relative to its base under the letter format and loses or gains less under cloze scoring; and length normalization changes the ranking on HellaSwag more than on ARC.
- **Setup.** Four checkpoints (BF16), ARC-Challenge test (1,172 items) and HellaSwag (2,000-item seeded subset), conventions cloze-sum, cloze-token, cloze-char, letter, 0-shot for cloze and 0-shot for letter (a 5-shot letter variant as an extra column). Fixed batch size 8, BF16, seed irrelevant (no sampling; the item subset seed is 0).
- **Procedure.** One forward pass per (model, item, format) yields all cloze norms via `ScoreComponents`; the letter format a second pass. Predictions stored per item.
- **Metrics.** Accuracy with bootstrap 95% interval over items; paired difference per model pair and convention; the sign pattern across conventions.
- **Plot (`fig_lm_scoring_flip`).** Left: x = convention (4 categories), y = accuracy on ARC-Challenge, one line per checkpoint with item-bootstrap error bars. Right: x = convention, y = paired accuracy difference between the two checkpoints that flip most (or the chosen base/post-trained pair), with paired-bootstrap interval; horizontal zero line.
- **Expected outcome.** Direction as in the hypothesis; the literature motivating it: cloze vs letter formats and the emergence of letter-format ability with scale (arXiv 2210.12353) and the Open LLM Leaderboard's MMLU discrepancies (the Hugging Face post cited below). Magnitudes: to be measured. If no pair flips, the section reports that and shows the convention-dependence of the margins instead.
- **Noise check.** Paired bootstrap over items for every difference; a flip is claimed only if the intervals exclude 0 on both sides; BF16 batch-size sensitivity is measured in item 7 and the disagreement rate between batch sizes 1 and 8 is reported next to the flip table so that margins below it are not interpreted.
- **T1 reduction.** HellaSwag subset 2,000 (full 10,042 is optional); `SMALL=True` uses 0.6B only.
- **Artifacts committed.** Per-item scores `lm-arc-scores.parquet` (model, item id, four components for each choice; about 5 MB), `lm-gsm8k-samples.jsonl` and `lm-humaneval-samples.jsonl` (generations, correctness flags, seeds, generation settings) so that pass@k and extraction analyses run offline; `lm_eval_reference_arc.json`.

#### Figures and tables

- `fig_lm_scoring_flip`; `tab_lm_conventions` (accuracy, interval, per convention and checkpoint); `tab_lm_extractors` (GSM8K accuracy under three extractors for the same generations); `fig_lm_passk` (x = k, y = pass@k with problem-bootstrap bands, per checkpoint and task); `fig_lm_reliability` (reliability diagrams per convention); `tab_lm_bpb` (tokens per byte, token perplexity, bits per byte for two tokenizers, with the identity check); `fig_lm_batch_noise` (maximum absolute logit difference vs batch size, FP32 vs BF16, log y).

#### Pitfalls and misconceptions

- Comparing accuracies computed under different conventions or formats as if they measured the same thing.
- Token-level perplexity across tokenizers.
- pass@k estimated as `1 - (1 - c/n)^k`, or computed from `n = k` samples only, or confused with majority voting.
- Answer extractors that give the model credit for the last number it mentions.
- Executing model-written code outside a sandbox; `run_python_tests` is not a sandbox.
- Treating BF16 near-ties as ranking evidence; the batch size is part of the protocol.
- Scoring a reasoning model by the likelihood of the answer option without its reasoning: the likelihood of an answer letter measures what the model would say with no thinking.
- Prompt leakage of the label words in the few-shot demonstrations of the letter format.

#### Summary (draft bullets)

- A likelihood score is a measurement under a convention; sum, per-token, per-character and per-byte normalizations are different decision rules and can rank checkpoints differently.
- Letter and cloze formats measure different abilities; small base models are weak at the former.
- pass@k is estimated without bias from `n >= k` samples by `1 - C(n-c, k)/C(n, k)`; the plug-in version is biased and majority voting is a different quantity.
- Bits per byte is the tokenizer-independent likelihood score; token perplexity is not comparable across vocabularies.
- Answer extraction and execution sandboxes are part of the metric; batch composition changes BF16 results.

#### Exercises

1. **(check) Which convention?** For each of four tasks (single-word answers; four sentence-long endings; numeric answers; code), say which normalization you would use and why. *Sketch:* single-word: `'sum'` or `'token'` equivalent; endings: `'char'` or `'byte'`; numeric: generative with extraction; code: execution.
2. **(check) What does `n - c < k` imply for the estimator?** *Sketch:* any size-`k` subset contains a correct sample, so `U = 1`.
3. **(derive) Variance at the extremes.** Derive `Var(U)` for `k = 1` and `k = n` and show the first is `p(1-p)/n`. *Sketch:* `U = C/n` and `U = 1[C >= 1]`.
4. **(derive) Benchmark-level interval.** If pass@k is averaged over `M` problems with per-problem success probabilities `p_j`, show that the variance of the average has two parts: within-problem estimator variance (shrinks with `n`) and between-problem variance of `pass@k(p_j)` (shrinks only with `M`), and conclude which to increase when the budget is fixed. *Sketch:* law of total variance; between-problem term is `Var_j(1-(1-p_j)^k) / M`, independent of `n`.
5. **(measure) Convention flips.** Pick a third task (for example `google/boolq` or `allenai/openbookqa`), predict which convention gives the post-trained model its largest gain over the base, then measure with paired intervals.
6. **(measure) Batch-size sensitivity.** Measure the fraction of ARC predictions that change between batch size 1 and 32 in BF16 and in FP32, with a bootstrap interval, and relate it to the distribution of score margins.
7. **(open) A scoring audit.** Take a published leaderboard score for Qwen3-1.7B-Base on one task, reproduce it with your harness, and explain each remaining difference (prompt, few-shot, normalization, extraction, batch size), adding the cheapest experiment that tests each.

#### Further reading

- Papers: [HumanEval/Codex](https://arxiv.org/abs/2107.03374) (pass@k and its estimator); [Lessons from the Trenches on Reproducible Evaluation](https://arxiv.org/abs/2405.14782) (harness design choices); new: [Leveraging Large Language Models for Multiple Choice Question Answering](https://arxiv.org/abs/2210.12353) (cloze vs letter); [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) (multi-metric reporting and calibration).
- Tutorials and posts: Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/); new: Hugging Face, [What's going on with the Open LLM Leaderboard?](https://huggingface.co/blog/open-llm-leaderboard-mmlu) (the same MMLU, three implementations, three rankings); CS336 evaluation lectures.
- Code: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness); [LightEval](https://github.com/huggingface/lighteval) (practical scorers); [human-eval](https://github.com/openai/human-eval) (execution protocol).

#### Risks and open decisions

- **Ranking flip may not occur.** The section's claim is conditional; fallback is to show the convention dependence of margins and of confidence (reliability) instead, and to add a pair from the `Qwen3.5-0.8B-Base`/`Qwen3-0.6B-Base` set; the notebook text is written after the grid is run.
- **Harness version.** `acc_norm`'s length definition and the continuation-splitting rule are read from the pinned harness version; the fixture is regenerated if the version changes.
- **Sandbox.** `run_python_tests` runs untrusted code on the build host: the build must run this cell in a container; fallback is to commit execution outcomes only and mark the cell `RERUN=False` by default.
- **Time.** Generation pools may not fit 15 minutes; the committed pools make the analysis cells T0.
- **Dataset licenses.** HellaSwag's license is read from the Hub card by the author; ARC is CC BY-SA 4.0; wikitext is CC BY-SA 3.0 and GFDL (the notebook does not redistribute text).

### 1.5 Evaluation suites, judges and error bars

- **Label:** `sec_lm_eval_suites` · **Tier:** T1 (reduced T0 config: the bootstrap, paired-variance and Bradley–Terry cells on committed per-item artifacts, no model; flag `RERUN=False`) · **Budget:** T1 about 15–30 min wall-clock for the judging cells (9B judge, 3,600 calls, single forward each; to be measured) with `RERUN=True`; T0 about 2 min from committed artifacts; peak GPU memory about 20 GB, a whole 24 GB GPU (Qwen3.5-9B BF16 text-only weights 17.9 GB plus a batch of 1,000-token prompts) · **Codex units:** Codex 15.3 (evaluation suite), 15.5 (human and model judges), 15.6 (statistical comparisons)
- **Builds on (book):**
  - `sec_mdl-statistics`: estimators, confidence intervals, the bootstrap with its validity conditions and the cluster caveat, the paired sign-flip permutation test. Reused: the percentile-interval definition and the sign-flip test as an independent check; not re-taught. The book defers "dependent or clustered observations" to a block or cluster bootstrap: this section is where that is built.
  - `sec_regularized`: Bradley–Terry likelihood for preferences. Reused: the model `P(i beats j) = sigmoid(s_i - s_j)`; this section fits it to judge outputs.
  - `sec_what-attention-computes`: the evidence-level ladder `tab_induction-evidence`, reused in the judge-validation paragraph (agreement with humans is behavioral evidence that the judge is useful; it is not evidence about quality).
- **Builds on (this part):** 1.1 (`d2l.load_hf`), 1.2 (`d2l.generate`, `d2l.Generation`), 1.3 (`d2l.fewshot_prompt`), 1.4 (`d2l.score_choices`, `d2l.format_mc`, `d2l.pass_at_k`, `d2l.extract_answer`, `d2l.run_python_tests`, `ScoreComponents`, the committed GSM8K sample pool).
- **Exports (`#@save`):**
  - `d2l.BootstrapResult` (`typing.NamedTuple`): `mean` (float), `ci` (`(lo, hi)`), `se` (float), `p_two_sided` (float or `None`), `n` (items), `n_clusters` (int), `n_boot` (int).
  - `d2l.bootstrap_ci(scores, n_boot=10000, cluster=None, *, seed=0, alpha=0.05) -> d2l.BootstrapResult`: percentile interval for the mean of `scores` (1-D array of floats, binary scores allowed). With `cluster=None` items are resampled with replacement; with `cluster` (array of hashable labels, same length) whole clusters are resampled and the statistic is the ratio estimator `sum of scores / sum of cluster sizes` over the resampled clusters (so unequal clusters are handled). Implementation: draw `n_boot x G` multinomial counts, then two matrix products; no Python loop over replicates. `p_two_sided` is `None` for a single sample. Raises on NaN, on fewer than 2 items (or 2 clusters), on length mismatch.
  - `d2l.paired_bootstrap(scores_a, scores_b, n_boot=10000, cluster=None, *, seed=0, alpha=0.05) -> d2l.BootstrapResult`: `bootstrap_ci(scores_a - scores_b)` with `mean` the mean paired difference and `p_two_sided = min(1, 2 min(P*(delta* <= 0), P*(delta* >= 0)))` over the bootstrap replicates (a bootstrap-based, not a permutation, p-value: the docstring says so and points to the sign-flip test of `sec_mdl-statistics` for the latter). `scores_a` and `scores_b` must be item-aligned; the function cannot check alignment, so the caller (`EvalResult.compare`) aligns by item id. If all differences are 0 the interval is `(0, 0)`, `se` 0 and `p_two_sided` 1.0.
  - `d2l.EvalTask` (dataclass) and `d2l.EvalSuite`: the harness.
    - `EvalTask`: `id`, `dataset`, `config`, `split`, `revision`, `kind` in `{'mc', 'gen', 'ppl'}`, `n` and `subset_seed` (seeded subsample without replacement of the split; `n=None` is all), `shots`, `fmt`/`norm` (for `mc`), `max_new_tokens`, `sampling` (dict passed to `d2l.generate`), `extract` (name of the extractor), `cluster` (name of the dataset column used as cluster key, or `None`), and `since` (optional date filter on a metadata column, for time-split benchmarks).
    - `EvalSuite(tasks, name='suite', batch_size=8, *, n=None, seed=0)`: `tasks` is a list of `d2l.EvalTask` objects or of task-id strings from `d2l.EVAL_TASKS` (strings are resolved through the registry); `n` overrides each task's subsample size (`EvalTask.n`) and `seed` overrides `subset_seed`, e.g. `d2l.EvalSuite(['arc_easy', 'piqa'], n=500, seed=0).run(model, tok)`; `EvalSuite.from_manifest(path_or_name)` reads `data/suites/<name>.json`; `suite.hash` is the sha256 of the canonical JSON of the manifest plus the sha256 of each task's sorted item-id list (so a changed dataset revision or subset changes the hash).
    - `suite.run(model, tokenizer, *, seed=0, cache_dir=None, rerun=False) -> d2l.EvalResult`. `model` is any object `d2l.score_choices` and `d2l.generate` accept. Per-item results are cached on disk under `cache_dir / "<model id>@<revision>" / "<suite hash>.parquet"` and the run resumes from the cache; sampling seeds are derived per item as `hash((seed, task id, item id))` so adding items never changes existing items' samples. The result's `meta` records model id, revision, dtype, batch size, library versions, suite hash, seed, wall-clock.
    - `EvalResult.per_item -> pandas.DataFrame` with columns `task, item_id, cluster, score` (`score` in `{0.0, 1.0}` for `mc` and `gen`, or the bits-per-byte of the item for `ppl`); `.scores(task=None)` (the per-item scores as an `np.ndarray` ordered by `item_id` for the given task, or a `dict[str, np.ndarray]` over all tasks when `task` is None; this is what `d2l.paired_bootstrap` consumes); `.summary(n_boot=2000) -> DataFrame` (task, mean, ci_lo, ci_hi, n); `.macro(n_boot=10000)` (mean of task means with a bootstrap stratified by task: items resampled within each task independently, then averaged); `.compare(other, *, n_boot=10000, seed=0) -> dict[str, BootstrapResult]` (per task and `'macro'`), aligning on `(task, item_id)`; it raises `ValueError` if the suite hashes differ or the item sets are not identical and lists the first mismatches; `.save(path)` and `EvalResult.load(path)`.
    - Built-in tasks (registry `d2l.EVAL_TASKS`): `arc_challenge`, `hellaswag`, `mmlu` (5-shot, letter format, subject as cluster key: reported as an example, not a claim that items within a subject are correlated), `mmlu_pro` (5-shot, letter format with up to 10 options), `gsm8k` (4-shot, greedy, `extract='gsm8k'`), `humaneval` (execution, greedy), plus four zero-shot multiple-choice tasks for small models (`kind='mc'`, `shots=0`, `norm='token'`; table `tab_lm_eval_zeroshot`, dataset IDs checked against the Hub API on 28 Oct 2026): `arc_easy` (`allenai/ai2_arc`, config `ARC-Easy`, split `test`, 2,376 items; CC-BY-SA-4.0), `piqa` (`baber/piqa` parquet mirror, split `validation`, 1,838 items; the original `ybisk/piqa` is a loading-script dataset with licence `unknown`, and the mirror records no licence on the Hub; the upstream PIQA README states the Academic Free License v3.0 (https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/README.md, read 29 Oct 2026), which the notebook cites as the upstream statement, not as the mirror's; downloaded at run time, no item text committed), `sciq` (`allenai/sciq`, default config, split `test`, 1,000 items; CC-BY-NC-3.0: non-commercial, so the notebook downloads it at run time only and never commits item text; per-item scores in `lm-eval-scores.parquet` are fine), `winogrande` (`allenai/winogrande`, config `winogrande_xl`, split `validation`, 1,267 items, two-option partial scoring of the sentence continuation after the blank; no licence tag on the Hub; the upstream README says the codebase is Apache-2.0 and the dataset CC-BY (https://raw.githubusercontent.com/allenai/winogrande/master/README.md, read 29 Oct 2026), which the notebook cites as the upstream statement; downloaded at run time, no item text committed). Item counts are the published split sizes, to be confirmed when the author pins `revision`. Manifests committed: `data/suites/dev_v1.json` (ARC-Challenge 500, HellaSwag 500, GSM8K 200: fast and held fixed) and `data/suites/report_v1.json` (MMLU-Pro 1,000-item seeded subset, MMLU 2,000, GSM8K full 1,319).
  - `d2l.JudgeResult` (`typing.NamedTuple`): `pref_a` (float in (0, 1): position-debiased probability that answer A is better), `delta` (float, logit units, `pref_a = sigmoid(delta)`), `position_bias` (float, logit units: positive favors the first slot), `consistent` (bool), `verdict` (`'a'`, `'b'` or `'tie'`; `'tie'` when the two orders disagree), `logits` (`(l1, l2)`).
  - `d2l.judge_pairwise(judge, tokenizer, question, answer_a, answer_b, *, rubric=JUDGE_RUBRIC_V1, batch_size=8) -> d2l.JudgeResult | list[d2l.JudgeResult]`. Each argument may also be a list of equal length (batched, returns a list). Procedure per item: build the judge prompt twice, once with `answer_a` shown first (order 1) and once with `answer_b` first (order 2), through the chat template with `enable_thinking=False`; the verdict is read as a single-token choice: the log-odds `l = logit("A") - logit("B")` at the first assistant position, where `"A"` and `"B"` are the first tokens of the strings `A` and `B` (asserted distinct, and each a single token in isolation). Let `l1` = log-odds that the first slot wins in order 1 (`answer_a` first) and `l2` the same in order 2 (`answer_b` first). Under the additive model of D1.5c, `delta = (l1 - l2) / 2`, `position_bias = (l1 + l2) / 2`, `consistent = (l1 > 0) == (l2 < 0)`, `verdict = 'a' if consistent and delta > 0 else 'b' if consistent and delta < 0 else 'tie'`. `JUDGE_RUBRIC_V1` is a committed string (neutral, asks which answer is more helpful and correct, forbids reasoning, asks for one letter) so that judge prompts are part of the committed artifact. Edge cases: identical answers give `delta = 0.0` exactly; answers longer than 1,500 tokens are truncated with a warning naming the item; swapping `answer_a` and `answer_b` negates `delta` and leaves `position_bias` unchanged (test).
  - `d2l.bradley_terry(winners, losers, n_items, *, weights=None, l2=1e-3, n_iter=200) -> numpy.ndarray`: maximum-likelihood (L2-regularized) Bradley–Terry strengths `s` (length `n_items`, mean zero, natural-log scale) from index arrays of winning and losing items; ties are expressed as two half-weighted records through `weights`. Solved by the minorization-maximization iteration of Hunter (2004) (or Newton on the logistic form: the notebook states which). The L2 term makes the solution unique when the comparison graph is not strongly connected.
- **Models and data:**
  - Judge: `Qwen/Qwen3.5-9B` (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Apache-2.0) in bf16, text-only and non-thinking (the part's judge, `00_conventions.md`). Second judge (optional, exercise): `Qwen/Qwen3-4B` (`1cfa9a7208912126459214e8b04321603b3df60c`), post-trained, non-thinking, kept as the older-family contrast for the self-preference check.
  - Answerers: the four checkpoints of 1.4 in their post-trained forms where they exist: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` (non-thinking), temperature 0.7, top-p 0.8, `max_new_tokens=256`, one answer per prompt, seed 0 (the Qwen3.5 card's non-thinking settings; the same settings for all four so that sampling is not a confound).
  - Prompts: `HuggingFaceH4/mt_bench_prompts` (80, Apache-2.0; first turn only) plus 220 seeded instructions from `databricks/databricks-dolly-15k` (CC BY-SA 3.0; categories `open_qa`, `general_qa`, `brainstorming`, `creative_writing`, instructions with empty context only): 300 prompts.
  - Judge validation: `lmsys/mt_bench_human_judgments` (CC BY-4.0), split `human`, turn 1, pairs with `winner` in `{model_a, model_b}` (ties dropped and counted), features `question_id, model_a, model_b, winner, judge, conversation_a, conversation_b, turn`; 3,000 pairs maximum, a seeded subset.
  - Benchmarks for the error-bar experiments: those of 1.4 plus `TIGER-Lab/MMLU-Pro` (MIT, test split, 12,032 items; 1,000 seeded) and `cais/mmlu` (MIT, `all`, test, 14,042 items; 2,000 seeded). Named but not run: GPQA (`Idavidrein/gpqa`, CC BY 4.0, gated: needs accepted terms and a token), Humanity's Last Exam (`cais/hle`, MIT, gated), LiveCodeBench (`livecodebench/code_generation_lite`), LiveBench (monthly releases: the notebook cannot pin a moving benchmark and says so).
- **Packages:** `pandas`, `pyarrow`, `numpy`, `scipy` (for the analytic t-interval used as a test), `datasets`, `transformers >= 5.2.0`.

#### Narrative arc

1. **The question.** Opening cell: two checkpoints scored on a 500-item benchmark (the dev suite's ARC-Challenge part, cloze-char). Print the two accuracies, their separate 95% intervals (they overlap), the item-level correlation of correctness, and the interval of the paired difference (it excludes zero, or not). The measured symptom: unpaired intervals overlapping while the paired interval is much narrower, because most items are easy or hard for both models. "Is the difference real, and what is the right error bar?"
2. **Suites: development vs reporting** (`subsec_lm_suites`).
   - A development suite is small, fast, fixed and drawn from failures you care about; a reporting suite is larger, external and touched rarely. Two manifests are committed and printed. Hash and revision pinning make "the same suite" checkable (`suite.hash`).
   - Saturation and discriminative power: for the four checkpoints, on MMLU (2,000 items) and MMLU-Pro (1,000 items), both 5-shot letter format, define `power = (max mean - min mean over checkpoints) / median half-width of the item-bootstrap interval`; the table reports it per benchmark (a quantity defined here; the values are to be measured). The qualitative claim from the MMLU-Pro paper (arXiv 2406.01574) is that the harder benchmark spreads models out more and is less sensitive to prompt variation; the notebook checks the first part on these four checkpoints only.
   - Current reporting suites in one table (`tab_lm_reporting_suites`): MMLU-Pro, GPQA, Humanity's Last Exam, LiveCodeBench, LiveBench: what each measures, size, access and cost to run, whether it is time-split. Agent benchmarks are 10.5. Contamination-limited evaluation: time-split sets (an item's release date vs a model's data cutoff, which Qwen3 and Qwen3.5 do not state in their cards or the Qwen3 report: the notebook says "cutoff not published" instead of assuming one), private holdouts and perturbation-based variants; `EvalTask.since` is shown on LiveCodeBench metadata (no model run).
3. **Error bars** (`subsec_lm_errorbars`).
   - The unit of resampling: an item. Implement `bootstrap_ci`; check it against the analytic t-interval on Gaussian data and against the Wilson-style normal interval on binary data (tests).
   - Paired comparisons: `paired_bootstrap` on item-aligned scores. Derivation D1.5a. Cell: paired vs unpaired interval widths for all six checkpoint pairs on ARC-Challenge and on HellaSwag; the ratio of widths against `sqrt(1 - rho)` from the item correlation `rho`.
   - Clusters: when items share a passage, a problem or a prompt, they are not independent. Derivation D1.5b. Cell: the GSM8K sample pool of 1.4 (8 samples per problem): per-sample iid bootstrap vs problem-cluster bootstrap; the ratio of interval widths is the empirical `sqrt(DEFF)`, compared with `1 + (m - 1) r` using the within-problem correlation `r` from the pool.
   - Minimum detectable difference as a function of `n`: iid, paired (with the measured disagreement rate) and clustered; `fig_lm_mde`.
   - Agreement with the book's sign-flip permutation test on the same pairs (they are different procedures answering nearly the same question; reported side by side).
4. **LLM judges** (`subsec_lm_judges`).
   - Setup: the judge prompt (printed), single-letter verdict read from logits in both orders, `JudgeResult` fields.
   - Position bias: over the 300 x 6 pairs of checkpoint answers, the distribution of `position_bias` (histogram), the fraction of position-inconsistent verdicts, and how many pairs would flip winner if only one order were used.
   - Verbosity bias: for each pair, regress `delta` on `log(len(answer_a)/len(answer_b))` (words); report the slope with a bootstrap interval over prompts; compute a length-adjusted win rate by holding the length term at 0 (the idea of length-controlled AlpacaEval, arXiv 2404.04475, in one logistic regression; not their estimator).
   - Self-preference: the judge is Qwen3.5-9B and two of the four answerers are Qwen3.5-family: the notebook shows the Qwen3 vs Qwen3.5 pairs and states what cannot be concluded from one judge (self-preference needs a judge outside the family or a second judge; exercise 6). The literature (arXiv 2404.13076) is cited for the effect.
   - Validation against humans: on the MT-Bench human judgments, the judge's `verdict` on non-tie pairs vs the human `winner`: agreement over consistent verdicts, coverage (fraction of pairs with a consistent verdict), and agreement when forced by the sign of `delta`; Cohen's kappa; bootstrap intervals over question ids (cluster = `question_id`, because several pairs share a question). Compared with the human-human agreement reported for MT-Bench in arXiv 2306.05685 (the value is read from the paper by the author; not stated here).
5. **Rankings from pairwise judgments** (`subsec_lm_rankings`). Turn the 6 x 300 verdicts into a Bradley–Terry ranking of four checkpoints with `bradley_terry`; bootstrap over prompts (resample prompts, refit) for 95% intervals of the strengths and of the pairwise probabilities that checkpoint `i` ranks above `j`; compare with the raw win-rate ranking. Mention that Chatbot Arena (arXiv 2403.04132) fits this model to human votes at scale. The ties/inconsistencies policy: two half-weighted records.
6. **Which differences survive?** (`subsec_lm_survive`) The summary table: every checkpoint pair x {ARC, HellaSwag, MMLU, MMLU-Pro, GSM8K dev, judge win rate}: paired difference, 95% interval, verdict "distinguishable / not distinguishable", and the minimum detectable difference at that `n`.

#### Derivations

**D1.5a: Variance of a paired difference** (`eq_lm_paired_var`).
- *Statement.* Let `a_i, b_i` be the scores of two models on item `i` (items iid draws), `delta = mean(a) - mean(b)`. Then `Var(delta) = (sigma_a^2 + sigma_b^2 - 2 rho sigma_a sigma_b) / n` with `rho` the correlation of `a_i` and `b_i`; the unpaired (independent-samples) formula omits the covariance. For equal variances the paired variance is `(1 - rho)` times the unpaired. For binary scores, with `d_10` = probability that `a` is right and `b` wrong and `d_01` the reverse, `Var(delta) = (d_10 + d_01 - (d_10 - d_01)^2) / n`: only the disagreements matter.
- *Assumptions.* Items are iid samples from the benchmark distribution; scores are per item; no clusters (D1.5b removes this).
- *Steps.* (1) `delta = mean(a_i - b_i)`; the `a_i - b_i` are iid. (2) `Var(a_i - b_i) = Var(a) + Var(b) - 2 Cov(a, b)`. (3) Divide by `n`. (4) Unpaired: `Cov` term dropped: this is the variance of the difference of means of independent samples, wrong when the same items are used. (5) Binary case: `a_i - b_i` takes values `{1, 0, -1}` with probabilities `d_10, 1 - d_10 - d_01, d_01`; its mean is `d_10 - d_01` and its second moment `d_10 + d_01`; variance is the second moment minus the squared mean. (6) This is McNemar's setting: the test depends on the discordant pairs only. (7) Illustration with stated inputs (not measured): `n = 500`, accuracies 0.55 and 0.52, discordance 0.12: half-widths `1.96 sqrt(...)` of 3.0 points (paired) vs 6.2 points (unpaired).
- *Check cell (CPU, seed 0).* Simulate item difficulty `b_i ~ N(0, 1)`, success probability `sigmoid(theta_m + b_i)` for two models (`theta_a - theta_b` small); 20,000 simulated benchmarks of `n = 500`: the empirical variance of `delta` matches formula (2)-(3) using the sample `sigma`s and `rho` to 3 sigma of Monte Carlo error; the empirical coverage of the paired-bootstrap interval is within `1.5/sqrt(2000)` of 0.95 over 2,000 of those benchmarks; the unpaired interval over-covers.

**D1.5b: Clustered standard errors** (`eq_lm_cluster_se`).
- *Statement.* With `G` clusters of equal size `m` and intra-cluster correlation `r`, the variance of the overall mean of `n = G m` items is `Var = (sigma^2 / n) [1 + (m - 1) r]`; `DEFF = 1 + (m - 1) r` is the design effect. The cluster bootstrap resamples clusters and estimates the same quantity without knowing `r`; for unequal sizes the estimator is the ratio `sum y / sum m_g`.
- *Assumptions.* Clusters are independent; within a cluster items are exchangeable with common correlation `r`; `G` is large enough for the bootstrap (a few dozen).
- *Steps.* (1) Mean of cluster means: `Var(mean) = (1/G^2) sum Var(sum_g / m)`. (2) For a cluster, `Var(sum of m items) = m sigma^2 + m(m-1) r sigma^2`. (3) Hence `Var(cluster mean) = (sigma^2 / m)[1 + (m - 1) r]`, and `Var(mean of G) = (sigma^2 / (G m))[1 + (m - 1) r]`. (4) The iid formula gives `sigma^2 / n`: it understates the variance by the factor `DEFF`. (5) Resampling clusters reproduces the variance of cluster means; resampling items does not. (6) Repeated samples of the same prompt are clusters: with 8 samples per problem and `r` near 0.5, `DEFF = 4.5` and intervals are `2.1` times too narrow if samples are treated as independent.
- *Check cell (CPU, seed 0).* Synthetic: `G = 50`, `m = 10`, `r = 1/3` (so `DEFF = 4`): the cluster-bootstrap interval width is 2.0 times the iid-bootstrap width within 10%; also compute `r` by one-way ANOVA estimator on the real GSM8K pool and report `DEFF` and the width ratio observed.

**D1.5c: Debiasing position in a judge and Bradley–Terry** (`eq_lm_judge_bias`, `eq_lm_bt`).
- *Statement.* If the judge's log-odds that the first-shown answer wins is additive, `l = beta + Delta * sign`, where `beta` is a position bias and `Delta` is the quality advantage of the answer in the first slot, then with `l1` (A first) and `l2` (B first) `Delta_A = (l1 - l2) / 2` and `beta = (l1 + l2) / 2`; the debiased preference is `sigmoid(Delta_A)`. Bradley–Terry: `P(i beats j) = sigmoid(s_i - s_j)`, so `Delta_A` estimates `s_A - s_B` per prompt, and the MLE over prompts is the BT solution.
- *Assumptions.* Additivity of position and quality in log-odds (checked, not assumed: plot `l1` against `-l2` over prompts; points on `y = x` mean no bias, a constant offset means pure position bias, slope other than 1 means the bias interacts with quality); verdict noise symmetric.
- *Steps.* (1) Order 1: first slot holds A: `l1 = beta + Delta`. (2) Order 2: first slot holds B, whose advantage is `-Delta`: `l2 = beta - Delta`. (3) Subtract and add. (4) The average of the two probabilities `(sigmoid(l1) + 1 - sigmoid(l2)) / 2` is a different (and non-additive) summary; the notebook shows both and that they rank pairs identically up to ties. (5) BT likelihood for `N` records with winners `w_k`, losers `l_k`: `sum_k log sigmoid(s_{w_k} - s_{l_k})`; strictly concave up to a constant shift in `s`; L2 regularization fixes the shift and disconnected components.
- *Check cell (CPU, seed 0).* Simulate 5,000 prompts with true `Delta ~ N(0.3, 1)` and `beta = 1.0` plus noise `N(0, 0.3^2)` in each `l`; the recovered `beta` and `Delta` match to `0.05`; simulate BT data for 4 items with strengths `(0.6, 0.2, -0.1, -0.7)` and 20,000 comparisons: `bradley_terry` recovers them (after centering) to `0.05`.

#### From-scratch implementation

Components (about 155 lines): `bootstrap_ci` and `paired_bootstrap` (30), `EvalTask`, `EvalSuite`, `EvalResult` (75), `judge_pairwise` and prompt template (35), `bradley_terry` (15).

Numerical tests:

| Test | Reference | Tolerance | Seed |
|---|---|---|---|
| `bootstrap_ci` on 500 Gaussian draws | `scipy.stats.t.interval` of the mean | each endpoint within 5% of the interval width | 0 |
| `bootstrap_ci` Monte Carlo error: two runs with different seeds | itself | endpoint difference below 3% of the width at `n_boot = 10000` (the notebook prints the observed) | 0, 1 |
| `paired_bootstrap` p-value vs sign-flip permutation p-value on the dev pairs | `sec_mdl-statistics` procedure, `B = 10000` | within `3 sqrt(p(1-p)/10000) * sqrt(2)` | 0 |
| `paired_bootstrap` with `cluster=` on `G = 50, m = 10, r = 1/3` | iid bootstrap | width ratio 2.0 within 10% | 0 |
| `paired_bootstrap` on identical vectors | exact | `mean 0, ci (0, 0), p 1.0` | none |
| `judge_pairwise` swap symmetry | itself | `delta(A,B) = -delta(B,A)` to `1e-6` (FP32), `position_bias` equal | none |
| `judge_pairwise` identical answers | exact | `delta == 0.0` | none |
| `bradley_terry` on simulated data | known strengths | `0.05` | 0 |
| `EvalSuite` resume | a run interrupted after half the items equals an uninterrupted run | exact per-item scores | 0 |
| `EvalSuite.compare` with different hashes | raises | n/a | none |

#### In production

- `lm-evaluation-harness` and `lighteval` for suites (task registries, caching); `scipy.stats.bootstrap` (BCa) as the library counterpart of `bootstrap_ci`; the Chatbot Arena and `alpaca_eval` code bases for judges and Bradley–Terry rankings. The from-scratch code is the oracle for what an interval means; the harness's own reported standard errors (`acc_stderr`) are compared against `bootstrap_ci` on the same items in one cell (the harness's analytic stderr for accuracy is the binomial one; it ignores clustering).

#### Headline experiment

- **Hypothesis.** (a) For checkpoint pairs on the same items, the paired interval is narrower than the unpaired one by roughly `sqrt(1 - rho)`, so that some differences invisible to overlapping unpaired intervals are distinguishable when paired; (b) treating the 8 samples of a GSM8K problem as independent understates the interval by `sqrt(DEFF)`; (c) the judge has a measurable position bias and a nonzero fraction of position-inconsistent verdicts, and ignoring order flips the winner of some pairs; (d) a subset of the six checkpoint pairs has judge-based differences whose bootstrap intervals exclude 0.
- **Setup.** Four checkpoints; ARC (500 and full), HellaSwag 500, MMLU 2,000, MMLU-Pro 1,000, GSM8K dev 200 plus the 8-sample pool; 300 prompts x four answerers; judge Qwen3.5-9B; MT-Bench human judgments for validation; all BF16, batch size 8, bootstrap `n_boot = 10000`, `seed = 0`.
- **Procedure.** Run (or load) per-item scores; compute unpaired and paired intervals for every pair; compute cluster intervals for the sample pool; run the judge in both orders over 6 x 300 answer pairs; compute position bias, consistency, verbosity slope, human agreement; fit Bradley–Terry with prompt-level bootstrap.
- **Metrics.** Interval widths and their ratios; fraction of pairs distinguishable under each method; `position_bias` distribution; consistency rate; human agreement (with coverage); BT strengths with intervals.
- **Plots.** `fig_lm_paired_vs_unpaired`: x = the six checkpoint pairs, y = accuracy difference (points) on ARC 500, two error bars per pair (unpaired normal-approximation, paired bootstrap), horizontal zero line. `fig_lm_mde`: x = `n` (100 to 5,000, log), y = half-width of the 95% interval (percentage points); series: iid unpaired, paired with the measured disagreement rate, clustered with the measured `DEFF`. `fig_lm_judge_bias`: scatter of `l1` against `-l2` per prompt with the `y = x` line and the fitted line, and a histogram of `position_bias`. `fig_lm_bt`: BT strengths with 95% intervals for the four checkpoints, next to raw win rates.
- **Expected outcome.** (a)–(d) as directions, with `sqrt(1 - rho)` and `sqrt(DEFF)` as the quantitative predictions checked against the measured ratios; the literature reports position and verbosity biases of LLM judges (arXiv 2306.05685, 2305.17926) and self-preference (arXiv 2404.13076); how large they are for a 4B Qwen judge is to be measured. The MMLU-Pro paper's claim that it separates models better than MMLU is tested only on these four checkpoints.
- **Noise check.** This section is the noise check: bootstrap Monte Carlo error is reported (two seeds), judge noise is exposed by the two-order design, and answer sampling noise by repeating the answer generation of one checkpoint pair with a second seed (a second pass of 300 answers for the pair `Qwen3-1.7B` vs `Qwen3.5-2B` only: if the judge win rate moves by more than its interval between the two answer seeds, that is stated).
- **T1 reduction.** Prompts 300 to 150 and the second judge omitted; MMLU 2,000 to 1,000.
- **Artifacts committed.** `lm-eval-scores.parquet` (per-item scores for all suites, about 10 MB), `lm-judge-pairs.jsonl` (prompt id, answerer pair, `l1`, `l2`, answer lengths; no answer text beyond ids plus a hash, to keep licenses simple; the answers themselves in `lm-answers.jsonl` under the datasets' licenses: Dolly CC BY-SA 3.0 and MT-Bench Apache-2.0 prompts), `lm-judge-human.jsonl` (per-pair judge logits and the human label), `data/suites/dev_v1.json`, `report_v1.json`.

#### Figures and tables

- `fig_lm_paired_vs_unpaired`, `fig_lm_mde`, `fig_lm_judge_bias`, `fig_lm_bt` (above); `tab_lm_reporting_suites` (the five reporting suites and their properties); `tab_lm_suite_power` (discriminative power per benchmark); `tab_lm_judge_validation` (agreement with human labels with coverage and intervals); `tab_lm_survive` (the final which-differences-survive table); `fig_lm_design_effect` (iid vs cluster intervals on the GSM8K pool).

#### Pitfalls and misconceptions

- Non-overlapping or overlapping unpaired intervals as a test of a paired difference.
- Resampling tokens, samples or sub-questions as if independent; clusters need cluster resampling.
- Reading a bootstrap p-value as a permutation p-value, or reporting 10,000 replicates as 10,000 observations.
- Judges: a single order; a rubric that rewards length; trusting agreement with humans on easy pairs; using a judge from the same family as one of the answerers without saying so.
- Position-consistent verdicts discard information: the coverage must be reported with the agreement.
- Many pairwise comparisons: six pairs per benchmark give multiple comparisons; the notebook reports uncorrected intervals and says how a Bonferroni-style widening would change the survival table (exercise).
- A benchmark with `n` items cannot separate differences smaller than its minimum detectable difference; "no significant difference" is not "no difference".
- Moving benchmarks (LiveBench) are not pinned by a revision hash: record the release.

#### Summary (draft bullets)

- A development suite and a reporting suite serve different purposes; both are pinned by manifest hash and revision.
- Report differences between models as paired differences with a bootstrap interval; pairing removes the item-difficulty variance, and the width follows `sqrt(1 - rho)`.
- Items that share a prompt, problem or passage are clustered; resample clusters, and expect an inflation of `sqrt(1 + (m - 1) r)`.
- A pairwise LLM judge must be run in both orders; the position bias and the inconsistency rate are measured, and agreement with humans is reported with coverage.
- Bradley–Terry turns pairwise verdicts into a ranking with intervals; only differences whose intervals exclude zero are reported as differences.

#### Exercises

1. **(check) Paired or not?** A leaderboard reports two models' accuracies with separate 95% intervals that overlap. State a necessary and a sufficient condition (in terms of the item correlation) under which the paired interval nevertheless excludes zero. *Sketch:* the paired half-width scales with `sqrt(1 - rho)`; with large `rho` the difference can be significant while the marginal intervals overlap; nothing in the marginal intervals is sufficient.
2. **(check) Cluster or item?** For each of: MMLU questions grouped by subject; five generated answers per prompt; ARC questions; reading-comprehension questions about one passage: say whether the cluster bootstrap is needed and what the cluster key is. *Sketch:* answers per prompt: yes (prompt); passage questions: yes (passage); ARC: no; MMLU subjects: not needed for the mean over questions, needed if the target is generalization to new subjects.
3. **(derive) McNemar form.** Derive `Var(delta)` for binary paired scores in terms of discordant probabilities, and the sample size needed for a 95% interval of half-width `w` given discordance `d`. *Sketch:* `n = 1.96^2 (d - delta^2) / w^2`, about `1.96^2 d / w^2` for small `delta`.
4. **(derive) Judge debiasing.** Derive `delta = (l1 - l2)/2`, and show that if the true model is `l = beta + c * Delta * sign` with `c != 1` (the judge compresses quality differences) the debiased `delta` is `c * Delta`, so BT strengths are compressed but the ranking is unchanged. *Sketch:* same subtraction; ranking depends only on the sign structure.
5. **(measure) Power vs n.** Subsample ARC-Challenge to `n` in {100, 200, 400, 800, 1172} for the pair with the smallest paired difference, repeat 200 times, and plot the fraction of subsamples in which the paired interval excludes zero; compare with the minimum detectable difference formula.
6. **(measure) Second judge.** Judge the same pairs with `Qwen/Qwen3-4B`; report position bias, consistency, human agreement and the BT ranking for each judge; does either judge favor its own family's answers (paired difference of win rates, with an interval over prompts)?
7. **(open) A suite for your task.** Pick a narrow capability, build a 100-item development suite from real failures plus a held-out reporting subset, write its manifest, and report two models with paired intervals; explain which decisions the suite can and cannot support.

#### Further reading

- Papers: [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (paired differences, clustered standard errors, power: the source of D1.5a–b); [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (judge biases, agreement with humans); [Chatbot Arena](https://arxiv.org/abs/2403.04132) (Bradley–Terry rankings from votes); [MMLU-Pro](https://arxiv.org/abs/2406.01574); [GPQA](https://arxiv.org/abs/2311.12022); [Humanity's Last Exam](https://arxiv.org/abs/2501.14249); [LiveBench](https://arxiv.org/abs/2406.19314); [Dror et al., Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/). New: [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926) (position bias); [LLM Evaluators Recognize and Favor Their Own Generations](https://arxiv.org/abs/2404.13076) (self-preference); [Length-Controlled AlpacaEval](https://arxiv.org/abs/2404.04475) (verbosity); [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) (suite design).
- Tutorials: Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/); the NeurIPS 2024 and ICML 2024 evaluation tutorials; CS336 evaluation lectures.
- Code: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (suites and standard errors); [LightEval](https://github.com/huggingface/lighteval); [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) (judges and length control).

#### Risks and open decisions

- **Judge capacity.** The 9B judge may be poorly calibrated on pairs of 0.6B-versus-1.7B answers. Fallback: report the human-agreement table first and restrict the ranking claims to what the validation supports; or use a larger judge through an API (`Qwen/Qwen3.8-27B` or `openai/gpt-oss-120b`, 1.7, exercise 6), never a 4-bit load (`00_conventions.md`).
- **Gated datasets.** GPQA and HLE need accepted terms and tokens; they are named and not run. Fallback: MMLU-Pro and ARC-Challenge carry the reporting-suite role.
- **Licenses of generated answers.** Answers from Apache-2.0 models to CC BY-SA prompts are redistributed with attribution; if the editor prefers, ship only logits and hashes.
- **Bootstrap with small `n` or `G`.** The notebook asserts at least 30 clusters and warns below 100 items; the percentile interval is not BCa (an exercise or a one-line `scipy` comparison).
- **MMLU subject clusters.** Declared as an example key only; the item-level interval remains the default for the mean.
- **Run time.** Judging 3,600 prompts with the 9B judge may exceed 15 minutes; fallback is 150 prompts and the committed judgments.

### 1.6 Interpretability

- **Label:** `sec_lm_interpretability` · **Tier:** T1 (reduced T0 config: R1 `Qwen/Qwen3-0.6B` on CPU for the patching, probing and from-scratch SAE cells with 5 prompts pairs and 100k cached activation vectors; released-SAE and steering cells need T1; flag `RETRAIN=False` loads committed lens, probe and SAE artifacts) · **Budget:** T1 about 15 min wall-clock with committed artifacts and a few minutes more per `RETRAIN=True` stage (tuned lens, SAE: to be measured); peak GPU memory about 4.5 GB (Qwen3-1.7B-Base BF16 3.4 GB, one released SAE layer in FP32 0.54 GB, a batch of activations) · **Codex units:** new (no Codex unit)
- **Builds on (book):**
  - `sec_what-attention-computes`: the residual stream and its update `eq_stream-update`, QK and OV circuits, induction heads, the warning that attention maps locate candidates, and `tab_induction-evidence` (the evidence ladder). Reused: this section extends the ladder with the interventions the book leaves as an exercise (`ex_what-attention-computes-circuit-ablation`). Circuits are not re-taught.
  - `sec_mdl-svd-low-rank`: low-rank structure; used for the PCA view of the diff-of-means and SAE directions (a single cell), not re-derived.
  - `sec_transformer-block`: RMSNorm, pre-norm residual arrangement (the final norm is what a lens must apply).
  - `sec_mdl-statistics`: bootstrap; from 1.5 the cluster bootstrap is used (cluster = document or sentence).
- **Builds on (this part):** 1.2 (`d2l.load_qwen3`, the `d2l.GPT` port with hooks, `d2l.generate`); 1.4 (`d2l.bits_per_byte`-style cross-entropy for the perplexity cost); 1.5 (`d2l.bootstrap_ci` with `cluster=`); 1.1 (`d2l.safetensors_header`, to inspect the SAE repo's file layout without downloading 537 MB files).
- **Exports (`#@save`):**
  - `d2l.intervene(model, sites, *, record=()) -> contextmanager`: instruments a `d2l.GPT`. `sites` maps `(kind, layer)` to a function `fn(x) -> x_new` (or `None` for record-only); `kind` in `'resid_post'` (output of block `layer`), `'resid_mid'` (residual after attention: the input of `norm2`), `'attn_out'` (output of `blk.attention` before the residual add), `'mlp_out'` (output of `blk.ffn`), `'head_z'` (input of `W_o`, shape `(B, T, H*head_dim)`; `fn` receives the tensor and may modify heads by slicing). `record` lists sites to store in `ctx.cache[(kind, layer)]` (detached, original dtype). Hooks are removed on exit, including on exceptions; nesting is allowed. `fn` must return a tensor of the input's shape and dtype. The context object also exposes `ctx.cache` and `ctx.grad_cache` when `record_grad=True` (keeps the graph for attribution patching).
  - `d2l.logit_lens(model, hidden, layer=None) -> torch.Tensor`: applies the model's final norm and tied output head to a residual stream `hidden` `(…, d)` and returns logits `(…, V)` in FP32 (so that `logit_lens(model, h_L)` equals the model's logits exactly, up to dtype).
  - `d2l.TunedLens(d_model, num_layers)`: one translator `h + A_l h + b_l` per layer, `A_l` initialized to zero (so training starts at the logit lens); `.forward(hidden, layer)` returns the translated stream; `.logits(model, hidden, layer)` composes with `d2l.logit_lens`; `.fit(model, token_batches, steps, lr, ...)` minimizes `KL(p_final || p_lens)` per layer by streaming batches through the frozen model; `state_dict` saved to `DATA_HUB['lm-tuned-lens-qwen3-0.6b']`.
  - `d2l.TopKSAE(d_model, d_sae, k)`: `nn.Module` with `W_enc (d_sae, d_model)`, `b_enc`, `W_dec (d_model, d_sae)`, `b_dec` (the layout and names of Qwen-Scope's files); `encode(x)`: `pre = x @ W_enc.T + b_enc`, `z = relu(topk(pre, k))` (`z` has at most `k` nonzeros; ties by index); `decode(z) = z @ W_dec.T + b_dec`; `forward(x) -> (x_hat, z)`; `loss(x) -> (mse, aux)`; `normalize_decoder_()` rescales each decoder column to unit norm in place (and the matching encoder row, preserving the function); `k=None` selects the ReLU variant with `l1(z)`. `d2l.train_sae(sae, activations, steps, lr, batch_size, ...)`: the notebook's training loop.
  - `d2l.load_qwen_scope_sae(repo_id, layer, revision=None, device=None, dtype=torch.float32) -> d2l.TopKSAE`: downloads one file `layer{layer}.sae.pt` with `huggingface_hub.hf_hub_download`, loads it with `torch.load(path, weights_only=True, map_location='cpu')`, checks keys `{W_enc, W_dec, b_enc, b_dec}` and shapes against the repo's `config.json` (`d_model`, `d_sae`, `k`), and returns a `TopKSAE` with `k` from the config. Edge cases: `layer` outside `config.json['layers']` raises; the README's encoder subtracts nothing before `W_enc` (no `b_dec` pre-subtraction): the loader keeps that convention and documents it; a mismatch between `d_model` and the model raises when used with `sae_metrics`.
  - `d2l.sae_metrics(sae, model, token_batches, layer) -> dict`: over held-out token batches, excluding position 0 and any position whose activation L2 norm exceeds `10 x` the batch median (the first-token outliers that the Qwen-Scope report says it filtered): `l0` (mean nonzeros of `z`), `fvu` (`sum ||x - x_hat||^2 / sum ||x - mean(x)||^2`), `ce_orig`, `ce_splice` (cross-entropy of the model with block `layer`'s output replaced by `x_hat`), `ce_mean_ablate` (replaced by the mean activation), `loss_recovered = (ce_mean_ablate - ce_splice) / (ce_mean_ablate - ce_orig)`, `dead_frac` (features that never fire over the batches), `density` (per-feature firing frequency array).
- **Models and data:**
  - Models: R1 `Qwen/Qwen3-0.6B` (`c1899de289a04d12100db370d81485cdf75e47ca`, 28 layers, `d = 1024`) for lenses, probes and patching; `Qwen/Qwen3-1.7B-Base` (`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, 28 layers, `d = 2048`) for the SAE part, loaded through the same `d2l.load_qwen3`.
  - SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` (revision `ce1a79d9c5163932d65c417380e53230e1086370`). Layout read through the Hub API on 28 Oct 2026: `config.json` (`model_type: topk_sae`, `base_model: Qwen/Qwen3-1.7B-Base`, `d_model: 2048`, `d_sae: 32768`, `k: 50`, `num_layers: 28`, `layers: [0..27]`, `hook_point: resid_post`, `dtype: float32`), `README.md`, `LICENSE`, `app.py` (a Gradio demo), and 28 files `layer0.sae.pt` to `layer27.sae.pt`, each 537,012,593 or 537,012,603 bytes: a `dict` of four FP32 tensors `W_enc (32768, 2048)`, `W_dec (2048, 32768)`, `b_enc (32768,)`, `b_dec (2048,)`. The sibling `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100` (revision `2756a49850...`, `k = 100`) provides a second released point. Hook point `resid_post` of layer `n` is the output of decoder block `n` (0-indexed), which is `blks[n]`'s output in `d2l.GPT`; the demo's steering adds `strength * W_dec[:, feature]` to that output through a forward hook (all positions, or the listed ones). Qwen-Scope's own description (arXiv 2605.11887, section 2.2): TopK SAEs trained per layer on in-house pretraining data, auxiliary dead-feature loss with weight 1/32, activations with extremely large L2 norm filtered, outliers most often at the first token for Qwen3-1.7B; the report gives no reconstruction numbers, so the notebook measures them. Related released groups the notebook mentions: Qwen3.5-2B-Base (`Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`, a hybrid: exercise 7), Qwen3-8B-Base.
  - License (read 29 Oct 2026): the card is tagged `license: other` (`license_name: qwen`, `license_link` to the repo's `LICENSE`), the `LICENSE` file is Apache-2.0 text, and the card's "Caution" section prohibits using interpretability tools for non-scientific purposes to interfere with model capabilities or to produce harmful content, with authorization terminated for violators. The notebook downloads the SAE at run time only, never commits or redistributes weights, prints the card's Caution section verbatim from the downloaded `README.md` before the first use, and states that the steering experiment is scientific research on a base model. The author reads the card again at authoring time. Fallback below (own SAEs) unchanged. See risks.
  - Text data: `HuggingFaceFW/fineweb-edu` config `sample-10BT` (ODC-By), first 6,000 documents truncated to 256 tokens for training activations (about 1.5M tokens) and the next 800 as held out; `wikimedia/wikipedia` config `20231101.zh` (CC BY-SA 3.0 and GFDL), 300 documents truncated to 128 tokens, for the language contrast; `openai/openai_humaneval` canonical solutions (MIT) for the code contrast.
  - Probing data: Universal Dependencies English EWT (`UniversalDependencies/UD_English-EWT` on GitHub, files `en_ewt-ud-{train,dev,test}.conllu`, license CC BY-SA 4.0, release pinned by commit at authoring time), UPOS tags (17 classes), a seeded 20,000-word training subset and the full test split's first 5,000 words; word-to-token alignment through the tokenizer's offset mapping, probing the last sub-token of each word.
  - Patching data: 40 committed `(clean, corrupted, correct, distractor)` prompt pairs, e.g. `"The Eiffel Tower is located in the city of"` / `"The Colosseum is located in the city of"` with `" Paris"` and `" Rome"`; both prompts tokenize to the same length; built from a committed list of landmark and city facts; pairs where the model's clean top-1 is not the correct answer are discarded and counted.
- **Packages:** `torch >= 2.5`, `scikit-learn` (the probe baseline), `datasets`, `huggingface_hub`, `matplotlib`. Optional oracle: `transformer_lens` (the author checks Qwen3 support at the pinned version; if absent, the patching oracle is the Transformers model with hooks).

#### Narrative arc

1. **The question.** Opening cell: on R1, the prompt `"The Eiffel Tower is located in the city of"`. Print the final top-5 next tokens, then the top-1 token the logit lens reads out at every 4th layer: early layers print junk or the input token's neighbors, and the answer appears only late. The symptom: "the model knows the answer at the end; where did it come from, and how would we know a story is true?" Then the evidence ladder of `tab_induction-evidence`, extended (`tab_lm_interp_evidence`, below): reading, probing, intervening.
2. **Reading the residual stream: logit lens and tuned lens** (`subsec_lm_lenses`).
   - Setup with `d2l.intervene` recording `resid_post` of each block; `d2l.logit_lens`. Derivation D1.6a: exact decomposition of the final logits into per-component contributions, and what the lens does to it.
   - Logit lens results: for the 40 patching prompts plus 200 held-out FineWeb-Edu documents: (i) rank of the final answer token by layer (factual prompts), (ii) `KL(p_final || p_lens_l)` and top-1 agreement with the final prediction by layer. Identity check at the last layer.
   - Tuned lens: train `d2l.TunedLens` on 1M streamed tokens of FineWeb-Edu (25% of positions per sequence sampled for the loss); curves of `KL` by layer for both lenses on 100k held-out tokens; cluster bootstrap over documents.
   - Limits: the lens at layer `l` is a statement about a linear readout, not about whether later layers use the information (so, behavioral/activation evidence only).
3. **Probes and control tasks** (`subsec_lm_probes`).
   - UPOS from the last sub-token's `resid_post` at each of 14 layers (even indices); multinomial logistic regression with `L2` regularization chosen on the dev split (`C` from `{0.01, 0.1, 1, 10}`).
   - Control task (Hewitt and Liang): each word type in the training vocabulary gets a random UPOS drawn from the empirical tag distribution, fixed per type; the probe is trained and tested on it; **selectivity = task accuracy minus control accuracy**. Probe accuracy alone cannot separate "the representation encodes POS" from "the probe memorizes word types". The cell also reports the trivial baseline (most frequent tag per word type from training) and a probe on the embedding layer (layer 0), which already scores well because most words have one tag.
   - Plot: accuracy and selectivity by layer, with sentence-cluster bootstrap intervals. Quantitative check: selectivity at each layer with its interval; claims are made only for layers whose interval excludes the baseline.
4. **Activation patching and attribution patching** (`subsec_lm_patching`).
   - Method: run clean and corrupted prompts, cache clean activations at a site, rerun the corrupted prompt with that site overwritten by the clean value, measure the effect on the logit difference `LD = logit(correct) - logit(distractor)`; normalized effect `(LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)`. Derivation D1.6b, including the first-order approximation (attribution patching).
   - Sweep: all 28 layers x token positions (subject's last token, the final token, the other positions pooled) for `resid_post`; then per-layer `attn_out` and `mlp_out` at the subject's last token. 40 pairs; mean effect map with bootstrap interval over pairs.
   - Sanity checks: patching every position at the last layer gives effect 1 (up to numerics); patching nothing gives 0; patching a random other prompt's activations at the same site gives an effect near 0 (a control for "any activation change moves LD").
   - Attribution patching: one forward and one backward pass over the pairs produce the AtP score for every `(layer, position)`; Spearman correlation with the exact map; wall-clock ratio.
   - Induction heads in R1 (tie to the book's lab): head-level mean-ablation through the `'head_z'` site on repeated random sequences; the 10 heads whose ablation raises second-copy loss most, vs the 10 heads with the highest prefix-matching score from the attention pattern (the book's measure); overlap reported; claim limited to what the ablation shows. This is the component-level row of `tab_induction-evidence` that the book left open.
5. **Sparse autoencoders** (`subsec_lm_sae`).
   - Objective and sparsity: derivation D1.6c (L1-relaxed sparse coding; why the decoder norm must be bounded; top-k as the L0-constrained alternative; shrinkage).
   - Train small SAEs on Qwen3-1.7B-Base layer 14 (`resid_post` of `blks[14]`), cached activations from 1.5M FineWeb-Edu tokens (position 0 and norm outliers excluded; stored as FP16, 1.5M x 2048 x 2 B = 6.1 GB, in host memory or a memmap), width 8,192, both variants: ReLU+L1 with `lambda` in a grid, and TopK with `k` in {8, 16, 32, 64, 128}; 3 seeds at each setting for the T0-sized R1 version, 1 seed at T1 for the 1.7B version plus a 3-seed repeat for `k = 32` only; a 32,768-wide TopK `k = 50` SAE trained on the same cache for the like-for-like comparison with the release.
   - Released SAE: load layer 14 (537 MB) and, as an instrument check, compute reconstruction before using any feature: `fvu`, `ce_splice`, `loss_recovered`, `dead_frac`, with the README's encoder convention and, if the numbers are implausible, the alternative `b_dec`-subtracted convention (the cell reports both; the one that reconstructs is kept and named).
   - Frontier plot: `FVU` against `L0` (and `loss_recovered` against `L0`) for our L1, our TopK, the released `k = 50` and `k = 100` points. The width mismatch (8K vs 32K) and the training-data mismatch (1.5M tokens vs the release's) are stated in the caption; the 32K point reduces the first.
   - Features and labels: for the released SAE, top-activating contexts for 10 features chosen by a contrastive rule (mean activation difference between code and prose text; AUC) and 5 by density rank; label by reading; then **test the label**: define the label as a rule on held-out tokens (for the code feature, "token is inside a Python program line"), report precision, recall and F1 of "feature active" against the rule, with bootstrap intervals over documents; then a causal check: clamp the feature to 0 in the spliced forward and measure the change in loss on the tokens it fires on versus on random tokens. Pitfalls of labeling (confirmation bias, dense features, polysemanticity, absorption) shown on one dense feature (`density` histogram).
   - Gemma Scope as the related suite (arXiv 2408.05147) and Neuronpedia as a dashboard host, in text.
6. **Steering** (`subsec_lm_steering`).
   - Three directions at layer 14 of Qwen3-1.7B-Base, for the concept "write Chinese": (a) the released SAE's decoder column of the feature chosen by contrast (mean activation difference between Chinese Wikipedia and English FineWeb-Edu text; the best-separating feature by AUC), (b) the diff-of-means vector (mean `resid_post` on Chinese text minus mean on English; contrastive activation addition, arXiv 2312.06681; the same idea as representation engineering, arXiv 2310.01405), (c) a random direction of equal norm (control, 5 seeds). Applied as `h <- h + alpha d / ||d||` at layer 14 at all positions (prompt and generated tokens), through `d2l.intervene`.
   - Behavior metric: fraction of generated characters in the CJK Unified Ideographs block (U+4E00 to U+9FFF), measured on 48 greedy new tokens for 100 English prompts (the first 100 held-out FineWeb-Edu document beginnings, 32 tokens each); a prompt counts as "switched" if that fraction is at least 0.3 (the threshold is a convention stated in the cell; the curve is also reported for 0.1 and 0.5). Strength grid: `alpha` in `{0, 0.25, 0.5, 1, 2, 4, 8} x median ||h||` at layer 14.
   - Cost metric: increase in cross-entropy (nats per token) of the steered model on held-out English text, teacher-forced, with the same intervention active (perplexity is `exp` of it).
   - Frontier: switched fraction against cost for the three directions; the notebook reports whether the SAE feature beats the diff-of-means at matched cost (AxBench, arXiv 2501.17148, found simple baselines competitive with or better than SAEs for steering on their benchmark; this is a small, single-concept test and the text says so).
   - Side effects: sample outputs at the chosen operating point for 5 prompts, including a prompt whose correct continuation is a factual completion: does the steered model still complete it? A second concept is an exercise.
7. **Attribution graphs, in outline** (`subsec_lm_circuits`). One page: transcoders replace MLPs by sparse feature layers, a replacement model reproduces outputs, an attribution graph records which features drive which; intervention experiments validate the graph; where it breaks (reconstruction error nodes, attention). The notebook does not run it; pointer to the circuit-tracing method paper page and to released graphs. `fig_lm_attribution_outline` is a drawn diagram (author-made).
8. **What each method can claim** (`subsec_lm_evidence`): fill `tab_lm_interp_evidence`.

#### Derivations

**D1.6a: The residual stream as an exact sum, and what the lens approximates** (`eq_lm_dla`, `eq_lm_lens`).
- *Statement.* For a pre-norm decoder with final RMSNorm weight `gamma` and tied unembedding `E` (rows are token vectors), the final logits at a position are `z = (1 / rho_L) E diag(gamma) (h_0 + sum_{l=1}^{L} Delta_l)` with `Delta_l` the output of block `l` (attention plus MLP) added to the stream and `rho_L = sqrt(mean(h_L^2) + eps)`. So the contribution of component `j` to the logit of token `t` is `c_{j,t} = e_t^T diag(gamma) Delta_j / rho_L`, and `sum_j c_{j,t} = z_t` exactly. The logit lens at layer `l` is `z^{(l)} = (1 / rho_l) E diag(gamma) h_l`: the partial sum `h_0 + sum_{j <= l} Delta_j` with the scale of the stream at layer `l`, not `L`.
- *Assumptions.* RMSNorm with no bias; the logit lens applies the final norm to `h_l`.
- *Steps.* (1) Unroll `h_L = h_0 + sum_l Delta_l` (`eq_stream-update`). (2) Substitute into the final norm: `RMSNorm(h_L) = diag(gamma) h_L / rho_L`. (3) The unembedding is linear: `z = E diag(gamma) h_L / rho_L`, and linearity in `h_L` gives the sum. (4) The logit lens at `l` replaces `h_L` by `h_l` and `rho_L` by `rho_l`; it equals the partial sum up to the factor `rho_L / rho_l`. (5) The error of the lens relative to the final logits is the contribution of blocks `l+1..L` plus the rescaling; the tuned lens learns the linear map `h_l -> h_l + A_l h_l + b_l` that best predicts the final distribution in KL. (6) Because the later blocks are nonlinear functions of `h_l`, no affine map is exact; the residual error of the tuned lens is the part of the later computation that is not linear in `h_l`.
- *Check cell.* On R1 and 8 prompts: `sum_j c_{j,t}` over `h_0`, `L` attention outputs and `L` MLP outputs equals the logit of the correct token to `1e-3` in FP32 (the notebook records the error); the lens at layer `L` equals the model's logits; the ratio `rho_L / rho_l` is printed per layer, showing why the logit lens at early layers is poorly scaled (to be measured).

**D1.6b: Activation patching as a causal-effect estimate, and attribution patching** (`eq_lm_patch`, `eq_lm_atp`).
- *Statement.* Fix a metric `m` (the logit difference). For a site `s` with clean value `a_s^c` and corrupted value `a_s^x`, define `PE(s) = m(run on x with a_s := a_s^c) - m(run on x)`, an intervention effect on the corrupted run; the normalized effect divides by `m(clean) - m(corrupted)`. A first-order expansion in the site activation gives `AtP(s) = (a_s^c - a_s^x)^T grad_{a_s} m |_{corrupted run}`; the error is `(1/2) Delta^T H Delta` plus higher order, where `Delta = a_s^c - a_s^x`.
- *Assumptions.* Clean and corrupted prompts have the same length and token alignment; the site is overwritten, not added to; `m` is differentiable in `a_s` with the rest of the computation as a function of `a_s` (the downstream path from the site, in the corrupted run).
- *Steps.* (1) Let `g(a) = m(run on x with a_s := a)`, a scalar function of the site's activation. (2) `PE(s) = g(a^c) - g(a^x)`. (3) Taylor: `g(a^c) = g(a^x) + grad g(a^x)^T Delta + R`. (4) The gradient is one backward pass of `m` through the corrupted run, giving `grad g` for every site at once; the clean activations are cached by one forward pass; so `AtP` for all `L x T x ...` sites costs about 3 passes, while exact patching costs one forward pass per site. (5) `R` vanishes when `g` is affine in `a`: sites close to the output (the last layer's residual at the final position) are nearly affine in the logit, so AtP is accurate there; for earlier sites the downstream nonlinearity (softmax attention saturation, MLP) makes `R` large. (6) Normalized effect sums: patching all positions at the output layer gives 1 by construction.
- *Check cell.* On the 40 pairs: Spearman and Pearson correlation between `AtP` and exact `PE` over all `(layer, position)` cells of `resid_post`, and the mean absolute error by layer (to be measured; direction: error rises toward early layers); wall-clock of both. Unit test: for a toy 2-layer MLP with a known `g`, AtP equals exact PE for affine `g` and differs by the second-order term for a quadratic `g` (CPU, closed forms).

**D1.6c: The sparse-autoencoder objective, decoder norms, and top-k** (`eq_lm_sae`, `eq_lm_sae_shrink`).
- *Statement.* The L1 SAE solves `min_{W, z(x)} (1/2) ||x - W z||^2 + lambda ||z||_1`, `z >= 0`, `||w_j|| <= 1`. (i) Without the norm bound the objective has no minimizer: scaling `w_j -> c w_j`, `z_j -> z_j / c` leaves the reconstruction unchanged and divides the penalty by `c`. (ii) For an orthonormal dictionary the minimizer is the soft threshold `z_j = max(y_j - lambda, 0)` with `y_j = w_j^T x`: active features are shrunk by exactly `lambda`. (iii) The top-k SAE minimizes `||x - W z||^2` subject to `||z||_0 <= k`; for an orthonormal dictionary the minimizer keeps the `k` largest `y_j` unshrunk: `z_j = y_j` on the support.
- *Assumptions.* Nonnegative activations (ReLU), a fixed dictionary for (ii) and (iii) (the statement is about the encoder problem; learning `W` adds the alternating problem), `||w_j|| = 1` in (ii).
- *Steps.* (i) For one feature, reconstruction depends only on the product `w_j z_j`; with penalty `lambda z_j` and `||w_j||` free, set `w_j -> c w_j`, `z_j -> z_j / c`: the penalty becomes `lambda z_j / c`, which tends to 0 as `c` grows, so the infimum is not attained; hence `||w_j|| <= 1` (or the norm is fixed to 1 after each step). (ii) With `W^T W = I`: `||x - W z||^2 = ||x||^2 - 2 y^T z + ||z||^2` (when `x` lies in the span; plus a constant otherwise): separable in `z_j`: minimize `(1/2) z_j^2 - y_j z_j + lambda z_j` over `z_j >= 0`: derivative `z_j - y_j + lambda = 0`, so `z_j = max(y_j - lambda, 0)`. (iii) Same separable form with the cardinality constraint: choosing the support `S` reduces the objective by `(1/2) sum_{j in S} y_j^2` (taking `z_j = y_j`), maximized by the `k` largest `y_j`. (iv) Consequences: L1 biases activations down (feature magnitudes are not comparable to true coefficients, and the reconstruction error includes the shrinkage `lambda`), top-k does not, and `L0` is set directly rather than through `lambda`; the cost is that `k` is the same for every input.
- *Check cell (CPU).* A random orthonormal dictionary `W` (`512 x 2048` via QR, seed 0), 10,000 synthetic `x = W z* + noise` with `||z*||_0 = 10` and nonzero values uniform in `[1, 2]`: the L1 encoder's active coefficients are lower than the true by `lambda` (mean difference matches `lambda` within Monte Carlo error), while top-10 recovers them with zero mean bias; reconstruction MSE of both is printed for matched `L0`; a second cell demonstrates (i): train a one-feature autoencoder with and without the norm constraint and show the decoder norm growing without bound in the unconstrained case.

#### From-scratch implementation

Components (about 165 lines; the text may replace the probe and the SAE loop by library calls to stay near 150): `intervene` (30), `logit_lens` and `TunedLens` (30), probe training (15), `TopKSAE` with L1 variant (35), `train_sae` (20), `load_qwen_scope_sae` (10), `sae_metrics` (15), the steering hook (10).

Numerical tests:

| Test | Reference | Tolerance | Seed |
|---|---|---|---|
| `intervene` record-only leaves logits unchanged | no hook | exact (bitwise) | none |
| `intervene` with identity `fn` at every site | no hook | exact | none |
| `resid_post` recorded at block `l` equals Transformers' `hidden_states[l+1]` for `l < L-1` (attention `eager`) | Transformers hooks on `model.model.layers[l]` | the 1.2 FP32 tolerance | none |
| `logit_lens(model, h_L)` vs model logits | forward | FP32 `1e-4` | none |
| D1.6a decomposition sums | logits | `1e-3` FP32 | none |
| `TunedLens` at initialization equals the logit lens | `d2l.logit_lens` | exact | none |
| `TopKSAE` `encode` for `k = 50` has at most 50 nonzeros per row; matches the demo's `topk_relu` (`app.py`: ReLU, then top-k) on a released layer, and the README snippet (top-k then scatter) except at positions with fewer than `k` positive pre-activations, which the cell counts | repo code paths | exact | none |
| Released SAE round trip on 256 random activations of the correct scale | `fvu < 1` and `fvu` equal to the same quantity computed by a separate NumPy reimplementation | `1e-5` relative | 0 |
| `normalize_decoder_` leaves `decode(encode(x))` unchanged | before | `1e-5` | 0 |
| Patching sanity: full-layer last-position patch gives effect 1; null patch gives 0 | by construction | `1e-3` | none |
| D1.6b toy AtP | closed form | exact for affine, matches second-order term for quadratic | none |
| D1.6c orthonormal-dictionary shrinkage | `lambda` | Monte Carlo `3 sigma` | 0 |

#### In production

- TransformerLens (`HookedTransformer`, activation cache, patching helpers) and ARENA's notebooks are the standard teaching and research tools; the notebook runs the layer-14 logit-lens ranking and one patching cell with TransformerLens as an oracle if the pinned version supports Qwen3, and otherwise with Transformers hooks. SAELens (training, evaluation, loading released SAEs) and the tuned-lens package are the production counterparts of `TopKSAE`/`train_sae` and `TunedLens`; Neuronpedia hosts feature dashboards (pointer). The from-scratch `d2l.intervene` on `d2l.GPT` (1.2) is the oracle for hooks on the Transformers model, the reverse of the usual direction.

#### Headline experiment

- **Hypothesis.** A single released SAE feature, used as a steering direction at layer 14 of Qwen3-1.7B-Base, shifts generation toward Chinese at a perplexity cost on English text that lies above the cost of a diff-of-means vector at equal behavior change or about equal to it, and both beat a random direction of the same norm by a wide margin. (A prediction, with AxBench as the prior that baselines compete with SAEs; the outcome is to be measured and may go either way.)
- **Setup.** Qwen3-1.7B-Base, BF16 via `d2l.load_qwen3`; layer 14; the released SAE (`W32K-L0_50`); contrast data of 300 Chinese Wikipedia and 300 English FineWeb-Edu documents (128 tokens) for choosing the feature and the diff-of-means vector, a disjoint set of 100 English prompts for the behavior measurement, and 800 held-out English documents for the cost measurement; greedy decoding of 48 tokens; strength grid `alpha` in `{0, 0.25, 0.5, 1, 2, 4, 8}` x median residual norm at layer 14.
- **Procedure.** (1) Choose the feature by the AUC of its activation separating Chinese from English tokens on the contrast set (report the top 5 and the chosen one's AUC). (2) Compute the diff-of-means vector on the same set. (3) For each direction and strength run the 100 prompts and compute the switched fraction (threshold 0.3) and the mean CJK character fraction; (4) compute the teacher-forced cross-entropy on the held-out set with the intervention active; (5) repeat the random direction with 5 seeds.
- **Metrics.** Switched fraction (with a bootstrap interval over prompts); `delta CE` in nats per token (with a document-cluster bootstrap interval); the area under the frontier (switched fraction against `delta CE` up to a cost cap of 0.5 nats: a cap chosen as a convention) per direction.
- **Plot (`fig_lm_steering_frontier`).** x = `delta CE` (nats per token on English text), y = switched fraction; three curves (SAE feature, diff-of-means, random direction mean with min-max band over 5 seeds), points labeled by `alpha`; horizontal dashed lines at 0.5 and 0.9 switched fraction; vertical dashed line at the cost cap.
- **Expected outcome.** Random direction: no switch at any cost below the cap (direction); both informed directions switch language at some strength; whether the SAE feature is cheaper than diff-of-means: unknown, to be measured. Magnitudes are not predicted. Qwen-Scope reports successful language steering with SAE features (arXiv 2605.11887, section 3); its report does not measure a perplexity cost, which is what this experiment adds.
- **Noise check.** Prompt-level bootstrap intervals on switched fraction; random-direction seeds; the claim "A beats B at matched cost" needs the area difference to exceed the larger of the two bootstrap intervals (area computed per bootstrap replicate); decoding is greedy, so there is no sampling noise; the 100 prompts are a limited sample and the notebook says what intervals they afford.
- **T1 reduction.** Already T1; `SMALL=True` uses Qwen3-0.6B with its own small SAE (trained in the notebook, `RETRAIN=True`) and the diff-of-means vector, without the released SAE.
- **Artifacts committed.** Tuned-lens translators (`DATA_HUB['lm-tuned-lens-qwen3-0.6b']`, about 57 MB in FP16), probe weights (small), our small SAEs for R1 and for Qwen3-1.7B-Base layer 14 in FP16 (`lm-sae-r1-l14`, about 34 MB for width 8,192 on `d = 1024`; `lm-sae-q17b-l14`, about 134 MB for width 8,192 on `d = 2048`), per-cell patching effect tables, steering results as CSV; none of the released SAE.

#### Figures and tables

- `fig_lm_lens_rank` (rank of the correct answer by layer, logit vs tuned lens); `fig_lm_lens_kl` (KL to the final distribution by layer, both lenses, with document-cluster bands).
- `fig_lm_probe_selectivity` (probe accuracy, control accuracy and selectivity by layer; baseline line).
- `fig_lm_patching_map` (heatmap: layers x positions, normalized effect; second panel `attn_out`/`mlp_out` at the subject's last token); `fig_lm_atp_vs_exact` (scatter of AtP against exact effect with the correlation); `tab_lm_induction_heads` (top heads by ablation vs by pattern score).
- `fig_lm_sae_frontier` (FVU and loss recovered against L0 for L1, TopK, released points); `tab_lm_sae_release` (per-layer FVU for the layers fetched, if the sweep is enabled); `fig_lm_feature_density` (histogram of firing frequencies, the dense feature highlighted); `tab_lm_feature_labels` (feature, label, rule, precision, recall, F1, clamp effect).
- `fig_lm_steering_frontier` (headline); `tab_lm_steering_samples` (sample generations at the operating point).
- `tab_lm_interp_evidence`: rows: logit lens; tuned lens; linear probe; probe with control task; head ablation; activation patching; attribution patching; SAE reconstruction; SAE feature label; feature clamping; steering; attribution graph. Columns: evidence level (behavioral, activation, weight-level, input intervention, component intervention, as in `tab_induction-evidence`), conclusion supported, what it cannot show.
- `fig_lm_attribution_outline` (diagram, author-made).

#### Pitfalls and misconceptions

- A probe's accuracy is not the model's use of the property; selectivity and interventions are needed.
- The logit lens's early-layer outputs are partly a scaling artifact (D1.6a), not "what the model believes" at that layer.
- Patching with a corrupted prompt that differs in more than the intended variable; unequal token lengths; patching after a hook that changes dtype.
- Attribution patching is a gradient estimate; it fails where the downstream function is far from linear.
- SAE features are not "the" concepts of the model: reconstruction error is large, features can be dense, split or absorbed, and a label made by reading top examples is a hypothesis.
- Applying an SAE trained on one checkpoint to another: the Qwen-Scope README says SAEs trained on base models are reasonable for post-trained checkpoints; it is an assumption to check by reconstruction (FVU, loss recovered) on that checkpoint.
- First-token activation outliers (attention sinks) dominate norms and metrics; exclude position 0 and say so.
- Steering strength is not comparable across directions unless normalized; always report the perplexity cost beside the behavior change.
- Loading `.pt` checkpoints: use `weights_only=True`; pickle files can execute code.

#### Summary (draft bullets)

- The residual stream is an exact sum of component outputs; the logit lens reads a partial sum with the wrong scale, and the tuned lens learns a linear correction per layer.
- A probe measures what is linearly decodable; a control task measures what the probe could learn without the property; only interventions show use.
- Activation patching estimates a component's causal effect on a metric; attribution patching approximates all effects with one backward pass and is accurate only where the downstream map is nearly linear.
- A sparse autoencoder is dictionary learning on activations; L1 shrinks coefficients and needs bounded decoder norms, top-k does not shrink and fixes L0; released suites make SAEs available for models too large to train on in a notebook, to be validated by reconstruction before use.
- Steering adds a direction at a layer; it must be reported as behavior change against its cost in perplexity and against a random-direction control.

#### Exercises

1. **(check) Lens at the last layer.** What does `d2l.logit_lens(model, h_L)` return and why is the KL to the final distribution zero there? What about at layer 0? *Sketch:* exactly the logits, since the final norm and head are the same function; at layer 0 it is the embedding stream, a bigram-like readout dominated by the input token.
2. **(check) Control task.** A probe reaches 95% accuracy at layer 12 and 92% on the control task. What do you conclude? *Sketch:* selectivity 3 points: the probe largely memorizes word types; the accuracy is not evidence that the layer encodes POS beyond lexical identity.
3. **(derive) Soft thresholding.** Derive the L1 SAE encoder for an orthonormal dictionary and show that top-k gives `z_j = y_j` on the support. *Sketch:* D1.6c, steps (ii)–(iii).
4. **(derive) AtP error.** Show that the AtP error for a site is `(1/2) Delta^T H Delta + O(||Delta||^3)` and give a condition on the downstream computation under which AtP is exact. *Sketch:* Taylor remainder; exact if the metric is affine in the site activation (for example a linear readout of the final stream without normalization).
5. **(measure) Where does the fact live?** Predict at which layers and positions the normalized patching effect is largest for the 40 factual pairs; run it; report the layer range at which the subject's last token carries more than half of the effect, with a bootstrap interval over pairs.
6. **(measure) A second concept.** Repeat the steering frontier for a concept of your choice (formal vs informal register, Python vs prose, or English vs French); report the frontier and whether the ordering of the three directions is the same as for Chinese.
7. **(open) SAEs on a hybrid.** Use `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50` with `Qwen/Qwen3.5-2B-Base` through Transformers hooks; measure FVU and loss recovered by layer and ask whether the Gated DeltaNet layers and the full-attention layers (1.1: 3 of every 4 layers are linear attention) have different reconstruction quality; propose and test an explanation.

#### Further reading

- Lenses and probes: [the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens); [Tuned Lens](https://arxiv.org/abs/2303.08112); new: [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368).
- Patching and circuits: new: [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (causal tracing: the template for the factual-recall patching map); [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) (metrics and methods); [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348) and Neel Nanda's [attribution patching post](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) (AtP); [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).
- Sparse autoencoders: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features); [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600); [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093) (top-k SAEs, auxiliary loss); new: [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) (why features share directions).
- Released suites: [Gemma Scope](https://arxiv.org/abs/2408.05147); [Qwen-Scope](https://arxiv.org/abs/2605.11887) and its [SAE for Qwen3-1.7B-Base](https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50).
- Steering: [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681); [Representation Engineering](https://arxiv.org/abs/2310.01405); new: [AxBench](https://arxiv.org/abs/2501.17148) (steering baselines vs SAEs).
- Code and courses: [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens); [ARENA](https://github.com/ARENA-education/ARENA_materials); new: [SAELens](https://github.com/decoderesearch/SAELens) and the [tuned-lens package](https://github.com/EleutherAI/tuned-lens).

#### Risks and open decisions

- **Qwen-Scope license and caution clause.** The card says `license: other` with a Qwen license name and carries a use-restriction "Caution" paragraph; the LICENSE file is Apache-2.0 text, so the two disagree, and the notebook treats the stricter reading as binding (run-time download only, no committed weights, the Caution quoted in the notebook). Fallback if the editor judges the restriction incompatible with the book's license: use only SAEs trained in the notebook (R1 and Qwen3-1.7B-Base) and Gemma Scope (Gemma terms apply, its own check), and keep Qwen-Scope as a pointer; the headline experiment then uses our own SAE feature instead of the release.
- **Released-SAE convention.** Whether the encoder should subtract `b_dec` is not stated in the card; the notebook decides by reconstruction and prints both; if neither reconstructs, the instrument check fails and the section stops using the release (and says so).
- **Download size.** 537 MB per layer; the layer sweep (4 layers, 2.1 GB) is optional.
- **Run time.** Tuned-lens training (about 1M tokens x 27 translators) and SAE training are the longest stages; the committed artifacts make T1 fit; `RETRAIN=True` is for the author's verification.
- **TransformerLens support.** Qwen3 support at the pinned version must be checked; if missing, the oracle is Transformers hooks.
- **Hook semantics across versions.** Transformers decoder-layer outputs changed between versions (tensor vs tuple); the oracle cell handles both and asserts the last-layer relation to the final-norm output.
- **Steering on a base model.** Greedy continuations of short English prompts by a base model may be degenerate; fallback: prompts with fixed prefixes that elicit prose (for example "The history of"), which the notebook commits.
- **Probe dataset.** UD English EWT is CC BY-SA 4.0; the PTB-derived POS sets on the Hub (for example `batterydata/pos_tagging`, which lists Apache-2.0 but is derived from Penn Treebank and CoNLL-2003) are avoided for licensing clarity.

### 1.7 Models behind an API

- **Label:** `sec_lm_api` · **Tier:** T0 by default (the notebook replays committed responses; flag `API_MODE='replay'`); live calls are optional (`API_MODE='record'` reads the cache first and calls only on a miss; needs a token and a stated cost cap); the local arm of the headline experiment is T1 (flag `LIVE_SMALL=True`: 50 problems, `Qwen/Qwen3.5-4B` locally, sweep levels {1, 4, 16}, 10 repeats) · **Budget:** T0 about 2 min (hash lookups, tables and plots); live about 15–25 min wall-clock (hosted calls plus the local server's start-up and 100 sequential generations; to be measured); GPU only for the local arm: `Qwen/Qwen3.5-9B` in BF16 behind `transformers serve`, about 20 GB, a whole 24 GB GPU on the build box (no other model on that GPU while the server runs); the hosted arm needs no GPU; live spend is capped by `COST_CAP_USD` (a constant the notebook sets; to be measured, expected to be small) · **Codex units:** none (added by the editor on 2026-10-29; not in v5 or the Codex plan)
- **Builds on (book):**
  - `sec_decoding`: temperature, top-p, top-k and stop conditions. Reused as request fields of the chat protocol; not re-taught.
  - `sec_repro`: seeds. Reused to say what a seed does and does not promise when the sampler is someone else's.
  - `sec_mdl-statistics`: bootstrap, through 1.5's shared functions.
- **Builds on (this part):** 1.3 (`d2l.fewshot_prompt` with `template='chat_multi'` builds the 4-shot messages), 1.4 (`d2l.extract_answer`, the seeded 100 GSM8K problem ids and the 4 demonstrations of the sample pool), 1.5 (`d2l.paired_bootstrap`, `d2l.bootstrap_ci`, the MDE formula of `fig_lm_mde`; `d2l.judge_pairwise`, `JUDGE_RUBRIC_V1` and `lm-judge-human.jsonl` for exercise 6). The local arm needs only a running `transformers serve`; `d2l.generate` (1.2) is not used.
- **Not this section:** 13.1's `d2l.OpenAIServer` is a *load-testing* client: `/v1/completions`, token-id prompts, `ignore_eos`, aimed at a server the reader controls, results discarded after the run. `d2l.ChatAPI` is a *measurement and replay* client: `/v1/chat/completions`, text messages, a provider's real semantics, every response cached and kept. Neither imports the other. Tool calls are named once (the `tools` field) and taught in 10.1; the Responses API of `transformers serve` and vLLM is out of scope.
- **Exports (`#@save`; names checked against `00_conventions.md` and `01_interfaces.md` on 2026-10-29, no collision):**
  - `d2l.ChatResult` (dataclass): `text`, `finish_reason`, `usage` (dict with `prompt_tokens`, `completion_tokens`, `cached_tokens`, `reasoning_tokens`; missing fields are `None`, never 0), `logprobs` (list of `(token, logprob, top)` for the generated tokens, or `None`), `model` (the id the server echoes), `system_fingerprint` (or `None`), `request_hash`, `from_cache`, `t_first` (seconds to the first content chunk when streamed, else `None`), `t_total`, `itl` (list of inter-chunk gaps, streamed only), `n_retries`, `n_429`, `created` (UTC ISO time of the live call). Timings are those of the live call that filled the cache; a replayed result carries them unchanged and `from_cache=True`.
  - `d2l.ChatAPI(base_url, model, api_key=None, *, cache=None, mode='replay', price=None, cost_cap_usd=None, timeout=120.0, max_retries=6, default_params=None, http_client=None)`: wrapper over `openai.OpenAI` and `openai.AsyncOpenAI` built with `max_retries=0` (the SDK retries twice by default, `openai.DEFAULT_MAX_RETRIES == 2` in 3.41.0, which would hide retries from the counters). `cache` is a JSONL path (append-only; loaded into a dict at start). `mode`: `'replay'` (cache only; a miss raises `d2l.ReplayMiss` naming the request, so a build can never reach the network), `'record'` (cache first, live on a miss, then append), `'live'` (always call and append; for repeated measurements). `api_key` is never read from a file and never enters the hash, the cache or a log line.
    - `.chat(messages, *, replica=0, stream=False, **params) -> ChatResult` and `async .achat(...)`: `params` are request fields (`temperature, top_p, max_tokens, stop, seed, n, logprobs, top_logprobs, response_format, extra_body`); `replica` is a client-side integer that is not sent but is part of the hash, so that K identical live requests are K distinct cache entries (experiment 2). Retries: connection errors, timeouts, HTTP 408, 429, 500, 502, 503, 504, with `backoff_delay` (honouring `Retry-After` as a floor); HTTP 400, 401, 403, 404, 422 raise at once.
    - `.models() -> list[dict]` (the `/v1/models` listing), `.spent_usd` (running cost from `price` and `usage`), `.manifest() -> dict` (base-URL host, model id, SDK version, date, request, cache-hit and retry counts; written next to the cache).
    - `request_hash(base_url, model, messages, params, replica) -> str`: sha256 of the canonical JSON (sorted keys, no whitespace, UTF-8) of the host part of `base_url`, `model`, `messages`, the sampling and format `params` and `replica`. It excludes `stream`, timeouts, headers and keys: streaming changes transport and timing, not the requested distribution (exercise 1).
  - `async d2l.chat_batch(api, requests, *, concurrency=8) -> list[ChatResult | Exception]`: `requests` is a list of dicts `{'messages': ..., 'replica': 0, **params}`; an `asyncio.Semaphore(concurrency)` bounds in-flight calls, results come back in input order, a failed request is returned as its exception (so one bad item does not discard 99 paid ones). Notebooks `await` it; plain scripts wrap it in `asyncio.run`.
  - `d2l.backoff_delay(attempt, base=1.0, cap=60.0, rng=None, retry_after=None) -> float`: full jitter, `Uniform(0, min(cap, base * 2**attempt))`, never below `retry_after` when the server gave one; `rng` is a `random.Random` so tests are seeded.
  - `d2l.PricePoint` (NamedTuple `input, output, cached_input=None, per=1_000_000`) and `d2l.api_cost(usage, price) -> float` (`eq_lm_api_cost`): `[(p - c) * input + c * (cached_input or input) + o * output] / per` with `p = prompt_tokens`, `c = cached_tokens or 0`, `o = completion_tokens` (reasoning tokens are inside `o` unless the provider says otherwise: asserted from the usage block). Prices are arguments: `d2l.price_from_listing(listing, model_id, provider) -> PricePoint` reads the `pricing` fields of a `/v1/models` entry; the committed snapshot carries its retrieval date.
  - `d2l.ReplayMiss`, `d2l.CostCapExceeded` (exceptions).
- **Models and data:**
  - Hosted: `Qwen/Qwen3.5-397B-A17B` (Hub revision `8472618112abcbd45acbcdc58436aff4233c23f7`, Apache-2.0, 403,397,928,944 parameters in the safetensors index, all components included: BF16 weights alone are 806.8 GB, more than 8 × 40 GB = 320 GB and more than 8 × 80 GB = 640 GB; the name states 17B active parameters). On 29 Oct 2026 `https://router.huggingface.co/v1/models` listed it with five providers (novita, featherless-ai, scaleway, ovhcloud, deepinfra), context length 262,144 where given, prices on four of the five and `supports_structured_output` true for three. The notebook pins one provider with the documented `model:provider` suffix (for example `Qwen/Qwen3.5-397B-A17B:deepinfra`, chosen at authoring time from the listing; the default `:fastest` may switch providers between calls, which changes price, latency and possibly numerics). The Hub revision does not identify what a provider serves: the manifest records the provider, the date and the served model id.
  - Local: `Qwen/Qwen3.5-9B` (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, the part's judge model, here as the answerer) in BF16 behind `transformers serve` (Transformers 5.22.0, `pip install transformers[serving]`; endpoints `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models` read from `src/transformers/cli/serving/server.py` at tag v5.22.0; the docs mark the CLI experimental). Same family as the hosted model, so the accuracy gap is about size and not about family or template. Optional second local server: vLLM 0.32.0 in its own environment (`--language-model-only` for the text-only 9B), reached over the same URL scheme.
  - Closed provider (optional, never committed): any provider with an OpenAI-compatible endpoint; the model id is read from its own `/v1/models` at authoring time and not named here (unverified here).
  - Data: `openai/gsm8k`, config `main`, split `test`, the seeded 100 problems of 1.4 and its 4 `train` demonstrations (MIT). Exercise 6: `openai/gpt-oss-120b` (Apache-2.0, 116,829,156,672 parameters, nine providers on the 29 Oct 2026 listing), `lm-judge-human.jsonl` from 1.5.
  - `DATA_HUB`: `lm-api-gsm8k-v1`, `lm-api-determinism-v1`, `lm-api-sweep-v1`, `lm-api-judge-v1` (exercise 6, optional; the four response caches, JSONL), `lm-api-listing-v1` (JSON, the `/v1/models` rows for the models used, with retrieval date; metadata only). Outputs of Apache-2.0 models to MIT prompts are committed with attribution; a closed provider's outputs are never committed (terms, below).
- **Packages:** `openai >= 3.41.0` (PyPI latest on 28 Oct 2026, `requires_python >= 3.10`; its `CompletionUsage` has `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens`, and `ChatCompletion` has `system_fingerprint`, all read from the installed package), `httpx` (comes with it; `httpx.MockTransport` drives the tests), `transformers[serving] >= 5.22.0` (this section only; the part's floor stays 5.2.0), `numpy`, `pandas`, `matplotlib`.

#### Narrative arc

1. **The question.** Opening cell: `Qwen/Qwen3.5-397B-A17B` weights are 806.8 GB in BF16 (the safetensors total times 2 bytes, printed with the arithmetic) against the book's tiers: 24 GB (T1), 40 GB (T2), 320 GB or 640 GB for the 8-GPU node (T3). Then one `ChatAPI.chat` call to the router prints the reply, the `usage` block, the wall-clock time and the cost from `api_cost`. The measured symptom: a model no tier can load answers in about a second, for a cost printed in dollars, and the book can say nothing about its weights. "What does it take to use such a model as a measured instrument rather than a chat window?"
2. **The chat-completions protocol** (`subsec_lm_chat_protocol`).
   - The request: `messages` (roles; the provider applies the chat template, so the rendered token sequence of 1.3 is not visible), `temperature`/`top_p`, `max_tokens` (the router's schema; the SDK's newer `max_completion_tokens` is rejected by `transformers serve`), `stop` (the router's schema allows up to 4), `n`, `seed`, `logprobs`/`top_logprobs` (the router's schema allows 0–5), `response_format` with `type: json_schema` and `strict`, `stream` with `stream_options.include_usage`. Tool calls: the `tools` field exists; 10.1 teaches it (one forward reference).
   - The response: `choices[].message.content`, `finish_reason` (`stop`, `length`, `tool_calls`), and the `usage` block: prompt, completion, total, `cached_tokens`, `reasoning_tokens`. Why `usage` is the ground truth for cost and for token counts: the client never sees the provider's tokenizer.
   - Thinking models: hidden reasoning is billed as completion tokens and consumes `max_tokens`; an empty answer with `finish_reason='length'` is the signature (cell: `max_tokens=64` on a hard prompt). The notebook asks for non-thinking with `extra_body={'chat_template_kwargs': {'enable_thinking': False}}` (accepted by `transformers serve`) and checks `reasoning_tokens` on the hosted side, since providers may ignore it.
   - Probe matrix (`tab_lm_api_fields`, one cell, about 10 tiny requests per endpoint): send each field to the router and to `transformers serve` and record accepted, ignored or rejected. Source for the local column: `UNUSED_CHAT_COMPLETION_FIELDS` in `transformers` v5.22.0 makes `logprobs`, `top_logprobs`, `n`, `response_format`, `stream_options`, `max_completion_tokens`, `tool_choice` and others raise, while `seed`, `stop`, `max_tokens` and `chat_template_kwargs` are accepted. The vLLM column is filled from its docs at 0.32.0 (its `extra_body` parameters; `prompt_logprobs` is a vLLM extension: the author checks it, unverified here). The router column is measured: its `/v1/models` rows carry `supports_tools` and `supports_structured_output` per provider but no logprobs flag.
3. **One client, several endpoints** (`subsec_lm_endpoints`).
   - Three presets built from one class: the router (`https://router.huggingface.co/v1`, token with the "Inference Providers" permission; the `:fastest`, `:cheapest`, `:preferred` and `:<provider>` suffixes; billed by Hugging Face with no markup per its docs; included monthly credits depend on the plan, a free account must buy credits: the notebook says so and `API_MODE='replay'` needs none), a local `transformers serve` (`http://localhost:8000/v1`, empty `api_key`), and vLLM (same scheme, own environment, pinned in 7.7's table).
   - `tab_lm_api_models`: from the committed listing, the large open-weight rows (id, number of providers, context length, listed input and output prices, structured-output flag) for six models; a column states that prices differ by provider (for the 397B model the four listed input prices differ by about 1.6× between the cheapest and the dearest, computed in the cell).
   - `huggingface_hub.InferenceClient` is the library's own client for the same service; the notebook shows the one-line equivalent and why the OpenAI SDK is used (works unchanged against the local servers).
4. **Measuring from the client** (`subsec_lm_client_measure`).
   - Latency: time to first token (`t_first`, first non-empty content chunk), inter-chunk gaps, end-to-end. Streaming chunks are not tokens: a provider may batch several tokens per chunk, so `itl` is "per chunk" and the per-token rate is `completion_tokens / (t_total - t_first)` from `usage`.
   - Throughput under concurrency: `chat_batch` with a semaphore; closed-loop sweep (experiment 3); Little's law `N = X W` as the check and the explanation of the knee (D1.7a, `eq_lm_littles_law`).
   - Rate limits: HTTP 429 and the `Retry-After` header (RFC 6585, MDN); a limit may be requests per minute or tokens per minute: the sweep's 429 counts do not tell which. Full-jitter backoff (D1.7b, `eq_lm_backoff_wait`); retried requests may still be billed (unverified; the author checks the billing page after a forced-429 run).
   - Cost: `api_cost` on `usage`, `eq_lm_api_cost`; prices are read from the listing (its `pricing` fields carry no unit; the notebook assumes USD per million tokens, which is the unit the billing page must reconcile: the live run compares the notebook's total with the billing page's usage for the same run, to be measured). A client-side cap raises `CostCapExceeded` before the call that would cross it.
   - Prompt caching and batch APIs: cached prompt tokens appear in `usage.prompt_tokens_details.cached_tokens` where a provider supports it; one cell sends a 2,000-token shared prefix twice and reports whether the router's provider returns it (exercise 5). Provider batch APIs (OpenAI's Batch guide, Anthropic's message batches) trade latency for price and are described only; the router documents none (unverified).
5. **Reproducibility and replay** (`subsec_lm_api_repro`).
   - What can change under a fixed model id: weights (alias re-pointed), quantization or kernels at the provider, the chat template, the provider behind `:fastest`. Record `model`, provider, `system_fingerprint` when returned, the date and the SDK version in `manifest()`; the Hub revision is a statement about weights, not about what served the request.
   - Experiment 2: temperature 0 with a seed is not determinism (`seed` is best-effort in the protocol: a request field, with no promise; the book says what the provider documents and what the experiment shows). Forward reference to 13.2 (batch-dependent kernels) and Thinking Machines' account of batch variance, already a source of 1.4.
   - The replay cache: JSONL keyed by `request_hash`; why every parameter that can change the output is in the key (stale replays are silent errors) and why nothing secret is; `mode='replay'` makes the book's build offline and exact; a changed prompt is a loud `ReplayMiss`, not a silent re-call. Distinguish: a cache hit replays *one draw*; it does not make the model deterministic.
6. **What you lose** (`subsec_lm_api_lost`; `tab_lm_api_lost`).
   - Prompt log-probabilities: the chat protocol returns log-probabilities of *generated* tokens only (router schema: top 0–5 alternatives), never of the prompt or a supplied continuation, so `d2l.score_choices` (1.4) cannot run: no cloze scoring, no bits per byte, no perplexity. Evaluation falls back to generation with an extractor (`kind='gen'` in 1.5). The one surviving likelihood readout: ask for a single-letter answer with `max_tokens=1`, `logprobs=True`, `top_logprobs=5` and read the letters among the top-5, if the provider returns them (cell; measured, not assumed).
   - Weights and activations (1.6), the rendered prompt (1.3), the tokenizer (counts only through `usage`), the sampler (top-k and min-p are not in the protocol), the exact build (hence experiment 2), and control of batching.
   - What stays available: structured output (`response_format` json_schema: 10.2), many samples (`n` or repeated requests: pass@k of 1.4), pairwise judging as generation. `d2l.EvalSuite` (1.5) is not wired to `ChatAPI` here; the headline experiment scores with `extract_answer` and `paired_bootstrap` directly, and says so.
7. **Terms and data** (`subsec_lm_api_terms`; `tab_lm_api_terms`).
   - Data leave the machine: every prompt goes to the router and then to a provider whose retention and logging terms are the provider's (each provider's page is read at authoring time; unverified here). The book sends only public benchmark text; the notebook warns against sending visitor, personal or unreleased data, and against pasting keys into prompts.
   - Output-use restrictions: Anthropic's Commercial Terms (section D.4, read 29 Oct 2026) bar using the service to build a competing product, including training competing models; OpenAI's terms page refused an automated fetch and is unverified, so the author reads it. Open-weight models served by a third party carry two layers: the weights' licence (Apache-2.0 for the two used here) and the provider's service terms. Consequence for 3.5, 7.3 and 9.1: before an API model generates training data or a distillation target, the notebook checks both layers and records the answer in the manifest.
   - Redistribution: committed caches hold only outputs of Apache-2.0 open-weight models; a closed provider's outputs stay local (git-ignored) and only derived scores are committed.

#### Derivations

**D1.7a: Little's law and the throughput knee of a closed loop** (`eq_lm_littles_law`).
- *Statement.* A client keeps `c` requests in flight (closed loop, no think time). Over a long run the mean number in flight equals throughput times mean latency, `c = X W` (Little). If the endpoint serves at most `X_max` requests per second and an unloaded request takes `W_0`, then `X(c) = min(c / W_0, X_max)` and `W(c) = max(W_0, c / X_max)`; the knee is at `c* = X_max W_0`. Beyond it extra concurrency buys latency, not throughput.
- *Assumptions.* Stationary run, the same request mix at every level, a single bottleneck rate `X_max` (a provider's capacity, a rate limit or the local GPU); no 429s (otherwise `c` includes backoff sleeps and the identity still holds for requests in the system, which the notebook counts separately).
- *Steps.* (1) Little: time-average of the number in system `L = X W` for any stable system. (2) Closed loop: `L = c` at every instant, so `X = c / W`. (3) Below capacity, `W = W_0`, so `X = c / W_0`. (4) At capacity `X = X_max`, so `W = c / X_max`. (5) Equate: `c* = X_max W_0`. (6) For token throughput multiply `X` by mean completion tokens; the knee in tokens per second is the same level.
- *Check cell (CPU, seed 0).* An in-process fake server (`asyncio.sleep(W_0 = 0.2 s)` behind a semaphore of capacity `K = 8`, so `X_max = K / W_0 = 40` per second): sweep `c` in {1, 2, 4, 8, 16, 32}, 200 requests each; measured `X` within 5% of the formula at every level, and `X * W` within 2% of `c`. The same two checks are then applied to the live sweep: the identity holds, the formula's shape is read off the plot.

**D1.7b: Expected waiting under full-jitter backoff** (`eq_lm_backoff_wait`).
- *Statement.* Each attempt fails independently with probability `p` (a 429 or 5xx); after the `k`-th failure (`k = 0, 1, ...`) the client sleeps `U(0, min(cap, base 2^k))`, and gives up after `K` retries. Then the expected number of attempts is `sum_{k=0}^{K} p^k` and the expected total sleep is `E[S] = sum_{k=0}^{K-1} p^{k+1} min(cap, base 2^k) / 2`. With `K -> infinity` and `cap` large, `E[S] -> (base p / 2) / (1 - 2p)` for `p < 1/2` and diverges for `p >= 1/2`.
- *Assumptions.* Independent failures (a rate-limited server violates this: that is why jitter exists; the check below uses a correlated server too); sleeps do not themselves change `p`.
- *Steps.* (1) Retry `k+1` happens only if the first `k+1` attempts failed: probability `p^{k+1}`. (2) The sleep before it has mean `min(cap, base 2^k) / 2`. (3) Sum over `k`; attempts: `1 + p + ... + p^K`. (4) Without the cap the terms are `(base/2) p (2p)^k`, a geometric series converging iff `2p < 1`; the cap restores convergence. (5) The reason for jitter is not the mean but the variance across clients: `n` clients that failed together and sleep identical `base 2^k` retry together again, while with `U(0, .)` their retries spread over the window instead of landing on one instant, which lowers the peak arrival rate the server sees.
- *Check cell (CPU, seed 0).* Monte Carlo with `p = 0.3`, `K = 6`, `base = 1`, `cap = 60`: simulated mean total sleep within 3 sigma of the formula (200,000 draws). Correlated case: 64 clients against a fake server that returns 429 beyond 10 requests per second: attempts to finish all clients, no jitter vs full jitter (the notebook prints both; direction: fewer with jitter; magnitude to be measured).

#### From-scratch implementation

Components (about 135 lines): `request_hash` (10), `backoff_delay` (8), `PricePoint`, `api_cost`, `price_from_listing` (18), `ChatResult` (12), `ChatAPI` with cache, retry, streaming timestamps and the cost cap (65), `chat_batch` (15), `ReplayMiss`, `CostCapExceeded` (4). No network in the tests: `openai.AsyncOpenAI(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))`.

Numerical tests:

| Test | Reference | Tolerance | Seed |
|---|---|---|---|
| `request_hash` is invariant to dict key order and to `stream`, `timeout`, the API key; changes with `seed`, `max_tokens`, `model` (provider suffix), a message character and `replica` | itself | exact | none |
| `backoff_delay` over 100,000 draws: every value in `[0, min(cap, base 2^k)]`, mean `min(cap, base 2^k)/2`; `retry_after=3` gives at least 3 | closed form | mean within 3 sigma | 0 |
| Mock server: 429 (with `Retry-After: 0`) twice then 200 | 3 attempts, `n_429 == 2`, `n_retries == 2` | exact | none |
| Mock server: 400 and 401 | exactly 1 attempt, exception raised | exact | none |
| Mock server: 503 always, `max_retries=3` | exactly 4 attempts, then raises | exact | none |
| `chat_batch`: 50 requests, `concurrency=5`, handler records in-flight maximum, one request fails | maximum in flight ≤ 5; 49 results plus 1 exception, in input order | exact | none |
| Cache: second identical call in `'record'` mode makes zero HTTP calls and sets `from_cache`; `'replay'` with an unseen request raises `ReplayMiss`; the cache file contains no key string | itself | exact | none |
| Streaming: concatenated chunks equal the non-streaming `text` of the same mock; `t_first <= t_total`; `usage` taken from the final chunk | itself | exact | none |
| `api_cost` on 5 hand-computed usage dicts (cached tokens, no cached price, `None` fields) | arithmetic | `1e-12` | none |
| `CostCapExceeded` raised before the call that would cross the cap, with no HTTP call for it | itself | exact | none |
| Fake-server Little's-law sweep (D1.7a) and the Monte Carlo of D1.7b | formulas | as stated | 0 |

#### In production

- The `openai` SDK (clients, typed responses, its own retry and timeout machinery, which the book switches off to count), `huggingface_hub.InferenceClient` (the router's native client), and eval frameworks' API backends: lm-evaluation-harness `local-completions` and `local-chat-completions` model types (its API guide), `inspect-ai` and LiteLLM (unverified here) for multi-provider routing, retries and caching. The from-scratch client is the oracle for what those do: the retry-count and cache-key tests are re-run against the SDK's behaviour in one cell, and the harness's chat backend is pointed at the local `transformers serve` to reproduce a 20-item run of experiment 1 (agreement of per-item correctness, to be measured).

#### Headline experiment

- **Hypothesis.** (a) The same client, prompts and extractor measure a 397B-parameter hosted model and the local 9B model of the same family; the hosted model's accuracy on 100 GSM8K problems is at least the local model's, but at `n = 100` the paired interval may well include 0, in which case the notebook states "not distinguishable at this `n`" with the MDE of 1.5. (b) Temperature 0 with a fixed seed gives repeated hosted requests more than one distinct completion for at least one of three long prompts; the local server run alone gives one; run under concurrent load it may give more (batch-dependent numerics). (c) Closed-loop throughput rises about linearly with concurrency until a knee and TTFT then grows; `X * W = c` holds to a few percent at every level.
- **Setup.** *Experiment 1:* hosted `Qwen/Qwen3.5-397B-A17B:<provider>` and local `Qwen/Qwen3.5-9B`, the 100 problems of 1.4, 4-shot as `chat_multi` messages (printed once), `temperature=0`, `seed=0`, `max_tokens=512`, non-thinking requested, answers by `d2l.extract_answer`; hosted at concurrency 8, local at 1; the hosted run is done twice (`replica` 0 and 1). *Experiment 2:* the three problems with the longest completions in experiment 1, K = 20 `replica`s each, same settings; hosted; local alone; local with 8 background requests in flight. *Experiment 3:* hosted, streaming, `max_tokens=128`, 32 requests per level, `c` in {1, 2, 4, 8, 16}, two passes with the level order shuffled by seed 0.
- **Procedure.** Run (record) or replay; per-item correctness; `paired_bootstrap` hosted-vs-local and replica-vs-replica; `bootstrap_ci` per model; tokens, cost, e2e latency p50/p90; per prompt count distinct completions and the first character at which each differs from the modal completion; per level the TTFT p50/p90, median `itl`, aggregate completion tokens per second, 429 count, retries.
- **Metrics.** Accuracy with 95% interval; paired difference with interval; replica disagreement (number of problems whose correctness flips); cost per 100 problems (`api_cost`, from the snapshot) and its reconciliation with the billing page when live; distinct-completion counts; `X * W / c`.
- **Plots.** `fig_lm_api_gsm8k`: left, accuracy per model with its interval and the paired difference with its interval and zero line; right, end-to-end latency (p50, p90 bars) per problem for both, hosted cost annotated. `fig_lm_api_determinism`: x = replica rank by first-divergence position, y = first-divergence character index (log), one panel per endpoint condition, markers per prompt. `fig_lm_api_concurrency`: left, x = `c` (log2), y = completion tokens per second, two passes as two series, the `min(c/W_0, X_max)` curve fitted to the first pass; right, TTFT p50 and p90 against `c`, 429 counts as labels.
- **Expected outcome.** Directions as in the hypothesis; no magnitudes: accuracy, cost, distinct counts and the knee are all to be measured. If every replica is identical on every prompt the notebook reports that the claim was not supported for that endpoint at this size, and says that identical outputs on 60 requests do not show determinism.
- **Noise check.** Experiment 1: the second hosted run gives the replica noise floor, and a model gap smaller than the replica disagreement is called unresolved; the paired bootstrap over problems gives the sampling interval. Experiment 2: counts of distinct completions, no p-values. Experiment 3: two interleaved passes; their difference is the noise floor of a shared service, and a knee is claimed only if both passes show it.
- **T1 reduction.** `LIVE_SMALL=True` as in the header; replay is T0 regardless.
- **Artifacts committed.** The four caches named above, `lm-api-listing-v1`, `manifest.json` per cache (model id, provider, dates, SDK version, counts), and `lm-api-fields.csv` (the probe matrix).

#### Figures and tables

- `fig_lm_api_gsm8k`, `fig_lm_api_determinism`, `fig_lm_api_concurrency` (above); `fig_lm_api_backoff` (attempts until all 64 clients finish, with and without jitter, from the D1.7b correlated-server check).
- `tab_lm_api_fields` (field by endpoint: accepted, ignored, rejected); `tab_lm_api_models` (large open-weight models on the router, providers, context, listed prices); `tab_lm_api_lost` (what you lose, the consequence and the workaround); `tab_lm_api_terms` (data retention, output-use restriction, licence layers: one row per provider read, with the date).

#### Pitfalls and misconceptions

- A model id is not a version: aliases are re-pointed, providers re-quantize, `:fastest` changes providers between calls. Pin the provider, record the date, keep the responses.
- `temperature=0` and a seed are not determinism; a cache hit replays one draw and proves nothing about the model.
- Timing whole responses and calling it latency; counting streamed chunks as tokens; using a `max_tokens` that hidden reasoning consumes.
- Leaving the SDK's own retries on (they hide 429s from the counters and multiply requests); retrying 400/401; retrying without jitter; sleeping a fixed time after a 429 that carried `Retry-After`.
- A cache key that omits a parameter (silent stale replays) or includes the API key (a secret in a committed file).
- Comparing local and hosted models under different prompts, thinking modes or extractors, then reading the gap as capability; reading a gap at `n = 100` without 1.5's interval.
- Trusting a price table without a date; summing cost without cached-token and reasoning-token terms; treating the router's listing as a contract.
- Sending non-public data; committing outputs whose terms forbid it; training a model on an API model's outputs without reading both layers of terms.

#### Summary (draft bullets)

- Models too large for any tier here are reached over HTTP; the OpenAI-compatible chat-completions protocol is the common interface, and the same client talks to a hosted router, a local server and an optional closed provider.
- The `usage` block, not the client, is the ground truth for tokens, cached tokens, reasoning tokens and cost; prices are dated inputs, never constants.
- Latency, throughput and rate limits are measured from the client: time to first token, closed-loop concurrency with a semaphore (Little's law explains the knee), 429 with `Retry-After`, full-jitter backoff.
- Hosted results are not reproducible by default: model ids drift and temperature 0 is not deterministic; the book records the provider and date and replays a request-hash cache, so the build is offline and exact.
- An API removes prompt log-probabilities, weights and the sampler: evaluation falls back to generation with an extractor, and terms of use decide what may be sent, trained on and committed.

#### Exercises

1. **(check) Cache keys.** For each change say whether `request_hash` must change: reorder JSON keys; change `seed`; change `max_tokens`; change the API key; turn on `stream`; change `:deepinfra` to `:novita` in the model id; change `replica`. *Sketch:* no for key order, API key and `stream` (transport and timing, not the requested distribution; the price is that a replay of a streamed request returns the recorded timings); yes for the rest.
2. **(check) Retry or raise?** For HTTP 400, 401, 408, 429, 500, 503 and a connection reset, say whether to retry and what to do with `Retry-After`. *Sketch:* retry 408, 429, 500, 503 and the reset with backoff; raise on 400 and 401 (the request or key is wrong and will stay wrong); take `Retry-After` as a floor for the delay.
3. **(derive) The knee.** From `X = min(c / W_0, X_max)` and Little's law derive `W(c)` for `c > c*`, and the extra TTFT a client sees when it runs 2 `c*` requests in flight. *Sketch:* `W = c / X_max`, so at `c = 2 c*` latency doubles to `2 W_0` with no throughput gain.
4. **(derive) Equal vs full jitter.** "Equal jitter" sleeps `d/2 + U(0, d/2)` with `d = min(cap, base 2^k)`. Derive its mean sleep and compare with full jitter's. *Sketch:* mean `3d/4` against `d/2`; equal jitter never sleeps less than `d/2`, so it desynchronizes less.
5. **(measure) Prompt caching.** Send the same 2,000-token prefix with five different suffixes, then again after 5 minutes; read `usage.prompt_tokens_details.cached_tokens` each time. Report the cached fraction, whether the provider lists a cached-input price, and the saving from `api_cost`.
6. **(measure) An API judge from another family (optional, live).** Judge the pairs of 1.5 with `openai/gpt-oss-120b` through `ChatAPI`: both orders, verdict as a generated letter (`max_tokens` large enough for its reasoning, or low reasoning effort per the router's `reasoning_effort` field), the 300 answer pairs of one checkpoint pair of 1.5 (the pair of its answer-seed check) and 200 human-labelled pairs of `lm-judge-human.jsonl`, both orders (1,000 calls). Report position bias, the inconsistent-pair rate, agreement with the human labels of `lm-judge-human.jsonl` with coverage, agreement with the local Qwen3.5-9B judge (Cohen's kappa), and cost. Independence (`00_conventions.md`): this judge's family differs from all four answerers, unlike the local judge's, so the comparison bears on 1.5's self-preference exercise; state that one run does not settle it.
7. **(open) Is it the same model?** Send the same 20 prompts at temperature 0 to two providers of one open-weight model and to the local copy; quantify differences in text, `usage` counts and latency; decide what evidence would show a provider serves a different model (arXiv 2410.20247 and 2504.04715 give tests).

#### Further reading

- Papers: [Model Equality Testing: Which Model Is This API Serving?](https://arxiv.org/abs/2410.20247) (statistical tests that an endpoint serves the model it claims); [Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs](https://arxiv.org/abs/2504.04715) (substitution and quantization at providers); the RFC for HTTP 429, [RFC 6585](https://www.rfc-editor.org/rfc/rfc6585).
- Tutorials and docs: Hugging Face, [Inference Providers](https://huggingface.co/docs/inference-providers/index) (router, provider policies), its [chat-completion specification](https://huggingface.co/docs/inference-providers/tasks/chat-completion) and [pricing and billing](https://huggingface.co/docs/inference-providers/pricing); Transformers, [Serving](https://huggingface.co/docs/transformers/main/en/serving); vLLM, [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html); AWS Architecture Blog, [Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/) (the source of D1.7b's schedules); MDN, [Retry-After](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After); OpenAI, [Structured outputs](https://platform.openai.com/docs/guides/structured-outputs), [Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) and [Batch](https://platform.openai.com/docs/guides/batch); Anthropic, [Message batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing); Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (why temperature 0 is not deterministic).
- Code: [openai-python](https://github.com/openai/openai-python) (clients and types); [lm-evaluation-harness API guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md) (`local-chat-completions`); [vLLM](https://github.com/vllm-project/vllm).

#### Risks and open decisions

- **Provider churn.** The hosted model, its providers and prices change weekly. The committed caches and listing snapshot keep the build exact (`API_MODE='replay'`). Fallback if the model leaves the router before authoring ends: `Qwen/Qwen3.5-122B-A10B` (Apache-2.0, listed with two providers on 29 Oct 2026), otherwise `openai/gpt-oss-120b`.
- **Token and credits.** Live runs need a Hub token with the "Inference Providers" permission; free accounts must buy credits, PRO accounts list a monthly credit (the pricing page said $2.00 on 29 Oct 2026). Replay needs neither. The cost cap is a notebook constant: an editor decision whether the build may spend at all (the build box needs a token held outside the repository).
- **Price units and reconciliation.** The listing's `pricing` fields have no stated unit; if the billing page disagrees with `api_cost` the notebook reports the ratio and uses the billing page.
- **Logprobs and thinking switches are provider-dependent.** The probe matrix is the evidence; if the chosen provider ignores `enable_thinking`, raise `max_tokens` and report the reasoning-token share as a confound, or pick another provider.
- **`transformers serve` limits.** The CLI is documented as experimental, rejects `logprobs`, `n`, `response_format` and `stream_options`, and may load the vision tower of Qwen3.5-9B (the Hub total is 9.65 billion parameters, 19.3 GB in BF16, against the judge's 17.9 GB text-only weights): it still fits a whole 24 GB GPU, with less KV headroom (to be measured). Fallback: vLLM 0.32.0 in its own environment with `--language-model-only`, or `Qwen/Qwen3.5-4B` as the local arm.
- **Terms.** If the editor judges any provider's output terms incompatible with committing responses, commit hashes, usage and per-item scores only, and keep the text local; the replay then works from the author's machine and the build runs the live-free subset.
- **Rate limits are unpublished.** The HF documentation read here states none; the sweep measures them for one provider on one day and says so.
