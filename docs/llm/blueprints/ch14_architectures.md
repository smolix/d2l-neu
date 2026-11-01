# Chapter 14: Advanced Architectures — Blueprint

**Label:** `chap_lm_architectures` · **Arc:** VII. Architecture (which architectural changes do current models make, and what does each buy?) · **Sections:** 14.1–14.9 (14.8 and 14.9 optional ◇) · **Tiers:** T0 (14.1, 14.9), T1 (14.3, 14.5, 14.8), T2 (14.2, 14.4, 14.7, each with a reduced T1 configuration; one A100 40 GB, plus a T2 pair for 14.2's routing run), T3 (14.6, with a T1 reduction; 8× A100 40 GB)

*Authoring status: blueprint v1, written 2026-10-28 against the Hugging Face hub as of that date. Every model fact in this file was read from a `config.json`, a safetensors header, the Hub API, or a technical report fetched that day; items that could not be confirmed are marked "(unverified)" or "to be measured". Numbers labelled "pre-check" were computed by the blueprint author with a throw-away script from the cited config files (CPU only); the notebooks must recompute them, not copy them.*

## Chapter overview

### Draft introduction (≈170 words)

The transformer of Advanced is a starting point, not a destination. Open the configuration file of a model released this year and most fields are familiar (RoPE, GQA, RMSNorm, SwiGLU, a mixture of experts), but a growing minority are not: a latent KV rank, an indexer that picks 2048 of a million keys, a layer type called `linear_attention`, a `hc_mult` of 4, an embedding table with 384 million rows. This chapter teaches those fields. It is organized by *axis* of the decoder, not by model: expert sparsity, KV layout, attention form, sequence mixing, the residual stream, and (optionally) tokens and decoding order. For each axis it asks what a change saves, what it costs, and which training and serving setup (Chapter 5) makes the trade worthwhile. Section 14.1 builds the table that the other sections annotate: a design table compiled from the configuration files of released models, with a function that turns any configuration into parameters, active compute and cache bytes per token. Every later section picks one column, derives the effect from first principles, and measures it at small scale. The chapter is designed to grow: a new release adds a row, and a new axis adds a section.

### Learning objectives

By the end of the chapter a reader can:

1. **Reconstruct** a released model's total parameters, active parameters, FLOPs per token and cache bytes per token from its configuration file alone, and reconcile the first with the safetensors index to the parameter (dense, MoE and MLA families) or to a stated residual (hybrids, vision towers, MTP heads, lookup tables). *(14.1)*
2. **Read** a technical report with a schema that separates disclosed facts, configuration-file facts and inferences, and trace a headline claim to the ablation that supports it, or to none. *(14.1)*
3. **Derive**, for each axis, a closed-form saving and the cost that pays for it, and verify the derivation in a numerical test: optimal MoE sparsity at fixed compute and all-to-all bytes with a latent width (14.2); absorbed-matrix MLA decode and cache bytes (14.3); why a softmax head needs a sink and what a gate replaces (14.4); the crossover context of learned sparse attention (14.5); state versus KV bytes of a hybrid (14.6); the spectral properties of doubly stochastic residual mixing (14.7). *(14.2–14.7)*
4. **Implement** each mechanism in at most ~150 lines of PyTorch, check it against a reference (Transformers module, the book's own code, or a closed form) within a stated tolerance, and **measure** its effect at small scale with an explicit noise check. *(14.2–14.8)*
5. **Load** a released hybrid checkpoint (Qwen3.5-2B) into the book's own layer code and match the Transformers logits. *(14.6)*
6. **Map** every field of a current configuration to the section that teaches it, and explain a design decision in terms of the hardware the model was trained and served on. *(all)*
7. **Design** a model and training plan under a fixed budget, with one measured baseline and one rejected design with evidence. *(Assignment, 14.9)*

### Prerequisites

Book labels (what is used; nothing is re-taught):

| Label | Used for |
|---|---|
| `sec_scaling-laws` | the $12Ld^2$ census, non-embedding $N$, $6ND$, `tab_modern-recipe` (the seven-row prose table that 14.1 turns into data), constructor-call idiom (`blk.attention = d2l.GQAAttention(...)`) |
| `sec_moe` | `moe_accounting` (a hand-run census of Mixtral and DeepSeek-V3 that `d2l.config_stats` generalizes), `MoELayer` (dense-gather, local to that section, **not** `#@save`), bias-controlled balancing, fine-grained and shared experts, `tab_moe-experts` |
| `sec_kv-cache` | `eq_kv-cache-bytes`, `d2l.GQAAttention`, `subsec_kv-cache-low-rank` (oracle SVD of GPT-2 keys and values; MLA named, decoupled RoPE named but not derived), sliding window and the GPT-2 first-token mass measurement, `tab` comparing cache-reduction methods |
| `sec_hybrids`, `sec_deltanet`, `sec_mamba`, `sec_matrix-state` | attention fraction and placement rules, `tab_hy-recipe` (nine shipped hybrids, 2024–2025), `eq_dn-gated`, `delta_chunked` (chunked WY form, un-gated, local), `GatedDeltaNetBlock` (toy: LayerNorm, GELU MLP, time loop), `segsum` and `matrix_state_chunked` |
| `sec_attention-at-scale`, `sec_positional-information` | sliding-window masks and receptive-field growth, online softmax; RoPE, ALiBi, NoPE, `TinyCharLM` (`#@save`), the extrapolation protocol (train at $L$, evaluate at $4L$) |
| `sec_transformer-block`, `sec_gpt` | `d2l.TransformerBlock` (`attn_factory`, `ffn_factory` seams), `d2l.FeedForward` (width fixed at $8d/3$), `d2l.GPT`, QK-norm |
| `sec_diffusion-discrete` | masked-diffusion bound, unmasking sampler, the $1/S$ independence error on a two-token example (for 14.8) |
| `sec_mdl-svd-low-rank`, `sec_init_param` | Eckart–Young and low-rank factorization (14.3); signal propagation at initialization (14.7) |
| `sec_perf_model`, `sec_hardware` | roofline, decode bound; the chip table (via `d2l.chip_specs()`) |

Earlier sections of this part:

| Section | Used for |
|---|---|
| 1.1 `sec_lm_hf_models` | configuration files, `d2l.load_hf`, `d2l.config_memory(config, seq_len, batch_size, dtype_bytes)`, the Qwen3.5-2B annotation |
| 1.2 `sec_lm_d2l_checkpoint` | `d2l.load_qwen3` (R1), the weight-name mapping and RoPE-layout vocabulary, the layer-by-layer logit matching protocol |
| 4.2 `sec_lm_reference_model` | R2 (`d2l.load_reference_model`), its tokenizer, data slice and training recipe (the base of 14.2, 14.4, 14.7) |
| 4.3 `sec_lm_instabilities` | QK-norm, logit growth (14.4, 14.7 stable-LR ranges) |
| 4.6 `sec_lm_long_context` | YaRN and RULER-style tasks (14.4) |
| 5.1, 5.2 `sec_lm_accelerator_design`, `sec_lm_workload_hardware` | `d2l.chip_specs()`, the decode-step model $t(b,L)=\max(\cdot,\cdot)$ (14.3, 14.5, 14.9) |
| 6.6 `sec_lm_expert_parallel` | all-to-all dispatch, node-limited routing (14.2 extends, does not repeat) |
| 12.3, 12.4 | MTP heads reused for drafting (14.8 is the pretraining side); inference-time KV compression (contrasted in 14.3 and 14.5) |
| 1.5 `sec_lm_eval_suites` | `d2l.paired_bootstrap` (every headline experiment) |

Assumed from other chapters, beyond the conventions' reserved list (each repeated under the section that needs it):

- `d2l.config_memory(config, seq_len, batch_size, dtype_bytes=2, *, state_bytes=4, ...)` (1.1) returns a `dict` with keys `params, weights, kv_cache, recurrent_state, activations, logits, total` (bytes except `params`); 14.1 cross-checks against it. Imported as is (01_interfaces.md section 1).
- `d2l.chip_specs()` returns the 5.1 `pandas.DataFrame` with columns `chip, vendor, kind, year, mem_type, mem_gb, mem_bw_tbs, sram_mb, p16_tflops, p8_tflops, p4_tflops, tdp_w, scaleup_gbs, scaleup_note, scaleup_domain, scaleout_gbs, sparsity_note, source_url, retrieved, notes` (14.3, 14.5, 14.9 use `p16_tflops` (bf16), `mem_bw_tbs` (TB/s) and `mem_gb`).
- `d2l.load_reference_model` returns the R2 `d2l.GPT`-compatible module (it returns the model only; the tokenizer is `d2l.r2_tokenizer()` (3.6)), with `d2l.load_reference_model(step=None, dtype=..., device=..., preset='r2')` (keys `lm-r2-config`, `lm-r2-s{step}`; 4.2); 14.2, 14.4 and 14.7 retrain variants from scratch with `d2l.train_lm_steps`. 14.2 exports `d2l.train_lm_steps` (below) for training caller-built variant modules, since 4.2's `d2l.pretrain(PretrainConfig)` builds R2 itself.
- `d2l.fla_available()` is **not** assumed; 14.6 checks for the `fla` package itself.

### Section dependency graph

```
                        14.1 Reading an architecture  (config_stats, design table, report schema)
                         │
   ┌─────────────┬───────┼──────────────┬────────────────┬───────────────┬────────────┐
 14.2 MoE      14.3 MLA/CLA        14.4 refinements   14.6 hybrids    14.7 residuals   14.8 ◇ tokens
 (FFN axis)    (KV layout)         (attention form,   (sequence       (depth axis)     and decoding
   │              │                 sinks, positions)  mixing)                          order
   │              └──► 14.5 learned sparse attention        ▲
   │                    (token mixing, selection)            │ sec_hybrids, sec_deltanet
   └───────────────────────────────┬──────────────────────────┘
                                   ▼
                      14.9 ◇ Designing under a budget  (+ Assignment)
```

- 14.1 is required by every other section (`config_stats`, `design_table`).
- 14.5 requires 14.3 (the MLA latent is the main-attention substrate of DSA) and reuses the `MLAAttention` class.
- 14.6 additionally imports `d2l.RefinedAttention` (gated attention, partial RoPE) from 14.4 for its full-attention layers; 14.6, 14.7 and 14.2 are otherwise mutually independent. 14.7's R2-scale harness shares the variant-training loop defined in 14.4 (`d2l.train_lm_steps`).
- 14.9 and the Assignment need 14.1 only; they use any other section's measured numbers when the student has them.
- Minimal course paths from v5: **LLM course** 14.1, 14.2, 14.6; **Systems for Large Models** 14.3, 14.5 (+14.1); **Architecture seminar**: the whole chapter.

### Running examples and shared artifacts

*Used:* R1 (`d2l.load_qwen3`, 1.2; the dense row, and the oracle for 14.6's attention layers), R2 (4.2; the base of every training experiment), R3 (the book as a corpus; the 14.8 segmentation corpus), Qwen3.5-2B (1.1), Qwen3-30B-A3B (14.2 routing analysis), `d2l.chip_specs()` (5.1), `d2l.paired_bootstrap` (1.5), `d2l.GPT`, `d2l.GQAAttention`, `d2l.TransformerBlock`.

*Produced (data, all under `data/arch/` in the repo, delivered through `d2l.DATA_HUB`):*

| Artifact | `DATA_HUB` key | Format | Produced in | Used in |
|---|---|---|---|---|
| Pinned `config.json` snapshots of every model in the design table (≈ 40 files) | `lm-arch-configs` | `.tar.gz` of `<org>__<name>.json`, plus `revisions.json` mapping model id to commit SHA | 14.1 | all (T0 notebooks run offline) |
| Model registry `models.yaml` (id, revision, release date, report URL, license, `axis_notes`) | `lm-arch-registry` | YAML | 14.1 | all |
| Disclosed-values file `expected.yaml` (values stated in reports or model cards: total/active parameters, cache bytes, layer counts, with the page or section cited) | `lm-arch-expected` | YAML | 14.1 | 14.1 tests, 14.9 |
| Design table `design_table.csv` (one row per model, one column per axis, generated, never hand-edited) | `lm-arch-design-table` | CSV | 14.1 | all, 14.9 |
| Parameter group census (per model, by tensor-name prefix, from safetensors headers) | `lm-arch-headers` | JSON, ≈ 1 MB total (headers only, no weights) | 14.1 | 14.1 tests |
| Sparsity/granularity sweep logs and fitted law | `lm-arch-moe-sweep` | JSONL + `fit.json` | 14.2 (T2) | 14.2 analysis, 14.9 |
| Routing traces of Qwen3-30B-A3B (expert ids and weights per token, per layer, three domains) | `lm-arch-routing-qwen3-30b` | `.npz`, int16 ids | 14.2 | 14.2 analysis |
| MHA→MLA conversion curves | `lm-arch-mla-conversion` | JSON | 14.3 | 14.3 |
| Variant training logs (sink, gate, partial-RoPE; seeds) | `lm-arch-refine-logs` | JSONL | 14.4 (T2) | 14.4 analysis |
| Selection-overhead profiles | `lm-arch-sparse-profiles` | JSON (Chrome-trace summaries) | 14.5 | 14.5 |
| Hybrid-vs-attention serving measurements (memory and throughput vs context) | `lm-arch-hybrid-serving` | JSONL | 14.6 (T3) | 14.6 analysis, 14.9 |
| Residual-variant logs (norm, gradient norm per layer, LR sweep) | `lm-arch-residual-logs` | JSONL | 14.7 (T2) | 14.7 analysis |

### Compute plan

| § | Tier | Budget (design target, not a measurement) | Peak memory | Committed artifacts |
|---|---|---|---|---|
| 14.1 | T0 | ≤ 5 min on CPU after a one-time config download; offline thereafter | < 2 GB | `lm-arch-configs`, `lm-arch-registry`, `lm-arch-expected`, `lm-arch-design-table`, `lm-arch-headers` |
| 14.2 | T2 (T1: 4 runs of a tiny grid) | T2 ≤ 3 h on 1× A100 40 GB for the sweep, + ≤ 20 min on a T2 pair for routing analysis; T1 ≤ 15 min | T2 ≤ 34 GiB sweep (design target), routing ≈ 30.5 GB of weights per GPU on the pair; T1 ≤ 12 GB | sweep logs, routing traces |
| 14.3 | T1 | ≤ 12 min on the build box | ≤ 10 GiB | conversion curves |
| 14.4 | T2 (T1: tiny model, 3 seeds) | T2 ≤ 3.5 h (4 variants + 2 extra baseline seeds); T1 ≤ 15 min | T2 ≤ 34 GiB (design target, A100 40 GB) | variant logs |
| 14.5 | T1 | ≤ 12 min | ≤ 12 GiB | selection profiles |
| 14.6 | T3 (T1 reduction: Qwen3.5-2B vs Qwen3-1.7B) | T3 ≤ 2 h on 8× A100 40 GB (≈ 16 GPU-hours); T1 ≤ 15 min | T3 ≤ 36 GiB per GPU (0.9 × 40 GB; weights 9.0 GB per GPU at TP = 8) | serving measurements, weight-conversion logit-match report |
| 14.7 | T2 (T1 reduced) | T2 ≤ 3.5 h; T1 ≤ 15 min | T2 ≤ 34 GiB (design target, A100 40 GB) | variant logs |
| 14.8 ◇ | T1 | ≤ 15 min | ≤ 12 GiB | byte-LM checkpoint, patch statistics |
| 14.9 ◇, Assignment | T0 (+T1 for the measured baseline) | T0 minutes; T1 baseline ≤ 15 min | — | the student's dossier |

### How to extend this chapter

This subsection is the editor's maintenance manual and also appears, shortened to one paragraph, at the end of 14.1's narrative. It has three parts: the row procedure, the axis rule and the refresh checklist.

**Entering a new model (a row).** A release enters the design table in five steps, none of which edits prose:

1. Add an entry to `data/arch/models.yaml`: `id` (Hub id), `revision` (full commit SHA; fetch through `huggingface_hub.HfApi().model_info(id).sha`), `released` (the Hub `createdAt` date, not the paper date), `report` (arXiv id or URL), `license`, `core: true|false` (core rows are the ≈ 12 the T0 notebook plots; extended rows are in the CSV only), and `axis_notes` (one free-text line per axis that the configuration file cannot express).
2. Run `d2l.config_stats` on its configuration. If the `model_type` is unknown, write a *layer plan* function and register it with `@d2l.register_arch('<model_type>')` (≤ 30 lines: it returns the list of `LayerSpec` objects of 14.1). Most releases reuse existing layer kinds (`gqa`, `swa`, `mla`, `dsa`, `gdn`, `kda`, `lightning`, `csa2`, `dense`, `moe`, `latent_moe`). A genuinely new layer kind is an `#@save` addition with a closed-form cost, and it is the first sign that an axis may be new.
3. Check the derived numbers against two references and record both in the notebook output: the safetensors header census (exact for BF16 backbones, a stated residual otherwise), and the `expected.yaml` entry (values the report or model card states; each with its source page). A mismatch is a finding: write it into `axis_notes`, never tune the calculator to hit a number.
4. Fill the report schema of 14.1 for the model (one YAML file in `data/arch/reports/`, ≤ 40 lines). The schema's `inferred` block must be non-empty for any claim the report does not support with an ablation.
5. Regenerate `design_table.csv` and the figures (`fig_lm_arch_axes_timeline`, `tab_lm_design_table`); commit the CSV and the diff of the figures.

**Deciding between a row, a paragraph and a section.** A release *moves an existing axis* when it changes a value in an existing column (a different top-$k$, a new window): row only. It earns a paragraph in the owning section when it changes the *analysis* (for example, a release whose KV cache is smaller than the section's derived lower bound for its axis, or a new router score function): add the paragraph, the row, and a check cell. A new axis earns its own section only if **all four** hold:

- (a) it changes a design-table column that no existing section owns;
- (b) it appears in at least two independent model families, or in one family together with an ablation in its report;
- (c) it has a cost and a benefit that a notebook can measure at T2 or below (a headline experiment exists);
- (d) it cannot be explained in under a page as a setting of an existing axis.

*Worked applications of the rule at the time of writing.* (i) **Engram / per-layer n-gram embeddings** (DeepSeek-V4.1-Flash `engram_*` fields; Qwen3.8-Flash-Next `ngram_*` and `ple_*` fields): conditions (a), (b: two families) and (c) hold; (d) arguably holds because it is "sparsity without arithmetic". It is therefore a sub-block of 14.2 (`subsec_lm_moe_lookup`) and stays one unless Engram-style tables spread across several releases (editor's decision, 2026-10-29). (ii) **Cross-layer KV and index reuse** (V4.1-Flash `kv_source_layer_ids`, `index_source_layer_ids`; GLM-5.3 `indexer_types`): split between 14.3 (KV sharing, CLA) and 14.5 (index sharing); the rule says no new section because (d) holds. (iii) **Speculative drafters shipped inside the checkpoint** (MTP, DSpark): 12.3 owns them; 14.8 owns only the pretraining objective; the table has a column, no section. (iv) **Output heads / multi-token prediction** is already an axis in v5's list and lives in optional 14.8.

**Refresh checklist per edition** (≈ half a day): (1) `python -m d2l_arch.refresh --since <date>` lists new Hub repos from the organizations in `models.yaml` (`HfApi().list_models(author=..., sort='created_at')`); (2) add rows for models that move an axis; (3) re-run the T0 notebook of 14.1 and diff `design_table.csv`; (4) update the "State of the art" sentence at the top of each section (≤ 3 lines); (5) re-run only the cheap checks (14.1, 14.3 equivalence, 14.4 identities, 14.7 spectral test); the T2/T3 experiments are re-run only when their model lines change; (6) delete rows whose Hub repo disappeared and keep the snapshot in `lm-arch-configs` (configs are pinned, so old rows keep working).

### Config fact sheet (fetched 2026-10-28; every field below was read from the model's `config.json`)

Used by the design table of 14.1 and quoted again in the sections that analyze each axis. "Release" is the Hub `createdAt`. "Safetensors total" is the Hub API `safetensors.total` (elements, packed FP4/FP8 expanded to elements by the API; see the risk in 14.1). Models marked † are the core rows (≈ 12) of the T0 table.

| Model (release) | Fields relevant to the axes |
|---|---|
| † Qwen3-8B (2025-04-27) | `hidden_size 4096`, `num_hidden_layers 36`, `num_attention_heads 32`, `num_key_value_heads 8`, `head_dim 128`, `intermediate_size 12288`, `vocab_size 151936`, `tie_word_embeddings false`, `rope_theta 1e6`; safetensors total 8,190,735,360 |
| † Qwen3-30B-A3B (2025-04-27) | `hidden_size 2048`, 48 layers, 32 heads / 4 KV, `head_dim 128`, `num_experts 128`, `num_experts_per_tok 8`, `moe_intermediate_size 768`, `norm_topk_prob true`, `mlp_only_layers []`, `router_aux_loss_coef 0.001`; total 30,532,122,624 |
| Qwen3-235B-A22B | `hidden_size 4096`, 94 layers, 64 heads / 4 KV, 128 experts, top-8, `moe_intermediate_size 1536`; total 235,093,634,560 |
| † gpt-oss-120b (2025-08-04) | `hidden_size 2880`, 36 layers, 64 heads / 8 KV, `head_dim 64`, `attention_bias true`, `layer_types` alternating `sliding_attention`, `full_attention`, `sliding_window 128`, `num_local_experts 128`, `experts_per_token 4`, `intermediate_size 2880`, `swiglu_limit 7.0`, `quantization_config.quant_method mxfp4`; total 116,829,156,672. gpt-oss-20b: 24 layers, 32 experts, total 20,914,757,184 |
| GLM-4.5 (2025-07-20) | `hidden_size 5120`, 92 layers, 96 heads / 8 KV, `head_dim 128`, `partial_rotary_factor 0.5`, `use_qk_norm true`, `attention_bias true`, 160 routed + 1 shared experts, top-8, `first_k_dense_replace 3`, `num_nextn_predict_layers 1`; total 358,337,791,296 |
| MiniMax-M2 (2025-10-22) | `hidden_size 3072`, 62 layers, 48 heads / 8 KV, 256 experts, top-8, `intermediate_size 1536`, `use_qk_norm true`, `attn_type_list` all 1 (62 of 62 softmax layers), `num_mtp_modules 3`; total 228,689,764,864 |
| MiniMax-Text-01 (MiniMax-01, 2025-01; `MiniMaxAI/MiniMax-Text-01`, `model_type minimax_text_01`) | `hidden_size 6144`, 80 layers, 64 heads / 8 KV, `head_dim 128`, `rotary_dim 64`, `num_local_experts 32`, `num_experts_per_tok 2`, `intermediate_size 9216`, `vocab_size 200064`, `max_position_embeddings 10240000`, `postnorm true`; `attn_type_list` has 10 ones (softmax every 8th) and 70 zeros (lightning attention). The original lightning-attention hybrid row; M1 reuses this layout. Config read; parameter total not computed (to be measured) |
| MiniMax-M1-80k (2025-06-13) | 80 layers, `attn_type_list` has 10 ones (softmax every 8th) and 70 zeros (lightning attention), 64 heads / 8 KV, 32 experts, top-2, `hidden_size 6144`; shows that M2 reverted to full attention |
| MiniMax-M3 (2026-06-02) | `hidden_size 6144`, 60 layers, 64 heads / 4 KV, 128 experts top-4 + 1 shared, `moe_layer_freq` has 57 ones of 60, `sparse_attention_config`: `sparse_block_size 128`, `sparse_topk_blocks 16`, `sparse_num_index_heads 4`, `sparse_index_dim 128`, active on 57 of 60 layers, `sparse_local_block 1`, `sparse_init_block 0`; total 427,040,140,160 |
| DeepSeek-V2 (2024-04-22) | `hidden_size 5120`, 60 layers, 128 heads, `kv_lora_rank 512`, `q_lora_rank 1536`, `qk_nope_head_dim 128`, `qk_rope_head_dim 64`, `v_head_dim 128`, 160 routed + 2 shared experts, top-6, `moe_intermediate_size 1536`, `first_k_dense_replace 1`, `n_group 8`, `topk_group 3`, `scoring_func softmax`, `routed_scaling_factor 16`; total 235,741,434,880 |
| † DeepSeek-V3 (2024-12-25) | `hidden_size 7168`, 61 layers, 128 heads, `kv_lora_rank 512`, `q_lora_rank 1536`, rope 64 / nope 128 / v 128, 256 routed + 1 shared, top-8, `moe_intermediate_size 2048`, `first_k_dense_replace 3`, `n_group 8`, `topk_group 4`, `scoring_func sigmoid`, `topk_method noaux_tc`, `num_nextn_predict_layers 1`, `quantization_config` fp8 e4m3 `weight_block_size [128,128]`; total 684,489,845,504 (671B + MTP module) |
| DeepSeek-V3.2 (2025-12-01) | V3's fields plus `index_n_heads 64`, `index_head_dim 128`, `index_topk 2048`, `scale_fmt ue8m0`; total 685,355,329,792 |
| Kimi-K2 (2025-07-11) | DeepSeek-V3 architecture: `hidden_size 7168`, 61 layers, 64 heads (not 128), 384 routed + 1 shared experts, top-8, `n_group 1`, `first_k_dense_replace 1`; total 1,026,408,235,864 |
| † DeepSeek-V4-Flash (2026-04-22) | `hidden_size 4096`, 43 layers + 1 MTP, 64 heads, `num_key_value_heads 1`, `head_dim 512`, `qk_rope_head_dim 64`, `q_lora_rank 1024`, `o_lora_rank 1024`, `o_groups 8`, `sliding_window 128`, `compress_ratios` length 44 starting `[0, 0, 4, 128, 4, 128, …]`, `index_n_heads 64`, `index_head_dim 128`, `index_topk 512`, 256 routed + 1 shared, top-6, `scoring_func sqrtsoftplus`, `num_hash_layers 3`, `hc_mult 4`, `hc_sinkhorn_iters 20`, `expert_dtype fp4`, `max_position_embeddings 1048576`; total 290,944,616,402. DeepSeek-V4-Pro: `hidden_size 7168`, 61 layers, 128 heads, 384 experts, `index_topk 1024`, `o_groups 16`; same family |
| † DeepSeek-V4.1-Flash (2026-09-10) | `text_config`: `hidden_size 5120`, `num_hidden_layers 40`, 64 heads, `num_key_value_heads 1`, `head_dim 512`, `q_lora_rank 1280`, `o_lora_rank 1024`, `o_groups 8`, `sliding_window 128`, `compress_ratios` (length 43 = 40 layers + 3 MTP/draft slots: `[0,0]`, then 18 × `2`, then 20 × `1`, then `[0,0,0]`), `kv_source_layer_ids [2,8,14,20]`, `index_source_layer_ids [2,8,14,20,24,28,32,36]`, `index_n_heads 32`, `index_head_dim 128`, `index_topk 512`, `candidate_block_size 8`, `candidate_topk_blocks 2048`, 384 routed + 1 shared experts, top-6, `moe_intermediate_size 2304`, `scoring_func sqrtsoftplus`, `hc_mult 4`, `hc_sinkhorn_iters 20`, `engram_layer_ids [1,14]`, `engram_num_embeddings [384006168, 384016682]`, `engram_head_dim 256`, `engram_n_heads 8`, `engram_max_ngram_size 4`, `num_nextn_predict_layers 3`, `dspark_block_size 5`, `dspark_n_routed_experts 128`, `quantization_config` fp8 block 32 with `expert_dtype fp4`, `max_position_embeddings 1048576`; vision tower in `vision_config` (32 layers, 1024 wide); safetensors total 763,205,315,794 (report: 552B backbone, 16B active in decode, 8B in prefill) |
| † Qwen3-Next-80B-A3B (2025-09-09) | `hidden_size 2048`, 48 layers, `full_attention_interval 4`, `head_dim 256`, 16 heads / 2 KV, `partial_rotary_factor 0.25`, `linear_num_key_heads 16`, `linear_num_value_heads 32`, `linear_key_head_dim 128`, `linear_value_head_dim 128`, `linear_conv_kernel_dim 4`, 512 experts top-10, `moe_intermediate_size 512`, `shared_expert_intermediate_size 512`; total 81,324,862,720 |
| † Qwen3.5-2B (2026-02-28) | `text_config`: `hidden_size 2048`, 24 layers (`layer_types`: 3 × `linear_attention` then `full_attention`, repeated; 18 + 6), `attn_output_gate true`, 8 heads / 2 KV, `head_dim 256`, `intermediate_size 6144` (dense), `linear_num_key_heads 16`, `linear_num_value_heads 16`, key/value head dim 128, `linear_conv_kernel_dim 4`, `rope_parameters`: `rope_theta 1e7`, `partial_rotary_factor 0.25`, `mrope_interleaved true`, `mrope_section [11,11,10]`; `mtp_num_hidden_layers 1`, `tie_word_embeddings true`, `vocab_size 248320`, a 24-layer vision tower; total 2,274,069,824 (of which the text backbone is 1.882B by the blueprint's count; the rest is the vision tower and the MTP head). Qwen3.5-0.8B: total 873,438,784 |
| † Qwen3.5-35B-A3B (2026-02-24) / Qwen3.6-35B-A3B (2026-04-15) | `qwen3_5_moe_text`: `hidden_size 2048`, 40 layers, 16 heads / 2 KV, `head_dim 256`, 256 experts, top-8, `moe_intermediate_size 512`, `shared_expert_intermediate_size 512`, `linear_num_value_heads 32`; Qwen3.6 total 35,951,822,704 |
| Qwen3.6-27B, Qwen3.8-27B (2026-08-05) | dense hybrids: `hidden_size 5120`, 64 layers (48 linear + 16 full), 24 heads / 4 KV, `intermediate_size 17408`, `linear_num_value_heads 48`, `output_gate_type swish`; Qwen3.8-27B total 27,781,427,952 |
| Qwen3.8-2.4T-A95B (2026-08-08, license `other`) | `hidden_size 8192`, 92 layers (69 linear + 23 full), 64 heads / 4 KV, 512 experts, top-10, `moe_intermediate_size 2048`, `linear_num_value_heads 128`; total 2,446,182,725,504 |
| Qwen3.8-Flash-Next (2026-08-24, license `other`) | `model_type qwen4_exp_text`: `hidden_size 2560`, 48 layers (`full_attention_interval 4`), 24 heads / 2 KV, 512 experts top-10, `moe_intermediate_size 640`, `hc_count 4`, `hc_lowrank 320`, `indexer_budget 2048`, `indexer_compress_ratio 4`, `indexer_n_heads 4`, `indexer_head_dim 128`, `ngram_size 3`, `ngram_vocab_size_base 20000000`, `ple_embed_dim 2560`, `ple_layer_ids [2]`, `output_gate_type sigmoid`, `mtp` hybrid; total 179,999,981,459 |
| Kimi-Linear-48B-A3B (2025-10-30) | `hidden_size 2304`, 27 layers: `kda_layers` 20, `full_attn_layers [4,8,12,16,20,24,27]` (MLA, `mla_use_nope true`, `kv_lora_rank 512`, `q_lora_rank null`, 32 heads), `linear_attn_config.num_heads 32`, `head_dim 128`, `short_conv_kernel_size 4`, 256 experts top-8 + 1 shared, `moe_intermediate_size 1024`; total 49,122,681,728 |
| † Kimi-K3 (2026-06-13, license `other`) | `text_config`: `hidden_size 7168`, 93 layers, `linear_attn_config`: 69 `kda_layers` + 24 `full_attn_layers` (every 4th through 92, plus 93), `num_heads 96`, `head_dim 128`, `short_conv_kernel_size 4`, `gate_lower_bound -5.0`, `use_full_rank_gate true`; MLA: `kv_lora_rank 512`, `q_lora_rank 1536`, rope 64 / nope 128 / v 128, `mla_use_nope true`, `mla_use_output_gate true` (weights: `kv_a_proj_with_mqa` is `[576, 7168]`); MoE: `num_experts 896`, `num_experts_per_token 16`, `num_shared_experts 2`, `moe_intermediate_size 3072`, `routed_expert_hidden_size 3584` (the latent width, half of 7168), `latent_moe_use_norm true`, `first_k_dense_replace 1` (`intermediate_size 33792`), `hidden_act situ`; `attn_res_block_size 12`; `quantization_config` MXFP4 (group 32) on routed experts only (attention, shared experts, dense MLP and `lm_head` ignored); total 2,779,931,837,184 (report: 2.8T total, 104B active) |
| GLM-5 (2026-02-11) | `glm_moe_dsa`: `hidden_size 6144`, 78 layers, 64 heads, `kv_lora_rank 512`, `q_lora_rank 2048`, `qk_nope_head_dim 192`, `qk_rope_head_dim 64`, `v_head_dim 256`, `index_n_heads 32`, `index_head_dim 128`, `index_topk 2048`, 256 routed + 1 shared, top-8, `first_k_dense_replace 3`; total 753,864,139,008 |
| GLM-5.3 (2026-08-25) | GLM-5 plus `indexer_types` (78 entries: 21 `full`, 57 `shared`: indexer results are reused across layers), `index_topk_freq 4`, `mlp_layer_types` (3 dense, 75 sparse), `max_position_embeddings 1048576` |
| GLM-5.3-Flash (2026-08-25) | `glm5_next_text`: `hidden_size 4096`, 45 layers: `layer_types` 34 `linear_attention` (KDA) + 11 `deepseek_sparse_attention`, `mhc true`, `hc_mult 4`, `hc_sinkhorn_iters 20`, `index_kpool 4`, 288 routed + 1 shared, top-8; total 321,323,031,390 |

*Not available:* `meta-llama/Llama-3.1-8B` returns HTTP 401 (gated), so Llama 3 is a report-only row (its architecture table in the paper) and the dense reference row is Qwen3-8B. `MiniMaxAI/MiniMax-01` returns 401 because it is the wrong id: the MiniMax-01 text model is `MiniMaxAI/MiniMax-Text-01` (public, HTTP 200; `config.json` fetched 2026-10-28), now a design-table row below.

### Facts v5 asserts: confirmed, qualified, or not confirmable

| v5 statement | Status |
|---|---|
| "V2 adds MLA; V3 adds aux-loss-free MoE, MTP, FP8, DualPipe; V3.2 adds learned sparse attention; V4.1-Flash adds KV compression" | V2/V3/V3.2 confirmed by configs (`kv_lora_rank` in V2, `scoring_func sigmoid`/`noaux_tc`/`num_nextn_predict_layers`/fp8 in V3, `index_*` in V3.2). DualPipe is a training-system fact not visible in any config (report-only). **Qualified for V4.1-Flash:** DeepSeek-V4 (Hub 2026-04-22; arXiv 2606.19348) sits between V3.2 and V4.1-Flash and already introduced the `compress_ratios` (CSA/HCA at 4 and 128), `hc_mult 4` (mHC) and `num_hash_layers`. V4.1-Flash adds cross-layer KV and index reuse, the causal encoder–decoder split, FP4 main KV, single-pass mHC, Engram and DSpark (report §2). The blueprint keeps v5's four-step line and adds V4-Flash as a fifth row (deviation 1). |
| "V4.1-Flash adds KV compression (12.4)" | **Not confirmed as written.** 12.4 is inference-time compression (quantizing and evicting a trained model's cache). V4.1-Flash's compression is architectural (sequence-dimension compression and layer-dimension sharing, trained in) and is taught in 14.3 and 14.5. The FP4 main KV cache is QAT (report §2.4.4) and is a pointer to 11.5/12.4. |
| "Qwen3.5: Gated DeltaNet with one gated-attention layer in four" | Confirmed (`full_attention_interval 4`, `attn_output_gate true`, `layer_types`). |
| "Kimi K3 (16 of 896 experts with Stable LatentMoE)" | Confirmed (`num_experts 896`, `num_experts_per_token 16`, `routed_expert_hidden_size 3584`, `latent_moe_use_norm`). |
| "Attention Residuals … as in Kimi K3" | Confirmed (`attn_res_block_size 12`; report §2.2: block AttnRes, 8 blocks of 12 layers plus the embedding source). |
| "gpt-oss learned sink logits" | Confirmed by the Transformers module (`self.sinks = nn.Parameter(num_attention_heads)`; the sink is concatenated as an extra logit column and dropped after the softmax). |
| "MiniMax's lightning attention" (14.6) | Confirmed for MiniMax-Text-01 and MiniMax-M1 (each 70 lightning + 10 softmax layers). **Qualified:** MiniMax-M2 (`attn_type_list` all ones) reverted to full attention and MiniMax-M3 uses block-sparse attention. 14.6 therefore uses M1 as the lightning row and M2/M3 as evidence that the choice is not monotone (deviation 3). |
| "Qwen3.6 / Qwen3.8" (editor's note, not in v5) | Exist on the Hub (Qwen3.6-35B-A3B, Qwen3.6-27B, Qwen3.8-27B, Qwen3.8-2.4T-A95B, Qwen3.8-Flash-Next). Qwen3.6 and 3.8-27B/-2.4T reuse the `qwen3_5` layout; the config differs only in sizes and `output_gate_type`. **Qwen3.8-Flash-Next is a new layout** (`qwen4_exp_text`: hyper-connection streams `hc_count 4`, an indexer, n-gram embeddings, per-layer embeddings). No technical report was located for it: treat its rows as config-only (no `expected` values). |
| "mHC" first appears in V4.1-Flash | No: V4-Flash/V4-Pro configs carry `hc_mult 4`, and the mHC paper (2512.24880) predates them. GLM-5.3-Flash (`mhc true`) and Qwen3.8-Flash-Next (`hc_count 4`) also use it. |

## Assignment: Design under a budget (the course capstone)

**Label:** `sec_lm_capstone`

*v5 places the capstone in Chapter 14 ("Capstone | Design under a budget | 14 | Any | One measured baseline and one rejected design, with evidence"). 14.9 is optional (◇) and carries the guided worked example. This assignment is written so that it stands alone with 14.1 (`d2l.config_stats`) plus the 5.2 decode model; every other section is an optional source of evidence. Editor's decision (2026-10-29): the capstone stays hosted by 14.9, so the assignment stays self-contained and 14.9 can still be cut, but the part's assignment table lists it unnumbered as a chapter-level entry.*

### Overview

A student receives a *scenario* (task distribution, data, training compute, serving hardware, context length, latency and concurrency targets, a quality gate) and must produce a defensible architecture and training plan. The work is judged on engineering judgment, not on matching a frontier leaderboard: every claim in the dossier must trace to a number that `d2l.config_stats`, the 5.2 decode model, or the student's own measurement produced. Nothing is trained at scenario scale; the scenario scale is *analytic*, and the evidence is measured at small scale.

### Scenarios (assignment parameters; the editor may change any value)

| | **A. Long-context agent serving** | **B. Small private assistant** (the T1 path) | **C. High-throughput batch generation** |
|---|---|---|---|
| Task distribution | multi-turn tool use over repositories; prompts of 100–128K tokens, mostly cached prefix, 2–4K new tokens per turn | short-turn chat and extraction, prompts ≤ 8K | document rewriting, prompts 2K, outputs 1K |
| Serving hardware | one 8× A100-80GB node (`d2l.chip_specs()` row, no FP8, bf16 weights) | one RTX 4090 (Ada, 24 GB, the readers' floor) | one 8× A100-80GB node |
| Concurrency target | 32 sessions at 128K context | 4 sessions at 8K | as many as fit; objective is tokens/s/GPU |
| Latency target | ≥ 25 tokens/s per user at that context | ≥ 20 tokens/s per user | none (throughput objective, TTFT ≤ 10 s) |
| Parameter budget | total ≤ 100B, bf16 | total ≤ 1B | total ≤ 100B, bf16 |
| Training budget (analytic) | $6 N_{\text{act}} D \le 1\times10^{23}$ FLOPs, $D \le 20$T tokens | $\le 3\times10^{20}$ FLOPs | same as A |
| Quality gate | not worse than the dense GQA baseline of the same active size by more than the ablation noise floor measured in the dossier (relative, see T3) | same | same |

*The serving hardware of A and C is an analytic target (the A100-80GB SXM datasheet row of `d2l.chip_specs()`); no run of this assignment uses it. The book's T3 node has 8× 40 GB (320 GB, 288 GB at 0.9) and could not hold scenario A's Qwen3-30B-A3B at 32 sessions × 128K (473 GB, 14.9 exercise 1).*

Training budgets are analytic constraints: the autograder checks $6 N_{\text{act}} D$ using `config_stats` active parameters and the student's declared `tokens` (the same $6ND$ as `sec_scaling-laws`, `eq_six_nd`; MoE and hybrid corrections are the student's to argue in the dossier).

### Tasks

**T1. The design (25 points).** Write `design.json`, a Hugging Face-style `config.json` (any `model_type` that `d2l.config_stats` supports, or a `layer_plan` the student registers), and `claims.json` with the student's *hand-derived* values. The design must satisfy all hard constraints of the chosen scenario:

1. `total_params ≤ budget` and $6N_{\text{act}}D$ within the training budget;
2. memory: weights + (KV cache + recurrent state) × concurrency × context ≤ 0.9 × HBM of the serving hardware (weights split across the 8 GPUs by tensor parallelism for A and C; one GPU for B), using `config_stats` per-token cache bytes and fixed state bytes;
3. latency: the 5.2 decode-step model $t(b,L)=\max(2 N_{\text{act}} b/P,\,(W + b L k_{KV})/\beta)$ with $P$ and $\beta$ from `d2l.chip_specs()` predicts the target tokens/s/user at the target concurrency (tensor-parallel communication is ignored for A and C and listed as a risk in the dossier);
4. `claims.json` must contain `total_params`, `active_params`, `cache_bytes_per_token`, `state_bytes` derived by hand from the config (with the formula used, as text). The autograder compares each claim to `config_stats(design)`: **relative difference ≤ 0.5%** per field (parameters), **exact** for `cache_bytes_per_token` and `state_bytes` when the design uses only `gqa`, `mla` or `gdn` layers (they are integer sums), ≤ 0.5% otherwise.

*Starter code* (`capstone/starter.py`, provided):

```python
Scenario = dataclass(name, hardware, concurrency, context, tokens_per_s_user, param_budget, train_flops_budget, ...)
def load_scenario(name: str) -> Scenario
def check_design(design: dict, claims: dict, scenario: Scenario) -> CheckReport   # runs 1-4 above; CheckReport.table() prints a pass/fail table
def predict_decode(design: dict, scenario: Scenario, batch: int, context: int) -> dict  # step time, tokens/s/user, binding resource ('compute'|'bandwidth'|'capacity')
def comparators(scenario: Scenario) -> pd.DataFrame      # config_stats rows of three released models near the scenario's size, from the 14.1 design table
```

**T2. One measured baseline (20 points).** On the build box (T1 path) or a T2 allocation, measure a *released* model that stands in for the baseline family at small scale (scenario A and C: `Qwen/Qwen3-1.7B` as the attention-only baseline and `Qwen/Qwen3.5-2B` as the hybrid; scenario B: `Qwen/Qwen3-0.6B` and `Qwen/Qwen3.5-0.8B`). Using `capstone.measure_decode(model_id, context, batch, steps)`:

1. peak memory versus context at three contexts, one per factor of 4, ending at the largest context that fits;
2. decode tokens/s at batch 1 and at the largest batch that fits, three repetitions each (report the median and the min–max range; the CUDA-graph or `torch.compile` setting is recorded in the log);
3. compare each with the prediction of `predict_decode` using the row of `d2l.chip_specs()` for the hardware that ran the measurement (the build box's 4090 row, or the A100 40 GB PCIe row that 5.1 adds for a T2 allocation) and the model's `config_stats`.

*Autograder:* the log must contain all six measurement cells; **measured peak memory within 15% of predicted** (weights + cache + state + the allocator workspace stated by `config_memory`), and **decode tokens/s within 25% of predicted** *or* the dossier contains a profile (14.6's profile cell or the PyTorch profiler) that localizes the gap to a named kernel category. (The 15% memory tolerance is the one of assignment A5; the 25% throughput tolerance is looser because batch-1 decode on a small model is launch-bound, which 5.2 shows.)

**T3. One rejected design, with evidence (25 points).** The student names a *specific alternative* to one axis of the T1 design, writes a pre-registered decision rule, runs a small-scale experiment with the chapter's building blocks, and rejects it, or states that the evidence cannot reject it.

- **Alternatives allowed** (each has a harness in the corresponding section, so the experiment is a config flag): MoE sparsity or granularity (14.2), GQA versus MLA at matched cache bytes (14.3), learned sink or output gate (14.4), top-$k$ block selection (14.5), attention fraction of a hybrid (`sec_hybrids` and 14.6), residual variant (14.7). The student may propose another, with the instructor's approval.
- **Pre-registration:** before any run, commit `rules.yaml` with the metric, the direction, the *threshold* below which the alternative is rejected, the number of seeds (≥ 3) and the T1 or T2 token budget. The run log records the SHA-256 of `rules.yaml`; the autograder checks that the hash in the log matches the file and that the file's commit precedes the results file's.
- **Measurement:** two metrics: *quality* (validation loss at a fixed token budget on the R2 data slice, and one task metric of the student's choice) and *cost* (the scenario's binding resource: cache bytes per token, decode step time from the 5.2 model, or training tokens/s). Uncertainty: `d2l.paired_bootstrap(scores_a, scores_b, cluster=...)` over seeds or per-document losses (1.5), returning a `d2l.BootstrapResult` (read `r.mean` and `r.ci`), and the baseline's seed-to-seed standard deviation.
- **Autograder:** (i) ≥ 3 seeds per arm, or a paired bootstrap interval; (ii) the verdict in `verdict.json` is one of `rejected`, `adopted`, `inconclusive` and is **consistent with the pre-registered rule applied to the interval** (an interval that straddles the threshold must be `inconclusive`); (iii) no verdict of `rejected` or `adopted` is allowed when the effect is within 2 × the baseline seed standard deviation.

**T4. The dossier (20 points).** `dossier.md`, ≤ 4 pages, with these headings (the autograder checks they are present; a human grades the content with the rubric below):

1. *Requirements:* the scenario's constraints restated in the student's words and which one binds (compute, bandwidth, capacity) according to `predict_decode`.
2. *Design table row:* the chosen design as one row of the 14.1 design table, next to the three comparators of `comparators(scenario)`.
3. *Capacity estimate:* total, active, cache and state bytes, and training FLOPs, each with its formula (cite the 14.1 equation labels).
4. *Experiment matrix:* every experiment the student considered, the uncertainty each would resolve, its cost in GPU-hours, and the order they would run in before committing a cluster. At least one experiment in the matrix must be *infeasible at small scale* and the dossier must say why (e.g., Muon-scale optimizer interaction, long-context data).
5. *Baseline and rejected design:* T2 and T3 results with their uncertainty.
6. *Quality gates and decision rule:* what must be true on the real run to continue, and the stop condition.
7. *Risks:* at least three, each with a fallback.

**Written analysis (10 points; answers ≤ 250 words each):**

1. Which resource binds in your design (compute, bandwidth or capacity), and what would change if the context doubled? Give the derivative of your binding term with respect to context from the formulas.
2. Name one design-table column in which your design is *worse* than a released comparator, and why you accept it.
3. Your T2 prediction missed by $x$%. Where does the error come from, and is it a modeling error or a measurement error?
4. In T3, what is the largest effect your experiment could have detected (use the seed standard deviation) and how does it compare with the effect size the original paper reports?
5. Which claim in a technical report for a model near your design did you *not* believe, and why? (Use the 14.1 schema: disclosed, in-config or inferred.)
6. What is the cheapest experiment that could overturn your decision?

**Rubric (T4 and written analysis, human-graded):** 0–4 on each of: requirement fidelity, quantitative consistency (every number traceable), appropriate skepticism of reports, quality of the experiment matrix, honesty about uncertainty. Deductions: any conclusion drawn from a difference within noise (−5), any number in the dossier without a formula or source (−2 each, up to −10).

### T1 path and hours

- **T1 path = scenario B**, entirely on the build box: `config_stats` and `predict_decode` are T0; T2 measures Qwen3-0.6B and Qwen3.5-0.8B (each ≤ 4 min); T3 runs R2-tiny variants (`d2l.GPT` width 256, 6 blocks, token budget calibrated to ≤ 3 min per run; 3 seeds × 2 arms ≤ 18 min, run as one job in the background).
- **Scenarios A and C** use the same code with the 8× A100-80GB analytic row of `chip_specs()`; only T2 changes: the baseline measurement runs at T2 (one A100 40 GB) with `Qwen/Qwen3-1.7B` and `Qwen/Qwen3.5-2B`. Fit (pilot arithmetic, Hub and `config.json` numbers): Qwen3-1.7B has 2,031,739,904 bf16 parameters (4.06 GB at most) and 114,688 B of KV per token, so one 32,768-token sequence takes 3.76 GB and about 8 sequences fit in 0.9 × 40 GB (about 33 at 8,192 tokens, 135 at 2,048); 32K is its native limit (40,960 positions), so memory never sets the largest context. Qwen3.5-2B caches 12,288 B per token (6 full-attention layers × 2 KV heads × head_dim 256) plus a fixed state, 3.2 GB at 262,144 tokens. Every context and batch cell of the protocol therefore runs on 40 GB; "the largest batch that fits" is read from the measured peak.
- **Estimated student time (an estimate to be calibrated in the first offering):** 3 h design and derivations, 2 h baseline, 4 h rejected design, 3 h dossier and analysis; 10–14 h in total.
- **Collaboration and honesty:** each student's `claims.json` is graded against the *student's own* `design.json`; two students with the same design must have distinct rejected designs.

### Failure modes the assignment is designed to expose

Using total instead of active parameters in the decode model (the autograder's `active_params` field catches it); forgetting that a hybrid's recurrent state is per sequence, not per token (14.6); treating a window layer's cache as growing; applying a learned-sparse-attention speedup at a context below the crossover (14.5); declaring victory on a 0.02-nat difference.

## Teaching sources

All of v5's chapter sources are kept; each is annotated with what to take from it.

- Sebastian Raschka, [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison): the diagrams-per-model style; take the "what changed relative to the previous model" framing for the per-row `axis_notes` of 14.1. The page predates 2026 releases; use it as the template for rows, not as a source of facts.
- Raschka, [From GPT-2 to gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the): the walk through sliding window, learned sinks, MoE and MXFP4 in one model; the reference for 14.4's sink variant and the `layer_types` of gpt-oss.
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook): architecture ablations at small scale and how to decide which to run; the model for 14.4/14.7's noise-check discipline and for 14.9's experiment matrix.
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the analytic cost-model style (derive, then measure); translate its attention and MoE arithmetic into the `config_stats` per-layer FLOPs and the 14.3/14.5 decode-intensity derivations. Do not adopt its JAX code.
- Hugging Face, [Mixture of Experts Explained](https://huggingface.co/blog/moe): vocabulary and the router/capacity picture for 14.2.
- Songlin Yang, [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/) and [flash-linear-attention](https://github.com/fla-org/flash-linear-attention): the WY/UT derivation and the Triton kernels behind 14.6's chunked form; `fla` is also the optional fast path in 14.6.
- Added by this blueprint: Stanford [CS336](https://cs336.stanford.edu/) (cited across the part): the architecture and MoE lectures and the scaling assignment as the style model for 14.2's fit and for the capstone's pre-registration; [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (v5's 14.9 source) for budget reasoning; the Transformers modeling files as the executable ground truth for every row: [`modeling_qwen3_5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), [`modeling_deepseek_v3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py), [`modeling_gpt_oss.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py), [`modeling_qwen3_moe.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) (all four returned HTTP 200 on 2026-10-28).

## Sections

### 14.1 Reading an architecture

- **Label:** `sec_lm_reading_architectures` · **Tier:** T0 (no reduced configuration needed) · **Budget:** ≤ 5 min on CPU; first run downloads ≈ 40 `config.json` files (≈ 0.3 MB) and ≈ 40 safetensors *headers* by HTTP range request (≈ 1 MB), then everything is read from the pinned snapshot `lm-arch-configs` and `lm-arch-headers` · **Peak memory:** < 2 GB · **Codex units:** 16.1, 16.3 (the dense reference case 16.2 is folded into the Qwen3-8B row; the design matrix 16.5 moves to 14.9)
- **Builds on (book):**
  - `sec_scaling-laws`: the $12Ld^2$ census, the non-embedding convention, `eq_six_nd`, `tab_modern-recipe`. 14.1 does **not** re-derive them; it makes `tab_modern-recipe` data, extends it to 2024–2026 and to MoE, MLA and hybrid layers.
  - `sec_moe`: `moe_accounting` (a manual census with a unit argument and a router term). `d2l.config_stats` is its generalization, and the DeepSeek-V3 row of `moe_accounting` (656B in experts, 23B active in experts) is the first regression test.
  - `sec_kv-cache`: `eq_kv-cache-bytes`. 14.1 adds the cache formulas for the other layer kinds.
  - `sec_hybrids`: `tab_hy-recipe` and the per-layer state formula `eq_ms-state-bytes`. 14.1 reuses the state formula and the "pay the cache per attention layer" rule.
  - `sec_read_write`: safetensors, the index file.
- **Builds on (this part):**
  - 1.1 (`sec_lm_hf_models`): configuration files, `d2l.config_memory`, the Qwen3.5-2B annotation (the field-to-section map of 1.1 is the seed of the axes table here). 14.1's cache and state formulas must agree with `config_memory` (test below). Imports `d2l.config_memory(config, seq_len, batch_size, dtype_bytes=2, *, state_bytes=4, ...)`, which returns a `dict` with keys `params, weights, kv_cache, recurrent_state, activations, logits, total`.
  - 5.2 (forward reference: the decode model consumes `ConfigStats`).
  - 1.2 (`sec_lm_d2l_checkpoint`): 1.2 subclasses the book's classes, which stay unchanged: `d2l.Qwen3Attention` (`head_dim`, `qk_norm`, `rope_theta`, `norm_eps`), `d2l.SwiGLU` (explicit hidden width) and `d2l.Qwen3GPT` (`num_kv_heads`, `tie_embeddings`). 1.1 owns `d2l.config_params(config, include_vision=False, include_mtp=False, by_group=False) -> int | dict[str, int]`: dense and hybrid configs only; a MoE config raises `NotImplementedError('use d2l.config_stats (14.1)')`; with `by_group=True` it returns per-group counts (`embed, attention, linear_attention, mlp, norm, lm_head, vision, mtp`) that sum to the total. **`d2l.config_stats` calls `config_params(config, include_vision=True, include_mtp=True, by_group=True)` for the dense and hybrid parts** (embedding, dense attention, linear attention, dense MLP, norm, head, vision, MTP) and maps its groups onto `params_by_group`; it computes the MoE parts (routed and shared experts, router, latent projections) and the MLA, DSA and lookup-table parts itself, and adds active parameters, FLOPs per token, cache and state bytes. The D1 formulas for kinds that `config_params` also handles are therefore a *cross-check* (test: equal to `config_params(by_group=True)`), not a second implementation. Fallback (one line): if a config has a layer kind `config_params` does not know, `config_stats` catches `NotImplementedError` and uses its own D1 formula for that config. Chapter 1 verified Qwen3.5-2B at LM 1,881,825,088, vision 331,416,576, MTP 60,828,160, total 2,274,069,824, matching the pre-check here.
  - 5.2 also exports `d2l.ServingModel(cfg, chip, ...)` with a `kv_bytes_override` hook for MLA or hybrid caches: **`ConfigStats` is the source feeding that hook** (`kv_bytes_override=stats.cache_bytes_per_token`, fixed state through `stats.state_bytes`; `ServingModel(cfg, chip, calib=None, dtype_bytes=2, kv_bytes=2, mem_util=0.9, kv_bytes_override=None)` is the 5.2 signature); 14.3, 14.6 and the Assignment pass `config_stats` outputs to it instead of recomputing cache bytes.
- **Exports (`#@save`):**
  - `d2l.config_stats(config, *, kv_bytes=2, state_bytes=4, groups=('text',), plan=None) -> d2l.ConfigStats` (reserved name). `config` is a `transformers.PretrainedConfig`, a `dict`, a path to a `config.json`, or a model id resolved through the pinned snapshot. Contract: one call returns total parameters, active parameters, FLOPs per token as an affine function of context, bytes of growing cache per token and bytes of fixed state per sequence, plus a per-layer table. Never downloads weights.
  - `d2l.ConfigStats` (dataclass): `total_params`, `active_params` (per token, embedding lookups excluded, output head included, lookup tables and MTP/draft modules excluded), `params_by_group` (dict: `embedding`, `head`, `attn`, `linear_mixer`, `dense_ffn`, `routed_experts`, `shared_experts`, `router`, `norm`, `lookup_tables`, `mtp`, `vision`), `flops_fixed` (= 2 × active matmul parameters), `flops_per_ctx_token` (attention-score and value-mix FLOPs per context token, summed over layers, windows and top-$k$ already applied through `flops_per_token(ctx)`), `cache_bytes_per_token` (grows with context), `cache_bytes_bounded` (sliding-window layers, saturates), `state_bytes` (constant per sequence), `layers` (a DataFrame), `notes` (list of strings: every assumption, unsupported field and inference the calculator made). Methods: `flops_per_token(ctx, mode='decode'|'prefill')`, `cache_bytes(ctx)`, `to_row()` (flat dict for the design table).
  - `d2l.LayerSpec` (dataclass): `mixer` (`'gqa'|'swa'|'mla'|'dsa'|'gdn'|'kda'|'lightning'|'csa2'`), `ffn` (`'dense'|'moe'|'latent_moe'`), and the numeric fields each needs; `d2l.register_arch(model_type)`: decorator registering `plan(cfg) -> list[LayerSpec]`.
  - `d2l.load_config(model_id, revision=None, offline=True) -> dict` and `d2l.safetensors_census(model_id, revision=None, groups=None) -> dict[str, dict]` (group name to `{'elements', 'dtypes', 'bytes'}` from the safetensors header by HTTP range request; handles sharded checkpoints through `model.safetensors.index.json`, and packed FP4/MXFP4 tensors through a documented per-format unpack factor).
  - `d2l.design_table(registry='lm-arch-registry', core_only=True) -> pd.DataFrame` and `d2l.ReportSchema` with `d2l.check_report(path)` (schema validation, below).
  - Data: `lm-arch-configs`, `lm-arch-registry`, `lm-arch-expected`, `lm-arch-design-table`, `lm-arch-headers` (formats in the chapter overview).
- **Models and data:** the core rows (marked † in the fact sheet): `Qwen/Qwen3-8B`, `Qwen/Qwen3-30B-A3B`, `openai/gpt-oss-120b`, `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4.1-Flash`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-35B-A3B`, `moonshotai/Kimi-K3`, plus (extended rows) Qwen3-235B-A22B, gpt-oss-20b, GLM-4.5, GLM-5, GLM-5.3-Flash, MiniMax-M1/M2/M3, DeepSeek-V2, V3.2, V4-Pro, Kimi-K2, Kimi-Linear, Qwen3.6/3.8 variants. Licenses (Hub): Apache-2.0 (Qwen3.x dense and the 35B-A3B MoE, gpt-oss), MIT (DeepSeek V3.2 onwards, V4 family, GLM, Kimi-Linear), "other" (Kimi K2/K3, Qwen3.8-2.4T and Flash-Next, MiniMax): only configuration files and safetensors headers are read, no weights, so no license acceptance is needed.
- **Packages:** `huggingface_hub>=0.30`, `transformers>=5.0` (for `PretrainedConfig` of `qwen3_5` and later types; the calculator itself reads plain dicts and does not import model classes), `pandas`, `matplotlib`, `pyyaml`.

#### Narrative arc

1. **The question.** *What is actually in a model's config, and what does it cost?* The opening cell loads Qwen3-8B's and DeepSeek-V4.1-Flash's `config.json`, flattens them, and prints (a) the number of keys (pre-check: 26 for Qwen3-8B; 87 for V4.1-Flash once `text_config`, `vision_config` and `quantization_config` are flattened, 59 in `text_config` alone, of which 41 do not occur in Qwen3-8B's config) and those 41 keys (`compress_ratios`, `hc_mult`, `engram_*`, …), and (b) three different parameter counts for the same repository: the report's backbone count (552B), the safetensors total (763,205,315,794) and the number a first-year student would compute from `hidden_size` and `num_hidden_layers` with $12Ld^2$ (to be printed; it is wrong by a large factor for a MoE). The symptom to print: *none of the three numbers agree, and each is right for a different question.*
2. **The axes of a decoder** (`subsec_lm_arch_axes`). Table `tab_lm_arch_axes`: axis → fields in the configuration → section that teaches it. Rows: token mixing (`layer_types`, `full_attention_interval`, `sliding_window`, `index_topk`, `compress_ratios`; 14.4–14.6), KV layout (`num_key_value_heads`, `kv_lora_rank`, `kv_source_layer_ids`; 14.3), feed-forward (`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `routed_expert_hidden_size`, `scoring_func`, `n_group`/`topk_group`; 14.2), lookup memory (`engram_*`, `ngram_*`, `ple_*`; 14.2 sub-block), residual paths and norms (`hc_mult`, `hc_count`, `mhc`, `attn_res_block_size`, `use_qk_norm`, `rms_norm_eps`; 14.7, 4.3), positions (`rope_theta`, `partial_rotary_factor`, `rope_scaling`, NoPE layers; 14.4, 4.6), output heads (`num_nextn_predict_layers`, `mtp_*`, `dspark_*`; 14.8, 12.3), tokenization (`vocab_size`, tied embeddings; 3.6, 14.8), numerics (`quantization_config`, `expert_dtype`, `hidden_act` such as `situ`, `swiglu_limit`; 4.4, 11.x). Print the table from a data file so the editor can extend it. Point out that 1.1 mapped one model's fields; here the map is the table's column set.
3. **From configuration to numbers** (`subsec_lm_arch_stats`). The `LayerSpec` abstraction: one record per layer, a mixer kind and an FFN kind; `plan(cfg)` is the *only* model-specific code. Walk through the three plans needed for Qwen3-8B (36 identical `gqa` + `dense`), DeepSeek-V3 (3 `mla` + `dense`, then 58 `mla` + `moe`) and Qwen3.5-2B (`layer_types` → `gdn` or `gqa`-with-gate, all `dense`). Equations `eq_lm_arch_params` through `eq_lm_arch_cache` (see Derivations). Show the per-layer DataFrame for one model.
4. **Checking against the safetensors index** (`subsec_lm_arch_census`). Run the census for four families and print the difference for each; the exact-match and residual-explained results are the section's first headline (see Experiment). Explain what the three kinds of residual are: *named extra groups* (vision tower, MTP head, draft module, lookup tables), *packed dtypes* (an element count versus a byte count), and *small tensors the calculator ignored* (router biases, correction biases).
5. **The design table** (`subsec_lm_arch_table`). One row per model; columns: release date, family, total and active parameters, active fraction, `layers`, attention mix (counts of each mixer kind), cache bytes per token, state bytes, expert count and top-$k$ and shared count, score function, dense-prefix layers, position scheme, residual scheme (`hc`/`attnres`/plain), heads shipped (MTP/draft), weight dtype. Plot `fig_lm_arch_axes_timeline`: for four axes a small-multiples scatter over release date (cache bytes per token on a log axis; active fraction; fraction of non-softmax layers; extra heads). Read the plot for *moves*, not trends: the point is which axis each release moves.
6. **The DeepSeek line as single-axis changes** (`subsec_lm_arch_deepseek`). Table `tab_lm_arch_deepseek` built *from the configs*, one row per release, with the "axis changed" column filled in by a diff of two consecutive configs: V2 → V3 (`scoring_func`, `topk_method`, `num_nextn_predict_layers`, `quantization_config`, `n_routed_experts 160 → 256`, `first_k_dense_replace 1 → 3`), V3 → V3.2 (`index_*` appear), V3.2 → V4-Flash (`compress_ratios`, `hc_mult`, `o_lora_rank`, `num_key_value_heads 1`, `head_dim 512`, hidden size 7168 → 4096, 61 → 43 layers; this release is the one v5's line skips), V4-Flash → V4.1-Flash (`kv_source_layer_ids`, `index_source_layer_ids`, `engram_*`, `dspark_*`, `candidate_*`, 43 → 40 layers, 3 `compress_ratios` regimes). The key-diff printer is a ten-line cell; its output is the table. Each line points forward to the section that teaches the changed field. State the "single-axis" claim honestly: V4-Flash and V4.1-Flash each change several axes at once, and the table shows it.
7. **Reading a report** (`subsec_lm_arch_reports`). The schema (see Figures and tables, `tab_lm_arch_schema`), then two worked examples: (a) DeepSeek-V4.1-Flash §2 and §4.2.1: fill the schema, tracing each claim to a table, figure or ablation; (b) Kimi K3 §2 and §3.2. For each report field, label it `disclosed` (stated in the report), `in-config` (read from the configuration file) or `inferred` (neither; the reader's guess). *Reconcile the report with the configuration file*: for V4.1-Flash the report says 40 layers, hidden 5120, 64 heads, 384 routed experts, top-6, expert width 2304, window 128, indexer 32 heads × 128, top-512, mHC factor 4 with 20 Sinkhorn iterations (all in the config and equal), and the report says 890 bytes of global KV cache per token, a number the config does not contain (reconstructed in Derivation D4). Where a report and a config disagree, the notebook prints the disagreement and the section records it in `axis_notes`; the pre-check found none for V4.1-Flash (to be confirmed by the author).
8. **Summary of the method and "How to extend this chapter"** (one paragraph, with a pointer to the overview's full manual).

#### Derivations

*Notation:* $d$ hidden size, $L$ layers, $V$ vocabulary, $H$ query heads, $H_{kv}$ KV heads, $d_h$ head dimension, $n$ context length, $b_{kv}$ bytes per cached element. For MLA: $r$ latent rank, $r_q$ query rank, $d_n$, $d_r$, $d_v$ the nope, rope and value head dimensions. For MoE: $E$ routed experts, $k$ active, $E_s$ shared, $w$ expert width, $\ell$ latent width (equal to $d$ unless LatentMoE). For linear layers: $H_v$ value heads, $d_k$, $d_v$ head dimensions, $\kappa$ conv kernel.

**D1. Parameters per layer (`eq_lm_arch_params`).** Statement: for each layer kind the parameter count is a closed form in the fields; the total is the sum plus embeddings, and the *active* count replaces $E$ by $k$ in the routed term.

Assumptions: SwiGLU experts (three matrices); biases and norms counted when the config says so; embedding counted once when `tie_word_embeddings`.

Steps:
1. GQA with optional output gate: $P_{\text{gqa}} = d\,H d_h (1+\gamma) + 2\,d\,H_{kv} d_h + H d_h\, d + (\text{bias}: H d_h + 2 H_{kv} d_h + d) + (\text{QK-norm}: 2 d_h) + (\text{sinks}: H)$, with $\gamma=1$ if `attn_output_gate` (the query projection is doubled; the Qwen3.5-2B shape is `q_proj [4096, 2048]` = $8\cdot256\cdot2$, see the header).
2. MLA with query LoRA: $P_{\text{mla}} = d r_q + r_q + r_q H (d_n + d_r) + d (r + d_r) + r + r H (d_n + d_v) + H d_v d$ (the two `+ r_q`, `+ r` are the LoRA norms). Without query LoRA (`q_lora_rank` null, Kimi-Linear) the first three terms become $d H (d_n + d_r)$.
3. Gated DeltaNet (Qwen3.5 layout): with $K = H_k d_k$, $V = H_v d_v$, $P_{\text{gdn}} = d(2K + V) + dV + 2 d H_v + (2K+V)\kappa + 2 H_v + d_v + V d$ (`in_proj_qkv`, `in_proj_z`, `in_proj_b/a`, depthwise conv, `A_log`, `dt_bias`, the gated RMSNorm weight, `out_proj`). *Pre-check:* this formula reproduces the Qwen3.5-2B text group exactly (1,881,825,088).
4. KDA (Kimi K3 layout): $P_{\text{kda}} = 3 d H d_k + d H d_v + H d_v d + d f + f H d_k + d H + 3 H d_k \kappa + H d_k + 2 d_k$ with $f=128$ the rank of the low-rank forget gate (`f_a_proj [128, d]`, `f_b_proj [H d_k, 128]`); the rank is **not in the config** (inferred from the header), so `notes` records it.
5. MoE FFN: $P_{\text{moe}} = E\cdot 3 d w + E_s\cdot 3 d w_s + d E\ (+E\ \text{bias})$; active $A_{\text{moe}} = k\cdot 3 d w + E_s \cdot 3 d w_s + d E$. LatentMoE replaces $d$ by $\ell$ in the routed term and adds $2 d\ell$ for the down- and up-projections: $P_{\text{lat}} = E\cdot 3\ell w + 2 d \ell + E_s \cdot 3 d w_s + d E$.
6. Totals: $N = V d\,(1+[\text{untied}]) + \sum_\ell (P^{\text{mix}}_\ell + P^{\text{ffn}}_\ell + 2d) + d$, $N_{\text{act}} = V d + \sum_\ell (P^{\text{mix}}_\ell + A^{\text{ffn}}_\ell + 2d) + d$.

*Check cell:* for each of seven checkpoints compare `config_stats(...).total_params` with `safetensors_census`. **Pre-check results (blueprint author, 2026-10-28):** Qwen3-8B 8,190,735,360, Qwen3-30B-A3B 30,532,122,624, Qwen3-235B-A22B 235,093,634,560 and DeepSeek-V2 235,741,434,880 match the header **to the parameter** (difference 0). gpt-oss-20b and -120b differ by 768 and 4,608, which are exactly $24\times32$ and $36\times128$ router biases (not in the base formula; add `router_bias` to the plan). Kimi-K2 differs by 26,456, of which 23,040 = $60\times384$ is the router correction bias; the remaining 3,416 are to be traced by the author. DeepSeek-V3 derived 671.026B versus the API's 684.49B: the 13.46B difference is the MTP module (`num_nextn_predict_layers 1`), reported separately in the `mtp` group. Qwen3.5-2B: text 1,881,825,088 (exact), visual 331,416,576, `mtp.*` 60,828,160, sum 2,274,069,824 = the API total. Active parameters: Qwen3-30B-A3B 3.042B + embedding 0.311B ≈ the "3.3B" of its model card; DeepSeek-V2 20.85B ≈ "21B"; gpt-oss-20b 3.608B ≈ "3.6B"; gpt-oss-120b 5.133B ≈ "5.1B"; Kimi-K2 31.69B ≈ "32B"; DeepSeek-V3 36.63B ≈ "37B" (all against model cards; the cards are the only sources of those numbers).

**D2. FLOPs per token (`eq_lm_arch_flops`).** Statement: one decoded token at context $n$ costs $F(n) = 2 N_{\text{mat}} + \sum_\ell c_\ell(n)$, with $N_{\text{mat}}$ the active matmul parameters (active parameters minus norms, router counted) and $c_\ell(n)$ the layer's attention or state work.

Steps:
1. Matmul FLOPs: $2$ per parameter per token (`sec_scaling-laws`); embedding lookup free; MoE counts the $k$ selected experts.
2. Softmax attention: $c = 2 H n d_{qk} + 2 H n d_v$ with $n \to \min(n, w)$ for windows. For GQA $d_{qk}=d_v=d_h$ so $c = 4 H n d_h$.
3. MLA, *expanded* (prefill): $c = 2Hn(d_n + d_r) + 2Hn d_v$. MLA, *absorbed* (decode, 14.3): $c = 2Hn(r + d_r) + 2Hn r$ (the latent plays both K and V, shared by all heads).
4. DSA (14.5): indexer $2 H_I d_I n$ plus main attention over $\min(n, k)$ selected tokens.
5. Linear layer: constant in $n$: $c = 6 H_v d_k d_v$ (read $S^\top k$, write the rank-one correction, read $S^\top q$) $+ 2\kappa (2K+V)$ for the conv.
6. The calculator returns $F(n) = F_0 + n\,F_1$ with $F_1$ summed over softmax layers (windowed layers saturate), which is what 5.2's decode model needs.

*Check:* the Qwen3-8B case reproduces $4Hnd_h L$ against a hand count; for $n \ll 6d$ the ratio $n F_1 / F_0$ is the book's $n/(6d)$ (`sec_scaling-laws`). No safetensors reference exists for FLOPs; the check is the identity with the book's formula and a PyTorch `FlopCounterMode` run of a *random-weight small instance* of Qwen3-8B's layer plan at width 256 (within 5%; the counter and the formula differ by softmax and norm FLOPs, which is the stated tolerance).

**D3. Cache and state (`eq_lm_arch_cache`).** Statement: bytes per token of growing cache and bytes of fixed state are sums over layers with kind-specific terms.

Steps:
1. GQA/MHA/MQA: $2 H_{kv} d_h b_{kv}$ per token per global layer (`eq_kv-cache-bytes` with $b=1$, $n=1$); window layers: the same times $\min(n, w)$ (bounded; reported in `cache_bytes_bounded`).
2. MLA: $(r + d_r) b_{kv}$ per token per layer, independent of $H$. (With `mla_use_nope`, Kimi's weight shapes still contain the 64 rope slots: `kv_a_proj_with_mqa` is `[576, 7168]`, so the cache is 576 elements per token in the implementation; the calculator reads the shape from the plan, flagged in `notes`.)
3. DSA: add the indexer key $d_I\, b_{idx}$ per token per *indexer-owning* layer.
4. Gated DeltaNet / KDA: fixed state $H_v d_k d_v\, b_{state} + (2K+V)(\kappa-1)\, b_{kv}$ per layer per sequence (fp32 state per `mamba_ssm_dtype`); no growth.
5. CSA2 (V4.1-Flash, derived in D4): per token $\sum_{\text{source layers}} e_{\text{main}}/c_i + e_{\text{idx}}/c_i$ for compression ratio $c_i$, plus a bounded window term $2\cdot n_{\text{win}}\ldots$ (the sliding-window KV is bounded and excluded from the "global" cache the report quotes).
6. Pre-check values (bf16, from the configs, blueprint author): Qwen3-8B 147,456 B/token; Qwen3-30B-A3B 98,304; Qwen3-235B-A22B 192,512; GLM-4.5 376,832; MiniMax-M2 253,952; DeepSeek-V2 69,120; DeepSeek-V3 and Kimi-K2 70,272; GLM-5 89,856; gpt-oss-120b 36,864 B/token on 18 full layers plus a bounded 4,718,592 B for the 18 window layers; Qwen3.5-2B 12,288 B/token plus fixed 19,537,920 B (18 layers); Qwen3.5-35B-A3B 20,480 B/token plus 64,389,120 B; Qwen3.6-27B 65,536 B/token plus 153,944,064 B; Qwen3.8-2.4T-A95B 94,208 B/token plus 587,292,672 B; Kimi K3 27,648 B/token on its 24 MLA layers (107,136 B/token if all 93 layers were MLA) plus 449,372,160 B fixed on 69 KDA layers (the same arithmetic as `fig_hy-cache`). The notebook must reproduce each to the byte.

*Check cell:* `d2l.config_memory` agreement: for every row, `config_stats(c).cache_bytes_per_token * n * B` equals `config_memory(c, n, B)['kv_cache']` for dense rows exactly, and `state_bytes * B` equals `config_memory(...)['recurrent_state']` for hybrids within 0.1% (the two functions may differ in how they count conv state; a difference is a bug report to 1.1's author, to be listed in the notebook).

**D4. Reconstructing a report's cache number (`eq_lm_arch_v41`).** Statement: with the layer plan read from `compress_ratios`, `kv_source_layer_ids` and the report's byte layout, the global KV cache of DeepSeek-V4.1-Flash is 890 bytes per token, the figure in the report's abstract.

Assumptions (marked): the main KV entry is a 512-channel latent stored as E2M1 with one E4M3 scale per 16 channels (report §2.4.4: "E2M1 with one E4M3 scale per 16 channels") → $512/2 + 512/16 = 288$ bytes. The indexer key is a 128-channel vector in MXFP4 (E2M1 plus an E8M0 scale per 32 channels, an OCP format) → $128/2 + 128/32 = 68$ bytes; *this layout is the blueprint's inference* (the report says only that V4 already uses FP4 indexer keys and that the format is MXFP4); the sliding-window cache is bounded and excluded ("global" cache, report abstract); only layers that *own* main KV (`kv_source_layer_ids`) store anything; reuse and reindex layers store nothing.

Steps:
1. From `compress_ratios[0:40] = [0,0] + [2]*18 + [1]*20`: layers 1–2 are window-only (ratio 0, no global KV); layers 3–20 (the causal encoder) have ratio 2; layers 21–40 (the decoder) ratio 1.
2. `kv_source_layer_ids = [2, 8, 14, 20]` (0-indexed): three encoder layers at ratio 2 (each stores $1/2$ main entry per token) and one decoder layer at ratio 1.
3. Entries per token: $3\times\tfrac12 + 1\times1 = 2.5$.
4. Bytes per token: $2.5 \times (288 + 68) = 890$. ✔ (main: 720; indexer: 170.)

*Check cell:* the calculator, run with `plan=csa2_plan(cfg)` and these two byte constants as explicit `overrides`, returns 890. If the author finds a different indexer layout reproduces 890 as well, both are listed in `notes`. The same plan with `compress_ratios` of V4-Flash (alternating 4 and 128 on all layers) and FP8 entries gives V4's figure; the report's Figure 1(b) says V4.1-Flash is "roughly 1/4" of V4-Flash, a second check (to be measured; the V4-Flash byte layout is not read here, the authors should fetch the V4 report §2 or mark (unverified)).

**D5 (smaller): prefill versus decode active parameters.** The report states 8B active in prefill and 16B in decode because the decoder half reads encoder-produced KV and need not run in prefill (CED). With the plan, prefill active = encoder layers + shared pieces; the calculator exposes `mode='prefill'` for CED plans. *Pre-check (rough, attention parameter counts approximated):* experts alone give $40\times7\times3\times5120\times2304 = 9.9$B active in the FFNs for decode; to be measured to within 5% of 16B after attention is added. This is an *open-ended* check, not a regression test.

#### From-scratch implementation

Components (≈ 150 lines of calculator + 60 lines of census, with plans kept ≤ 30 lines each):

```python
@dataclass
class LayerSpec:
    mixer: str; ffn: str
    # mixer fields (only those the kind needs)
    d: int; H: int = 0; Hkv: int = 0; dh: int = 0; window: int | None = None
    gate: bool = False; qk_norm: bool = False; bias: bool = False; sinks: bool = False
    r: int = 0; rq: int | None = None; dn: int = 0; dr: int = 0; dv: int = 0     # MLA
    Hv: int = 0; dk: int = 0; dvv: int = 0; kappa: int = 0; Hk: int = 0; f: int = 0  # linear
    topk: int | None = None; HI: int = 0; dI: int = 0                              # DSA
    cache_ratio: float | None = None; kv_entry_bytes: float | None = None; owns_kv: bool = True  # CSA2
    # ffn fields
    I: int = 0; E: int = 0; k: int = 0; w: int = 0; Es: int = 0; ws: int = 0; ell: int = 0; router_bias: bool = False
def register_arch(model_type): ...
def config_stats(config, *, kv_bytes=2, state_bytes=4, groups=('text',), plan=None) -> ConfigStats
def safetensors_census(model_id, revision=None, groups=None) -> dict   # HTTP range: first 8 bytes = header length, then the JSON header
def design_table(registry='lm-arch-registry', core_only=True) -> pd.DataFrame
```

Registered plans: `qwen3`, `qwen3_moe`, `deepseek_v2`, `deepseek_v3`, `deepseek_v32` (adds `dsa`), `kimi_k2` (same as V3), `gpt_oss`, `glm4_moe`, `minimax_m2`, `qwen3_next`, `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe_text`, `kimi_linear`, `kimi_k3` (`latent_moe`, `kda`+`mla`), `deepseek_v4` and `deepseek_v41` (`csa2`), `glm_moe_dsa`. A plan that sees an unknown field containing no known mapping appends a string to `notes` and does not guess.

Numerical tests (seeded where random; tolerances stated):

| Test | Reference | Tolerance |
|---|---|---|
| Total parameters, 4 dense/MoE/MLA models | safetensors header elements | exactly equal |
| gpt-oss, K2 residuals | itemized (router/correction biases) | exactly equal after `router_bias` and `e_score_correction_bias` are added to the plans |
| Qwen3.5-2B text group | header elements with prefix `model.language_model.` | exactly equal |
| Kimi K3 total | Hub API 2,779,931,837,184 | within 0.5% (pre-check: derived 2,777.4B, −0.09%); active 102.1B vs report 104B: within 3% (pre-check −1.8%; embeddings and unmodelled gate bias terms are the residual) |
| V4.1-Flash routed experts | 40 × 384 × 3 × 5120 × 2304 = 543.6B | exact |
| V4.1-Flash lookup tables | `engram.embed.weight` shapes `[384016682, 256]` and `[384006168, 256]` from the header: 196.6B elements | exact |
| Cache bytes per token | listed in D3 | exactly equal |
| Cache bytes of V4.1-Flash | 890 (report) | exactly equal under the stated layout assumptions |
| `config_stats` vs `config_memory` | 1.1 | exact (dense), 0.1% (hybrid) |
| `FlopCounterMode` on a width-256 instance | D2 | within 5% |

#### In production

- `transformers.AutoConfig` (≥ 5.0) parses configs; `AutoModel.from_config(config)` on the `meta` device (`torch.device('meta')`) gives a *parameter-shape oracle* that needs no weights: summing `p.numel()` over `model.named_parameters()` is a second, independent census, and the notebook compares it with the calculator and the header (it exists for dense, MoE, `deepseek_v3`, `qwen3_5`, and `gpt_oss`; for `deepseek_v41` and `kimi_k3` the modeling code is `trust_remote_code` and is **not** executed: the notebook uses the header only, consistent with 1.1's rule).
- `huggingface_hub.HfApi().model_info(id, files_metadata=True)` for `safetensors.parameters` and `createdAt`; `huggingface_hub.get_safetensors_metadata(id)` for headers (the notebook uses range requests directly to show how it works, then cross-checks against the library for one model).
- The from-scratch calculator is the oracle for `d2l.config_memory` (1.1) and for the decode-time model of 5.2 and 13.x; the capstone's autograder imports it.

#### Headline experiment

- **Hypothesis:** the closed forms D1–D3 reproduce released checkpoints' parameter counts and cache bytes to the parameter or the byte for the model families that the book's own layers cover, and to a *named, itemized* residual otherwise.
- **Setup:** CPU only; 18 extended-or-core models; no training. Seeds: only the random-weight FLOP-counter check uses seed 0.
- **Procedure:** (1) derive per model; (2) compare with the header census at the group level; (3) print one table of `derived`, `header`, `difference`, `explained by`; (4) compare derived cache bytes with `config_memory` and with report values in `lm-arch-expected`.
- **Metrics:** relative error per model and per group; list of unexplained residuals (target: empty for the 4 exact families, one entry otherwise).
- **Plot specification:** `fig_lm_arch_census` — horizontal bar chart, one bar group per model, bars for backbone / experts / lookup tables / vision / MTP parameters in billions on a log x-axis, a marker at the report's headline total; title "Where do the parameters live?". Second panel: cache bytes per token (log x) split into growing and bounded, with fixed state shown as a separate hatched bar (different unit, labeled "per sequence, not per token").
- **Expected outcome:** exact agreement for dense/MoE/MLA (pre-check: confirmed for four models); residuals itemized for the rest (pre-check). Nothing here is an empirical effect, so no noise check applies; the only random element is seeded. The published headline numbers (report totals and "active" values) are used as *expected* values with a stated tolerance, never to tune the calculator.
- **Artifacts committed:** `lm-arch-design-table`, `lm-arch-headers`, `lm-arch-expected`, `lm-arch-configs`.

#### Figures and tables

| Label | Content | Cell |
|---|---|---|
| `tab_lm_arch_axes` | axis → config fields → section | `axes.yaml` rendered |
| `tab_lm_design_table` | core rows × axis columns (generated) | `d2l.design_table().style` |
| `fig_lm_arch_axes_timeline` | four small multiples over release date | `plot_timeline(df)` |
| `fig_lm_arch_census` | parameter groups and cache bytes (above) | `plot_census(df)` |
| `tab_lm_arch_deepseek` | DeepSeek line, key diffs between consecutive configs | `diff_configs(a, b)` |
| `tab_lm_arch_schema` | the report schema: `model`, `report`, `architecture` (each field tagged disclosed / in-config / inferred), `training` (tokens, stages, optimizer, precision), `systems` (parallelism, hardware, MFU if disclosed), `claims` (claim, evidence type: ablation / final-model-only / none, baseline, resource budget), `reconciliation` (report vs config disagreements) | `ReportSchema` |

#### Pitfalls and misconceptions

- "Parameters" has at least five meanings: total, active, backbone (the report's number), checkpoint elements (the Hub's), and bytes. Always say which.
- The Hub API counts packed FP4/MXFP4 elements as *elements* (observed: gpt-oss `U8` total 19.1B equals the element count of the experts, not bytes), but a per-tensor header gives the *packed* shape; the census must unpack by a per-format factor. A formula that is "off by 2×" is almost always this.
- Lookup tables (Engram, n-gram embeddings) are parameters that cost memory and lookup traffic but ≈ 0 matmul FLOPs: they inflate total but not active. DeepSeek-V4.1-Flash's checkpoint holds 196.6B of them (inferred from `engram.embed.weight` shapes) in addition to the 552B backbone.
- `mla_use_nope` does not remove the rope slots from the weights; read shapes, not flag names.
- `num_key_value_heads` is meaningless for MLA (it reads 128 in DeepSeek-V3) and 1 for V4's single-KV-head layers; the cache comes from `kv_lora_rank + qk_rope_head_dim` or from `head_dim`, depending on the type.
- A config in the repository can lag the report (or the weights): print disagreements, do not hide them.
- Do not infer a mechanism from a field name: `hc_mult` is the number of residual streams, `compress_ratios` is per-layer and has *extra entries beyond the layer count* for MTP slots (43 entries for 40 layers).

#### Summary (draft bullets)

- A configuration file is a compact program for a cost model: a layer plan, not a list of hyperparameters.
- Total, active, FLOPs and cache are four different cost axes; each has a closed form per layer kind (D1–D3), and the sum reproduces released checkpoints exactly where the book's layers cover them.
- Residuals are findings: vision towers, MTP heads, router biases, lookup tables and packed dtypes explain every gap seen so far.
- The design table is data, regenerated from configs; each later section of the chapter is one of its columns.
- A technical report is a set of claims with different evidence: separate what is disclosed, what is in the config and what you inferred.
- The DeepSeek line is a sequence of releases that each change several axes, not one.

#### Exercises

*Check (answer sketches given).*

1. **Four numbers.** DeepSeek-V4.1-Flash has 552B backbone parameters, 763B safetensors elements and 16B active parameters in decode. Which of the three is memory-relevant for serving, which for FLOPs, and which for the question "how big is the model?" *Sketch:* serving memory uses elements × bytes (the 763B includes 196.6B Engram elements in FP8 that sit in host or HBM memory), FLOPs use active (16B), and the headline "552B" excludes lookup tables and the vision tower; the third question has no single answer.
2. **Which layers store cache?** In `layer_types` of Qwen3.5-35B-A3B (30 linear + 10 full) and in the 24 full-attention layers of Kimi K3, which layers contribute to the growing cache, which to the fixed state? Give each model's growing bytes per token. *Sketch:* only full layers grow; 20,480 B/token and 27,648 B/token respectively (D3 values); K3 would need 107,136 B/token if all 93 layers were MLA.

*Derive (answer sketches given).*

3. **Latent MoE parameters.** Using the Kimi K3 fields (`routed_expert_hidden_size 3584`, `moe_intermediate_size 3072`, 896 experts), derive the routed expert parameters per layer with and without the latent projection (at $\ell = d = 7168$) and the extra parameters of $W^\downarrow, W^\uparrow$. *Sketch:* $896\cdot3\cdot3584\cdot3072 = 29.59$B per layer versus $896\cdot3\cdot7168\cdot3072 = 59.2$B at full width; the projections add $2\cdot7168\cdot3584 = 51.4$M. Latent width halves expert parameters (and all-to-all bytes) for 0.17% extra.
4. **Cache of a CED model.** Re-derive 890 B/token if the indexer keys were stored in FP8 (1 byte per element plus no scale) instead; what would the report's number have been? *Sketch:* indexer 128 B instead of 68 B → $2.5 \times (288+128) = 1040$ B/token.

*Measure (predict before running).*

5. **The residual census.** Add a new row of your choice to `models.yaml` (a model not in the table: one released after 2026-10-01, or an older one such as Mixtral-8x7B), run `config_stats` and the census, and *predict* the residual before running. Report the explained and unexplained parts.
6. **A model that moves two axes.** Pick Qwen3.8-Flash-Next (config-only row): which of its fields can `config_stats` not account for (`hc_count`, `ngram_*`, `ple_*`, `indexer_*`)? Add the missing plan terms for the ones that have closed forms, state which remain `notes`, and compare the derived total with the Hub's 179,999,981,459.

*Open.*

7. **Schema stress test.** Fill the report schema for a report of your choice that is *not* in the chapter (for example the Qwen3 report or the Llama 3 paper) and find one headline claim the paper does not support with an ablation. Write it up in ≤ 200 words in the `claims` block.

#### Further reading

- Papers (all in v5): [Qwen3](https://arxiv.org/abs/2505.09388) (an open, dense+MoE family with explicit configuration table); [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (MLA, DeepSeekMoE, and the KV-cache comparison the report builds on); [DeepSeek-V3](https://arxiv.org/abs/2412.19437); [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556); [DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1) (§2 architecture, §4.2.1 model setup: the source of the report-vs-config reconciliation); [Llama 3](https://arxiv.org/abs/2407.21783) (a dense report-only row); [gpt-oss model card](https://arxiv.org/abs/2508.10925).
- *New (verified):* [DeepSeek-V4](https://arxiv.org/abs/2606.19348) (title confirmed 2026-10-28: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"), the release that introduced `compress_ratios`, `hc_mult` and `num_hash_layers`; [Kimi K3](https://arxiv.org/html/2607.24653v1) (v5; §2 architecture; abstract states 2.8T total and 104B active).
- Teaching: Raschka's architecture comparison; [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook).
- Code and docs: [Transformers configuration docs](https://huggingface.co/docs/transformers/main_classes/configuration); [safetensors metadata parsing](https://huggingface.co/docs/safetensors/metadata_parsing) (the range-request recipe; HTTP 200); [Hub API `models/<id>`](https://huggingface.co/api/models/Qwen/Qwen3-8B) for `safetensors.parameters`.

#### Risks and open decisions

| Risk | Fallback |
|---|---|
| A Hub repository is renamed, gated or removed | `lm-arch-configs` pins snapshots and headers; rows keep working offline; the registry marks the row `status: archived` |
| `trust_remote_code` model types (Kimi K3, V4.1-Flash) cannot be instantiated on `meta` | header-only census (already the primary path) |
| Packed-dtype unpacking factors differ per format (MXFP4 `U8`, compressed-tensors `weight_packed`, FP4 stored one per `I8` byte) | per-format table in the code, one test per format; unresolved formats go into `notes` and are excluded from the exact-match set |
| The V4.1-Flash indexer byte layout (68 B) is inferred | the notebook states it as an assumption; if the V4 report (arXiv 2606.19348) contradicts it, change the constant and the D4 numbers, keep the structure |
| `config_memory` (1.1) and `config_stats` disagree | 14.1 files the disagreement; `config_stats` is authoritative for per-layer-kind accounting, `config_memory` for the activation workspace |
| Qwen3.8-Flash-Next has no technical report located | config-only row; `expected` empty; flagged in the table with a dagger |
| The design table grows past what one figure can show | `core: true` keeps ≈ 12 rows; extended rows live in the CSV and in the exercises |

### 14.2 Mixture-of-experts design at scale

- **Label:** `sec_lm_moe_design` · **Tier:** T2 (reduced T1 configuration: `cfg.scale='t1'`: `d2l.GPT` width 256, 6 blocks, a 5-point grid, two seeds, `allenai/OLMoE-1B-7B-0924` for routing analysis) · **Budget:** T2 ≈ 2.5–3 h on 1× A100 40 GB for the sweep (17 runs capped at ≈ 8 min each by token budget; calibrated in cell 3 from measured tokens/s, which absorbs the 40 GB part's lower memory bandwidth) plus ≤ 20 min on a T2 pair for routing analysis (Qwen3-30B-A3B, bf16, 61.06 GB of weights split over the two GPUs with `device_map='auto'`, about 30.5 GB each, batch 4 × 1024 tokens; one 40 GB GPU cannot hold the weights; readers without a pair analyse the committed traces); T1 ≤ 15 min total (5 configurations × 2 seeds at ≤ 90 s, plus ≤ 4 min of OLMoE routing at 13.8 GB weights, sequences of 512 tokens, batch 4) · **Peak memory:** T2 design target ≤ 34 GiB sweep (the usable ≈ 38 GiB of a 40 GB part less a margin; if the calibration cell shows more at E = 128, halve the micro-batch and double the gradient accumulation, which leaves tokens, global batch and the comparison unchanged), routing ≈ 30.5 GB of weights per GPU plus activations (to be measured, ≲ 38 GiB per GPU); T1 ≤ 12 GiB sweep and ≈ 14 GiB OLMoE routing (tight; fallback in Risks) · **Codex units:** 3.2, 3.8 (router mechanics are `sec_moe`/6.6; "changing an existing architecture" appears as an exercise pointing to 7.x upcycling and 6.6)
- **Builds on (book):**
  - `sec_moe`: the dense-gather `MoELayer` (local, not saved), `eq_moe-layer`, the aux-loss versus bias-controller comparison, fine-grained and shared experts, `tab_moe-experts`, `moe_accounting`. **Taken as given:** *why* routing collapses, the Switch loss `eq_moe-aux`, the bias update `eq_moe-bias`, why shared and narrow experts help. 14.2 does not re-derive or re-run them; it starts at *how much* sparsity and granularity to buy and how released models route.
  - `sec_scaling-laws`: Chinchilla form `eq_chinchilla_law`, `eq_six_nd`; the fit protocol of the miniature scaling study.
  - `sec_transformer-block`: `ffn_factory` seam.
- **Builds on (this part):**
  - 14.1: `d2l.config_stats` for total/active parameters of the released configurations used in the router table, `d2l.design_table`.
  - 6.6 (`sec_lm_expert_parallel`): all-to-all dispatch and combine, capacity factor versus dropless, node-limited routing, and the all-to-all volume as a function of top-$k$, hidden size and EP degree. **14.2 does not re-derive that volume** (it calls `d2l.a2a_bytes`, 6.6, for the baseline and adds only the latent width); it adds the *latent width* and applies it to released configurations (Derivation B).
  - 5.2 (`sec_lm_workload_hardware`): decode traffic is set by active parameters and by the number of *distinct* experts touched.
  - 4.2 (R2): the base model, data slice and recipe.
  - `d2l.paired_bootstrap` (1.5) for the pairwise claims.
- **Exports (`#@save`):**
  - `d2l.MoEFFN(d, num_experts, top_k, width, *, shared=0, shared_width=None, score='sigmoid'|'softmax'|'sqrtsoftplus', norm_topk=True, scale=1.0, n_group=1, topk_group=None, latent=None, zero_experts=0, bias_rate=1e-3, dispatch='sorted')` — a drop-in `ffn_factory` product. Forward returns `y` and stores `self.stats` (per-expert token counts, router entropy, mean experts' compute per token, bytes dispatched) without altering the autograd graph.
  - `d2l.moe_dispatch_bytes(num_experts, top_k, hidden, *, latent=None, nodes=8, node_limit=None, b_dispatch=2, b_combine=2, exact=False) -> float` — expected inter-node bytes per token per layer (Derivation B).
  - `d2l.expert_stats(ids, weights, domains, tokens, num_experts)` — load, entropy, domain specialization (JS divergence from the global domain mix), token-identity information, co-activation; returns a `dict` of arrays.
  - `d2l.fit_sparsity_law(df) -> FitResult` and `d2l.optimal_sparsity(coefs, C, grid)` — the parametric fit and its optimizer (Derivation A).
  - `d2l.train_lm_steps(model, batches, steps, *, lr, warmup, decay_frac, seed, eval_every, eval_fn, hooks=()) -> dict` — the shared small-scale training loop (AdamW, linear warm-up, WSD decay from `sec_scheduler`, bf16 autocast, gradient clipping at 1.0); reused by 14.4 and 14.7. Relationship to 4.2: `d2l.pretrain(model, stream, cfg, val_fn=None, hooks=(), resume=None)` already takes a caller-built module, so `train_lm_steps` is a thin wrapper (≈ 30 lines): it builds a `d2l.PretrainConfig` from the keyword arguments (WSD via `d2l.wsd_lr`, parameter groups via `d2l.param_groups`), adapts `eval_fn`/`eval_every` to `val_fn`, calls `d2l.pretrain`, and returns a summary dict. It adds no training logic of its own. *Assumed from 4.2:* `d2l.load_reference_model` (4.2); train/validation token batches come from `d2l.PackedStream` over the FineWeb-Edu `d2l.TokenShards` (4.1, 3.6).
  - Data: `lm-arch-moe-sweep` (JSONL of runs: grid cell, seed, tokens, final and best loss, tokens/s, per-layer load statistics; `fit.json`), `lm-arch-routing-qwen3-30b` (`.npz`: `ids[int16, (tokens, layers, k)]`, `weights[float16]`, `domain[int8]`, `token_id[int32]`; ≈ 40 MB for 3 domains × 100K tokens × 48 layers × 8).
- **Models and data:** R2 with its FineWeb-Edu slice (4.2); `Qwen/Qwen3-30B-A3B` (Apache-2.0, 30,532,122,624 parameters, 61 GB bf16, pinned revision); `allenai/OLMoE-1B-7B-0924` (Apache-2.0, 6,919,161,856 parameters, 64 experts top-8, 16 layers, `norm_topk_prob false`; the T1 routing model). Routing corpora (all ODC-By; ≈ 100K tokens each, fixed sampling seed): prose `HuggingFaceFW/fineweb-edu` (`sample-10BT`), math `HuggingFaceTB/finemath` (`finemath-3plus`), code `HuggingFaceTB/smollm-corpus` (`python-edu`; the source files carry their own repository licenses, so only token ids and statistics are committed, not text).
- **Packages:** `torch>=2.8` (for `torch._grouped_mm` if available; otherwise the sorted loop), `transformers>=5.0` (`Qwen3MoeForCausalLM`, `output_router_logits=True`), `scipy` (fit), `datasets`.

#### Narrative arc

1. **The question.** Spend the same active compute, but store more experts: what do you get, and what do you pay? The opening cell prints, from `config_stats` (14.1), for six released MoE models (Qwen3-30B-A3B, Qwen3-235B-A22B, gpt-oss-120b, DeepSeek-V3, Kimi-K2, Kimi K3): total parameters, active parameters, the ratio, experts per layer, top-$k$, expert width over $d$, shared experts, router score function, number of dense prefix layers. The symptom: the ratio ranges from about 4 (older style) to ≈ 27 (K3: 2.78T/102B by the 14.1 pre-check), while the per-model design choices *do not move monotonically with it*. Then the question narrows to the two knobs (sparsity, granularity) and the two costs (memory, traffic).
2. **Sparsity and granularity as design axes** (`subsec_lm_moe_axes`).
   - Definitions at fixed active width: $S = E/k$ (expansion), $G = k/k_0$ (granularity: how finely a fixed active width is cut); the three matched-comparison rules (matched active parameters, matched total parameters, matched training FLOPs) and which each paper uses.
   - Reproduce `tab_moe-experts` automatically from configs and extend it with a column $E/k$ and a column expert-width-to-$d$ ratio: Qwen3-30B-A3B 0.375, gpt-oss-120b 1.0, DeepSeek-V3 0.286, K3's routed experts $3072/3584 = 0.857$ of the *latent* width.
   - The two published scaling laws on one slide (equations `eq_lm_moe_krajewski`, `eq_lm_moe_abnar` below) and what each says: granularity (Krajewski et al.) and optimal sparsity at fixed compute (Abnar et al.).
3. **Router design in released models** (`subsec_lm_moe_routers`). One table, built from the configs (`tab_lm_moe_routers`): score function (softmax in V2 and Qwen3, sigmoid in V3, K2, K3, GLM and MiniMax, `sqrtsoftplus` in V4), top-$k$ weight normalization (`norm_topk_prob`), the post-scale (`routed_scaling_factor`: 16 in V2, 2.5 in V3, 2.827 in K2, 1.5 in V4-Flash), selection bias (`topk_method noaux_tc`, `e_score_correction_bias`), group-limited routing (`n_group 8`, `topk_group 4` in V3; `n_group 1` in K2, K3 and GLM: no group limit), hash layers (`num_hash_layers 3` in V4: the first three layers have fixed hash routing; the report was **not** read for this, "(unverified)" until the author checks arXiv 2606.19348). Implement each choice as an option of `d2l.MoEFFN` in ≤ 40 lines and show on a toy batch that: (a) sigmoid scores with normalized weights are *not* a softmax over the top-$k$ (print both); (b) the bias moves selection, not weights (assert by gradient); (c) group-limited routing with `topk_group` $g$ bounds the number of nodes per token by $g$ (assert on 10,000 random tokens).
4. **Dense early layers, shared experts and the first-$k$ question** (`subsec_lm_moe_dense_prefix`). `first_k_dense_replace` is 3 (V3, V3.2, GLM-4.5, GLM-5, 5.3), 1 (K2, K3, Kimi-Linear), 0 for Qwen3 MoE (`mlp_only_layers []`) and gpt-oss. What the router can do in layer 0: it sees only embeddings. Measurement in the notebook's routing part: *how much of the expert choice is predictable from the token identity alone*, per layer (normalized mutual information between token id and expert id on the traces). Hypothesis to test, with direction only: routing is most lexical in the first layers (this is what V4's hash layers would exploit; the OLMoE paper reports vocabulary specialization of experts). Do not claim that this *explains* the dense prefix: it motivates it.
5. **Variable compute per token and zero-computation experts** (`subsec_lm_moe_zero`). LongCat-Flash routes some tokens to identity experts so that compute per token varies; add `zero_experts=Z` to `d2l.MoEFFN` (router has $E+Z$ outputs; the zero experts return $x$ unchanged and cost no FLOPs), log mean active experts per token, and show on a toy corpus (the book's TimeMachine characters) that the mean compute falls below the nominal top-$k$ when easy tokens exist. Treat the large-scale claims as the paper's (v5's source), not reproduced here.
6. **Shrinking the traffic: LatentMoE** (`subsec_lm_moe_latent`). $z = W^\downarrow x$, route $z$, aggregate in the latent space, then $W^\uparrow \mathrm{RMSNorm}(u)$ (K3's eq. 11 form; `latent_moe_use_norm true`). Show the byte formula (Derivation B), implement `latent=ℓ` in `d2l.MoEFFN`, and verify the layer is exactly a LatentMoE by comparing against a literal transcription of eq. 11 of the K3 report on random weights. The cost is the 14.1 exercise: $2d\ell$ extra parameters, and the stability issue that K3 mentions (four nearly consecutive matmuls: ill-conditioned) which is a pitfall, not an experiment here.
7. **Sparsity without arithmetic: conditional memory (Engram)** (`subsec_lm_moe_lookup`; sub-block added by this blueprint, see deviation 2). Lookup tables indexed by hashed n-grams (`engram_max_ngram_size 4`, `engram_num_embeddings` ≈ 384M rows per layer, head dim 256, 8 heads; Qwen3.8-Flash-Next has `ngram_size 3`, `ngram_vocab_size_base 20,000,000`). From 14.1: 196.6B of V4.1-Flash's 763B checkpoint elements are Engram tables. Why it is a *different kind of sparsity*: memory capacity and lookup traffic scale, FLOPs do not. Implement a miniature (hash the previous $n$ token ids into a table of $M$ rows, gate the lookup into the residual) in ≤ 40 lines and measure parameter count, lookup bytes per token, and collision rate as a function of $M$ on R3. The quality claims are the paper's (arXiv 2601.07372); do **not** test them at this scale (a figure with no conclusion is fine). *Stays a sub-block of 14.2* unless Engram-style tables spread across several releases (the chapter's extension rule; editor's decision, 2026-10-29).
8. **MoE at inference: total sets memory, active sets traffic, and the middle** (`subsec_lm_moe_inference`). Derivation C: the fraction of experts touched by a decode batch. Plot predicted versus measured touched-expert fraction on the Qwen3-30B-A3B traces. Consequence for 5.2: at small batch the decode step reads $\approx b\,k/E$ of each layer's experts; at $b \gtrsim E/k \cdot \ln(\cdot)$ almost all experts are read, and the step time approaches the dense-weights bound.
9. **What experts specialize in** (`subsec_lm_moe_specialization`). Load Qwen3-30B-A3B (T2 pair) or OLMoE (T1) with `output_router_logits=True`; run 100K tokens each of prose, math and code; compute `d2l.expert_stats`: per-layer load imbalance (max/mean), router entropy, domain specialization (JS divergence of each expert's domain mix from the global mix), token-identity information, co-activation. Report layers where specialization is strongest. Caution to include in the text: domain and token statistics describe *correlations*, not the function of an expert.

#### Derivations

**A. Optimal sparsity at fixed training FLOPs (`eq_lm_moe_abnar`, `eq_lm_moe_opt`).**

*Statement.* Under the parametric law of Abnar et al. (their Eq. 6, quoted from arXiv 2501.12370) the loss at fixed total training compute $C=6 N_a D$ has an interior optimum in the sparsity $S$ that depends on $N$ and $C$; the optimum is the root of a sum of three power terms and has no closed form but is computed by one-dimensional root finding.

*Law (as published).* $L(N,D,S)=\dfrac{a}{N^{\alpha}}+\dfrac{b}{D^{\beta}}+\dfrac{c}{(1-S)^{\lambda}}+\dfrac{d}{(1-S)^{\delta}N^{\gamma}}+e$, where $N$ is total parameters, $S$ the ratio of non-active to total experts and the paper fits $\alpha=0.5962$, $\beta=0.3954$, $\lambda=-0.1666$, $\delta=0.1603$, $\gamma=0.1595$, $a=16612.50$, $b=5455.67$, $c=0.4598$, $d=17.26$, $e=0.94$ (their Table 3; the paper reports $R^2=99\%$ on the fit set and $68\%$ on held-out $S=0.98$; the units of $N$ and $D$ are the paper's, so the check below is *structural*, not a prediction). The Krajewski et al. law for granularity, for comparison: $L(N,D,G)=c+\left(\tfrac{g}{G^{\gamma}}+a\right)N^{-\alpha}+bD^{-\beta}$ (`eq_lm_moe_krajewski`).

*Assumptions.* (i) $N_a=(1-S)N$ (non-expert parameters ignored; state the error this makes for R2-scale and for released models using `config_stats`); (ii) $C=6N_aD$ exactly; (iii) the law holds in the region searched.

*Steps.*
1. Let $u=1-S\in(0,1]$ and eliminate $D$: $D = C/(6uN)$.
2. $L(N,u;C) = aN^{-\alpha} + b\,(6uN/C)^{\beta} + c\,u^{-\lambda} + d\,u^{-\delta}N^{-\gamma} + e$.
3. At fixed $N$: $\partial L/\partial u = b\beta\,(6N/C)^{\beta}u^{\beta-1} - \lambda c\,u^{-\lambda-1} - \delta d\,N^{-\gamma}u^{-\delta-1} = 0$. With the published signs ($\lambda<0$, $\delta>0$) the second term is positive and the third negative, so the three terms can balance: an interior root exists when the data term and the capacity term cross.
4. At fixed $u$: $\partial L/\partial N = -\alpha aN^{-\alpha-1} + b\beta(6u/C)^{\beta}N^{\beta-1} - \gamma d\,u^{-\delta}N^{-\gamma-1}=0$.
5. Solve the pair (3)–(4) by nested bisection (outer in $N$, inner in $u$; both are one-dimensional and monotone on bracketing intervals; use `scipy.optimize.brentq`).
6. Sensitivity: differentiate (3) implicitly in $C$ to see how $u^*$ moves with compute; the sign gives the paper's claim that optimal sparsity rises (the active fraction falls) with compute.

*Numerical check cell.* For a range of $C$: (i) solve (3)–(4) by root finding; (ii) minimize $L$ by brute force on a $200\times200$ grid of $(\log N, u)$; (iii) assert the two agree to the grid resolution; (iv) assert $u^*(C)$ is monotone in $C$ on the grid in the direction the implicit derivative predicts; (v) repeat with the coefficients fitted in the notebook's own sweep (Experiment). No absolute value of $S^*$ is claimed from the paper's coefficients.

**B. All-to-all bytes per token per layer with a latent width (`eq_lm_moe_a2a`).**

*Statement.* With experts uniformly placed on $n$ nodes, a token routed to $k$ experts crosses the node boundary for $M_{\text{eff}}$ distinct remote nodes, and the bytes sent are $B = M_{\text{eff}}\,\ell\,(b_{\text{disp}} + b_{\text{comb}})$, where $\ell$ is the routed width ($\ell=d$ for a standard MoE, the latent width for LatentMoE) and $b$ the bytes per element; node-limited routing caps $M_{\text{eff}}\le M$.

*Assumptions.* Uniform independent routing (an upper bound on spread); dedup within a node (the token is sent once per node and fanned out over NVLink, DeepSeek-V3's scheme); activations in the dispatched precision; ignores the shared-expert path (computed locally); not the 6.6 formula's dependence on EP degree (recapped by `:numref:` only).

*Steps.*
1. The number of distinct nodes among $k$ draws without replacement from $E$ experts split equally into $n$ nodes: $\mathbb{E}[\#\text{nodes}] = n\bigl(1-\binom{E-E/n}{k}/\binom{E}{k}\bigr)$; with replacement, $n(1-(1-1/n)^{k})$.
2. Subtract the local node (probability $\approx$ the same expression for one node) to get remote nodes $M_{\text{eff}}$.
3. Node-limited routing with `topk_group` $g$ chooses $g$ node-sized groups first: $M_{\text{eff}} \le g$ regardless of $k$ (V3: $g=4$ with `n_group 8`).
4. Dispatch carries $\ell$ elements, combine returns $\ell$ elements; $B=M_{\text{eff}}\ell(b_{\text{disp}}+b_{\text{comb}})$.
5. Latent MoE: $\ell<d$ reduces $B$ by $\ell/d$ at the price of $2d\ell$ parameters per layer (14.1 exercise 3) and of one up-projection after combine.

*Numerical check cell.* (a) Monte-Carlo: draw 10⁶ tokens with `torch.randint`/`topk` of random scores over $E=256$, $n=8$; compare mean distinct nodes with step 1 to 3 decimals (seeded); (b) apply the group-limited selection of V3 (`n_group 8`, `topk_group 4`) to random scores and assert the max distinct nodes is 4; (c) evaluate `d2l.moe_dispatch_bytes` from `config_stats` fields for Qwen3-30B-A3B, gpt-oss-120b, V3, K2 (`n_group 1`: no cap), K3 ($\ell=3584$), V4.1-Flash on $n=8$ nodes at 2 bytes and tabulate (values to be measured; the cell prints them).

**C. Experts touched by a decode batch (`eq_lm_moe_touched`).**

*Statement.* If each of $b$ tokens in a decode step picks $k$ of $E$ experts independently and uniformly, the expected fraction of the layer's experts whose weights must be read is $1-(1-k/E)^{b}$, and weight bytes per step $\approx W_{\text{nonexp}} + L_{\text{moe}}\,E\,P_{\text{exp}}\bigl(1-(1-k/E)^b\bigr)$.

*Steps.* (1) Expert $i$ is untouched by one token with probability $1-k/E$ (exact under uniform marginals); (2) independence across tokens gives $(1-k/E)^b$; (3) linearity of expectation gives the touched fraction; (4) the batch at which 90% of experts are touched is $b_{90} = \ln 0.1/\ln(1-k/E)$.

*Check.* Monte-Carlo with `torch.rand` top-$k$ for $E=128,k=8$ (to 3 decimals); then **replace** the uniform router by the measured Qwen3-30B-A3B routing traces and plot the measured touched fraction versus $b$ against the formula per layer; the gap is the effect of correlated routing (direction expected: fewer experts touched than the formula, to be measured).

#### From-scratch implementation

Budget ≈ 140 lines: `MoEFFN` (router with three score functions and bias, group-limited selection, sorted dispatch, shared experts, zero experts, latent), `moe_dispatch_bytes`, `expert_stats`, the fit.

```python
class MoEFFN(nn.Module):  #@save
    def __init__(self, d, num_experts, top_k, width, *, shared=0, shared_width=None, score='sigmoid',
                 norm_topk=True, scale=1.0, n_group=1, topk_group=None, latent=None, zero_experts=0,
                 bias_rate=1e-3, dispatch='sorted'): ...
    def route(self, x):            # -> idx (T,k), w (T,k), scores (T,E+Z)
    def forward(self, x):          # x: (B,T,d) -> (B,T,d); sets self.stats
    def update_bias(self):         # sign-controller of sec_moe eq_moe-bias, called by the training loop
```

*Sorted dispatch:* flatten tokens, `argsort` the flat expert ids, `torch.bincount` for counts, `torch.split` the gathered activations, loop over experts (≤ 128 iterations), scatter-add the weighted outputs with `index_add_`. Use `torch._grouped_mm` when present (a flag); the loop is the oracle.

| Test | Reference | Tolerance / seed |
|---|---|---|
| Sorted dispatch equals dense-gather output | `sec_moe`'s `MoELayer` run with the same weights (copy parameters; softmax router, no bias) | `max abs diff < 1e-5` in fp32; seed 0; also bf16 within 3e-3 |
| Gradients agree with dense-gather | same | relative error < 1e-4 on all parameters |
| Grouped matmul path equals the loop | the loop | exact up to fp reorder: `< 1e-5` fp32 |
| Sigmoid + normalized top-$k$ weights | literal transcription of the Qwen3 MoE block (`Qwen3MoeSparseMoeBlock` in Transformers, softmax + `norm_topk_prob`) for the softmax case, and of `DeepseekV3MoE` for sigmoid with `e_score_correction_bias`, `n_group`, `topk_group`, `routed_scaling_factor` | `< 1e-5` on random weights and 256 tokens, seed 0 |
| Group-limited node count | assertion over 10⁴ tokens | max distinct groups = `topk_group` |
| LatentMoE | literal eq. 11 of the K3 report (shared + $W^\uparrow\mathrm{RMSNorm}(\sum p_iE_i(W^\downarrow x))$) | `< 1e-5` |
| Zero experts | an identity-expert reference | exact outputs for tokens routed only to zero experts |
| `config_stats` parameter count of an `MoEFFN` layer | `sum(p.numel())` | exact |

#### In production

- **Transformers:** `Qwen3MoeSparseMoeBlock` (softmax router, `norm_topk_prob`, loop over experts with `index_add_`), `DeepseekV3MoE` (sigmoid, `n_group`, `topk_group`, `e_score_correction_bias`); the from-scratch layer is the test oracle for both, on random weights built from the real configs of *small* instances (width 256, 16 experts) so that nothing large is loaded.
- **Kernel libraries (pointers only):** MegaBlocks (grouped GEMM, dropless; v5-listed in 6.6) and DeepEP (expert-parallel all-to-all; 6.6); `torch._grouped_mm`; in vLLM and SGLang the fused MoE Triton kernels (13.x). The notebook times the sorted loop against `torch._grouped_mm` on one A100 and reports the ratio (to be measured), not as a recommendation.
- **Routing capture:** `output_router_logits=True` returns `router_logits` per layer; the notebook recomputes top-$k$ from them using the model's `norm_topk_prob`, `num_experts_per_tok` and asserts agreement with a forward hook on `mlp.experts`.

#### Headline experiment

- **Hypothesis (directional, from sources):** at fixed active parameters and tokens, validation loss decreases as total parameters grow (larger $S$) and as granularity grows (larger $G$), with diminishing returns in both; at a *fixed* token budget the benefit of $S$ saturates or reverses when experts become data-starved. Krajewski et al. report that loss improves with granularity at fixed $N$ and $D$ (their fitted $\gamma$), and Abnar et al. report that at fixed compute the optimal sparsity increases with model size; the magnitudes at R2 scale are **to be measured**.
- **Setup (T2).** Base: R2 (`d2l.GPT`-compatible; width and depth from 4.2) with every FFN replaced by `MoEFFN`, except layer 0, which stays dense (the V3 convention at `first_k_dense_replace` $=1$). Fixed active FFN width $A$ = the dense FFN width of R2 (a shared expert is *not* used so that active parameters match exactly; router parameters are the only active-parameter difference and are logged). Grid: $G\in\{1,2,4,8\}$ ($k=G$ active experts of width $A/G$), $S\in\{2,4,8,16\}$ ($E=S\cdot k$), plus the dense baseline: 17 runs. Router: sigmoid scores with normalized top-$k$ weights and the bias controller of `sec_moe` (`bias_rate=1e-3`), no auxiliary loss. Tokens: set by a calibration cell so the slowest run (E=128) takes ≤ 8 min; **the same token count for every cell**; batch, learning rate, WSD schedule from the 4.2 recipe; one seed per cell plus 2 extra seeds at one reference cell $(S,G)=(8,4)$ to measure seed noise. Data: FineWeb-Edu slice of R2; evaluation: the R2 validation set, final and best loss.
- **Procedure.** (1) Calibrate tokens/s for the slowest and fastest cell; (2) run the grid; (3) fit $L(S,G)=e+(a+g\,G^{-\gamma_G})\,S^{-\alpha}$ (five parameters, 16 points) by least squares in log-space; (4) compute residual-bootstrap intervals for the parameters; (5) report the reference-cell seed standard deviation $\sigma$ and mark all cell differences $<2\sigma$ as ties; (6) compare the fitted $S$-dependence with the published form.
- **Metrics:** final validation loss; *loss versus total parameters at fixed active*; tokens/s; peak memory; per-layer load imbalance (max/mean) and router entropy at the end; total-to-active ratio.
- **Plot (`fig_lm_moe_sweep`).** x-axis: total parameters (log), y-axis: validation loss; one line per $G$, markers for cells, a shaded band of $\pm2\sigma$ around the dense baseline, the fitted surface as thin lines; second panel: tokens/s relative to dense versus $S$ (shows what the extra capacity costs in wall time at this implementation's maturity). Third panel: per-layer load imbalance by $S$ at $G=4$.
- **Expected outcome:** direction as in the hypothesis; the *size* of the effect and the position of any interior optimum in $S$ are to be measured. The fitted $\alpha$ and $\gamma_G$ are reported with intervals; **no comparison with the paper's exponents is claimed**, only the sign.
- **Noise check:** 3 seeds at the reference cell and at the dense baseline give $\sigma$; a pair of cells is a "win" only if the loss gap exceeds $2\sigma$ *and* `d2l.paired_bootstrap` over validation documents excludes 0; otherwise a tie.
- **T1 reduction (`cfg.scale='t1'`):** `d2l.GPT` width 256, 6 blocks, FineWeb-Edu 20M-token slice; configurations: dense, $(S,G)\in\{(4,1),(16,1),(4,4),(16,4)\}$; 2 seeds each; ≤ 90 s each. T1 output is a smoke test of the pipeline and of the *direction*; it carries no fitted exponent (the fit cell is skipped with a message).
- **Artifacts committed (T2):** `lm-arch-moe-sweep` (run logs and `fit.json`), `lm-arch-routing-qwen3-30b` (recorded on the T2 pair; readers with one GPU analyse it on CPU, and T1 records OLMoE routing itself).
- **Second experiment (analysis).** Routing traces on three domains: panels of per-layer load imbalance, specialization (JS divergence), token-identity information, touched-expert fraction versus batch (Derivation C), top co-activated pairs. Expected: some layers show domain specialization well above a within-domain shuffle baseline (to be measured; the shuffle baseline permutes domain labels across documents with 1,000 permutations and reports the 95th percentile).

#### Figures and tables

| Label | Content | Cell |
|---|---|---|
| `tab_lm_moe_models` | the opening table (six released MoEs: totals, active, ratio, $E$, $k$, width/$d$, shared, score, dense prefix) | cell 1 |
| `tab_lm_moe_routers` | router choices per release from configs | `routers_from_configs()` |
| `fig_lm_moe_sweep` | headline plot (3 panels) | `plot_sweep(df, fit)` |
| `fig_lm_moe_opt_sparsity` | $S^*(C)$ from Derivation A with the fitted law, and the surface of $L$ | `plot_optimal()` |
| `fig_lm_moe_a2a` | bytes per token per layer for released configs at 8 nodes | `plot_a2a()` |
| `fig_lm_moe_touched` | touched fraction vs batch: formula, Monte-Carlo, measured | `plot_touched()` |
| `fig_lm_moe_specialization` | per-layer specialization, load imbalance, token information | `plot_routing(stats)` |
| `fig_lm_moe_lookup` | Engram miniature: rows vs collisions vs lookup bytes | `plot_lookup()` |

#### Pitfalls and misconceptions

- "A 3B-active MoE is a 3B model": memory scales with total parameters (Qwen3-30B-A3B needs 61 GB of weights at bf16), and decode traffic at modest batch reads most experts (Derivation C).
- Matched active parameters is not matched FLOPs: attention and the router are part of the active compute; more experts also means more small matmuls and lower arithmetic intensity (to be measured: tokens/s versus $S$ in the headline plot).
- A fitted exponent from 16 points is an estimate with a wide interval; the Abnar et al. fit itself has $R^2=68\%$ on its held-out slice. Do not extrapolate $S^*$ beyond the grid.
- Load balance is not specialization; a perfectly balanced router can be random. Report both (14.2's stats).
- Normalizing top-$k$ weights, the `routed_scaling_factor`, the score function and the bias all interact: V2 (softmax, scale 16) and V3 (sigmoid, normalized, scale 2.5) are not interchangeable.
- The "dense-gather" `sec_moe` layer is correct but its cost grows with $E$; sweeps need the sorted dispatch.
- Expert statistics from 100K tokens are noisy for rare experts; report counts with the bootstrap interval, and do not interpret experts with fewer than a stated number of tokens.

#### Summary (draft bullets)

- Sparsity and granularity are two knobs on one budget: at fixed active compute, total parameters buy loss at the price of memory and traffic.
- Released routers differ in four independent choices (score, normalization/scale, selection bias, group limits); they are options of one layer.
- Latent routing shrinks expert communication in proportion to $\ell/d$; Engram-style lookup adds parameters that cost memory but not FLOPs.
- The optimal sparsity at fixed compute is the root of a sum of power terms; the published fit supports the direction, not the digits.
- Decode reads the experts the batch touches: $1-(1-k/E)^b$ of them under uniform routing.
- Experts specialize partly by domain and strongly by token identity in early layers; that is a correlation to report with a shuffle baseline, not a mechanism.

#### Exercises

*Check.*

1. **Which matched comparison?** A paper compares a dense 7B with an MoE with 1B active and 7B total at equal training tokens. Which of "matched active", "matched total", "matched FLOPs" is it? What would the third-party reader conclude if the paper claimed "same quality, 6× cheaper"? *Sketch:* matched total parameters, unmatched FLOPs; the claim holds for FLOPs per token only; training tokens are equal so the MoE got 7× fewer FLOPs, and memory is equal, not 6× cheaper.
2. **Zero experts.** With $E=64$ real experts, $Z=16$ zero experts and top-$k=8$ over all 80, what is the mean number of FLOP-bearing experts per token if the router picks zero experts with probability 25% of the slots? What if one token is routed to 8 zero experts? *Sketch:* 6 of 8 on average ($8\times0.75$); that token costs no expert FLOPs but still a router call, and its output is the weighted sum of $x$ copies.

*Derive.*

3. **Touched fraction under non-uniform routing.** If expert $i$ has marginal selection probability $q_i$ ($\sum_i q_i=k$) per token, show that the expected touched fraction is $1-\frac1E\sum_i(1-q_i)^b$, that it is *at most* the uniform formula by Jensen, and evaluate it for $q_i\propto i^{-s}$ with $s=0.5$ and $E=128,k=8$. *Sketch:* $(1-q)^b$ is convex in $q$, so $\frac1E\sum(1-q_i)^b\ge(1-\bar q)^b$ with $\bar q=k/E$; equality iff uniform.
4. **A latent width that equalizes traffic.** For Kimi K3 ($k=16$, $\ell=3584$) and DeepSeek-V3 ($k=8$, $d=7168$, node limit $M=4$ of $n=8$ nodes) find the latent width $\ell$ at which the two have the same dispatch bytes per token under Derivation B (uniform, no node limit for K3). *Sketch:* ignoring the local-node correction, K3's uncapped $M_{\text{eff}}=8(1-(7/8)^{16})\approx7.06$ and V3's is capped at $M=4$ (uncapped it would be $8(1-(7/8)^{8})\approx5.25$, so the cap binds). Equal bytes need $7.06\,\ell=4\cdot7168$, i.e. $\ell\approx4{,}061$. K3's $\ell=3584$ is about 12% below that, so under these assumptions (uniform independent routing, equal byte widths for dispatch and combine) K3 sends about 12% fewer bytes per token per layer than V3 despite $k=16$; state that real routing is neither uniform nor independent and that K3's expert-parallel system uses redundant experts to balance load (report §5.2.1).

*Measure.*

5. **Granularity at fixed S.** Predict the sign of the loss difference between $G=1$ and $G=8$ at $S=8$ in the headline experiment, then run it with 3 seeds each and report with `d2l.paired_bootstrap`. State the minimal detectable difference.
6. **Router entropy versus load.** On the Qwen3-30B-A3B traces, compute per-layer router entropy of the *softmax over 128 experts* and the load imbalance of the selected top-8; predict whether the two are correlated across layers, then compute the correlation with a bootstrap interval. Compare with OLMoE (T1).

*Open.*

7. **Dense prefix or hash layers.** Using the token-identity information measured per layer, design a small experiment on the R2-scale MoE comparing (i) a dense first layer, (ii) a learned-router first layer, (iii) a hash-routed first layer (fixed random map from token id to expert). Run at T1 with ≥ 3 seeds and write down the decision rule before running.

#### Further reading

- Papers (v5): [DeepSeekMoE](https://arxiv.org/abs/2401.06066) (fine-grained and shared experts); [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664) (taught in `sec_moe`; compare here); [Scaling Laws for Fine-Grained MoE](https://arxiv.org/abs/2402.07871) (granularity law); [Parameters vs FLOPs](https://arxiv.org/abs/2501.12370) (optimal sparsity; coefficients quoted from its Table 3); [LongCat-Flash](https://arxiv.org/abs/2509.01322) (zero-computation experts); [LatentMoE](https://arxiv.org/abs/2601.18089); [Kimi K3](https://arxiv.org/html/2607.24653v1) (Stable LatentMoE: eq. 11, SiTU-GLU, quantile balancing, §2.3); [Nemotron 3](https://arxiv.org/abs/2512.20856); [Kimi K2](https://arxiv.org/abs/2507.20534); [DeepSeek-V3](https://arxiv.org/abs/2412.19437).
- *New (verified 2026-10-28):* [OLMoE](https://arxiv.org/abs/2409.02060) (title confirmed: "OLMoE: Open Mixture-of-Experts Language Models"; open routing analysis: domain and vocabulary specialization); [ST-MoE](https://arxiv.org/abs/2202.08906) (router z-loss and stability; carried over from Codex 3.3); [Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372) (Engram); [Switch Transformers](https://arxiv.org/abs/2101.03961) and [MegaBlocks](https://arxiv.org/abs/2211.15841) (already in v5 for 6.6; Codex 3.2); [Sparse Upcycling](https://arxiv.org/abs/2212.05055) (v5, 6.6; exercise pointer for "changing an existing architecture").
- Models and teaching: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Mixture of Experts Explained](https://huggingface.co/blog/moe); [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924).
- Code: [`modeling_qwen3_moe.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) (router and expert loop; HTTP 200); [`modeling_deepseek_v3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py) (sigmoid router with group limits and correction bias); [Engram code](https://github.com/deepseek-ai/Engram) (HTTP 200).

#### Risks and open decisions

| Risk | Fallback |
|---|---|
| The sorted-dispatch loop is too slow at $E=128$ on the A100 | use `torch._grouped_mm`; else cap the grid at $E\le64$ ($G\le4$ at $S=16$) and say so |
| Seed noise is comparable to the effect (likely for $S\le4$, $G\le2$) | report ties honestly; raise tokens or add seeds at the two corners only; the plot already shows the $2\sigma$ band |
| 4.2's `load_reference_model` is unavailable or its config differs | `d2l.train_lm_steps` + `d2l.PackedStream` (4.1); documented here |
| Qwen3-30B-A3B (61.06 GB of weights) does not fit one A100 40 GB | the routing run uses the T2 pair (`device_map='auto'`, layers split over two GPUs, no NVLink needed for correctness); if activations at batch 4 × 1024 exceed 38 GiB per GPU, batch 2 × 1024; the traces are only for statistics |
| OLMoE (13.8 GB bf16) does not fit a 15 GiB slot with activations | claim a whole 24 GB GPU; in a slot, sequence 256, batch 2, `torch.cuda.empty_cache()` per layer group; else run OLMoE's first 8 layers only (`config.num_hidden_layers=8`) and say so |
| `num_hash_layers` semantics in V4 not confirmed | marked "(unverified)" until the author reads arXiv 2606.19348 §2 |
| Domain corpora licensing (code) | commit only token ids and statistics, not text |
| Paper's scaling-law units are unknown | the check is structural; the notebook's own fit is the quantitative result |

### 14.3 Latent attention and cross-layer KV sharing

- **Label:** `sec_lm_mla` · **Tier:** T1 · **Budget:** ≤ 14 min on the build box: equivalence and cost model (≈ 1 min), single-layer decode latency sweep at DeepSeek-V3 layer dimensions (≈ 3 min), GPT-2 conversion curves (≈ 4 min), tiny trained comparison MLA/GQA/CLA at equal cache (2 seeds × 3 arms × ≈ 1 min) · **Peak memory:** ≤ 10 GiB (the single-layer benchmark holds ≈ 0.4 GB of weights at bf16 and a cache of ≤ 2.7 GB in the worst arm) · **Codex units:** 3.4 (all), 2.5 (the YOCO/CLA cache-sharing part only; the "local/global and asymmetric" parts are in 14.4 and the V4.1 discussion)
- **Builds on (book):**
  - `sec_kv-cache`: `eq_kv-cache-bytes`; `d2l.GQAAttention` (`#@save`; the equal-cache baseline); `subsec_kv-cache-low-rank`. **Taken as given from that subsection:** the *premise* that realized keys and values of GPT-2 are approximately low rank (an oracle, per-sequence SVD of the concatenated 1536-wide K and V, tested on one passage); the *name* MLA with rank 512 in DeepSeek-V2; and the *statement* that RoPE does not commute with a shared down-projection so MLA carries a separate rotary component. The section never derives either the absorbed path or the obstacle. 14.3 starts there and derives both.
  - `sec_mdl-svd-low-rank`: Eckart–Young. 14.3 uses it, and adds the activation-weighted variant.
  - `sec_positional-information`: the RoPE rotation $R_{i}$ with $R_t^\top R_s=R_{s-t}$.
  - `sec_transformer-block`: the `attn_factory` seam; `d2l.GPT`.
- **Builds on (this part):**
  - 14.1: `d2l.config_stats` (cache bytes of the released MLA configs, equality tests), `d2l.ConfigStats.flops_per_token` with `mode='decode'`.
  - 5.2: the decode-step model $t(b,L)=\max(\cdot,\cdot)$ and `d2l.chip_specs()` (5.1 `DataFrame`, columns `p16_tflops`, `mem_bw_tbs` in TB/s, `mem_gb`; the A100-80GB row is read from the NVIDIA datasheet by 5.1 and is the analytic target of the capstone scenarios, an A100 40 GB PCIe row from the same datasheet serves the T2 runs, and the build-box row is measured by 5.1/5.2).
  - 1.2: `d2l.load_qwen3` for the optional RoPE conversion cell (R1).
  - 5.2 `d2l.ServingModel(cfg, chip, ..., kv_bytes_override=...)` (editor's interface note): the latency prediction in the headline experiment passes `config_stats` cache bytes (for MLA: $(r+d_r)b\cdot L$ per token) through `kv_bytes_override` instead of recomputing them.
  - 12.4: contrast: inference-time compression of an *already trained* cache (KIVI, eviction) versus architecture-level compression that is trained in.
- **Exports (`#@save`):**
  - `d2l.MLAAttention(d, num_heads, dn, dr, dv, r, rq=None, *, rope_base=10000.0, nope=False)` — causal MLA with `forward(x, cache=None, mode='expanded'|'absorbed', pos=0)`; `cache` is a `d2l.MLACache` holding `(c, k_rope)` of shapes `(B, n, r)` and `(B, n, dr)`; `mode` changes the *computation order* only (Derivation A).
  - `d2l.MLACache` — preallocated latent cache with `append`, `bytes()`.
  - `d2l.mla_decode_cost(H, dn, dr, dv, r, n, *, strategy, bytes_per=2) -> dict(flops, bytes, intensity)` for `strategy in ('cache_expanded', 'latent_expand_each_step', 'absorbed')`.
  - `d2l.CrossLayerKV(share: int)` — a helper that wraps a list of `GQAAttention`-compatible modules so layers $j\cdot s,\dots,(j+1)s-1$ read the K/V of layer $j\cdot s$ (CLA); `d2l.cla_gpt(vocab, d, H, Hkv, L, share)` builds a `d2l.GPT` with it.
  - `d2l.joint_svd_kv(W_k, W_v, r, *, calib=None) -> (W_down, W_up_k, W_up_v)` — data-free joint SVD of $[W_K\ W_V]$, or the activation-aware variant when `calib` (a tensor of inputs $X$) is given.
  - `d2l.mha_to_mla(attn, r, *, calib=None)` — converts a `d2l.GPT` attention block with learned or absent RoPE (GPT-2) into an `MLAAttention` with `nope=True`.
  - Data: `lm-arch-mla-conversion` (JSON: rank, cache elements/token/layer, method, loss with and without recovery steps, seeds).
- **Models and data:** random-weight layers at DeepSeek-V3 dimensions from its config (`d=7168, H=128, r=512, rq=1536, dn=128, dr=64, dv=128`; no checkpoint is downloaded); GPT-2 124M with `d2l.DATA_HUB` weights as in `sec_kv-cache` (MIT); evaluation text: *The Time Machine* (as in the book) and eight 1,024-token passages from R3 (the book's own text) as a second evaluation set; optional R1 `Qwen/Qwen3-0.6B` (Apache-2.0) for the partial-RoPE conversion cell; recovery data: the first 2M tokens of the R2 FineWeb-Edu slice (ODC-By).
- **Packages:** `torch>=2.8`, `transformers>=5.0` (oracle: `DeepseekV3Attention`).

#### Narrative arc

1. **The question.** `sec_kv-cache` showed that the cache is what fills the accelerator and that GPT-2's keys and values are nearly low rank. *If you can store 576 numbers per token instead of 40,960 (DeepSeek-V3's 128 heads × (128+64 key dims, 128 value dims)), can you also attend with them directly, without ever rebuilding the full keys?* The opening cell prints, from `config_stats` (14.1), cache bytes per token of seven released models: Qwen3-8B 147,456 B (GQA 32:8), GLM-4.5 376,832 B (GQA 96:8), DeepSeek-V3 70,272 B (MLA), Kimi-K2 70,272 B, GLM-5 89,856 B, and the expanded, never-stored K/V size of DeepSeek-V3 for comparison (61 layers × 128 heads × (192+128) × 2 B = 4,997,120 B/token; pre-check). The symptom: MLA stores 71× less than the K/V it implies, and 2.1× less than Qwen3-8B's GQA while having 4× more heads.
2. **The MLA computation** (`subsec_lm_mla_def`). Down-projection to a latent $c_t\in\mathbb{R}^r$, up-projections to per-head keys and values, a separate rotary slice $k^R_t\in\mathbb{R}^{d_r}$ shared by all heads, queries with an optional LoRA path. Equations `eq_lm_mla_def`. Parameter shapes read from the header of a real config (14.1's `d2l.safetensors_census`): `kv_a_proj_with_mqa [576, 7168]`, `kv_b_proj [24576, 512]` in Kimi K3 (the same shapes as V3 with 96 heads).
3. **Absorption: attend in the latent space** (`subsec_lm_mla_absorb`). Derivation A. Present the three decode strategies (cache the expanded K/V; cache the latent and re-expand every step; absorbed) in a single table (`tab_lm_mla_strategies`) with FLOPs, bytes and arithmetic intensity per context token per layer, computed by `d2l.mla_decode_cost` at V3's dimensions. Only the absorbed strategy has both the small cache *and* a sane FLOP count.
4. **Why RoPE has to be decoupled** (`subsec_lm_mla_rope`). Derivation B, with the numerical demonstration that the absorbed path equals the explicit path with decoupled RoPE and *fails* with RoPE applied after the up-projection. State what Kimi Linear and K3 do instead (`mla_use_nope true`: no rotation on the MLA layers, the KDA layers carry position; the rope slots remain in the weight shapes: unresolved whether they are used, "(unverified)": the author reads `modeling_kimi_linear.py` from the model repo) and what V4.x do (one KV head of 512 channels including 64 rotary channels, output rotated back by a conjugate RoPE per the FlashMLA README's fused kernel description).
5. **Decode latency and cache against GQA** (`subsec_lm_mla_latency`). Run a single layer at V3's dimensions against `GQAAttention` with 128 query heads and 8 KV heads of 128 (the same query width, a 3.6× larger cache) on the build box: latency per decode step versus context $n\in\{1\text{K},2\text{K},\dots,32\text{K}\}$ and batch $b\in\{1,4,16\}$; overlay the prediction of the 5.2 model. This is the headline experiment (below).
6. **Prefill is different** (`subsec_lm_mla_prefill`). Prefill compute is the bottleneck, and absorbed prefill costs $(2r+d_r)/(d_n+d_r+d_v)=1088/320\approx3.4\times$ more attention FLOPs per head-pair than the expanded form; the expanded path (up-project K and V once per chunk, then run ordinary MHA) is the right prefill mode. Show the two modes' costs and confirm they agree numerically (expanded prefill, absorbed decode on the same cache).
7. **Sharing across layers: CLA** (`subsec_lm_mla_cla`). Owner layers compute K/V; the next $s-1$ layers reuse them. Cache bytes divide by $s$. Implement `CrossLayerKV`, verify the cache arithmetic against `config_stats` for a toy config, and compare the *equal-cache* options at R2-tiny: GQA with $H_{kv}=2$ versus CLA-2 with $H_{kv}=4$ versus MLA with $r$ chosen to match. State the CLA paper's claim (2× on top of MQA at nearly unchanged accuracy, 1B and 3B models trained from scratch) as the source's, not as a result here. Connect to V4.1: its `kv_source_layer_ids [2,8,14,20]` (14.1) is CLA with a *non-uniform* share pattern in the layer dimension and the report §2.3 calls cache sharing one of three multiplicative dimensions (entry size, sequence compression, layer sharing); the YOCO decoder–decoder design (v5 lists V4.1's CED as "inspired by YOCO") is cited, not rebuilt.
8. **Converting a trained model** (`subsec_lm_mla_convert`). Joint SVD of $[W_K\ W_V]$ (data-free) versus the activation-aware variant (Derivation C); head-mean pooling (the GQA conversion of Ainslie et al., already in the book) at equal cache; the partial-RoPE step of MHA2MLA described and demonstrated on R1 as an optional cell (rank the 32 frequency pairs of a 64-dimensional rotary slice by their contribution to attention scores and keep the top $m$ rotated). Report the sources' magnitudes verbatim as the sources': MHA2MLA reports a 92.19% KV cache reduction on Llama-2-7B with a 0.5% LongBench drop after fine-tuning on 0.3–0.6% of the data; TransMLA reports 93% compression and a 6B-token fine-tune. The notebook's own experiment is GPT-2 (no RoPE), so it needs no decoupling.

#### Derivations

**A. Absorbed MLA decode (`eq_lm_mla_absorb`).**

*Statement.* Let the explicit MLA score between query position $t$ and key position $s\le t$ in head $h$ be $\sigma_{ts}^h=\bigl(q^{N}_{t,h}\!\cdot k^{N}_{s,h}+q^{R}_{t,h}\!\cdot k^{R}_{s}\bigr)/\sqrt{d_n+d_r}$ with $k^N_{s,h}=W^{UK}_h c_s$, $v_{s,h}=W^{UV}_h c_s$. Then attention can be computed using only the cached pairs $(c_s,k^R_s)$, with $\hat q_{t,h}=(W^{UK}_h)^\top q^N_{t,h}$, scores $\sigma^h_{ts}=(\hat q_{t,h}\!\cdot c_s+q^R_{t,h}\!\cdot k^R_s)/\sqrt{d_n+d_r}$ and output $y_t=\sum_h W^O_h\,W^{UV}_h\,\hat o_{t,h}$, $\hat o_{t,h}=\sum_s\alpha^h_{ts}c_s$. The result is identical to the explicit path.

*Assumptions.* Causal softmax; the same scaling factor $1/\sqrt{d_n+d_r}$ in both paths (it is a property of the score, not of the path); decoupled rotary slice (Derivation B); no normalization between the up-projection and the dot product.

*Steps.*
1. Associativity: $q^N_{t,h}\!\cdot(W^{UK}_hc_s)=\bigl((W^{UK}_h)^\top q^N_{t,h}\bigr)\!\cdot c_s$, with $(W^{UK}_h)^\top\in\mathbb{R}^{r\times d_n}$ applied once per query head per step instead of once per cached position.
2. The rotary part never touches the latent: $q^R_{t,h}\!\cdot k^R_s$ is computed against the $d_r$-dimensional cached slice, shared by all heads (as MQA).
3. Concatenate: query $[\hat q_{t,h};q^R_{t,h}]\in\mathbb{R}^{r+d_r}$, key $[c_s;k^R_s]$, value $c_s$ (the first $r$ coordinates of the key). This is an MQA layer with one KV head and head width $r+d_r$ for K and $r$ for V: `F.scaled_dot_product_attention` applies with `k.expand` over heads.
4. Values: $o_{t,h}=\sum_s\alpha^h_{ts}W^{UV}_hc_s=W^{UV}_h\hat o_{t,h}$ by linearity, so the up-projection moves *after* the weighted sum, once per head per step.
5. Output projection: $y_t=\sum_hW^O_ho_{t,h}$. The product $W^O_hW^{UV}_h\in\mathbb{R}^{d\times r}$ *could* be pre-merged, but merged it has $Hdr=469.8$M parameters per layer at V3 dimensions versus $r H d_v + H d_v d=125.8$M for the two factors: **do not merge**; reordering the products is the absorption.
6. Costs per decoded token per layer and per context token: expanded-cache strategy $2H(d_n+d_r+d_v)=81{,}920$ FLOPs and $H(d_n+d_r+d_v)\cdot2$ B $=81{,}920$ B read (intensity 1); latent-cache-with-re-expansion strategy $2rH(d_n+d_v)+2H(d_n+d_r+d_v)=33.6$M FLOPs and $(r+d_r)\cdot2=1{,}152$ B (intensity ≈ 29,000, hopelessly compute-bound); absorbed $2H(2r+d_r)=278{,}528$ FLOPs and 1,152 B (intensity 241.8). All at V3's $H=128$, bf16. Against the A100-80GB ridge (peak bf16 / HBM bandwidth from `d2l.chip_specs()`; datasheet 312 TFLOP/s and 2.039 TB/s give about 153 FLOP/B, to be read from the table by the notebook) the absorbed path sits slightly on the compute side: MLA turns the decode attention from bandwidth-bound to near-ridge.
7. Cache: $(r+d_r)\,b_{kv}$ bytes per token per layer; times 61 layers gives 70,272 B for V3 at $b_{kv}=2$.

*Numerical check cell.* Random $W$ at small dimensions (`d=256, H=8, dn=32, dr=16, dv=32, r=64, rq=96`); seeded; assert (i) explicit versus absorbed outputs: max abs difference $<10^{-10}$ in float64, relative $<10^{-5}$ in float32; (ii) the three strategies' outputs agree to the same tolerance; (iii) the byte counts of `MLACache.bytes()` equal $L(r+d_r)b\,n$; (iv) report the *bf16* deviation of absorbed versus explicit (the tolerance is "to be measured"; the direction to check is that it exceeds the fp32 one because the latent product is a longer chain of bf16 matmuls); (v) with the same weights built into the Transformers `DeepseekV3Attention` (a small config without YaRN: `rope_scaling=None`), assert agreement within $10^{-5}$ in fp32 for the explicit path.

**B. Why RoPE has to be decoupled (`eq_lm_mla_rope`).**

*Statement.* If RoPE rotates the up-projected keys, $k_{s,h}=R_sW^{UK}_hc_s$, no vector $\tilde q$ independent of $s$ exists such that $q_{t,h}^\top k_{s,h}=\tilde q^\top c_s$ for all $s$; absorption would need a position-dependent matrix per cached key.

*Assumptions.* $R_s$ block-diagonal with $2\times2$ rotations at fixed frequencies, $R_t^\top R_s=R_{s-t}$ (`sec_positional-information`), generic $W^{UK}_h$.

*Steps.*
1. $q^\top(R_sW^{UK}c_s)$ with query $R_tq^0$: $(R_tq^0)^\top R_sW^{UK}c_s=q^{0\top}R_{s-t}W^{UK}c_s$.
2. To read it as $\tilde q^\top c_s$ we need $\tilde q_{t,s}=(W^{UK})^\top R_{s-t}^\top q^0$, which depends on $s-t$.
3. Absorbing $W^{UK}$ into the query therefore requires $n$ different $r$-vectors per head per step, $O(n\,r\,d_n)$ work per head: the absorption saving is lost (it is the re-expansion strategy in disguise).
4. Alternatively put a rotation in latent space: we need $R'_s$ with $R_sW^{UK}=W^{UK}R'_s$ for all $s$; this holds only if the column space of $W^{UK}$ is invariant under every rotation, which fails for generic trained weights (the set of $R_s$ for $s=0,1,\dots$ generates a large commutative group acting on $d_n$ dims).
5. Decoupling adds $d_r$ coordinates that carry *only* position: $q^R_{t,h}=R_tW^{QR}_hc^Q_t$ and $k^R_s=R_sW^{KR}x_s$; their dot product is $q^{R0\top}R_{s-t}k^{R0}$, relative by step 1, and the nope part $q^N\!\cdot k^N$ has no rotation. Cost: $d_r$ extra cached elements per token (64 of 576 in V3: 11%).

*Numerical check cell.* Same random weights; (a) explicit path with RoPE applied after the up-projection and a *reference* attention that computes it directly: its output differs from the "absorbed ignoring position" path by a relative error of order one (print it; no threshold, the point is that it is not small); (b) the decoupled explicit and absorbed paths agree to $10^{-5}$; (c) measure the cost of the patched position-dependent absorption at $n=4096$: it is ≈ $n$ times the absorbed cost (print measured/predicted ratio).

**C. Activation-aware joint low-rank (`eq_lm_mla_awsvd`).**

*Statement.* Given inputs $X\in\mathbb{R}^{m\times d}$ with Gram matrix $X^\top X=LL^\top$ and weights $W=[W_K\ W_V]\in\mathbb{R}^{d\times 2d_{kv}}$, the rank-$r$ matrix $\tilde W$ minimizing $\|X(W-\tilde W)\|_F$ is $\tilde W=L^{-\top}\,\mathrm{SVD}_r(L^\top W)$, i.e. $\mathrm{SVD}_r$ in the whitened space; with $X=I$ it is the data-free joint SVD.

*Steps.* (1) $\|X(W-\tilde W)\|_F^2=\operatorname{tr}\bigl((W-\tilde W)^\top LL^\top(W-\tilde W)\bigr)=\|L^\top W-L^\top\tilde W\|_F^2$; (2) $\mathrm{rank}(L^\top\tilde W)=\mathrm{rank}(\tilde W)$ for invertible $L$; (3) Eckart–Young (`sec_mdl-svd-low-rank`) gives the best rank-$r$ approximation of $L^\top W$ as $U_r\Sigma_rV_r^\top$; (4) map back: $\tilde W=L^{-\top}U_r\Sigma_rV_r^\top$, factors $W^D=L^{-\top}U_r\Sigma_r^{1/2}\in\mathbb{R}^{d\times r}$ and $[W^{UK}\ W^{UV}]=\Sigma_r^{1/2}V_r^\top$; (5) the value of the objective is $\sum_{i>r}\sigma_i^2$.

*Check.* Equality of the closed form with brute-force minimization over a random-restart gradient descent on $X$ of shape $2000\times64$ (seeded), and the residual equal to $\sum_{i>r}\sigma_i^2$ to $10^{-8}$ in float64.

#### From-scratch implementation

Budget ≈ 130 lines: `MLAAttention` (≈ 70: projections, RoPE on the rotary slice, `expanded` and `absorbed` forward, causal mask for prefill via SDPA, cache handling), `MLACache` (≈ 15), `mla_decode_cost` (≈ 15), `CrossLayerKV` (≈ 20), `joint_svd_kv` (≈ 15).

| Test | Reference | Tolerance / seed |
|---|---|---|
| Explicit vs absorbed vs re-expand | each other | fp64 $<10^{-10}$; fp32 $<10^{-5}$ rel.; seed 0 |
| Prefill (expanded) then decode (absorbed) vs full-sequence forward | full forward | fp32 $<10^{-5}$ |
| `MLAAttention` vs `DeepseekV3Attention` | Transformers | fp32 $<10^{-5}$ on a small random config, `rope_scaling=None`, interleaved RoPE flag matched |
| Cache bytes of `MLACache` | $L(r+d_r)bn$ | exact |
| `config_stats(DeepSeek-V3).cache_bytes_per_token` | $61\times576\times2$ | exactly 70,272 |
| `CrossLayerKV(share=2)` cache bytes, and `share=1` outputs | half the bytes of `share=1`; a plain `GQAAttention` stack | bytes exact; outputs $<10^{-6}$ (fp32) |
| `joint_svd_kv` at full rank | original $W_K,W_V$ | $<10^{-5}$ rel.; at rank $r$ the residual equals $\sum_{i>r}\sigma_i^2$ |
| `mha_to_mla` at full rank | original GPT-2 logits | $<10^{-4}$ (fp32) |

#### In production

- **Transformers:** `DeepseekV3Attention` implements the *explicit* path (and caches expanded K/V in the default `DynamicCache`); the from-scratch `expanded` mode is its oracle, the `absorbed` mode is checked against it. **vLLM/SGLang** implement absorbed decode with MLA-specific backends (pointers; v5 lists them for 13.x).
- **Kernels.** DeepSeek's [FlashMLA](https://github.com/deepseek-ai/FlashMLA) is the reference kernel library. Its 2026-09-30 release **removed Hopper support** and targets Blackwell (B200) and Ascend for the V4.1 sparse kernels, with the previous commit kept for Hopper (README, fetched 2026-10-28); there is no FlashMLA for A100 or Ada. The notebook therefore uses `F.scaled_dot_product_attention` in MQA form for the absorbed path (head width $r+d_r=576$ for K is outside FlashAttention's supported head sizes: SDPA falls back to the efficient or math backend; print `torch.backends.cuda` flags and the kernel chosen by the profiler). Measured absorbed latencies are therefore an *upper bound* on what a tuned kernel gives; say so on the plot.
- **TransMLA / MHA2MLA code** for conversion of real checkpoints (v5 sources; the notebook's GPT-2 experiment is the from-scratch small-scale counterpart).

#### Headline experiment

- **Hypothesis.** At the same query width (128 heads × 128), absorbed MLA decode has lower per-step latency than 8-KV-head GQA at long context and large batch, where the cache bytes dominate, and *not* at short context or batch 1, where fixed costs and the larger absorbed FLOPs dominate; the crossover context is predicted by the 5.2 model to within a stated factor. The 5.2-model prediction (derived, not measured): GQA per context token per layer is bandwidth-bound at $4{,}096$ B per sequence-token; MLA is bounded by $\max(278{,}528/P,\;1{,}152/\beta)$; on the A100 row of `chip_specs()` this predicts MLA ≈ 2× faster per context token (to be recomputed from the table), and on the build-box row *to be measured*.
- **Setup.** One layer at V3 dimensions (random weights, bf16, no checkpoint), arms: GQA-128:8 (`d2l.GQAAttention`, d_h=128, `enable_gqa=True`), MLA expanded-cache, MLA latent-re-expand, MLA absorbed. Context $n\in\{1\text{K},2\text{K},4\text{K},8\text{K},16\text{K},32\text{K}\}$ (the largest that fits the arm), batch $b\in\{1,4,16\}$ (largest that fits), cache prefilled with random data (the latency does not depend on values), 50 warm-up and 200 timed steps with CUDA events; CUDA graphs on (`torch.compile(mode='reduce-overhead')` or manual capture, recorded). Tier T1 (build box); the A100 row of the prediction uses `chip_specs()`.
- **Procedure.** Time each arm; fit time $=\tau_0+\tau_1\,n\,b$ per arm by least squares; compare $\tau_1$ with the 5.2 prediction and report the achieved fraction of the roofline.
- **Metrics:** per-step latency (median over 200 steps, 5 repetitions of the whole sweep, report the median across repetitions and the min–max range); cache bytes; achieved FLOP/s and GB/s (counter-based: formula FLOPs and bytes divided by time); maximum batch that fits at $n=32$K.
- **Plot (`fig_lm_mla_latency`).** x: context length (log), y: latency per step (ms, log), one panel per batch size, one line per arm, with the predicted line dashed and shaded run-to-run range; second panel: cache bytes per token (log) per arm, with the max-batch-that-fits annotated.
- **Expected outcome.** Direction as in the hypothesis; the crossover and the size of the gap are to be measured. Sources for the direction: DeepSeek-V2 reports a large reduction of KV cache and an increase of maximum generation throughput (see the paper; the notebook does not quote the number).
- **Noise check.** Five full repetitions; a latency difference counts only if the arms' min–max ranges do not overlap; the fitted $\tau_1$ has a bootstrap interval over the repetitions.
- **Second experiment (T1, secondary).** Conversion curves on GPT-2: rank $r\in\{32,64,128,256,512\}$ (cache elements per token per layer $r$ versus 1,536 originally) for (i) per-sequence oracle SVD (the book's), (ii) data-free joint SVD, (iii) activation-aware SVD from a 64K-token calibration batch, (iv) mean-pooled GQA at equal cache ($g\in\{1,2,3,4,6,12\}$ groups: $2\cdot64\cdot g$ elements); each with 0 and 200 recovery steps (AdamW, lr $2\times10^{-5}$, R2 slice), evaluated on eight 1,024-token passages (R3 and *The Time Machine*) with `d2l.paired_bootstrap` over passages; 3 recovery seeds at $r=128$ only. Hypothesis (direction): at equal cache, joint SVD (iii) degrades less than mean-pooled GQA, and recovery steps close part of the gap; the magnitudes are to be measured. A difference smaller than the passage-bootstrap interval is a tie.
- **Third (optional) experiment.** MLA versus GQA versus CLA-2 trained from scratch at width 256 / 6 blocks with matched cache elements per token (MLA $r+d_r$ equal to $2H_{kv}d_h$): 2 seeds × 3 arms; tie unless the gap exceeds the seed range; the sources' claims are quoted as theirs.
- **T1 reduction:** all of the above is T1. **Artifacts committed:** `lm-arch-mla-conversion`.

#### Figures and tables

| Label | Content | Cell |
|---|---|---|
| `fig_lm_mla_dataflow` | MLA dataflow with the explicit and absorbed routes (drawn with matplotlib patches; the absorbed route in a different color) | `draw_mla()` |
| `tab_lm_mla_strategies` | FLOPs, bytes, intensity per context token per layer for the three strategies at V3 dims and at Kimi-K2 dims (64 heads) | `mla_decode_cost` |
| `fig_lm_mla_latency` | headline | `plot_latency()` |
| `fig_lm_mla_conversion` | loss vs cache elements for (i)–(iv), with and without recovery; bootstrap bars | `plot_conversion()` |
| `tab_lm_mla_cache_released` | cache bytes per token of released models (GQA, MLA, hybrid) from `config_stats` | cell 1 |

#### Pitfalls and misconceptions

- "MLA stores a latent, so it is cheap": only the *absorbed* order of operations is cheap; caching the latent and re-expanding each step costs ≈ 120× the absorbed attention FLOPs at V3 dimensions (33.6M versus 278,528 per context token per layer).
- Absorption is a reordering of matmuls, not a pre-merge of weights (merged $W^OW^{UV}$ is 3.7× larger than the factors).
- With 128 heads the absorbed path has intensity ≈ 242 FLOP/B at bf16: decode becomes *compute-bound* on bandwidth-rich chips; fewer heads (Kimi-K2: 64) lower it.
- `num_key_value_heads` (128 in the V3 config) is a compatibility field; the cache is $(r+d_r)$.
- `mla_use_nope` does not remove the rope slots from the weight shapes; read `kv_a_proj_with_mqa`'s shape (576), not the flag.
- bf16 absorbed attention accumulates error through a longer matmul chain: validate in fp32/fp64 first, then quantify the bf16 deviation.
- Conversion without recovery fine-tuning is a *capability probe*, not a deployable model; and joint SVD of weights ignores the input distribution (hence the activation-aware variant).
- CLA shares K/V *after* any per-layer norm and rotation choices: with RoPE the borrower must rotate its queries consistently with the owner's keys.

#### Summary (draft bullets)

- MLA caches one $r$-dimensional latent plus $d_r$ rotary channels per token and layer (576 in V2/V3/K2/K3); at V3's dimensions that is 71× less than the K/V it represents.
- Decode attends in the latent space by absorbing $W^{UK}$ into the query and $W^{UV}$ into the output order; the result is exactly the explicit one.
- RoPE after the up-projection would make absorption position-dependent, so a small decoupled rotary slice carries position.
- Absorbed MLA is compute-heavier per token (3.4× attention FLOPs at V3 dimensions) and far lighter in bytes; whether latency improves depends on batch, context and chip.
- Cross-layer sharing divides the cache by the sharing factor; V4.1's `kv_source_layer_ids` is the non-uniform case.
- A trained MHA/GQA checkpoint can be converted by joint low-rank factorization plus a short recovery run; the sources report 92–93% cache reduction on a 7B model.

#### Exercises

*Check.*

1. **Bytes of an MLA model.** From `kv_lora_rank`, `qk_rope_head_dim` and `num_hidden_layers` compute the cache bytes per token of DeepSeek-V2 (60 layers), V3, GLM-5 (78 layers) and check against 69,120, 70,272 and 89,856. Why does GLM-5 have a bigger cache than V3 despite the same $r=512$? *Sketch:* 78 layers versus 61 (78 × 576 × 2 = 89,856).
2. **Which strategy?** For $H=16$, $d_n=d_r=d_v=64$, $r=128$ compute the intensity of the three decode strategies; for which $H$ does the absorbed path stop being compute-bound on a chip with ridge point 150 FLOP/B? *Sketch:* absorbed FLOPs $2H(2r+d_r)=2H\cdot320$, bytes $(r+d_r)\cdot2=384$; intensity $=H\cdot1.67$; crosses 150 at $H\approx90$.

*Derive.*

3. **MLA versus GQA at equal cache.** Show that for any GQA layer with $H_{kv}$ KV heads of width $d_h$ there is an MLA layer with $r=2H_{kv}d_h$ whose attention can represent the same function (rank argument), and say what the converse fails to hold. *Sketch:* GQA's keys and values are $[K\ V]=X[W_K\ W_V]$, a rank-$\le 2H_{kv}d_h$ factorization; set $W^D=[W_K\ W_V]$, $W^{UK}_h$ selecting the block of $h$'s group, no-op for the rest; the converse fails because an MLA layer's per-head key/value matrices need not share a block structure.
4. **Decoupling cost.** If $d_r$ rotary channels are added per token, the cache is $(r+d_r)$. With a nope-MLA (no rotation) the cache is $r$. Derive the cache saving of removing rotation as a fraction and evaluate for V3 ($r=512,d_r=64$); then argue why Kimi Linear can afford NoPE on its MLA layers but a pure-MLA model such as V3 could not. *Sketch:* $d_r/(r+d_r)=11.1\%$; the three KDA layers per MLA layer supply order information through convolution and decay; a pure-attention stack has no other source of position (NoPE decoders can recover relative position from the causal mask alone only weakly, `sec_positional-information`).

*Measure.*

5. **Predict the crossover.** Using the 5.2 model and the build-box row of `chip_specs()`, predict the context at which absorbed MLA overtakes GQA-128:8 at batch 4; then measure it and report the ratio of predicted to measured crossover with the repetition ranges.
6. **Partial RoPE on R1.** On Qwen3-0.6B, rank the 32 rotary frequency pairs per head by the contribution of each pair to the attention scores on 64K tokens of R2 text; keep the top 8, zero the rotation on the rest, and measure the perplexity change *before any fine-tuning*; which pairs survive (low or high frequencies)? Compare with the MHA2MLA paper's observation (to be read in the paper; mark your statement of its finding "(unverified)" until checked).

*Open.*

7. **A cache budget.** Choose a model from the design table and a context of 128K; you have 20 GB of HBM for the cache at batch 8. Design the cheapest (in quality, by your own estimate and a 14.4/14.3-style experiment) combination of GQA width, MLA rank, CLA sharing and window fraction that fits; write the pre-registered decision rule first.

#### Further reading

- Papers (v5): [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (§2.1: MLA, decoupled RoPE, the cache comparison table); [Cross-Layer Attention](https://arxiv.org/abs/2405.12981) (abstract: another 2× KV reduction over MQA at nearly the same accuracy, 1B and 3B models trained from scratch); [MHA2MLA](https://arxiv.org/abs/2502.14837) (partial-RoPE selection and joint SVD; Llama-2-7B −92.19% KV, −0.5% LongBench, 0.3–0.6% of data).
- *New (verified):* [TransMLA](https://arxiv.org/abs/2502.07864) (title: "TransMLA: Multi-Head Latent Attention Is All You Need"; GQA→MLA conversion, 93% of the KV cache of LLaMA-2-7B compressed, 6B tokens of fine-tuning, 10.6× speed-up at 8K context as the paper claims); [You Only Cache Once](https://arxiv.org/abs/2405.05254) (decoder–decoder layer sharing that V4.1's CED builds on; Codex 2.5).
- Teaching: the DeepSeek-V2 cache-reduction discussion; [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) for the roofline arithmetic; Smola's efficiency lecture (Codex 3.4; the matrix-absorption and byte-budget examples) at alex.smola.org/posts/45-mlss-efficiency/ (verified in v5's reference list).
- Code: [FlashMLA](https://github.com/deepseek-ai/FlashMLA) (HTTP 200; current release Blackwell/Ascend, old commit for Hopper); [`modeling_deepseek_v3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py) (explicit MLA oracle).

#### Risks and open decisions

| Risk | Fallback |
|---|---|
| `scaled_dot_product_attention` has no fused kernel for head width 576 on the build box (Ada) | the math/efficient backend; label the plot "SDPA (unfused)"; also report the arithmetic-only prediction |
| The build-box row of `chip_specs()` is missing or lacks the Ada GPU | measure peak bandwidth with a `torch.empty_like` copy and peak FLOPs with a large matmul in the notebook; record both |
| Transformers' `DeepseekV3Attention` signature changed (v5.x) | the oracle cell is the small config test only; pin the Transformers version in the notebook and record it |
| GPT-2 conversion shows no gain from activation-aware SVD at the tested ranks | report the tie; the derivation and the equality check stand alone |
| Kimi's `mla_use_nope` rope-slot semantics unresolved | marked "(unverified)"; the notebook only states shapes |
| 128-head layer at batch 16, n=32K does not fit the Ada GPU for the expanded-cache arm | drop that cell; the "max batch that fits" panel reports it |

### 14.4 Refinements of softmax attention

- **Label:** `sec_lm_attention_refinements` · **Tier:** T2 (reduced T1 configuration: `cfg.scale='t1'`: `d2l.GPT` width 256, 6 blocks, context 256, 3 seeds per variant) · **Budget:** T2 ≈ 3.5 h on 1× A100 40 GB: 4 variants plus 2 extra baseline seeds = 6 training runs capped at ≈ 30 min each by token budget (calibrated from R2's measured tokens/s) plus ≈ 20 min of evaluation; T1 ≤ 15 min (4 variants × 3 seeds × ≈ 1 min, ≈ 2 min of evaluation, the released-model probes of cell 8) · **Peak memory:** T2 design target ≤ 34 GiB (14.2's micro-batch rule if exceeded); T1 ≤ 8 GiB (probes: Qwen3-1.7B and Qwen3.5-2B at bf16 with eager attention at 2K tokens ≤ 12 GiB) · **Codex units:** new in v5 (Codex 2.5's local/global discussion is folded in; Mistral 7B and StreamingLLM carried over as sources)
- **Builds on (book):**
  - `sec_kv-cache` (`subsec_kv-cache-sinks`): the GPT-2 measurement of attention mass on the first token (a third to a half of the mass from mid-stack up, over one passage), the sliding window with 0, 1 or 4 sink tokens, and the remark that gpt-oss gives every head a learned sink logit. **Taken as given:** *that* sinks exist in GPT-2 and *that* a window without them fails. 14.4 does not repeat the GPT-2 experiment; it explains why, and trains the alternatives.
  - `sec_positional-information`: RoPE, ALiBi, NoPE, the extrapolation protocol (train at $L$, test at $4L$), `TinyCharLM` (`#@save`). **Taken as given:** that NoPE can work in a causal decoder (Kazemnejad et al.) and the protocol. 14.4 adds partial rotary and NoPE-layer interleaving at R2 scale; YaRN and position interpolation are 4.6's.
  - `sec_transformer-block`: QK-norm, the `attn_factory` seam; `sec_attention-at-scale`: sliding-window masks.
- **Builds on (this part):**
  - 14.1: `d2l.config_stats` for cache bytes of window layouts (`cache_bytes_bounded`), the design table's position and attention-mix columns.
  - 14.2: `d2l.train_lm_steps` (the shared small-scale trainer) and the R2 data path. Imports `d2l.load_reference_model` (4.2) and R2's validation set (`lm-fwedu-val`, 3.6).
  - 4.2 (R2), 4.3 (`sec_lm_instabilities`: attention-logit growth, QK-norm; the gated attention paper claims higher stable learning rates, so 14.4's optional LR sweep reuses 4.3's divergence criterion), 4.6 (long-context data; not required), 1.6 (activations in the residual stream).
  - 1.1/1.2: `d2l.load_hf` for the released-model probes.
- **Exports (`#@save`):**
  - `d2l.RefinedAttention(d, num_heads, num_kv_heads, head_dim, *, rope=True, rope_frac=1.0, rope_base=10000.0, window=None, qk_norm=True, sink=False, gate=None)` — GQA attention with: a learned per-head sink logit (`sink=True`, gpt-oss style); an output gate (`gate='sigmoid'` elementwise per head dimension from a doubled query projection as in Qwen3.5, `'swish'` as in `output_gate_type`, or `None`); partial rotary (`rope_frac<1`, half-split convention as Transformers; the book's `_rope` uses adjacent pairs, so the class includes both with a flag and a test that they are related by a fixed permutation); `rope=False` for NoPE layers; sliding `window`.
  - `d2l.attention_with_sink(q, k, v, sink, mask)` — the reference implementation (explicit softmax over the concatenated logit column, then drop the sink column).
  - `d2l.refined_gpt(vocab, d, num_heads, num_kv_heads, num_blks, *, sink=False, gate=None, rope_frac=1.0, nope_every=0, window=None, window_every=0, max_len=...)` — a `d2l.GPT` whose blocks use `RefinedAttention` via `attn_factory`.
  - `d2l.sink_metrics(model, batch) -> dict` — per layer and head: attention mass on position 0 (queries from position 64), mean sink probability, max-to-median residual magnitude, fraction of gate values below 0.1.
  - `d2l.repeat_copy_task(vocab, m, seed)` and `d2l.eval_copy(model, lengths)` — random-token copy test (an induction probe) whose accuracy is a function of the lookback distance; no training on it.
  - `d2l.diff_attention(q1, k1, q2, k2, v, lam)` — differential attention (the "in brief" item; ≤ 15 lines).
  - Data: `lm-arch-refine-logs` (JSONL: variant, seed, tokens, validation loss, per-layer metrics, copy accuracy by distance, gate statistics).
- **Models and data:** R2 (4.2) and its FineWeb-Edu slice (ODC-By); released-model probes (T1): `Qwen/Qwen3-1.7B` (Apache-2.0; 28 layers, GQA 16:8 of width 128, hidden 2048, tied embeddings, 2,031,739,904 parameters per the Hub API; softmax attention with QK-norm and no sink or gate; 114,688 B/token of KV cache at bf16) versus `Qwen/Qwen3.5-2B` (Apache-2.0; 6 gated full-attention layers, partial RoPE 25%, 18 Gated DeltaNet layers; 14.1's fact sheet); optional T2 probe `openai/gpt-oss-20b` (Apache-2.0; learned sinks; loads with MXFP4 dequantized to bf16 on A100, ≈ 42 GB, a statement the author must confirm in the model card; that does not fit one A100 40 GB, so the probe runs on a T2 pair with `device_map='auto'` (about 21 GB per GPU) or is skipped).
- **Packages:** `torch>=2.8`, `transformers>=5.0` (oracle `GptOssAttention`; eager attention for `output_attentions=True`).

#### Narrative arc

1. **The question.** Softmax attention must put its probability somewhere. `sec_kv-cache` measured that GPT-2 puts a third to a half of the mass on token 0 and that deleting that token wrecks perplexity. *Is that a bug to be designed out, and what does each fix cost?* The opening cell prints, for the released configurations (from 14.1's table): which models carry a sink logit (gpt-oss: `sinks` parameter of shape `[num_attention_heads]` per layer), an output gate (Qwen3-Next, Qwen3.5/3.6/3.8: `attn_output_gate true` or `output_gate_type`; Kimi K3: `mla_use_output_gate true`; MiniMax-M3: `attention_output_gate false`), partial rotary (`partial_rotary_factor`: Qwen3.5 0.25, GLM-4.5 0.5, MiniMax-M3 0.5), NoPE layers (Granite 4, `sec_hybrids`; K3's and Kimi Linear's MLA layers), windows (gpt-oss 128 on alternate layers), and which do none of these (Qwen3, MiniMax-M2). *One measured symptom:* the first-token attention mass and the max-to-median residual magnitude of Qwen3-1.7B versus Qwen3.5-2B on the same 2K-token passage (to be measured; the cell is the T1 probe).
2. **Where the mass goes: sinks and massive activations** (`subsec_lm_attn_sinks`). The convex-hull argument (Derivation 1): a head whose output should be near zero needs a key whose value is near zero, and the margin in logits must grow like $\ln n$. Then the evidence: massive activations (Sun et al.: a few residual-stream activations are up to 10⁵ times larger than the rest, constant across inputs, behaving like bias terms, and concentrate attention probability on their tokens) and the alternative reading (Barbero et al.: sinks let a model avoid over-mixing). The honest statement: the convex-hull argument gives a *mechanical* reason, not a proof that this is why trained models do it. The book's measurement of the first-token mass is *recalled in one paragraph*, not re-run.
3. **Three ways out** (`subsec_lm_attn_fixes`). (a) *Learned sink logit* (gpt-oss): a per-head scalar $z$ that adds an extra column to the logits and is dropped after the softmax (Derivation 2); (b) *output gate* (Qwen): $o\odot\sigma(g(x))$ (Derivation 3); (c) *differential attention* (brief): $\mathrm{softmax}(q_1k_1^\top)-\lambda\,\mathrm{softmax}(q_2k_2^\top)$, which can express negative weights and cancel common-mode noise; unit test: $\lambda=0$ recovers standard attention. A T0 toy (below) shows the three mechanisms on a task whose correct output is zero for half the queries.
4. **Positions at scale** (`subsec_lm_attn_positions`). Partial RoPE: rotate $r$ of $d_h$ channels; the unrotated channels are position-free content channels. Table `tab_lm_attn_positions` from configs: $\theta$, rotated channels, the number of rotated pairs whose wavelength $2\pi\theta^{2i/r}$ exceeds the maximum context (a pair with wavelength beyond the context barely rotates: it behaves as NoPE), for Qwen3-8B, Qwen3.5-2B, GLM-4.5, gpt-oss-120b (YaRN factor 32 noted), Kimi K3 (MLA, nope). Pre-check: Qwen3.5-2B ($\theta=10^7$, $r=64$): 10 of 32 pairs have wavelength above 262,144 tokens. Interleaving NoPE layers: every fourth layer without RoPE (SmolLM3; Llama 4 per `tab_modern-recipe`'s prose; the "Rope to Nope and Back Again" hybrid), and what each buys for extrapolation (hypothesis only; measured below).
5. **Local–global layouts** (`subsec_lm_attn_layouts`). Window size $w$ and the fraction $f_g$ of global layers as a dial on the cache: $\text{bytes}(n)=2H_{kv}d_hb\,L\,[f_g\,n+(1-f_g)\min(n,w)]$. Evaluate for gpt-oss-120b ($f_g=\tfrac12$, $w=128$: 36,864 B/token growing, 4,718,592 B bounded; 14.1's pre-check) and the Gemma 3 report's 5:1 local:global with window 1024 (value from v5's source; its config is not read here). What the dial costs in recall: the receptive-field argument of `sec_attention-at-scale` (`subsec_attn-receptive-field`) says information crosses $L_{\text{local}}\cdot w$ tokens through local layers; the global layers are the recall layers (`sec_hybrids`'s lesson in another place).
6. **The experiment** (`subsec_lm_attn_experiment`): four variants at R2 scale (below).
7. **What the released models do** (`subsec_lm_attn_released`). First-token mass and residual magnitude of Qwen3-1.7B versus Qwen3.5-2B (T1) and optionally gpt-oss-20b (T2 pair), using `sink_metrics` and the probes in cell 8. Summarize what *can't* be concluded: the models differ in many other ways (data, recipe, hybrid layers); a difference is a pointer, not an ablation. Contrast with the Gated Attention paper's own controlled claim (30 variants, 15B MoE and 1.7B dense models, 3.5T tokens; gating mitigates attention sink and improves length extrapolation, per its abstract).

#### Derivations

**1. Why a head needs a sink, and how large its margin must be (`eq_lm_sink_hull`, `eq_lm_sink_margin`).**

*Statement.* For attention weights $\alpha\in\Delta^{n-1}$ and values $v_j\in\mathbb{R}^{d_h}$, $\min_{\alpha\in\Delta}\|\sum_j\alpha_jv_j\|=\operatorname{dist}(0,\operatorname{conv}\{v_j\})$. If all values lie in the open half-space $\{v:\langle u,v\rangle\ge m\}$, $m>0$, no weighting yields $\|o\|<m$. If one key has value $0$ and the others have $\|v_j\|\le V$, a logit margin $z-s\ge\ln(nV/\varepsilon)$ over $n$ keys with equal logits $s$ yields $\|o\|\le\varepsilon$.

*Assumptions.* Single head, single query; softmax weights $\alpha_j\propto e^{s_j}$; the "equal logits $s$" case is the worst case for the margin.

*Steps.*
1. $o=\sum\alpha_jv_j$ is a point of the convex hull; its norm is at least the distance from the origin to the hull.
2. Half-space case: $\langle u,o\rangle=\sum\alpha_j\langle u,v_j\rangle\ge m$, so $\|o\|\ge m$.
3. With a sink $v_*=0$ and real keys at logit $s$: $o=\dfrac{e^s\sum_{j\le n}v_j}{ne^s+e^z}$, so $\|o\|\le \dfrac{nVe^s}{ne^s+e^z}\le nVe^{s-z}$.
4. Requiring $nVe^{s-z}\le\varepsilon$ gives $z-s\ge\ln(nV/\varepsilon)$: the margin grows logarithmically with the number of competing keys (and context).

*Numerical check cell.* (a) Random values in a half-space ($n=64$, $d_h=16$, seed 0): solve $\min\|V^\top\alpha\|^2$ on the simplex with `scipy.optimize.minimize(method='SLSQP')`; assert the minimum is $\ge m^2-10^{-8}$; with an added zero vector the minimum is 0. (b) For $n=2^k$, $k=1..14$ and $V=1$: bisect on $z-s$ until $\|o\|\le\varepsilon=0.01$; assert the bisected margin is $\le\ln(nV/\varepsilon)$ and differs by a constant (print the offset; the slope in $\ln n$ is 1 to two decimals).

**2. A learned sink logit is an extra key with zero value (`eq_lm_sink_equiv`).**

*Statement.* Let $s_{tj}$ be the logits of a head and $z$ a learned scalar. Softmax over $[s_{t1},\dots,s_{tt},z]$ with the last column's value set to zero gives the same output as appending a key $k_*$ with $q_t^\top k_*/\sqrt{d_h}=z$ and value $0$. The output equals $g_t\,\bar o_t$ with $\bar o_t$ the ordinary softmax output over the real keys and $g_t=\sigma\bigl(\operatorname{LSE}_j(s_{tj})-z\bigr)$.

*Steps.* (1) $p_{tj}=e^{s_{tj}}/(Z_t+e^z)$, $Z_t=\sum_je^{s_{tj}}$; (2) $o_t=\sum_jp_{tj}v_j=\frac{Z_t}{Z_t+e^z}\sum_j\frac{e^{s_{tj}}}{Z_t}v_j$; (3) $\frac{Z_t}{Z_t+e^z}=\frac{1}{1+e^{z-\log Z_t}}=\sigma(\log Z_t-z)$. (4) A query-independent $z$ cannot be produced by a dot product with a fixed key for all queries; the equivalence to an extra key requires either a bias-like key coordinate or reading $z$ as a per-head constant (the Transformers implementation does the latter).

*Check.* (a) `d2l.attention_with_sink` versus (i) a direct softmax over the concatenated logits (exact), (ii) $\sigma(\mathrm{LSE}-z)\cdot$ standard SDPA output, assert $<10^{-6}$ in fp32; (b) versus Transformers' `GptOssAttention` with a small `GptOssConfig` (4 heads, `head_dim 16`, `sliding_window 8`) and copied random weights: $<10^{-5}$ in fp32 (the Transformers implementation subtracts the row maximum before the softmax; the tolerance covers it).

**3. An output gate lets a head emit zero without a sink (`eq_lm_gate_null`).**

*Statement.* With $o_t=\bar o_t\odot\sigma(W_gx_t)$ (elementwise per head dimension) a head can output (near) zero for any attention distribution, so no key needs to absorb mass; the pattern can stay spread. The learned-sink head is the special case of a *scalar, score-determined* gate $\sigma(\mathrm{LSE}-z)$ (Derivation 2): the gate in the sink head cannot depend on $x$ except through the scores.

*Steps.* (1) $\|o_t\|\le\|\bar o_t\|\max_i\sigma(g_i)$; (2) driving $W_gx_t\to-\infty$ gives $o_t\to0$ for any $\alpha$; (3) hence the loss gradient no longer pushes the logits to concentrate on a null key; (4) conversely the gate adds parameters ($Hd_h\cdot d$ per layer: the doubled `q_proj` of Qwen3.5-2B: $+2048\cdot2048$ per full-attention layer, derivable from 14.1's D1) and one elementwise multiply.

*Check (the T0 toy).* A single `RefinedAttention` layer, $d=32$, $H=2$, trained 500 steps (AdamW, seed 0,1,2) on random sequences of 32 vectors where the target is the mean of values for queries flagged "relevant" and exactly zero for the others. Measure: final loss and, on the "irrelevant" queries, the maximum attention probability over keys (concentration) for baseline, sink, gate. *Expected (direction):* the baseline's attention concentrates on one key (high max probability) on irrelevant queries; the gate variant reaches low loss with a spread pattern; the sink variant uses its sink column (mass on the sink probability near 1). Magnitudes to be measured, 3 seeds each, reported as ranges.

**4. Partial rotary and the wavelength budget (`eq_lm_pr_wavelength`).** *Statement:* with $r$ rotated channels (so $r/2$ pairs) and base $\theta$, pair $i$ rotates by $\theta^{-2i/r}$ radians per position, wavelength $\lambda_i=2\pi\theta^{2i/r}$; a pair is *effectively static* over a context $n$ if $\lambda_i>n$. *Steps:* (1) $\lambda_i>n\iff i>\frac{r}{2}\log_\theta(n/2\pi)$; (2) the count of static pairs is $\lceil r/2-\frac r2\log_\theta(n/2\pi)\rceil$ clipped to $[0,r/2]$; (3) rotated-but-static pairs are functionally NoPE channels. *Check:* count via the closed form versus by brute force over $i$ for the five released configs of cell 4 (exact integers).

#### From-scratch implementation

Budget ≈ 120 lines: `RefinedAttention` (≈ 70: doubled-query gate path, sink column, partial rope both conventions, NoPE flag, window mask, GQA via SDPA with an explicit mask because SDPA has no sink argument; use `enable_gqa=True`; for the sink, compute the softmax manually in fp32 over the concatenated column, or use the log-sum-exp identity of Derivation 2 to *reuse* the fused kernel: $o=\sigma(\mathrm{LSE}-z)\cdot\mathrm{SDPA}$ with LSE from `torch.logsumexp` of the scores, which is a documented trick, tested against the explicit version), `attention_with_sink` (≈ 12), `diff_attention` (≈ 15), `sink_metrics` (≈ 25), `repeat_copy_task` (≈ 12).

| Test | Reference | Tolerance / seed |
|---|---|---|
| `attention_with_sink` vs concatenated softmax; vs LSE-gate identity | each other | fp32 $<10^{-6}$ |
| `RefinedAttention(sink=True)` vs `GptOssAttention` | Transformers, small config, copied weights | fp32 $<10^{-5}$ |
| `RefinedAttention(gate='sigmoid')` vs a literal transcription of `Qwen3_5Attention` (doubled `q_proj`, `chunk`, `sigmoid`) | the module in `modeling_qwen3_5.py` on a small config | fp32 $<10^{-5}$ |
| Partial rope, half-split | `apply_rotary_pos_emb` from `modeling_qwen3_5.py` with `rotary_dim = rope_frac·d_h` | fp32 $<10^{-6}$ |
| Interleaved vs half-split | related by a fixed channel permutation (the loader of 1.2 applies the same permutation) | exact up to permutation |
| `rope=False` layer equals no rotation | SDPA on unrotated q,k | exact |
| `diff_attention(lam=0)` | standard attention | exact |
| Causality with `window` | mask equals band mask of `sec_attention-at-scale` | exact |
| Param count of variants | `config_stats` of the same layout | exact |

#### In production

- **Transformers:** `GptOssAttention` (sinks are an `nn.Parameter` of shape `[num_attention_heads]` concatenated as a logit column, dropped after the softmax; the `s_aux` argument of the attention interface), `Qwen3_5Attention` (doubled query projection, `sigmoid(gate)` on the attention output, zero-centered RMSNorm for QK-norm, partial RoPE of 25% in `rotate_half` convention). The from-scratch layer is the oracle for both.
- **Kernels:** FlashAttention-style kernels accept `window_size`; sinks are supported through `s_aux`/`sink` arguments in recent kernel releases (gpt-oss reference code uses a Triton kernel; "which kernels support sinks on A100" is **not confirmed** here: the author checks the `flash-attention` and `gpt-oss` repos; the LSE-gate identity makes the fused-kernel route available without kernel support).
- **Serving:** a window layer's cache is a ring buffer of $w$ tokens (`cache_bytes_bounded`); a sink logit costs nothing in the cache; a gated layer stores the same K/V (the gate is applied to the output).

#### Headline experiment

- **Hypothesis (directional, from sources).** Relative to the baseline GQA+RoPE+QK-norm: the learned sink and the output gate *reduce first-token attention mass and the max-to-median residual magnitude* (the gate paper reports that gating mitigates attention sink and massive-activation concentration; gpt-oss ships sinks); partial RoPE with interleaved NoPE layers may improve accuracy beyond the training length (the "Rope to Nope" paper reports gains for a hybrid of local RoPE and global NoPE attention in long-context tasks), and validation loss changes are small (within the noise at R2 scale is plausible: the gate paper reports improvements at 1.7B dense and 15B MoE scale; the effect at R2 scale is **to be measured**).
- **Setup (T2).** R2 architecture and recipe (4.2), variants: **B** baseline (GQA, full RoPE, QK-norm); **S** `sink=True`; **G** `gate='sigmoid'`; **P** `rope_frac=0.25` on all layers and `rope=False` on every fourth layer. Training context $L$ from R2 (1,024 or R2's value), the same tokens and seed 0 for the four variants, and seeds 1 and 2 for B only (to measure baseline noise $\sigma$). Token budget identical across variants; data, order and learning rate schedule identical.
- **Procedure.** (1) Train; (2) evaluate validation loss; (3) `sink_metrics` on 64 held-out sequences: mean first-token attention mass per layer, max-to-median residual magnitude per layer, gate-value distribution for G; (4) length extrapolation: validation loss by position on sequences of $4L$ tokens without any scaling (pure extrapolation) and copy-task accuracy at lookback distances $\{L/4,L/2,L,2L,4L\}$ using `repeat_copy_task`; (5) report differences from B in units of $\sigma$.
- **Metrics:** validation loss; first-token mass; residual max/median; copy accuracy versus distance; loss at positions $>L$ relative to position $\le L$; throughput (tokens/s) and parameters (G adds $H d_h d$ per layer).
- **Plot (`fig_lm_attn_variants`).** Four panels: (i) validation loss by variant with the baseline seed range shaded; (ii) first-token attention mass per layer, one line per variant; (iii) max/median residual magnitude per layer (log y); (iv) copy accuracy versus lookback distance (log-2 x) with a vertical line at $L$.
- **Expected outcome.** Direction as above; sizes **to be measured**. *Not predicted:* which variant wins on loss. An outcome in which all four variants tie on loss but differ in (ii)–(iv) is a fully acceptable result and is the one the sources most suggest for small scale.
- **Noise check.** Baseline seeds 0, 1, 2 give $\sigma$ for loss and for each metric; differences below $2\sigma$ are reported as ties; for (ii)–(iv) the 64 evaluation sequences give a bootstrap interval; no conclusion from a single seed.
- **T1 reduction (`cfg.scale='t1'`).** `d2l.GPT` width 256, 6 blocks, context 256, R2's FineWeb-Edu 20M-token slice, 3 seeds per variant (12 runs, ≈ 1 min each); extrapolation to 1,024; the same metrics; a smoke test of every metric and plot. **Artifacts committed:** `lm-arch-refine-logs`.
- **Secondary (T1/T2 probes).** First-token mass and residual magnitude of released models (cell 8) with the explicit disclaimer above.

#### Figures and tables

| Label | Content | Cell |
|---|---|---|
| `tab_lm_attn_released` | which released models carry sink / gate / partial rope / NoPE / window (from configs) | cell 1 |
| `fig_lm_sink_margin` | bisected logit margin versus $\ln n$ (Derivation 1) with the bound | `plot_margin()` |
| `fig_lm_attn_toy` | T0 toy: loss and attention concentration for baseline, sink, gate | `plot_toy()` |
| `tab_lm_attn_positions` | $\theta$, rotated channels, static pairs for released configs | `positions_table()` |
| `fig_lm_attn_variants` | headline (4 panels) | `plot_variants()` |
| `fig_lm_attn_layouts` | cache bytes vs context for layouts ($f_g$, $w$) including gpt-oss-120b | `plot_layouts()` |

#### Pitfalls and misconceptions

- A sink is not necessarily the first token; it is whatever token the model makes a no-op (a delimiter, or position 0 when the first token's value is small). `sink_metrics` measures position 0 only; say so.
- Removing the first token from a window cache moves the softmax's mass onto tokens the head had refused; retaining *positions* matters (book), and the sink *logit* removes the need.
- "Gated attention" and "gated DeltaNet" share a word and not a mechanism: the first is an output gate on softmax attention, the second a state-update gate (14.6).
- `attn_output_gate true` doubles the query projection: parameter counts and `q_proj` shapes change (Qwen3.5-2B: `[4096, 2048]` for 8 heads of 256).
- Partial RoPE's unrotated channels are *not* NoPE heads: they sit inside every head; NoPE *layers* drop rotation in the whole layer.
- The book's `_rope` pairs adjacent channels; Transformers rotates halves; loading released weights needs a permutation (1.2); partial rotary applies to the first `rope_frac · head_dim` channels in the Transformers convention.
- Extrapolation tests without scaling measure the architecture; production long-context models use YaRN/interpolation on top (4.6); do not read the former as the latter.
- A variant that improves $4L$ copy accuracy at the cost of $L$ loss is a trade-off, not a win; plot both.

#### Summary (draft bullets)

- Softmax forces each head to place its probability mass somewhere; a head with nothing to say needs a key with (near) zero value, with a logit margin that grows like $\ln n$.
- A learned sink logit is an extra zero-value key (equivalently a scalar gate $\sigma(\mathrm{LSE}-z)$ on the head's output); an output gate generalizes it and removes the pressure to concentrate attention.
- Released models choose differently (gpt-oss: sinks; Qwen3-Next/3.5/3.6/3.8: output gates; Qwen3 and MiniMax-M2: neither) and nothing in the configs says which was better.
- Partial RoPE leaves content channels free, and low-frequency rotated pairs behave as NoPE; interleaving NoPE layers is a coarser version of the same idea.
- Local–global layouts are a dial on cache bytes, $f_g n+(1-f_g)\min(n,w)$; recall lives in the global layers.
- At R2 scale the measurable differences are in attention mass, activation outliers and extrapolation, not necessarily in loss; report ties as ties.

#### Exercises

*Check.*

1. **Bytes of a layout.** For gpt-oss-120b compute growing and bounded cache bytes at $n=131{,}072$ and compare with a hypothetical all-global model of the same $H_{kv},d_h,L$. *Sketch:* growing $18\cdot2\cdot8\cdot64\cdot2=36{,}864$ B/token $\times n=4.83$ GB; bounded $18\cdot128\cdot2048=4.72$ MB; all-global would be 73,728 B/token $=9.66$ GB: the layout halves the cache at 128K and the bounded part is negligible.
2. **Is it a gate?** A head computes $o=\bar o\,(1-p_{\text{sink}})$ with $p_{\text{sink}}=e^z/(Z+e^z)$. Is that a gate whose value depends on $x$? Which function of the head's own scores? *Sketch:* yes, $g=\sigma(\mathrm{LSE}-z)$; it depends on $x$ only through the query-key scores, not on $x$ directly (Derivation 2).

*Derive.*

3. **Margin at 1M tokens.** Using Derivation 1 with $V=1,\varepsilon=0.01$, what logit margin is needed at $n=2^{20}$, and what does it imply for the attention-logit growth discussed in 4.3? *Sketch:* $\ln(2^{20}\cdot100)=\ln(1.05\times10^8)=18.5$, a larger logit gap than shorter contexts need ($\ln(4096\cdot100)=12.9$), so sink logits (and the norms of $W_Q,W_K$) must grow with context unless the head is gated; this is one way long-context training stresses QK-norm.
4. **Static pairs.** Show that for $r=64$, $\theta=10^7$ the number of pairs with $\lambda_i>n$ at $n=262{,}144$ is 10, and find $\theta$ such that no pair is static at that context. *Sketch:* $i>32\log_\theta(n/2\pi)$; at $\theta=10^7$, $\log_\theta(41{,}722)=4.62/7=0.66$, so $i\ge22$: pairs 22–31 (10 pairs); no static pair needs $\theta^{62/64}\le n/2\pi$, i.e. $\theta\le41{,}722^{64/62}\approx5.9\times10^{4}$.

*Measure.*

5. **Gate granularity.** Predict whether a head-wise scalar gate ($H$ parameters instead of $Hd_h$ per token) matches the elementwise gate on first-token mass and loss; implement it as `gate='sigmoid_head'` and test with 3 seeds. The gate paper compares such positions and granularities; read its table afterwards and state whether your result agrees.
6. **Learning-rate range.** For B and G, sweep the peak learning rate over a factor 8 range (T1) and find the largest stable LR with 4.3's criterion; the gate paper reports that gating tolerates larger learning rates: does it at this scale? Report with the seed range.

*Open.*

7. **A sink for a window cache.** Train a window-128 variant with and without a learned sink logit and with 0 or 4 retained initial positions at inference; evaluate on streaming perplexity over 8K tokens. State which combination you would ship, and the pre-registered criterion.

#### Further reading

- Papers (v5): [StreamingLLM](https://arxiv.org/abs/2309.17453); [Massive Activations](https://arxiv.org/abs/2402.17762) (abstract: very few activations up to 100,000× larger, largely constant across inputs, acting as bias terms and concentrating attention); [Gated Attention](https://arxiv.org/abs/2505.06708) (30 variants, 15B MoE and 1.7B dense, 3.5T tokens; head-specific sigmoid gate after SDPA improves performance and stability and mitigates the sink, per the abstract); [Rope to Nope and Back Again](https://arxiv.org/abs/2501.18795) (hybrid of global NoPE and local RoPE attention); [Gemma 3](https://arxiv.org/abs/2503.19786) (5:1 local:global); [gpt-oss model card](https://arxiv.org/abs/2508.10925) (learned sinks, alternating window); [SmolLM3 blog](https://huggingface.co/blog/smollm3) (NoPE layers); [Differential Transformer](https://arxiv.org/abs/2410.05258).
- *New (verified 2026-10-28):* [Why do LLMs attend to the first token?](https://arxiv.org/abs/2504.02732) (the over-mixing account of sinks); [Mistral 7B](https://arxiv.org/abs/2310.06825) (sliding window and rolling buffer cache; carried over from Codex 2.5).
- Models: [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) (gated attention in a hybrid); [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (the probe model).
- Code: [`modeling_gpt_oss.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py) (HTTP 200; eager attention with the sink column); [`modeling_qwen3_5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) (output-gate attention); [gpt-oss repository](https://github.com/openai/gpt-oss) (reference kernels, HTTP 200).
- Teaching: Raschka's "From GPT-2 to gpt-oss" (sinks and windows explained with code).

#### Risks and open decisions

| Risk | Fallback |
|---|---|
| R2 recipe differences make variants not directly comparable with 4.2's reported loss | compare only within the notebook (same recipe, same seed policy); report that absolute loss is not R2's |
| Effects at R2 scale are below noise (likely for loss) | the plan reports ties and focuses on the diagnostic metrics; the T0 toy carries the mechanism |
| gpt-oss-20b probe (≈ 42 GB in bf16) does not fit one A100 40 GB, and MXFP4 kernels are unavailable on A100 | run it on a T2 pair, or skip it; the Transformers oracle test on a small random config stands alone |
| Fused-kernel support for sinks on A100/Ada is unconfirmed | LSE-gate identity route (tested), or the explicit softmax in fp32 |
| Copy-task accuracy at R2 scale is near zero for all variants | use the loss on the second copy instead of accuracy; or raise training tokens for the induction phase; state it |
| Half-split vs interleaved RoPE mismatch with 1.2's loader | the permutation test is part of the section; coordinate with 1.2's author |

### 14.5 Learned sparse attention

- **Label:** `sec_lm_sparse_attention` · **Tier:** T1 · **Budget:** ≤ 14 min on the build box: selection-quality study on a frozen Qwen3-0.6B (≈ 5 min, includes fitting a 2-layer indexer for ≈ 2 min), trained-in sparsity on synthetic retrieval (3 arms × 2 seeds × ≈ 1 min), profile of the selection step (≈ 2 min) · **Peak memory:** ≤ 12 GiB (Qwen3-0.6B bf16 plus captured q/k of two layers at 8K tokens; profile at ≤ 64K tokens uses one layer) · **Codex units:** 3.5
- **Builds on (book):**
  - `sec_attention-at-scale`: sliding-window masks, receptive-field growth with depth (`subsec_attn-receptive-field`), the linear-cost window implementation, blockwise online softmax and memory traffic as the bottleneck. **Taken as given:** *fixed* sparsity patterns (windows) and why they cut arithmetic but not retrieval reach. 14.5 starts from what a window cannot do and adds *learned, content-dependent* selection.
  - `sec_kv-cache`: `eq_kv-cache-bytes`; the sink-plus-window result (a sparse pattern that *works* because of a content-independent addition).
  - `sec_mdl-svd-low-rank`, `sec_mdl-divergences-distances`: only for the KL loss of Derivation 2.
- **Builds on (this part):**
  - 14.3: `d2l.MLAAttention` (absorbed mode; DSA runs MLA in MQA form) and the cache-bytes accounting; the FLOP formulas `eq_lm_mla_absorb`.
  - 14.1: `d2l.config_stats` extended with `dsa` layers (`index_n_heads`, `index_head_dim`, `index_topk`) and `csa2` plans; the design table's attention-mix column.
  - 14.4: `d2l.RefinedAttention` (window flag) for the local branch; sinks as the content-independent special case.
  - 1.2: `d2l.load_qwen3` (R1) for the frozen model; 5.2: the decode model for the crossover.
  - 12.4: contrast only: eviction and quantization of the cache of an already trained *dense* model.
- **Exports (`#@save`):**
  - `d2l.block_gate_scores(q, k, block, *, causal=True)` — MoBA gate: mean-pooled key per block, score = dot product with the query; returns `(B, H, T, num_blocks)` with future blocks masked.
  - `d2l.page_bound_scores(q, k, page)` — Quest-style upper bound: per-page channel-wise min and max of keys, score $\sum_c\max(q_cm_c,\,q_cM_c)$.
  - `d2l.block_topk_attention(q, k, v, *, block, topk, window=0, init_blocks=1, group_shared=True, impl='mask'|'gather')` — attention over (initial blocks) ∪ (top-k gated blocks) ∪ (the local window and the query's own block); `impl='mask'` builds a dense boolean mask (the oracle), `impl='gather'` gathers the selected key/value blocks (the one whose cost scales with $k\cdot$block).
  - `d2l.LightningIndexer(d, num_heads_I, dim_I)` — $I_{t,s}=\sum_jw_{t,j}\,\mathrm{ReLU}(q^I_{t,j}\cdot k^I_s)$ with `forward(x_q, k_I)` returning scores; `d2l.token_topk_attention(q, k, v, scores, topk)` — attention restricted to the top-$k$ tokens per query.
  - `d2l.indexer_kl_loss(I, p, selected=None)` — Derivation 2's two losses (full support or restricted to the selected set), with `p` the head-summed, L1-normalized attention of the main model; `d2l.mass_recall(attn, selected)`.
  - `d2l.recall_task(num_pairs, ctx_len, vocab, seed)` — associative-recall sequences (random key tokens, random value tokens, a query key at the end) used for trained-in sparsity; returns tokens, target and the *position* of the needed pair (for selection recall).
  - Data: `lm-arch-sparse-profiles` (JSON: per $n$ and arm: kernel-category times from `torch.profiler`, memory, selection recall).
- **Models and data:** `Qwen/Qwen3-0.6B` (R1; Apache-2.0; 28 layers, 16 heads / 8 KV of width 128; pinned revision); documents: R3 (the book as a corpus, the longest contiguous chapters) and *The Time Machine* as a second source; synthetic retrieval from `recall_task`; random-weight layers for profiling at the dimensions of DeepSeek-V3.2 (`H=128, r=512, dr=64, HI=64, dI=128, topk=2048`, from its config).
- **Packages:** `torch>=2.8` (`torch.topk`, `index_select`, `scaled_dot_product_attention`), `transformers>=5.0`, optional `fla` (for the NSA Triton kernel; CUDA capability not guaranteed on Ada, see Risks).

#### Narrative arc

1. **The question.** At 128K tokens a decode step of a dense MLA layer reads all 128K cached latents. DeepSeek-V3.2 reads 2,048 of them. *How does a model decide which 2,048, how does that cost, and can it be trained so it does not hurt?* The opening cell prints (from 14.1's tables) for the released sparse-attention configs the fraction of the context attended at $n=128$K and at the model's own maximum context: DeepSeek-V3.2 and GLM-5 (token-level top-2,048: 1.56% of 128K, 1.25% of V3.2's `max_position_embeddings` 163,840 and 1.01% of GLM-5's 202,752), DeepSeek-V4-Flash and V4.1-Flash (top-512 over compressed entries plus a 128-token window: the compress ratio changes what a "token" is), MiniMax-M3 (blocks of 128, top-16 blocks, plus a local block: 2,176 positions of $n$, 0.21% of its 1,048,576 maximum; 57 of 60 layers), Qwen3.8-Flash-Next (`indexer_budget 2048`, compress ratio 4). The symptom: sparsity per layer is now 98–99.8% at long context, and every released design pairs it with an *indexer that is itself $O(n)$*.
2. **Why a window is not enough** (`subsec_lm_sparse_window`). A one-paragraph recap of `sec_attention-at-scale` (receptive field $\approx$ layers × $w$) and a 30-second demonstration on `recall_task`: a window-$w$ model answers recall queries only when the needed pair lies inside the window; the accuracy curve against the distance to the pair drops to chance past $w$ (a trained 1-layer toy, T0, seeded). What is missing is *content-dependent reach*.
3. **Four selection rules** (`subsec_lm_sparse_rules`). All choose a subset $S_t$ of positions for query $t$ at budget $k$:
   - *MoBA*: partition the context into blocks of size $B$, score each block by $q\cdot\bar k_{\text{block}}$ (mean-pooled key), top-$k$ gating over blocks; the query's own block is always included and causality is enforced inside it (MoBA paper, Eq. 3–5).
   - *NSA*: three branches with separate K/V and a learned sigmoid gate: compressed (block summaries, $l=32$, stride $d=16$), selected (top-$n$ blocks of size $l'=64$ chosen by the compressed branch's own attention scores: $n=16$ including 1 initial and 2 local blocks) and sliding window ($w=512$); the settings are the paper's §4. Selection is shared by all query heads of a GQA group.
   - *Quest* (inference-time, training-free): per-page channel-wise min and max of keys give an upper bound on $q\cdot k$; select the top pages per head.
   - *DSA*: a lightweight indexer scores *every* preceding token, $I_{t,s}=\sum_{j=1}^{H^I}w^I_{t,j}\mathrm{ReLU}(q^I_{t,j}\cdot k^I_s)$ (DeepSeek-V3.2, Eq. 1; V3.2 has $H^I=64$, $d^I=128$ from its config); the main attention (MLA in MQA mode) then attends to the top-2,048.
   Implement the four *scoring* rules as functions that return a score tensor, and a common `select_topk` that returns indices; the four attention paths differ only in the scoring function and in whether selection is per head or shared by the group.
4. **Hardware alignment: blocks, groups and the union problem** (`subsec_lm_sparse_hw`). Derivation 3: if each head of a GQA group selects its own set, the KV bytes read are the *union*. NSA's critique of Quest (per-head selection) in the paper's §2 is the reason NSA shares the selection across a group; DSA runs in MQA mode (one KV head), so token-level selection is already shared by all 128 query heads. Contiguous blocks coalesce memory reads; scattered tokens need a gather kernel (FlashMLA's sparse decode kernels exist for exactly this; Hopper/Blackwell only).
5. **Training for sparsity** (`subsec_lm_sparse_training`). (a) *Sparse from scratch*: NSA and MoBA train with their selection from the start (NSA: 270B tokens at 8K context then long-context continued training with YaRN, per its §4; the paper reports comparison with full attention trained to convergence). (b) *Retrofitting*: DeepSeek-V3.2 changes only attention from V3.1-Terminus, in two stages (report §2.1.1): a *dense warm-up* in which all parameters except the indexer are frozen and the indexer is trained with a KL loss to the head-summed, L1-normalized main attention (lr $10^{-3}$, 1,000 steps of 16 × 128K tokens = 2.1B tokens), then a *sparse training* stage on top-2,048 selection (lr $7.3\times10^{-6}$, 15,000 steps of 480 × 128K tokens = 943.7B tokens) with the main model trained on the language-modeling loss only and the indexer on the KL loss restricted to the selected set (detached from the main graph). State the point: the indexer has *no gradient path from the language-modeling loss*; its only teacher is the main model's own attention. (c) the V4.x line: sequence compression ($m$ tokens into one entry; V4: CSA at $m=4$ with an indexer, HCA at $m=128$ without), and CSA2's cross-layer reuse of *indices* (IndexCache, YOIO, HISA as the prior work the V4.1 report names; GLM-5.3's `indexer_types` has 21 `full` and 57 `shared` layers).
6. **Cost of the selection step** (`subsec_lm_sparse_cost`). Derivation 1 gives the crossover and the cap; the profile cell (Experiment C) measures where time goes.
7. **Experiments** (Experiments A–C, below).
8. **What the released models report** (`subsec_lm_sparse_reports`). One paragraph each, with quotes limited to short phrases: DeepSeek-V3.2's "only architectural modification" and the parity evaluation (report §2.2); NSA's claim to match or exceed full attention on general, long-context and instruction-reasoning benchmarks with large kernel speed-ups (paper abstract; not tested here); V4.1-Flash's hierarchical sparse indexer (report §2.3.2) and the 890 bytes of 14.1.

#### Derivations

**1. Cost per query token and the sparsity crossover (`eq_lm_sparse_cost`, `eq_lm_sparse_cap`).**

*Statement.* Let dense attention cost $a\,n$ FLOPs per query token per layer ($a$ per cached position), an indexer cost $b$ per scored position, and sparse attention cost $a$ per *selected* position. Then $C_{\text{full}}(n)=an$, $C_{\text{sparse}}(n)=bn+a\min(n,k)$; the sparse path is cheaper iff $n>k\,a/(a-b)$ and its speedup is bounded by $a/b$. For DeepSeek-V3.2 ($a=2H(2r+d_r)=278{,}528$ for MLA absorbed with $H=128$, $b=2H^Id^I=16{,}384$, $k=2048$): cap 17.0; at $n=131{,}072$ speedup $13.4$; the break-even context is $k\,a/(a-b)\approx2{,}176$.

*Assumptions.* FLOP counts only (no bandwidth, no launch overhead, no gather inefficiency); indexer cost $2H^Id^I$ per scored position (one dot product per indexer head, ReLU and weighting negligible); main attention in absorbed-MQA form (14.3).

*Steps.* (1) Full: $a n$. (2) Sparse: the indexer scores all $n$ positions ($bn$), then attention over $\min(n,k)$ positions. (3) $C_{\text{sparse}}<C_{\text{full}}\iff bn+ak<an$ for $n>k$, i.e. $n>ak/(a-b)$. (4) $\lim_{n\to\infty}C_{\text{full}}/C_{\text{sparse}}=a/b$. (5) For MoBA/NSA replace $bn$ by the block-gate cost $2Hd\,n/B$ (mean-pooled keys scored per block) and $ak$ by $a(kB+w+B_{\text{init}})$. (6) For DeepSeek-V4.1-Flash's hierarchy: indexer layers that run in `Full` mode scan their own entries ($n/c$ at compression $c$) and `Reindex` layers scan only the candidate pool of $2048\times8=16{,}384$ positions; the total indexer cost per query token is $\sum_{\text{Full}}2H^Id^I\,n/c_\ell+\sum_{\text{Reindex}}2H^Id^I\min(n/c_\ell,16{,}384)$.

*Numerical check cell.* (a) Evaluate $C_{\text{full}},C_{\text{sparse}}$ by formula and by `config_stats(...).flops_per_token(n)` on DeepSeek-V3.2's and GLM-5's configs for $n\in\{4\text{K},\dots,1\text{M}\}$; assert agreement to $10^{-9}$ relative (same formula, different path: catches off-by-one in plans). (b) Pre-check for V4.1-Flash at $n=1{,}048{,}576$ with $H^I=32$, $d^I=128$ (14.1's fact sheet): indexer FLOPs per query token ≈ $2.2\times10^{10}$ with the hierarchy and ≈ $5.6\times10^{10}$ if the four `Reindex` layers scanned the full decoder context: the hierarchy is a ≈ 2.5× saving of indexer work at 1M (arithmetic from the stated formula; the author recomputes and states the assumption that compression divides the scanned entries). (c) Plot the speedup curve and mark each released config's $(n/k)$ operating points.

**2. Selection quality, the two-stage indexer loss and the error bound (`eq_lm_sparse_err`, `eq_lm_sparse_kl`).**

*Statement.* For one head and query with attention $\alpha\in\Delta^{n-1}$ and selected set $S$ with captured mass $m=\sum_{s\in S}\alpha_s$, the sparse output $o_S=\sum_{s\in S}\frac{\alpha_s}{m}v_s$ satisfies $\|o-o_S\|=(1-m)\,\|\bar o_{S^c}-o_S\|\le 2V(1-m)$ where $\bar o_{S^c}$ is the normalized output on the complement and $V=\max_j\|v_j\|$; the set maximizing $m$ at budget $k$ is the top-$k$ of $\alpha$. The DSA warm-up loss $\sum_tD_{\mathrm{KL}}(p_t\|\mathrm{softmax}(I_t))$ is minimized (to 0) iff $\mathrm{softmax}(I_t)=p_t$, in which case the indexer's top-$k$ equals the oracle top-$k$ of $p_t$ (the head-summed attention) and maximizes the summed-head captured mass; the second-stage loss restricted to $S_t$ trains the indexer to match $p_t$ *on the set it selects* and thus does not penalize omissions directly.

*Steps.* (1) $o=\sum_s\alpha_sv_s=m\,o_S+(1-m)\,\bar o_{S^c}$; (2) subtract $o_S$: $o-o_S=(1-m)(\bar o_{S^c}-o_S)$; (3) triangle inequality with $\|\cdot\|\le V$ gives $\le2V(1-m)$; (4) among sets of size $k$, $m$ is maximized by the $k$ largest $\alpha_s$ (exchange argument); (5) the head-summed target is $p_{t,s}\propto\sum_h\alpha^h_{ts}$, the quantity whose top-$k$ maximizes the *summed* captured mass over heads (per head, the optimum can differ: the price of sharing); (6) the stage-two loss involves only $s\in S_t$ by construction; an element wrongly excluded receives no gradient from this loss, which is why stage one (full support) runs first and why the selection must be re-evaluated as the model changes.

*Numerical check cell.* (a) Random $\alpha$ (Dirichlet, seed 0, $n=512$) and values; assert the bound for 1,000 random sets and that the top-$k$ set maximizes $m$ among them; (b) on captured q/k of two layers of Qwen3-0.6B at 4K tokens, compute the per-head versus head-summed optimum captured mass (the *price of sharing*, a measured quantity); (c) `indexer_kl_loss` equals the explicit KL (`torch.nn.functional.kl_div`) to $10^{-6}$, and its gradient with respect to a fixed `p` is zero.

**3. The union of per-head selections in a GQA group (`eq_lm_sparse_union`).**

*Statement.* If each of $G$ query heads sharing a KV head selects $k$ of $n$ positions independently and uniformly, the expected number of distinct KV positions read is $n\bigl(1-(1-k/n)^G\bigr)$, between $k$ and $Gk$; with group-shared selection it is $k$. With correlated selections it is smaller.

*Steps.* (1) A position is missed by one head w.p. $1-k/n$; (2) by all $G$ heads w.p. $(1-k/n)^G$ (independence); (3) linearity of expectation. *Check:* Monte-Carlo ($n=4096$, $k=256$, $G=4$, 2,000 draws, seeded) to 3 decimals; then replace uniform selections by the per-head top-$k$ of the captured Qwen3-0.6B attention (GQA 16:8, $G=2$) and report the measured union-to-$k$ ratio, expected to be smaller than the uniform formula (to be measured).

#### From-scratch implementation

Budget ≈ 150 lines: the four score functions (≈ 40), `select_topk` and the two attention paths (`mask` ≈ 25, `gather` ≈ 45), `LightningIndexer` and `token_topk_attention` (≈ 30), `mass_recall` and the KL loss (≈ 15), `recall_task` (≈ 15).

| Test | Reference | Tolerance / seed |
|---|---|---|
| `impl='gather'` equals `impl='mask'` | the mask oracle | fp32 $<10^{-5}$; seed 0; block 32, topk 4, window 64 |
| With `topk` $\ge$ all blocks, output equals full causal attention | `F.scaled_dot_product_attention(is_causal=True)` | fp32 $<10^{-5}$ |
| Block gate with a single block equals dense attention within the block | same | exact up to fp |
| `token_topk_attention` with $k\ge n$ | full attention | fp32 $<10^{-6}$ |
| Causality: no position selects a future block | assertion over all $(t,\text{selected})$ | exact |
| `LightningIndexer` vs a literal einsum of Eq. 1 of the V3.2 report | explicit loops | fp32 $<10^{-6}$ |
| `mass_recall` of the oracle top-$k$ $\ge$ any other set of the same size | 1,000 random sets | exact inequality |
| Error bound (Derivation 2) | direct | holds for all 1,000 sets |
| FLOP formula of Derivation 1 vs `config_stats(...).flops_per_token` | each other | $10^{-9}$ relative |

#### In production

- **DeepSeek:** the V3.2-Exp repository (`deepseek-ai/DeepSeek-V3.2-Exp`, HTTP 200) contains the reference model code with the indexer; FlashMLA's sparse kernels power DSA and the V4.1 kernels; the README (fetched 2026-10-28) states that its 2026-09-30 release dropped Hopper support and targets Blackwell and Ascend, with an older commit retained for Hopper; **nothing there runs on A100 or Ada**, which is why the notebook's `gather` path in PyTorch is the artifact.
- **NSA / MoBA:** `fla-org/native-sparse-attention` (Triton kernels for NSA; HTTP 200) and `MoonshotAI/MoBA` (built on FlashAttention varlen; HTTP 200); the notebook uses them as *references for design*, not as test oracles on the build box (Risks).
- **Serving:** vLLM and SGLang ship DSA support for V3.2-class models (13.x pointers); index reuse across layers (GLM-5.3 `indexer_types`, V4.1 reuse modes) is a scheduler-visible change in what each layer computes and caches.

#### Headline experiment

- **Experiment A (selection quality on a frozen real model; the learned-indexer content).**
  - *Hypothesis (direction):* at equal budget (tokens attended per query), a trained lightning indexer captures more of the head-summed attention mass of a dense model than a mean-pooled block gate (MoBA) or a Quest page bound, and the gap widens as the budget shrinks; the oracle top-$k$ is the ceiling. Magnitudes **to be measured**.
  - *Setup:* Qwen3-0.6B (R1, bf16, frozen), 2 layers (one early, one late: the layer indices are chosen in the notebook by the largest and smallest head-summed attention entropy on a 4K-token sample, to avoid cherry-picking), documents of 4,096 tokens from R3 (train: 200 documents; held-out: 50 documents; no overlap of chapters). Capture $q,k$ and the attention weights with hooks (eager attention for the two layers only). Indexer: `LightningIndexer(H^I=4, d^I=64)` fitted with the stage-one KL loss to the head-summed attention (AdamW, lr $10^{-3}$, 1,000 steps; ≈ 2 min), then *sparse-stage* fitting restricted to the selected set for 300 steps (to show the distinction of Derivation 2). Budgets $k\in\{64,128,256,512\}$ tokens per query ($B=32$ blocks for MoBA and Quest pages of 16).
  - *Metrics:* captured head-summed mass (`mass_recall`) on held-out documents with a bootstrap interval over documents; per-head versus summed optimum; then the *inference-time* effect: validation loss of Qwen3-0.6B with attention restricted to the selected set in those two layers only (no retraining), versus full.
  - *Plot (`fig_lm_sparse_recall`):* x: budget $k$ (log), y: captured mass; lines: oracle, indexer (stage 1), indexer (stage 2), MoBA gate, Quest bound, random; bands: bootstrap 95% intervals; second panel: loss increase over full versus $k$.
  - *Noise check:* 50 held-out documents × queries: bootstrap over documents; with two fitted indexers per setting (2 seeds) the seed range is shown; a gap smaller than the interval is a tie.
- **Experiment B (trained-in sparsity on retrieval; v5's notebook).**
  - *Hypothesis:* a window-plus-top-$k$-block model matches full attention on associative recall when the budget exceeds the number of needed blocks, and *fails* when the window alone is used; the sparse arm's accuracy depends on selection recall (the fraction of queries whose needed block is selected).
  - *Setup:* `d2l.GPT`-compatible models, width 256, 6 blocks, context 2K, `recall_task` (num_pairs 64, vocab 512) mixed 1:1 with FineWeb-Edu LM batches; arms: **F** full attention; **W** window 128 only; **M** window 128 + top-4 blocks of 64 (MoBA-style gate, trained from scratch); each 1 min on the build box, 2 seeds; evaluate recall accuracy at contexts 2K (train), 4K and 8K (the number of blocks grows, the budget is fixed) and LM loss.
  - *Metrics:* recall accuracy, selection recall, LM loss, tokens/s.
  - *Plot (`fig_lm_sparse_recall_acc`):* accuracy versus context length for F, W, M with seed ranges; selection recall as dashed lines.
  - *Expected:* W near chance beyond its window; M between W and F, with the amount depending on whether the gate learns to find the pair (to be measured; at 1-minute training this may not converge: Risks).
- **Experiment C (selection overhead).** Single layer at DeepSeek-V3.2's dimensions (random weights, bf16, MQA form with $r+d_r=576$): arms dense-absorbed, `token_topk` (indexer $H^I=64$, $d^I=128$, top-2,048), `block_topk` (block 128, top-16); $n\in\{4\text{K},8\text{K},\dots,64\text{K}\}$, batch 1 and 8; per-step times for *score*, *topk*, *gather*, *attention* from `torch.profiler` kernel categories, 5 repetitions. Fit time $=\tau_0+\tau_1n$ per category; compare $\tau_1^{\text{score}}/\tau_1^{\text{dense}}$ with $b/a$ (Derivation 1). *Expected (direction):* the score term grows linearly in $n$ and dominates the sparse path's time at long context; the crossover context in *time* is larger than the FLOP crossover of 2,176 because of top-$k$ and gather overheads (to be measured). **Artifacts committed:** `lm-arch-sparse-profiles`.
- **T1 reduction:** this section is already T1.

#### Figures and tables

| Label | Content | Cell |
|---|---|---|
| `tab_lm_sparse_released` | released sparse-attention configs: indexer heads/dim, top-$k$, block, compress ratio, fraction attended at 128K and 1M | cell 1 |
| `fig_lm_sparse_rules` | schematic of the four selection rules (matplotlib patches: query row, blocks, selected cells) | `draw_rules()` |
| `fig_lm_sparse_speedup` | speedup $C_{\text{full}}/C_{\text{sparse}}$ versus $n$ with operating points | `plot_speedup()` |
| `fig_lm_sparse_recall` | Experiment A | `plot_recall()` |
| `fig_lm_sparse_recall_acc` | Experiment B | `plot_acc()` |
| `fig_lm_sparse_profile` | Experiment C stacked time by category versus $n$ | `plot_profile()` |
| `tab_lm_sparse_union` | union of selected KV positions versus group size (formula, Monte-Carlo, measured) | cell 7 |

#### Pitfalls and misconceptions

- Top-$k$ selection is *not* $O(k)$: the indexer scores all $n$ positions; only the main attention is $O(k)$.
- FLOP speedups are not wall-clock speedups: top-$k$, gather and launch overheads add a context-independent term and the gather is bandwidth-bound; kernels for it exist only on specific hardware.
- Per-head selection in a GQA/MQA model reads the union of the selections; group-shared selection is what makes the traffic $k$, at the price of a worse per-head optimum.
- A sparse model must be *trained* (or retrofitted) for sparsity; applying top-$k$ at inference to a dense model (Quest, Experiment A's loss panel) is a different, usually weaker, regime.
- The indexer receives no gradient from the language-modeling loss in DSA: its quality is bounded by how well the main model's own attention is a good teacher.
- "Compression ratio" changes the unit of selection: V4's top-512 over entries at $m=128$ covers $512\times128$ tokens, not 512.
- Mean-pooled keys can hide a single sharp key inside a block (the weakness the Quest bound addresses); recall tasks with one needle per block expose it.
- Training/inference mismatch: block boundaries, ties in top-$k$, and numerical noise in scores must be identical in both paths (indices computed in the same precision; DSA scores in FP8 at inference per the V3.2 report is a *different* arithmetic; state that the notebook uses bf16 for both).

#### Summary (draft bullets)

- Learned sparse attention keeps the cache but reads a small, content-chosen subset: 2,048 of 128K positions in DeepSeek-V3.2.
- Four selection rules (mean-pooled block gate, compressed-branch gate, page bound, token-level indexer) differ in what they score and who shares the selection.
- The cost is $bn+ak$: an indexer $O(n)$ term with a small constant plus a $k$-sized attention; the speedup is capped at $a/b$ (17× for V3.2 by FLOPs).
- The error of restricting to a set is at most $2V$ times the missed attention mass, and the top-$k$ of the attention is the best set; an indexer is trained to predict that distribution.
- DSA's two stages (indexer warm-up, then sparse training with a detached indexer loss) make retrofitting feasible; NSA and MoBA train sparse from scratch.
- Index reuse and hierarchical candidate pools (GLM-5.3, V4.1-Flash) attack the remaining $O(n)$ indexer term.

#### Exercises

*Check.*

1. **What fraction is attended?** For MiniMax-M3 (`sparse_block_size 128`, `sparse_topk_blocks 16`, local block 1) compute the fraction of a 1M-token context attended by a query, ignoring initial blocks. *Sketch:* $(16+1)\times128/1{,}048{,}576=0.21\%$ ($2{,}176$ positions).
2. **Which layers choose?** In V4.1-Flash `index_source_layer_ids=[2,8,14,20,24,28,32,36]` and `kv_source_layer_ids=[2,8,14,20]`: which layers run an indexer, which own main KV, and which only reuse? *Sketch:* indexers on all eight; main KV owned by four; layers 24, 28, 32, 36 reindex over a shared candidate pool but reuse main KV; the other layers reuse both (the report's Reuse mode).

*Derive.*

3. **The cap.** Show that $C_{\text{full}}/C_{\text{sparse}}\to a/b$ and compute it for GLM-5 ($H^I=32$, $d^I=128$, $H=64$, $r=512$, $d_r=64$, MLA absorbed). *Sketch:* $a=2\cdot64\cdot1088=139{,}264$, $b=2\cdot32\cdot128=8{,}192$; cap $17.0$ (the same as V3.2: both halve $H$ and $H^I$).
4. **Union under correlation.** For $G=4$ heads with pairwise selection overlap fraction $\rho$ (each pair shares a fraction $\rho$ of its $k$ positions, independent otherwise), derive the expected union size to first order in $\rho$. *Sketch:* inclusion–exclusion truncated at second order: $|\cup|\approx Gk-\binom G2\rho k$ for small overlaps.

*Measure.*

5. **Budget sweep.** In Experiment A, predict the budget at which the indexer's captured mass first exceeds 90% for the late layer, then measure it. Report the budget as a fraction of the 4K context.
6. **Block size.** Repeat Experiment B's arm M with block 16, 64, 256 at a fixed token budget (top-$k\times$block = 256); predict which block size gives the best selection recall and measure with 2 seeds.

*Open.*

7. **Index reuse.** Implement "reindex every second layer, reuse indices otherwise" in a 6-block model and measure the loss and selection-recall cost against indexing every layer (T1; pre-register the criterion). Relate your result to GLM-5.3's 21/57 split.

#### Further reading

- Papers (v5): [Native Sparse Attention](https://arxiv.org/abs/2502.11089) (the three-branch design, $l=32,d=16,l'=64,n=16,w=512$; the GQA group-sharing argument against per-head selection); [MoBA](https://arxiv.org/abs/2502.13189) (top-$k$ gating over blocks from MoE); [Quest](https://arxiv.org/abs/2406.10774) (query-aware pages, training-free); [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) (§2.1 DSA; the two training stages and their token counts quoted above).
- *New (verified 2026-10-28):* [IndexCache](https://arxiv.org/abs/2603.12201) (cross-layer index reuse); [HISA](https://arxiv.org/abs/2603.28458) (hierarchical indexing; the V4.1 report's §2.3.2 prior work); [HySparse](https://arxiv.org/abs/2602.03560) (sparse layers reusing the KV of dense layers); [You Only Index Once](https://arxiv.org/abs/2606.06467) (shared routing across layers); [DeepSeek-V4](https://arxiv.org/abs/2606.19348) (CSA and HCA).
- Teaching: the NSA paper's kernel section for the group-centric loading argument; [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) for the bandwidth arithmetic.
- Code: [`fla-org/native-sparse-attention`](https://github.com/fla-org/native-sparse-attention) and [`MoonshotAI/MoBA`](https://github.com/MoonshotAI/MoBA) (HTTP 200 on 2026-10-28); [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp); [FlashMLA](https://github.com/deepseek-ai/FlashMLA).

#### Risks and open decisions

| Risk | Fallback |
|---|---|
| Experiment B's 1-minute training does not teach the gate to find the pair | lengthen to ≈ 3 min for one seed pair and drop arm W's second seed; or report "gate did not converge at this budget" with the selection-recall curve (itself a result) |
| Qwen3-0.6B eager attention capture of two layers at 4K tokens is slow or memory-heavy | 2K tokens, one layer; budgets scaled by the same factor |
| NSA's Triton kernels do not run on the build box's Ada GPU (compute capability, Triton version) | PyTorch gather implementation only; the kernels are design references |
| Sparse kernels on A100 unavailable (FlashMLA) | stated; Experiment C measures the PyTorch gather path and says so on the plot |
| The indexer FLOP convention ($2H^Id^I$ per scored entry) differs from the production kernel's actual work (FP8 indexer) | state the convention; report both FLOPs and measured time |
| The compression-divides-scanned-entries assumption for V4.1's indexers is an inference | marked "assumed"; the report §2.3 is the check |

### 14.6 Hybrid models at scale

- **Label:** `sec_lm_hybrids_at_scale` · **Tier:** T3 (T1 reduction below; the weight-loading half is T1 by itself) · **Budget:** T1 ≤ 15 min (logit match in fp32 on Qwen3.5-0.8B ≈ 3 min, bf16 on Qwen3.5-2B ≈ 4 min, state-decode test ≈ 1 min, kernel micro-benchmark ≈ 3 min, T1 serving reduction Qwen3.5-2B vs Qwen3-1.7B ≈ 4 min); T3 ≤ 2 h on 8× A100 40 GB (≈ 16 GPU-hours; planned, to be calibrated; the plan fits 8× 40 GB and runs once) · **Peak memory:** T1 ≤ 12 GiB (2B bf16 for both our model and the Transformers oracle ≈ 8.5 GB); T3 ≈ 72 GB of bf16 weights for the hybrid (9.0 GB per GPU at TP = 8) and 61 GB for the attention-only arm (7.6 GB per GPU), ≤ 36 GiB per GPU in total (0.9 × 40 GB) · **Codex units:** 16.4 (Kimi K3 as a hybrid), 3.7 (the hybrid comparison); 3.6 and 3.7's from-scratch work is `sec_hybrids`/`sec_deltanet`
- **Builds on (book):**
  - `sec_hybrids`: attention fraction and placement rules, `tab_hy-recipe` (nine shipped hybrids), `fig_hy-cache`, `eq_ms-state-bytes`, distillation from transformers. **Taken as given:** *why* 1 attention layer in 4–8 works and the cache-per-attention-layer rule. 14.6 does not re-argue the ratio; it maps the recipe table to *configs*, and loads one.
  - `sec_deltanet`: `eq_dn-gated`, the WY form (`delta_chunked`, un-gated, local), `GatedDeltaNetBlock` (a toy: LayerNorm, GELU MLP, time loop, no convolution). **14.6 does not re-derive the delta rule or the ungated WY representation**; it extends the chunked form with gating and replaces the toy block by the released layout.
  - `sec_matrix-state`: `segsum`, `matrix_state_chunked` (scalar decay), state-bytes formula.
  - `sec_mamba`: recurrent `step`. `sec_kv-cache`: `GQAAttention`, `d2l.GPT` weight-loading idiom (GPT-2 in `sec_kv-cache`).
- **Builds on (this part):**
  - 1.1: Transformers logits of Qwen3.5-2B (the oracle), `d2l.load_hf`, `d2l.config_memory`. 1.2: `d2l.Qwen3Attention`, `d2l.SwiGLU` and `d2l.Qwen3GPT` (`head_dim`, `qk_norm`, `rope_theta`, `norm_eps`, `tie_embeddings`), the weight-name and RoPE-layout vocabulary and the layer-by-layer matching protocol; `d2l.SwiGLU(d, hidden)` takes the explicit width (Qwen3.5-2B has `intermediate_size 6144` = 3d, not the book's 8d/3).
  - 14.4: `d2l.RefinedAttention(gate='sigmoid', rope_frac=0.25, qk_norm=True)` is Qwen3.5's full-attention layer. 14.1: `config_stats` for the memory prediction; 5.2: `d2l.ServingModel` with `kv_bytes_override` fed by `ConfigStats`. 13.x: pointers only (state management in serving).
- **Exports (`#@save`):**
  - `d2l.GatedDeltaNetLayer(d, num_k_heads, num_v_heads, dk, dv, kappa=4, eps=1e-6)` — the released Qwen3.5 layer: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, depthwise causal conv with SiLU over the concatenated q,k,v channels, L2-normalized q and k, query scale $d_k^{-1/2}$, $g=-e^{A}\,\mathrm{softplus}(a+\mathrm{dt})$, $\beta=\sigma(b)$, head replication when $H_v>H_k$, gated RMSNorm output, `out_proj`; `forward(x, state=None, mode='chunked'|'recurrent')`.
  - `d2l.gated_delta_recurrent(q, k, v, g, beta, S0=None)` and `d2l.gated_delta_chunked(q, k, v, g, beta, chunk=64, S0=None)` — returns `(o, S_final)`; fp32 state.
  - `d2l.HybridCache` — per layer either KV or `(conv_state, S)`; `bytes()`.
  - `d2l.hybrid_gpt(config)` — builds a `d2l.GPT`-compatible model from a Qwen3.5-style text config by replacing blocks' `attention`, `ffn` and norms (see the mapping table below); `d2l.load_qwen35(model_id='Qwen/Qwen3.5-2B', revision=None)` — builds it and fills it from safetensors, ignoring `model.visual.*` and `mtp.*`.
  - (No FFN class is defined here: 14.6 uses 1.2's `d2l.SwiGLU(num_hiddens, hidden)`, with the book's `W_g, W_1, W_2`.) `d2l.measure_serving(engine, model_id, contexts, concurrencies, reps)` — the T3 harness.
  - Data: `lm-arch-hybrid-serving` (JSONL: model, context, concurrency, rep, engine, cache capacity, peak memory per GPU, TTFT, decode tokens/s).
- **Models and data:** `Qwen/Qwen3.5-2B` (Apache-2.0, 2,274,069,824 params of which LM 1,881,825,088; one safetensors file; T1 oracle) and `Qwen/Qwen3.5-0.8B` (fp32 check); `Qwen/Qwen3-1.7B` (T1 attention-only baseline; 114,688 B/token KV); T3: `Qwen/Qwen3.5-35B-A3B` (Apache-2.0, 35,951,822,704 params, 40 layers: 10 full-attention + 30 Gated DeltaNet; 20,480 B/token + 64,389,120 B fixed state, pre-check) versus `Qwen/Qwen3-30B-A3B-Instruct-2507` (Apache-2.0, same 30,532,122,624 parameters as Qwen3-30B-A3B, 262,144 native context, 98,304 B/token): both 262K native, so no YaRN confound. Prompts: random token ids (latency does not depend on content) and 20 R3 passages for sanity.
- **Packages:** `transformers>=5.0`, `flash-linear-attention` (`fla`, optional fast path; Triton runs on A100 and Ada, confirm in the notebook), `causal-conv1d` (optional), an engine for T3: vLLM or SGLang, **whichever supports `qwen3_5` hybrid cache management at the pinned version (unverified; Risks)**.

#### Narrative arc

1. **The question.** The recipe table of `sec_hybrids` was a table of *papers*. *What does a shipped hybrid's configuration look like, and does it load into the code we built?* The opening cell prints (from 14.1) the layer-kind counts and cache/state bytes of Qwen3-Next, Qwen3.5-2B/35B-A3B, Qwen3.6-27B, Qwen3.8-2.4T-A95B (3:1 linear:full everywhere via `full_attention_interval 4`), Kimi-Linear (20 KDA + 7 MLA, 3:1), Kimi K3 (69 KDA + 24 MLA), MiniMax-M1 (70 lightning + 10 softmax), GLM-5.3-Flash (34 KDA + 11 DSA layers + mHC), Nemotron-H-8B (Mamba-2 and attention, pattern string in `hybrid_override_pattern`), Nemotron-3-Nano-30B-A3B (Mamba, MoE and attention, pattern string `MEMEM*EMEMEM*…`; both `model_type nemotron_h`, 200 on the Hub), plus MiniMax-M2 (all softmax). Symptom: **seven families converged on ≈ 25% full attention for gated-delta-rule hybrids and 8–12% for Mamba hybrids, but the 2025–26 MiniMax line moved away from lightning attention** (M2: 62/62 softmax layers; M3: block-sparse), so "hybrid" is not monotone.
2. **From recipe to config** (`subsec_lm_hy_configs`): table `tab_lm_hy_released` with attention fraction, placement (`layer_types`, `full_attn_layers`, patterns), recurrent kind, attention kind (GQA gated, MLA nope, DSA), state bytes per sequence and KV bytes per token (from `config_stats`), state/KV break-even context (Derivation 2). Layer-placement plot: strips per model with full-attention layers highlighted. Does each model obey the book's "none first, evenly spread" rule? (Qwen: layers 4, 8, … one-indexed; K3: every 4th plus the last two; Nemotron patterns start with `M`.) Report violations (K3's layers 92 and 93 adjacent) as findings.
3. **What is inside the released layer** (`subsec_lm_hy_layer`): the mapping table `tab_lm_hy_mapping` (below) from the book's toy `GatedDeltaNetBlock` to the released layer: each difference with its effect: depthwise short convolution (κ=4) on q,k,v; Mamba-style decay $g=-e^{A}\mathrm{softplus}(a+\mathrm{dt})$ per value head; separate $\beta$ projection; $L_2$-normalized $q,k$ plus $d_k^{-1/2}$; output gate $z$ with RMSNorm-then-SiLU ordering; $H_k\ne H_v$ replication; fp32 state; no MLP inside the block (the MLP is the ordinary FFN of the decoder layer).
4. **Chunked gated form** (`subsec_lm_hy_chunked`): Derivation 1; implement, test against the recurrence and against Transformers' `torch_chunk_gated_delta_rule` and `fla`'s Triton kernel; micro-benchmark vs sequence length.
5. **Loading Qwen3.5-2B into the book's code** (`subsec_lm_hy_load`): the mismatch checklist (below) as a table; layer-by-layer matching with first-divergence localization; the fp32 check on 0.8B and the bf16 check on 2B with a *noise-floor tolerance*; stateful decode equals full forward.
6. **Recurrent state versus KV cache at long context** (`subsec_lm_hy_state`): Derivation 2; serving consequences: the state is per sequence, not per token, so *prefix caching needs state snapshots* (a state at token $t$ cannot be truncated: the KDA-aware prefix cache of the K3 report §5.4.1 is the production answer, cited not rebuilt), preemption and swapping move a fixed block, and batch-size limits come from the state when contexts are short. Feed `ConfigStats` into `d2l.ServingModel(kv_bytes_override=…)` to predict maximum concurrency.
7. **Hybrid versus attention-only on a node** (`subsec_lm_hy_serving`, T3): the headline experiment.
8. **What the 2026 line says** (`subsec_lm_hy_lessons`): sizes of the linear-layer state against the KV of the few attention layers at each model's context; the new variants (KDA's per-channel gate with `gate_lower_bound -5.0` and `use_full_rank_gate`; Qwen3.8-Flash-Next's addition of an indexer and mHC streams: the hybrid is becoming a platform that carries several other axes of this chapter at once).

#### Mismatch checklist: the book's toy → Qwen3.5 (`tab_lm_hy_mapping`)

| Item | Book / `d2l` default | Qwen3.5-2B | Handling |
|---|---|---|---|
| RMSNorm | $x\,w$, init 1, `eps` default | $x(1+w)$, init 0; `eps=1e-6` | fold $w\leftarrow1+w_{\text{HF}}$ in fp32 at load; set `eps=1e-6` (1.2's `norm_eps`) |
| Gated RMSNorm in linear layer | none | plain weight (not $1+w$), stored fp32, applied *before* the SiLU gate | load as is; implement norm-then-gate |
| FFN width | $8d/3$ | `intermediate_size 6144` | `d2l.SwiGLU(d, 6144)` (1.2; `W_g, W_1, W_2`) |
| Full-attention layer | `GQAAttention` | `q_proj [4096,2048]` holds query and gate per head (chunked along the last dim of `view(.., H, 2·256)`), `q_norm`/`k_norm` $(1+w)$ on the head dim, sigmoid gate on the output | `RefinedAttention(gate='sigmoid')` with the same split |
| RoPE | adjacent pairs, full head | half-split, first 64 of 256 channels, $\theta=10^7$, text-only positions identical in the three M-RoPE streams | 14.4's half-split partial rotary |
| Linear layer weights | `W_qkv`, `W_ab`, `W_gate` | `in_proj_qkv [6144,2048]`, `in_proj_z [2048,2048]`, `in_proj_b [16,2048]`, `in_proj_a [16,2048]`, `conv1d.weight [6144,1,4]`, `A_log [16]` (fp32), `dt_bias [16]`, `norm.weight [128]` (fp32), `out_proj [2048,2048]` | name map; channel order in `in_proj_qkv` and the conv is $[q\,|\,k\,|\,v]$ |
| Embeddings | `token_emb` | `model.language_model.embed_tokens [248320,2048]`, tied | `tie_embeddings` (1.2) |
| Extra tensors | — | `model.visual.*` (331,416,576), `mtp.*` (60,828,160) | ignored; counts logged |
| Dtypes | one dtype | bf16 weights; `A_log`, linear `norm` fp32; state fp32 (`mamba_ssm_dtype`) | keep fp32 where stored |

#### Derivations

**1. The gated chunked delta rule equals the recurrence (`eq_lm_hy_chunked`).**

*Statement.* With $a_t=e^{g_t}\in(0,1]$ the recurrence $S_t=a_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$, $o_t=S_t^\top q_t$ (the book's `eq_dn-gated` with a per-head scalar $\alpha_t=a_t$) is computed exactly, chunk by chunk, by one unit-lower-triangular solve per chunk.

*Assumptions.* Chunk length $C$; $\gamma_t=\prod_{i\le t}a_i$ within the chunk (cumulative from the chunk start); inputs already $L_2$-normalized and $q$ scaled; $\gamma_i/\gamma_j=e^{G_i-G_j}$ computed in log space (never divide).

*Steps.* (1) Unrolling the chunk from entering state $S_0$: $S_i=\gamma_iS_0+\sum_{j\le i}\frac{\gamma_i}{\gamma_j}k_ju_j^\top$ with update vectors $u_j=\beta_j(v_j-S_j'^\top k_j)$, $S'_j=a_jS_{j-1}$. (2) Substituting $S'_i=\gamma_iS_0+\sum_{j<i}\frac{\gamma_i}{\gamma_j}k_ju_j^\top$: $u_i=\beta_i\bigl(v_i-\gamma_iS_0^\top k_i-\sum_{j<i}\frac{\gamma_i}{\gamma_j}(k_j^\top k_i)u_j\bigr)$. (3) Matrix form $(I+A)U=\mathrm{diag}(\beta)(V-\mathrm{diag}(\gamma)KS_0)$ with $A_{ij}=\beta_i\frac{\gamma_i}{\gamma_j}k_i^\top k_j$ for $j<i$, else 0: the book's `eq_dn-wy-system` with a decay factor; solve by forward substitution against the right-hand side. (4) Outputs $O=\mathrm{diag}(\gamma)QS_0+\bigl(\mathrm{tril}(QK^\top)\odot\Gamma\bigr)U$, $\Gamma_{ij}=\gamma_i/\gamma_j$ for $j\le i$. (5) State carry: $S_C=\gamma_CS_0+\sum_j\frac{\gamma_C}{\gamma_j}k_ju_j^\top$. (6) Cost per head per token: building $A$ and $QK^\top$ is $O(Cd_k)$, the solve is $O(C(d_k+d_v))$, state read/write is $O(d_kd_v)$ amortized over $C$.

*Numerical check cell.* Random inputs (seed 0, $H=4$, $d_k=d_v=64$, $T=512$): (a) chunked versus recurrent in fp64 $<10^{-10}$ relative for $C\in\{16,64,256\}$; (b) versus `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule` from Transformers (call with `use_qk_l2norm_in_kernel=True`) in fp32 $<10^{-5}$; (c) versus `fla.ops.gated_delta_rule.chunk_gated_delta_rule` when `fla` imports (fp32/bf16 tolerance to be measured; mark skipped otherwise); (d) with $g\equiv0$ it reduces to the book's `delta_chunked` (after the $d_k^{-1/2}$ scaling) $<10^{-6}$.

**2. State versus KV: bytes per sequence and the break-even context (`eq_lm_hy_state`).**

*Statement.* For a hybrid with $L_f$ full-attention layers (GQA, $2H_{kv}d_hb$ per token per layer) and $L_\ell$ linear layers (state $H_vd_kd_vb_s+(2K+V)(\kappa-1)b$ per layer), the per-sequence bytes are $M(n)=L_f\,c_{kv}\,n+L_\ell\,c_{st}$; the context at which the linear state equals the KV of the full layers is $n^\*=L_\ell c_{st}/(L_fc_{kv})$, and the memory ratio to an attention-only model of $L$ layers and the same $c_{kv}$ is $M_{\text{hy}}(n)/M_{\text{att}}(n)=\frac{L_f}{L}+\frac{L_\ell c_{st}}{L\,c_{kv}\,n}$.

*Steps and pre-check numbers* (bf16 KV, fp32 state, from the configs, blueprint author): Qwen3.5-2B: $L_f=6,c_{kv}=2048$ B; $L_\ell=18$, $c_{st}=1{,}085{,}440$ B ⇒ $M=12{,}288\,n+19{,}537{,}920$, $n^\*\approx1{,}590$ tokens; ratio to Qwen3-1.7B's 114,688 B/token → 9.3× smaller at long context. Qwen3.5-35B-A3B: 20,480 B/token + 64,389,120 B ⇒ $n^\*\approx3{,}144$; versus Qwen3-30B-A3B 98,304 B/token: at $n=131{,}072$: 12.88 GB vs 2.75 GB (≈ 4.7× less). Kimi K3: 27,648 B/token + 449,372,160 B ⇒ $n^\*\approx16{,}253$. The notebook recomputes all with `config_stats` and asserts equality to the byte, then prints the break-even context for every row of `tab_lm_hy_released`: *below $n^\*$ the state, not the KV, is what you pay for*, which is why short-context batch size gains are small.

*Check.* `HybridCache.bytes()` after a prefill of $n$ tokens equals $M(n)$ exactly on the loaded 2B model; `config_memory(...)['kv_cache'] + config_memory(...)['recurrent_state']` agrees within 0.1% (disagreement is reported to 1.1).

#### From-scratch implementation

Budget ≈ 150 lines: `gated_delta_recurrent` (≈ 18), `gated_delta_chunked` (≈ 40), `GatedDeltaNetLayer` (≈ 50: projections, conv with state, gates, norm), `HybridCache` and `hybrid_gpt`/`load_qwen35` (≈ 40).

| Test | Reference | Tolerance / seed |
|---|---|---|
| Chunked vs recurrent (fp64) | each other | $10^{-10}$ rel.; $C\in\{16,64,256\}$ |
| Both vs Transformers torch implementations; vs `fla` if present | Transformers `modeling_qwen3_5.py` functions | fp32 $10^{-5}$ |
| `GatedDeltaNetLayer` vs `Qwen3_5GatedDeltaNet` with copied random weights | Transformers module, small config | fp32 $10^{-5}$ |
| Stateful decode vs full forward (conv state + $S$) | full forward | fp32 $10^{-5}$ |
| fp32 logits, Qwen3.5-0.8B, 3 prompts × 512 tokens | Transformers fp32 | max abs logit diff to be measured; report and fix a tolerance at 10× the observed value only if the diff is $<10^{-3}$; else localize the first diverging layer |
| bf16 logits, Qwen3.5-2B | Transformers bf16 | **noise-floor rule:** our max abs diff $\le2\times$ the max abs diff between HF-bf16 and HF-fp32 on the same prompts; plus top-1 agreement over all positions reported |
| Layer-by-layer hidden states | `output_hidden_states=True` | first layer whose relative error exceeds $3\times$ the previous layer's is flagged |
| Parameter counts | 1,881,825,088 (LM), `config_stats` and `d2l.config_params` | exact |
| Cache bytes of `HybridCache` | $M(n)$ | exact |

#### In production

Transformers' `Qwen3_5GatedDeltaNet` uses `fla`'s kernels (`chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`) and `causal-conv1d` when installed, with the PyTorch functions quoted above as fallback; this notebook's code is the *test oracle* for both. vLLM/SGLang manage hybrid caches with fixed-size state slots next to paged KV (13.x). K3's report §5.1 describes training/prefill kernels for KDA and an intra-device context-parallel scheme (cited).

#### Headline experiment

- **Hypothesis (derived, then to be measured).** The hybrid's per-sequence cache bytes follow $M(n)$ of Derivation 2 to the byte, so at 128K its per-sequence bytes are ≈ 4.7× smaller than those of the attention-only model of the same active size, and its maximum concurrency is higher by a smaller factor on this node (≈ 2.5× in the pilot arithmetic above, because TP = 8 replicates the hybrid's 2 KV heads 4× and the peer's 4 KV heads 2×; before allocator overhead, to be measured); decode throughput per sequence at long context is higher for the hybrid because less is read per step, but at short context (below $n^\*$) the advantage disappears and prefill of the linear layers may cost more than attention at the same length (to be measured).
- **Setup (T3).** 8× A100 40 GB, one node. Arms: `Qwen3.5-35B-A3B` (hybrid) and `Qwen3-30B-A3B-Instruct-2507` (attention-only), both bf16, tensor parallel over the node (TP = 8; TP = 4 as a second configuration if the engine's expert-parallel path fails), same engine and version, `max_model_len = 262,144`. Contexts $n\in\{4\text{K},32\text{K},128\text{K},256\text{K}\}$; concurrency $c\in\{1,8,32\}$ (a cell above the engine's reported capacity runs at that maximum instead, and the duplicate is not repeated); 64 generated tokens (`ignore_eos`); 3 repetitions per cell, first discarded as warm-up. Prefix caching **off**. Capacity at $M=36\times10^9$ B per GPU (pilot arithmetic before engine reservations; at TP = 8 each of the hybrid's 2 KV heads is stored on 4 GPUs and each of the attention-only model's 4 KV heads on 2): about 20 sequences of 128K for the hybrid against 8 for the attention-only arm, and 10 against 4 at 256K, so $c=32$ is never reached at 128K; the TP = 4 fallback leaves 18.0 and 20.7 GB per GPU for cache.
- **Procedure.** Record (i) engine-reported KV/state capacity and maximum concurrency at $n$; (ii) peak memory per GPU (`torch.cuda.max_memory_allocated` via the engine's worker or `nvidia-smi` sampling at 1 Hz, whichever the engine permits; document); (iii) TTFT of one prompt of $n$ tokens; (iv) decode tokens/s per sequence and in total; (v) the same cells *predicted* by 5.2's `ServingModel` fed with `config_stats` bytes.
- **Metrics:** measured/predicted cache bytes per sequence; max concurrency at 128K; TTFT versus $n$; decode tokens/s per user versus $n$ at $c=8$.
- **Plot (`fig_lm_hy_serving`).** Three panels: (a) bytes per sequence versus $n$ (log–log) for both models, measured markers and the $M(n)$ lines, $n^\*$ marked; (b) max concurrency versus $n$; (c) decode tokens/s per user versus $n$ for $c=8$, min–max bands over repetitions.
- **Expected outcome.** (a) lines agree within allocator/page-size rounding (the engine's page granularity for hybrid state is an unknown; measured/predicted ratio **to be measured**); (b) and (c) follow the direction of the hypothesis; magnitudes to be measured. Do not quote the Qwen blog's speedups as results.
- **Noise check.** Three repetitions; differences count only if min–max bands do not overlap; the two arms differ in *more than* the attention layer (data, vision tower, expert count 256 vs 128), so the plot caption states that the comparison is architectural-system, not an ablation.
- **T1 reduction.** Qwen3.5-2B vs Qwen3-1.7B on the build box with Transformers `generate` (no engine): contexts 2K, 8K, 32K (Qwen3-1.7B native 32K), batch 1 and the largest that fits; the same panels (a) and (c); same assertion of Derivation 2. **Artifacts committed (T3):** `lm-arch-hybrid-serving`.
- **Second experiment (T1, kernels).** One layer at 2B dimensions ($H_v=16,d_k=d_v=128$): tokens/s versus $T\in\{1\text{K},\dots,32\text{K}\}$ for recurrent loop, PyTorch chunked ($C=64$), `fla` chunked; expected order of magnitude gaps between the three to be measured; shows why the chunked form is what is shipped.

#### Figures and tables

| Label | Content | Cell |
|---|---|---|
| `tab_lm_hy_released` | released hybrids: layer counts by kind, fraction, placement, state/KV bytes, $n^\*$ | `d2l.design_table` subset |
| `fig_lm_hy_placement` | placement strips | `plot_placement()` |
| `tab_lm_hy_mapping` | the mismatch checklist above | static |
| `fig_lm_hy_match` | per-layer relative error of ours vs HF (bf16 and fp32 cases) with the noise-floor line | `plot_match()` |
| `fig_lm_hy_kernels` | tokens/s vs $T$ for the three implementations | `plot_kernels()` |
| `fig_lm_hy_serving` | headline | `plot_serving()` |

#### Pitfalls and misconceptions

- A recurrent layer's state is **per sequence**; a prefix cache must store a state snapshot at the shared prefix boundary, and a cached state cannot be truncated or extended backwards.
- The fixed state is *not small* in absolute terms: 1.09 MB per Qwen3.5-2B linear layer (fp32); at short contexts it exceeds the KV of the attention layers.
- Zero-centered RMSNorm weights ($1+w$) and the unscaled gated-norm weight are different conventions in the same layer; fold per tensor, not globally.
- Two "gated" mechanisms: the output gate of attention (14.4) and the decay/beta gates of the linear layer.
- Equal *active* size is not equal capability or equal data; the T3 comparison is a systems measurement.
- The hybrid's per-token FLOPs for the linear layers are constant in $n$ but not small; throughput at short context may be lower.
- A passing logit match at bf16 means "within the noise floor", not "identical": say which.

#### Summary (draft bullets)

- Released hybrids put one full-attention layer in four (gated-delta families) or one in eight to twelve (Mamba families); the 2026 line also shows retreat from linear layers (MiniMax-M2) and combination with sparse attention (GLM-5.3-Flash).
- The released linear layer extends the book's toy with a short convolution, Mamba-style decay, normalized q/k, an output gate and fp32 state; the gated chunked form is exact.
- Qwen3.5-2B loads into the book's `d2l.GPT` blocks with a short mismatch list, and its logits match Transformers within the bf16 noise floor.
- Per-sequence bytes are $L_fc_{kv}n+L_\ell c_{st}$; below $n^\*$ the state dominates; at 128K a 3B-active hybrid needs ≈ 4.7× fewer cache bytes than its attention-only peer (derived).
- Serving a hybrid means managing state snapshots as well as pages.

#### Exercises

*Check.*
1. **Which dominates?** For Qwen3.5-35B-A3B at $n=1{,}000$ and $n=100{,}000$ compute state bytes and KV bytes. *Sketch:* state 64.4 MB fixed; KV 20.5 MB and 2.05 GB.
2. **Why fold $1+w$?** What error results if you load the HF weight as the book's $w$ without folding? *Sketch:* the norm output is scaled by $w$ instead of $1+w$; since $|w|\ll1$ the activations shrink by ≈ 1/(1+w)…; logits are wrong at the first layer, localized by the layer-by-layer protocol.

*Derive.*
3. **Break-even.** Derive $n^\*$ for Kimi K3 and show it is independent of batch size. *Sketch:* $n^\*=L_\ell c_{st}/(L_fc_{kv})=449{,}372{,}160/27{,}648\approx16{,}253$; both terms scale with batch.
4. **Cost of chunking.** Show the per-token FLOPs of the chunked form are minimized at a $C$ of order $\sqrt{d_kd_v/(d_k+d_v)}$ in a model where the solve costs $C(d_k+d_v)$ and the state work costs $d_kd_v/C$. *Sketch:* minimize $C(d_k+d_v)+d_kd_v/C$; $C^\*=\sqrt{d_kd_v/(d_k+d_v)}$ (≈ 8 for 128×128): hardware, not FLOPs, picks 64.

*Measure.*
5. **Placement.** Using `lm-arch-configs`, compute for every hybrid the position of the first and the last full-attention layer as a fraction of depth; test the "never first" rule and report exceptions.
6. **Kernel ratio.** Predict the ratio of chunked to recurrent time at $T=8$K, then measure with 5 repetitions and report the range.

*Open.*
7. **Distill your own.** Replace three of every four attention layers of `Qwen3-0.6B` by `GatedDeltaNetLayer` initialized from the attention projections (as `sec_hybrids` describes) and recover with a short fine-tune; pre-register the quality gate (T1–T2).

#### Further reading

- Papers (v5): [Kimi Linear](https://arxiv.org/abs/2510.26692); [Nemotron-H](https://arxiv.org/abs/2504.03624); [Nemotron 3](https://arxiv.org/abs/2512.20856) (title verified 2026-10-28: "NVIDIA Nemotron 3: Efficient and Open Intelligence"); [MiniMax-01](https://arxiv.org/abs/2501.08313); [Jamba](https://arxiv.org/abs/2403.19887); [Kimi K3](https://arxiv.org/html/2607.24653v1) (§2.1 KDA and gated MLA; §5.1, §5.4.1); [Gated DeltaNet](https://arxiv.org/abs/2412.06464).
- Models: [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct); [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B); `nvidia/Nemotron-H-8B-Base-8K` and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (configs read 2026-10-28).
- Teaching and code: [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/); [flash-linear-attention](https://github.com/fla-org/flash-linear-attention); [`modeling_qwen3_5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) (the oracle; HTTP 200).

#### Risks and open decisions

| Risk | Fallback |
|---|---|
| Engine does not support Qwen3.5 hybrid cache at the pinned version (unverified) | Transformers + `fla` with static batch; report only (i)–(iii) predicted/measured memory and per-sequence decode; say so |
| The engine's page granularity inflates measured state bytes | report measured/predicted and the engine's block size; the exact-byte assertion is applied to `HybridCache` only |
| fp32 logit diff on 0.8B is larger than $10^{-3}$ | layer localization; likely culprit order: norm convention, RoPE half-split, gate split |
| `fla` Triton kernel unavailable on the build box | PyTorch chunked only; label the plot |
| Qwen3.5-35B-A3B + 128K prompts exceed A100 time budget | drop 256K, keep 4K–128K |
| Qwen3.8-Flash-Next-style layouts (mHC, indexer) are not loadable by this code | out of scope; table row only |

### 14.7 The residual stream across depth

- **Label:** `sec_lm_residual_stream` · **Tier:** T2 (reduced T1: `d2l.GPT`-sized, 12 blocks, width 256, 2 LRs per variant) · **Budget:** T2 ≈ 2.5–3.5 h on 1× A100 40 GB (4 variants × 4 learning rates = 16 runs capped at ≈ 8 min each, plus 2 extra baseline seeds at the reference LR, plus diagnostics); T1 ≤ 15 min (4 variants × 2 LRs × ≈ 1.2 min, 2 baseline seeds, diagnostics) · **Peak memory:** T2 design target ≤ 34 GiB (HC with $n=4$ streams multiplies residual activations by ≈ 4, which took 45 GB in the earlier 80 GB plan: the H and M arms run at half the micro-batch with doubled gradient accumulation, leaving global batch, tokens and the comparison unchanged; to be measured); T1 ≤ 10 GiB · **Codex units:** new in v5 (the Kimi K3 report's §2.2 and the V4.1 report's §2.4.1 replace Codex 16.4/16.3 for this axis)
- **Builds on (book):**
  - `sec_transformer-block`: pre-norm, the residual stream, `d2l.TransformerBlock`; `sec_init_param`: signal propagation at initialization and residual scaling (variance growth of a sum of $L$ branches). **Taken as given:** the identity-path argument and the $1/\sqrt{2L}$-style residual scaling. 14.7 starts from the *trained* consequence (norm growth and dilution) and from learned alternatives to the unit-weight sum.
  - `sec_scaling-laws` (`tab_modern-recipe`: OLMo's off-stream post-norm is the lone dissent that "concedes the identity path"), `sec_gpt`.
- **Builds on (this part):**
  - 14.2: `d2l.train_lm_steps` and R2 data path (`d2l.load_reference_model` from 4.2; validation set `lm-fwedu-val`). 14.1: `config_stats` for the extra parameters and activation traffic of each variant (`hc_mult`, `attn_res_block_size`). 14.4: `sink_metrics` (massive activations per layer) reused for the hidden-norm profile.
  - 4.3 (`sec_lm_instabilities`): the stable-LR criterion (loss diverges or spikes above a stated factor) and logit growth; 1.6: the residual stream as a workspace (what "dilution" means for readouts); 5.2: memory traffic as the cost of anything that touches the stream.
- **Exports (`#@save`):**
  - `d2l.sinkhorn(logits, iters=20, eps=1e-6)` — alternating row/column normalization of $\exp(\text{logits})$ returning an (approximately) doubly stochastic matrix; differentiable by unrolling.
  - `d2l.HyperConnection(n, d, mode='hc'|'mhc', sinkhorn_iters=20)` — wraps a sublayer $F$: input streams $X\in\mathbb{R}^{B\times T\times n\times d}$, token-wise coefficients $(A,B,C)=\mathcal{H}(X)$ from a normalization and a small projection plus static parts, update $X'=BX+C\,F(AX)$ (the form of the V4.1 report's Eq. 2); `mode='mhc'` constrains $B$ with `sinkhorn` and $A,C$ nonnegative (see Derivation 2 and the exact parameterization to be read from the mHC paper; the notebook asserts *properties*, not constants).
  - `d2l.AttnRes(num_layers, d, block=None)` — Attention Residuals: per-layer pseudo-query $w_l$, sources the embedding and previous layer outputs, kernel $\exp(w_l^\top\mathrm{RMSNorm}(k_i))$; `block=S` gives Block AttnRes with partial-sum merging by online softmax.
  - `d2l.ResidualGPT(variant, vocab, d, H, Hkv, L, ...)` — a GPT-compatible model whose blocks reuse `RefinedAttention` and the FFN but whose residual wiring is `variant in ('prenorm','hc','mhc','attnres','attnres_block','dwa','value_residual')`.
  - `d2l.depth_diagnostics(model, batch)` — per-layer RMS of the residual stream and per-layer gradient norms (norm of $\partial L/\partial h_l$ at the input of each block); `d2l.stable_lr_range(runs, tol)`.
  - Data: `lm-arch-residual-logs`.
- **Models and data:** R2 (4.2; FineWeb-Edu slice, ODC-By); no released checkpoint is trained or loaded. Configuration evidence only: `hc_mult 4`, `hc_sinkhorn_iters 20` (DeepSeek-V4-Flash/Pro/V4.1-Flash, GLM-5.3-Flash `mhc true`, Qwen3.8-Flash-Next `hc_count 4`), `attn_res_block_size 12` (Kimi K3).
- **Packages:** `torch>=2.8` (`torch.compile` for the HC arms), optional `triton` (not required).

#### Narrative arc

1. **The question.** A pre-norm transformer computes $h_L=h_0+\sum_lf_l(h_l)$: every layer's output enters with weight 1. *Is that the best way to route information across depth, and what does it cost to do better?* Opening cell: a pre-norm stack at initialization and after the notebook's short training (T0, width 128, 24 layers, seeded): RMS of the residual stream per layer (grows with $l$) and the relative size of each layer's contribution $\|f_l\|/\|h_l\|$ (shrinks). The symptom: the last layers' outputs are a small fraction of the stream they write to.
2. **Hidden-state growth and dilution under pre-norm** (`subsec_lm_res_dilution`): Derivation 1 (growth $\sqrt L$ to $L$, dilution $\propto1/\sqrt l$), the T0 measurement, the link to massive activations (14.4) and to `sec_init_param`'s residual scaling. State the AttnRes paper's claim as its own (abstract: uniform aggregation causes uncontrolled hidden-state growth with depth and dilutes each layer's contribution).
3. **Hyper-connections** (`subsec_lm_res_hc`): $n$ parallel streams mixed by learned, token-dependent matrices; what each of $A,B,C$ does (read mixing, stream-to-stream mixing, write distribution); parameter and memory cost ($n\times$ activations at the residual, the extra coefficient projections); the HC paper's claim (seesaw between gradient vanishing and representation collapse) as its own. Implement `mode='hc'`; with $n=1$, $A=B=C=1$ it is the standard residual (test).
4. **Why unconstrained mixing is unstable and what the constraint fixes** (`subsec_lm_res_mhc`): Derivation 2 (products of mixing matrices; doubly stochastic closure; spectral norm ≤ 1; mean preservation; Sinkhorn, finite iterations); the mHC claim (HC compromises the identity-mapping property, causing instability; projecting onto a manifold restores it) as its own; V4.x ship it with 4 streams and 20 Sinkhorn iterations (configs). Include the *single-pass* kernel argument: the V4.1 report counts activation traffic $(4n+4)d$ for the multi-pass implementation versus a lower bound $(2n+2)d$ and fuses to approach it; at $n=4$ that is 20$d$ and 10$d$ elements per token per layer, against ≈ $3d$ for a plain residual update (read $x$, read $y$, write $x+y$; blueprint's count): *the cost of the axis is memory traffic, not FLOPs*.
5. **Learned aggregation over depth** (`subsec_lm_res_depth`): three designs on one axis: *DenseFormer* (depth-weighted averages: $h_l\leftarrow\sum_{i\le l}w_{l,i}h_i$, scalar weights initialized to the identity), *value residuals* (ResFormer: add the first layer's value to later layers' values; SVFormer shares them and halves the KV cache, per the paper's abstract), *Attention Residuals* (softmax attention over earlier layer outputs, Derivation 3). Implement all three minimally (DWA and value residual ≤ 15 lines each).
6. **Attention Residuals in Kimi K3** (`subsec_lm_res_attnres`): full versus block form, $N\approx8$ blocks recovering most of the benefit (K3 report: 8 blocks of 12 layers plus the embedding source, 9 total; config `attn_res_block_size 12`); the cost accounting of Derivation 3 and the online-softmax merge of inter-block and intra-block partial sums; pipeline-parallel communication consequences are cited (report §5.2.2), not rebuilt.
7. **The experiment** (below): norm and gradient profiles, loss, stable LR range, memory traffic.

#### Derivations

**1. Growth and dilution of the pre-norm stream (`eq_lm_res_growth`).**

*Statement.* If layer outputs $f_l$ have RMS $\sigma$ and pairwise correlation $\rho\in[0,1]$, then $\mathbb{E}\|h_L\|^2\approx\|h_0\|^2+d\sigma^2\bigl(L+\rho L(L-1)\bigr)$ (per-coordinate variance form), so the RMS of the stream grows between $\sqrt L$ ($\rho=0$) and $L$ ($\rho=1$), and layer $l$'s share of the *final* stream is $\propto\sigma/\sqrt{\sigma^2L(1+\rho(L-1))}$.

*Steps.* (1) $h_L=h_0+\sum_lf_l$ (pre-norm: each $f_l$ is a function of the *normalized* $h_l$ so $\sigma$ stays $O(1)$ however large $h_l$ is); (2) variance of a sum: $L\sigma^2+L(L-1)\rho\sigma^2$; (3) share of layer $l$: $\sigma/\|h_L\|_{\text{rms}}$; (4) normalized-input consequence: because layer $l$ reads $h_l/\mathrm{rms}(h_l)$, its *effective relative update* is $\sigma/\mathrm{rms}(h_l)\to0$ with depth.

*Numerical check cell.* Seeded random pre-norm stack (width 256, 48 layers, random $f_l$ = frozen random 2-layer MLPs on the normalized input with unit output RMS): measure RMS of $h_l$ versus $l$ and fit the exponent; assert it lies in $[0.4,1.05]$ and print $\rho$ estimated from the outputs; then train for 200 steps on the R2 slice and print the same curve for the trained T0 model (no assertion, a measurement).

**2. Products of mixing matrices; doubly stochastic closure (`eq_lm_res_ds`).**

*Statement.* For stream-mixing matrices $B_l\in\mathbb{R}^{n\times n}$, the effective map across $L$ layers is $P_L=B_{L-1}\cdots B_0$. (a) For a fixed $B$ with spectral radius $\rho(B)$, $\|B^L\|_2\ge\rho(B)^L$ and $\|B^L\|\to0$ if $\rho<1$; for i.i.d. random $B_l$ the top Lyapunov exponent $\lambda_1=\lim\frac1L\log\|P_L\|$ exists and is generically nonzero: *geometric growth or decay*. (b) If every $B_l$ is doubly stochastic then so is $P_L$; $\|P_L\|_2\le1$; $P_L\mathbf 1=\mathbf 1$ and $\mathbf 1^\top P_L=\mathbf 1^\top$, so the stream mean $\frac1n\mathbf 1^\top x$ is preserved exactly: the identity-mapping property survives in the mean.

*Assumptions.* Nonnegative entries; exact row and column sums of 1 (the implementation uses finitely many Sinkhorn iterations: the check measures the drift).

*Steps.* (1) $\|B^L\|\ge\rho^L$ because $\rho(B)\le\|B^L\|^{1/L}$; (2) closure: $(AB)\mathbf 1=A\mathbf 1=\mathbf 1$, $\mathbf 1^\top AB=\mathbf 1^\top$, entries stay nonnegative; (3) $\|B\|_2\le\sqrt{\|B\|_1\|B\|_\infty}=1$ for doubly stochastic $B$ (Riesz–Thorin/Hölder bound on the induced norms); (4) mean preservation from $\mathbf 1^\top B=\mathbf 1^\top$; (5) Birkhoff: the set of doubly stochastic matrices is the convex hull of permutation matrices, so $B=\sum_k\theta_kP_k$ is a *soft permutation* (streams are re-ordered, not amplified); (6) Sinkhorn: for a positive matrix, alternating row and column normalization converges to a unique doubly stochastic matrix of the form $D_1MD_2$; after $T$ iterations the column sums are exactly 1 and the row sums are within a residual $r_T$ that decreases with $T$.

*Numerical check cell.* (seed 0, $n=4$, $L=96$): (a) unconstrained: $B_l=I+0.1\,\mathcal N(0,I)$ entries: print $\log\|P_L\|/L$ (nonzero Lyapunov estimate) and the min and max singular values of $P_L$; (b) Sinkhorn with $T=20$ on $\exp(\mathcal N(0,1))$ logits: assert column sums within $10^{-6}$, report max row-sum deviation $r_{20}$ as a function of $T\in\{1,2,5,10,20,50\}$ (a measurement); assert $\|P_L\|_2\le1+L\,r_{20}$ and $|\frac1n\mathbf 1^\top P_Lx-\frac1n\mathbf 1^\top x|\le L\,r_{20}\|x\|$ (the bound uses the finite-iteration residual; say so); (c) print the spectral norm of $P_L$ for the constrained product: it should stay $\le1$ and the *second* singular value should decay (streams decorrelate while the mean persists).

**3. Attention Residuals: boundedness and cost (`eq_lm_res_attnres`).**

*Statement.* AttnRes layer inputs $h_l=\sum_{i<l}\alpha_{i\to l}v_i$ with $\alpha\in\Delta$ (softmax of $w_l^\top\mathrm{RMSNorm}(v_i)$) are convex combinations, so $\|h_l\|\le\max_{i<l}\|v_i\|$ (no growth with depth by construction); the per-token cost of the full form is $4\sum_l ld\approx2L^2d$ FLOPs and $Ld$ stored elements; the block form with $N$ blocks stores $Nd$ elements.

*Steps.* (1) Convexity and the triangle inequality; (2) scores and weighted sum are two length-$l$ dot-product passes of width $d$: $2ld+2ld$ per layer; (3) summing over layers $2L(L+1)d$; (4) block form: sources are $N$ block sums plus one partial sum; (5) the merge: inter-block attention (over completed blocks, computed once per block) and the intra-block partial-sum source combine by online-softmax rescaling, exactly equal to the single softmax.
*Pre-check (arithmetic from Kimi K3's config and report):* $L=93$, $d=7168$: full stores $93\times7168\times2$ B $=1.33$ MB per token in bf16; block form with 9 sources $0.13$ MB (≈ 10× less); FLOPs $2\cdot93^2\cdot7168\approx1.2\times10^8$ per token versus ≈ $2\times10^{11}$ of the model's active matmul FLOPs (0.06%, using the 14.1 pre-check of 102.1B active parameters).

*Numerical check cell.* (a) Convexity bound on random sources; (b) full AttnRes versus Block AttnRes with block size 1: identical outputs ($<10^{-6}$); (c) the online-softmax merge equals the direct softmax over the concatenated sources ($<10^{-6}$ fp32); (d) cost counters equal the formulas above via `FlopCounterMode`.

#### From-scratch implementation

Budget ≈ 140 lines: `sinkhorn` (≈ 10), `HyperConnection` (≈ 45), `AttnRes` incl. block form and merge (≈ 50), DWA and value residual (≈ 25), `ResidualGPT` wiring and `depth_diagnostics` (≈ 40 additional, counted separately in the notebook).

| Test | Reference | Tolerance / seed |
|---|---|---|
| `HyperConnection(n=1)` with $A=B=C=1$ equals a standard pre-norm residual block | `d2l.TransformerBlock` (pre-norm) | fp32 $<10^{-6}$ |
| `mhc`: row sums of $B$ within $r_T$, column sums exactly 1, $B\ge0$ | property checks, 1,000 random tokens | column $10^{-6}$; row deviation reported |
| Product closure and norm bound (Derivation 2) | property | as stated |
| Streams: expanding $h_0$ to $n$ copies and reading the mean at the output reproduces the baseline when all coefficients are the identity-like initialization | baseline model with the same weights | fp32 $<10^{-5}$ |
| AttnRes: convexity, block = full at block size 1, merge exactness | Derivation 3 | $10^{-6}$ |
| DWA with identity-initialized weights equals the baseline | baseline | exact |
| Parameter counts of variants | `config_stats` extension (`hc_mult`, `attn_res`) | exact |

#### In production

DeepSeek-V4.x and GLM-5.3-Flash ship mHC with $n=4$ and 20 Sinkhorn iterations (configs); the V4.1 report's §2.4.1 describes a *single-pass* fused implementation of residual update, coefficient computation and input mixing that reads $(n+1)d$ values and approaches the $(2n+2)d$ traffic bound. K3 ships Block AttnRes (8 blocks of 12 layers) with cache-based pipeline communication and a two-phase computation (arXiv 2603.15031; K3 §5.2.2 for activation memory). The from-scratch modules are the numerical oracle; their naive PyTorch cost is the *measured baseline* that makes the kernel argument concrete (headline panel iv).

#### Headline experiment

- **Hypothesis (directional, from sources).** Relative to pre-norm residuals: (i) unconstrained HC trains to a similar or lower loss at moderate LR but loses stability at higher LR (the mHC paper's claim: loss of the identity-mapping property causes instability); (ii) mHC restores the stable range while keeping the loss benefit; (iii) AttnRes flattens the growth of residual-stream RMS with depth (by Derivation 3) and improves loss at equal tokens (the AttnRes paper reports consistent improvement across scales); (iv) HC variants cost memory traffic and throughput, AttnRes less so. Sizes at R2 scale **to be measured**; a tie on loss with differences in (i), (iii), (iv) is an acceptable outcome.
- **Setup (T2).** R2 architecture and recipe (4.2) with the residual variant changed: **P** pre-norm; **H** `hc` ($n=4$); **M** `mhc` ($n=4$, 20 iterations); **A** `attnres_block` (block size so that $N\approx L/8$, at least 4 blocks). Peak learning rates: $\{0.5,1,2,4\}\times$ R2's recipe LR; identical tokens (≈ 8 min each by calibration), data order and schedule; seed 0 for all; seeds 1 and 2 for P at $1\times$ to measure $\sigma$.
- **Procedure.** (1) Train 16 runs, log loss every 50 steps and any non-finite value; (2) at the end, `depth_diagnostics` on 32 held-out sequences: per-layer residual RMS and per-layer gradient norm; (3) mark a run *unstable* if the loss is non-finite or exceeds the run's earlier minimum by more than 0.5 nats after step 100 (4.3's criterion; the threshold is a pre-registered constant); (4) compute `stable_lr_range`: the set of LRs whose final loss is within $2\sigma$ of the best of that variant and which are stable; (5) throughput and peak memory.
- **Metrics:** final validation loss per (variant, LR); stable-LR range per variant; residual RMS versus layer; gradient norm versus layer (ratio of first to last layer); tokens/s; peak memory; extra parameters.
- **Plot (`fig_lm_res_variants`).** Four panels: (i) validation loss versus LR (log-x) per variant, unstable runs as open markers, the $2\sigma$ band around the best baseline loss; (ii) residual RMS versus layer at $1\times$ LR; (iii) gradient norm versus layer; (iv) tokens/s and peak memory relative to P.
- **Expected outcome.** As in the hypothesis; direction only for (i)–(iii); (iv) *predicted* qualitatively from the traffic counts (H and M slower than P; A close to P), magnitudes to be measured.
- **Noise check.** Baseline seed std $\sigma$ at $1\times$ LR; all loss differences below $2\sigma$ are ties; the stable-LR boundary is resolved only to the grid factor 2 (stated); no claim about "the" optimal LR.
- **T1 reduction.** `d2l.GPT`-like width 256, **12** blocks (depth matters for this axis), 20M-token slice, LRs $\{1,3\}\times$ the book's default AdamW LR, 2 baseline seeds; ≈ 15 min total. **Artifacts committed (T2):** `lm-arch-residual-logs`.

#### Figures and tables

| Label | Content | Cell |
|---|---|---|
| `fig_lm_res_growth` | T0: RMS of the stream and relative layer contribution versus depth (pre-norm) | `plot_growth()` |
| `fig_lm_res_product` | Derivation 2: singular values of $P_L$ for unconstrained and doubly stochastic mixing; Sinkhorn residual versus $T$ | `plot_product()` |
| `tab_lm_res_cost` | extra parameters, activation traffic per token per layer in units of $d$, stored elements for HC, mHC (multi-pass and fused), DWA, AttnRes full and block (K3 numbers) | formulas |
| `fig_lm_res_variants` | headline | `plot_variants()` |
| `tab_lm_res_released` | which released models use which scheme (configs: `hc_mult`, `hc_count`, `mhc`, `attn_res_block_size`) | cell 1 |

#### Pitfalls and misconceptions

- "More streams" is not "more capacity" for free: activations at the residual scale with $n$ and every layer touches all streams.
- Sinkhorn with finitely many iterations is only *approximately* doubly stochastic; its residual compounds in products (Derivation 2's bound); 20 iterations in the configs is a trade-off, not exactness.
- Doubly stochastic $B$ preserves the stream *mean*, not each stream: streams may mix and decorrelate.
- AttnRes layer inputs are convex combinations, so unlike pre-norm there is no identity path unless the attention learns to put weight on the previous output; initialization of the pseudo-queries matters (zero gives uniform weights).
- The AttnRes block granularity (decoder layer versus sublayer) is a modeling choice; the notebook states which it uses and the author must read arXiv 2603.15031 for the paper's.
- Residual-stream RMS is not comparable across variants that expand to $n$ streams; compare the *read-out* (the mean) or the input to the first sublayer.
- Stability comparisons need a pre-registered divergence criterion; a single-seed stable/unstable verdict at one LR is not a boundary.

#### Summary (draft bullets)

- Under pre-norm every layer adds with weight 1: the stream's RMS grows with depth and each layer's relative contribution shrinks.
- Hyper-connections widen the stream to $n$ copies mixed by learned matrices; unconstrained mixing multiplies, across depth, into geometric growth or decay.
- Doubly stochastic mixing is closed under products, has spectral norm ≤ 1 and preserves the stream mean; Sinkhorn projection provides it approximately.
- Attention Residuals replace the unit-weight sum by softmax attention over earlier layers' outputs: bounded by construction, $O(L^2d)$ FLOPs, $O(Ld)$ or $O(Nd)$ memory.
- The price of this axis is activation memory and traffic, which is why DeepSeek and Moonshot ship fused kernels and block structure.
- At R2 scale expect differences in stability and norm profiles before differences in loss.

#### Exercises

*Check.*
1. **Dilution.** With $\rho=0$ and $L=64$ layers, by what factor is layer 64's share of the final stream smaller than layer 1's share of $h_2$? *Sketch:* $\sigma/\sqrt{64}\,\sigma$ versus $\sigma/\sqrt2\,\sigma$: ratio $\sqrt{2/64}\approx0.18$ (relative to the stream each writes to).
2. **Traffic.** For $n=4$ compute the elements moved per token per layer by multi-pass mHC, by the fused lower bound and by a plain residual update. *Sketch:* $(4n+4)d=20d$, $(2n+2)d=10d$, $3d$.

*Derive.*
3. **Closure.** Prove that the set of $n\times n$ doubly stochastic matrices is closed under multiplication and has operator 2-norm ≤ 1. *Sketch:* Derivation 2 steps 2–3; $\|B\|_2^2\le\|B\|_1\|B\|_\infty=1$.
4. **Block cost.** For $L$ layers and block size $S$, derive the stored elements and the FLOPs per token of Block AttnRes and the $S$ minimizing FLOPs plus a memory penalty $\mu$ per stored element. *Sketch:* sources per layer $\approx L/S+1$; FLOPs $\approx4Ld(L/S+1)$, stored $(L/S+1)d$; both decrease in $S$, so the optimum is set by quality, not cost: the cost curve is monotone (state this and say what measurement would pick $S$).

*Measure.*
5. **Number of streams.** Compare $n\in\{2,4,8\}$ for mHC at T1; predict whether loss improves monotonically with $n$ and the throughput cost; test with 2 seeds.
6. **Sinkhorn iterations.** Replace 20 iterations by 2 and 5: does stability change? Report the row-sum residual and the stable-LR range.

*Open.*
7. **Depth weighting versus attention.** Implement DWA and a value residual at T1 and compare with AttnRes on the T1 stability and loss panels; write a one-paragraph decision memo with a pre-registered criterion.

#### Further reading

- Papers (v5): [Hyper-Connections](https://arxiv.org/abs/2409.19606); [mHC](https://arxiv.org/abs/2512.24880); [Attention Residuals](https://arxiv.org/abs/2603.15031) (abstract: PreNorm accumulation with fixed unit weights causes hidden-state growth and dilution; Block AttnRes for memory); [Value Residual Learning](https://arxiv.org/abs/2410.17897) (ResFormer: equal loss with 16.11% fewer parameters; SVFormer halves KV cache); [DenseFormer](https://arxiv.org/abs/2402.02622) (depth-weighted averaging); [Kimi K3](https://arxiv.org/html/2607.24653v1) (§2.2, §5.2.2).
- *New (verified in this blueprint):* [DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1) §2.4.1 (single-pass mHC and its memory-traffic count, already v5-listed in 14.1) and [DeepSeek-V4](https://arxiv.org/abs/2606.19348) (first model to ship mHC per its config).
- Teaching and code: Raschka's architecture comparison for where each lab put its norms; `d2l` `sec_init_param` for the initialization view; the mHC paper's infrastructure section and the V4.1 report for kernels; no public reference implementation of AttnRes was located (the K3 repository [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) returned HTTP 200 and may carry modeling code: the author checks).

#### Risks and open decisions

| Risk | Fallback |
|---|---|
| Exact mHC parameterization (activations on $A$, $C$, normalization in $\mathcal H$) not matched | tests assert properties (nonnegativity, DS, closure); document deviations in the notebook; read the paper §3 |
| AttnRes granularity or initialization differs from the paper | the notebook states its choice; `granularity` flag; report a tie if results are sensitive |
| HC arms exceed memory/time at R2 scale (the 38 GiB usable on A100 40 GB) | quarter micro-batch with more accumulation; then $n=2$, or fewer layers; or `torch.compile`; report the reduced setting |
| Stability events are rare at the tested LRs | extend the LR grid by $\times8$ for P only; or use a deeper model (T1: 24 blocks) |
| 16 runs × 8 min exceed the T2 budget | drop LR $0.5\times$ for all but P |

### 14.8 ◇ Beyond fixed tokens and left-to-right decoding

- **Label:** `sec_lm_beyond_tokens` · **Tier:** T1 · **Budget:** ≤ 15 min on the build box: byte-level AR LM on R3 (≈ 3 min), entropy patching and BPE comparison (≈ 1 min), MTP auxiliary-head arms (2 arms × 2 seeds × ≈ 1 min), tiny masked-diffusion LM (≈ 3 min) and the steps sweep (≈ 1 min) · **Peak memory:** ≤ 8 GiB · **Codex units:** 2.6 (auxiliary prediction; conditional memory went to 14.2's `subsec_lm_moe_lookup`), 2.8 (diffusion LMs; the text-scale example is proposed for `sec_diffusion-discrete` under v5's "Changes outside this part")
- **Builds on (book):** `sec_text-sequence` (byte-level BPE, tokens per byte; **not re-taught**); `sec_diffusion-discrete` (masked-diffusion bound `eq_diffusion-masked-loss`, unmasking sampler, the $1/S$ independence error on a two-token example, "masked LM = one-step diffusion"); **taken as given:** the objective and sampler; 14.8 adds text scale, confidence-based unmasking, and a quality-versus-steps measurement. `sec_gpt`, `sec_language-model` (bits per byte).
- **Builds on (this part):** 3.6 (`sec_lm_tokenizers`: tokenizer statistics; pinned Qwen3 tokenizer), 12.3 (`sec_lm_drafters`: reuses MTP heads for drafting; the acceptance-rate derivation lives there), 14.1 (the `num_nextn_predict_layers`, `mtp_*`, `dspark_*` columns), 14.2 (`d2l.train_lm_steps`), 1.5 (`d2l.paired_bootstrap`), R3 (`d2l.book_corpus`, 2.4).
- **Exports (`#@save`):** `d2l.mtp_loss(h, heads, tokens, k=2, weights=(1.0, 0.3))` (sequential-module and parallel-head variants); `d2l.entropy_patches(logits, threshold, *, max_len=None)` (patch boundaries where next-byte entropy exceeds the threshold, per the BLT description); `d2l.patch_flops(n_global, n_local, patch_len)` and `d2l.bpe_flops(n_params, bytes_per_token)`; `d2l.MaskedDiffusionLM(vocab, d, H, L)` (bidirectional, mask token, weighted cross-entropy of `eq_diffusion-masked-loss`); `d2l.sample_masked_diffusion(model, length, steps, remask='confidence'|'random', seed)`. Data: none beyond logs (≤ 1 MB JSON).
- **Models and data:** R3 text as bytes (the book; license of the repo); `Qwen/Qwen3-0.6B` tokenizer (Apache-2.0); header-only inspection of `Qwen/Qwen3.5-2B` for `mtp.*` (`mtp.fc [2048, 4096]`, one attention+MLP layer, `mtp.pre_fc_norm_embedding`; 60,828,160 parameters); optional T2 pointer: `GSAI-ML/LLaDA-8B-Instruct` (v5 link; license read from the card by the author).
- **Packages:** `torch>=2.8`, `tokenizers`/`transformers`.

#### Narrative arc

1. **The question.** *Why does a language model predict one token, from a fixed vocabulary, left to right?* Opening cell: the design-table columns of 14.1 for output heads: Qwen3.5 (`mtp_num_hidden_layers 1`), DeepSeek-V3 (`num_nextn_predict_layers 1`), V4.1-Flash (3 plus a DSpark draft module with `dspark_block_size 5`), MiniMax-M2 (`num_mtp_modules 3`), MiniMax-M3 (`num_mtp_modules 7`), GLM-4.5 (1). Symptom: shipped checkpoints carry extra prediction machinery that the base transformer does not need.
2. **Multi-token prediction as a training objective** (`subsec_lm_btok_mtp`): $\mathcal L=\mathcal L_1+\sum_{j=2}^{k}\lambda_j\mathcal L_j$; parallel heads (the MTP paper) versus a sequential module that conditions on the embedding of the next token (DeepSeek-V3). What it changes in training versus what it enables at inference (drafting: 12.3). Arms in the experiment: baseline NTP versus NTP + one auxiliary head, on the byte LM.
3. **Learned segmentation of bytes** (`subsec_lm_btok_bytes`): why tokenizers are a fixed preprocessing step and what dynamic patching changes: entropy-based patches (BLT) place boundaries where the next byte is hard to predict; H-Net learns boundaries end to end. Derivation: compute per byte as a function of mean patch length. Measure on R3: patch length distribution at several entropy thresholds from the notebook's byte LM versus the Qwen3 BPE token lengths in bytes.
4. **Masked diffusion at text scale** (`subsec_lm_btok_diffusion`): LLaDA's masked-diffusion LM (the bidirectional network predicts all masked tokens; training objective is the weighted masked cross-entropy of the book); sampling by iterative unmasking with *remasking*; the independence error of parallel unmasking from the book (`1/S` on the two-token example) as the quantity to measure at text scale. Tiny model on R3 bytes in windows of 64; quality metric: NLL of samples under the AR byte LM of step 1 (an external judge: the notebook states its bias) and distinct-n; versus number of steps $S\in\{1,2,4,8,16,32,64\}$; two remasking rules (random vs confidence-based). Commercial diffusion LMs (Mercury) are cited, not reproduced.
5. **What this axis is not** (`subsec_lm_btok_limits`): honesty about scale: tiny models do not reproduce LLaDA's or BLT's reported results; the notebook checks mechanisms (compute formulas, sampler behavior), not frontier claims.

#### Derivations

**1. Compute per byte of a patch-based model (`eq_lm_btok_flops`).** *Statement.* A byte-level model with a local encoder/decoder of $N_l$ parameters running per byte and a global latent transformer of $N_g$ parameters running per patch of mean length $p$ bytes costs $C_{\text{patch}}=2N_l+2N_g/p$ FLOPs per byte (matmul FLOPs only); a BPE model of $N_b$ parameters with $t$ bytes per token costs $C_{\text{bpe}}=2N_b/t$. They are equal when $N_g=p\,(N_b/t-N_l)$.
*Steps.* (1) FLOPs per forward token $2N$ (`sec_scaling-laws`); (2) local layers see every byte; global layers see $1/p$ of positions; (3) BPE sees $1/t$ of bytes as tokens; (4) equate and solve; (5) **corollary:** at fixed $N_g$, increasing $p$ lowers cost until $N_l$ dominates; at fixed FLOPs, longer patches buy a larger global model.
*Numerical check cell.* Count FLOPs with `FlopCounterMode` on the notebook's tiny patch-model instance (local conv/linear + global `d2l.GPT`, byte inputs) and on a BPE model of matched FLOPs; assert the formula within 5% (matmul-only accounting); evaluate the equality condition for the measured $p$ and $t$ of R3 (values to be measured).

**2. Why parallel unmasking costs quality (`eq_lm_btok_indep`).** *Statement (book's result, applied).* Tokens revealed in the same step are drawn independently from their marginals; for a pair the error is $\mathrm{TV}(p^1\otimes p^2,p)/S$ with equally spaced levels (the book's two-token example). *Text-scale check:* estimate the pairwise dependence between adjacent masked bytes under the trained model and verify on a synthetic distribution with known joint (two perfectly correlated bits) that the sampler's error at $S$ steps matches $1/S$ of the book's formula within sampling error (seeded, 20,000 samples, a pair of tokens; tolerance 0.01 absolute). *Steps for text:* none new; the experiment measures how the NLL-of-samples degrades with $S$.

#### From-scratch implementation

Budget ≈ 130 lines: `mtp_loss` (≈ 20), `entropy_patches` (≈ 15), `MaskedDiffusionLM` (reuses `d2l.GPT` blocks with `causal=False` via `GQAAttention(causal=False)`; ≈ 25), sampler (≈ 35), flops helpers (≈ 10).

| Test | Reference | Tolerance / seed |
|---|---|---|
| `mtp_loss` with $k=1$ equals the standard NTP loss | `F.cross_entropy` | exact |
| MTP target alignment (head $j$ at position $t$ predicts $t+j$) | brute-force loop | exact |
| `entropy_patches` with threshold $+\infty$ gives one patch per sequence; $-\infty$ one per byte | direct | exact |
| Masked loss with all tokens masked at weight 1 equals BERT-style cross-entropy | direct | $10^{-6}$ |
| Sampler with $S=L$ and random remask reveals one token per step | counting | exact |
| Two-bit synthetic joint: sampler error versus $1/S$ | book's formula | 0.01 abs. |
| FLOP formula vs `FlopCounterMode` | Derivation 1 | 5% |

#### In production

DeepSeek-V3's MTP module (sequential, shares embedding and head, used for training and for speculative decoding), Qwen3.5's `mtp.*` head (inspect shapes; 12.3 drafts with it), MiniMax's multiple MTP modules (configs). BLT code (`facebookresearch/blt`, v5-listed) and H-Net (arXiv 2507.07955); LLaDA's open implementation (github.com/ML-GSAI/LLaDA, HTTP 200) and sampling code as the reference for remasking; Mercury is a commercial reference only.

#### Headline experiment

- **Hypothesis.** (i) Adding a 2-token MTP head does not hurt and may slightly lower main-head validation loss at this scale (the MTP paper reports gains growing with model size and for code; at 5M parameters the effect is **to be measured** and ties are expected); (ii) entropy-patch lengths grow as the threshold rises and exceed BPE token lengths only for high thresholds (measured); (iii) NLL-of-samples under the AR judge improves as steps increase and saturates, with confidence-based remasking better than random at small $S$ (the direction the LLaDA sampling literature suggests; to be measured).
- **Setup.** `d2l.GPT` width 256, 6 blocks, byte vocabulary 256, context 256, R3 train/val split by chapter, ≈ 3 min of training for the AR byte LM; MTP arms: NTP versus NTP+1 head (weight 0.3), 2 seeds, ≈ 1 min each; masked-diffusion model width 256, 6 blocks, windows of 64 bytes, ≈ 3 min; sampler sweep $S\in\{1,\dots,64\}$ with 200 samples per $S$ and 3 sampling seeds.
- **Metrics:** bits per byte (AR); patch length versus threshold; compute per byte (Derivation 1); NLL of samples under the AR judge (bits per byte), distinct-2; the MTP head's accuracy for token $t+2$ given the correct $t+1$.
- **Plot (`fig_lm_btok`).** Panels: (a) bits per byte, NTP vs +MTP with seed ranges; (b) mean patch length versus entropy threshold with the BPE mean bytes per token as a horizontal line; (c) sample NLL versus steps for the two remasking rules with seed ranges.
- **Noise check.** 2 seeds for (a): differences within the seed range are ties; (b) is deterministic given the model; (c) three sampling seeds and 200 samples per point with a bootstrap interval.
- **Artifacts committed:** the three JSON logs (no `DATA_HUB` weights).

#### Figures and tables

`tab_lm_btok_heads` (released output-head columns from configs); `fig_lm_btok` (above); `fig_lm_btok_flops` (compute per byte versus $p$ for several $N_g$ against a BPE baseline); `tab_lm_btok_patches` (threshold, mean patch length, bytes per BPE token).

#### Pitfalls and misconceptions

- MTP as training objective and MTP as drafter are different claims; the second needs acceptance measurements (12.3).
- Entropy patching needs a *separate* small model to produce entropies; its cost is part of the system.
- Bits per byte, not per token, is the comparable metric across tokenizers and patchers.
- A diffusion LM's "steps" are full bidirectional forward passes over the whole sequence; compare wall time and model evaluations with AR decoding at matched quality, not step counts.
- An AR judge flatters AR-like samples; report distinct-n as well.

#### Summary (draft bullets)

- Released models carry extra output machinery (MTP heads, draft modules); as a training objective MTP supervises several future tokens.
- Dynamic patching moves the unit of computation from the token to a learned segment; its cost is $2N_l+2N_g/p$ per byte.
- Masked diffusion decodes by iterative unmasking; revealing several tokens per step trades quality for fewer sequential steps, with the independence error derived in the book.
- At small scale the notebook verifies mechanisms, not the published scale results.

#### Exercises

*Check.* 1. **Which units?** Convert bits per token of a BPE model to bits per byte given $t=4.1$ bytes per token. *Sketch:* divide by $t$. 2. **Patch extremes.** What do thresholds $0$ and $\log_2256$ bits give? *Sketch:* one patch per byte (entropy ≥ 0 always) and one per sequence (entropy ≤ 8 bits).
*Derive.* 3. **Breakeven patch length.** From Derivation 1 find the patch length at which a model with $N_g=N_b$ equals BPE's cost for $N_l=0.1N_b$, $t=4$. *Sketch:* $p=N_g/(N_b/t-N_l)=1/(0.25-0.1)=6.7$ bytes. 4. **Two-bit error.** Reproduce $\mathrm{TV}/S$ for two perfectly correlated fair bits. *Sketch:* $p^1\otimes p^2$ is uniform on 4 outcomes, $p$ on 2: TV $=1/2$.
*Measure.* 5. **MTP weight.** Sweep the auxiliary weight $\{0,0.1,0.3,1\}$ and report the seed-range-qualified optimum. 6. **Remasking.** Add a third rule (low-confidence remask of already revealed tokens) and compare at $S=8$.
*Open.* 7. **Patch-aware attention budget.** Combine entropy patching with 14.5's block selection (patches as blocks) in a 150-line experiment; pre-register the criterion.

#### Further reading

- Papers (v5): [Multi-token prediction](https://arxiv.org/abs/2404.19737); [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) and [code](https://github.com/facebookresearch/blt); [H-Net](https://arxiv.org/abs/2507.07955); [LLaDA](https://arxiv.org/abs/2502.09992) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct); [Mercury](https://arxiv.org/abs/2506.17298).
- *New (verified 2026-10-28):* [D3PM](https://arxiv.org/abs/2107.03006) (structured discrete diffusion; Codex 2.8); [DSpark](https://arxiv.org/abs/2607.05147) (confidence-scheduled speculative decoding with semi-autoregressive drafting; title verified); [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (v5, MTP module).
- Teaching and code: [LLaDA repository](https://github.com/ML-GSAI/LLaDA) (HTTP 200); the book's `sec_diffusion-discrete` exercises.

#### Risks and open decisions

| Risk | Fallback |
|---|---|
| MTP effect is below noise at 5M parameters (expected) | report as tie; the mechanism tests carry the section |
| Diffusion sample NLL judged by a weak AR byte LM is noisy | use more samples; add distinct-n and a likelihood bound on held-out windows |
| `sec_diffusion-discrete` text-scale example (v5's outside change) is not yet written | 14.8 is self-contained; it will cite it when it exists |
| LLaDA-8B at T2 is not run | pointer only |

### 14.9 ◇ Designing under a budget

- **Label:** `sec_lm_design_budget` · **Tier:** T0 (the measured baseline of the Assignment is T1) · **Budget:** minutes on CPU for the design matrix; the Assignment's T1 measurement ≤ 15 min · **Peak memory:** < 2 GB · **Codex units:** 16.5, 16.6 (16.2–16.4 are the dense, V4.1 and K3 cases already in 14.1 and 14.3–14.7)
- **Builds on (book):** `sec_scaling-laws` (`eq_six_nd`, the miniature fit), `sec_hardware` (roofline); nothing is re-derived.
- **Builds on (this part):** 14.1 (`d2l.config_stats`, `d2l.design_table`, report schema); 5.2 (decode-step model and `d2l.ServingModel(cfg, chip, ..., kv_bytes_override=...)`: **`ConfigStats` feeds the override hook** for MLA and hybrid rows, per the editor's note); 5.6 (cost of owned versus rented clusters; GPU-hour prices as inputs, not invented here); 4.7 (`sec_lm_budgets`: lifetime-cost calculator and overtraining); `d2l.chip_specs()` (5.1). Measured numbers from 14.2–14.7 are optional inputs (`lm-arch-*` artifacts).
- **Exports (`#@save`):** `d2l.Scenario` and `d2l.load_scenario(name)`; `d2l.check_design(design, claims, scenario)` and `d2l.predict_decode(design, scenario, batch, context)` (the Assignment's starter code, with `predict_decode` implemented as `ServingModel` plus `config_stats`); `d2l.design_matrix(scenario, candidates)`; `d2l.rank_experiments(experiments)`. No data artifacts.
- **Models and data:** candidates taken from the design table (Qwen3-30B-A3B, Qwen3.5-35B-A3B, DeepSeek-V3, Kimi-Linear-48B-A3B, gpt-oss-120b, Kimi K3 as the large extreme) through their pinned configs; hardware rows from `d2l.chip_specs()`.
- **Packages:** `pandas`, `numpy`.

#### Narrative arc

1. **The question.** *Given a task, data, compute, hardware and a serving objective, which of the axes of this chapter do you pull, and what do you measure first?* Opening cell: the Assignment's Scenario A printed as a requirements table, then one line: *which resource binds* for each candidate, computed by `predict_decode` (the notebook prints it; no numbers are pre-claimed here except the derived cache bytes in the sections above, e.g. Qwen3-30B-A3B 12.9 GB per sequence at 128K versus Qwen3.5-35B-A3B 2.75 GB).
2. **From requirements to constraints** (`subsec_lm_bud_constraints`): turn each requirement into an inequality on `ConfigStats`: weights ≤ memory share, KV+state × concurrency × context ≤ cache share, decode step time ≤ SLO from the 5.2 model, $6N_{\text{act}}D\le C_{\text{train}}$. Output: a feasible region over (total, active, cache bytes per token, state bytes).
3. **The design matrix** (`subsec_lm_bud_matrix`): rows = candidate designs (released archetypes: dense GQA, MoE GQA, MoE MLA, hybrid MoE, sparse attention), columns = constraints and the binding resource, with a sensitivity column: the derivative of the binding term with respect to context and batch (finite differences). Mark which cells come from `config_stats` (derived), from a section's measurement (artifact), or from a report (disclosed) using the 14.1 schema.
4. **What to measure first** (`subsec_lm_bud_experiments`): Derivation. A transparent heuristic for ranking experiments: score $=\dfrac{\text{prior uncertainty of the decision-relevant quantity}\times\text{decision sensitivity}}{\text{cost}}$, with uncertainty taken from the spread of published or measured values (e.g. the range of $\tau_1$ across repetitions in 14.3), sensitivity from the matrix's finite differences, cost in GPU-hours. Show it picks (for Scenario A) the measurement of the cache/state bytes and decode slope of the hybrid before any quality ablation, and an ablation of an axis only after the constraint analysis says that axis binds.
5. **The dossier and the rejected design** (`subsec_lm_bud_dossier`): pre-registration (a `rules.yaml` written before a run), the verdict logic (rejected / adopted / inconclusive; interval versus threshold; no verdict from a difference inside $2\sigma$), and a worked example in prose: a rejected design (for example, GQA→MLA at matched cache at R2-tiny scale) with its evidence figure.
6. **The Assignment** (`subsec_lm_bud_assignment`): points to the chapter's Assignment; walks scenario B end to end at T1 as the worked example (design, baseline of Qwen3-0.6B vs Qwen3.5-0.8B, one rejected design, dossier skeleton).

#### Derivations

**1. Binding resource of one decode step (`eq_lm_bud_bind`).** *Statement.* With batch $b$, context $L$, active parameters $N_a$, weight bytes $W$, cache bytes per token $k$ and fixed state $s$: $t=\max\!\bigl(\tfrac{2N_ab+F_1bL}{P},\ \tfrac{W_{\text{touched}}(b)+b(kL+s)}{\beta}\bigr)$, subject to $W+b(kL+s)\le C$; the binding resource is the argument of the max, or capacity if the constraint is active. *Steps:* (1) 5.2's model with `config_stats` fields ($F_1$ is `flops_per_ctx_token`, `k` is `cache_bytes_per_token`, `s` is `state_bytes`); (2) for MoE, $W_{\text{touched}}(b)=W_{\text{nonexp}}+W_{\exp}(1-(1-k_e/E)^b)$ (14.2 Derivation C); (3) maximum batch from the capacity constraint $b_{\max}=\lfloor(C-W)/(kL+s)\rfloor$; (4) sensitivities $\partial t/\partial L$ and $\partial b_{\max}/\partial L$ by closed form where possible. *Check cell:* `predict_decode` equals the direct formula to $10^{-9}$ for three candidates; the per-sequence bytes of the hybrid at 128K are ≈ 4.7× smaller than its attention-only peer's, so its maximum batch is ≈ 4.7× larger when both have the same $C-W$ on one device and *before* overheads; with the cache replicated across a TP = 8 group the ratio is lower (≈ 2.5× in 14.6's pilot arithmetic).

**2. Ranking experiments (`eq_lm_bud_rank`).** *Statement.* For decision $D\in\{$design A, design B$\}$ determined by $\hat q=\arg\min$ of a predicted cost, the value of resolving the uncertainty of one input $x_i$ is bounded by $\sigma_i\,|\partial q/\partial x_i|$ (first-order), a *heuristic* upper bound on how far the decision statistic can move. *Steps:* linearize, divide by GPU-hours. *Check:* on a toy with two inputs of known variances the ranking matches a Monte-Carlo estimate of decision flips (10,000 draws, seed 0, ranking agreement).

#### From-scratch implementation

≈ 100 lines: `Scenario`, `design_matrix` (≈ 35), `predict_decode` wrapper (≈ 25), `check_design` (≈ 25), `rank_experiments` (≈ 15). Tests: constraint table for a hand-built toy scenario with known answers (exact); `predict_decode` against the direct formula (10⁻⁹); `check_design` flags a design that violates each constraint in turn (5 unit tests, exact); ranking toy (above).

#### In production

The same arithmetic is what inference-capacity planning does (vLLM's `max_concurrency` reports, SGLang's token budgets); MLPerf Inference and the lm-evaluation-harness (Codex 16.6) supply measurement protocols; the Ultra-Scale Playbook supplies the training-side analogue. The Assignment's autograder imports `check_design`.

#### Headline experiment

*This section's experiment is the Assignment's T2/T3 pair (see "Assignment"); the notebook only runs the T0 design matrix.* **Hypothesis:** for Scenario A at 128K context the binding resource differs across archetypes (capacity for attention-only designs, compute or bandwidth for hybrids), so the cheapest first measurement differs too. **Setup:** four candidates, `predict_decode` at $b\in\{1,8,32\}$ and $L\in\{8\text{K},32\text{K},128\text{K}\}$ on the 8× A100-80GB analytic row. **Metrics:** binding resource, predicted tokens/s/user, $b_{\max}$. **Plot (`fig_lm_bud_matrix`):** heatmap candidates × (b, L) coloured by predicted tokens/s/user with the binding resource as the cell label. **Expected outcome:** to be computed; no measurement, so no noise check beyond the stated input uncertainty (error bars from the min–max of the `chip_specs` values used). **T1:** n/a (T0).

#### Figures and tables

`tab_lm_bud_scenario`, `fig_lm_bud_matrix`, `tab_lm_bud_experiments` (ranked list with uncertainty, sensitivity, cost), `tab_lm_bud_dossier` (the dossier skeleton).

#### Pitfalls and misconceptions

- Optimizing a metric the scenario does not bind (e.g. FLOPs when capacity binds).
- Treating a derived number as measured: the matrix labels provenance per cell.
- Ranking experiments by curiosity rather than by decision sensitivity.
- Forgetting tensor-parallel communication in the decode model (a listed risk, not modeled).
- Concluding "rejected" from a one-seed difference.

#### Summary (draft bullets)

- A design is a point in the chapter's design table plus a training plan; constraints are inequalities on `ConfigStats`.
- The 5.2 decode model with `config_stats` bytes names the binding resource per candidate.
- Experiments are ranked by uncertainty × decision sensitivity ÷ cost.
- A rejected design needs a pre-registered rule and evidence beyond noise.

#### Exercises

*Check.* 1. **Which binds?** For Qwen3-30B-A3B at 128K, 32 sessions on the analytic 8× A100-80GB row (a 40 GB node would not hold it): does capacity or bandwidth bind? *Sketch:* KV 12.9 GB × 32 = 412 GB + 61 GB weights = 473 GB ≤ 0.9 × 640 GB: feasible; the decode step reads that much per token step across the node, so bandwidth binds first (compute the step time from the table). 2. **Provenance.** Label each cell of a three-row matrix as derived, measured or disclosed. *Sketch:* `config_stats` outputs are derived; the report's 890 B/token is disclosed; a latency from 14.3 is measured.
*Derive.* 3. **Max batch.** Derive $b_{\max}(L)$ for the hybrid and show it is not proportional to $1/L$ at small $L$. *Sketch:* the fixed state $s$ makes $b_{\max}=\lfloor(C-W)/(kL+s)\rfloor$ flatten below $L=s/k$ ($n^\*$ of 14.6). 4. **Sensitivity.** Derive $\partial t/\partial L$ in the bandwidth-bound regime. *Sketch:* $b\,k/\beta$ per context token.
*Measure.* 5. **Replace a chip.** Re-run the matrix with the build-box row; predict which candidate's binding resource changes, then check. 6. **Prediction error.** Compare `predict_decode` with the T1 measurement of 14.6 for Qwen3.5-2B and report the ratio with the repetition range.
*Open.* 7. **Your dossier.** Complete the Assignment's dossier for a scenario of your own, with one measured baseline and one rejected design.

#### Further reading

[Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook); [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook); [nanochat](https://github.com/karpathy/nanochat) (HTTP 200; an end-to-end budgeted training stack); Stanford [CS336](https://cs336.stanford.edu/); [How to Scale Your Model](https://jax-ml.github.io/scaling-book/); the Assignment's sources in 14.1–14.7.

#### Risks and open decisions

| Risk | Fallback |
|---|---|
| 5.2's `ServingModel` signature differs from the assumed `(cfg, chip, ..., kv_bytes_override=...)` | thin adapter in `predict_decode`; the formula of Derivation 1 is the oracle |
| GPU-hour prices (5.6) unavailable | cost column in GPU-hours only |
| 14.9 is optional but hosts the capstone | the Assignment is self-contained (see its note) |
