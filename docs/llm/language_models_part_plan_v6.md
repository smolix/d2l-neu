# Language Models Part: Revised Plan (v6)

**Replacement for the current Language Models part of Dive into Deep Learning. Revised 5 November 2026.**

**What changed from v5.**

- 78 sections plus 4 Further Reading (FR) notebooks, 82 units, from 90; no headroom language. PyTorch only.
- Reader floor one 16 GB GPU (two scheduler slots, 15 GiB): judge Qwen3.5-4B in bf16, QLoRA headline Qwen3-8B in NF4.
- The Tools appendix stays and is linked, never absorbed; it owns usage, pinning, licences and buying.
- Models: the Qwen3 dense ladder, Qwen3.5 as the current small generation, Gemma 4 E2B as the one second family; other names in Chapter 14 FR.
- Nine-line entries with 1 to 3 citation keys, one Resources list per chapter, FR notebooks in Chapters 2, 8, 11, 14; a third of v5's "New" bullets restated Advanced material and are now Restate lines.
- Label fixes: `tab_rl_notation` (owner `sec_mdp`), not `tab_rl_notation_contract`; SGNS exercise in `sec_mdl-mi-estimation`; perplexity across tokenizers in `sec_mdl-coding-compression`; `sec_scaling-laws` trains on `d2l.WikiText103`. Book-side moves: see "Changes outside this part".
- Merges 4.2+4.3, 5.3+5.4, 5.5+5.6, 6.1+6.2, 6.3+6.5, 7.4+7.5, 7.7+7.8, 8.3+8.4, 9.6+9.7, 11.2+11.4, 11.3+11.6; 13.5, 14.8, 14.9 demoted; 8.7 split into 8.6 Reward Overoptimization and 8.7 Safety Evaluation and Red-Teaming; 1.7 Models behind an API new.
- Course additions: 8.7 safety, 2.2 token classification, 1.5 reproduction protocol, 12.1 Triton kernel, A5, A8, A9, a project scaffold, per-lecture paths.

**Inputs and checks.** The Codex plan (the bibliography), its review and `blueprints/README.md`; every book label here was checked against `chapter_*/` on 5 November 2026 and resolves.

## Goal

Building, scaling, aligning, extending and deploying language models, and the architectures of released models, starting where Advanced stops: a reader can build a GPT with RoPE, GQA, RMSNorm and MoE layers (`sec_gpt`, `sec_kv-cache`, `sec_moe`); train it under WSD (`sec_adamw`, `sec_muon`, `sec_scheduler`); reason about rooflines, KV caches and data parallelism (`sec_perf_model`, `sec_hardware`, `sec_multi_gpu`); derive policy gradients, PPO, DPO and the KL-regularized optimum (`chap_reinforcement_learning`, `chap_deep_rl`).

## Size

14 chapters, 78 sections, 4 FR notebooks: 82 units. A section runs 2,000 to 3,500 prose words; an FR notebook 1,000 to 2,000, no training run. Seven arcs: I Using and measuring (1, 2); II Building a base model (3, 4); III Scaling out (5, 6); IV Post-training (7, 8, 9); V Acting (10); VI Deploying (11, 12, 13); VII Architecture (14).

## Teaching approach

Advanced undergraduate or first-year graduate level, after Basics and Advanced, with Mathematics cited by label. Per the writing guide a section has an opening of 150 to 300 words from the reader's starting point and one concrete example, ending on the open question; two to five `##` subsections on one mechanism; a closing, a summary, exercises. One idea per section; one experiment per notebook, a single seeded run checked against a reference or prediction, any second measurement a cell on a committed artifact or an exercise. Plan entries:

```text
### N.M Noun-Phrase Title · T<k> · [KEEP | REVISE | MERGE(v5 x.y+x.z) | NEW] · [Codex u.v]
**Idea:** one jargon-free sentence.
**Opening:** the reader's starting point, one concrete example, the closing open question.
**Teaches:** 3 to 4 bullets; the mechanism in full, relatives only with their one difference.
**Derivation:** the one derivation with its smallest numeric case, or "none".
**Notebook:** the one question, model and data, the check, the noise-safe claim, reused `d2l` pieces.
**Figures:** 1 to 3 ids with a phrase each; "(plot)" marks a computed plot.
**Restate / forward:** labels restated in one sentence each; at most one forward reference.
**Exercises:** 5 to 8 (up to 10 over 3,000 words); the 2 or 3 load-bearing ones.
**Cite:** 1 to 3 citation keys.
```

A chapter block replaces objectives, assignment lines and teaching sources: label; one-sentence hook; route as one question per section; scope left out; at most 8 annotated Resources; FR topics or "none". An FR entry uses the nine lines with Notebook = "none (reading)". Objectives, assignments and course paths live only here; tier labels never appear in a heading.

### Course paths

One line per lecture, "|"-separated; a lecture is one or two sections; T3 and T4 sections appear only as read-the-logs lectures on committed artifacts, marked "(logs)".

- **Large Language Models** (26 lectures, at most T2): 1.1, 1.2 | 1.3, 1.4 | 1.5, 1.6 | 1.7, 2.3 | 2.4, 2.5 | 3.2, 3.3 | 3.6, 4.1 | 4.2 | 4.6, 5.2 | 7.2, 7.4 | 7.5, 8.1 | 8.3 | 8.4 | 8.6, 8.7 | 9.2, 9.5 | 9.6 | 10.1, 10.3 | 10.5 | 11.1, 12.2 | 13.2 | 14.1, 14.2 | 14.6 | 4 project lectures.
- **Systems for Large Models** (26 lectures): 1.1, 1.4 | 4.1 | 4.3 | 5.1 | 5.2 | 5.3 (logs) | 5.4 (logs) | 6.1 (logs) | 6.2 (logs) | 6.3 | 6.4 (logs) | 6.5, 6.6 (logs) | 7.6 (logs) | 8.5 (logs) | 11.1 | 11.2 | 11.3, 11.4 | 12.1 | 12.2, 12.3 | 12.4 | 13.1, 13.2 | 13.3 | 13.4 (logs) | 14.3, 14.5 | 2 project lectures.
- **Post-Training and Alignment** (20 lectures): 1.1, 1.4 | 1.5 | 7.2, 7.3 | 7.4, 7.5 | 8.1 | 8.2 | 8.3 | 8.4 | 8.5 (logs), 8.6 | 8.7 | 9.1, 9.2 | 9.3 (logs) | 9.4, 9.5 | 9.6 | 10.1, 10.2 | 10.3 | 10.4 (logs) | 10.5 | 2 project lectures.
- **NLP with Pretrained Models** (16 lectures): 1.1, 1.2 | 1.3 | 1.4 | 1.5 | 2.1 | 2.2 | 2.2 lab | 2.3 | 2.4 | 2.5 | 2.5 lab | 7.1 | 7.2 | 7.3 | 7.4 | project.
- **Model Architecture** (seminar, 16 meetings): 1.1, 1.2 | 1.6 | 4.2 | 5.2 | 14.1 | 14.2 | 14.3 | 14.4 | 14.5 | 14.6 | 14.7 | Chapter 14 FR | 4 paper meetings.

### Assignments

Committed numeric targets with a noise floor in the grader; every assignment has a T1 path.

- **A1** (Ch. 1, T1) run, port, measure: bf16 logit tolerance; published scores within interval.
- **A2** (Ch. 2, T1) RAG over the book: Recall@10 and answer accuracy above floors with paired intervals.
- **A3** (Ch. 3, T0 to T1) data pipeline: deduplication F1 on labelled pairs; manifest reproducible from hashes.
- **A4** (Ch. 4, T1; T2 full) pretrain with packing, masks, WSD: validation loss within a band at fixed tokens.
- **A5** (Ch. 4, T0, new) scaling-law fit on 4.2's committed ladder logs: exponents within tolerance; a lifetime-cost prediction.
- **A6** (Ch. 5, 6, T1; T3 full) performance model and TP/PP blocks: predictions within 15%; outputs and gradients match one GPU.
- **A7** (Ch. 7, 8, T1 to T2) SFT, DPO, GRPO on Qwen3-0.6B: win rate and verifiable accuracy above floors.
- **A8** (Ch. 8, T1, new, in 8.7) safety and red-teaming: refusal, over-refusal, attack-success and extraction rates on the published sets.
- **A9** (Ch. 10, T1, new) agent harness: pass^k on a fixed task set above a floor at fixed tokens.
- **A10** (Ch. 12, 13, T1) inference engine: throughput and P99 targets on a trace; batch-invariant outputs.
- **Mini** (Ch. 9, 11, T1) and **Capstone** (Chapter 14 FR, design under a budget), graded on checks and a report.
- **Project scaffold** (the 11-711 and CS288 pattern): proposal naming claim and baseline; baseline reproduction with error bars (1.5's protocol); ablations fixed before the runs; report at the precision that survives re-execution; T1 by default.

## Threads from the rest of the book

- `sec_gpt` (`subsec_gpt-loading`), `sec_transformer-block`, `sec_read_write` → 1.1, 1.2, 4.2, 12.1, 14.6.
- `sec_text-sequence` (`subsec_fertility-digits-glitch`), `sec_mdl-coding-compression` → 1.4, 3.6.
- `sec_what-attention-computes` → 1.3, 1.6. `sec_transformer` → 2.2, 2.3.
- `sec_attention-at-scale`, `sec_kv-cache` (`subsec_kv-cache-low-rank`, `subsec_kv-cache-batching`), `sec_attention-scoring-functions` → 4.1, 4.5, 6.2, 12.4, 13.3, 14.3 to 14.5.
- `sec_positional-information` → 4.5, 14.4. `sec_moe` → 6.4, 13.4, 14.2.
- `sec_adamw`, `sec_muon`, `sec_scheduler` (`subsec_scheduler-branching`), `sec_scaling`, `sec_practice` → 4.2, 4.6.
- `sec_scaling-laws` → 4.6, 8.4, 9.5, 14.1, 14.2.
- `sec_perf_model`, `sec_hardware` (`subsec_hw-two-workloads`, `tab_gpu_specs`, `eq_hw_decode_bound`), `sec_compilation` (`subsec_comp-overhead`) → Ch. 5, 11.1, 12.1.
- `sec_memory_precision`, `sec_numerics` → 4.3, 5.2, 7.5, 11.1.
- `sec_multi_gpu`, `sec_multi_gpu_concise` (`tab_mgp-zero`, `ex_mgp-tensor`) → Ch. 5, 6.
- `sec_policygradient`, `sec_baselines`, `sec_ppo`, `sec_regularized`, `sec_offline` → Ch. 8.
- `sec_mdp` (`tab_rl_notation`), `sec_rl_sequences` (`eq_dpo_reward`, `ex_rl-sequences-kl-loss`) → Ch. 8 to 10.
- `sec_imitation` (`subsec_il_sft`, `subsec_il_dagger`) → 7.2, 9.2, 10.3. `sec_custom_layer` → 11.4.
- `chap_modern_rnn` (`sec_hybrids`, `sec_deltanet`) → 14.6. `sec_transformer-block`, `sec_init_param` → 4.2, 14.7.
- `sec_seq2seq` → 1.4.
- `sec_mdl-svd-low-rank` → 2.1, 7.4, 14.3. `subsec_mdl-pmi`, `sec_mdl-infonce`, `sec_mdl-mi-estimation` → 2.1, 2.3.
- `sec_mdl-divergences-distances`, `sec_mdl-information_theory` → 8.2, 9.1, 9.2. `sec_mdl-constrained-optimization-duality` → 8.3.
- `sec_mdl-statistics` (`subsec_mdl-paired-comparisons`, `ex_mdl-statistics-clustered`) → 1.5, 2.5, 8.7, 10.5.
- `sec_mdl-numerical-stability-conditioning` → 11.3; `sec_mdl-concentration-generalization` → 11.2; `sec_mdl-bayesian-computation` → 8.5.

## Contract with the rest of the book

Topics Advanced defers to this part, and retiring labels surviving parts cite; v5 rows without a real citation, or whose owner stays, are gone.

| Promised or cited in | Promise | Delivered in |
|---|---|---|
| `chap_performance` intro, `sec_multi_gpu`, `sec_multi_gpu_concise`, moe.md:116 | Multi-node training; TP, PP, CP, EP | Ch. 5, 6 |
| `sec_kv-cache`, `sec_hardware` | Continuous batching, paged KV, speculation | 12.2, 13.2, 13.3; usage in `sec_model_serving` |
| `chap_performance`, `sec_memory_precision` | Quantization for inference | 7.5, Ch. 11 |
| `sec_fast_transformer`, `sec_hardware`, `sec_memory_precision` | fp8 and fp4 training | 4.3 |
| attention Further Reading (four sites) | Split-KV decode kernels; long-context training; patching, attribution graphs | 12.1; 4.5; 1.6 |
| `chap_reinforcement_learning`, `chap_deep_rl`, `sec_rl_sequences` | Post-training, reward models and multi-turn agents at LM scale (DPO already derived there; 8.3 restates) | Ch. 8, 10.4 |
| `sec_mdp` (`tab_rl_notation`) | Notation used verbatim and extended | Ch. 8 to 10 |
| `sec_scaling-laws` (three sites) | Data curation; budgets beyond the optimum; pretraining at scale | Ch. 3; 4.6; 4.2 |
| transformers/index.md:43, encoders-decoders.md:1262 | Fine-tuning | Ch. 7 |
| `sec_transformer`, `sec_scaling-laws`, preface → `chap_nlp_pretrain`, `chap_nlp_app` | Pretrained encoders; pretraining at scale | `chap_lm_encoders`, `chap_lm_pretraining` (encoders-decoders.md:361 reworded) |
| `sec_mdl-mi-estimation` → `subsec_negative-sampling`, `sec_word2vec_data` | SGNS loss; 0.75-power noise | 2.1 keeps both labels |
| transformers/further-reading.md:257; comp-perf/further-reading.md:202 | Vision encoders; exported graphs for serving | Image Models part; 13.1 |

## What this part does not re-teach

**Rule:** restate a fact owned elsewhere in one sentence with its label; never re-derive or re-implement it. v5's owner table stands with these additions and corrections:

- `sec_attention-scoring-functions`, `sec_attention-at-scale`: document masks, FlexAttention masks. `sec_positional-information`: YaRN, NTK-aware scaling, position interpolation. `sec_practice`, `sec_muon`: z-loss, loss-spike rewind, QK-clip.
- `sec_rl_sequences`, `sec_baselines`: DPO derivation, GRPO variants (DAPO, GSPO, Dr. GRPO), k3 as forward-KL gradient. `sec_what-attention-computes`, `subsec_kv-cache-sinks`: learned sink logits, gated attention, massive-activation sinks.
- Transformers Further Reading: the speculative acceptance rule. `sec_kv-cache`: per-channel key and per-token value quantization. `sec_scaling-laws` (`tab_modern-recipe`): the configuration table, canonical copy in 14.1.
- `sec_mdl-coding-compression`: arithmetic coding, perplexity across tokenizers. `subsec_mdl-distillation`: soft-target distillation. `sec_seq2seq`: chrF, BLEU, the fixed-vector bottleneck. `sec_repro`: deterministic arithmetic and seeds.
- Corrections: the SGNS exercise is owned by `sec_mdl-mi-estimation`, the PiSSA idea by `chapter_mdl-linear-algebra/further-reading.md`. Inside the part: 7.5 owns the quantizers (absmax, blockwise, NF4), 4.3 block-scaling error, 1.4 pass@k, 1.5 clustered standard errors.

## Compute tiers

| Tier | Hardware | Budget per notebook |
|---|---|---|
| T0 | CPU, any build host | Minutes |
| T1 | One 16 GB GPU: two scheduler slots, at most 15 GiB; existing scheduler | About 15 min |
| T2 | One A100 40 GB ("T2 pair": two, NVLink); new resource class | A few hours |
| T3 | One 8x A100 40 GB node; new resource class, once per release | About 2 h (4.2's ladder: 2.5 h) |
| T4 | Two 8x A100 nodes; cluster launch, committed logs | About 1 h |

**Rules for T2 and above:** a reduced configuration runs at T1 with the same code; analysis cells use committed artifacts via `d2l.DATA_HUB`. **Hardware limits:** FP8 is native on Ada and Hopper, not A100 (4.3 simulates otherwise); NVFP4 and MXFP4 are simulated on Ada and A100, native on Blackwell. **Build system:** T2 to T4 resource classes in `runtime_env.notebook_resource`; a `d2l.launch` helper for `torchrun --nnodes` or SLURM jobs; T3 and T4 re-executed per release.

## Running examples

Three shared artifacts, called by name: **the Qwen3-0.6B checkpoint in the book's own code** (1.2; `Qwen3GPT`, a `d2l.GPT` subclass with QK-norm and an explicit head dimension; reused in 4.2, 12.1, 14.6); **the reference model** (4.2; a 138M decoder on 3.15B FineWeb-Edu tokens with a three-size ladder at two token budgets, committed; fitted in 4.6, reused by Chapters 6, 11, 12, 14); **the book corpus** (2.4; used by 2.3 to 2.5, Chapter 10, 4.4).

Model roles: book's own code, Qwen3-0.6B/1.7B/4B/8B; current small generation and judge, Qwen3.5-0.8B/2B/4B (judge 4B); second family, Gemma 4 E2B; effort levels and MXFP4, gpt-oss-20b; open trajectory, Olmo 3 7B stage checkpoints; NoPE, SmolLM3-3B; MoE, Qwen3-30B-A3B and Qwen3.6-35B-A3B; node-scale fine-tuning, Qwen3.6-27B (fallback Qwen3-32B); encoders, ModernBERT-base, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B; SAE suites, Qwen-Scope and Gemma Scope 2 (exercise); API flagships, Qwen3.8-2.4T-A95B, DeepSeek V4-Flash, gpt-oss-120b.

## Conventions

- **PyTorch only**, a deliberate exception to Advanced's PyTorch+JAX policy; PyTorch-only symbols from Advanced (`generate_cached`, FlexAttention, `d2l.torchrun`, `fully_shard`) are therefore unproblematic. Reader floor, model families and the Tools boundary (`sec_model_serving`, `sec_training_systems`, `sec_software_ecosystem`, `sec_hardware_buyers`, `sec_cloud_instances` stay, linked) are as in "What changed".
- **Experiments:** one per notebook, single seeded runs, no conclusion from a difference within noise. **Technical reports** in depth only in Chapter 14. **Private authoring sources** (ETH slides) stay out of bibliographies. **Cross-references** use labels only (`chap_lm_<topic>`, `sec_lm_<topic>`); book text never names a chapter or section number.

## State of the art: what is covered and where

Dated 5 November 2026; Chapter 14's data file and FR notebook carry freshness, elsewhere the origin plus one current exemplar. Graded reasoning effort 1.1, 9.6; SAE suites with transcoders 1.6; current corpora Ch. 3; optimizer reality check and committed ladder 4.2; NVFP4 pretraining in production 4.3, 11.4; Olmo 3 stage checkpoints 4.4; rack-scale domains 5.4; wide EP and disaggregation 6.4, 13.4; RLVR and RL scaling 8.4; asynchronous RL and its 2026 corrections 8.5; overoptimization and safety 8.6, 8.7; on-policy distillation 9.2; adaptive test-time compute 9.6; MCP governance, Agent Skills, 2026 agent benchmarks Ch. 10; rotations and native low-bit formats 11.2, 11.4; Triton decode kernels, 2026 drafters, KV compression 12.1, 12.3, 12.4; DeepSeek V4 14.1, 14.2, 14.5, 14.7; Gemma 4 local-global design 14.4; hybrids 14.6; diffusion LMs, byte patching, test-time training Ch. 14 FR.

## References policy

- **Per section:** 1 to 3 citation keys; no section-end Further Reading. **Per chapter:** one Resources list of at most 8 annotated items; FR notebooks in Chapters 2, 8, 11, 14 say what each work adds. Benchmark and model names live in per-edition data files.
- **Bibliography:** the Codex plan is the source of record. **Verification:** v5 sources checked on 28 October 2026; v6 additions verified on 5 November 2026 on their primary pages; "verify" marks what a writer must confirm. **Cited by title:** BM25 (Robertson and Zaragoza), Levy and Goldberg 2014 (`Levy.Goldberg.2014`), Cormack et al. 2009, the SIGCOMM 2024 network papers, Young 1974 and Daly 2006.

## Chapter overview

| # | Chapter | Sections | Tiers |
|---|---|---:|---|
| 1 | Working with Pretrained Language Models | 7 | T0 to T1 |
| 2 | Encoders, Embeddings and Retrieval | 5 | T1 |
| 3 | Data for Language Models | 6 | T0 to T2 |
| 4 | Pretraining | 6 | T0 to T2 |
| 5 | Hardware for Training and Inference | 4 | T0 to T4 |
| 6 | Parallelism Beyond One Node | 6 | T3 to T4 |
| 7 | Fine-Tuning | 7 | T1 to T3 |
| 8 | Preference Optimization and Reinforcement Learning | 7 | T1 to T3 |
| 9 | Distillation and Reasoning | 6 | T1 to T3 |
| 10 | Agents and Tool Use | 5 | T0 to T3 |
| 11 | Model Compression | 4 | T1 |
| 12 | Efficient Inference | 4 | T1 to T2 |
| 13 | Serving Systems | 4 | T1 to T3 |
| 14 | Advanced Architectures | 7 | T0 to T3 |
| | **Total** | **78** | plus 4 FR notebooks |

Sections per tier (heaviest tier, FR excluded): T0 7, T1 39, T2 18, T3 11, T4 3.
## Chapter 1 Working with Pretrained Language Models

**Label:** `chap_lm_pretrained` · 7 sections · T0 to T1. **Purpose:** checkpoints as instruments; the part's scoring tools.
**Hook:** The reader has trained the GPT of `sec_gpt` and seen released models answer what it cannot; here those models are loaded, ported, scored and opened.
**Route:** What is in a checkpoint (1.1)? Do its tensors run the same in the book's code (1.2)? How much does the prompt decide (1.3)? Can scoring conventions disagree (1.4)? When does a difference survive a paired test (1.5)? How is a claim about computation tested (1.6)? What can an API measure (1.7)?
**Scope:** No fine-tuning (Chapter 7), serving (Chapter 13) or architecture beyond a configuration (Chapter 14).
**Resources:** CS336 evaluation lectures (1.4, 1.5); HF LLM course, chapter 2 (1.1); Defeating Nondeterminism in LLM Inference (1.4, 1.7); ARENA, TransformerLens (1.6); openai-python (1.7).
**Further Reading notebook:** none

### 1.1 Pretrained Models with Hugging Face · T1 · REVISE · [Codex opening unit]
**Idea:** A released model is a configuration file plus tensors; memory and speed follow from it.
**Opening:** Three lines load Qwen3.5-2B and it answers what the book's GPT cannot; closes on the book's code.
**Teaches:**
- Repository anatomy: card, configuration, sharded safetensors, tokenizer; revision as commit.
- Auto classes, `dtype`, `device_map`, `generate`; one chat template for base, instruct, thinking.
- Qwen3.5's configuration field by field; Qwen3.6 and Qwen3.8 in one sentence; `reasoning_effort`.
**Derivation:** Bytes from a configuration: weights, KV per token, state, workspace.
**Notebook:** Qwen3.5-2B and Gemma 4 E2B against derived bytes and `eq_hw_decode_bound`; a `reasoning_effort` sweep on gpt-oss-20b (tokens, latency). Claim: memory within 10%, decode within 2x.
**Figures:** `fig_lm_hub_repo`; `fig_lm_memory_vs_config` (plot).
**Restate / forward:** `subsec_hw-two-workloads` (decode reads every weight); `sec_software_ecosystem` (pinning, licences). Forward: 14.1.
**Exercises:** 6 to 8; derive Gemma 4 E2B's KV bytes per token; measure decode against batch.
**Cite:** Transformers 1910.03771; Qwen3 2505.09388; Qwen3.8 README.

### 1.2 A Released Checkpoint in the Book's Own Code · T1 · REVISE · [Codex opening unit, 2.1]
**Idea:** The same tensors run in a subclass of `d2l.GPT` and match the reference layer by layer.
**Opening:** Qwen3-0.6B's first block almost fits the book's constructor; closes on what a match exposes.
**Teaches:**
- `d2l.Qwen3Attention` (QK-norm, `head_dim`), `d2l.SwiGLU`, `d2l.Qwen3GPT`, `load_qwen3` (Decision 4).
- `generate_cached` gains `#@save`; fused projections and RoPE layout as the silent breakages.
- Why Qwen3 (last pure-transformer Qwen ladder), not Qwen3.5 (needs `chap_modern_rnn` code).
**Derivation:** none.
**Notebook:** Qwen3-0.6B in `d2l.Qwen3GPT` against Transformers on 32 prompts, per layer, then `d2l.generate`. Claim: bf16 agreement within a stated tolerance; identical greedy tokens for 64 steps.
**Figures:** `fig_lm_weight_map`; `fig_lm_layer_divergence` (plot).
**Restate / forward:** `subsec_gpt-loading` (GPT-2 weights mapped by name and shape); `sec_transformer-block` (QK-norm).
**Exercises:** 5 to 7; derive the parameter count from the index; measure a wrong RoPE layout's drift.
**Cite:** Qwen3 2505.09388; safetensors documentation.

### 1.3 Prompting and In-Context Learning · T1 · KEEP · [new]
**Idea:** The prompt is tokens, and few-shot accuracy depends on how the demonstrations are laid out.
**Opening:** Four solved examples lift accuracy and the rendered template shows why; closes on what a score measures.
**Teaches:**
- A chat template as token ids: roles, special tokens, generation prefix; `d2l.fewshot_prompt`.
- Zero- and few-shot formats; sensitivity to number, order and label correctness.
- In-context learning against a weight update.
**Derivation:** none.
**Notebook:** Qwen3-1.7B base and instruct on 200 items over demonstration count, four orders, shuffled labels. Claim: order alone moves accuracy by several points; shuffled labels cost less than none.
**Figures:** `fig_lm_chat_template`; `fig_lm_icl_sensitivity` (plot).
**Restate / forward:** `sec_what-attention-computes` (an induction head copies what followed an earlier copy).
**Exercises:** 5 to 7; measure the spread over 20 orders; derive why a calibration prompt shifts priors.
**Cite:** GPT-3 2005.14165; Rethinking the Role of Demonstrations 2202.12837.

### 1.4 Scoring Language Models · T1 · REVISE · [Codex 15.2]
**Idea:** A score is a convention, and two conventions can rank two models differently.
**Opening:** One item scored by letter log-probabilities and by generation gives two verdicts; closes on sample size.
**Teaches:**
- Log-likelihood scoring: cloze against letter formats, length normalization; determinism controls.
- Generative answers, extraction, execution checks; `d2l.score_choices`, `d2l.extract_answer`, `d2l.pass_at_k`.
- Bits per byte as the tokenizer-free score; chrF and BLEU as string-overlap conventions.
**Derivation:** The unbiased pass@k estimator, 1 - C(n-c,k)/C(n,k), and its variance; case n=4, c=1, k=2.
**Notebook:** Qwen3-0.6B and Qwen3.5-0.8B on 500 items under cloze, letter and generative scoring. Claim: one convention flips the ranking by more than the paired interval.
**Figures:** `fig_lm_scoring_conventions`; `fig_lm_ranking_flip` (plot).
**Restate / forward:** `subsec_bits-per-byte`, `sec_mdl-coding-compression` (bits per byte); `sec_seq2seq` (chrF, BLEU); `subsec_deterministic_arithmetic`.
**Exercises:** 6 to 8; derive the n at which the pass@k variance halves; measure length normalization.
**Cite:** Codex 2107.03374; Lessons from the Trenches 2405.14782.

### 1.5 Evaluation Suites, Judges and Error Bars · T1 · REVISE · [Codex 15.3, 15.5, 15.6]
**Idea:** A benchmark has a lifecycle, and a difference counts only if it survives a paired test.
**Opening:** A model gains two points on a suite; closes on writing a result another lab can check.
**Teaches:**
- The lifecycle (introduced, climbed, saturated, replaced), one dated example each; names in a data file.
- Paired comparison, clustered errors (`d2l.paired_bootstrap`, `d2l.EvalSuite`); judge biases in both orders.
- Protocol `subsec_lm_reproduce_protocol`: prompts, seeds, samples, interval, detectable effect.
**Derivation:** Clustered standard error for G clusters of m items with correlation rho; case G=2, m=2.
**Notebook:** Qwen3.5-4B judges two Qwen3 checkpoints on 300 pairs in both orders; paired bootstrap; biases as rates. Claim: one headline interval includes zero.
**Figures:** `fig_lm_benchmark_lifecycle`; `fig_lm_mde` (plot).
**Restate / forward:** `subsec_mdl-paired-comparisons` (pairing removes item difficulty); `ex_mdl-statistics-clustered` (cluster bootstrap).
**Exercises:** 6 to 8; derive the design effect 1+(m-1)rho; measure judge agreement with 200 human labels.
**Cite:** Adding Error Bars to Evals 2411.00640; Judging LLM-as-a-Judge 2306.05685.

### 1.6 Interpretability · T1 · REVISE · [new]
**Idea:** Claims about what a model computes are tested by intervening on activations, not by reading them.
**Opening:** The answer to a recall prompt rises through Qwen3-0.6B's layers; closes on what a pushed feature costs.
**Teaches:**
- The residual stream read by the logit and tuned lens; probes with a control task.
- Activation patching as a causal effect; attribution patching as its first-order form.
- SAEs, Qwen-Scope as the instrument; one steering vector; chain-of-thought monitorability.
**Derivation:** Patching as a causal effect and its gradient form; the SAE objective (L1, bounded decoder norms); top-k.
**Notebook:** Patching on Qwen3-0.6B localizes a recall behaviour to layer and position; lenses, probe, small SAE against Qwen-Scope, steering vector, one cell each. Claim: a few cells carry most of the effect.
**Figures:** `fig_lm_patching`; `fig_lm_lens_rank` (plot); `fig_lm_sae_frontier` (plot).
**Restate / forward:** `sec_what-attention-computes` (induction heads by ablation); `sec_mdl-svd-low-rank` (a probe is rank one). Forward: Chapter 9.
**Exercises:** 6 to 8; derive attribution patching; measure steering against perplexity.
**Cite:** Tuned Lens 2303.08112; Scaling SAEs 2406.04093; Circuit Tracing 2025; ADAG 2604.07615.

### 1.7 Models behind an API · T0 · NEW · [blueprint Decision 10]
**Idea:** A hosted model is a service with a protocol, a price and limits; what it withholds is why the part uses open weights.
**Opening:** Qwen3.8-2.4T-A95B exceeds every tier yet answers in a second for a printed cost; closes on what survives.
**Teaches:**
- The chat-completions protocol; `usage` as ground truth for tokens and cost; a field probe matrix.
- `d2l.ChatAPI`: replay cache, full-jitter backoff, cost cap; router, local server, closed provider.
- Model ids drift, temperature 0 is not determinism; lost: prompt log-probabilities, weights, sampler.
**Derivation:** Little's law for a closed loop, X = min(c/W0, Xmax), knee c* = Xmax W0; expected full-jitter sleep.
**Notebook:** Qwen3.8-2.4T-A95B (licence checked) and gpt-oss-120b via the router against Qwen3.5-4B behind `transformers serve`, 100 GSM8K items; replicas, concurrency sweep, cost; replayed at T0. Claim: X W = c within a few percent.
**Figures:** `fig_lm_api_concurrency` (plot); `fig_lm_api_lost`.
**Restate / forward:** `sec_decoding` (request fields are sampler parameters); `sec_repro` (seeds). Forward: 10.1 (`tools`).
**Exercises:** 5 to 7; derive W(c) beyond the knee; measure prompt-cache hits on a shared prefix.
**Cite:** Model Equality Testing 2410.20247; Inference Providers docs; AWS, Backoff and Jitter.

## Chapter 2 Encoders, Embeddings and Retrieval

**Label:** `chap_lm_encoders` · 5 sections + FR · T1. **Purpose:** encoders, embeddings and retrieval; replaces the retiring NLP chapters.
**Hook:** The reader has a decoder that answers questions but no way to find, among a million passages, the one that holds the answer.
**Route:** What statistic does a word vector factorize (2.1)? What does an encoder give per token (2.2)? Why do pooled states retrieve poorly until contrastive training (2.3)? What does an index trade for speed (2.4)? Where does a RAG system fail (2.5)?
**Scope:** No encoder pretraining at scale (`sec_transformer`), no agent loops (Chapter 10), no multimodal retrieval (Image Models part).
**Resources:** CS224N word-vector and pretraining lectures (2.1, 2.2); Sentence Transformers docs (2.3); ModernBERT blog (2.2); Faiss wiki, Pyserini (2.4); Jason Liu, Systematically Improving Your RAG (2.5).
**Further Reading notebook:** `sec_lm_encoders_fr`

### 2.1 Word Embeddings · T1 · REVISE · [replaces retired word2vec and GloVe sections]
**Idea:** A word vector is a factorization of co-occurrence statistics.
**Opening:** Two words that share contexts show what the rows of `sec_gpt`'s table mean; closes on two meanings.
**Teaches:**
- SGNS, 0.75-power noise, subsampling; labels `subsec_negative-sampling` and `sec_word2vec_data` kept.
- Shifted PMI, stated (derivation in `sec_mdl-mi-estimation`'s exercise); SVD of PPMI; GloVe in one sentence.
- Analogies, bias, tied tables; closing paragraph `subsec_lm_categorical_embeddings` (any categorical variable).
**Derivation:** none (stated; derived in `sec_mdl-mi-estimation`).
**Notebook:** SGNS on a word-level `d2l.WikiText103` slice against SVD of PPMI on a similarity benchmark; neighbours in Qwen3-0.6B's table. Claim: both correlate with human judgments alike.
**Figures:** `fig_lm_sgns_pmi`.
**Restate / forward:** `subsec_mdl-pmi` (PMI exceeds independence); `sec_mdl-svd-low-rank` (truncated SVD is optimal).
**Exercises:** 5 to 7; derive subsampling's effect on rare words; measure analogy accuracy against dimension.
**Cite:** word2vec 1301.3781; Levy and Goldberg 2014; GloVe D14-1162.

### 2.2 Pretrained Encoders · T1 · REVISE · [replaces retired BERT, sentiment and NLI sections]
**Idea:** A masked-token encoder gives one vector per token, and a task head is fine-tuned per task.
**Opening:** Ten thousand sentences an hour cost more to prompt than to label; closes on one vector for every task.
**Teaches:**
- The encoder and `subsec_mlm-loss-by-position` restated; ModernBERT worked in full; RoBERTa, ELECTRA, T5 as relatives.
- Three heads: sentence classification, sentence pairs (NLI), token classification (named-entity tagging).
- When an encoder beats prompting (cost, latency, labels); naive Bayes baseline; text CNN.
**Derivation:** none.
**Notebook:** ModernBERT-base with a sentence head (NLI, sentiment) and a token head (CoNLL-style tagging) against few-shot Qwen3-1.7B. Claim: the encoder matches few-shot at a tenth of the cost; the token head reaches a stated F1.
**Figures:** `fig_lm_encoder_heads`; `fig_lm_encoder_vs_prompting` (plot).
**Restate / forward:** `sec_transformer` (a bidirectional encoder predicts masked tokens); `sec_mdl-naive_bayes` (classical baseline).
**Exercises:** 5 to 7; measure memory against batch size (cloud-instances.md:212); derive the masked-position loss.
**Cite:** BERT 1810.04805; ModernBERT 2412.13663; CoNLL-2003 W03-0419.

### 2.3 Text Embeddings from Contrastive Training · T1 · REVISE · [Codex 2.7]
**Idea:** Pooled encoder states retrieve poorly until contrastive training spreads them on the sphere.
**Opening:** Mean-pooled 2.2 vectors retrieve near-random passages, with mean cosine near one; closes on scale.
**Teaches:**
- Anisotropy and pooling (CLS, mean, last token); normalization.
- In-batch negatives with temperature, hard negatives and false negatives; weak pairs then supervised pairs.
- Decoder-derived embedders (Qwen3-Embedding), query instructions; Matryoshka, binary quantization; MTEB.
**Derivation:** The large-batch limit of InfoNCE as alignment plus uniformity on the sphere; case four points on a circle.
**Notebook:** Mean-pooled ModernBERT on the book corpus, then in-batch and hard-negative training, then Qwen3-Embedding-0.6B; Recall@10, nDCG, a Matryoshka sweep. Claim: training lifts Recall@10 by tens of points.
**Figures:** `fig_lm_anisotropy`; `fig_lm_matryoshka` (plot).
**Restate / forward:** `sec_mdl-infonce` (B negatives bound mutual information by log B). Re-check the Hub for a 2026 embedder.
**Exercises:** 5 to 7; derive the uniformity term; measure recall against temperature.
**Cite:** SimCSE 2104.08821; Alignment and Uniformity 2005.10242; Qwen3 Embedding 2506.05176.

### 2.4 Sparse, Dense and Approximate Retrieval · T1 · REVISE · [new]
**Idea:** An index trades recall for latency and memory, and BM25 is the floor a dense index must beat.
**Opening:** One query scanned against a million passages is too slow to serve; closes on what the model reads.
**Teaches:**
- Inverted indexes and BM25 (term saturation, length normalization) as the baseline.
- IVF with k-means, product quantization, HNSW; recall against latency and memory as the one trade.
- Reciprocal-rank fusion; reranking (Qwen3-Reranker-0.6B); Recall@k, MRR, nDCG; SPLADE and ColBERT to FR.
**Derivation:** Asymmetric distance in product quantization, memory per vector and its error.
**Notebook:** BM25, flat, IVF-PQ and HNSW over the chunked book (shared artifact), fusion, then the reranker. Claim: BM25 within a band of dense on exact terms; HNSW reaches flat recall far faster.
**Figures:** `fig_lm_ivf_pq`; `fig_lm_index_tradeoff` (plot).
**Restate / forward:** `sec_mdl-svd-low-rank` (a product quantizer is a block low-rank code). The index serves 2.5 and Chapter 10.
**Exercises:** 5 to 7; derive PQ memory per vector; measure HNSW recall against search width.
**Cite:** Product Quantization 10.1109/TPAMI.2010.57; HNSW 1603.09320; Cormack et al., Reciprocal Rank Fusion.

### 2.5 Retrieval-Augmented Generation · T1 · REVISE · [new]
**Idea:** A RAG system is a chain of stages, and a failure is located by measuring each stage.
**Opening:** The 2.4 retriever wired to Qwen3-4B answers wrongly with the gold passage retrieved; closes on long context.
**Teaches:**
- Stages: chunking, query rewriting, reranking, context assembly, grounded generation with citations.
- Per-stage metrics with intervals; a failure taxonomy (not retrieved, ranked low, ignored, unsupported).
- RAG-Sequence and RAG-Token as marginalizations over documents.
**Derivation:** The two marginalizations, the top-k approximation, the retriever gradient.
**Notebook:** On the committed evaluation set (240 dev, 240 test; Decision 13): which stage change moves accuracy outside its paired interval? Three parts; a 50-case taxonomy. Claim: one stage carries most of the gain.
**Figures:** `fig_lm_rag_stages`; `fig_lm_rag_failures` (plot).
**Restate / forward:** `subsec_mdl-paired-comparisons` (paired intervals per stage). Forward: 10.4 (retrieval inside an agent loop).
**Exercises:** 6 to 8; derive the top-k approximation error; measure position effects.
**Cite:** Retrieval-Augmented Generation 2005.11401; Lost in the Middle 2307.03172; RAGAS 2309.15217.

### FR Further Reading: Encoders and Retrieval · T0 · NEW · [new]
**Idea:** The retrieval methods that fail the teachable-core test, each with the clause saying what it adds.
**Opening:** none (reading).
**Teaches:** Learned sparse retrieval (SPLADE: expansion in vocabulary space); late interaction (ColBERT: per-token matching at index time); late chunking and contextual headers; training for RAG (REALM, RAFT, Self-RAG: what each learns); long context against retrieval.
**Derivation:** none.
**Notebook:** none (reading); 1,000 to 2,000 words.
**Figures:** none.
**Restate / forward:** none.
**Exercises:** none.
**Cite:** SPLADE 2107.05720; ColBERTv2 2112.01488; Self-RAG 2310.11511.

## Chapter 3 Data for Language Models

**Label:** `chap_lm_data` · 6 sections · T0 to T2. **Purpose:** crawl to corpus: extraction, filtering, deduplication, mixing, synthesis, tokenization.
**Hook:** The reader has trained on a text file someone else prepared; released models start from billions of pages, and every step before the first batch changes what can be learned.
**Route:** What does extraction lose (3.1)? What does a filter remove, and who checks it (3.2)? How are near-duplicates found without pairwise comparison (3.3)? How much of each domain goes in (3.4)? When does rephrased text help (3.5)? How large should the vocabulary be (3.6)?
**Scope:** BPE (`sec_text-sequence`), n-gram statistics (`sec_language-model`) and distribution shift are not re-taught; instruction data is Chapter 7.
**Resources:** CS336 data lectures and assignment (3.1 to 3.4); FineWeb blog post and Datatrove (3.1, 3.2); minbpe (3.6); Smol Training Playbook, data chapters (3.4, 3.5).
**Further Reading notebook:** none

### 3.1 Document Extraction from Web Crawls · T0 · REVISE · [Codex 1.1, 1.4]
**Idea:** Extraction decides what the model can ever learn.
**Opening:** One crawl record holds menus, a banner and an article two extractors render differently; closes on quality.
**Teaches:**
- WARC and WET records; HTML extraction and how it fails on code, tables and mathematics.
- Language identification and Unicode normalization, and what each destroys.
- Personal data, provenance, licences, opt-outs; a manifest with source ids; FineWeb-2 multilingual.
**Derivation:** none.
**Notebook:** One Common Crawl shard as a pinned WARC byte range (never hosted, Decision 13) through two extractors; loss per document class; a manifest from hashes. Claim: disagreement concentrated in code and tables.
**Figures:** `fig_lm_crawl_pipeline`.
**Restate / forward:** `sec_text-sequence` (a tokenizer sees bytes, so normalization decides which bytes exist).
**Exercises:** 5 to 7; measure extraction loss on one page class; derive documents per language under identifier error.
**Cite:** FineWeb 2406.17557; Common Crawl documentation.

### 3.2 Quality Filtering · T1 · KEEP · [Codex 1.5]
**Idea:** A quality filter is a classifier trained on model-annotated labels, and its false removals must be audited.
**Opening:** A model rates a page low that a human would keep; closes on how what the filter keeps is biased.
**Teaches:**
- Heuristic rules and perplexity filters; a classifier on model labels as the FineWeb-Edu recipe.
- The threshold as a precision-recall choice on held-out human labels.
- Auditing removals by language, domain and dialect.
**Derivation:** none.
**Notebook:** The classifier trained on Qwen3.5-4B labels, applied to the 3.1 documents, audited against 200 human labels. Claim: removal rate differs by domain beyond its interval; precision stated with an interval.
**Figures:** `fig_lm_filter_audit` (plot).
**Restate / forward:** `sec_environment-and-distribution-shift` (a two-set classifier measures shift).
**Exercises:** 5 to 7; measure agreement between two labellers' classifiers; derive the cost of a false removal.
**Cite:** DCLM 2406.11794; Gopher 2112.11446.

### 3.3 Deduplication and Decontamination · T0 · KEEP · [Codex 1.5, 15.7]
**Idea:** Near-duplicates are found by hashing, and the hash threshold is a tunable S-curve.
**Opening:** The same licence text sits in a thousand documents, two differing by a date; closes on the test set.
**Teaches:**
- Exact hashing; MinHash signatures and LSH banding; suffix-array substring deduplication.
- Decontamination by n-gram overlap; what deduplication does to memorization.
**Derivation:** MinHash agreement equals Jaccard similarity; the S-curve 1-(1-s^r)^b and its threshold; case b=2, r=2.
**Notebook:** MinHash and LSH from scratch; b and r tuned on labelled near-duplicate pairs; planted benchmark items detected. Claim: the fitted S-curve predicts measured agreement; all planted items found.
**Figures:** `fig_lm_lsh_scurve` (plot); `fig_lm_minhash`.
**Restate / forward:** `sec_prob` (a MinHash value is a minimum over a random permutation).
**Exercises:** 5 to 7; derive the expected false positives of banding; measure F1 against shingle size.
**Cite:** Deduplicating Training Data 2107.06499; Broder 1997.

### 3.4 Data Mixtures · T2 · REVISE · [Codex 1.6]
**Idea:** Domain weights are hyperparameters, fitted on proxy runs.
**Opening:** Two small runs with different web, code and mathematics weights have losses that cross; closes on transfer.
**Teaches:**
- Domain weights and temperature sampling across languages; annealing on high-quality data at the end.
- Proxy-model regression from weights to per-domain loss (RegMix); DoReMi's one difference (excess loss).
- Dolma 3 as the open, documented mixture.
**Derivation:** none.
**Notebook:** Proxies at 20M parameters on web, code and mathematics mixtures; regression from weights to per-domain loss; one held-out mixture. Claim: predictions fall within the proxies' seed noise of the measured losses.
**Figures:** `fig_lm_mixture_regression` (plot).
**Restate / forward:** `sec_scaling-laws` (power-law loss makes small proxies informative).
**Exercises:** 5 to 7; derive the temperature for a target language share; measure weights against proxy size.
**Cite:** DoReMi 2305.10429; RegMix 2407.01492; Olmo 3 2512.13961.

### 3.5 Synthetic and Rephrased Data · T1 · REVISE · [Codex 1.7]
**Idea:** Rephrasing trades diversity for quality, and the trade is measurable at equal tokens.
**Opening:** A small model turns one messy page into a clean paragraph; closes on who may train on generated text.
**Teaches:**
- Rephrasing web text and textbook-style generation; quality against diversity as the one trade.
- Nemotron-CC-v2 and FinePhrase as current rephrased corpora each.
- Licences of generated data; the terms check of 1.7 before an API model generates training data.
**Derivation:** none.
**Notebook:** A slice rephrased with Qwen3.5-2B; small models on original, rephrased and mixed data at one budget; held-out web loss and a small suite. Claim: mixed beats both alone; rephrased-only loses on web loss.
**Figures:** `fig_lm_rephrase_tradeoff` (plot).
**Restate / forward:** `sec_environment-and-distribution-shift` (training on model outputs shifts the data).
**Exercises:** 5 to 7; measure distinct n-grams, original against rephrased; derive the rephrasing budget.
**Cite:** Rephrasing the Web 2401.16380; Nemotron-CC-v2 dataset card; FinePhrase 2604.13977.

### 3.6 Tokenizers at Scale · T0 · REVISE · [Codex 1.2, 1.3]
**Idea:** Vocabulary size is a compute decision, and the unigram model is the alternative to merges.
**Opening:** The BPE of `sec_text-sequence` splits a Hindi sentence into three times the tokens; closes on extension.
**Teaches:**
- Unigram tokenization with Viterbi segmentation; SentencePiece; the one difference from merges.
- Vocabulary size against embedding and softmax cost and against sequence length; the vocabulary fit.
- Extension and initialization of new rows; the reference tokenizer (byte BPE, 32,768, ids 0 to 31; Decision 1).
**Derivation:** none.
**Notebook:** BPE and unigram on a FineWeb-2 slice at three sizes; fertility and bits per character by language; extension with two initializations. Claim: tokenizer type matters less than size.
**Figures:** `fig_lm_fertility_by_language` (plot); `fig_lm_unigram_viterbi`.
**Restate / forward:** `subsec_fertility-digits-glitch` (fertility, digits); `sec_mdl-coding-compression` (bits per character).
**Exercises:** 5 to 7; derive the vocabulary size that balances cost and length; measure digit chunking.
**Cite:** SentencePiece D18-2012; Scaling Laws with Vocabulary 2407.13623.

## Chapter 4 Pretraining

**Label:** `chap_lm_pretraining` · 6 sections · T0 to T2. **Purpose:** run, read and extend a real pretraining run; the reference model and its ladder.
**Hook:** The reader has trained the GPT of `sec_gpt` for an afternoon; a real run lasts days, is interrupted, spikes, and is read from its logs.
**Route:** How does a data path stay exact across an interruption (4.1)? What do a run's logs say, and why do runs spike (4.2)? When can the arithmetic drop to eight or four bits (4.3)? What does a new domain cost (4.4)? How is a longer window trained and measured (4.5)? Where is the optimum once inference tokens count (4.6)?
**Scope:** Optimizers, schedules, batch size, muP, the Chinchilla fit, mixed precision and clipping are restated, not re-taught; multi-node execution is Chapter 6.
**Resources:** CS336 assignment 1 (4.1, 4.2); build-nanogpt, nanochat (4.1, 4.2); Smol Training Playbook, SmolLM3 blog (4.2, 4.5); OLMo code, Olmo 3 checkpoints (4.4, 4.6).
**Further Reading notebook:** none

### 4.1 The Pretraining Data Path · T1 · REVISE · [Codex 1.8, 5.1, 8.4]
**Idea:** A data path is exact and resumable, or the run is not reproducible.
**Opening:** A restarted run sees a batch the dead run never would have; two packed documents leak attention; closes on the run.
**Teaches:**
- Pre-tokenized sharded streams with a manifest; per-document positions inside a packed sequence.
- Token-weighted loss across unequal documents; resumable cursors that name the next batch.
- The book's `d2l.DataModule` and `d2l.BPETokenizer` stack as the main flow; HF `datasets` named once.
**Derivation:** none.
**Notebook:** Packed batches with document masks against unpacked, per-document logits compared; interrupt and resume, the next ten batches compared. Claim: logits equal within bf16 rounding; batches identical.
**Figures:** `fig_lm_packing_mask`; `fig_lm_data_cursor`.
**Restate / forward:** `sec_attention-scoring-functions` (document masks); `sec_attention-at-scale` (FlexAttention); `sec_repro` (seeds).
**Exercises:** 5 to 7; derive token-weighted against sequence-weighted gradients; measure the mask's cost.
**Cite:** FlexAttention blog (pytorch.org/blog/flexattention).

### 4.2 The Reference Pretraining Run · T2 · MERGE(v5 4.2+4.3) · [Codex 5.1, 5.9, 16.2]
**Idea:** A run is read through its logs, and the instabilities in them have a mechanism.
**Opening:** Last week's logs show a gradient-norm spike at hour six; closes on sizing for a served model.
**Teaches:**
- What to log: per-domain loss, update-to-weight ratio, gradient norm, MFU, downstream scores.
- Instabilities, one `##` subsection: logit growth and its cures (z-loss, QK-norm, QK-clip, rewind), restated.
- Optimizers as a fact: matrix optimizers at most about 1.4x over AdamW, about 1.1x at scale.
- Reference model: 138M on 3.15B FineWeb-Edu tokens (Decision 2) plus a three-size ladder at two budgets (T3).
**Derivation:** Attention-logit growth from the Q and K weight norms, and why clipping their product bounds it.
**Notebook:** The 138M model at the T2 schedule (1.05B tokens): a high-learning-rate arm spikes, a QK-clip arm (`d2l.qk_clip_`, Decision 19) does not. Claim: the clipped arm trains where the baseline cannot.
**Figures:** `fig_lm_run_dashboard` (plot); `fig_lm_stable_lr_range` (plot).
**Restate / forward:** `sec_practice` (z-loss, rewind); `sec_muon` (QK-clip); `subsec_scheduler-branching` (WSD). Forward: 4.6.
**Exercises:** 6 to 8; derive the logit bound under QK-clip; measure update-to-weight ratios.
**Cite:** Small-Scale Proxies 2309.14322; Kimi K2 2507.20534; Fantastic Pretraining Optimizers 2509.02046.

### 4.3 Low-Precision Training · T1 (Ada or newer for native fp8) · REVISE (v5 4.4) · [Codex 4.5, 5.6]
**Idea:** Low-precision training works with per-block scales and the sensitive tensors kept in higher precision.
**Opening:** fp8 doubles matrix throughput, but one outlier sets a block's scale; closes on what stays bf16.
**Teaches:**
- Which GEMMs run in fp8; tensorwise, rowwise, blockwise scaling; delayed against current; E4M3, E5M2.
- NVFP4 and MXFP4; Nemotron 3 as the production case (25T tokens); Four Over Six as the refinement.
- Tier line: fp8 native on Ada and Hopper; NVFP4 and MXFP4 simulated on Ada and A100, native on Blackwell.
**Derivation:** Block-scaling error against block size and in-block dynamic range.
**Notebook:** The 20M proxy of 3.4 in bf16 and fp8 (torchao) with overflow statistics; NVFP4 simulated for the MLP GEMMs. Claim: fp8 matches bf16 loss within seed noise at a stated speedup.
**Figures:** `fig_lm_block_scaling`; `fig_lm_fp8_loss_throughput` (plot).
**Restate / forward:** `sec_numerics` (E4M3 trades range for precision); `sec_memory_precision` (fp32 master weights). Pays hardware.md:322.
**Exercises:** 5 to 7; derive the error ratio of block sizes 16 and 128; measure overflow under delayed scaling.
**Cite:** FP8 Formats 2209.05433; NVFP4 pretraining 2509.25149; Nemotron 3 2512.20856; Four Over Six 2512.02010.

### 4.4 Continued Pretraining and Mid-Training · T2 · REVISE (v5 4.5) · [Codex 5.8]
**Idea:** A new domain is bought with forgetting unless replay and a re-warmed schedule pay for it.
**Opening:** Qwen3-0.6B continued on the book learns it and loses a general item; closes on a released model's stages.
**Teaches:**
- Re-warming and re-decaying the learning rate; replay; forgetting against domain gain as the trade.
- Mid-training stages (reasoning, code, long context) before post-training; Olmo 3's stage as the exemplar.
- Reading an open trajectory: Olmo 3 7B stage checkpoints scored with the Chapter 1 suite in one cell.
**Derivation:** none.
**Notebook:** Qwen3-0.6B base on the book corpus plus a technical corpus, with and without replay; domain loss, general loss, the suite. Claim: replay recovers most of the general-loss gap.
**Figures:** `fig_lm_replay_tradeoff` (plot); `fig_lm_olmo_stages` (plot).
**Restate / forward:** `sec_scheduler` (warmup tames a cold optimizer state, which a restart recreates).
**Exercises:** 5 to 7; measure forgetting against replay fraction; derive the re-warm loss bump.
**Cite:** Don't Stop Pretraining 2004.10964; Continual Pretraining 2403.08763; Olmo 3 2512.13961.

### 4.5 Long-Context Training · T2 · REVISE (v5 4.6) · [Codex 2.4, 5.8, 15.4]
**Idea:** A longer window is trained in stages on data that needs it, and measured on tasks that need it.
**Opening:** After RoPE extension the model reads 16K tokens without using them; closes on the road to a million.
**Teaches:**
- Progressive length schedules; data with real long-range dependencies, not padded short text.
- Effective context on synthetic (RULER) and natural (HELMET) tasks; needle tests necessary, not sufficient.
- The gap to 1M-token flagships and what closes it (sparse or linear mixing, context parallelism, KV tiers).
**Derivation:** none (YaRN and interpolation are restated, not re-derived).
**Notebook:** A small checkpoint from 4K to 16K with interpolation and with YaRN on long-dependency data, scored on RULER-style and document-QA tasks. Claim: YaRN keeps more of the 4K accuracy at 16K.
**Figures:** `fig_lm_length_schedule`; `fig_lm_effective_context` (plot).
**Restate / forward:** `sec_positional-information` (interpolation compresses all frequencies; YaRN scales the low ones). Forward: 14.5, 14.6.
**Exercises:** 5 to 7; measure accuracy against needle depth; derive the attention-entropy change.
**Cite:** How to Train Long-Context LMs 2410.02660; RULER 2404.06654; HELMET 2410.02694.

### 4.6 Training Budgets beyond the Compute Optimum · T0 · REVISE (v5 4.7, now core) · [Codex 5.5]
**Idea:** Once inference tokens enter the objective, the compute-optimal model is smaller and trained longer.
**Opening:** The Chinchilla rule calls the reference model undertrained, yet serving dwarfs training; closes on hardware.
**Teaches:**
- The compute-optimal size with lifetime inference cost; overtraining small models as the consequence.
- Repeated data under a token shortage: the diminishing value of an epoch.
- Fitting a scaling law on real logs: exponents, intervals, what a three-point ladder can say. Carries A5.
**Derivation:** The optimum with D_inf inference tokens in the objective.
**Notebook:** The Chinchilla form of `sec_scaling-laws` fitted to 4.2's committed ladder logs and the miniature grid; a lifetime-cost calculator. Claim: exponents agree within intervals; the optimum shifts smaller.
**Figures:** `fig_lm_ladder_fit` (plot); `fig_lm_lifetime_cost` (plot).
**Restate / forward:** `sec_scaling-laws` (loss follows a power law in parameters and tokens). Pays scaling-laws.md:498 and :825.
**Exercises:** 5 to 7; derive the optimum under repeated data; measure exponent sensitivity to one ladder point.
**Cite:** Beyond Chinchilla-Optimal 2401.00448; Scaling Data-Constrained LMs 2305.16264.

## Chapter 5 Hardware for Training and Inference

**Label:** `chap_lm_hardware` · 4 sections · T0 to T4. **Purpose:** why accelerators, servers and clusters are built as they are; predicting step time and decode throughput.
**Hook:** The reader knows one GPU from `sec_hardware`; a released model is trained on thousands of chips and served from racks whose links and prices decide what a token costs.
**Route:** What does a chip's area, power and perimeter buy (5.1)? Why do training, prefill and decode want different hardware (5.2)? Where inside a server can state live (5.3)? What bounds parallelism across a rack and a cluster, and what does a token cost (5.4)?
**Scope:** `sec_hardware` and `sec_perf_model` are restated, not re-taught; buying and renting stay in `sec_hardware_buyers` and `sec_cloud_instances`; reliability is 6.6.
**Resources:** How to Scale Your Model (5.1, 5.2); Stas Bekman, Machine Learning Engineering (5.3, 5.4); GPU MODE lectures, Modal GPU Glossary (5.1); The Datacenter as a Computer (5.4).
**Further Reading notebook:** none

### 5.1 Accelerator Design · T0 (T2 microbenchmarks) · REVISE · [Codex 4.8]
**Idea:** A chip is a budget of area, power and package perimeter, and three ratios place it.
**Opening:** Two datasheets, more FLOP/s against more bandwidth, and one decode step to run; closes on workloads.
**Teaches:**
- Die area among matrix units, SRAM, memory controllers; power; perimeter caps stacks and I/O.
- HBM, GDDR, LPDDR, SRAM-only; three ratios: ridge point, capacity over bandwidth, bytes per FLOP.
- Specification table as a data file: A100 to Rubin, Ironwood, Trn3, MI455X, Groq (NVIDIA licence), Cerebras.
**Derivation:** How the SRAM against matrix-unit split moves the ridge point; capacity over bandwidth as a decode floor.
**Notebook:** The table in the ridge-point against capacity-over-bandwidth plane (T0); A100 microbenchmarks of matrix throughput by shape and HBM, L2, shared-memory bandwidth (T2). Claim: within 15% of the datasheet.
**Figures:** `fig_lm_chip_budget`; `fig_lm_accelerator_plane` (plot).
**Restate / forward:** `subsec_hw-shoreline` (compute grows with area, bandwidth with perimeter); `tab_gpu_specs` (extended).
**Exercises:** 5 to 7; derive how many SRAM-only chips hold an 8B model; measure the half-peak matrix size.
**Cite:** AI and Memory Wall 2403.14123; Ampere Architecture 2208.11174; Trn3, Ironwood, Helios, Groq vendor pages.

### 5.2 Training, Prefill and Decode Workloads · T2 · REVISE · [Codex 4.8, 14.1]
**Idea:** Training, prefill and decode want different hardware because they sit on different sides of the ridge.
**Opening:** Qwen3-8B at batch one leaves the GPU idle and at batch 128 is compute-bound; closes on the server.
**Teaches:**
- Training: FLOPs per token (6N), state bytes per parameter, activation memory, communication per step.
- Prefill compute-bound; decode bandwidth- and capacity-bound; MoE: total sets capacity, active sets traffic.
- The throughput-interactivity frontier and cost per million tokens; separate prefill and decode pools.
**Derivation:** t(b, L) = max(2 N_act b / P, (W + b L k_KV) / beta) under capacity C; crossover batch; frontier shift with L.
**Notebook:** Qwen3-8B on one A100: step time, prefill latency, decode throughput over batch and context; the fit predicts H100, B200 and an SRAM part against InferenceX. Claim: within 25% on the A100.
**Figures:** `fig_lm_three_workloads`; `fig_lm_frontier` (plot).
**Restate / forward:** `subsec_hw-two-workloads`; `eq_hw_decode_bound` (weight bytes over bandwidth); the batch-164 crossover. Forward: 13.4.
**Exercises:** 5 to 7; derive the crossover batch for an MoE; measure KV growth against context.
**Cite:** Scaling Transformer Inference 2211.05102; DistServe 2401.09670; InferenceX.

### 5.3 Memory Tiers and Data Paths inside a Server · T3 · MERGE(v5 5.3+5.4) · [Codex 1.8, 4.7, 7.7]
**Idea:** Every byte in a server crosses a link with an alpha-beta cost; state lives where its link hides behind compute.
**Opening:** The optimizer state does not fit the GPU and host memory is twenty times slower; closes on the cluster.
**Teaches:**
- Tiers (SRAM, HBM, host DDR, NVMe, file systems): bandwidth, latency, capacity, cost; CXL, disks as rows.
- The server as a graph (GPUs, NVSwitch, PCIe, CPUs, NICs, drives); NUMA; the small-transfer latency floor.
- Offloading as moving state down a tier (ZeRO-Infinity) and the overlap condition; KV tiers.
**Derivation:** Overlap condition from intensity and link bandwidth; alpha-beta per link; bisection, switch against tree.
**Notebook:** On the 8x A100 node: host-device bandwidth per NUMA node, p2p matrix, all-reduce, all-to-all, save and load of an 8B state (reused by 6.6). Claim: alpha, beta within 20% of specification.
**Figures:** `fig_lm_server_graph`; `fig_lm_offload_condition` (plot).
**Restate / forward:** `subsec_hw-bytes` (one GPU's ladders); `sec_multi_gpu` (alpha-beta); `subsec_hw-interconnects` (the PCIe tree).
**Exercises:** 5 to 7; derive the batch at which KV offload overlaps; measure checkpoint bandwidth against shards.
**Cite:** ZeRO-Infinity 2104.07857; Demystifying NCCL 2507.04786; Mooncake 2407.00079.

### 5.4 Networks, Scale-Up Domains and Cost · T4 (T0 calculator) · MERGE(v5 5.5+5.6) · [Codex 4.7, 4.8, 6.2, 7.7]
**Idea:** The scale-up domain bounds tensor and expert parallelism; the network carries the rest; cost follows.
**Opening:** A 70B model needs eight nodes, so an all-reduce crosses a NIC, not an NVSwitch; closes on execution.
**Teaches:**
- Scale-up domains (8-GPU node, NVL72, Helios, Trn3, TPU pods) against the scale-out fabric.
- Cross-node collectives: ring against tree, hierarchical all-reduce, all-to-all for experts; which layouts fit.
- Cost per million tokens from price, throughput, utilization; owned against rented; power and cooling.
**Derivation:** Ring against tree all-reduce across nodes and the crossover size; hierarchical speedup.
**Notebook:** All-reduce and all-to-all across two nodes, flat against hierarchical; a layout calculator (T0) for a node, NVL72 and Helios; dollars per million tokens from measured MFU. Claim: crossover within 2x of prediction.
**Figures:** `fig_lm_two_networks`; `fig_lm_ring_tree_crossover` (plot).
**Restate / forward:** `sec_multi_gpu` (ring all-reduce moves 2(n-1)/n per rank); `sec_cloud_instances` (rental prices). Forward: 6.5.
**Exercises:** 5 to 7; derive hierarchical speedup for n nodes of g GPUs; measure all-to-all against size.
**Cite:** DeepSeek-V3 Hardware 2505.09343; Llama 3 2407.21783; RDMA at Meta Scale (SIGCOMM 2024).
## Chapter 6 Parallelism Beyond One Node

**Label:** `chap_lm_parallelism`.
**Hook:** The reader ran DDP on four GPUs and derived the ZeRO stages; an 8B model's 128 GB of state fits no card, so the state is cut up and the pieces must talk.
**Route:** How do ranks on different machines share one state (6.1)? What does splitting an activation cost (6.2)? How much of a pipeline idles (6.3)? What does a remote expert cost (6.4)? Which layout of 16 GPUs is fastest (6.5)? How often should a failing run save (6.6)?
**Scope:** Mechanisms and measurements; library choice is `sec_training_systems`, kernels Chapter 12, serving Chapter 13. Multi-node runs are committed logs.
**Resources:** Ultra-Scale Playbook (6.5); How to Scale Your Model (6.2, 6.5); PyTorch distributed docs (6.1, 6.6); `nccl-tests` (6.1); Megatron Core guide (6.2); TorchTitan (6.5); torchft (6.6); CS336 systems assignment (6.2).
**Further Reading notebook:** none.

### 6.1 Sharded Data Parallelism across Nodes · T4 (T3 core) · MERGE(v5 6.1+6.2) · [Codex 6.1, 6.7, 6.8]
**Idea:** FSDP shards every state across all ranks, and the ranks find each other through a rendezvous and a mesh.
**Opening:** Eight ranks each hold an eighth of the state and gather it before use. Closing: a collective with no partner.
**Teaches:** The `fully_shard` life cycle, wrapping and prefetch as overlap dials; rendezvous (`torchrun` per node, SLURM, `d2l.launch`); meshes; hangs read from the flight recorder.
**Derivation:** ZeRO-3 bytes per rank (2+2+12)N/W; N = 8B, W = 8 gives 16 GB.
**Notebook:** Do an 8B model's memory and step time under `fully_shard` on 8x A100 follow ZeRO-3? Wrap and prefetch sweep; two-node repeat (T4 logs); one induced hang. Check: memory within 10%. Claim: the dials move exposed communication.
**Figures:** `fig_lm_fsdp_lifecycle_timeline` (step phases); `fig_lm_fsdp_mem` (plot: memory by wrap).
**Restate / forward:** `tab_mgp-zero`; `subsec_adamw-memory`; `sec_training_systems` linked. Forward: 6.5.
**Exercises:** 6 to 8. Derive ZeRO-2 vs ZeRO-3 memory at W = 64; measure all-gather bandwidth; open: resume on 16 ranks.
**Cite:** ZeRO; PyTorch FSDP.

### 6.2 Tensor, Sequence and Context Parallelism · T3 · MERGE(v5 6.3+6.5) · [Codex 6.5, 6.6, 7.4]
**Idea:** Sharding an activation along a dimension costs the collective that restores the dependence.
**Opening:** Split the MLP's first weight by columns and the second by rows and nothing is gathered until the end. Closing: a sequence too long for one GPU.
**Teaches:** Column- and row-parallel linears with the conjugate f and g operators; sequence parallelism; ring attention as distributed online softmax; TP in the node, CP across.
**Derivation:** TP bytes per layer against 24·b·s·d² FLOPs at d = s = 4096.
**Notebook:** Does two-way TP of the reference model's block match one GPU, and does TP 2, 4, 8 step time follow the alpha-beta prediction? Ring vs all-to-all CP at 32K tokens. Check: bf16 tolerance; 15%. Claim: exposed communication grows with TP degree.
**Figures:** `fig_lm_tp_fg` (f and g around a column-row pair); `fig_lm_cp_scaling` (plot: ring vs all-to-all).
**Restate / forward:** `sec_attention-at-scale`; `ex_mgp-tensor`. Forward: 6.5.
**Exercises:** 7 to 9. Derive bytes per layer with and without sequence parallelism; measure where ring overlap fails; open: CP with TP.
**Cite:** Megatron-LM; Ring Attention.

### 6.3 Pipeline Parallelism · T3 (T0 simulator) · KEEP · [Codex 7.1 to 7.3]
**Idea:** A pipeline's cost is its bubble, and schedules shrink it at the price of memory.
**Opening:** Four stages, one batch: three GPUs idle at every moment, and micro-batches fill the gaps. Closing: can the bubble reach zero.
**Teaches:** GPipe and its bubble; 1F1B, which keeps at most p micro-batches of activations alive; interleaved stages; backward split into input- and weight-gradient halves (zero bubble, DualPipe).
**Derivation:** The bubble fraction (p-1)/(m+p-1); p = 4, m = 8 gives 3/11.
**Notebook:** Does a schedule simulator predict the step time and peak memory of a four-stage run? A T0 simulator against `torch.distributed.pipelining` at m = 4, 8, 16. Check: bubble within 10%. Claim: GPipe holds more activation memory than 1F1B.
**Figures:** `fig_lm_pp_gantt` (three schedules); `fig_lm_pp_bubble` (plot: bubble against m).
**Restate / forward:** `sec_perf_model`. Forward: 6.5.
**Exercises:** 6 to 8. Derive the stage balance with embedding and head on the end stages; measure activation memory per stage; open: split backward.
**Cite:** GPipe; Zero Bubble; DualPipe.

### 6.4 Expert Parallelism · T3 · KEEP · [Codex 3.2, 3.8, 7.5]
**Idea:** Experts live on different devices, so routing is an all-to-all whose cost depends on skew.
**Opening:** With 64 experts over eight GPUs each token's top-k choice names a device. Closing: what the busiest GPU costs.
**Teaches:** Dispatch and combine as two all-to-alls; capacity factors vs dropless grouped GEMMs; skew and the straggler rule; node-limited routing; host-synchronization pitfalls.
**Derivation:** All-to-all bytes per token, 2·k·d·2 times (1-1/E); d = 1024, k = 2, E = 8 gives 7 KB.
**Notebook:** Does a small MoE trained with EP = 8 match one GPU's loss, and how does dispatch time grow with skew? The `sec_moe` block with 64 experts; skew 1, 2, 4. Check: loss within noise. Claim: dispatch time tracks the busiest rank.
**Figures:** `fig_lm_ep_dispatch` (tokens per rank); `fig_lm_ep_skew` (plot: time against skew).
**Restate / forward:** `sec_moe`; `tab_collectives`. Pays moe.md:116. Forward: 13.4.
**Exercises:** 6 to 8. Derive node-limited bytes for E = 64; measure grouped GEMM vs padding; open: upcycle a dense checkpoint.
**Cite:** Switch Transformers; MegaBlocks; DeepEP.

### 6.5 Composed Parallelism and Topology · T4 · REVISE (v5 6.7) · [Codex 7.6, 7.8]
**Idea:** A configuration is a mesh over DP x TP x PP x CP x EP, and a cost model ranks the feasible ones.
**Opening:** A 7B model on 16 GPUs admits dozens of layouts; most do not fit, some that fit are slow. Closing: when weak scaling stops paying.
**Teaches:** The chattiest axis on the fastest link; feasibility from the memory formulas of 6.1 to 6.3; step time as compute at measured MFU plus one exposed term per axis; `tab_training_tools` linked.
**Derivation:** The composed step-time model; DP 8 x TP 2 against DP 4 x TP 4.
**Notebook:** Does the cost model's ranking for a 7B model on two nodes agree with measurement? The top three and one predicted-slow layout with TorchTitan at a 512 x 4096 global batch (eight runs, about 15 GPU-hours). Check: within 15%. Claim: the measured order is the predicted order.
**Figures:** `fig_lm_compose_mesh` (mesh axes on two nodes' links); `fig_lm_compose_rank` (plot: predicted vs measured).
**Restate / forward:** `sec_perf_model`; 5.4's hierarchical all-reduce. Forward: none.
**Exercises:** 6 to 8. Derive memory per rank for DP 2 x TP 4 x PP 2; measure strong scaling from 8 to 16 GPUs; open: add EP 16 and re-rank.
**Cite:** TorchTitan; Ultra-Scale Playbook.

### 6.6 Checkpoints, Failures and Long Runs · T3 · REVISE (v5 6.8) · [Codex 7.7]
**Idea:** At scale something fails every few hours, and the checkpoint interval follows from the failure rate.
**Opening:** A published 54-day run on 16,384 GPUs logged 419 interruptions. Closing: surviving a failure without stopping.
**Teaches:** Failure rates per node-day, `d2l.failure_rate(gpus)`; the Young-Daly interval; sharded checkpoints resharded and saved asynchronously; rank replacement (torchft); `subsec_training_long_run` linked.
**Derivation:** Half an interval lost per failure plus one save per interval gives sqrt(2C/lambda); 23 minutes at C = 90 s.
**Notebook:** How much of a run is lost at each interval, and does a killed rank resume on a different world size? Interval curves from 5.3's checkpoint cost; the reference model on 8 GPUs, one rank killed, resumed on 4. Check: identical batches. Claim: waste under 5% at the optimum.
**Figures:** `fig_lm_long_waste` (plot: waste against interval); `fig_lm_long_ckpt_layout` (resharded checkpoint).
**Restate / forward:** `subsec_training_long_run`; `sec_read_write`; 4.1's cursor. Forward: none.
**Exercises:** 6 to 8. Derive Daly's correction at lambda·C = 0.1; measure the cost of asynchronous saves; open: a 100,000-GPU failure budget.
**Cite:** Young (1974) and Daly (2006), by title; Revisiting Reliability; torchft.

## Chapter 7 Fine-Tuning

**Label:** `chap_lm_finetuning`.
**Hook:** The reader has a pretrained model that continues text and a task it should do instead; changing some weights on a few thousand examples is the oldest route, and the questions are which weights, in what precision.
**Route:** How much of the model should move (7.1)? How does a conversation become a training example (7.2)? What makes an example worth training on (7.3)? Why does a rank-8 update do most of the work (7.4)? Can frozen weights sit in four bits (7.5)? How does the recipe reach a node (7.6)? Do adapters add up (7.7)?
**Scope:** Reward-based training is Chapter 8; retrieval for new facts is 2.5; pinning and licences are `sec_software_ecosystem`.
**Resources:** LoRA Without Regret (7.4); Practical Tips for Finetuning LLMs Using LoRA (7.4); smol course (7.2, 7.3); PEFT docs (7.4 to 7.6); FSDP-QLoRA (7.6); torchtune, Unsloth, Axolotl docs (7.6); Tulu 3 (7.3).
**Further Reading notebook:** none.

### 7.1 Full Fine-Tuning · T2 · KEEP · [Codex 8.1]
**Idea:** Unfreezing more buys task accuracy and costs memory and forgetting.
**Opening:** The reader froze a backbone in Basics; a 1.7B decoder has 28 blocks, and the choice is which may move. Closing: must every weight move.
**Teaches:** Unfreezing depth and layer-wise learning rates; a task head vs a generative answer; the memory ledger, why 1.7B fits a 40 GB card and 8B does not; forgetting and replay.
**Derivation:** none.
**Notebook:** How do task accuracy and suite regression move with unfreezing depth? Qwen3-1.7B on one A100 40 GB; four unfreezing depths. Check: paired bootstrap. Claim: the fully unfrozen arm gains on the task and loses on the suite.
**Figures:** `fig_lm_ft_depth` (plot: task and suite scores against depth); `fig_lm_ft_ledger` (memory ledger).
**Restate / forward:** `subsec_freezing`; `subsec_adamw-memory`. Forward: none.
**Exercises:** 6 to 8. Derive the ledger for 8B under ZeRO-3; measure learning rate against epochs; open: the replay fraction that restores the suite.
**Cite:** ULMFiT; LoRA Learns Less and Forgets Less.

### 7.2 Supervised Instruction Tuning · T1 · KEEP · [Codex 8.2, 8.4]
**Idea:** Instruction tuning is behaviour cloning on serialized conversations with the loss on the assistant's tokens only.
**Opening:** Base Qwen3-0.6B continues a question with more questions; a conversation is one string with role markers. Closing: where conversations come from.
**Teaches:** The part's chat template (`d2l_chatml_v1`) and `render_chat` as the only serializer; completion-only masks and end-of-turn tokens; packing through 4.1's masks; the prefix invariant.
**Derivation:** none.
**Notebook:** Does the inference prompt equal the training prefix token by token, and what does the mask change? Qwen3-0.6B base on 10,000 Tulu 3 conversations; completion-only vs every-token loss. Check: the prefix test. Claim: the accuracy difference with its interval.
**Figures:** `fig_lm_sft_mask` (a two-turn conversation with its loss mask); `fig_lm_sft_arms` (plot: the two loss arms).
**Restate / forward:** `subsec_il_sft`; 4.1's masks; 3.6's reserved ids. Forward: 10.1.
**Exercises:** 6 to 8. Derive the loss-bearing fraction from turn lengths; measure a missing end-of-turn token; open: weighting later turns.
**Cite:** InstructGPT; Tulu 3.

### 7.3 Instruction Data · T1 · KEEP · [Codex 8.3]
**Idea:** A small curated set beats a large random one at equal tokens.
**Opening:** A public mixture holds a million conversations and a thousand chosen well can match it. Closing: whether the model can write its own.
**Teaches:** Sources: human-written, templated, model-generated (Self-Instruct; Magpie from the user-turn opener); selection by quality, difficulty and diversity; decontamination; the generator outside the judge's family.
**Derivation:** none.
**Notebook:** At equal tokens, does a curated 5,000-example subset beat a random one and approach the full set? Qwen3-0.6B base with the 7.2 recipe; Magpie generation decontaminated. Check: paired bootstrap. Claim: curated beats random outside the interval.
**Figures:** `fig_lm_instr_select` (the selection funnel); `fig_lm_instr_arms` (plot: three arms).
**Restate / forward:** 3.3's near-duplicate detection; 3.5's collapse; `subsec_mdl-paired-comparisons`. Forward: none.
**Exercises:** 5 to 7. Derive the expected test overlap at a given contamination rate; measure cluster count against accuracy; open: a difficulty curriculum.
**Cite:** LIMA; Magpie; Self-Instruct.

### 7.4 Low-Rank Adaptation · T1 · MERGE(v5 7.4+7.5) · [Codex 8.5, 8.6]
**Idea:** A rank-r update captures most of fine-tuning at a fraction of the state.
**Opening:** One 7.1 weight delta's singular values fall off within a few dozen directions, so the update can be two thin factors. Closing: frozen weights that do not fit.
**Teaches:** `LoRALinear` with B zero at start; targets and rank; the learning rate and the A/B ratio; exact merge and unmerge; one table of relatives: PiSSA, DoRA, VeRA, prompt tuning, GaLore.
**Derivation:** alpha/sqrt(r) keeps the update's variance rank-independent (rsLoRA); B's gradient scales with A's norm (LoRA+).
**Notebook:** At which rank and targets does LoRA match 7.1, and does the scaling matter? Qwen3-1.7B on one 16 GB card; ranks 1, 8, 64; attention vs attention and MLP; both scalings. Check: PEFT oracle. Claim: attention and MLP at r = 8 lands inside 7.1's interval.
**Figures:** `fig_lm_lora_merge` (adapter and merge); `fig_lm_lora_targets` (plot: accuracy against rank).
**Restate / forward:** `sec_mdl-svd-low-rank`; `subsec_adamw-memory`. Forward: none.
**Exercises:** 7 to 9. Derive the variance argument and the LoRA+ ratio; measure a DoRA layer against PEFT's; open: GaLore's memory against LoRA's.
**Cite:** LoRA; rsLoRA; LoRA+.

### 7.5 Quantized Fine-Tuning · T1 · REVISE (v5 7.6) · [Codex 8.7, 12.1]
**Idea:** Frozen weights can be stored in four bits while adapters train in sixteen.
**Opening:** Qwen3-8B in bf16 is 16.4 GB, more than a 15 GiB slot; under LoRA those weights never change. Closing: what the adapter must absorb.
**Teaches:** The part's quantizers, owned here: absmax and affine grids, per-block scales, their error; NF4 at normal quantiles with double quantization; dequantization inside the matmul; LoftQ.
**Derivation:** NF4 as equal-probability bins against Lloyd-Max, four levels on a standard normal; Delta^2/12 restated from 4.3.
**Notebook:** Does QLoRA of Qwen3-8B within 15 GiB match bf16 LoRA of Qwen3-1.7B, and what does LoftQ change? The three quantizers on Qwen3 tensors; 8B in NF4; LoftQ vs zero initialization. Check: bitsandbytes; the ledger. Claim: peak memory under 15 GiB.
**Figures:** `fig_lm_nf4_entropy` (NF4 next to Lloyd-Max); `fig_lm_quant_blocksize` (plot: error by block size).
**Restate / forward:** 4.3's block-scaling error; `sec_numerics`. Forward: 11.1.
**Exercises:** 6 to 8. Derive Lloyd-Max for four levels; measure double quantization's saving and error; open: the rank that absorbs one layer's error.
**Cite:** QLoRA; LoftQ; 8-bit Optimizers.

### 7.6 Fine-Tuning at Scale and Toolchains · T3 (T1 reduced) · MERGE(v5 7.7+7.8) · [new in v5]
**Idea:** The same recipe scales to a node by sharding frozen weights and adapters, and a toolchain is that recipe as configuration.
**Opening:** A 27B model is 54 GB in bf16; sharded over eight 40 GB cards LoRA fits, full fine-tuning does not. Closing: models that fit no node.
**Teaches:** LoRA and QLoRA under FSDP: frozen weights sharded like any state, memory from the ledger; the 7.4 recipe in PEFT with TRL and in torchtune matched to the oracle; hosted fine-tuning (Tinker).
**Derivation:** none.
**Notebook:** Does LoRA of Qwen3.6-27B on one node follow the predicted memory and step time, and do two toolchains reproduce the recipe? FSDP on 8x A100 40 GB; Qwen3-32B as the fallback (blueprint Decision 11); T1 reduced: 7.4 in both toolchains. Check: 20 losses within 1e-3. Claim: memory per GPU within 10%.
**Figures:** `fig_lm_ft_fsdp_memory` (plot: memory per GPU by method); `fig_lm_ft_fsdp_timeline` (one step).
**Restate / forward:** 6.1's ZeRO-3 ledger and `fully_shard` wrapping; `sec_software_ecosystem` linked. Forward: none.
**Exercises:** 5 to 7. Derive the node count for full fine-tuning of 27B; measure the throughput gap between toolchains; open: configuration diffs as ablation records.
**Cite:** FSDP-QLoRA; PyTorch FSDP; Tinker (Thinking Machines, verified 2026-11-05).

### 7.7 Merging and Retaining Capabilities · T1 · KEEP (v5 7.9) · [Codex 8.8]
**Idea:** Adapters and checkpoints add, and the sum keeps both tasks if the interfering entries are handled.
**Opening:** Adding two task adapters' deltas gives a model that does both, sometimes; sign conflicts are where it fails. Closing: fine-tuning or retrieval for new facts.
**Teaches:** Task vectors and task arithmetic; TIES (trim, elect a sign, merge) and DARE (drop at random, rescale); soups of checkpoints; composing adapters; forgetting and replay.
**Derivation:** DARE's unbiasedness, dropping a fraction p and rescaling by 1/(1-p).
**Notebook:** Which merge keeps both tasks and the suite? Two LoRA adapters on Qwen3-0.6B merged by sum, TIES and DARE; interference as the sign-disagreement fraction. Check: both tasks and the suite with intervals. Claim: TIES and DARE keep both where addition loses one.
**Figures:** `fig_lm_merge_additivity` (task vectors and sign conflicts); `fig_lm_merge_pareto` (plot: task A vs B).
**Restate / forward:** 7.4's exact merge; 2.5's retrieval route; `subsec_scheduler-branching`. Forward: none.
**Exercises:** 5 to 7. Derive DARE's variance in p; measure a soup of the reference model's branch checkpoints; open: new facts by fine-tuning vs retrieval.
**Cite:** Task Arithmetic; TIES; DARE.

## Chapter 8 Preference Optimization and Reinforcement Learning

**Label:** `chap_lm_rl`.
**Hook:** The reader derived GRPO and DPO on a toy verifier task; on a real language model the reward is a second network, sampling dominates, and the policy exploits what the reward left unsaid.
**Route:** How is a reward learned from comparisons (8.1)? How do four models fit one GPU (8.2)? Why can pairs train the policy directly (8.3)? What changes with a verifier (8.4)? Why is generation the cost (8.5)? How far can a proxy be optimized (8.6)? How is safety measured under attack (8.7)?
**Scope:** Derivations stay in `sec_regularized` and `sec_rl_sequences`; agent RL is 10.4, reasoning 9.3, injection 10.5. Notation: `tab_rl_notation` (owner `sec_mdp`), extended by the token-level state, reference policy and policy lag.
**Resources:** RLHF Book (8.1 to 8.3); Approximating KL Divergence (8.2); Reward Hacking in RL (8.6); TRL docs (8.1 to 8.4); verl docs (8.5); CS336 alignment assignment (8.4); HarmBench and XSTest repositories (8.7).
**Further Reading notebook:** asynchronous RL corrections, constitutional and deliberative alignment, reward-model ensembles, RL scaling studies, watermarking.

### 8.1 Reward Models · T2 · KEEP · [Codex 9.1]
**Idea:** A reward model is a classifier of comparisons on a language-model backbone, and what it rewards must be measured.
**Opening:** The reader fitted a Bradley-Terry reward on a gridworld; here a head on the last token must generalize from 10,000 comparisons. Closing: what else it tracks.
**Teaches:** The reward head and the pairwise loss; preference data and the agreement ceiling; shift invariance and calibration; length and format correlation; the best-of-n curve.
**Derivation:** The pairwise gradient as a push on the margin; best-of-n reward at accuracy p, n = 2.
**Notebook:** How accurate is a 0.6B reward model and how much of its score is length? A head on Qwen3-0.6B trained on a public preference slice; failure probes; best-of-n with the 1.5 judge. Check: held-out accuracy. Claim: accuracy beats the length-only baseline.
**Figures:** `fig_lm_rm_head` (the head on a pair's last token); `fig_lm_rm_acc_vs_bon` (plot: best-of-n).
**Restate / forward:** `sec_regularized`; 7.2's template. Forward: 8.6.
**Exercises:** 6 to 8. Derive the margin gradient and the unidentifiable shift; measure annotator agreement against accuracy; open: a length-debiased head.
**Cite:** InstructGPT; RewardBench.

### 8.2 RLHF with PPO at Language-Model Scale · T2 · REVISE · [Codex 9.2, 9.4]
**Idea:** Four models share the GPU and a per-token KL penalty keeps the policy near the reference.
**Opening:** The reader's PPO had one network and a simulator; here four copies of a 0.6B model share one GPU. Closing: can the reward model be skipped.
**Teaches:** The four roles and their memory; the per-token KL penalty as reward shaping, GAE on token trajectories, whitening; k1, k2, k3 as KL estimators, in the reward or in the loss.
**Derivation:** Bias and variance of k1 and k3 on a two-token vocabulary with ratios 1/2 and 2.
**Notebook:** Do the advantages the code computes equal hand-computed ones, and what does each role cost? PPO on Qwen3-0.6B instruct with 8.1's reward model; seconds per role from `d2l.PhaseTimer`. Check: three hand-computed trajectories. Claim: role memory matches the ledger.
**Figures:** `fig_lm_ppo_roles` (the four models and their flows); `fig_lm_ppo_curves` (plot: reward and KL).
**Restate / forward:** `sec_ppo`; `sec_actorcritic`; `ex_rl-sequences-kl-loss`. Forward: 8.5.
**Exercises:** 6 to 8. Derive mean and variance of k1, k2, k3 on the two-token case; measure whitening's effect; open: critic-free PPO with a leave-one-out baseline.
**Cite:** N+ Implementation Details of RLHF with PPO; Approximating KL Divergence.

### 8.3 Direct Preference Optimization and Its Variants · T2 · MERGE(v5 8.3+8.4) · [Codex 9.3]
**Idea:** The policy itself is the reward model, and the choice of reference and of pairing decides what the gradient does.
**Opening:** In Advanced the KL-regularized optimum made the reward a log-ratio of policy to reference, so comparisons train the policy directly. Closing: pairs drawn from the policy itself.
**Teaches:** The DPO gradient, a sigmoid weight of the implicit margin times a difference of log-probability gradients; reference dependence; likelihood displacement; one table of relatives: IPO, KTO, SimPO, ORPO, online DPO.
**Derivation:** none.
**Notebook:** How do chosen and rejected log-probabilities move under DPO, and does online pairing change the win rate? DPO on 7.2's checkpoint with 8.1's pairs; offline vs policy-sampled pairs. Check: the 1.5 judge, both orders. Claim: both log-probabilities fall while the margin grows.
**Figures:** `fig_lm_dpo_logps` (plot: chosen, rejected, margin); `fig_lm_po_family` (relatives by assumption).
**Restate / forward:** `eq_dpo_reward` in `sec_rl_sequences`; `sec_offline`. Forward: none.
**Exercises:** 6 to 8. Derive the DPO gradient and IPO's target; measure a beta sweep against displacement; open: SimPO on the same pairs.
**Cite:** DPO; Unintentional Unalignment.

### 8.4 RL with Verifiable Rewards · T2 · REVISE (v5 8.5) · [Codex 9.5 to 9.7]
**Idea:** With a verifier the reward is exact, and the remaining choices are how to aggregate tokens and how to keep entropy.
**Opening:** An arithmetic problem has one checkable answer; the checker must find it inside hundreds of tokens, and entropy falls as the policy learns. Closing: why sampling sets the cost.
**Teaches:** Robust verification; GRPO on multi-token responses; the aggregation choices restated (clip-higher, sequence ratio, constant divisor); new: overlong filtering, CISPO, entropy collapse, predictable scaling.
**Derivation:** The first-order entropy change under one step, minus the step size times the covariance of log-probability and advantage.
**Notebook:** Does GRPO on Qwen3-0.6B base solve unseen arithmetic templates, and how does one aggregation choice change length and entropy? Sequence-normalized vs constant divisor. Check: 200 held-out items with pass@k intervals. Claim: accuracy rises above base; length moves as derived.
**Figures:** `fig_lm_grpo_loop` (sample, verify, normalize, update); `fig_lm_grpo_curves` (plot: accuracy, length, entropy).
**Restate / forward:** `sec_rl_sequences`, `subsec_ppo_asymmetric`, `eq_seq_ratio`; `subsec_pg_loss_divisor`. Forward: 9.3.
**Exercises:** 7 to 9. Derive the entropy formula and the fastest-collapse case; measure a loose checker; open: fit the sigmoid scaling on the logs.
**Cite:** DeepSeekMath; DAPO; The Art of Scaling RL Compute.

### 8.5 RL Training Systems · T3 · REVISE (v5 8.6) · [Codex 9.8, 14.7]
**Idea:** Generation dominates RL compute, so the system separates rollouts from updates and controls policy lag.
**Opening:** In 8.4 sampling took most of each step; an engine samples faster but holds its own weights. Closing: a gradient from a sampler several steps old.
**Teaches:** Engine and trainer as two services with weight synchronization; colocated vs disaggregated placement; policy lag and effective sample size; the 2026 corrections as one design space: clipped ratios, VCPO, OAPL, SAO.
**Derivation:** Effective sample size and the lag at which it halves, on four samples.
**Notebook:** How does wall-clock per RL step split across phases in colocated and split layouts, and what does policy lag do to accuracy? Qwen3-1.7B with vLLM rollouts on 8x A100; lag 0, 1, 4, 16; five T3 runs. Check: effective sample size. Claim: accuracy falls beyond the predicted lag.
**Figures:** `fig_lm_rls_layouts` (both placements); `fig_lm_rls_lag` (plot: accuracy against lag).
**Restate / forward:** `sec_mdl-bayesian-computation`; `sec_hardware`. Forward: 13.2.
**Exercises:** 6 to 8. Derive effective sample size at lag k; measure sync time against model size; open: masked ratios and VCPO at lag 16 on the logs.
**Cite:** HybridFlow; AReaL; VCPO (arXiv 2602.17616).

### 8.6 Reward Overoptimization · T1 · REVISE (v5 8.7, first half) · [Codex 9.9]
**Idea:** A proxy reward is exploited, and the language-specific exploits are measured with checkers the policy never saw.
**Opening:** The 8.1 score correlated with length; a policy pointed at it learns to be long, then to agree with the user. Closing: what the policy must refuse.
**Teaches:** Proxy against true reward as a function of KL; the language exploits, length, format, sycophancy, hedging, judge quirks; remedies, ensembles, early stopping by KL, length control; an independent referee.
**Derivation:** R(d) = d(alpha minus beta log d) with d the square root of KL, its peak at exp(alpha/beta minus 1).
**Notebook:** How far does optimizing against a length-biased reward go before an independent checker says quality falls? 8.4's GRPO against 8.1's head plus a length term; remedies as cells on the run. Check: the Qwen3.5-4B judge and a sycophancy probe. Claim: the proxy rises while the checker peaks and falls.
**Figures:** `fig_lm_oo_gao` (plot: proxy and true reward against KL); `fig_lm_oo_hedge` (plot: length and sycophancy).
**Restate / forward:** `sec_regularized`; `subsec_rl_sequences_hacking`. Forward: none.
**Exercises:** 6 to 8. Derive the peak and the stopping KL from fitted alpha and beta; measure a three-head ensemble; open: a length-controlled win rate.
**Cite:** Scaling Laws for Reward Model Overoptimization; Towards Understanding Sycophancy.

### 8.7 Safety Evaluation and Red-Teaming · T1 · NEW (v5 8.7, second half) · [Codex 9.9]
**Idea:** A model's safety is a pair of error rates, harmful compliance and over-refusal, measured at rest and under attack.
**Opening:** After 7.2 the model answers everything, including what it should decline; after a safety fine-tune it also declines a cooking question about knives. Closing: harm inside a tool's output.
**Teaches:** Should-refuse and should-comply sets with paired, clustered error rates (aggregate rates published only, blueprint Decision 13); three jailbreak families; memorization extraction; bias and written principles, one paragraph each.
**Derivation:** none.
**Notebook:** What does a small safety fine-tune change in harmful compliance, over-refusal and attack success? Qwen3-0.6B instruct before and after a safety slice; three jailbreak families; extraction from 1,000 prefixes. Check: Qwen3.5-4B with the rubric. Claim: compliance falls and over-refusal rises. Carries A8.
**Figures:** `fig_lm_safety_rates` (plot: compliance against over-refusal); `fig_lm_safety_attacks` (plot: success per family).
**Restate / forward:** `subsec_mdl-paired-comparisons`, `ex_mdl-statistics-clustered`. Forward: 10.5.
**Exercises:** 6 to 8. Derive the clustered standard error for unequal categories; measure refusal against capability across safety-data fractions; open: extraction against prefix length.
**Cite:** HarmBench; XSTest; Extracting Training Data.

### FR Further Reading: Post-Training · T0 · NEW · [new]
**Idea:** The post-training results that fail the teachable-core test but that a practitioner will meet.
**Opening:** The reader has run PPO, DPO, GRPO and a safety evaluation; this notebook names what each leaves out.
**Teaches:** Asynchronous RL corrections (SAO, arXiv 2607.07508; OAPL, arXiv 2602.19362; ICML 2026 Stable Asynchrony); Constitutional AI and Deliberative Alignment; RL scaling studies; A Watermark for Large Language Models.
**Derivation:** none.
**Notebook:** none (reading).
**Figures:** none.
**Restate / forward:** none; nothing depends on this notebook.
**Exercises:** none.
**Cite:** as listed, each with its one-clause contribution.

## Chapter 9 Distillation and Reasoning

**Label:** `chap_lm_reasoning`.
**Hook:** The reader has a 1.7B model that solves the arithmetic family after RL and a 0.6B model that does not; the chapter moves capability into the smaller model and buys accuracy with test-time tokens.
**Route:** What can a student learn from the teacher's distribution (9.1)? Why does scoring the student's own samples help (9.2)? What do SFT, RL and distillation each change (9.3)? What does a step-level scorer see (9.4)? Why do more samples stop helping (9.5)? How is a reasoning budget set and trained in (9.6)?
**Scope:** Tool calls inside a trace are Chapter 10; monitorability is 1.6's paragraph; no single named efficient-reasoning method in prose.
**Resources:** On-Policy Distillation (9.2); Why We Think (9.5, 9.6); Reasoning from Scratch (9.3); Open R1 (9.3); PRM800K repository (9.4); gpt-oss model card, the effort levels (9.6).
**Further Reading notebook:** none.

### 9.1 Knowledge Distillation · T1 · REVISE · [Codex 10.1 to 10.3]
**Idea:** A student can learn from the teacher's distribution, not only its samples.
**Opening:** A student trained on the teacher's answers sees one token per position; the teacher's distribution holds a probability for every entry. Closing: what the samples never show.
**Teaches:** Sequence-level against token-level distillation at matched compute; top-k logit storage; the divergence table, forward KL, reverse KL, generalized JSD, each with its use.
**Derivation:** The bias of top-k truncated soft targets on a four-token vocabulary with k = 2.
**Notebook:** At matched compute, does token-level distillation beat sequence-level? Qwen3-1.7B into Qwen3-0.6B; 20,000 teacher samples vs top-32 logits on the same prompts. Check: accuracy with intervals and a compute ledger. Claim: the logit arm reaches the same accuracy at fewer tokens.
**Figures:** `fig_lm_kd_modes` (targets at one position); `fig_lm_kd_cost` (plot: accuracy vs compute).
**Restate / forward:** `subsec_mdl-distillation`; `sec_mdl-divergences-distances`. Forward: none.
**Exercises:** 6 to 8. Derive the top-k bias and the k that keeps it under 1%; measure a temperature sweep; open: distillation across tokenizers.
**Cite:** Distilling the Knowledge in a Neural Network; Sequence-Level KD.

### 9.2 On-Policy Distillation · T2 · KEEP · [Codex 10.4 to 10.7]
**Idea:** Scoring the student's own samples removes the compounding error of offline distillation.
**Opening:** The 9.1 student goes wrong at token 40 and the teacher's samples never show the way back; the teacher is the expert DAgger asks. Closing: how much cheaper than RL.
**Teaches:** The on-policy objective: sample from the student, score each token by the teacher's log-probability minus the student's, a dense reward with no critic; the cost ledger against RL.
**Derivation:** DAgger with a soft expert, the compounding bound from quadratic to linear.
**Notebook:** At matched generated tokens, how do offline distillation, on-policy distillation and GRPO compare? A Qwen3-0.6B LoRA student from A7's SFT checkpoint, a Qwen3-1.7B teacher. Check: accuracy with intervals; per-position KL. Claim: on-policy reaches the offline accuracy at fewer tokens. Hosts the mini-assignment.
**Figures:** `fig_lm_opd_toy` (a student path with teacher scores); `fig_lm_opd_tokens` (plot: accuracy against tokens).
**Restate / forward:** `subsec_il_dagger`; 9.1's divergence table. Forward: none.
**Exercises:** 6 to 8. Derive the DAgger bound for a soft expert; measure teacher size against student gain; open: on-policy distillation as the last stage after RL.
**Cite:** GKD; On-Policy Distillation.

### 9.3 Training Reasoning Models · T3 (T2 reduced) · KEEP · [Codex 10.2, 11.2]
**Idea:** A reasoning model is made by cold-start SFT, RL with verifiable rewards and distillation, and each stage changes something different.
**Opening:** After RL on checkable problems a base model's trace grows and accuracy rises; after SFT on stronger traces the format appears at once. Closing: unseen templates.
**Teaches:** The pipeline: cold-start SFT on a few thousand traces, RL with verifiable rewards, rejection-sampling SFT, distillation; RL from base and the behaviours that appear; prolonged RL and its plateau.
**Derivation:** none.
**Notebook:** At matched compute, does SFT on traces or RL from base generalize better to unseen templates? Qwen3-1.7B base; SFT on 4,000 traces from Qwen3.5-4B vs GRPO from base; T2 reduced. Check: seen and unseen accuracy with pass@k intervals. Claim: the seen-unseen gap differs between arms.
**Figures:** `fig_lm_reason_pipeline` (the stages); `fig_lm_reason_passk` (plot: seen vs unseen).
**Restate / forward:** 8.4's GRPO and verifier; 9.1's distillation; `sec_mdl-statistics`. Forward: none.
**Exercises:** 6 to 8. Derive the compute split of a three-stage pipeline; measure trace length against RL steps; open: rejection-sampling SFT.
**Cite:** DeepSeek-R1; ProRL; GSM-Symbolic.

### 9.4 Verifiers and Process Supervision · T2 · KEEP · [Codex 11.3]
**Idea:** A step-level scorer sees what an outcome scorer cannot, at the price of labels.
**Opening:** Two candidates reach the same wrong answer by different routes; an outcome scorer rates them alike, a step scorer finds the first wrong step. Closing: paying for the labels.
**Teaches:** Outcome against process reward models; Monte Carlo step labels from rollouts per prefix; generative verifiers that critique before a verdict; calibration across generators.
**Derivation:** The variance of a Monte Carlo step label against rollouts per step, on eight rollouts.
**Notebook:** On the same candidate pools, does step-level reranking beat outcome reranking, and does the gain survive a change of generator? Outcome and process heads on Qwen3-0.6B. Check: best-of-16 reranking with intervals. Claim: the process scorer's gain and its cross-generator drop.
**Figures:** `fig_lm_prm_labels` (step labels on a trace); `fig_lm_prm_rerank` (plot: best-of-n by scorer).
**Restate / forward:** 8.1's reward head; `subsec_mdl-paired-comparisons`. Forward: none.
**Exercises:** 6 to 8. Derive rollouts per step for a target label confidence; measure step aggregation (minimum, product, last); open: a generative verifier.
**Cite:** Let's Verify Step by Step; Math-Shepherd.

### 9.5 Parallel Test-Time Compute · T1 · KEEP · [Codex 11.4]
**Idea:** More samples help until they are correlated.
**Opening:** Sample 16 answers and take the majority, and accuracy rises; at 256 it stops rising while coverage keeps climbing. Closing: one longer trace instead.
**Teaches:** Self-consistency, best-of-n with a scorer, weighted voting; pass@k as coverage against deployable selection; correlated samples and why the majority saturates; accuracy against generated tokens.
**Derivation:** Majority accuracy for n independent samples at accuracy p (n = 3 gives 3p^2 minus 2p^3) and its erosion under a shared error mode.
**Notebook:** How does accuracy scale with generated tokens for majority vote, scorer reranking and the oracle? 9.3's 0.6B model, n up to 256. Check: intervals per n. Claim: the majority saturates below pass@k.
**Figures:** `fig_lm_tts_majority_theory` (plot: majority against n); `fig_lm_tts_frontier` (plot: frontier).
**Restate / forward:** 1.4's pass@k estimator; `sec_decoding`. Forward: none.
**Exercises:** 6 to 8. Derive the vote formula for n = 5 and the two-mode case; measure temperature against coverage; open: 64 small-model samples against one 4B answer.
**Cite:** Self-Consistency; Large Language Monkeys.

### 9.6 Sequential and Adaptive Test-Time Compute · T2 (T1 reduced) · MERGE(v5 9.6+9.7) · [Codex 10.6, 11.5 to 11.7]
**Idea:** A reasoning budget is a dial, set by effort level at inference or by a length penalty at training.
**Opening:** gpt-oss-20b at low effort answers in a few hundred tokens and at high effort in a few thousand, with accuracy differing on hard items only. Closing: a tool call mid-trace.
**Teaches:** Effort levels (gpt-oss, Qwen3.8's `reasoning_effort`) as deployed budget forcing; adaptive allocation by difficulty, fitted on validation; overthinking and length penalties in RL giving an accuracy-token frontier.
**Derivation:** Allocation as a knapsack over items, spend where marginal accuracy per token is highest.
**Notebook:** How do accuracy and tokens move with effort by difficulty, and does a length penalty in training shift the frontier? The effort sweep on gpt-oss-20b (T1); a length-penalized GRPO variant of 8.4's model (T2) with an allocation rule fitted on validation. Check: intervals per item. Claim: tokens rise with effort everywhere; accuracy on hard items only.
**Figures:** `fig_lm_seq_budget` (plot: accuracy and tokens by effort); `fig_lm_effic_frontier` (plot: frontier).
**Restate / forward:** 9.5's frontier; 8.4's reward; 1.1's effort cell. Forward: 10.3.
**Exercises:** 6 to 8. Derive the knapsack allocation and when a uniform budget is optimal; measure budget forcing against the effort dial; open: overthinking detection.
**Cite:** s1; Reasoning on a Budget (arXiv 2507.02076); adaptive allocation (arXiv 2602.01070).
## Chapter 10 Agents and Tool Use

**Label:** `chap_lm_agents` (5 sections, T0 to T3).
**Hook:** The reader has prompted and fine-tuned a model; now it must act: call a tool, run code, and keep going.
**Route:** How does a model ask for a tool (10.1)? How is an output kept parseable (10.2)? What makes an agent loop succeed (10.3)? How is an agent trained (10.4)? How is it judged and attacked (10.5)?
**Scope:** Left out: computer-use and multimodal agents (Image Models part), RL algorithms (Chapter 8), index construction (Chapter 2).
**Resources:** Berkeley LLM Agents course (10.3); HF Agents and MCP courses (10.1, 10.3); Anthropic, Building Effective Agents (10.3); mini-swe-agent (10.3); Outlines (10.2); verl (10.4).
**Further Reading notebook:** none.

### 10.1 Tool Calling · T1 · REVISE · [new]
**Idea:** A tool call is a structured turn in the chat template, and its accuracy is measured like any other output.
**Opening:** The chat model gets arithmetic wrong and cannot ask a calculator in parseable form. Closing: trusting a tool's text.
**Teaches:**
- Tool schemas in the template; parsing; the result as a tool turn; parallel calls.
- Synthetic traces kept only if they execute; Toolformer and Gorilla in one sentence.
- MCP (Agentic AI Foundation since Dec 2025); accuracy by name, arguments, execution.
**Derivation:** none.
**Notebook:** Does SFT on executed traces raise call accuracy on held-out tools? Qwen3-1.7B, three tools, 2,000 traces. Claim: seen tools gain beyond the interval.
**Figures:** `fig_lm_tool_turn` (call and result as turns); (plot) accuracy by tool.
**Restate / forward:** `subsec_il_sft`. Forward: 10.2.
**Exercises:** 5 to 8. Derive: parallel calls as one turn. Measure: distractor tools. Open: an MCP server.
**Cite:** Toolformer (2302.04761); AAIF announcement (Anthropic, 2025); Writing Tools for Agents.

### 10.2 Structured Outputs and Constrained Decoding · T0 · KEEP · [Codex 13.9]
**Idea:** A grammar becomes a token-level automaton, and masking logits with it changes the distribution.
**Opening:** Some of 10.1's calls fail to parse; the reader wants a decoder that cannot emit an invalid string. Closing: the format's cost.
**Teaches:**
- A JSON schema as a character automaton; a token is allowed if it reaches a live state.
- Tokens spanning several transitions or ending mid-string; the state tracked per token.
- Token-by-token renormalization is not sequence conditioning; masking after thinking.
**Derivation:** The token automaton and the masked distribution; case: vocabulary {a, ab}, language {ab}.
**Notebook:** Does a tokenizer-aware masker keep accuracy at 100% parse rate? Qwen3-0.6B on CPU, 10.1's schema, checked vs brute force. Claim: within the interval.
**Figures:** `fig_lm_token_automaton` (token edges); (plot) mask sizes.
**Restate / forward:** `sec_decoding`; `sec_text-sequence`. Forward: none.
**Exercises:** 5 to 8. Derive: the masked-vs-conditioned gap. Measure: mask precomputation. Open: masking after thinking.
**Cite:** Outlines (2307.09702); XGrammar (2411.15100).

### 10.3 Agent Loops and Harnesses · T1 · REVISE · [new]
**Idea:** An agent is a loop whose success depends on the interface it is given and the context it keeps.
**Opening:** On a task needing five calls in a row, 10.1's model redoes a finished step. Closing: model failure or harness failure.
**Teaches:**
- The loop (observe, think, act), code as the action space, the agent-computer interface.
- Long horizons: truncation, compaction, file memory, subagents; interleaved thinking.
- Agent Skills: a `SKILL.md` folder loaded by progressive disclosure; a failure taxonomy.
**Derivation:** none.
**Notebook:** Interface or compaction? A 100-line loop with a sandboxed shell, Qwen3.5-4B on 30 repair tasks, three arms. Claim: interface wins beyond the interval. Hosts A9.
**Figures:** `fig_lm_agent_loop` (loop, context, compaction, skills); (plot) success vs tokens.
**Restate / forward:** `sec_imitation`; `sec_mdp`. Forward: 10.5.
**Exercises:** 5 to 8. Derive: tokens per episode vs horizon. Measure: a test-running subagent. Open: a skill for the build.
**Cite:** ReAct (2210.03629); SWE-agent (2405.15793); Agent Skills standard (agentskills.io).

### 10.4 Training Agents with Reinforcement Learning · T3 (T1 reduced) · KEEP · [new]
**Idea:** In a multi-turn episode only the agent's tokens are actions, and the loss mask says so.
**Opening:** An agent's episode interleaves its tokens with tool outputs it never produced. Closing: reward for a late failure.
**Teaches:**
- The episode as an MDP with tool outputs as transitions; the gradient over agent tokens.
- Trajectory vs turn-level rewards; environments as code; throughput as the bottleneck.
- Search and software agents as shipped forms; task synthesis; hacking (editing the test).
**Derivation:** The multi-turn policy gradient with its mask; case: two turns, one call.
**Notebook:** Does RL as a search agent beat prompting and single-turn RAG? Qwen3-1.7B, GRPO over the 2.4 retriever (T1: 0.6B); masks unit-tested. Claim: beats prompting beyond the interval.
**Figures:** `fig_lm_multiturn_mask` (agent vs environment tokens); (plot) accuracy.
**Restate / forward:** `sec_rl_sequences`; `subsec_rl_sequences_hacking`; `tab_rl_notation`. Forward: none.
**Exercises:** 5 to 8. Derive: turn rewards as trajectory reward plus shaping. Measure: sandbox throughput. Open: synthetic tasks.
**Cite:** Search-R1 (2503.09516); SWE-RL (2502.18449); Kimi K2 (2507.20534).

### 10.5 Evaluating and Securing Agents · T1 · REVISE · [new]
**Idea:** An agent is judged by reliability and cost, and attacked through what it reads.
**Opening:** The 10.3 agent solves six tasks in ten once; a daily user wants the same task every time. Closing: a lasting defence.
**Teaches:**
- pass^k vs pass@k; cost-normalized scores; the time horizon at 50% success.
- Harness sensitivity, reproducible environments, leakage; names in a data file.
- Injection through tool outputs; a malicious `SKILL.md` as supply chain; defences measured.
**Derivation:** pass^k as the mean of p_i^k; case: rates 0.9 and 0.5 give pass^5 near 0.31, pass@5 near 0.98.
**Notebook:** How much does one defence cut attack success, at what cost? The 10.3 agent, five runs per task, 60 attacks, with and without a confirmation gate. Claim: beyond the interval.
**Figures:** `fig_lm_injection_paths` (tool output, page, skill file); (plot) pass^k vs k.
**Restate / forward:** `subsec_mdl-paired-comparisons`; 1.4's pass@k estimator. Forward: none.
**Exercises:** 5 to 8. Derive: the variance of pass^k. Measure: attack success vs model size. Open: an injection that passes.
**Cite:** Indirect prompt injection (2302.12173); AgentDojo (2406.13352); Terminal-Bench 2.0 (2601.11868).

## Chapter 11 Model Compression

**Label:** `chap_lm_compression` (4 sections plus a Further Reading notebook, T1).
**Hook:** In 7.5 the reader stored a frozen model in four bits and it ran no faster; a served model must be fast, and activations enter.
**Route:** What does quantization speed up (11.1)? Why do a few channels break the grid (11.2)? Why are rounding and zeroing one problem (11.3)? What does training through the quantizer recover (11.4)?
**Scope:** Left out: KV-cache compression (12.4), number formats (`sec_hardware`, `sec_numerics`), fine-tuning quantizers (7.5).
**Resources:** MIT 6.5940 (11.1 to 11.4); Grootendorst, A Visual Guide to Quantization (11.1); EMNLP 2025 efficient-inference tutorial (11.1 to 11.3); torchao (11.1, 11.4).
**Further Reading notebook:** `sec_lm_compression_fr`: codebooks and lattices, learned rotations, lossless exponent coding, structured pruning, k-bit laws.

### 11.1 Quantization for Inference · T1 · KEEP · [Codex 12.1, 12.2]
**Idea:** Decode speed tracks weight bytes, so weight-only quantization speeds decode while activation quantization speeds prefill.
**Opening:** The four-bit model from 7.5 was no faster in eager PyTorch; through a fused int4 kernel it is. Closing: int4 and prefill.
**Teaches:**
- Per-channel vs per-group scales; effective bits with scales (int4, group 128: 4.125).
- W4A16 vs W8A8 and fp8: dequantize in registers vs integer tensor cores.
- Why W8A8 does not help small-batch decode; bits vs parameters at fixed memory; GGUF.
**Derivation:** Decode time as the larger of compute and weight bytes over bandwidth; case: Qwen3-0.6B.
**Notebook:** Does decode speed follow weight bytes? Qwen3-1.7B: group sizes at int4, int8, W8A8 with torchao kernels at batch 1 and 32. Claim: ordered by bytes within about 10%.
**Figures:** `fig_lm_quant_granularity` (scale granularities); (plot) speed vs bits.
**Restate / forward:** `eq_hw_decode_bound`; 7.5's quantizers. Forward: 11.2.
**Exercises:** 5 to 8. Derive: effective bits under double quantization. Measure: where W8A8 overtakes W4A16. Open: the ladder.
**Cite:** Integer-arithmetic-only inference (1712.05877); LLM.int8() (Dettmers.Lewis.Shleifer.ea.2022).

### 11.2 Outliers, Rescaling and Rotations · T1 · MERGE(v5 11.2+11.4) · [Codex 12.3, 12.5]
**Idea:** Outlier channels break uniform grids, and a rescaling or a rotation spreads them.
**Opening:** Int8 activations make Qwen3-1.7B's perplexity jump though the weights survived; a few channels are huge. Closing: a free rotation.
**Teaches:**
- Outlier channels; LLM.int8()'s mixed decomposition (outlier columns stay in 16 bits).
- Rescaling X W = (X S^-1)(S W) and the balancing S; AWQ (scales found by search).
- Invariance under an orthogonal rotation; Hadamard gives max over RMS of O(sqrt(log d)).
**Derivation:** The rescaling identity and the concentration bound; case: (8, 0, 0, 0) under a 4-point Hadamard.
**Notebook:** How much W8A8 and W4A4 loss do rescaling and rotation recover? Qwen3-1.7B: round-to-nearest, SmoothQuant, AWQ, Hadamard. Claim: rotation closes most of the gap.
**Figures:** `fig_lm_outlier_channels` (magnitude by channel, before and after); (plot) perplexity.
**Restate / forward:** `sec_mdl-concentration-generalization`; `sec_transformer-block`. Forward: none.
**Exercises:** 5 to 8. Derive: the balancing S; the Hadamard transform in O(d log d). Measure: calibration size. A codebook.
**Cite:** SmoothQuant (2211.10438); AWQ (2306.00978); QuaRot (2404.00456).

### 11.3 Reconstruction-Based Quantization and Pruning · T1 · MERGE(v5 11.3+11.6) · [Codex 12.4, 12.7]
**Idea:** Rounding and zeroing are the same layer-wise reconstruction problem, solved column by column with the Hessian.
**Opening:** Rounding one column and compensating in the rest loses less than rounding all at once. Closing: whether per-layer is right.
**Teaches:**
- The layer objective with H = X^T X; the Optimal Brain Surgeon step.
- GPTQ: columns in order, lazy block updates, a Cholesky factor of H^-1, dampening.
- SparseGPT (target zero); Wanda (no update); 2:4 as the hardware pattern; depth pruning.
**Derivation:** The OBS step from a quadratic with one constraint; case: a 2x2 weight vs every grid point.
**Notebook:** How much does Hessian compensation recover at int4 and half sparsity? GPTQ vs brute force on 16x16, then Qwen3-1.7B; SparseGPT and Wanda at 2:4, timed. Claim: most of the gap.
**Figures:** `fig_lm_obs_update` (one weight fixed, compensation); (plot) perplexity.
**Restate / forward:** `sec_mdl-numerical-stability-conditioning`; 7.5's NF4 grid. Forward: none.
**Exercises:** 5 to 8. Derive: OBS removal equals GPTQ with target zero. Measure: 2:4 speed. Open: prune four layers.
**Cite:** GPTQ (2210.17323); SparseGPT (2301.00774); Wanda (2306.11695).

### 11.4 Quantization-Aware Training and Native Low-Bit Formats · T1 · REVISE (v5 11.5) · [Codex 12.6]
**Idea:** Training through the quantizer recovers what post-training quantization loses, and new models ship in block formats.
**Opening:** The int4 GPTQ model lost a point that no calibration recovers; a few hundred training steps do. Closing: train in the shipped format.
**Teaches:**
- Fake quantization forward, straight-through backward, learnable scales; recovery training.
- Quantization-aware distillation (frozen bf16 teacher, KL): Nemotron 3 Nano in NVFP4.
- Block formats from 4.3; gpt-oss ships MXFP4 experts; native on Blackwell; ternary weights.
**Derivation:** The straight-through estimator as the gradient of a surrogate, the identity inside the clip range.
**Notebook:** At matched compute, does QAT or QAD beat PTQ at int4 and simulated MXFP4? Qwen3-0.6B: GPTQ vs 500 steps of QAT and QAD; gpt-oss-20b's experts inspected. Claim: most of the gap.
**Figures:** `fig_lm_fake_quant` (quantizer forward, straight-through backward); (plot) perplexity.
**Restate / forward:** `sec_custom_layer`; 4.3's block-scaling error; `subsec_mdl-distillation`. Forward: none.
**Exercises:** 5 to 8. Derive: the bias of the straight-through gradient. Measure: learnable vs fixed scales. Open: ternary weights.
**Cite:** Straight-through estimator (Bengio.Leonard.Courville.2013); QAD for NVFP4 (2601.20088); gpt-oss (2508.10925).

### FR Further Reading: Compression · T0 · NEW · [Codex 12.5, 12.8]
**Idea:** The compression methods outside the teachable core, each with the one thing it adds.
**Opening:** A reader who has rotated, reconstructed and retrained wants the two-bit and lossless corners. Closing: the next format.
**Teaches:** Codebooks and lattices (QuIP#, AQLM: what a codebook buys below three bits); learned rotations (SpinQuant). Lossless exponent coding (DFloat11); structured pruning plus distillation (Minitron); k-bit inference scaling laws.
**Derivation:** none.
**Notebook:** none (reading).
**Figures:** none.
**Restate / forward:** 11.2's rotation and 11.3's reconstruction. Forward: none.
**Exercises:** none.
**Cite:** QuIP# (2402.04396); DFloat11 (2504.11651); k-bit inference scaling laws (2212.09720).

## Chapter 12 Efficient Inference

**Label:** `chap_lm_inference` (4 sections, T1 to T2).
**Hook:** The reader has a compressed model and a decode bound from Advanced, and the eager loop reaches a fraction of it.
**Route:** Where do the milliseconds of a decode step go (12.1)? Can a cheap model propose tokens without changing the output (12.2)? What makes a good drafter (12.3)? Which parts of the cache tolerate compression (12.4)?
**Scope:** Left out: scheduling across requests (Chapter 13), changes made before pretraining (Chapter 14), weight quantization (Chapter 11).
**Resources:** PyTorch, Accelerating Generative AI II (12.1); Horace He, Making Deep Learning Go Brrrr (12.1); Triton tutorials (12.1); vLLM speculative-config post, 2026-07-13 (12.2, 12.3).
**Further Reading notebook:** none.

### 12.1 A Fast Decoder on One GPU · T1 · REVISE · [Codex 13.1, 13.3]
**Idea:** A decode step is a stack of small kernels, and the gap to the bandwidth bound is overhead, unfused kernels and a cache read that does not fill the GPU.
**Opening:** `generate_cached` on Qwen3-0.6B runs at a fraction of the Advanced bound; the GPU idles between kernels. Closing: other requests.
**Teaches:**
- CPU overhead vs kernel time; static shapes, a preallocated cache, CUDA-graph replay.
- A fused int4 weight-only GEMV that dequantizes in registers.
- Split-KV decode attention with merged softmax statistics as a Triton kernel (over 30 lines).
**Derivation:** Speed as the larger of overhead and bytes over bandwidth; bandwidth falls when few blocks read the cache.
**Notebook:** How close to the bound does each change bring Qwen3-0.6B at batch 1 and 8K? Stages: eager, compiled, graphs, int4, split-KV; kernel checked. Claim: a stated fraction of the bound.
**Figures:** `fig_lm_decode_trace` (a step before and after graphs); `fig_lm_split_kv`; (plot) speed.
**Restate / forward:** `sec_kv-cache`; `subsec_comp-overhead`; `sec_attention-at-scale`. Forward: 13.2.
**Exercises:** 5 to 8. Derive: the splits that fill the SMs. Measure: speed vs context. Open: a fused RMSNorm plus GEMV.
**Cite:** gpt-fast (PyTorch blog); Triton (Tillet.Kung.Cox.2019); FlashInfer (2501.01005).

### 12.2 Speculative Decoding · T1 · REVISE · [Codex 13.7]
**Idea:** A cheap drafter proposes and the target verifies, and the output distribution is unchanged.
**Opening:** 12.1's decoder reads every weight per token while the next few are often obvious; a small model could guess them. Closing: a drafter's cost.
**Teaches:**
- Draft gamma tokens, verify in one pass, accept with min(1, p/q), resample from max(0, p - q).
- Expected accepted length vs the acceptance rate; the choice of gamma.
- A wall-clock model with draft cost and batch gamma+1; where it loses; n-gram drafting.
**Derivation:** Exactness and the expected accepted length (1 - alpha^(gamma+1)) / (1 - alpha); case: three symbols.
**Notebook:** Does measured speedup follow the wall-clock model? A toy matches p; Qwen3-0.6B drafts for Qwen3-4B: speedup vs gamma, n-gram baseline, batch 16 loses. Claim: within about 20%.
**Figures:** `fig_lm_spec_accept` (draft, accept or reject, resample); (plot) speedup vs gamma.
**Restate / forward:** transformers Further Reading; the `sec_kv-cache` verification exercise. Forward: 12.3.
**Exercises:** 5 to 8. Derive: the optimal gamma; exactness under top-p. Measure: acceptance by domain. Open: batching.
**Cite:** Leviathan.Kalman.Matias.2023; Chen.Borgeaud.Irving.ea.2023; MagicDec (2408.11049).

### 12.3 Drafters and Multi-Token Prediction · T2 (T1 reduced) · REVISE · [Codex 2.6, 13.8]
**Idea:** A drafter can be a head on the target's own features, and a multi-token head trained with the model serves both pretraining and drafting.
**Opening:** 12.2's drafter disagrees with the target in style; a head on the target's hidden state would know what comes next. Closing: what MTP changes.
**Teaches:**
- Medusa (Cai.Li.Geng.ea.2024; independent heads) and EAGLE, with EAGLE-3 worked in full.
- Multi-token prediction as a pretraining objective; the MTP module reused for drafting.
- Tree verification in a paragraph; 2026 pointers: P-EAGLE (2602.01469), DFlash (2602.06036).
**Derivation:** none (12.2's acceptance statistics restated).
**Notebook:** Does a feature-level head accept more than a separate drafter at equal cost? An EAGLE-style head for Qwen3-0.6B vs a two-layer drafter; Qwen3.5-2B's MTP head. Claim: higher acceptance.
**Figures:** `fig_lm_eagle_head` (draft head on target features; MTP); (plot) acceptance vs cost.
**Restate / forward:** 12.2's acceptance model; `sec_language-model`. Forward: 14.1.
**Exercises:** 5 to 8. Derive: tree vs chain verification cost. Measure: MTP acceptance on code vs prose. Open: an MTP head on 4.2.
**Cite:** EAGLE-3 (2503.01840); Multi-token prediction (Gloeckle.Idrissi.Roziere.ea.2024); DeepSeek-V3 (2412.19437).

### 12.4 KV-Cache Compression at Inference Time · T1 · REVISE · [Codex 13.5, 13.6]
**Idea:** Keys and values tolerate different compressions, and eviction must be judged on more than needle tests.
**Opening:** Qwen3-4B at 32K context carries about 4.5 GB of cache on 8 GB of weights. Closing: whether eviction drops what is needed later.
**Teaches:**
- Keys quantized per channel before RoPE, values per token; two bits with group scales.
- Eviction by accumulated attention plus a recent window; SnapKV as the prefill-only variant.
- Needle tests pass while aggregation and multi-hop tasks fail; RULER-style tasks plus QA.
**Derivation:** Quantization error of a key channel pair under the RoPE rotation; case: (10, 0.1) rotated by 45 degrees.
**Notebook:** On Qwen3-4B at 32K, which bit width and eviction budget keep accuracy at the smallest cache? Bits 16, 4, 2 and budgets 100%, 50%, 25% on RULER-style and QA tasks. Claim: four-bit loses nothing.
**Figures:** `fig_lm_kv_eviction` (window, heavy hitters, evicted tokens); (plot) accuracy vs bytes.
**Restate / forward:** `sec_kv-cache`; `subsec_kv-cache-sinks`; `sec_positional-information`. Forward: 13.3.
**Exercises:** 5 to 8. Derive: bytes per token per scheme. Measure: eviction with sinks. Open: a multi-hop task.
**Cite:** KIVI (2402.02750); H2O (Zhang.Sheng.Zhou.ea.2023); SnapKV (2404.14469).

## Chapter 13 Serving Systems

**Label:** `chap_lm_serving` (4 sections, T1 to T3).
**Hook:** The reader has 12.1's fast decoder for one request; a hundred users want it at once, and each one is waiting.
**Route:** What does a server promise, and where is the knee (13.1)? How does per-iteration scheduling keep the GPU full (13.2)? How does cache memory stop fragmenting (13.3)? How are prefill and decode split across GPUs (13.4)?
**Scope:** Left out: running vLLM or SGLang as a user (`sec_model_serving`), rental cost (`sec_cloud_instances`), training parallelism (Chapter 6).
**Resources:** Inside vLLM (13.2, 13.3); LMSYS, large-scale expert parallelism with disaggregation (13.4); ICML 2024 serving tutorial; NVIDIA Dynamo documentation (13.4); LMCache (13.4).
**Further Reading notebook:** none.

### 13.1 Workloads and Service Objectives · T1 · REVISE · [Codex 14.1]
**Idea:** A serving system is judged by goodput under objectives, and queueing theory predicts the knee.
**Opening:** Two clients send requests to 12.1's decoder and the second waits; at ten clients the wait dominates. Closing: keeping the knee away.
**Teaches:**
- Workloads: arrivals, prompt and output lengths from a trace; open- vs closed-loop load.
- TTFT, ITL, throughput; goodput as requests per second meeting both objectives.
- Little's law and the M/G/1 mean wait as the first model of the knee; exported graphs.
**Derivation:** Little's law and the M/G/1 wait lambda E[S^2] / (2 (1 - rho)); case: utilization 0.8.
**Notebook:** At what load does P99 TTFT cross the objective for static batching and a reference server? A Poisson generator vs 12.1's decoder and vLLM on Qwen3-0.6B. Claim: the knee within one step.
**Figures:** `fig_lm_goodput` (latency vs load, goodput region); (plot) P99 TTFT vs load.
**Restate / forward:** `sec_model_serving` (`tab_serving_workloads`); `subsec_kv-cache-prefill-decode`. Forward: 13.2.
**Exercises:** 5 to 8. Derive: goodput vs batch size under two objectives. Measure: closed-loop load. Open: replay a public trace.
**Cite:** DistServe (2401.09670); Sarathi-Serve (Agrawal.Kedia.Panwar.ea.2024).

### 13.2 Continuous Batching and Chunked Prefill · T1 · REVISE · [Codex 14.2]
**Idea:** Scheduling per iteration keeps the GPU full, and chunked prefill keeps decode latency flat.
**Opening:** The static batch waits for its longest output while finished rows sit idle. Closing: what determinism costs when batches change.
**Teaches:**
- Iteration-level scheduling: rows at different positions, admission, completion, the queue.
- A long prefill stalls every decode row; chunked prefill under a token budget per step.
- Preemption (recompute vs swap); batch-invariant kernels for reproducible outputs.
**Derivation:** The idle fraction of static batching; case: rows of 10, 20, 40 and 80 tokens leave 53% idle.
**Notebook:** On one trace, what do continuous batching and chunked prefill do to throughput and ITL? A 150-line scheduler around 12.1's decoder vs static batching and vLLM; batch-invariant mode checked. Hosts A10.
**Figures:** `fig_lm_continuous_batching` (rows joining, prefill chunks); (plot) throughput, P99 ITL.
**Restate / forward:** `sec_model_serving` (`fig_tools_continuous_batching`); `subsec_kv-cache-batching`. Forward: 13.3.
**Exercises:** 5 to 8. Derive: the token budget that holds ITL under a limit. Measure: recompute vs swap. Open: two-tenant priorities.
**Cite:** Orca (Yu.Jeong.Kim.ea.2022); Sarathi-Serve (Agrawal.Kedia.Panwar.ea.2024); Defeating Nondeterminism (2025).

### 13.3 Paged KV Memory and Prefix Caching · T1 · REVISE · [Codex 14.3, 14.4]
**Idea:** Fixed-size blocks remove fragmentation, and shared prefixes are shared blocks.
**Opening:** 13.2's scheduler reserves each cache at maximum length and runs out at a quarter of the possible batch. Closing: writing a shared block.
**Teaches:**
- Block tables: logical to physical blocks, allocation on demand, the kernel reading the table.
- Fragmentation: internal waste bounded by one block per sequence, external waste gone.
- Prefix sharing: block hashes, a radix tree, copy-on-write, prefix-aware routing.
**Derivation:** Waste bounds for reservation vs blocks; case: 64 requests, maximum 4K, mean 500, block 16.
**Notebook:** How much batch does paging buy, and how much TTFT does prefix caching save? A block allocator with invariant tests, a page-aware kernel, a shared 1K prompt. Claim: the derived factor within 20%.
**Figures:** `fig_lm_block_table` (shared blocks, reference counts); (plot) batch vs block size.
**Restate / forward:** `sec_model_serving` (`fig_tools_prefix_cache`); `sec_kv-cache`; `sec_hybrids`. Forward: 13.4.
**Exercises:** 5 to 8. Derive: the block size balancing waste and overhead. Measure: radix-tree hit rate. Also: hybrid state; multi-LoRA.
**Cite:** PagedAttention (Kwon.Li.Zhuang.ea.2023); SGLang (Zheng.Yin.Xie.ea.2024); S-LoRA (2311.03285, exercise).

### 13.4 Serving across GPUs and Nodes · T3 (T2 pair) · REVISE · [Codex 14.5]
**Idea:** Prefill and decode want different parallelism, so they are split across pools and joined by a KV transfer.
**Opening:** One GPU holds Qwen3-30B-A3B only at 8 bits, and prefills stall its decode; eight GPUs offer several splits. Closing: the transfer's cost.
**Teaches:**
- Tensor vs expert parallelism for inference: all-reduce per layer vs all-to-all; wide EP.
- Disaggregation: a prefill pool at high TP, a decode pool at high batch; NIXL transfer.
- KV stores across replicas (Mooncake); prefix-local routing; crossovers in rate and length.
**Derivation:** Transfer time as latency plus KV bytes over bandwidth vs the stall removed; case: NVLink vs PCIe.
**Notebook:** Where are the TP vs EP and colocated vs disaggregated crossovers for Qwen3-30B-A3B? vLLM on 8x A100 (T2 pair: two GPUs at 8 bits); two prefill plus six decode with NIXL. Claim: within 30%.
**Figures:** `fig_lm_disaggregation` (pools, transfer, router); (plot) goodput vs load.
**Restate / forward:** 6.4's all-to-all bytes per token; `subsec_hw-two-workloads`; 5.3's per-link costs. Forward: none.
**Exercises:** 5 to 8. Derive: the prefill-to-decode ratio from a trace. Measure: TTFT under TP 2 vs TP 4. Open: prefix-local routing.
**Cite:** DistServe (2401.09670); Mooncake (2407.00079); NVIDIA Dynamo disaggregated-serving documentation.

## Chapter 14 Advanced Architectures

**Label:** `chap_lm_architectures` (7 sections plus a Further Reading notebook, T0 to T3).
**Hook:** The reader has the transformer of Advanced and a configuration file from 1.1 with fields that do not match it: a latent, expert counts, windows.
**Route:** What does a configuration file determine (14.1)? How sparse should a model be (14.2)? How is the cache shrunk in training (14.3)? Where does unneeded softmax mass go (14.4)? Can a model learn which blocks to read (14.5)? What does a hybrid cost to serve (14.6)? Why does depth dilute layers (14.7)?
**Scope:** Left out: the transformer block, MoE basics, SSM and DeltaNet layers (Advanced); multimodal encoders (Image Models part); the release calendar.
**Resources:** Raschka, The Big LLM Architecture Comparison (14.1); Smol Training Playbook (14.2, 14.4); How to Scale Your Model (14.2, 14.6); flash-linear-attention (14.6); nanochat (capstone).
**Further Reading notebook:** `sec_lm_architectures_fr`: release calendar, byte patching, diffusion language models, depth aggregation, zero-computation experts, test-time training, the design checklist.

### 14.1 Architecture Configurations · T0 · REVISE · [Codex 16.1, 16.3]
**Idea:** A configuration file determines parameters, active compute and cache bytes per token, and a technical report is read against it.
**Opening:** The configuration files of Qwen3-0.6B and Gemma 4 E2B do not yield their cards' counts from the obvious fields. Closing: what a report omits.
**Teaches:**
- The design axes, one per section: mixing, KV layout, feed-forward, lookup memory, residuals, positions.
- `tab_modern-recipe` extended and automated from configuration files; canonical here.
- Reading a report; DeepSeek's line V2 (MLA), V3 (MoE, MTP, fp8), V3.2 (indexer), V4 (mHC, Engram).
**Derivation:** Total, active, FLOPs and cache bytes per token from the fields; case: Qwen3-0.6B vs its safetensors index.
**Notebook:** Do the derived counts match the safetensors indexes? About a dozen configurations (Qwen3 ladder, Qwen3.5, Qwen3.6, Qwen3.8, Gemma 4, DeepSeek V4-Flash, SmolLM3, gpt-oss, Kimi K3). Claim: exact for dense.
**Figures:** `fig_lm_design_axes` (a block annotated with the axes); (plot) cache bytes vs active parameters.
**Restate / forward:** `tab_modern-recipe` in `sec_scaling-laws`; `sec_kv-cache` (`kv_cache_bytes`); `sec_moe`. Forward: none.
**Exercises:** 5 to 8. Derive: cache bytes per token for a hybrid with MLA. Measure: indexes that disagree. Open: a report schema.
**Cite:** Qwen3 (Qwen.Team.2025); DeepSeek-V3 (2412.19437); Gemma 4 model card (ai.google.dev, Apache-2.0).

### 14.2 Mixture-of-Experts Design at Scale · T2 · REVISE · [Codex 3.2, 3.8]
**Idea:** Sparsity is a design axis with its own scaling law, and the router is co-designed with the network.
**Opening:** The `sec_moe` model picked 8 experts of 64 by hand; released models choose 256 small experts, 8 active, one shared. Closing: lookup memory.
**Teaches:**
- Total vs active parameters and granularity; the fitted law and the optimum at fixed FLOPs.
- Router design in released models: sigmoid scores, normalized top-k, node-limited routing.
- All-to-all bytes per token; LatentMoE (2601.18089); Engram's hashed n-gram lookup memory.
**Derivation:** Optimal sparsity at fixed FLOPs; all-to-all bytes per token with and without a latent; case: hidden 4096, top-8, EP 64.
**Notebook:** At fixed active compute, how does loss move with total parameters and granularity? A sweep of the `sec_moe` GPT at 138M active (T2); Qwen3-30B-A3B's routing by domain. Claim: an exponent to one digit.
**Figures:** `fig_lm_moe_axes` (sparsity axes, Engram); (plot) loss vs total.
**Restate / forward:** `sec_moe`; `sec_scaling-laws`; 6.4. Forward: none.
**Exercises:** 5 to 8. Derive: the inference cost model. Measure: load imbalance vs skew. Open: zero-computation experts.
**Cite:** DeepSeekMoE (Dai.Deng.Zhao.ea.2024); Fine-grained MoE laws (Krajewski.Ludziejewski.Adamczewski.ea.2024); Engram (2601.07372).

### 14.3 Latent Attention and Cross-Layer KV Sharing · T1 · KEEP · [Codex 3.4, 2.5]
**Idea:** A latent KV can be absorbed into the projections at decode, if RoPE is decoupled.
**Opening:** The GQA cache still stores eight heads per layer; one small latent per token would cut it but costs compute. Closing: converting a GQA model.
**Teaches:**
- MLA: a down-projection to a latent, up-projections for K and V; absorption into Q and O.
- Why RoPE blocks absorption and the decoupled rotary key.
- Cross-layer sharing (one cache across adjacent layers); GQA conversion by SVD.
**Derivation:** The absorption identity and why a rotation between the factors breaks it; case: two heads, dimension 8, latent 4.
**Notebook:** Do the explicit and absorbed paths agree, and what does the latent buy vs GQA? MLA in `d2l.GPT`'s block, both forms; MLA and GQA trained (T1); Qwen3-0.6B converted by SVD. Claim: paths equal in bf16.
**Figures:** `fig_lm_mla_absorb` (explicit vs absorbed flow, decoupled path); (plot) decode latency vs context.
**Restate / forward:** `subsec_kv-cache-low-rank`; `sec_mdl-svd-low-rank`. Forward: none.
**Exercises:** 5 to 8. Derive: FLOPs per token for both paths. Measure: latent dimension vs loss. Open: one latent across two layers.
**Cite:** DeepSeek-V2 (2405.04434); Cross-Layer Attention (2405.12981); MHA2MLA (2502.14837).

### 14.4 Softmax Attention Refinements · T2 · REVISE · [new]
**Idea:** A softmax head must put its mass somewhere, and the fixes (sinks, gates, windows, partial positions) each trade something measurable.
**Opening:** Most heads of Qwen3-0.6B dump mass on the first token; Gemma 4 E2B sees 512 tokens in most layers and works at 128K. Closing: which fixes last.
**Teaches:**
- Sinks: mass with nowhere to go lands on a tiny-value token; massive activations; gates.
- Partial RoPE and interleaved NoPE layers (SmolLM3-3B) beyond training length.
- Local-global layouts as a KV dial (Gemma 4: 512 and 1024 windows); per-layer embeddings.
**Derivation:** A learned sink logit s equals one extra key of logit s and zero value; case: logits (2, 1, 0), sink 3.
**Notebook:** What do a learned sink, a gated output and partial RoPE with NoPE layers change? Four arms at 138M (T2): loss, first-token mass, maximum activation, length generalization. Claim: sink mass falls beyond noise.
**Figures:** `fig_lm_sink_gate` (a sink token vs an output gate); (plot) sink mass.
**Restate / forward:** `sec_what-attention-computes`; `subsec_kv-cache-sinks`; `sec_positional-information`. Forward: none.
**Exercises:** 5 to 8. Derive: cache bytes per token for ratio r and window w. Measure: window size vs loss. Open: per-layer embeddings.
**Cite:** StreamingLLM (Xiao.Tian.Chen.ea.2024); Massive Activations (Sun.Chen.Kolter.ea.2024); Gated Attention (Qiu.Wang.Zheng.ea.2025).

### 14.5 Learned Sparse Attention · T1 · KEEP · [Codex 3.5]
**Idea:** A learned indexer selects which blocks a query reads, and training and inference must agree on the selection.
**Opening:** 12.4's eviction threw tokens away for good; a model could keep everything and learn which blocks each query needs. Closing: selection's cost.
**Teaches:**
- Block selection: block summaries scored vs the query, top-k blocks plus a window (NSA).
- A learned indexer matching full attention's block mass (DSA); what Quest misses.
- Train-inference consistency with one selection kernel; scoring cost linear in blocks.
**Derivation:** FLOPs per query for full attention vs scoring plus k blocks; case: 64K context, block 64, k 16.
**Notebook:** Does window plus top-k blocks match full attention on retrieval, and what does selection cost? A small model (T1), block-sparse vs full, planted retrieval at 8K and 16K, selection profiled. Claim: matched.
**Figures:** `fig_lm_block_select` (query, block summaries, selected blocks, window); (plot) accuracy vs context.
**Restate / forward:** `sec_attention-at-scale` (FlexAttention); 12.4's eviction. Forward: none.
**Exercises:** 5 to 8. Derive: the block size balancing scoring and reading. Measure: a post-hoc indexer. Open: 14.3's latent.
**Cite:** Native Sparse Attention (Yuan.Gao.Dai.ea.2025); DeepSeek-V3.2 (2512.02556); Quest (2406.10774).

### 14.6 Hybrid Models at Scale · T3 (T1 load) · REVISE · [Codex 16.4, 3.7]
**Idea:** A released hybrid is the Advanced recipe (one attention layer in a few) at scale, and its serving cost follows from the attention fraction.
**Opening:** The `sec_hybrids` model had two layers; Qwen3.5-2B has gated DeltaNet in three positions of four, gated attention in the fourth. Closing: the state's limits.
**Teaches:**
- The released recipe: where the attention layers sit, what the chunked kernels do in practice.
- `d2l.Qwen35Hybrid`, from `d2l.GatedDeltaNet` and `d2l.Qwen3Attention` (as in 1.2).
- Relatives: Qwen3.6 and Qwen3.8, Kimi Linear, Nemotron 3 (Mamba-2 plus MoE), Mamba-3.
**Derivation:** Cache bytes at context L for attention fraction f: f L k_KV plus (1 - f) times the state; case: 128K.
**Notebook:** Does the book's class reproduce Qwen3.5-2B, and how do memory and throughput scale with context? (T1) logits matched in `d2l.Qwen35Hybrid`; (T3) Qwen3.6-35B-A3B vs Qwen3-30B-A3B. Claim: memory ratio within 10%.
**Figures:** `fig_lm_hybrid_layout` (layer stack, state vs cache); (plot) memory vs context.
**Restate / forward:** `sec_hybrids` (`fig_hy-cache`); `sec_deltanet`. Forward: none.
**Exercises:** 5 to 8. Derive: the context where hybrid memory falls below half. Measure: recall at 32K. Open: change the fraction.
**Cite:** Gated DeltaNet (2412.06464); Kimi Linear (2510.26692); Mamba-3 (2603.15569).

### 14.7 The Residual Stream across Depth · T2 · REVISE · [new]
**Idea:** Pre-norm dilutes each layer's contribution with depth, and learned mixing across streams or layers restores it only under a constraint.
**Opening:** Hidden-state norm in Qwen3-0.6B grows layer by layer while updates stay the same size, so late layers matter less. Closing: depth or width.
**Teaches:**
- Norm growth under pre-norm and the dilution of a layer's contribution; post-norm.
- Hyper-connections: streams mixed by learned matrices; unconstrained products blow up; mHC.
- Attention Residuals (a layer attends over earlier outputs; Kimi K3); DenseFormer as a pointer.
**Derivation:** Doubly stochastic matrices are closed under products, have spectral norm at most one and fix the uniform vector.
**Notebook:** Does the constraint widen the stable learning-rate range and flatten the norm profile? Pre-norm vs mHC with four streams, Attention Residuals optional, 138M (T2), three rates. Claim: a flatter profile.
**Figures:** `fig_lm_residual_mixing` (one stream vs several with mixing matrices); (plot) norm vs depth.
**Restate / forward:** `sec_transformer-block`; `sec_init_param`; 1.6's residual stream. Forward: none.
**Exercises:** 5 to 8. Derive: norm growth for independent vs aligned updates; Sinkhorn. Measure: streams vs traffic. Open: depth.
**Cite:** Hyper-Connections (2409.19606); mHC (2512.24880); Attention Residuals (2603.15031).

### FR Further Reading: Architectures · T0 · NEW · [Codex 2.6, 2.8, 16.5, 16.6]
**Idea:** The reading notebook holds what dates fastest, the release calendar and the designs not yet in the core, plus the checklist that hosts the capstone.
**Opening:** A reader who has worked seven axes wants to know what else released models tried, and what to try with a budget. Closing: the next axis.
**Teaches:** The open-weight release calendar (names and dates from a data file, no prose claims); byte-level patching (BLT, H-Net). Diffusion language models (LLaDA2.0; a text-scale example requested from `sec_diffusion-discrete` once that chapter settles). DenseFormer and value residuals; zero-computation experts (LongCat-Flash); test-time training and memory layers (Titans; Memory Layers). The design-under-a-budget checklist (task, data, compute, hardware, serving objective), hosting the capstone: one measured baseline and one rejected design with evidence.
**Derivation:** none.
**Notebook:** none (reading).
**Figures:** none.
**Restate / forward:** 14.1's table as the data source. Forward: none.
**Exercises:** the capstone only.
**Cite:** LLaDA2.0 (2512.15745); Byte Latent Transformer (2412.09871); Titans (Behrouz.Zhong.Mirrokni.2025).
## Changes outside this part

Retiring-chapter labels cited from surviving parts (every `:numref:` scanned on 5 November 2026) and the edits that travel with the deletion.

| Item | Cited at | Action |
|---|---|---|
| `chap_nlp_pretrain`, `chap_nlp_app` | preface index.md:303, encoders-decoders.md:361, scaling-laws.md:498 | Re-point to `chap_lm_encoders`, `chap_lm_pretraining`; reword encoders-decoders.md:361 |
| `subsec_negative-sampling`, `sec_word2vec_data` | mdl-mi-estimation.md:1230-1247 | Defined anew in 2.1 |
| `sec_seq2seq`, `sec_machine_translation`, `sec_encoder-decoder`, `sec_seq2seq_training` | attention-scoring.md:655, lstm.md:369, decoding.md:29, text-sequence.md:633 | Move `seq2seq.md` into `chapter_recurrent-modern`, labels preserved (handoff line 107); delete the stale `seq2seq.qmd` there |
| `DATA_HUB['ptb']` | word-embedding-dataset.md:76 only; `d2l.PennTreebank` (lstm.md:786) needs it | Move the registration into lstm.md |
| `sec_sentiment_cnn` | why-conv.md:274 | Cite 2.2's sentence on 1-D convolutions over tokens, or drop |
| `sec_bert-pretraining` | cloud-instances.md:212 | Re-point the exercise to 2.2 (T1 fine-tuning) |
| `sec_word2vec_pretraining` | none (pandas.md:253 is prose) | Optional pointer from `sec_pandas` to `subsec_lm_categorical_embeddings` |
| Library and build | `_quarto.yml`; `CHAPTER_NUMBERING` `[18]`, `[19]` in `tools/d2l_preprocess.py` (dict order equals `_quarto.yml` order); `tools/runtime_env.py:298`; `tools/build_lib.py:34`; `tools/output_noise_allow.txt:44-47`; `outputs/*/chapter_natural-language-processing-*` | Clean in the same change; rewrite `chapter_language-models/index.md` as the part intro; fix "Chapters 17-19" in `legacy-multigpu-lib.md:24`; rename `d2l.generate` to `d2l.generate_ids`, mark `generate_cached` `#@save` in `sec_kv-cache`, add `d2l.qk_clip_` to `sec_muon` and a Mamba-3 pointer to `chap_modern_rnn` FR. `sec_fine_tuning`, `sec_utils` and the Tools sections are unchanged |

**Other changes:** (1) retire the two NLP chapters; (2) request a text-scale masked-diffusion example from `sec_diffusion-discrete` after its rewrite; (3) multimodal models stay in the Image Models part; (4) scheduler extension for T2 to T4 and `d2l.launch`; (5) Advanced forward references cite the `chap_lm_*` labels of the contract table.

## v5 sections not carried over

No topic is dropped: merged sections keep their content, demoted ones become exercises or FR pointers. 4.3 → 4.2 subsection; 5.3 and 5.4 → 5.3; 5.5 and 5.6 → 5.4; 6.1 and 6.2 → 6.1; 6.3 and 6.5 → 6.2; 7.5 → 7.4; 7.7 and 7.8 → 7.6; 8.4 → 8.3; 8.7 → split into 8.6 and 8.7; 9.7 → 9.6; 11.4 → 11.2, codebooks to Chapter 11 FR; 11.6 → 11.3, DFloat11 to Chapter 11 FR; 13.5 → 13.3 exercises; 14.8 → 12.3, byte patching and diffusion LMs to Chapter 14 FR; 14.9 → Chapter 14 FR.

**Codex units not carried over.** v5's table stands with two rows updated: 14.8 → 13.1 for service objectives, operation stays in `sec_model_serving`; 2.8 → a text-scale example requested from `sec_diffusion-discrete`, diffusion LMs in Chapter 14 FR, multi-token prediction in 12.3.
