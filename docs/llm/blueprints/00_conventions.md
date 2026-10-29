# Blueprint Conventions for the Language Models Part

This file is the shared brief for every chapter blueprint in `blueprints/`. It covers four things:

- what a blueprint is for;
- the inputs a blueprint writer must read;
- the template each section follows;
- the names (labels, shared `d2l` functions, artifacts) that chapters must agree on.

## Purpose

A chapter blueprint is the specification from which an authoring agent writes the chapter's notebooks: D2L Markdown sources with executable PyTorch cells. It should be detailed enough that the author makes no decisions about:

- content and its order;
- derivations;
- code structure and tests;
- experiments, figures, exercises or sources.

The author still writes the prose and the code. The blueprint fixes what the prose says and what the code does.

## Inputs (read before writing)

1. **`language_models_part_plan_v5.md`** is authoritative for scope, numbering, tiers and sources. Read your chapter, then these part-wide sections:
   - Teaching approach;
   - Threads from the rest of the book;
   - Contract with the rest of the book;
   - What this part does not re-teach;
   - Compute tiers;
   - Running examples;
   - Conventions;
   - State of the art;
   - References policy.
2. **`language_models_part_plan.md`** (the Codex plan) holds extra bibliography for the Codex units named in each section's bracket.
3. **The book**, `smolix/d2l-neu` (the `chapter_*/` sources). Read every section listed under "Builds on" before you plan, to learn what it already teaches, its notation, and its `#@save` code. A section here may recap a topic owned elsewhere in at most one paragraph with a `:numref:`, and must not re-derive or re-implement it.
4. **Style references** in the book: `chapter_transformers/kv-cache.md`, `chapter_computational-performance/hardware.md`, `chapter_recurrent-modern/hybrids.md`. They show the source format:
   - cells as ```` ```{.python .input #cell-id} ````, each starting with `%%tab pytorch`;
   - `#@save` for library code;
   - `:label:`, `:numref:`, `:cite:` and `:eqlabel:` for cross-references;
   - `d2l.DATA_HUB` for downloads;
   - `d2l.Benchmark` for timing.

## Ground rules

- **PyTorch only.** Cells carry `%%tab pytorch`.
- **Compute tiers** (v5, "Compute tiers"):

  | Tier | Hardware | Budget per notebook |
  |---|---|---|
  | T0 | CPU | minutes |
  | T1 | one 24 GB Ada GPU (RTX 4090 class), the readers' floor. The build box (4× RTX 4090 in one AMD host) runs two notebooks per GPU at ≤ 15 GiB each; a notebook that needs more (the judge below) claims a whole GPU. Its GPUs have no NVLink and unreliable peer-to-peer transfers (driver issues on AMD hosts), so no multi-GPU experiment runs there | ≤ ~15 min |
  | T2 | one A100 40 GB (PCIe). Two-GPU experiments (tensor or pipeline parallelism over two GPUs, a prefill/decode split, NCCL bandwidth) run on two A100 40 GB joined by an NVLink bridge, written **T2 pair** | ≤ a few hours |
  | T3 | one 8× A100 40 GB node. Access is limited: each T3 experiment runs once and commits its logs. The node's interconnect (NVSwitch or NVLink-bridged PCIe pairs) is recorded (`nvidia-smi topo -m`) and decides which rows apply | ≤ ~2 h |
  | T4 | 2 or more nodes of 8× A100, for a handful of runs | ≤ ~1 h; the notebook analyzes committed logs |

  Every T2+ section specifies a reduced T1 configuration of the same code, behind a config flag, and lists the artifacts it commits.
- **Hardware facts that constrain the plans:**
  - A100 has no FP8 tensor cores and runs neither FlashAttention-3 nor native MXFP4/NVFP4.
  - FP8 runs on the Ada build box.
  - MXFP4 and NVFP4 are simulated.
  - The A100 40 GB PCIe has 40 GB of HBM2 at about 1.6 TB/s, the same dense bf16 rate as the SXM part, and links to its partner over an NVLink bridge; other traffic goes over PCIe Gen4. 5.1's `chip_specs` row carries the datasheet values.
- **Experiments sized to the claim.** Run the smallest experiment that supports the stated conclusion: no extra arms, seeds or model sizes because compute allows. T3 and T4 runs are few; each section lists its T3/T4 runs and the claim each one supports.
- **Software.** `torch==2.15.0` with Python 3.12 for every notebook (the book's pin moves with it; README, book-side edits). Serving engines pin their own `torch` (vLLM 0.32.0 needs 2.13.0, SGLang 0.5.22 needs 2.14.1), so they run in separate environments and notebooks talk to them over HTTP; 7.7 records each environment.
- **Licences.** The book's text is CC BY-SA 4.0 and its code MIT-0. Data the book writes follows suit: prose under CC BY-SA 4.0, code and test suites under MIT-0. Publicly available third-party data under a permissive licence (Apache-2.0, MIT, CC-BY, ODC-By and similar) may be shipped with attribution. Data that is non-commercial, unlicensed or unclear is downloaded by the notebook from its source and never committed.
- **No invented numbers.**
  - Expected outcomes are stated as a direction, or as a magnitude taken from a cited source ("the paper reports …"), or marked "to be measured".
  - Hardware specifications cite a datasheet or vendor page.
- **No conclusions from noise.** Every headline experiment states its noise check: seeds, a paired comparison, or a bootstrap interval from 1.5.
- **Notation.** Use the book's notation. Chapters 8–10 use `tab_rl_notation_contract` (in `sec_rl_sequences`) verbatim.
- **References.**
  - Keep every source v5 lists for a section, annotated with what to take from it.
  - New sources are welcome, but each must be verified:
    - **arXiv papers:** fetch `https://arxiv.org/abs/<id>` and match the title. Space requests at least 3 seconds apart, and do not use the arXiv export API, which is rate-limited.
    - **Other URLs:** must return HTTP 200.
    - **Hugging Face models and datasets:** check `https://huggingface.co/api/models/<id>` or `https://huggingface.co/api/datasets/<id>`.
  - Mark anything you could not verify with "(unverified)".
- **Output.**
  - Write only your own chapter file. Do not edit the plans or the book.
  - Aim for about 150–300 lines per section: dense and specific, not padded.

## Chapter file template

```
# Chapter N: Title — Blueprint

**Label:** `chap_lm_…` · **Arc:** … · **Sections:** … · **Tiers:** …

## Chapter overview
- Draft intro paragraph (what the chapter does and why, 120–200 words).
- Learning objectives (measurable; refine v5's).
- Prerequisites: book labels and earlier sections of this part, each with what is used.
- Section dependency graph (text form).
- Running examples and shared artifacts used and produced.
- Compute plan: per-section tier, GPU-hours, committed artifacts.

## Assignment (if the chapter has one)
Tasks; starter code; deliverables; autograder checks with exact tolerances; written-analysis questions; T1 path; estimated student hours.

## Teaching sources
Annotated list (keep all of v5's).

## Sections
(one block per section, template below)
```

## Section template

```
### N.M Title

- **Label:** `sec_lm_…` · **Tier:** … (reduced T1 config: …) · **Budget:** wall-clock and peak memory per tier · **Codex units:** …
- **Builds on (book):** each label with what is reused, and what must not be re-taught.
- **Builds on (this part):** sections, plus the artifacts and `d2l` names imported.
- **Exports:** `#@save` functions/classes with signatures and one-line contracts; data artifacts with their `DATA_HUB` keys and formats.
- **Models and data:** exact Hugging Face IDs with sizes, dataset IDs/subsets/splits, and licenses.
- **Packages:** with minimum versions where it matters.

#### Narrative arc
1. The question: the opening cell and the measured symptom it prints.
2. Subsections in order (proposed headings). Under each: 3–6 bullets naming the concept, equations (with `eq_lm_…` labels), code cells, and figures.

#### Derivations
For each: a precise statement, the assumptions, numbered proof steps, the final formula, and the cell that checks it numerically.

#### From-scratch implementation
Components with signatures; a line budget (≤ ~150 lines in total); numerical tests (reference, tolerance, seed).

#### In production
Library, APIs and versions; how the from-scratch code serves as the test oracle.

#### Headline experiment
Hypothesis; setup (model, data, steps, batch, learning rate, seeds, tier); procedure; metrics; plot specification (axes, series, labels); expected outcome with its source; noise check; T1 reduction; artifacts committed for T2+.

#### Figures and tables
`fig_lm_…` and `tab_lm_…` labels, each with its contents and the cell that makes it.

#### Pitfalls and misconceptions

#### Summary (draft bullets, 4–6)

#### Exercises
Two check, two derive, two measure, one open. Write each out in full; give an answer sketch for check and derive.

#### Further reading
Annotated: at least one paper, one tutorial/course/blog, and one code base. Include all of v5's sources for the section.

#### Risks and open decisions
Each risk with a fallback (model availability, kernel support on Ada or A100, licenses, data access).
```

## Labels

- Chapter labels are given in v5.
- Section labels are fixed by the registry below.
- New figure, table and equation labels follow `fig_lm_<slug>`, `tab_lm_<slug>` and `eq_lm_<slug>`, with a section-specific prefix so they cannot collide (for example `fig_lm_rag_failures`).
- 2.1 also defines the legacy labels `sec_word2vec_data` and `subsec_negative-sampling`, which the book cites.

| § | Label | § | Label | § | Label |
|---|---|---|---|---|---|
| 1.1 | `sec_lm_hf_models` | 6.1 | `sec_lm_multinode_jobs` | 10.1 | `sec_lm_tool_calling` |
| 1.2 | `sec_lm_d2l_checkpoint` | 6.2 | `sec_lm_fsdp` | 10.2 | `sec_lm_structured_outputs` |
| 1.3 | `sec_lm_prompting` | 6.3 | `sec_lm_tensor_parallel` | 10.3 | `sec_lm_agent_loops` |
| 1.4 | `sec_lm_scoring` | 6.4 | `sec_lm_pipeline_parallel` | 10.4 | `sec_lm_agentic_rl` |
| 1.5 | `sec_lm_eval_suites` | 6.5 | `sec_lm_context_parallel` | 10.5 | `sec_lm_agent_eval` |
| 1.6 | `sec_lm_interpretability` | 6.6 | `sec_lm_expert_parallel` | 11.1 | `sec_lm_quant_inference` |
| 1.7 | `sec_lm_api` | 6.7 | `sec_lm_composing_parallelism` | 11.2 | `sec_lm_outliers` |
| 2.1 | `sec_lm_word_embeddings` | 6.8 | `sec_lm_long_runs` | 11.3 | `sec_lm_gptq` |
| 2.2 | `sec_lm_bert` | 7.1 | `sec_lm_full_finetuning` | 11.4 | `sec_lm_rotations` |
| 2.3 | `sec_lm_text_embeddings` | 7.2 | `sec_lm_sft` | 11.5 | `sec_lm_qat` |
| 2.4 | `sec_lm_retrieval` | 7.3 | `sec_lm_instruction_data` | 11.6 | `sec_lm_pruning` |
| 2.5 | `sec_lm_rag` | 7.4 | `sec_lm_lora` | 12.1 | `sec_lm_fast_decoder` |
| 3.1 | `sec_lm_crawl` | 7.5 | `sec_lm_peft` | 12.2 | `sec_lm_speculative` |
| 3.2 | `sec_lm_quality_filtering` | 7.6 | `sec_lm_qlora` | 12.3 | `sec_lm_drafters` |
| 3.3 | `sec_lm_dedup` | 7.7 | `sec_lm_ft_toolchains` | 12.4 | `sec_lm_kv_compression` |
| 3.4 | `sec_lm_data_mixtures` | 7.8 | `sec_lm_ft_multi_gpu` | 13.1 | `sec_lm_serving_workloads` |
| 3.5 | `sec_lm_synthetic_data` | 7.9 | `sec_lm_merging` | 13.2 | `sec_lm_continuous_batching` |
| 3.6 | `sec_lm_tokenizers` | 8.1 | `sec_lm_reward_models` | 13.3 | `sec_lm_paged_kv` |
| 4.1 | `sec_lm_data_path` | 8.2 | `sec_lm_rlhf_ppo` | 13.4 | `sec_lm_distributed_serving` |
| 4.2 | `sec_lm_reference_model` | 8.3 | `sec_lm_dpo` | 13.5 | `sec_lm_multi_model_serving` |
| 4.3 | `sec_lm_instabilities` | 8.4 | `sec_lm_po_family` | 14.1 | `sec_lm_reading_architectures` |
| 4.4 | `sec_lm_low_precision` | 8.5 | `sec_lm_rlvr` | 14.2 | `sec_lm_moe_design` |
| 4.5 | `sec_lm_continued_pretraining` | 8.6 | `sec_lm_rl_systems` | 14.3 | `sec_lm_mla` |
| 4.6 | `sec_lm_long_context` | 8.7 | `sec_lm_overoptimization` | 14.4 | `sec_lm_attention_refinements` |
| 4.7 | `sec_lm_budgets` | 9.1 | `sec_lm_distillation` | 14.5 | `sec_lm_sparse_attention` |
| 5.1 | `sec_lm_accelerator_design` | 9.2 | `sec_lm_onpolicy_distillation` | 14.6 | `sec_lm_hybrids_at_scale` |
| 5.2 | `sec_lm_workload_hardware` | 9.3 | `sec_lm_reasoning_training` | 14.7 | `sec_lm_residual_stream` |
| 5.3 | `sec_lm_memory_tiers` | 9.4 | `sec_lm_verifiers` | 14.8 | `sec_lm_beyond_tokens` |
| 5.4 | `sec_lm_server_paths` | 9.5 | `sec_lm_parallel_tts` | 14.9 | `sec_lm_design_budget` |
| 5.5 | `sec_lm_networks` | 9.6 | `sec_lm_sequential_tts` | |  |
| 5.6 | `sec_lm_scale_up_out` | 9.7 | `sec_lm_efficient_reasoning` | |  |

## Shared names reserved across chapters

The owning section defines each name below with `#@save`; every other section imports it under exactly this name. A blueprint may add further `#@save` names, but they must be unique, and each must be listed under its section's **Exports**.

| Name | Owner | Contract |
|---|---|---|
| `d2l.load_hf(model_id, revision=None, dtype=torch.bfloat16, device_map='auto')` | 1.1 | Returns `(model, tokenizer)` from Transformers, pinned to a revision. |
| `d2l.config_memory(config, seq_len, batch_size, dtype_bytes=2)` | 1.1 | Bytes for weights, KV cache, recurrent state and activations, from a Hugging Face config. |
| `d2l.load_qwen3(model_id='Qwen/Qwen3-0.6B', revision=None)` | 1.2 | Returns a `d2l.Qwen3GPT` with Qwen3 weights (R1). |
| `d2l.Qwen3Attention`, `d2l.SwiGLU`, `d2l.Qwen3GPT` | 1.2 | Subclasses of the book's `GQAAttention`, `FeedForward` and `GPT` with Qwen3's knobs (`head_dim`, QK-norm, `rope_theta`, explicit FFN width, `num_kv_heads`, untied head); the book's classes are unchanged. |
| `d2l.generate(model, tokenizer, prompts, max_new_tokens, **sampling)` | 1.2 | The book's sampler, for any causal LM returning logits. |
| `d2l.score_choices(model, tokenizer, prompts, choices, norm='token')` | 1.4 | Log-likelihood scoring under an explicit convention. |
| `d2l.pass_at_k(n, c, k)` | 1.4 | Unbiased pass@k estimator. |
| `d2l.EvalSuite`, `d2l.paired_bootstrap(scores_a, scores_b, n_boot=10000, cluster=None)` | 1.5 | The shared harness; paired differences with confidence intervals. |
| `d2l.judge_pairwise(judge, tokenizer, question, answer_a, answer_b)` | 1.5 | Judges both orders and reports position-consistent preference. |
| `d2l.ChatAPI(base_url, model, api_key=None, *, cache=None, mode='replay', price=None, cost_cap_usd=None)`, `d2l.chat_batch(api, requests, *, concurrency=8)`, `d2l.api_cost(usage, price)` | 1.7 | Client for any OpenAI-compatible endpoint with a request-hash response cache (`mode` is `'replay'`, `'record'` or `'live'`), bounded concurrency with retry, and cost from the `usage` block. |
| `d2l.book_corpus(chunking='sections')` | 2.4 | R3: chunks of the book with ids, headings and text. |
| `d2l.BookRetriever` | 2.4 | BM25, dense and hybrid retrieval over R3, with an optional reranker; `.search(query, k)`. |
| `d2l.book_qa(split='dev')` | 2.5 | The book QA set: questions, gold passage ids, reference answers. |
| `d2l.PackedStream` | 4.1 | Resumable stream of packed token batches with document ids for masks. |
| `d2l.load_reference_model(step=None)` | 4.2 | R2 checkpoint from `DATA_HUB`. |
| `d2l.chip_specs()` | 5.1 | Accelerator specification table (DataFrame) with sources. |
| `d2l.launch(script, nodes=1, gpus_per_node=8, **env)` | 6.1 | Submits a `%%writefile` script with torchrun/SLURM and collects logs. |
| `d2l.LoRALinear` | 7.4 | LoRA wrapper for `nn.Linear` with merge/unmerge. |
| `d2l.quantize_blockwise`, `d2l.quantize_nf4`, `d2l.dequantize` | 7.6 | Weight quantizers reused by Chapter 11. |
| `d2l.FastGenerator` | 12.1 | Static-cache, CUDA-graph decoder reused by 12.2–12.4 and Chapter 13. |
| `d2l.config_stats(config)` | 14.1 | Parameters, active parameters, FLOPs per token and cache bytes per token from a config. |

## Models (default choices)

| Use | Model |
|---|---|
| Current hybrid model through Transformers | `Qwen/Qwen3.5-2B` (T1), `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-0.8B` |
| R1, rebuilt from scratch | `Qwen/Qwen3-0.6B` |
| Workhorse models | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, and their `-Base` variants |
| Judge (1.5, 2.5 and A2, 3.5, 7.3, 7.9, 8.x) | `Qwen/Qwen3.5-9B` (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`) in bf16, text-only: `AutoModelForCausalLM` builds `Qwen3_5ForCausalLM` and skips the vision tower and the MTP head, 17.9 GB of weights (vLLM: `--language-model-only`). Non-thinking (`enable_thinking=False`, which renders an empty think block), greedy. Only 8 of its 32 layers use full attention, so its KV cache is 32 KB per token. It needs a whole 24 GB GPU. Smaller fallback: `Qwen/Qwen3.5-4B` in bf16 |
| Helpers (generation inside a system under test, rewriting, paraphrasing, authoring-time checks) | `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-2B` in bf16, text-only, non-thinking; `Qwen/Qwen3.8-27B` (bf16, tensor-parallel on a T2 pair, or through an API) for authoring-time checks. Exception: 7.3's Magpie generator stays `Qwen/Qwen3-1.7B` (rules below) |
| Mixture of experts | `Qwen/Qwen3-30B-A3B` |
| Encoders and embedders | `answerdotai/ModernBERT-base`, `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B` |
| Native MXFP4 | `openai/gpt-oss-20b` |
| Fully open checkpoints | `allenai/OLMo-2-1124-7B` |
| Models too large to run locally | reached through an OpenAI-compatible API (1.7); notebooks cache every response, so the build replays them at T0 |

Pin revisions in the notebooks. A blueprint may choose otherwise, with a reason.

Which roles need different models:

- **Different models** wherever one model's output is graded, filtered, selected or labelled by another: answers and their judge; a test-set filter, the drafter and the judge; a data generator and the scorer that selects its data; training data and the judge that evaluates the trained model; the training reward and the evaluation; a text producer and its checker. Prefer a different family where outputs are compared pairwise (1.5, 7.3, 8.x); grading against a gold reference (2.5, 7.9) tolerates the same family.
- **The same model** where a role measures that model (2.5's closed-book probe is the generator) or where weights are shared by construction (policy and reference in 8.x; LoRA arms on one base in 7.9).
- **The same tokenizer** where two models exchange logits or token ids: teacher and student in logit distillation (9.1, 11.6), draft and target in speculative decoding (12.2). Qwen3 (vocabulary 151,936) and Qwen3.5 (248,320) cannot be mixed in such a pair.
- **Free to share** for stages of one system under test (2.5's query rewriting, HyDE and contextual headers use its generator).

No section loads a model in 4-bit just to fit memory: it uses a smaller Qwen3.5 model in bf16 instead. 4-bit formats appear only where quantization is the subject (7.6, Chapter 11, 12.1).
