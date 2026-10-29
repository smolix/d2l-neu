# Language Models Part: Revised Plan (v5)

**Dive into Deep Learning: replacement for the current Language Models part**
**Revised 28 October 2026**

**What changed from v4.**

- **Chapter 1 starts with Hugging Face.**
  - The new 1.1 runs a current model, Qwen3.5, through Transformers. It covers the Hub, revisions, licenses, chat templates, thinking modes and configuration files.
  - `d2l.GPT` follows as 1.2, so every later section starts from a current model.
  - 1.6 is renamed Interpretability, and its notebook adds released Qwen-Scope SAEs.
- **Chapter 2 separates encoders from embeddings and gives RAG its own section.**
  - 2.2 and 2.3 now state their boundary. 2.2 covers masked-token encoders: one vector per token, fine-tuned per task. 2.3 covers contrastive embeddings: one vector per text, used without fine-tuning.
  - The two share a headline experiment: 2.2's encoder makes a poor retriever until 2.3 trains it contrastively.
  - Retrieval (2.4) and RAG (2.5) are separate sections. RAG has a long notebook in six parts.
- **Chapter 5 is rebuilt as Hardware for Training and Inference.** It now covers:
  - accelerator design trade-offs (5.1);
  - why training, prefill and decode need different hardware (5.2);
  - every memory and storage tier from SRAM to disk (5.3);
  - data paths inside a server (5.4) and between servers (5.5);
  - scale-up vs scale-out, with power and cost (5.6).

  Reliability and checkpoint intervals move to 6.8.
- **Quantization is introduced where it is first needed.** 7.6 teaches quantizers (absmax, blockwise and NF4) before QLoRA uses them. 11.1 builds on 7.6 for inference.
- **Chapter 14 is now Advanced Architectures.**
  - It replaces Frontier Models with one section per design axis: MoE design, latent attention, refinements of softmax attention, learned sparse attention, hybrids, residual streams, and alternatives to fixed tokens.
  - A design table built from configuration files is refreshed each edition.
  - MLA and learned sparse attention move here from Chapter 12.
  - The OLMo checkpoint study moves to 4.2, and the DeepSeek case study becomes the worked example of 14.1.
- **Assignments.** A new A5 (a performance model of a node and a cluster) belongs to Chapter 5. The later assignments are renumbered A6–A8.
- **References.**
  - Sections that moved keep all their sources.
  - 55 new arXiv papers and 45 other pages (documentation, courses, blogs, code and vendor material) were added. All were checked on 28 Oct 2026: arXiv titles against the abstract pages, other URLs for HTTP 200.

**What changed from v3 to v4.**

- **Multimodal models moved out.** v3's multimodal chapter (contrastive image–text encoders, vision–language models, speech and audio models) moves to the later Image Models part; see "Changes outside this part".
- **Interpretability is core.** It is no longer optional, and its notebook covers lenses, probes, activation patching, sparse autoencoders and steering.

**What changed from v2 to v3.**

- **Scope of the rest of the book.** Only Basics, Advanced and Mathematics for Deep Learning are assumed to precede this part. Dependencies on the NLP chapters and the Tools appendix are removed, and the material this part needs from them is taught here. The Image Models part comes after this one and is not a prerequisite.
- **New Chapter 2, Encoders, Embeddings and Retrieval.** It covers word vectors, BERT and its descendants, text embeddings and RAG briefly, and replaces the retiring NLP Pretraining and NLP Applications chapters.
- **New Chapter 10, Agents and Tool Use.**
- **New section 1.5** (now 1.6). Interpretability tools.
- **Teaching apparatus.** Learning objectives, assignments and course paths.
- **References.** Every chapter lists teaching sources (courses, tutorials, blogs, code) alongside papers.

**Inputs and checks.**

- `language_models_part_plan.md` (Codex) stays the source bibliography.
- `language_models_part_review.md` explains the restructuring.
- Book labels were checked against the book source, `smolix/d2l-neu` (23 Oct 2026, and again on 28 Oct 2026 for v5). Every label under "Builds on" lies in Basics, Advanced or Mathematics.

## Goal

The part teaches how modern language models are built, scaled across machines, adapted, aligned, extended to tools and multi-step tasks, compressed and served, and how current architectures depart from the transformer built in Advanced. It starts where Advanced stops. By then a reader can:

- build a GPT with RoPE, GQA, RMSNorm and mixture-of-experts layers (`sec_gpt`, `sec_kv-cache`, `sec_moe`);
- train it with AdamW or Muon under a WSD schedule (`sec_adamw`, `sec_muon`, `sec_scheduler`);
- reason about rooflines, KV caches and data-parallel training (`sec_perf_model`, `sec_hardware`, `sec_multi_gpu`);
- derive policy gradients, PPO and the KL-regularized optimum (`chap_reinforcement_learning`, `chap_deep_rl`).

Every section here is either new material or applies those tools at a scale where new problems appear.

## Size

- **Total:** 14 chapters and 90 sections. Basics has 58 sections and Advanced has 72. The hard cap for this part is 130 (Basics plus Advanced).
- **Optional (◇):** 5 sections, the first to cut. Without them the core is 85 sections.
- **Headroom:** 40 sections remain under the cap. They are a reserve for authoring, not a target. Chapter 14 is expected to draw on them as new architectures appear.

The part is organized in seven arcs, so a course can take whole arcs:

| Arc | Chapters | Question it answers |
|---|---|---|
| I. Using and measuring | 1 Pretrained LMs, 2 Encoders, Embeddings and Retrieval | What does a trained model do, and how do we measure it? |
| II. Building a base model | 3 Data, 4 Pretraining | How is a base model made? |
| III. Scaling out | 5 Hardware for Training and Inference, 6 Parallelism Beyond One Node | What hardware do training and serving need, and how does training run on many machines? |
| IV. Post-training | 7 Fine-Tuning, 8 Preference Optimization and RL, 9 Distillation and Reasoning | How does a base model become useful? |
| V. Acting | 10 Agents and Tool Use | How does a model use tools and act over many steps? |
| VI. Deploying | 11 Compression, 12 Efficient Inference, 13 Serving Systems | How is it run cheaply and fast? |
| VII. Architecture | 14 Advanced Architectures | Which architectural changes do current models make, and what does each buy? |

## Teaching approach

The part targets an advanced undergraduate or first-year graduate course at a research university. The prerequisites are Basics, plus the Advanced chapters on attention, transformers, optimization, computational performance and reinforcement learning. The Mathematics part is a reference that sections cite by label.

### Section template

Every section follows D2L's pattern, tightened for an upper-level course:

1. **The question.** What problem appears at this scale, with one measured symptom.
2. **Mechanism and derivation.** The central result stated precisely and derived where a derivation exists. Each section entry below names its derivation on a **Derivation:** line.
3. **From scratch.** An implementation in at most ~150 lines, checked against a reference by a numerical test (logits, gradients, distributions or invariants).
4. **In production.** The same thing in the library practitioners use (PyTorch distributed, TorchTitan, PEFT/TRL, vLLM/SGLang, torchao), with the from-scratch version as the test oracle.
5. **One headline experiment.** A single seeded run, with no conclusion drawn from differences within run-to-run noise.
6. **Summary.**
7. **Exercises** of four kinds:
   - *check*: conceptual;
   - *derive*: extend or prove a result;
   - *measure*: change the notebook and predict before running;
   - *open*: a mini-project with no reference answer.
8. **Further reading.** At least one primary paper, one tutorial, course or blog, and one code base.

### Chapter apparatus

Every chapter opens with learning objectives and closes with an assignment. Assignments follow the style of graded projects in strong LM courses: each has a machine-checkable target (logits within tolerance, loss within a band, throughput above a floor, a passing invariant test) and a written analysis. Every assignment has a T1 path that fits a 16 GB GPU, so it can be taught without a cluster.

### Course paths

| Course | Weeks | Sections |
|---|---:|---|
| **Large Language Models** (graduate, one semester) | 13 | 1.1–1.6; 2.3–2.5; 3.2–3.3; 4.1–4.3, 4.6; 5.2; 6.2–6.4; 7.2, 7.4, 7.6; 8.1, 8.3, 8.5; 9.2, 9.3, 9.5; 10.1, 10.3; 11.1, 11.3; 12.1–12.2; 13.2–13.3; 14.1–14.2, 14.6 |
| **Systems for Large Models** | 13 | 1.4; 4.1, 4.4; Chapters 5–6; 7.8; 8.6; 10.4; Chapters 11–13; 14.3, 14.5 |
| **Post-Training and Alignment** | 10 | 1.1–1.5; Chapters 7–9; 10.1–10.5 |
| **NLP with Pretrained Models** (replaces the retired NLP chapters) | 8 | 1.1–1.5; Chapter 2; 7.1–7.4 |
| **Model Architecture** (advanced seminar) | 8 | 1.1–1.2, 1.6; 4.2–4.3; 5.2; Chapter 14 |

### Assignments

| # | Assignment | Chapters | Tier | Checkable target |
|---|---|---|---|---|
| A1 | Run Qwen3.5 through Transformers; load Qwen3 into `d2l.GPT`; build the evaluation harness with error bars | 1 | T1 | Logits match Transformers within BF16 tolerance; harness reproduces published scores within their confidence interval |
| A2 | RAG system over the book | 2 | T1 | Recall@10 above a stated floor; answer accuracy and citation precision above the 2.5 baseline, with paired-bootstrap intervals |
| A3 | Data pipeline: extraction, MinHash deduplication, quality classifier | 3 | T0–T1 | Deduplication F1 on labeled pairs; manifest reproducible from hashes |
| A4 | Pretrain a small model with packing, document masks and WSD | 4 | T1 (T2 full) | Validation loss within a band at a fixed token budget |
| A5 | Performance model of a node and a cluster | 5 | T1 (build box), T2–T3 full | Predicted training step time, time to first token and decode throughput within 15% of measurement |
| A6 | Tensor- and pipeline-parallel blocks verified against one GPU | 6 | T1 (2 GPUs), T3 full | Outputs and gradients match; measured bubble within 10% of the simulator |
| A7 | SFT → DPO → GRPO on Qwen3-0.6B | 7–8 | T1–T2 | Win rate and verifiable accuracy above a baseline at a fixed budget |
| A8 | Inference engine: paged KV, continuous batching, speculative decoding | 12–13 | T1 | Throughput and P99 latency targets on a fixed trace; bit-exact greedy outputs |
| Capstone | Design under a budget | 14 | Any | One measured baseline and one rejected design, with evidence |

## Threads from the rest of the book

The part continues threads that the surviving parts start. The table shows where each thread is picked up.

| Thread | Starts in | Continues in |
|---|---|---|
| `d2l.GPT` and loading released weights | `sec_gpt`, `sec_transformer-block`, `sec_read_write` | 1.1–1.2 (R1), 4.2 (R2), 12.1, 14.6 |
| Tokenization | `sec_text-sequence` | 3.6, 14.8 |
| Induction heads and attention circuits | `sec_what-attention-computes` | 1.3, 1.6 |
| Bidirectional encoders and masked-token objectives | `sec_transformer` | 2.2, 2.3 |
| Attention at scale and KV caches | `sec_attention-at-scale`, `sec_kv-cache` | 4.6, 6.5, 12.4, 13.3, 14.3–14.5 |
| Positional encodings | `sec_positional-information` | 4.6, 14.4 |
| Mixture of experts | `sec_moe` | 6.6, 13.4, 14.2 |
| Optimizers, schedules, muP | `sec_adamw`, `sec_muon`, `sec_scheduler`, `sec_scaling` | 4.2, 4.3, 4.5 |
| Scaling laws | `sec_scaling-laws` | 4.7, 8.5, 9.5, 14.2 |
| Performance models and hardware | `sec_perf_model`, `sec_hardware`, `sec_compilation` | Chapter 5, 11.1, 12.1 |
| Memory, precision and number formats | `sec_memory_precision`, `sec_numerics` | 4.4, 5.2, 7.6, 11.1 |
| Collectives and data parallelism | `sec_multi_gpu`, `sec_multi_gpu_concise` | Chapters 5 and 6 |
| Policy gradients, PPO, KL regularization | `sec_policygradient`, `sec_baselines`, `sec_ppo`, `sec_regularized` | Chapter 8 |
| Token-level MDP and the RL notation contract | `sec_rl_sequences` (`tab_rl_notation_contract`) | Chapters 8, 9, 10 |
| Imitation and compounding error | `sec_imitation` | 7.2, 9.2, 10.3 |
| Custom gradients and the straight-through estimator | `sec_custom_layer` | 11.5 |
| Recurrent and hybrid layers | `chap_modern_rnn` | 1.1, 13.3, 14.6 |
| Residual streams and normalization | `sec_transformer-block`, `sec_init_param` | 4.3, 14.7 |
| Masked diffusion | `sec_diffusion-discrete` | 14.8 |
| SVD and low rank | `sec_mdl-svd-low-rank` | 2.1, 7.4, 7.5, 11.4, 14.3 |
| PMI and InfoNCE | `subsec_mdl-pmi`, `sec_mdl-infonce` | 2.1, 2.3 |
| Divergences and coding | `sec_mdl-divergences-distances`, `sec_mdl-information_theory` | 8.2, 9.1, 9.2 |
| Duality | `sec_mdl-constrained-optimization-duality` | 8.3 |
| Statistics and the bootstrap | `sec_mdl-statistics` | 1.5, 2.5, 10.5 |
| Numerical conditioning | `sec_mdl-numerical-stability-conditioning` | 11.3 |

## Contract with the rest of the book

The table covers two kinds of commitment:

- topics that Advanced explicitly defers to this part;
- labels in retiring parts that Basics, Advanced or Mathematics still cite.

The cited labels must be re-pointed to the section of this part listed in the last column.

| Promised or cited in | Promise or target | Delivered in |
|---|---|---|
| `chap_performance` intro, `sec_multi_gpu`, `sec_multi_gpu_concise` | Multi-node training; tensor, pipeline and expert parallelism; network fabrics | Chapters 5 and 6 |
| `chap_performance` intro, `sec_hardware`, `sec_kv-cache` | Serving engines: continuous batching, paged KV, speculative decoding | 12.2, 13.2, 13.3 |
| `chap_performance` intro | Quantization as compression for inference | 7.6 (quantizers), Chapter 11 |
| `sec_fast_transformer` | FP8 training recipe | 4.4 |
| `chap_reinforcement_learning`, `sec_rl_sequences` | DPO derivation; post-training at language-model scale | 8.3, Chapter 8 |
| `sec_rl_sequences` (`tab_rl_notation_contract`) | RL notation, used verbatim | Chapters 8, 9, 10 |
| `sec_scaling-laws` | Data curation, instruction tuning, learning from feedback | Chapter 3, 7.2–7.3, Chapter 8 |
| `sec_hybrids` | Pretrained hybrid stacks | 13.3, 14.6 |
| `sec_transformer`, `sec_scaling-laws` → `chap_nlp_pretrain` | Pretrained encoders; pretraining at scale | `chap_lm_encoders` (Chapter 2), `chap_lm_pretraining` (Chapter 4) |
| `sec_mdl-mutual-information` → `subsec_negative-sampling`, `sec_word2vec_data` | Skip-gram with negative sampling; the 0.75-power noise distribution | 2.1 keeps both labels |
| `sec_scaling-laws` → `sec_word2vec_data` | Penn Treebank as a word-level corpus | 2.1 (same corpus) |
| `sec_pandas` → `sec_word2vec_pretraining` | "Embeddings for categorical columns" | 2.1 |
| `sec_parameters`, `sec_read_write` → `sec_fine_tuning` (8 references across 4 frameworks) | When fine-tuning helps and how far to unfreeze | 7.1 |
| Advanced (5 files) → `sec_training_systems` | Scaling ladder, library landscape, keeping a long run alive | 6.1, 6.7, 6.8 |
| `sec_hardware`, `chap_performance` → `sec_hardware_buyers` | Buying hardware | 5.6 covers cost and owned vs rented clusters; 5.1–5.2 cover how to choose accelerators for training and for serving. Desktop and unified-memory buying advice has no home here (see "Changes outside this part"). |

## What this part does not re-teach

**Rule:** a section may recap a topic owned elsewhere in at most one paragraph with a `:numref:`. It must not re-derive or re-implement it.

| Topic | Owner |
|---|---|
| Byte-level BPE, pre-tokenization, special tokens | `sec_text-sequence` |
| n-grams, perplexity, bits per byte | `sec_language-model` |
| Greedy, beam, temperature, top-k/top-p/min-p decoding | `sec_decoding` |
| Floating-point formats incl. FP8/FP4 layouts, loss scaling | `sec_numerics`, `sec_hardware` |
| Parameter/optimizer memory, activation checkpointing, freezing parameters | `sec_parameters`, `sec_use_gpu`, `sec_memory_precision`, `sec_adamw` |
| safetensors, run checkpoints, seeds | `sec_read_write`, `sec_repro` |
| Initialization and residual scaling | `sec_init_param` |
| AdamW, Muon, WSD and branching schedules, critical batch size, muP, clipping, EMA | `sec_adamw`, `sec_muon`, `sec_scheduler`, `sec_batch_size`, `sec_scaling`, `sec_practice` |
| RoPE, ALiBi, position interpolation, online softmax/FlashAttention, sliding windows, linear attention | `sec_positional-information`, `sec_attention-at-scale` |
| Induction heads, attention circuits | `sec_what-attention-computes` |
| RMSNorm, pre-norm, SwiGLU, QK-norm, GPT from scratch, loading GPT-2 | `sec_transformer-block`, `sec_gpt` |
| Encoder-only/decoder-only/encoder–decoder wiring; a bidirectional encoder; the masked-token objective; cross-attention and Perceiver-style latents | `sec_transformer` |
| KV cache, prefill vs decode, MQA/GQA, low-rank KV sketch, attention sinks | `sec_kv-cache` |
| MoE layer, balancing, fine-grained and shared experts | `sec_moe` |
| 6ND, Chinchilla fit, miniature scaling study | `sec_scaling-laws` |
| SSMs, Mamba, DeltaNet, hybrid stacks | `chap_modern_rnn` |
| Roofline, profiling, the memory hierarchy with its bandwidth and latency ladders, the shoreline argument, tensor cores, the accelerator specification table, energy, the build box's PCIe topology, the prefill/decode roofline | `sec_perf_model`, `sec_hardware` |
| torch.compile, CUDA graphs | `sec_compilation` |
| Custom autograd functions, the straight-through estimator | `sec_custom_layer` |
| Collectives, ring all-reduce, α–β model, DDP, FSDP/ZeRO derivation | `sec_multi_gpu`, `sec_multi_gpu_concise` |
| REINFORCE, baselines, RLOO, PPO, GAE | `sec_policygradient`, `sec_baselines`, `sec_ppo`, `sec_actorcritic` |
| Bradley–Terry reward model, KL-regularized optimum, Goodhart curves | `sec_regularized` |
| Token-level MDP, GRPO formula, verifier toy problem | `sec_rl_sequences` |
| Behavior cloning, compounding error, DAgger | `sec_imitation` |
| Masked-diffusion ELBO | `sec_diffusion-discrete` |
| KL and f-divergences, cross-entropy, coding, soft-target distillation | `sec_mdl-divergences-distances`, `sec_mdl-information_theory` |
| PMI, InfoNCE and its bound; the SGNS-as-shifted-PMI exercise | `subsec_mdl-pmi`, `sec_mdl-infonce`, `sec_mdl-mutual-information` |
| SVD, Eckart–Young, LoRA parameter count, PiSSA idea | `sec_mdl-svd-low-rank` |
| Lagrangian duality | `sec_mdl-constrained-optimization-duality` |
| Importance sampling | `sec_mdl-bayesian-computation` |
| Bootstrap, confidence intervals, tests | `sec_mdl-statistics` |

## Compute tiers

| Tier | Hardware | Budget per notebook | How it runs |
|---|---|---|---|
| T0 | CPU | Minutes | Any build host |
| T1 | Build box: ≤ 2 slots on one Ada GPU (≤ 15 GiB), or 2×1/2×2 | ≤ ~15 min | Existing scheduler |
| T2 | 1× A100 80 GB | ≤ a few hours | New resource class |
| T3 | 8× A100 80 GB, one node | ≤ ~2 h | New resource class |
| T4 | 2–4 nodes of 8× A100 | ≤ ~1 h | Cluster launch; the notebook analyzes committed logs |

Per-tier section counts are in the chapter overview.

**Rules for T2 and above:**

- Every T2+ notebook has a reduced configuration that runs at T1 with the same code (a config flag).
- Analysis cells run on committed artifacts (logs, traces, small checkpoints via `d2l.DATA_HUB`).

**Hardware limits:**

- A100 has no FP8 tensor cores and does not run FlashAttention-3. Native FP8 (4.4) therefore runs on the build box's Ada GPUs.
- MXFP4/NVFP4 needs Blackwell. These formats are simulated (4.4, 11.5).

**Build system:**

- Add T2–T4 resource classes to `runtime_env.notebook_resource`.
- Run them on dedicated hosts and commit the outputs.
- Add a `d2l.launch` helper that submits a `%%writefile` script with `torchrun --nnodes` or SLURM and collects the logs.
- Re-execute T3/T4 notebooks per release, not per build.

## Running examples

- **R1. `d2l.GPT` loading released checkpoints (1.2).** Extended with GQA and QK-norm, both already pluggable code in `sec_kv-cache` and `sec_transformer-block`. It loads Qwen3 dense models (0.6B–8B).
- **R2. The book's reference model (4.2).** A ~150–400M decoder trained on a FineWeb-Edu slice. Used in Chapters 4, 6, 11 and 12.
- **R3. The book as a corpus (2.4–2.5).** D2L's own text, chunked and indexed. It is the retrieval corpus for 2.3–2.5, the knowledge source for agents in Chapter 10, and a fine-tuning domain in 4.5.
- **Released models:**
  - Qwen3-0.6B/1.7B/4B/8B for fine-tuning, RL, distillation and evaluation;
  - Qwen3.5-0.8B/2B/4B (hybrid Gated DeltaNet, Apache-2.0) as the current model in 1.1, as a comparison model in evaluations, and in 14.6;
  - Qwen-Scope SAEs for Qwen3 base models in 1.6;
  - Qwen3-30B-A3B for expert parallelism, serving and routing analysis (14.2);
  - OLMo 2 for open-data analysis and checkpoint trajectories (4.2);
  - ModernBERT-base and Qwen3-Embedding-0.6B for encoders and embeddings;
  - gpt-oss-20b for native MXFP4.
- **Shared harness.** The evaluation harness from 1.4–1.5 is used by every later notebook.

## Conventions

- **PyTorch only.** This is a deliberate exception to Advanced's PyTorch+JAX policy, with the RL chapter as precedent.
- **Experiments.** One headline experiment per notebook, single seeded runs, and no conclusions drawn from noise.
- **Technical reports.** Analyzed in depth only in Chapter 14. Elsewhere they get at most one-sentence pointers.
- **Private authoring sources** (the ETH slide decks) stay out of bibliographies.
- **Cross-references** use labels only. New chapter labels follow `chap_lm_<topic>`.

## State of the art: what is covered and where

The plan is dated October 2026. The table lists the 2025–2026 developments that the core sections teach. Chapter 14 is refreshed every edition. Elsewhere a section cites the method's origin paper plus one current exemplar, so that it ages slowly.

| Development | Where |
|---|---|
| A hybrid released model used from the first notebook (Qwen3.5) | 1.1 |
| Sparse autoencoders including released SAE suites, attribution graphs, steering | 1.6 |
| LLM-derived embedding and reranking models | 2.3, 2.4 |
| Contextual retrieval, late chunking, reasoning-intensive retrieval, RAG evaluation | 2.4, 2.5 |
| Model-based filtering, rephrasing and synthetic pretraining data | 3.2, 3.5 |
| QK-clip/MuonClip stability, NVFP4 pretraining | 4.3, 4.4 |
| Prefill-oriented and SRAM-based accelerators; throughput–interactivity benchmarks | 5.1, 5.2 |
| Rack-scale NVLink domains, wide expert parallelism | 5.6, 6.6, 13.4 |
| RL with verifiable rewards: GRPO, Dr. GRPO, DAPO, GSPO, CISPO, RL scaling laws | 8.5 |
| Asynchronous RL systems | 8.6 |
| On-policy and self-distillation | 9.2 |
| Reasoning models, test-time compute, efficient reasoning | 9.3–9.7 |
| Tool calling, MCP, agent harnesses, agentic RL, agent security | Chapter 10 |
| Native MXFP4 releases, rotation-based 4-bit inference | 11.4, 11.5 |
| KV-cache compression at inference time | 12.4 |
| Prefill/decode disaggregation, tiered KV-cache stores | 5.3, 13.4 |
| Sparsity scaling laws, zero-computation experts, LatentMoE | 14.2 |
| MLA, learned sparse attention (NSA, DSA) | 14.3, 14.5 |
| Gated attention, learned sinks, partial RoPE and NoPE layers | 14.4 |
| Hybrid linear-attention models (Qwen3.5, Kimi Linear, Kimi K3, Nemotron 3) | 14.6 |
| Hyper-connections, mHC, Attention Residuals | 14.7 |
| Byte-level patching, diffusion LMs | 14.8 |

## References policy

- **Section sources:** at least one primary paper, one tutorial/course/blog, and one code base.
- **Chapter teaching sources:** the courses, tutorials and blogs that teach the chapter well.
- **Verification:** all links were checked on 28 Oct 2026. arXiv titles were confirmed through the arXiv API; other URLs returned HTTP 200.
- **Exceptions:** works whose landing pages block scripted checks, or that have no stable open landing page, are cited by title:
  - Robertson and Zaragoza's BM25 monograph;
  - Levy and Goldberg (2014), already in the book's bibliography as `Levy.Goldberg.2014`;
  - Cormack, Clarke and Büttcher (2009) on reciprocal-rank fusion;
  - the Meta and Alibaba network papers from SIGCOMM 2024;
  - Young (1974) and Daly (2006) on checkpoint intervals.

**Courses cited across the part:**

- Stanford [CS336 Language Modeling from Scratch](https://cs336.stanford.edu/) and [CS224N](https://web.stanford.edu/class/cs224n/);
- [CMU Advanced NLP](https://cmu-l3.github.io/anlp-fall2025/) and the [CMU 11-968 LLM Systems syllabus](https://www.cmu.edu/online/welcome/pdfs/11968-llm-systems-syllabus-f25.pdf);
- [MIT 6.5940 Efficient AI](https://hanlab.mit.edu/courses/2026-fall-65940);
- the [Berkeley Scalable AI syllabus](https://scalable-ai.eecs.berkeley.edu/assets/syllabus.pdf), [Berkeley LLM Agents](https://llmagents-learning.org/f24) and [Advanced LLM Agents](https://rdi.berkeley.edu/adv-llm-agents/sp25), and [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/);
- the Hugging Face [LLM Course](https://huggingface.co/learn/llm-course/chapter1/1), [smol course](https://huggingface.co/learn/smol-course/en/unit0/1) and [Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction);
- Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html);
- Raschka's [from-scratch books](https://www.sebastianraschka.com/from-scratch/).

---

## Chapter overview

| # | Chapter | Sections | Tiers | Main artifact |
|---|---|---:|---|---|
| 1 | Working with Pretrained Language Models | 6 | T1 | Qwen3.5 through Transformers, R1 and the evaluation harness |
| 2 | Encoders, Embeddings and Retrieval | 5 | T1 | Fine-tuned encoder, embedder, retrieval index and RAG system over R3 |
| 3 | Data for Language Models | 6 | T0–T2 | Versioned corpus manifest |
| 4 | Pretraining | 7 | T0–T2 | R2 and its training report |
| 5 | Hardware for Training and Inference | 6 | T0, T2–T4 | Measured performance model of a chip, a node and a cluster |
| 6 | Parallelism Beyond One Node | 8 | T3–T4 | Multi-node training configurations |
| 7 | Fine-Tuning | 9 | T1–T3 | Quantizers, adapters and fine-tuned checkpoints |
| 8 | Preference Optimization and Reinforcement Learning | 7 | T1–T3 | Reward model, DPO and GRPO checkpoints |
| 9 | Distillation and Reasoning | 7 | T1–T3 | Accuracy–compute frontiers |
| 10 | Agents and Tool Use | 5 | T0–T3 | Agent harness and an RL-trained search agent |
| 11 | Model Compression | 6 | T1 | Compressed checkpoints with quality reports |
| 12 | Efficient Inference | 4 | T1–T2 | Fast single-GPU generator |
| 13 | Serving Systems | 5 | T1, T3 | Scheduler and workload replay |
| 14 | Advanced Architectures | 9 | T0–T3 | Design table and a small-scale ablation of each axis |
| | **Total** | **90** | | |

Sections per compute tier: T0 8, T1 47, T2 18, T3 14, T4 3. A section with two tiers (for example, T2 reduced and T3 full) is counted at its heavier tier.

Bracketed numbers after each section title (e.g., `[Codex 1.4]`) give the Codex plan units it absorbs.

---

## Chapter 1 Working with Pretrained Language Models

**Label:** `chap_lm_pretrained`.

The part starts from released checkpoints, so every later chapter has a capable model to measure. Readers first use a current model as practitioners do, through Hugging Face Transformers. They then rebuild a released model in the book's own code and check that it matches. The chapter also builds the evaluation harness used by every later notebook.

**Learning objectives:**
- Find, load, pin and run a current released model with Hugging Face Transformers, and read its configuration file.
- Load a released checkpoint into the book's own model code and prove that it matches the reference.
- Explain how prompt format, chat templates and in-context demonstrations change behavior.
- Score models under explicit conventions, and report differences with error bars that survive a paired test.
- Test claims about what a model computes by intervening on its activations, not only by reading them.

**Assignment A1.**

**Teaching sources:**
- [CS336](https://cs336.stanford.edu/) (evaluation lectures);
- [HF LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) (chapter 2 covers the Transformers API);
- [NeurIPS 2024 tutorial on evaluating LLMs](https://neurips.cc/virtual/2024/tutorial/99524) and [ICML 2024 tutorial on LM evaluation](https://icml.cc/virtual/2024/tutorial/35227);
- Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/);
- Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/);
- [ARENA](https://github.com/ARENA-education/ARENA_materials) (interpretability curriculum) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

### 1.1 Pretrained models with Hugging Face · T1 · [Codex walkthrough]

**Builds on:** `sec_text-sequence` (tokenizers), `sec_decoding`, `sec_read_write`, `sec_hardware` (memory and the decode bound).

**New:**
- The Hub: model cards, configuration files, sharded safetensors and their index, revisions, gated models.
- Pinning a model by revision hash; licenses; why to avoid `trust_remote_code`. This absorbs the "Pin the Identity" and "Trust and Licenses" material from the retiring Tools appendix.
- Transformers' Auto classes: tokenizer, model, and generation configuration; `dtype` and `device_map`; `generate` and its sampling arguments, mapped to `sec_decoding`.
- Base, instruct and thinking variants; applying a chat template; switching thinking on and off.
- Reading a configuration file. Every field of a current hybrid model is mapped to the section that teaches it. Qwen3.5 has Gated DeltaNet layers with one full-attention layer in four (`sec_deltanet`, 14.6), an output gate on attention (14.4), partial RoPE (14.4), a multi-token-prediction head (12.3), and a vision tower (the later multimodal chapters).
- What Transformers is for and where it stops: high-throughput serving goes to vLLM and SGLang (13.1); the book's own code (1.2) exposes every tensor.

**Derivation:** The memory needed to run a model, from its configuration:
- weight bytes per format;
- the KV cache per token for attention layers;
- the fixed recurrent state for linear-attention layers;
- the activation workspace for one forward pass.

**Notebook:**
1. Load Qwen3.5-2B (T1; 4B as an option) pinned to a revision.
2. Annotate its configuration with section references, and reconcile the parameter count with the safetensors index.
3. Chat with thinking on and off.
4. Compare base and post-trained outputs on the same prompts.
5. Measure memory and tokens per second against the derived memory and the `sec_hardware` decode bound.

**Sources:**
- Papers: [Transformers](https://arxiv.org/abs/1910.03771); [Qwen3](https://arxiv.org/abs/2505.09388).
- Documentation: [Transformers docs](https://huggingface.co/docs/transformers/index); [Transformers quick tour](https://huggingface.co/docs/transformers/quicktour); [generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies); [model cards](https://huggingface.co/docs/hub/model-cards); [safetensors](https://huggingface.co/docs/safetensors/index); [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
- Course: [HF LLM Course, chapter 2](https://huggingface.co/learn/llm-course/chapter2/1).
- Model cards: [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B); [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B); [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).

### 1.2 From `d2l.GPT` to a released checkpoint · T1 · [Codex walkthrough, 2.1]

**Builds on:** `sec_gpt` (loading GPT-2), `sec_transformer-block`, `sec_kv-cache` (GQA), `sec_scaling-laws` (Qwen3 configuration row), `sec_read_write`, 1.1.

**New:**
- GQA and QK-norm in `d2l.GPT`; mapping a Hugging Face configuration onto its constructor.
- Mapping weight names between code bases: fused vs separate projections, the RoPE layout (interleaved or split halves), tied embeddings.
- Reconstructing the parameter count from tensor shapes.
- Matching layer by layer, and localizing a mismatch to the first diverging layer.
- Why the book reimplements Qwen3 (dense, full attention) and not Qwen3.5: the hybrid layers need `chap_modern_rnn` code. 14.6 loads Qwen3.5 into the book's hybrid model.

**Notebook:** Load Qwen3-0.6B into `d2l.GPT` and match the Transformers logits of 1.1 within a stated BF16 tolerance, layer by layer. Then generate with the book's sampler. This model is R1.

**Sources:** [Qwen3](https://arxiv.org/abs/2505.09388); [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B); [safetensors](https://huggingface.co/docs/safetensors/index); [LLMs from scratch](https://github.com/rasbt/LLMs-from-scratch); Raschka's [from-scratch materials](https://www.sebastianraschka.com/from-scratch/).

### 1.3 Prompting and in-context learning · T1 · [new]

**Builds on:** `sec_what-attention-computes` (induction heads), `sec_decoding`, 1.1.

**New:**
- Chat templates as tokenized input: roles, special tokens, the generation prefix, system prompts.
- Zero- and few-shot formats.
- Sensitivity to the order, format and label correctness of demonstrations.
- In-context learning contrasted with weight updates.

**Notebook:** Measure few-shot accuracy of a base and an instruct checkpoint as the number, order and correctness of the demonstrations vary.

**Sources:** [GPT-3](https://arxiv.org/abs/2005.14165); [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837); [Induction heads](https://arxiv.org/abs/2209.11895); [chat templating](https://huggingface.co/docs/transformers/chat_templating).

### 1.4 Scoring language models · T1 · [Codex 15.2]

**Builds on:** `sec_language-model` (perplexity, bits per byte), `sec_softmax` (calibration).

**New:**
- Log-likelihood scoring: cloze vs letter formats, length normalization.
- Generative answers and answer extraction; execution-based checks.
- The pass@k estimator.
- Batch-dependent nondeterminism in inference and how to control it.

**Derivation:** The unbiased pass@k estimator from n samples with c correct, and its variance.

**Notebook:** Score two checkpoints under three conventions and show their ranking flip. The scoring functions become `d2l` utilities.

**Sources:** [HumanEval/Codex](https://arxiv.org/abs/2107.03374); [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness); [Lessons from the Trenches](https://arxiv.org/abs/2405.14782).

### 1.5 Evaluation suites, judges and error bars · T1 · [Codex 15.3, 15.5, 15.6]

**Builds on:** `sec_mdl-statistics` (bootstrap, confidence intervals), `sec_regularized` (Bradley–Terry).

**New:**
- Development vs reporting suites; saturation.
- Current reporting suites: MMLU-Pro, GPQA, Humanity's Last Exam, LiveCodeBench, LiveBench. Agent benchmarks are in 10.5.
- Contamination-limited evaluation.
- Paired item-level comparisons; clustered standard errors.
- LLM judges and their position, verbosity and self-preference biases.
- Rankings from pairwise judgments.

**Derivation:** The variance of a paired difference, and why pairing shrinks it. Clustered standard errors for items that share a passage.

**Notebook:** Judge two checkpoints with a small model in both answer orders, then report which differences survive a paired bootstrap.

**Sources:** [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640); [LLM-as-a-judge](https://arxiv.org/abs/2306.05685); [Chatbot Arena](https://arxiv.org/abs/2403.04132); [MMLU-Pro](https://arxiv.org/abs/2406.01574); [GPQA](https://arxiv.org/abs/2311.12022); [Humanity's Last Exam](https://arxiv.org/abs/2501.14249); [LiveBench](https://arxiv.org/abs/2406.19314); [Dror et al.](https://aclanthology.org/P18-1128/).

### 1.6 Interpretability · T1 · [new]

**Builds on:** `sec_what-attention-computes` (induction heads, attention circuits), `sec_mdl-svd-low-rank`, 1.2 (R1).

**New:**
- The residual stream as a shared workspace; reading it with the logit lens and the tuned lens.
- Linear probes, with control tasks that separate what a representation encodes from what the probe learns.
- Activation patching: localizing a behavior by intervention rather than by correlation.
- Sparse autoencoders as dictionary learning on the residual stream; labeling features and its pitfalls.
- Released SAE suites (Gemma Scope, Qwen-Scope) as instruments for models too large to train SAEs on in a notebook.
- Steering vectors and their side effects.
- Attribution graphs, in outline.

**Derivation:** Activation patching as an estimate of a component's causal effect on a metric, and its first-order approximation (attribution patching). The sparse-autoencoder objective as L1-relaxed sparse coding, and why the decoder columns need bounded norm. Top-k SAEs as the L0-constrained alternative.

**Notebook:** On R1:
1. Track the rank of the correct answer across layers with the logit lens and with a tuned lens.
2. Fit a linear probe for a simple property, with a control task to measure selectivity.
3. Localize a factual-recall behavior by patching activations between clean and corrupted prompts.
4. Train a small sparse autoencoder on one layer's residual stream, plot reconstruction loss against sparsity, and label a few features. Compare with a released Qwen-Scope SAE for Qwen3-1.7B-Base.
5. Add a steering vector, and measure the behavior change against its perplexity cost.

**Sources:**
- Lenses and probes: [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens); [Tuned Lens](https://arxiv.org/abs/2303.08112).
- Sparse autoencoders: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features); [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600); [Scaling and evaluating SAEs](https://arxiv.org/abs/2406.04093).
- Released SAE suites: [Gemma Scope](https://arxiv.org/abs/2408.05147); [Qwen-Scope](https://arxiv.org/abs/2605.11887) and its [SAE for Qwen3-1.7B-Base](https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50).
- Steering: [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681); [Representation Engineering](https://arxiv.org/abs/2310.01405).
- Circuits: [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).
- Code and courses: [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens); [ARENA](https://github.com/ARENA-education/ARENA_materials).

## Chapter 2 Encoders, Embeddings and Retrieval

**Label:** `chap_lm_encoders`.

Decoder-only models are now central. Encoders and embeddings still do much of the work in search, classification, data filtering (Chapter 3) and retrieval for agents (Chapter 10).

`sec_transformer` already builds a bidirectional encoder and trains a masked-token objective on characters. The chapter is brief on encoders. It covers the pretrained encoder family and the embedding models built on it. It spends its length on the systems that use them: retrieval, and retrieval-augmented generation (RAG). It replaces two retiring chapters: NLP Pretraining (word2vec, GloVe, BERT) and NLP Applications (sentiment, NLI).

**How 2.2 and 2.3 divide the work.** Both start from a pretrained encoder, but they produce different things.
- **2.2:** an encoder pretrained with masked tokens. It yields one contextual vector per token, and a task head on top is fine-tuned for each task (classification, NLI, tagging).
- **2.3:** a model trained contrastively. It yields one vector per text, built so that cosine similarity means relatedness. A single model serves search, clustering, deduplication and classifier features without task-specific fine-tuning.

2.3's headline experiment shows why the second step is needed. Mean-pooled vectors from 2.2's encoder retrieve poorly, and contrastive training repairs them.

**Learning objectives:**
- Explain static word vectors as factorizations of co-occurrence statistics, and contextual embeddings as their successors.
- Fine-tune a pretrained encoder for classification and pair tasks, and decide when it beats prompting a decoder.
- Explain why pretrained token representations make poor sentence embeddings, and train and evaluate a contrastive embedding model.
- Build sparse, dense and hybrid retrieval indexes, and measure their recall, latency and memory.
- Build a RAG system stage by stage, evaluate each stage with error bars, and locate where its failures arise.

**Assignment A2.**

**Teaching sources:**
- [CS224N](https://web.stanford.edu/class/cs224n/) (word vectors, pretraining);
- Jay Alammar, [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) and [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/);
- [Sentence Transformers documentation](https://sbert.net/);
- [ModernBERT blog](https://huggingface.co/blog/modernbert);
- [Faiss wiki](https://github.com/facebookresearch/faiss/wiki);
- Eugene Yan, [Patterns for Building LLM-based Systems](https://eugeneyan.com/writing/llm-patterns/);
- Jason Liu, [Systematically Improving Your RAG](https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/);
- Hamel Husain, [Stop Saying RAG Is Dead](https://hamel.dev/notes/llm/rag/not_dead.html);
- Hugging Face cookbook, [Advanced RAG](https://huggingface.co/learn/cookbook/advanced_rag).

### 2.1 Word embeddings in brief · T1 · [replaces retired word2vec/GloVe sections]

**Builds on:** `subsec_mdl-pmi`, `sec_mdl-mutual-information` (the SGNS exercise), `sec_mdl-svd-low-rank`, `sec_gpt` (the token-embedding table).

**New:**
- The distributional hypothesis.
- Skip-gram with negative sampling (keeps the labels `subsec_negative-sampling` and `sec_word2vec_data`).
- The 0.75-power noise distribution and subsampling of frequent words.
- GloVe as weighted least squares on log co-occurrences.
- The SGNS optimum as shifted PMI. The result is stated here; its derivation stays the exercise in `sec_mdl-mutual-information`.
- Truncated SVD of PPMI as the closed-form counterpart.
- Analogies and their caveats; bias in embeddings.
- Static vectors vs the input-embedding table of a language model; tied input and output embeddings.
- Embeddings for categorical features (target of the `sec_pandas` reference).

**Notebook:** Train SGNS on Penn Treebank (the corpus `sec_scaling-laws` already loads) in minutes. Compare it with SVD of PPMI on word-similarity judgments, then inspect nearest neighbors in R1's token-embedding table.

**Sources:** [word2vec](https://arxiv.org/abs/1301.3781); [Negative sampling](https://arxiv.org/abs/1310.4546); [GloVe](https://aclanthology.org/D14-1162/); Levy and Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (NeurIPS 2014); [Debiasing word embeddings](https://arxiv.org/abs/1607.06520).

### 2.2 Pretrained encoders: BERT and its descendants · T1 · [replaces retired BERT, sentiment and NLI sections]

**Builds on:** `sec_transformer` (bidirectional encoder, masked-token objective, `subsec_mlm-loss-by-position`), `sec_text-sequence`.

**Output:** contextual token vectors plus a fine-tuned task head, one model per task.

**New:**
- The BERT recipe: [CLS]/[SEP], 80/10/10 masking, next-sentence prediction and why RoBERTa dropped it.
- RoBERTa's lessons: data, dynamic masking, batch size.
- ELECTRA's replaced-token detection and its sample efficiency.
- T5's span corruption, in one paragraph (`sec_transformer` owns encoder–decoder wiring).
- ModernBERT: RoPE, alternating local/global attention, unpadding, long context, code data.
- Fine-tuning heads for sequence classification, sentence pairs (NLI) and token tagging.
- When a small encoder beats prompting a decoder: cost per query, latency, many labels.

**Notebook:** Fine-tune ModernBERT-base on an NLI subset and on a sentiment set. Compare it with few-shot prompting of Qwen3-1.7B (1.3) on accuracy and cost per thousand examples.

**Sources:** [BERT](https://arxiv.org/abs/1810.04805); [RoBERTa](https://arxiv.org/abs/1907.11692); [ELECTRA](https://arxiv.org/abs/2003.10555); [T5](https://arxiv.org/abs/1910.10683); [ModernBERT](https://arxiv.org/abs/2412.13663); [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base).

### 2.3 Text embeddings from contrastive training · T1 · [Codex 2.7]

**Builds on:** `sec_mdl-infonce`, 2.2.

**Output:** one normalized vector per text, used without task-specific fine-tuning.

**New:**
- Why pretrained token representations make poor sentence vectors: anisotropy, and frequency dominating the geometry.
- Pooling: CLS, mean or last token; normalization and cosine similarity.
- Contrastive training with in-batch negatives (temperature; batch size as the number of negatives) and with mined hard negatives (the false-negative risk).
- Weakly supervised pair pretraining followed by supervised fine-tuning.
- Instruction-prefixed queries; asymmetric query and document encoders.
- Embedders derived from LLMs: bidirectional attention vs last-token pooling.
- Matryoshka dimensions; int8 and binary embedding quantization.
- MTEB and its pitfalls.

**Derivation:**
- Why InfoNCE with B in-batch negatives bounds mutual information by log B (`sec_mdl-infonce`), and what that implies for batch size.
- InfoNCE's large-batch limit as an alignment term plus a uniformity term on the sphere, which explains why contrastive training removes anisotropy.

**Notebook:**
1. Retrieve over R3 with mean-pooled ModernBERT from 2.2. Measure Recall@k and nDCG on held-out questions, and the anisotropy (mean pairwise cosine).
2. Train it contrastively with in-batch and then hard negatives, and remeasure.
3. Compare it with Qwen3-Embedding-0.6B.
4. Sweep the Matryoshka dimension and binary quantization.

**Sources:**
- Sentence embeddings and anisotropy: [Sentence-BERT](https://arxiv.org/abs/1908.10084); [On the Sentence Embeddings from Pre-trained LMs](https://arxiv.org/abs/2011.05864); [SimCSE](https://arxiv.org/abs/2104.08821); [Alignment and Uniformity](https://arxiv.org/abs/2005.10242).
- Embedding models: [E5](https://arxiv.org/abs/2212.03533); [E5-Mistral](https://arxiv.org/abs/2401.00368); [NV-Embed](https://arxiv.org/abs/2405.17428); [Qwen3 Embedding](https://arxiv.org/abs/2506.05176).
- Compression and evaluation: [Matryoshka](https://arxiv.org/abs/2205.13147); [embedding quantization](https://huggingface.co/blog/embedding-quantization); [MTEB](https://arxiv.org/abs/2210.07316).

### 2.4 Retrieval: sparse, dense and approximate search · T1 · [new]

**Builds on:** 2.3, `sec_mdl-svd-low-rank`.

**New:**
- Inverted indexes and BM25 as the baseline that dense retrieval must beat; learned sparse retrieval (SPLADE), in brief.
- Approximate nearest-neighbor indexes: IVF with k-means, product quantization, HNSW. The trade-offs between recall, latency and memory.
- Hybrid retrieval and reciprocal-rank fusion.
- Reranking: bi-encoders vs cross-encoders; LLM-based rerankers; late interaction (ColBERT).
- Retrieval metrics (Recall@k, MRR, nDCG) and benchmarks: BEIR, and reasoning-intensive retrieval (BRIGHT).

**Derivation:** Asymmetric distance computation in product quantization, its memory cost, and its error bound.

**Notebook:**
1. Index R3 with BM25 and with a dense embedder from 2.3 under flat, IVF-PQ and HNSW indexes.
2. Fuse BM25 and dense results.
3. Add a cross-encoder reranker.
4. Plot recall against latency and memory.

The resulting index is the retriever for 2.5 and Chapter 10.

**Sources:**
- Sparse retrieval: Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009); [SPLADE](https://arxiv.org/abs/2107.05720); Cormack, Clarke and Büttcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (SIGIR 2009).
- Dense retrieval and late interaction: [DPR](https://arxiv.org/abs/2004.04906); [ColBERT](https://arxiv.org/abs/2004.12832); [ColBERTv2](https://arxiv.org/abs/2112.01488).
- Approximate search: [Product quantization](https://doi.org/10.1109/TPAMI.2010.57); [HNSW](https://arxiv.org/abs/1603.09320); [Faiss](https://arxiv.org/abs/2401.08281).
- Benchmarks: [BEIR](https://arxiv.org/abs/2104.08663); [BRIGHT](https://arxiv.org/abs/2407.12883).
- Code and models: [Pyserini](https://github.com/castorini/pyserini); [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).

### 2.5 Retrieval-augmented generation · T1 (long notebook) · [new]

**Builds on:** 2.4, 1.3 (prompting), 1.5 (judges, paired bootstrap).

**New:** The RAG pipeline is treated as a chain of stages, each measured on its own:
- **Chunking:** fixed windows; structure-aware chunks along sections, equations and code cells; overlap; contextual chunk headers (contextual retrieval); late chunking.
- **Query processing:** rewriting; decomposition of multi-hop questions; hypothetical-document expansion (HyDE).
- **Context assembly:** ordering and position effects ("lost in the middle"); token budgets; removing near-duplicate passages.
- **Grounded generation:** citation formats; citation precision and recall; abstaining when the evidence is missing.
- **End-to-end evaluation:**
  - retrieval recall vs answer correctness vs faithfulness;
  - LLM judges with the 1.5 error bars;
  - a failure taxonomy: not retrieved, ranked too low, retrieved but ignored, unsupported claim.
- **Long context vs retrieval:** when putting whole documents in the context wins.
- **Training for RAG, in outline:**
  - joint retriever–generator training (REALM, RAG);
  - fine-tuning the generator on retrieved context (RAFT);
  - learning when to retrieve (Self-RAG);
  - iterative retrieval as an agent loop (10.3–10.4).

**Derivation:** RAG-Sequence and RAG-Token as marginalizations over retrieved documents. The top-k approximation to that sum. The gradient that reaches the retriever through the marginal likelihood.

**Notebook:** A long notebook (about twice the usual length) over R3, with Qwen3-4B as the generator. Its six parts can be assigned separately:
1. **Evaluation set.** Build questions over the book with gold passages and answers, partly LLM-generated and all checked by hand, with a held-out split.
2. **Chunking.** Compare three chunking strategies by retrieval recall.
3. **Query side.** Add query rewriting and the 2.4 reranker, and measure the gain of each stage.
4. **Grounded answers.** Generate answers with citations. Measure answer accuracy, citation precision and faithfulness with a judge, and report paired-bootstrap intervals.
5. **Failure analysis.** Classify 50 failures by stage, fix the largest class, and remeasure.
6. **Long context.** Compare against placing whole chapters in a 32K context at an equal latency budget.

**Sources:**
- Foundations: [RAG](https://arxiv.org/abs/2005.11401); [REALM](https://arxiv.org/abs/2002.08909); [Lost in the Middle](https://arxiv.org/abs/2307.03172).
- Pipeline stages: [HyDE](https://arxiv.org/abs/2212.10496); [Late Chunking](https://arxiv.org/abs/2409.04701); Anthropic, [Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval).
- Training for RAG: [RAFT](https://arxiv.org/abs/2403.10131); [Self-RAG](https://arxiv.org/abs/2310.11511).
- Evaluation and failures: [RAGAS](https://arxiv.org/abs/2309.15217) ([code](https://github.com/vibrantlabsai/ragas)); [Seven Failure Points](https://arxiv.org/abs/2401.05856); [FRAMES](https://arxiv.org/abs/2409.12941).
- Long context vs retrieval: [Long-Context LLMs Meet RAG](https://arxiv.org/abs/2410.05983).
- Practice: [HF Advanced RAG cookbook](https://huggingface.co/learn/cookbook/advanced_rag); Jason Liu, [Systematically Improving Your RAG](https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/).

## Chapter 3 Data for Language Models

**Label:** `chap_lm_data`.

This chapter covers what text goes into the model, and how it is extracted, filtered, deduplicated, mixed and tokenized.

**Does not re-teach:** BPE (`sec_text-sequence`), Zipf and n-gram statistics (`sec_language-model`), distribution shift (`sec_environment-and-distribution-shift`).

**Learning objectives:**
- Turn raw crawl data into a documented, versioned training corpus.
- Derive and tune MinHash/LSH deduplication.
- Choose filters and mixtures with small proxy experiments rather than intuition.
- Choose a tokenizer and vocabulary size for a multilingual corpus.

**Assignment A3.**

**Teaching sources:**
- [CS336](https://cs336.stanford.edu/) (data lectures and assignment);
- [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1);
- [Datatrove](https://github.com/huggingface/datatrove);
- [minbpe](https://github.com/karpathy/minbpe);
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (data chapters).

### 3.1 From crawl to documents · T0 · [Codex 1.1, 1.4]

**New:**
- WARC/WET records.
- HTML extraction and how it fails on code, tables and mathematics.
- Language identification.
- Unicode normalization and what it destroys.
- PII removal; provenance, licenses and opt-outs; dataset cards.

**Notebook:** Extract one Common Crawl shard with two extractors, compare what each loses, and emit a manifest with source IDs and transformation logs.

**Sources:** [Common Crawl](https://commoncrawl.org/get-started); [Trafilatura](https://trafilatura.readthedocs.io/); [fastText LID](https://fasttext.cc/docs/en/language-identification.html); [Unicode TR15](https://www.unicode.org/reports/tr15/); [FineWeb](https://arxiv.org/abs/2406.17557).

### 3.2 Quality filtering · T1 · [Codex 1.5]

**Builds on:** 2.3 (embedding classifiers).

**New:**
- Heuristic rules.
- Classifiers trained on LLM-annotated labels: fastText, or an embedding model with a linear head (the FineWeb-Edu recipe).
- Perplexity filters.
- Auditing removals by language, domain and dialect.

**Notebook:** Train a quality classifier from LLM-annotated labels, apply it to the 3.1 corpus, and audit false removals.

**Sources:** [DCLM](https://arxiv.org/abs/2406.11794); [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu); [Gopher](https://arxiv.org/abs/2112.11446) (filter rules); [Nemotron-CC](https://arxiv.org/abs/2412.02595).

### 3.3 Deduplication and decontamination · T0 · [Codex 1.5, 15.7]

**Builds on:** `sec_prob`, 2.3.

**New:**
- Exact hashing.
- MinHash and LSH banding.
- Suffix-array substring deduplication.
- Semantic deduplication with embeddings.
- Benchmark decontamination by n-gram overlap.
- Deduplication and memorization.

**Derivation:** The probability that MinHash signatures agree equals the Jaccard similarity. The S-curve of b bands × r rows, and where to set its threshold. None of this is in the Mathematics part.

**Notebook:** Implement MinHash/LSH, tune b × r against labeled near-duplicate pairs, and detect benchmark items planted in a training slice.

**Sources:** Broder, "On the resemblance and containment of documents" (1997); [Deduplicating Training Data](https://arxiv.org/abs/2107.06499); [SemDeDup](https://arxiv.org/abs/2303.09540); [Extracting Training Data](https://arxiv.org/abs/2012.07805).

### 3.4 Data mixtures and curricula · T2 · [Codex 1.6]

**New:**
- Domain weights as hyperparameters.
- Temperature sampling across languages.
- Proxy-model regression; DoReMi.
- Annealing on high-quality data.

**Notebook:** Train proxy models on web/code/math mixtures, fit a regression from mixture weights to per-domain loss, and test it on a held-out mixture.

**Sources:** [DoReMi](https://arxiv.org/abs/2305.10429); [RegMix](https://arxiv.org/abs/2407.01492); [OLMo 2](https://arxiv.org/abs/2501.00656); [SmolLM2](https://arxiv.org/abs/2502.02737).

### 3.5 Synthetic and rephrased data · T1 · [Codex 1.7]

**New:**
- Textbook-style generation.
- Rephrasing web text.
- Quality vs diversity.
- Recursive training and collapse.
- Licenses of generated data.

**Notebook:** Rephrase a web slice with a local instruct model. Compare small models trained on original, rephrased and mixed data at equal token budgets.

**Sources:** [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644); [Rephrasing the Web](https://arxiv.org/abs/2401.16380); [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia); [Nemotron-CC](https://arxiv.org/abs/2412.02595); [Model collapse](https://arxiv.org/abs/2305.17493).

### 3.6 Tokenizers at scale · T0 · [Codex 1.2, 1.3]

**Builds on:** `sec_text-sequence`, `sec_mdl-information_theory`.

**New:**
- Unigram tokenization and Viterbi segmentation; SentencePiece.
- Vocabulary size against embedding and softmax cost.
- Fertility across scripts; numbers and code.
- Extending a vocabulary and initializing the new embeddings (2.1).
- Byte-patch models as an aside.

**Notebook:** Train BPE and unigram tokenizers on one multilingual slice and compare bits per byte and fertility by language. Extend a pretrained vocabulary and compare initializations.

**Sources:** [SentencePiece](https://aclanthology.org/D18-2012/); [Subword Regularization](https://aclanthology.org/P18-1007/); [Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623); [WECHSEL](https://aclanthology.org/2022.naacl-main.293/); [Byte Latent Transformer](https://arxiv.org/abs/2412.09871).

## Chapter 4 Pretraining

**Label:** `chap_lm_pretraining`.

**Does not re-teach:**
- optimizers and schedules (`sec_adamw`, `sec_muon`, `sec_scheduler`);
- batch size and muP (`sec_batch_size`, `sec_scaling`);
- the Chinchilla fit (`sec_scaling-laws`);
- mixed precision (`sec_memory_precision`);
- clipping and EMA (`sec_practice`).

**Learning objectives:**
- Build an exact, resumable data path with packing and document masks.
- Train and monitor a reference model, and diagnose instabilities from logs.
- Apply FP8 training, continued pretraining and context extension, and measure what each costs and gains.

**Assignment A4.**

**Teaching sources:**
- [CS336](https://cs336.stanford.edu/) (assignment 1);
- Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) and [nanochat](https://github.com/karpathy/nanochat);
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook);
- [SmolLM3 blog](https://huggingface.co/blog/smollm3);
- [OLMo code](https://github.com/allenai/OLMo).

### 4.1 The pretraining data path · T1 · [Codex 1.8, 5.1, 8.4]

**Builds on:** `sec_gpt`, `sec_repro`, `sec_read_write`, `sec_adamw` ("What Not to Decay").

**New:**
- Pre-tokenized sharded streams.
- Sequence packing with document masks (FlexAttention) and per-document positions.
- Token-weighted loss reduction.
- Resumable data cursors.

**Notebook:** Check that packed batches with masks give the same per-document logits as unpacked batches. Interrupt and resume a run, and check that the next batches are identical.

**Sources:** [FlexAttention](https://pytorch.org/blog/flexattention/); [datasets streaming](https://huggingface.co/docs/datasets/stream); [torchdata](https://github.com/pytorch/data).

### 4.2 Training the reference model · T2 · [Codex 5.1, 5.9, 16.2]

**Builds on:** `sec_scheduler` (WSD), `sec_practice` (EMA), 1.4–1.5.

**New:**
- What to log: per-domain loss, update/weight RMS by module, gradient norm, MFU, downstream evaluations over tokens.
- When downstream metrics move relative to loss.
- Choosing and averaging checkpoints.
- Reading a fully open model's trajectory: OLMo 2's data, code and intermediate checkpoints, and how capabilities change over training. This was v4's section 14.2.

**Notebook:**
1. Train a ~150–400M decoder on a few billion FineWeb-Edu tokens on one A100; the reduced configuration runs at T1. The result is R2.
2. Evaluate a sequence of public OLMo 2 checkpoints with the Chapter 1 suite, and compare their curves with R2's.

**Sources:** [OLMo 2](https://arxiv.org/abs/2501.00656) and its [checkpoints](https://huggingface.co/allenai/OLMo-2-1124-7B); [Olmo 3](https://arxiv.org/abs/2512.13961); [Dolma](https://arxiv.org/abs/2402.00159); [OLMES](https://arxiv.org/abs/2406.08446); [SmolLM2](https://arxiv.org/abs/2502.02737); [build-nanogpt](https://github.com/karpathy/build-nanogpt).

### 4.3 Training instabilities · T1 · [Codex 5.9]

**Builds on:** `sec_practice`, `sec_transformer-block` (QK-norm), `sec_muon`.

**New:**
- Attention-logit growth and output-logit divergence.
- Reproducing them in small models at high learning rates.
- z-loss, QK-norm and QK-clip (MuonClip).
- Diagnosing a loss spike and restarting while skipping batches.

**Derivation:** How attention-logit growth follows from the growth of the norms of W_Q and W_K, and why clipping their product bounds it.

**Notebook:** Induce an instability, add each fix in turn, and map the stable learning-rate range of each variant.

**Sources:** [Small-scale proxies for instabilities](https://arxiv.org/abs/2309.14322); [PaLM](https://arxiv.org/abs/2204.02311) (z-loss); [Kimi K2](https://arxiv.org/abs/2507.20534) (MuonClip); [Muon is Scalable](https://arxiv.org/abs/2502.16982).

### 4.4 Low-precision training · T1 on Ada · [Codex 4.5, 5.6]

**Builds on:** `sec_hardware`, `sec_numerics`, `sec_memory_precision`, `sec_fast_transformer`.

**New:**
- Which GEMMs run in FP8; tensorwise, rowwise and blockwise scaling.
- Delayed vs current scaling; E4M3 and E5M2.
- Precision of master weights and optimizer state.
- Block-scaled 4-bit training (NVFP4), simulated.

**Derivation:** The quantization error of block scaling as a function of block size and the dynamic range within a block.

**Notebook:** Train the same small model in BF16 and FP8 (torchao) on Ada GPUs, and compare loss, throughput and overflow statistics. Simulate NVFP4 for the MLP GEMMs.

**Sources:** [FP8 Formats](https://arxiv.org/abs/2209.05433); [DeepSeek-V3](https://arxiv.org/abs/2412.19437); [Scaling FP8 training](https://arxiv.org/abs/2409.12517); [NVFP4 pretraining](https://arxiv.org/abs/2509.25149); [Microscaling](https://arxiv.org/abs/2310.10537); [torchao](https://github.com/pytorch/ao); [Transformer Engine](https://github.com/NVIDIA/TransformerEngine).

### 4.5 Continued pretraining and mid-training · T2 · [Codex 5.8]

**New:**
- Re-warming and re-decaying the learning rate.
- Replay.
- Forgetting vs domain gain.
- Mid-training stages: reasoning, code and long-context data before post-training.

**Notebook:** Continue Qwen3-0.6B base on R3 plus a technical corpus, with and without replay. Track domain loss, general loss and the Chapter 1 suite.

**Sources:** [Don't Stop Pretraining](https://arxiv.org/abs/2004.10964); [Simple and Scalable Continual Pretraining](https://arxiv.org/abs/2403.08763); [OLMo 2](https://arxiv.org/abs/2501.00656).

### 4.6 Long-context training · T2 · [Codex 2.4, 5.8, 15.4]

**Builds on:** `sec_positional-information` (which only names YaRN).

**New:**
- NTK-aware and YaRN frequency scaling in code.
- Progressive length schedules.
- Data with real long-range dependencies.
- Effective context measured with synthetic and natural tasks.

**Derivation:** Which RoPE frequencies interpolation compresses, and why YaRN treats high and low frequencies differently.

**Notebook:** Extend a small checkpoint from 4K to 16K with position interpolation and with YaRN, and compare them on RULER-style and document-QA tasks.

**Sources:** [YaRN](https://arxiv.org/abs/2309.00071); [How to Train Long-Context LMs](https://arxiv.org/abs/2410.02660); [RULER](https://arxiv.org/abs/2404.06654); [HELMET](https://arxiv.org/abs/2410.02694).

### 4.7 ◇ Budgets beyond Chinchilla · T0 · [Codex 5.5]

**Builds on:** `sec_scaling-laws`.

**New:**
- Lifetime cost including inference.
- Overtraining small models.
- Repeated data under a token shortage.

**Derivation:** The compute-optimal model size when inference tokens enter the objective.

**Notebook:** A lifetime-cost calculator built from the `sec_scaling-laws` fit and measured decode rates.

**Sources:** [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448); [Scaling Data-Constrained LMs](https://arxiv.org/abs/2305.16264).

## Chapter 5 Hardware for Training and Inference

**Label:** `chap_lm_hardware`.

`sec_perf_model` and `sec_hardware` cover one GPU and the book's build box:
- the memory hierarchy, with its bandwidth and latency ladders;
- the shoreline argument for why compute outruns bandwidth;
- tensor cores and formats, and a table of accelerator specifications;
- rooflines and the prefill/decode contrast;
- energy, and the box's PCIe topology.

This chapter does not repeat them. It asks three questions at datacenter scale:
1. Why accelerators are designed as they are, and why training, prefill and decode want different designs (5.1–5.2).
2. How data moves through every memory tier and every link of a server and a cluster (5.3–5.5).
3. How scale-up and scale-out domains are built and paid for (5.6).

Measurements use one A100, the 8× A100 node and the multi-node allocation. Reliability is treated with long runs in 6.8.

**Learning objectives:**
- Explain an accelerator's design as an allocation of area, power and package among compute, on-chip memory, off-chip memory and I/O.
- Derive what training, prefill and decode each need from the hardware, and predict step time, time to first token and decode throughput from a specification sheet. Check the prediction by measurement.
- Model every tier and link a byte crosses in a server and a cluster with bandwidth, latency, capacity and cost: from SRAM to disk, and from NVLink to Ethernet.
- Choose between scale-up and scale-out for a workload, and convert utilization and prices into cost per token.

**Assignment A5.**

**Teaching sources:**
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/), especially [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) and [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/);
- Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering);
- [Modal GPU Glossary](https://modal.com/gpu-glossary);
- [GPU MODE lectures](https://github.com/gpu-mode/lectures);
- [Efficiency in LLMs (MLSS)](https://alex.smola.org/posts/45-mlss-efficiency/);
- the [Berkeley Scalable AI syllabus](https://scalable-ai.eecs.berkeley.edu/assets/syllabus.pdf);
- Barroso, Hölzle and Ranganathan, [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2) (3rd ed.);
- [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832).

### 5.1 Accelerator design: compute, memory and I/O · T2 · [Codex 4.8]

**Builds on:** `sec_hardware` (`subsec_hw-shoreline`, `subsec_hw-gpu`, `tab_gpu_specs`), `sec_perf_model`.

**New:**
- A chip as a budget of three resources:
  - die area (reticle limit, chiplets), split among matrix units, on-chip SRAM and memory controllers;
  - power (watts per chip and per rack);
  - package perimeter, which sets how many memory stacks and I/O lanes fit.
- Off-chip memory choices and what each buys:
  - HBM: bandwidth, at a high cost per GB and a limited number of stacks;
  - GDDR: cheap bandwidth, little capacity;
  - LPDDR: capacity per dollar and per watt, little bandwidth;
  - SRAM only: very high bandwidth, tiny capacity per chip.
- Five numbers characterize a chip for language models: peak FLOP/s per format, memory bandwidth, memory capacity, scale-up bandwidth and scale-out bandwidth. Three ratios matter:
  - FLOP per byte (the ridge point);
  - seconds to read all of memory once (capacity over bandwidth);
  - network bytes per FLOP.
- Design points compared on those ratios:
  - GPUs from A100 to Blackwell and Rubin;
  - TPU v7 (Ironwood), Trainium, MI355X;
  - SRAM-heavy inference chips (Groq, Cerebras wafer-scale);
  - prefill-oriented parts with GDDR memory (Rubin CPX).
- Microbenchmarking the hierarchy inside one A100: achieved matmul throughput as a function of shape, and the bandwidth of HBM, L2 and shared memory.

**Derivation:**
- How the split of a fixed area and power budget between SRAM and matrix units moves the ridge point.
- Why capacity over bandwidth is a floor on per-token decode latency for a model that fills memory.
- How many chips an SRAM-only design needs just to hold a model, and what that implies for its cost per token.

**Notebook:**
- (T0) A specification table of about ten accelerators, with derived ratios and a plot of each chip in the plane of ridge point against capacity over bandwidth.
- (T2) Microbenchmarks on one A100: matmul throughput against matrix size and shape, and the bandwidth of HBM, L2 and shared memory. Compare each with the datasheet.

**Sources:**
- Microbenchmarking: [Demystifying the Ampere Architecture](https://arxiv.org/abs/2208.11174); [Benchmarking and Dissecting the Hopper Architecture](https://arxiv.org/abs/2402.13499).
- Design trade-offs: [AI and Memory Wall](https://arxiv.org/abs/2403.14123); [Chiplet Cloud](https://arxiv.org/abs/2307.02666); [WaferLLM](https://arxiv.org/abs/2502.04563).
- TPUs: [TPU v4](https://arxiv.org/abs/2304.01433); Google, [Ironwood](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/).
- Other accelerators: NVIDIA, [Rubin CPX announcement](https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference); [AMD Instinct MI350](https://www.amd.com/en/products/accelerators/instinct/mi350.html); [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/); Groq, [The LPU Explained](https://groq.com/blog/the-groq-lpu-explained); Cerebras, [Inference 3× faster](https://www.cerebras.ai/blog/cerebras-inference-3x-faster).
- Teaching: [Scaling Book: GPUs](https://jax-ml.github.io/scaling-book/gpus/); [Modal GPU Glossary](https://modal.com/gpu-glossary).

### 5.2 Training, prefill and decode: why their hardware needs differ · T2 · [Codex 4.8, 14.1]

**Builds on:** `subsec_hw-two-workloads` (`eq_hw_decode_bound`), `sec_kv-cache`, `sec_memory_precision`, `sec_moe`, 5.1.

**New:**
- **Training** needs FLOPs, capacity and network bandwidth together:
  - FLOPs per token (6N);
  - bytes of state per parameter, and activation memory;
  - communication per step.
- **Prefill** is compute-bound and sets time to first token. A chip used only for prefill can therefore trade HBM for cheaper memory.
- **Decode** is bound by bandwidth and capacity:
  - the batch at which decode turns compute-bound;
  - KV-cache growth with context and batch;
  - for MoE models, total parameters set capacity and active parameters set traffic.
- **The throughput–interactivity frontier.** Tokens per second per GPU are plotted against tokens per second per user, and the cost per million tokens is read off the curve.
- **Consequences for system design:**
  - separate prefill and decode pools (13.4);
  - SRAM chips for low-latency decode;
  - large-memory parts for long contexts;
  - why a training cluster is rarely the cheapest place to serve.

**Derivation:** The decode step time is t(b, L) = max(2·N_act·b / P, (W + b·L·k_KV) / β), subject to W + b·L·k_KV ≤ C. Here:
- b is the batch, L the context and N_act the active parameters;
- W is the weight bytes and k_KV the KV bytes per token;
- P is peak FLOP/s, β the memory bandwidth and C the capacity.

From this model, derive:
- the batch at which decode becomes compute-bound;
- the maximum batch that fits;
- the throughput–interactivity frontier and how it shifts with context length.

**Notebook:**
1. On one A100, measure Qwen3-8B training step time, prefill latency and decode throughput across batch and context length.
2. Fit the model to these measurements.
3. Predict the frontier for H100, B200 and an SRAM-based design from datasheets, and compare the predictions with published InferenceX numbers.

**Sources:**
- Papers: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102); [Splitwise](https://arxiv.org/abs/2311.18677); [DistServe](https://arxiv.org/abs/2401.09670); [LLM Inference Unveiled](https://arxiv.org/abs/2402.16363).
- Benchmarks: SemiAnalysis [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) and its [dashboard](https://inferencex.semianalysis.com/).
- Teaching: [Scaling Book: inference](https://jax-ml.github.io/scaling-book/inference/).

### 5.3 Memory and storage tiers, from SRAM to disk · T3 · [Codex 1.8, 7.7]

**Builds on:** `subsec_hw-bytes` (`fig_memory_hierarchy`, `fig_bandwidth_ladder`, `fig_latency_ladder`), `subsec_hw-cpu` (pinned memory), `sec_read_write`, 4.1.

**New:**
- Every tier with its bandwidth, latency, capacity and cost per GB:
  - registers and shared memory, L2, HBM;
  - host DDR5, and LPDDR5X coherent CPU memory on Grace-class nodes;
  - CXL-attached memory;
  - local NVMe;
  - parallel file systems and object stores;
  - hard disks.
- What lives where in language-model workloads: weights, optimizer state, activations, KV cache, datasets and checkpoints.
- Offloading as moving state down a tier (ZeRO-Offload/Infinity, FlexGen), and when the link makes it pay.
- KV-cache tiers: HBM, then host memory, then SSD-backed stores shared across servers (13.3–13.4).
- Checkpoint and data I/O: sizes and write bandwidth; asynchronous and sharded writes; host-memory staging; GPUDirect Storage.
- Hard disks and object storage for datasets and checkpoint archives: sequential bandwidth, the cost of random reads, and why shards are large.

**Derivation:** When offloading pays: the overlap condition between transfer time over a link and compute time on the GPU, as a function of arithmetic intensity and link bandwidth. Applied to optimizer-state offload and to KV-cache offload.

**Notebook:**
1. On one 8× A100 node, measure host↔device bandwidth (pageable and pinned, per NUMA node).
2. Measure NVMe sequential and random reads with fio, GPUDirect Storage against a bounce buffer, and the shared file system.
3. Measure synchronous and asynchronous save and load of an 8B model's full training state, and compute the training time lost per checkpoint.

**Sources:**
- Offloading: [ZeRO-Infinity](https://arxiv.org/abs/2104.07857); [FlexGen](https://arxiv.org/abs/2303.06865).
- KV-cache stores: [Mooncake](https://arxiv.org/abs/2407.00079); [LMCache](https://github.com/LMCache/LMCache).
- Storage at cluster scale: [Fire-Flyer AI-HPC](https://arxiv.org/abs/2408.14158); [3FS](https://github.com/deepseek-ai/3FS); [Llama 3](https://arxiv.org/abs/2407.21783) (storage for checkpoints).
- Tools and documentation: [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/); [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html); [fio](https://github.com/axboe/fio).
- Reference: [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2).

### 5.4 Data paths inside a server · T3 · [Codex 4.7]

**Builds on:** `subsec_hw-interconnects` (the build box, `fig_pcie_topology`), `sec_multi_gpu` (α–β model), 5.3.

**New:**
- An eight-GPU server as a graph whose nodes are GPUs, NVSwitches, PCIe switches, CPUs, NICs and NVMe drives. SXM vs PCIe parts.
- NVLink and NVSwitch; PCIe generations and switch trees; CPU–GPU links (PCIe vs NVLink-C2C); AMD Infinity Fabric.
- GPUDirect peer-to-peer and RDMA, with one NIC per GPU; NUMA and CPU/NIC affinity.
- Copy engines vs SM-driven copies, and the latency floor of a small transfer.
- How the server's topology decides where each parallelism axis goes (6.3–6.7).

**Derivation:** α–β parameters for each link from measurements. Bisection bandwidth of a switched node vs a PCIe tree, and why all-to-all suffers far more than all-reduce on the tree.

**Notebook:**
1. Read `nvidia-smi topo -m`.
2. Measure the GPU-to-GPU bandwidth and latency matrix (p2pBandwidthLatencyTest), and host-to-GPU bandwidth per NUMA node.
3. Measure intra-node all-reduce and all-to-all with nccl-tests.
4. Refit the `sec_multi_gpu` α–β model to NVLink, and compare it with the build box.

**Sources:**
- Papers: [Evaluating Modern GPU Interconnect](https://arxiv.org/abs/1903.04611); [Demystifying NCCL](https://arxiv.org/abs/2507.04786).
- Documentation and tools: [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/); [nccl-tests](https://github.com/NVIDIA/nccl-tests); [CUDA samples](https://github.com/NVIDIA/cuda-samples) (p2pBandwidthLatencyTest).
- Teaching: [Scaling Book](https://jax-ml.github.io/scaling-book/).

### 5.5 Networks between servers · T4 · [Codex 4.7, 6.2]

**Builds on:** 5.4, `sec_multi_gpu`.

**New:**
- NICs and link speeds; InfiniBand vs RDMA over Converged Ethernet (RoCE); Ultra Ethernet.
- The latency budget of a cross-node message: NIC, switch hops, cables and software.
- Topologies: rail-optimized fat trees; oversubscription; rail-only designs that lean on the scale-up domain; congestion control and adaptive routing.
- Collectives across nodes:
  - hierarchical algorithms;
  - NCCL's ring, tree and in-network reduction (SHARP) algorithms;
  - bus bandwidth vs algorithm bandwidth;
  - all-to-all across nodes for expert parallelism.

**Derivation:**
- α–β costs of ring and tree all-reduce across nodes, and the message size at which tree beats ring.
- Hierarchical all-reduce, which uses NVLink inside a node and the network between nodes, and its speedup over a flat ring.

**Notebook:** Measure all-reduce and all-to-all across 2–4 nodes. Test the predicted ring/tree crossover, and compare flat with hierarchical all-reduce.

**Sources:**
- Production networks: "RDMA over Ethernet for Distributed AI Training at Meta Scale" (SIGCOMM 2024); "Alibaba HPN" (SIGCOMM 2024); [Rail-only](https://arxiv.org/abs/2307.12169).
- Collective algorithms: [Demystifying NCCL](https://arxiv.org/abs/2507.04786); [SHARP](https://docs.nvidia.com/networking/display/sharpv300).
- Tools and standards: [nccl-tests](https://github.com/NVIDIA/nccl-tests); [Ultra Ethernet Consortium](https://ultraethernet.org/).

### 5.6 Scale-up, scale-out and what it costs · T0 · [Codex 4.8, 7.7]

**Builds on:** 5.2, 5.4–5.5.

**New:**
- Two networks:
  - a scale-up domain with memory-class bandwidth: NVLink racks such as NVL72, TPU pods with torus links and optical circuit switches, Trainium UltraServers, UALink;
  - a scale-out network (5.5).
- How the size of the scale-up domain bounds tensor and expert parallelism, and the memory per domain available for serving large MoE models.
- Scale-across: training across several datacenters.
- Power and cooling: rack power density, liquid cooling, power capping and thermal throttling.
- Cost: cost per token from MFU, price and utilization; owned vs rented clusters. This takes over the datacenter half of the retiring `sec_hardware_buyers`.

**Derivation:**
- Which parallel layouts fit inside one domain: the bytes each axis moves per step, against link bandwidth.
- Cost per million tokens as a function of price per GPU-hour, throughput and utilization, for training and for serving.

**Notebook:**
1. A calculator that reports which layouts fit in one scale-up domain. Apply it to three published systems: an eight-GPU node, an NVL72 rack and a TPU pod slice.
2. Convert the measured MFU (6.7) and decode throughput (5.2) into dollars per million tokens, for owned and for rented hardware.

**Sources:**
- Papers: [TPU v4](https://arxiv.org/abs/2304.01433); [Insights into DeepSeek-V3: hardware](https://arxiv.org/abs/2505.09343); [Llama 3](https://arxiv.org/abs/2407.21783) (cluster design); [MegaScale](https://arxiv.org/abs/2402.15627).
- Vendor material: NVIDIA, [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) and [NVL72 for training and real-time inference](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/); [UALink Consortium](https://ualinkconsortium.org/).
- Teaching and reference: [Scaling Book](https://jax-ml.github.io/scaling-book/); [The Datacenter as a Computer](https://link.springer.com/book/10.1007/978-3-031-01761-2).

## Chapter 6 Parallelism Beyond One Node

**Label:** `chap_lm_parallelism`.

Advanced stops at one node: hand-written data parallelism, DDP, and an FSDP/ZeRO derivation that is not executed (`sec_multi_gpu`, `sec_multi_gpu_concise`). This chapter executes sharded training at scale, adds tensor, pipeline, context and expert parallelism, and composes them across nodes. It takes over the retiring `sec_training_systems` (scaling ladder, library landscape, long-run practice).

**Learning objectives:**
- Implement each form of parallelism from collectives and verify it against a single-GPU run.
- Derive the communication and memory cost of each, and where it must sit in the topology.
- Choose and launch a composed configuration across nodes, and explain its measured MFU.
- Keep a long run alive through failures, with checkpoint intervals derived from failure statistics.

**Assignment A6.**

**Teaching sources:**
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook);
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/);
- [CS336](https://cs336.stanford.edu/) (systems assignment);
- the [CMU 11-968 syllabus](https://www.cmu.edu/online/welcome/pdfs/11968-llm-systems-syllabus-f25.pdf);
- [Megatron Core guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html);
- [TorchTitan](https://github.com/pytorch/torchtitan).

### 6.1 Running jobs across machines · T4 · [Codex 6.1]

**Builds on:** `sec_multi_gpu_concise`.

**New:**
- The scaling ladder from one GPU to many nodes, and what breaks at each rung.
- Multi-node rendezvous and SLURM.
- Process groups and device meshes.
- Rank-tagged logs.
- Debugging hangs with timeouts, NCCL debug output and the flight recorder.

**Notebook:** Launch the same DDP job on one and on two nodes, then reproduce and diagnose a hang caused by mismatched collective order.

**Sources:** [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html); [DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html); [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/).

### 6.2 Sharded data parallelism at scale · T3 · [Codex 6.7, 6.8]

**Builds on:** `sec_multi_gpu_concise`.

**New:**
- FSDP2 (`fully_shard`) and wrapping granularity.
- Prefetching; resharding after forward.
- Mixed-precision policies; CPU offload.
- Predicted vs measured peak memory.

**Notebook:** Train an 8B model with FSDP2 on one 8× A100 node, sweep wrapping and prefetch, and explain the trade-offs from traces.

**Sources:** [ZeRO](https://arxiv.org/abs/1910.02054); [PyTorch FSDP](https://arxiv.org/abs/2304.11277); [fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html); [TorchTitan](https://arxiv.org/abs/2410.06511).

### 6.3 Tensor and sequence parallelism · T3 · [Codex 6.5, 6.6]

**New:**
- Column- and row-parallel linear layers.
- Attention heads across ranks.
- Sequence parallelism for norms and dropout; DTensor placements.
- Why tensor parallelism stays inside the NVLink domain.

**Derivation:** The conjugate f/g operators and their gradients. The communication volume per layer, against compute per layer.

**Notebook:** Implement a two-way tensor-parallel MLP and attention block and verify outputs and gradients. Measure TP = 2/4/8 and predict the cross-node cost from 5.5.

**Sources:** [Megatron-LM](https://arxiv.org/abs/1909.08053); [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198); [PyTorch TP](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html).

### 6.4 Pipeline parallelism · T3 · [Codex 7.1–7.3]

**New:**
- GPipe and 1F1B.
- Interleaved stages.
- Splitting backward into input- and weight-gradient work (zero bubble, DualPipe).

**Derivation:** The bubble fraction (p−1)/(m+p−1), how interleaving changes it, and the activation memory per schedule.

**Notebook:** A schedule simulator (T0), validated against a four-stage `torch.distributed.pipelining` run.

**Sources:** [GPipe](https://arxiv.org/abs/1811.06965); [Efficient Large-Scale Training](https://arxiv.org/abs/2104.04473); [Zero Bubble](https://arxiv.org/abs/2401.10241); [DualPipe](https://github.com/deepseek-ai/DualPipe).

### 6.5 Context parallelism · T3 · [Codex 7.4]

**Builds on:** `sec_attention-at-scale` (online softmax).

**New:**
- Ring attention as distributed online softmax.
- Causal load balancing.
- Ulysses all-to-all.
- Combining context with tensor parallelism.

**Derivation:** The correctness of merging partial softmax statistics across ranks, and the overlap condition under which ring communication hides behind compute.

**Notebook:** Train on 128K-token sequences across 8 GPUs, verify against dense attention on a short case, and compare ring with all-to-all.

**Sources:** [Ring Attention](https://arxiv.org/abs/2310.01889); [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509); [TorchTitan](https://arxiv.org/abs/2410.06511).

### 6.6 Expert parallelism · T3 · [Codex 3.2, 3.8, 7.5]

**Builds on:** `sec_moe`.

**New:**
- All-to-all dispatch and combine.
- Capacity factors vs dropless execution; grouped GEMMs.
- Routing skew.
- Host-synchronization pitfalls.
- Upcycling a dense model.

**Derivation:** All-to-all volume per token as a function of top-k, hidden size and expert-parallel degree, and the effect of node-limited routing.

**Notebook:** Train a small MoE with EP = 8, verify it against one GPU, and measure dispatch time as skew grows.

**Sources:** [Switch Transformers](https://arxiv.org/abs/2101.03961); [MegaBlocks](https://arxiv.org/abs/2211.15841); [DeepEP](https://github.com/deepseek-ai/DeepEP); [Sparse Upcycling](https://arxiv.org/abs/2212.05055).

### 6.7 Composing parallelism across nodes · T4 · [Codex 7.6, 7.8]

**Builds on:** 5.4–5.6, 6.2–6.6.

**New:**
- Device meshes over DP × TP × PP × CP × EP.
- Mapping mesh axes onto the topology.
- Enumerating feasible configurations.
- MFU at scale; weak vs strong scaling.
- The library landscape (TorchTitan, Megatron-Core, DeepSpeed, nanotron) and what to use at which scale.

**Notebook:** Rank configurations for a 7–8B model on 2–4 nodes with a cost model. Run the top three with TorchTitan and compare predicted with measured step times.

**Sources:** [Efficient Large-Scale Training](https://arxiv.org/abs/2104.04473); [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook); [TorchTitan](https://arxiv.org/abs/2410.06511).

### 6.8 Checkpoints, failures and long runs · T3 · [Codex 7.7]

**Builds on:** 5.3 (checkpoint I/O), 6.2.

**New:**
- Failure rates at scale and their causes; stragglers and silent data corruption. This material moved from v4's 5.5.
- Distributed checkpoints and resharding to a new world size.
- Asynchronous saves.
- Detecting and replacing failed ranks; fault-tolerant data parallelism.
- Run hygiene: alerts, loss-spike playbooks, provenance. This is the "Keeping a Long Run Alive" material from the retiring Tools appendix.

**Derivation:** The Young–Daly optimal checkpoint interval, and its higher-order correction.

**Notebook:**
1. Fit failure rates to published interruption tables, and derive checkpoint intervals for several cluster sizes from 5.3's measured checkpoint costs.
2. Kill a rank mid-run and restart from a distributed checkpoint on a different number of GPUs. Verify the loss trajectory and the data cursor.

**Sources:** [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html); [torchft](https://github.com/pytorch/torchft); [Llama 3](https://arxiv.org/abs/2407.21783); [Revisiting Reliability in Large-Scale ML Clusters](https://arxiv.org/abs/2410.21680); [MegaScale](https://arxiv.org/abs/2402.15627); Young (CACM 1974); Daly (FGCS 2006).

## Chapter 7 Fine-Tuning

**Label:** `chap_lm_finetuning`.

**Builds on:**
- `sec_parameters`: freezing parameters;
- `sec_mdl-svd-low-rank`: LoRA as a rank-r update; PiSSA;
- `sec_imitation`: SFT as behavior cloning;
- `sec_numerics` and 4.4: number formats and block scaling, the starting point for the quantizers of 7.6.

Every method is implemented from scratch first, then run with PEFT, TRL, torchtune and Unsloth.

**Learning objectives:**
- Decide between full fine-tuning, parameter-efficient methods and retrieval for a given goal.
- Implement LoRA and its variants, and predict their memory and quality trade-offs.
- Quantize frozen weights for fine-tuning, and measure the quantization error and its effect on the adapted model.
- Serialize, mask and pack instruction data correctly, and verify it token by token.
- Fine-tune models up to ~70B across GPUs, and merge or compose the results.

**Teaching sources:**
- Thinking Machines, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/);
- Sebastian Raschka, [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms);
- [HF smol course](https://huggingface.co/learn/smol-course/en/unit0/1);
- [PEFT docs](https://huggingface.co/docs/peft/index);
- Answer.AI, [FSDP-QLoRA](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html).

### 7.1 Full fine-tuning · T2 · [Codex 8.1]

**Builds on:** `sec_parameters`, `sec_read_write`, `sec_memory_precision`, `sec_adamw`.

**New:**
- The pretrain-then-adapt paradigm.
- When fine-tuning helps, and how far to unfreeze: layer freezing, discriminative learning rates. This section is the new target of the `sec_fine_tuning` references in Basics.
- Task heads vs text-to-text adaptation.
- Learning rates and epochs for pretrained LMs.
- The memory budget of full fine-tuning.
- Forgetting, measured with the Chapter 1 suite.

**Notebook:** Fully fine-tune Qwen3-1.7B on one A100, then compare unfreezing depths and a classification head against generative answers.

**Sources:** [ULMFiT](https://arxiv.org/abs/1801.06146); [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673); [torchtune](https://github.com/pytorch/torchtune).

### 7.2 Supervised instruction tuning · T1 · [Codex 8.2, 8.4]

**Builds on:** 1.3, 4.1, `sec_imitation` (`subsec_il_sft`).

**New:**
- Serializing multi-turn conversations.
- Completion-only loss masks; end-of-turn tokens.
- Packing conversations without leakage between them.

**Notebook:** Instruction-tune Qwen3-0.6B base. Print token/mask tables for edge cases, and check that inference prompts match the training prefixes token by token.

**Sources:** [InstructGPT](https://arxiv.org/abs/2203.02155); [TRL SFT](https://huggingface.co/docs/trl/sft_trainer); [Tülu 3](https://arxiv.org/abs/2411.15124).

### 7.3 Instruction data · T1 · [Codex 8.3]

**Builds on:** 3.3, 3.5.

**New:**
- Human, templated and synthetic instructions; Self-Instruct and Magpie.
- Selection by quality, difficulty and diversity.
- Decontamination.

**Notebook:** Compare a random subset with a curated subset at equal token budgets.

**Sources:** [Self-Instruct](https://arxiv.org/abs/2212.10560); [LIMA](https://arxiv.org/abs/2305.11206); [Magpie](https://arxiv.org/abs/2406.08464); [Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture).

### 7.4 LoRA from scratch · T1 · [Codex 8.5]

**Builds on:** `sec_mdl-svd-low-rank`.

**New:**
- A LoRA linear layer.
- Initialization and α/r scaling.
- Target modules.
- Learning rates relative to full fine-tuning.
- Merging and unmerging; memory accounting.

**Derivation:** Why the update scale must be α/√r for the update to stay stable as the rank grows (rsLoRA), and the learning-rate ratio between the A and B matrices (LoRA+).

**Notebook:** Implement LoRA, check that merged and unmerged outputs agree, and compare ranks and targets against full fine-tuning.

**Sources:** [LoRA](https://arxiv.org/abs/2106.09685); [rsLoRA](https://arxiv.org/abs/2312.03732); [LoRA+](https://arxiv.org/abs/2402.12354); [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/).

### 7.5 Beyond LoRA · T1 · [Codex 8.5, 8.6]

**New:** the PEFT family, organized by what each method changes:
- **Initialization:** PiSSA.
- **Parameterization:** DoRA, VeRA.
- **Where parameters enter:** adapters, prefix and prompt tuning, IA³.
- **Optimizer state:** GaLore.

**Notebook:** Compare DoRA, prompt tuning and GaLore with the 7.4 baseline at matched memory.

**Sources:** [DoRA](https://arxiv.org/abs/2402.09353); [PiSSA](https://arxiv.org/abs/2404.02948); [VeRA](https://arxiv.org/abs/2310.11454); [Adapters](https://arxiv.org/abs/1902.00751); [Prefix-Tuning](https://arxiv.org/abs/2101.00190); [Prompt Tuning](https://arxiv.org/abs/2104.08691); [IA³](https://arxiv.org/abs/2205.05638); [GaLore](https://arxiv.org/abs/2403.03507).

### 7.6 Quantization for fine-tuning: QLoRA · T1 · [Codex 8.7, 12.1]

**Builds on:** `sec_numerics` (formats), 4.4 (block scaling), 7.4.

Quantization is first needed here, so this section introduces it. It covers only what fine-tuning needs: storing frozen weights in fewer bits to save memory. 11.1 builds on it for inference, where activations and speed matter as well.

**New:**
- Quantizing a weight tensor: absmax and affine integer grids (scale and zero point), per-tensor vs per-block scales, and how to measure the error.
- NF4: a 16-value grid placed at quantiles of a normal distribution. Double quantization of the block scales.
- Dequantizing on the fly inside the matmul.
- Gradients through frozen quantized weights into LoRA adapters; an initialization that absorbs the quantization error (LoftQ).
- 8-bit and paged optimizers.

**Derivation:**
- The expected squared error of uniform quantization (Δ²/12), and of blockwise absmax as the block size varies.
- NF4's grid as equal-probability bins of a normal distribution, which maximize the entropy of the codes. How this differs from the MSE-optimal Lloyd–Max grid.

**Notebook:**
1. Implement absmax, blockwise and NF4 quantizers, and compare their error on Qwen3 weights.
2. QLoRA-fine-tune an 8B model within 15 GiB.
3. Compare it with BF16 LoRA on 1.7B, and with LoftQ initialization.

**Sources:** [QLoRA](https://arxiv.org/abs/2305.14314); [8-bit Optimizers](https://arxiv.org/abs/2110.02861); [LoftQ](https://arxiv.org/abs/2310.08659); [k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720); [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes); Maarten Grootendorst, [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization).

### 7.7 Fine-tuning toolchains · T1 · [new]

**Builds on:** 7.4.

**New:**
- One recipe expressed in PEFT+TRL, torchtune and Unsloth; Axolotl and LLaMA-Factory as configuration-driven wrappers.
- Configuration files as experiment records.
- Checking a library run against the from-scratch oracle: masks, schedules, effective batch.
- Where throughput differences come from.

**Notebook:** Reproduce the 7.4 run in two toolchains, match the first steps' losses, and compare throughput.

**Sources:** [PEFT](https://huggingface.co/docs/peft/index); [TRL](https://huggingface.co/docs/trl/index); [torchtune](https://github.com/pytorch/torchtune); [Unsloth](https://github.com/unslothai/unsloth); [Axolotl](https://github.com/axolotl-ai-cloud/axolotl); [LLaMA-Factory](https://arxiv.org/abs/2403.13372).

### 7.8 Fine-tuning large models across GPUs · T3 · [new]

**Builds on:** 6.2.

**New:**
- Full fine-tuning and LoRA of 30–70B models on one node.
- LoRA and QLoRA under FSDP; ZeRO-3 with offload.
- Sharded adapters.

**Notebook:** LoRA-fine-tune a ~70B dense model on 8× A100. Report memory per GPU and step time, and estimate the cost of full fine-tuning.

**Sources:** [FSDP-QLoRA](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html); [ZeRO-Infinity](https://arxiv.org/abs/2104.07857); [torchtune](https://github.com/pytorch/torchtune).

### 7.9 Merging and retaining capabilities · T1 · [Codex 8.8]

**New:**
- Forgetting and replay.
- Task vectors, TIES and DARE; model soups.
- Composing adapters.
- Fine-tuning vs retrieval (2.5) for new knowledge.

**Notebook:** Train adapters on two tasks, merge them several ways, and evaluate both tasks and the general suite.

**Sources:** [Task Arithmetic](https://arxiv.org/abs/2212.04089); [TIES](https://arxiv.org/abs/2306.01708); [DARE](https://arxiv.org/abs/2311.03099); [Model Soups](https://arxiv.org/abs/2203.05482).

## Chapter 8 Preference Optimization and Reinforcement Learning

**Label:** `chap_lm_rl`.

**Builds on:**
- `chap_reinforcement_learning` and `chap_deep_rl`;
- REINFORCE, baselines, RLOO: `sec_policygradient`, `sec_baselines`;
- PPO, GAE: `sec_ppo`, `sec_actorcritic`;
- Bradley–Terry and the KL-regularized optimum: `sec_regularized`;
- the token MDP and GRPO: `sec_rl_sequences`.

The chapter uses `tab_rl_notation_contract` verbatim. What is new is running these methods on real language models.

**Learning objectives:**
- Train a reward model and diagnose what it rewards.
- Derive DPO and its relatives from the KL-regularized objective, and state what each one assumes.
- Run PPO and GRPO on real models, and explain the effect of each implementation choice on bias and variance.
- Design an RL training system around the fact that generation dominates the cost.

**Assignment A7.**

**Teaching sources:**
- Nathan Lambert, [RLHF Book](https://rlhfbook.com/);
- HF, [Illustrating RLHF](https://huggingface.co/blog/rlhf);
- John Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html);
- Lilian Weng, [Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/);
- [CS336](https://cs336.stanford.edu/) (alignment assignment);
- [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/).

### 8.1 Reward models for language · T2 · [Codex 9.1]

**Builds on:** `sec_regularized`.

**New:**
- A reward head on an LM backbone.
- Preference data and annotation instructions.
- Calibration; length and style bias.
- Accuracy vs usefulness as an optimization target.

**Notebook:** Train a reward model on Qwen3-0.6B, measure accuracy and its correlation with length, and probe its failures.

**Sources:** [InstructGPT](https://arxiv.org/abs/2203.02155); [RewardBench](https://arxiv.org/abs/2403.13787); [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf).

### 8.2 RLHF with PPO for language models · T2 · [Codex 9.2, 9.4]

**Builds on:** `sec_ppo`, `sec_actorcritic`, `sec_rl_sequences`.

**New:**
- Actor, critic, reference and reward models in memory.
- The per-token KL penalty.
- Reward on the final token; whitening.
- The implementation details that matter.

**Derivation:** Bias and variance of the k1, k2 and k3 KL estimators (`sec_mdl-divergences-distances`).

**Notebook:** PPO-RLHF on a 0.6B instruct model. Verify advantages on hand-computable trajectories, and report memory and time per role.

**Sources:** [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html); [The N+ Implementation Details of RLHF with PPO](https://arxiv.org/abs/2403.17031); [OpenRLHF](https://arxiv.org/abs/2405.11143).

### 8.3 Direct preference optimization · T2 · [Codex 9.3]

**Builds on:** `sec_regularized`, `sec_mdl-constrained-optimization-duality`.

**New:**
- Implicit rewards; dependence on the reference model.
- Likelihood displacement.

**Derivation:** DPO from the closed-form KL-regularized optimum and the Bradley–Terry model. This delivers the derivation that `sec_rl_sequences` defers.

**Notebook:** Implement DPO, train on the 8.1 pairs, and plot chosen and rejected log-probabilities alongside win rate.

**Sources:** [DPO](https://arxiv.org/abs/2305.18290); [Unintentional Unalignment](https://arxiv.org/abs/2410.08847); [TRL DPO](https://huggingface.co/docs/trl/dpo_trainer).

### 8.4 The preference-optimization family · T1 · [Codex 9.3]

**New:** methods organized by the assumption each one changes:
- IPO: bounded preferences.
- KTO: unpaired feedback.
- SimPO and ORPO: no reference model.
- Online and iterative DPO: fresh samples.

**Notebook:** Compare offline and online DPO on the same prompts, judged by a small model.

**Sources:** [IPO](https://arxiv.org/abs/2310.12036); [KTO](https://arxiv.org/abs/2402.01306); [SimPO](https://arxiv.org/abs/2405.14734); [ORPO](https://arxiv.org/abs/2403.07691); [Online AI Feedback](https://arxiv.org/abs/2402.04792).

### 8.5 RL with verifiable rewards · T2 · [Codex 9.5–9.7]

**Builds on:** `sec_baselines` (`subsec_pg_loss_divisor`), `sec_rl_sequences`.

**New:**
- Robust answer verification.
- GRPO on multi-token responses.
- Token- vs sequence-level aggregation and its length bias (Dr. GRPO).
- DAPO: clip-higher, dynamic sampling, overlong filtering.
- Sequence-level importance ratios (GSPO) and clipped importance weights (CISPO), and why they matter for MoE policies.
- Entropy collapse.
- Predictable scaling of RL compute.

**Derivation:** How each aggregation choice weights tokens, and the resulting gradient bias with respect to length.

**Notebook:** GRPO on Qwen3-0.6B/1.7B base for arithmetic or GSM8K-style problems. Ablate one aggregation choice, and plot accuracy, length and entropy.

**Sources:** [DeepSeekMath](https://arxiv.org/abs/2402.03300); [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783); [DAPO](https://arxiv.org/abs/2503.14476); [GSPO](https://arxiv.org/abs/2507.18071); [MiniMax-M1](https://arxiv.org/abs/2506.13585); [The Art of Scaling RL Compute](https://arxiv.org/abs/2510.13786); [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer).

### 8.6 RL training systems · T3 · [Codex 9.8, 14.7]

**New:**
- Generation dominates RL compute.
- An inference engine for rollouts and a trainer for updates; colocated vs disaggregated placement.
- Weight synchronization.
- Asynchronous rollouts with off-policy corrections (`sec_mdl-bayesian-computation`, importance sampling).

**Notebook:** GRPO with vLLM rollouts on 8× A100, in colocated and split layouts. Attribute time to generation, scoring, update and synchronization, and measure policy lag.

**Sources:** [HybridFlow/verl](https://arxiv.org/abs/2409.19256); [OpenRLHF](https://arxiv.org/abs/2405.11143); [AReaL](https://arxiv.org/abs/2505.24298); [verl](https://github.com/verl-project/verl).

### 8.7 ◇ Overoptimization, safety and evaluating post-trained models · T1 · [Codex 9.9]

**Builds on:** `sec_regularized` (Goodhart curves), `sec_rl_sequences` (`subsec_rl_sequences_hacking`).

**New:**
- Exploits specific to language models: length, sycophancy, format.
- Reward-model ensembles; length control.
- Training against written principles (Constitutional AI, deliberative alignment).
- Over-refusal and regression suites.

**Notebook:** Optimize against a deliberately biased reward and evaluate with an independent checker.

**Sources:** [Reward Model Overoptimization](https://arxiv.org/abs/2210.10760); [Sycophancy](https://arxiv.org/abs/2310.13548); [Constitutional AI](https://arxiv.org/abs/2212.08073); [Deliberative Alignment](https://arxiv.org/abs/2412.16339).

## Chapter 9 Distillation and Reasoning

**Label:** `chap_lm_reasoning`.

**Learning objectives:**
- Choose a distillation objective (forward, reverse or generalized divergence; offline or on-policy) and justify it.
- Build reasoning models from SFT, RL and distillation, and measure what each stage changes.
- Spend test-time compute efficiently, both in parallel (sampling and verification) and sequentially (longer or revised traces).

**Teaching sources:**
- Thinking Machines, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/);
- Lilian Weng, [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/);
- Sebastian Raschka, [Reasoning from Scratch](https://github.com/rasbt/reasoning-from-scratch);
- Jay Alammar, [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1);
- [Open R1](https://github.com/huggingface/open-r1).

### 9.1 Knowledge distillation for language models · T1 · [Codex 10.1–10.3]

**Builds on:** `sec_mdl-information_theory`, `sec_mdl-divergences-distances`.

**New:**
- Sequence-level vs token-level distillation.
- Top-k teacher logits.
- Forward KL, reverse KL and generalized JSD on token distributions.
- Teachers with different tokenizers.

**Notebook:** Distill Qwen3-1.7B into 0.6B with sequence-level and logit-level objectives at matched compute.

**Sources:** [Hinton et al.](https://arxiv.org/abs/1503.02531); [Sequence-Level KD](https://arxiv.org/abs/1606.07947); [MiniLLM](https://arxiv.org/abs/2306.08543).

### 9.2 On-policy distillation · T2 · [Codex 10.4–10.7]

**Builds on:** 9.1, `sec_imitation` (`subsec_il_dagger`).

**New:**
- The teacher scores the student's own samples.
- Reverse KL as a dense per-token reward.
- GKD.
- Self-distillation with privileged context (OPSD).
- Cost relative to RL.

**Derivation:** On-policy distillation as DAgger with a soft expert, and why it removes the compounding error of offline distillation.

**Notebook:** Compare offline distillation, on-policy distillation and GRPO (8.5) on one math task at matched generated tokens.

**Sources:** [GKD](https://arxiv.org/abs/2306.13649); [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/); [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734).

### 9.3 Training reasoning models · T3 (T2 reduced) · [Codex 10.2, 11.2]

**New:**
- Pipelines: cold-start SFT, RL with verifiable rewards, rejection sampling, distillation.
- R1-Zero-style RL from a base model.
- Prolonged RL.
- What changes and what does not.

**Notebook:** Compare SFT on traces with RL from base on one problem family at matched compute, and evaluate on unseen templates.

**Sources:** [DeepSeek-R1](https://arxiv.org/abs/2501.12948); [ProRL](https://arxiv.org/abs/2505.24864); [GSM-Symbolic](https://arxiv.org/abs/2410.05229); [Open R1](https://github.com/huggingface/open-r1).

### 9.4 Verifiers and process supervision · T2 · [Codex 11.3]

**New:**
- Outcome vs process reward models.
- Monte Carlo step labels.
- Generative verifiers.
- Calibration across generators.

**Notebook:** Train outcome and step-level scorers, and use each to rerank the same candidate pools.

**Sources:** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050); [PRM800K](https://github.com/openai/prm800k); [Math-Shepherd](https://arxiv.org/abs/2312.08935).

### 9.5 Parallel test-time compute · T1 · [Codex 11.4]

**New:**
- Self-consistency, best-of-N, weighted voting.
- pass@k vs deployable selection.
- Correlated samples.

**Derivation:** Majority-vote accuracy for independent samples, and how correlation between samples erodes it.

**Notebook:** Plot accuracy against generated tokens for majority vote, verifier reranking and the oracle.

**Sources:** [Self-Consistency](https://arxiv.org/abs/2203.11171); [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314); [Large Language Monkeys](https://arxiv.org/abs/2407.21787).

### 9.6 Sequential and adaptive test-time compute · T1 · [Codex 11.5, 11.6]

**New:**
- Budget forcing.
- Revision and early stopping.
- Allocating compute by difficulty.
- Many samples from a small model vs one answer from a large one.

**Notebook:** Sweep budget forcing, then fit a difficulty-based allocation rule on validation data and freeze it for the test set.

**Sources:** [s1](https://arxiv.org/abs/2501.19393); [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314).

### 9.7 Efficient reasoning · T2 · [Codex 10.6, 11.7]

**New:**
- Overthinking.
- Length penalties.
- Distilling into shorter traces.
- Accuracy–token frontiers by difficulty.

**Notebook:** Train a length-penalized variant and compare its frontier with the 8.5 model.

**Sources:** [Overthinking](https://arxiv.org/abs/2412.21187); [Kimi k1.5](https://arxiv.org/abs/2501.12599); [CRISP](https://arxiv.org/abs/2603.05433).

## Chapter 10 Agents and Tool Use

**Label:** `chap_lm_agents`.

Current frontier models are trained and evaluated as agents: they call tools, run code, search, and act over many turns. Three parts of the book support this chapter:

- **Advanced:** the MDP (`sec_mdp`), the token-level MDP and reward hacking (`sec_rl_sequences`), and compounding error (`sec_imitation`).
- **This part:** chat templates (1.3), retrieval and RAG (2.4–2.5), RL with verifiable rewards and RL systems (8.5–8.6), and reasoning (Chapter 9).

**Learning objectives:**
- Add tool calling to a model, with outputs guaranteed to parse.
- Build an agent harness, and measure how the interface and context management change success.
- Train a multi-turn agent with RL, with environment tokens masked from the loss.
- Evaluate agents for reliability and cost, and defend them against prompt injection.

**Teaching sources:**
- [Berkeley LLM Agents](https://llmagents-learning.org/f24) and [Advanced LLM Agents](https://rdi.berkeley.edu/adv-llm-agents/sp25);
- [HF Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) and [MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction);
- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) and [Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents);
- Lilian Weng, [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/);
- Chip Huyen, [Agents](https://huyenchip.com/2025/01/07/agents.html).

### 10.1 Tool calling · T1 · [new]

**Builds on:** 1.3, 7.2.

**New:**
- Tool schemas in the chat template; parsing calls; returning results as turns.
- Parallel and nested calls.
- Training data for tool use: self-supervised insertion (Toolformer), synthetic API traces.
- MCP as a protocol between models and tools.
- Measuring function-call accuracy.

**Notebook:** Give Qwen3-1.7B a calculator, a Python sandbox and the 2.4 retriever, and measure call accuracy on a small BFCL-style set. Fine-tune on synthetic tool traces and measure again.

**Sources:** [Toolformer](https://arxiv.org/abs/2302.04761); [Gorilla](https://arxiv.org/abs/2305.15334); [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html); [tool use in chat templates](https://huggingface.co/docs/transformers/chat_extras); [Model Context Protocol](https://modelcontextprotocol.io/).

### 10.2 Structured outputs and constrained decoding · T0 · [Codex 13.9]

**Builds on:** `sec_decoding`, 3.6.

**New:**
- Token-level finite-state masks.
- Grammars and JSON schemas.
- Tokens that span several automaton transitions.
- How masking changes the output distribution, and when it hurts accuracy.

**Derivation:** Building a token-level automaton from a character-level one, and the distribution that masked sampling actually produces.

**Notebook:** A tokenizer-aware finite-state masker for a JSON schema, used for the 10.1 tool arguments.

**Sources:** [Outlines](https://arxiv.org/abs/2307.09702); [XGrammar](https://arxiv.org/abs/2411.15100); [outlines code](https://github.com/dottxt-ai/outlines).

### 10.3 Agent loops and harnesses · T1 · [new]

**Builds on:** `sec_mdp`, `sec_imitation` (compounding error), 10.1–10.2.

**New:**
- The observe–think–act loop (ReAct).
- Code as the action space (CodeAct).
- Agent–computer interfaces.
- Context management over long horizons: truncation, summarization and compaction, external memory, subagents.
- Accounting for cost and latency.
- A taxonomy of failures.

**Notebook:** A ~100-line agent loop with a sandboxed shell, run on a small suite of repository-repair tasks. Ablate the tool interface and context compaction, and report success, steps and tokens.

**Sources:** [ReAct](https://arxiv.org/abs/2210.03629); [CodeAct](https://arxiv.org/abs/2402.01030); [SWE-agent](https://arxiv.org/abs/2405.15793); [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent); [smolagents](https://github.com/huggingface/smolagents); [Writing Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents).

### 10.4 Training agents with reinforcement learning · T3 · [new]

**Builds on:** `sec_rl_sequences`, 8.5–8.6, 10.3.

**New:**
- Multi-turn episodes as MDPs, with environment tokens masked from the policy loss.
- Trajectory- vs turn-level rewards.
- Environments as code: sandboxes and containers, and their throughput.
- Asynchronous rollouts at long horizons.
- Search agents and software-engineering agents.
- Synthesizing tasks and environments at scale.
- Reward hacking in agentic settings (`subsec_rl_sequences_hacking`).

**Derivation:** The policy gradient for a multi-turn episode in which only the agent's tokens are actions, and the resulting loss mask.

**Notebook:** Train Qwen3-1.7B with GRPO as a search agent over the 2.4 retriever on 8× A100 (reduced T1 configuration with 0.6B). Compare it with prompting alone and with single-turn RAG (2.5).

**Sources:** [Search-R1](https://arxiv.org/abs/2503.09516); [SWE-RL](https://arxiv.org/abs/2502.18449); [Kimi K2](https://arxiv.org/abs/2507.20534) (agentic data synthesis); [AReaL](https://github.com/inclusionAI/AReaL); [verl](https://github.com/verl-project/verl).

### 10.5 Evaluating and securing agents · T1 · [new]

**Builds on:** 1.5, `sec_mdl-statistics`.

**New:**
- Benchmarks: SWE-bench Verified, τ-bench and τ²-bench, Terminal-Bench, WebArena.
- pass@k vs pass^k reliability; cost-normalized scores.
- Harness sensitivity; reproducibility of environments.
- Indirect prompt injection.
- Least-privilege tools, sandboxing and confirmation gates.

**Derivation:** pass^k from per-task success rates, and why it falls far faster than pass@k rises.

**Notebook:** Evaluate the 10.3 agent with pass^k. Measure attack success under injected tool outputs, with and without a defense.

**Sources:** [SWE-bench](https://arxiv.org/abs/2310.06770); [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/); [τ-bench](https://arxiv.org/abs/2406.12045); [τ²-bench](https://arxiv.org/abs/2506.07982); [Terminal-Bench](https://www.tbench.ai/); [WebArena](https://arxiv.org/abs/2307.13854); [Indirect prompt injection](https://arxiv.org/abs/2302.12173); [AgentDojo](https://arxiv.org/abs/2406.13352).

## Chapter 11 Model Compression

**Label:** `chap_lm_compression`.

**Builds on:** formats (`sec_numerics`, `sec_hardware`), the quantizers of 7.6, the decode bound (`sec_hardware`), SVD (`sec_mdl-svd-low-rank`), conditioning (`sec_mdl-numerical-stability-conditioning`).

`chap_performance` defers quantization as compression to this chapter. 7.6 introduced quantizers for fine-tuning, where only memory matters. This chapter treats quantization for inference, where speed and activations matter as well. KV-cache compression is in 12.4.

**Learning objectives:**
- Derive the error and the speed of weight-only and weight-plus-activation quantization.
- Implement GPTQ, AWQ-style rescaling and rotation-based methods, and check each against a reference.
- Choose between post-training quantization, quantization-aware training, pruning and native low-bit formats for a deployment target.

**Teaching sources:**
- [MIT 6.5940](https://hanlab.mit.edu/courses/2026-fall-65940);
- Maarten Grootendorst, [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization);
- the [EMNLP 2025 efficient inference tutorial](https://aclanthology.org/2025.emnlp-tutorials.1/);
- [torchao](https://github.com/pytorch/ao);
- [llama.cpp](https://github.com/ggml-org/llama.cpp).

### 11.1 Quantization for inference · T1 · [Codex 12.1, 12.2]

**Builds on:** 7.6 (absmax, blockwise and NF4 quantizers), `sec_hardware` (the decode bound), 5.2.

**New:**
- What changes from 7.6: speed matters, not only memory, and activations are quantized as well as weights.
- Per-channel and per-group granularity; effective bits including scales.
- Weight-only vs weight-plus-activation quantization (W4A16, W8A8, FP8), and the kernels that make each fast.
- GGUF-style block formats for local inference.
- Bits per parameter vs parameter count at a fixed memory budget.

**Derivation:** Why decode speed tracks weight bytes (the `sec_hardware` decode bound), and why W8A8 speeds up compute-bound prefill but not bandwidth-bound decode at small batch.

**Notebook:** A group-size sweep. Measure perplexity and decode speed of int8 and int4 weight-only models and of a W8A8 model, with real kernels.

**Sources:** [Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877); [k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720); [torchao](https://github.com/pytorch/ao); [llama.cpp](https://github.com/ggml-org/llama.cpp).

### 11.2 Outliers and equivalent rescaling · T1 · [Codex 12.3]

**New:**
- Activation outlier channels.
- LLM.int8() decomposition.
- SmoothQuant.
- AWQ.

**Derivation:** The rescaling identity XW = (X diag(s)⁻¹)(diag(s) W), and the choice of s that balances the quantization error of the two factors.

**Notebook:** Plot per-channel statistics, then compare round-to-nearest, SmoothQuant and AWQ at W8A8 and W4.

**Sources:** [LLM.int8()](https://arxiv.org/abs/2208.07339); [SmoothQuant](https://arxiv.org/abs/2211.10438); [AWQ](https://arxiv.org/abs/2306.00978).

### 11.3 GPTQ and reconstruction-based quantization · T1 · [Codex 12.4]

**Builds on:** `sec_mdl-numerical-stability-conditioning`.

**New:**
- The Hessian and its Cholesky factorization.
- Lazy block updates.
- How calibration data matters.

**Derivation:** The layer-wise reconstruction objective, and the Optimal Brain Surgeon update that GPTQ applies column by column.

**Notebook:** Implement GPTQ, check it against brute force on a small matrix, and apply it to Qwen3-1.7B.

**Sources:** [GPTQ](https://arxiv.org/abs/2210.17323); [Optimal Brain Compression](https://arxiv.org/abs/2208.11580); [SparseGPT](https://arxiv.org/abs/2301.00774).

### 11.4 Rotations and vector quantization · T1 · [Codex 12.5]

**Builds on:** `sec_mdl-geometry-linear-algebraic-ops`, 2.4 (product quantization).

**New:**
- Hadamard transforms and incoherence. Neither is in the Mathematics part.
- Computational invariance.
- Learned rotations.
- Codebooks and lattices.

**Derivation:** Why a random orthogonal rotation spreads outliers, giving a max/RMS ratio of O(√log d) with high probability.

**Notebook:** Verify rotation invariance numerically. Compare W4A4 before and after a Hadamard rotation, then work a small codebook exercise.

**Sources:** [QuaRot](https://arxiv.org/abs/2404.00456); [SpinQuant](https://arxiv.org/abs/2405.16406); [QuIP#](https://arxiv.org/abs/2402.04396); [AQLM](https://arxiv.org/abs/2401.06118).

### 11.5 Quantization-aware training and native low-bit formats · T1 · [Codex 12.6]

**Builds on:** `sec_custom_layer` (custom gradients, the straight-through estimator), 4.4 (block-scaled formats).

**New:**
- Fake quantization, with `sec_custom_layer`'s straight-through estimator applied to LLM weights.
- Recovery training.
- MXFP4/NVFP4 block formats (simulated); models released natively in MXFP4.
- Ternary weights.

**Notebook:** Compare int4 QAT with post-training quantization at a matched budget, simulate MXFP4, and inspect gpt-oss-20b's MXFP4 expert weights.

**Sources:** [Straight-through estimator](https://arxiv.org/abs/1308.3432); [Microscaling](https://arxiv.org/abs/2310.10537); [gpt-oss model card](https://arxiv.org/abs/2508.10925); [BitNet b1.58](https://arxiv.org/abs/2402.17764); [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b).

### 11.6 Pruning and lossless compression · T1 · [Codex 12.7, 12.8]

**New:**
- Unstructured, 2:4 and structured pruning.
- SparseGPT and Wanda.
- Width and depth pruning followed by distillation (9.1).
- Entropy coding of BF16 exponents.

**Notebook:** Apply Wanda and 2:4 sparsity to Qwen3-1.7B and measure speed. Implement DFloat11-style exponent coding with bit-exact reconstruction.

**Sources:** [Wanda](https://arxiv.org/abs/2306.11695); [SparseGPT](https://arxiv.org/abs/2301.00774); [Minitron](https://arxiv.org/abs/2407.14679); [DFloat11](https://arxiv.org/abs/2504.11651).

## Chapter 12 Efficient Inference

**Label:** `chap_lm_inference`.

**Builds on:** `sec_kv-cache`, `sec_attention-at-scale`, `sec_compilation`, `sec_hardware`, 5.2.

This chapter covers techniques applied to a trained model within one request. Scheduling across requests is in Chapter 13. Architectural changes that must be made before pretraining, such as latent and learned sparse attention, are in Chapter 14.

**Learning objectives:**
- Take a decoder from eager mode to near the memory-bandwidth bound.
- Prove that speculative decoding is exact, and predict when it speeds decoding up.
- Train a drafter for a released model and measure its acceptance rate.
- Compress the KV cache of a trained model, and measure what the compression costs in accuracy.

**Teaching sources:**
- PyTorch, [Accelerating Generative AI II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/);
- Horace He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html);
- Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/);
- the [ICML 2024 tutorial on efficient LLM serving](https://icml.cc/virtual/2024/tutorial/35229);
- the [EMNLP 2025 efficient inference tutorial](https://aclanthology.org/2025.emnlp-tutorials.1/).

### 12.1 A fast decoder on one GPU · T1 · [Codex 13.1, 13.3]

**New:**
- Statically allocated caches.
- CUDA graphs over decode steps; fixed-shape compilation.
- Fused int4/int8 weight-only GEMMs.
- Separating CPU overhead from kernel time.

**Notebook:** Take R1 from eager mode to compiled with CUDA graphs and int4, and measure tokens per second against the `sec_hardware` decode bound.

**Sources:** [gpt-fast](https://github.com/pytorch-labs/gpt-fast); [Marlin](https://github.com/IST-DASLab/marlin); [FlashInfer](https://arxiv.org/abs/2501.01005).

### 12.2 Speculative decoding · T1 · [Codex 13.7]

**Builds on:** `sec_kv-cache` (which defers speculative decoding here), `sec_hardware`.

**New:**
- Expected tokens per step.
- A wall-clock model including draft cost and batch size.

**Derivation:** The acceptance rule and residual distribution, a proof that the output distribution is exact, and the expected accepted length.

**Notebook:** Verify exactness on a categorical toy problem. Then draft with Qwen3-0.6B for Qwen3-4B, including a regime where speculation loses.

**Sources:** [Leviathan et al.](https://arxiv.org/abs/2211.17192); [Chen et al.](https://arxiv.org/abs/2302.01318); [MagicDec](https://arxiv.org/abs/2408.11049); [SpecDecode-Bench](https://specdecode-bench.github.io/).

### 12.3 Drafters and multi-token prediction · T2 · [Codex 2.6, 13.8]

**New:**
- Medusa heads.
- EAGLE feature-level drafters.
- Multi-token prediction heads, trained with the model, reused for drafting. The pretraining side of multi-token prediction is in 14.8.
- Tree verification.

**Notebook:** Train an EAGLE-style head for R1 and compare its acceptance rate with a separate draft model. Then draft with the MTP head that ships with Qwen3.5 (1.1).

**Sources:** [Medusa](https://arxiv.org/abs/2401.10774); [EAGLE-3](https://arxiv.org/abs/2503.01840); [Multi-token prediction](https://arxiv.org/abs/2404.19737); [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (MTP module); [EAGLE code](https://github.com/SafeAILab/EAGLE).

### 12.4 Compressing the KV cache at inference time · T1 · [Codex 13.5, 13.6]

**New:**
- Per-channel keys vs per-token values.
- Quantizing keys before RoPE.
- Eviction by accumulated attention.
- Evaluation beyond needle-in-a-haystack tests.

**Notebook:** Sweep KV bit width and eviction budget on R1, evaluated with RULER-style and QA tasks.

**Sources:** [KIVI](https://arxiv.org/abs/2402.02750); [KVQuant](https://arxiv.org/abs/2401.18079); [H2O](https://arxiv.org/abs/2306.14048); [SnapKV](https://arxiv.org/abs/2404.14469).

## Chapter 13 Serving Systems

**Label:** `chap_lm_serving`.

The retiring `sec_model_serving` showed how to use vLLM and SGLang. This chapter uses them as reference implementations and builds the mechanisms inside them.

**Learning objectives:**
- Specify a serving workload and its service-level objectives, and measure goodput.
- Build continuous batching and a paged KV allocator, and prove their invariants.
- Choose parallelism and prefill/decode placement for multi-GPU serving from measurements.

**Assignment A8.**

**Teaching sources:**
- [Inside vLLM](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html);
- LMSYS, [Large-scale expert parallelism with prefill/decode disaggregation](https://lmsys.org/blog/2025-05-05-large-scale-ep/);
- the [ICML 2024 tutorial on efficient LLM serving](https://icml.cc/virtual/2024/tutorial/35229);
- the [CMU 11-968 syllabus](https://www.cmu.edu/online/welcome/pdfs/11968-llm-systems-syllabus-f25.pdf);
- the [vLLM docs](https://docs.vllm.ai/) and [SGLang docs](https://docs.sglang.ai/).

### 13.1 Workloads and service objectives · T1 · [Codex 14.1]

**New:**
- Running vLLM and SGLang behind an OpenAI-compatible API as reference servers, and local serving with llama.cpp. This absorbs the usage material of `sec_model_serving`.
- Arrival processes.
- TTFT, inter-token latency, throughput and goodput under SLOs.
- Open- vs closed-loop load.

**Derivation:** Little's law and an M/G/1 approximation, used as a first model of queueing delay under load.

**Notebook:** A load generator that plots latency percentiles against load, for static batching and for a reference server.

**Sources:** [DistServe](https://arxiv.org/abs/2401.09670); [Sarathi-Serve](https://arxiv.org/abs/2403.02310); [vLLM](https://github.com/vllm-project/vllm); [SGLang](https://github.com/sgl-project/sglang).

### 13.2 Continuous batching and chunked prefill · T1 · [Codex 14.2]

**New:**
- Iteration-level scheduling.
- Mixing prefill and decode; chunked prefill.
- Admission control and preemption.
- Batch-invariant kernels when outputs must be reproducible.

**Notebook:** A continuous-batching scheduler around 12.1, compared with static batching and with vLLM on the same trace.

**Sources:** [Orca](https://www.usenix.org/conference/osdi22/presentation/yu); [Sarathi-Serve](https://arxiv.org/abs/2403.02310); [Defeating Nondeterminism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

### 13.3 Paged KV memory and prefix caching · T1 · [Codex 14.3, 14.4]

**New:**
- Block tables; copy-on-write.
- Radix-tree prefix reuse; cache-aware scheduling.
- Offloading KV blocks to host memory and SSD (5.3).
- Caching recurrent state for hybrid models.

**Derivation:** Fragmentation bounds for fixed-size blocks vs contiguous allocation.

**Notebook:** A block allocator with invariant tests and a paged lookup, and a measurement of prefix caching under shared system prompts.

**Sources:** [PagedAttention](https://arxiv.org/abs/2309.06180); [SGLang](https://arxiv.org/abs/2312.07104); [Mooncake](https://arxiv.org/abs/2407.00079).

### 13.4 Serving across GPUs and nodes · T3 · [Codex 14.5]

**Builds on:** Chapter 6, 5.2, 5.4–5.6.

**New:**
- Tensor and expert parallelism for inference; wide expert parallelism.
- Prefill/decode disaggregation and KV transfer; KV-cache stores.
- Routing across replicas.

**Notebook:** Serve Qwen3-30B-A3B on 8× A100 with TP vs EP, and with colocated vs disaggregated prefill/decode. Find the crossover points.

**Sources:** [DistServe](https://arxiv.org/abs/2401.09670); [Splitwise](https://arxiv.org/abs/2311.18677); [Mooncake](https://arxiv.org/abs/2407.00079); [Dynamo](https://github.com/ai-dynamo/dynamo); [LMCache](https://github.com/LMCache/LMCache).

### 13.5 ◇ Serving many models · T1 · [Codex 14.6]

**New:**
- Batched multi-LoRA computation and adapter paging.
- Cascades and routing between models.

**Notebook:** Serve several 7.4 adapters over one base model with heterogeneous batches.

**Sources:** [S-LoRA](https://arxiv.org/abs/2311.03285); [Punica](https://arxiv.org/abs/2310.18547); [RouteLLM](https://arxiv.org/abs/2406.18665).

## Chapter 14 Advanced Architectures

**Label:** `chap_lm_architectures`.

Advanced builds the transformer that current models start from:
- RoPE, GQA, RMSNorm, QK-norm and SwiGLU (`sec_transformer-block`, `sec_kv-cache`);
- MoE with loss-free balancing and shared experts (`sec_moe`);
- sliding windows and attention sinks (`sec_kv-cache`);
- SSMs, DeltaNet and hybrid recipes (`chap_modern_rnn`).

This chapter covers the changes released models make beyond that baseline, along one design axis per section. For each change it covers what it saves, what it costs, and why a team with a given training and serving setup (Chapter 5) would make it.

**Designed to grow.** The chapter is organized by axis, not by model, so new releases rarely need a new section:
- **A release that moves an existing axis** adds a row to the design table of 14.1. If it changes the analysis, it also adds a paragraph to that axis's section.
- **A release that opens a new axis** gets a new section.
- **Upkeep.** The design table and its plots are regenerated from configuration files, so they can be refreshed each edition without rewriting prose. This is the only chapter that treats technical reports in depth.

**Learning objectives:**
- Reconstruct a model's parameters, active compute and cache bytes per token from its configuration file, and read a technical report critically.
- For each design axis (expert sparsity, KV compression, attention form, sequence mixing, residual paths), derive what it saves and what it costs, and measure both at small scale.
- Map every design decision in a current model to the section that teaches it, and explain it in terms of the hardware it was trained and served on.
- Design a model and training plan under a fixed budget, and defend it with measurements.

**Teaching sources:**
- Sebastian Raschka, [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) and [From GPT-2 to gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the);
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (architecture ablations);
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/);
- Hugging Face, [Mixture of Experts Explained](https://huggingface.co/blog/moe);
- Songlin Yang, [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/), and [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

### 14.1 Reading an architecture · T0 · [Codex 16.1, 16.3]

**Builds on:** 1.1 (configuration files), `sec_scaling-laws`, `sec_kv-cache`, `sec_moe`.

**New:**
- The design axes of a decoder; each later section of this chapter covers one:
  - token mixing (full, local, sparse, latent or linear attention);
  - KV layout;
  - feed-forward (dense or MoE);
  - residual paths and normalization;
  - positions;
  - output heads (multi-token prediction);
  - tokenization.
- A design table compiled from the configuration files of released models, with one column per axis, kept as data and regenerated each edition.
- Reading a technical report:
  - a schema for reports;
  - disclosed vs inferred facts;
  - tracing claims to ablations;
  - reconciling the report with the configuration file.
- A worked line of releases from DeepSeek, as a sequence of single-axis changes:
  - V2 adds MLA (14.3);
  - V3 adds auxiliary-loss-free MoE, MTP, FP8 and DualPipe (`sec_moe`, 12.3, 4.4, 6.4);
  - V3.2 adds learned sparse attention (14.5);
  - V4.1-Flash adds KV compression (12.4).

**Derivation:** The following, as functions of configuration fields, for dense, MoE, MLA and hybrid models:
- total parameters;
- active parameters per token;
- FLOPs per token;
- cache bytes per token: KV for attention layers, fixed state for linear layers.

**Notebook:**
1. Parse the configurations of about a dozen released models into the design table.
2. Check the derived parameter counts against each safetensors index.
3. Plot how each axis has moved over release dates.
4. Fill the report schema for one report, and reconstruct DeepSeek's state and communication budgets from its configuration files.

**Sources:**
- Technical reports: [Qwen3](https://arxiv.org/abs/2505.09388); [DeepSeek-V2](https://arxiv.org/abs/2405.04434); [DeepSeek-V3](https://arxiv.org/abs/2412.19437); [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556); [DeepSeek-V4.1-Flash](https://arxiv.org/html/2609.19969v1); [Llama 3](https://arxiv.org/abs/2407.21783); [gpt-oss model card](https://arxiv.org/abs/2508.10925).
- Teaching: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison).

### 14.2 Mixture-of-experts design at scale · T2 · [Codex 3.2, 3.8]

**Builds on:** `sec_moe` (balancing with and without a loss, fine-grained and shared experts), `sec_scaling-laws`, 6.6 (expert parallelism), 5.2.

**New:**
- Sparsity as a design axis: total vs active parameters, expert granularity, and the scaling laws that set them.
- Router design in released models:
  - sigmoid scores;
  - normalized top-k weights;
  - group- and node-limited routing, co-designed with the network (6.6).
- Dense early layers, and why models keep them.
- Variable compute per token through zero-computation experts.
- Shrinking expert traffic: LatentMoE projects tokens to a smaller dimension before dispatch.
- What experts specialize in, measured on a released model.
- MoE at inference time: total parameters set memory, active parameters set traffic (5.2).

**Derivation:**
- The optimal sparsity at fixed training FLOPs, from a parametric scaling law in total and active parameters.
- All-to-all bytes per token as a function of top-k, hidden size and latent dimension.

**Notebook:**
1. At fixed active compute, sweep the sparsity and granularity of the `sec_moe` GPT at R2 scale, and fit loss against total parameters.
2. Log the routing decisions of Qwen3-30B-A3B over code, math and prose, and measure expert specialization and load.

**Sources:**
- Expert design: [DeepSeekMoE](https://arxiv.org/abs/2401.06066); [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664) (taught in `sec_moe`; cited for the comparison).
- Sparsity scaling laws: [Scaling Laws for Fine-Grained MoE](https://arxiv.org/abs/2402.07871); [Parameters vs FLOPs](https://arxiv.org/abs/2501.12370).
- Released designs: [LongCat-Flash](https://arxiv.org/abs/2509.01322) (zero-computation experts); [LatentMoE](https://arxiv.org/abs/2601.18089); [Kimi K3](https://arxiv.org/html/2607.24653v1) (16 of 896 experts with Stable LatentMoE); [Nemotron 3](https://arxiv.org/abs/2512.20856); [Kimi K2](https://arxiv.org/abs/2507.20534); [DeepSeek-V3](https://arxiv.org/abs/2412.19437).
- Model and teaching: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); [Mixture of Experts Explained](https://huggingface.co/blog/moe).

### 14.3 Latent attention and cross-layer KV sharing · T1 · [Codex 3.4, 2.5]

**Builds on:** `sec_kv-cache` (`subsec_kv-cache-low-rank`: the low-rank KV sketch and why MLA needs decoupled RoPE), `sec_mdl-svd-low-rank`.

**New:**
- Trainable MLA with decoupled RoPE.
- Sharing KV across layers.
- Converting MHA/GQA checkpoints to MLA.

**Derivation:** Absorbing the up-projections into the query and output projections for decode, and why RoPE has to be decoupled for the absorption to work.

**Notebook:** Build MLA with explicit and absorbed paths and verify that they agree. Compare cache bytes and latency with GQA.

**Sources:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434); [Cross-Layer Attention](https://arxiv.org/abs/2405.12981); [MHA2MLA](https://arxiv.org/abs/2502.14837).

### 14.4 Refinements of softmax attention · T2 · [new]

**Builds on:** `sec_kv-cache` (sliding windows and attention sinks), `sec_positional-information` (RoPE, ALiBi and NoPE), `sec_transformer-block` (QK-norm), 4.3.

**New:**
- Sinks and gates:
  - attention sinks as a consequence of softmax normalization;
  - massive activations;
  - learned sink logits (gpt-oss);
  - output gating as a sink-free alternative (gated attention in Qwen3-Next and Qwen3.5).
- Positions at scale: partial RoPE; interleaving NoPE layers with RoPE layers; the effect of each on length extrapolation.
- Local–global layouts: window size and the ratio of local to global layers (Gemma 3, gpt-oss) as a dial on KV memory.
- Differential attention, in brief.

**Derivation:**
- A softmax head must place its mass somewhere, so a head with nothing to attend to puts it on a sink token.
- A learned sink logit is equivalent to an extra key with zero value.
- An output gate lets a head emit zero without a sink.

**Notebook:**
1. At R2 scale, train a baseline and three variants: a learned sink, a gated attention output, and partial RoPE with interleaved NoPE layers.
2. Compare loss, attention mass on the first token, and maximum activation magnitude.
3. Compare accuracy beyond the training length.

**Sources:**
- Sinks and gates: [StreamingLLM](https://arxiv.org/abs/2309.17453); [Massive Activations](https://arxiv.org/abs/2402.17762); [Gated Attention](https://arxiv.org/abs/2505.06708).
- Positions and layouts: [Rope to Nope and Back Again](https://arxiv.org/abs/2501.18795); [Gemma 3](https://arxiv.org/abs/2503.19786); [gpt-oss model card](https://arxiv.org/abs/2508.10925); [SmolLM3 blog](https://huggingface.co/blog/smollm3) (NoPE layers).
- Variants: [Differential Transformer](https://arxiv.org/abs/2410.05258).
- Model: [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct).

### 14.5 Learned sparse attention · T1 · [Codex 3.5]

**Builds on:** `sec_attention-at-scale` (windowed and sparse attention), 14.3.

**New:**
- Block selection with compressed and selected branches.
- Learned indexers, including DeepSeek Sparse Attention's lightweight indexer.
- Consistency between training and inference.
- The cost of the selection step.

**Notebook:** Compare window-plus-top-k-block attention with full attention on retrieval tasks, and profile the selection overhead.

**Sources:** [Native Sparse Attention](https://arxiv.org/abs/2502.11089); [MoBA](https://arxiv.org/abs/2502.13189); [Quest](https://arxiv.org/abs/2406.10774); [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556).

### 14.6 Hybrid models at scale · T3 · [Codex 16.4, 3.7]

**Builds on:** `sec_hybrids` (attention fraction, placement, the recipe table), `sec_deltanet`, `sec_mamba`, 1.1.

**New:**
- From the recipe table to released models:
  - Qwen3.5: Gated DeltaNet with one gated-attention layer in four;
  - Kimi Linear;
  - Nemotron-H and the hybrid MoE of Nemotron 3;
  - MiniMax's lightning attention.
- The chunked kernels that make linear layers fast in practice.
- Recurrent state vs KV cache at long context, and the consequences for serving (13.3).

**Notebook:**
1. (T1) Load Qwen3.5-2B into the book's hybrid model from `sec_hybrids` and `sec_deltanet`, and match the Transformers logits of 1.1.
2. (T3) Run a released hybrid model on 8× A100. Measure memory and throughput against context length, compared with an attention-only model of similar active size.

**Sources:**
- Hybrid releases: [Kimi Linear](https://arxiv.org/abs/2510.26692); [Nemotron-H](https://arxiv.org/abs/2504.03624); [Nemotron 3](https://arxiv.org/abs/2512.20856); [MiniMax-01](https://arxiv.org/abs/2501.08313); [Jamba](https://arxiv.org/abs/2403.19887); [Kimi K3](https://arxiv.org/html/2607.24653v1).
- Layers: [Gated DeltaNet](https://arxiv.org/abs/2412.06464).
- Models: [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct); [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B).
- Kernels and teaching: [flash-linear-attention](https://github.com/fla-org/flash-linear-attention); [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/).

### 14.7 The residual stream across depth · T2 · [new]

**Builds on:** `sec_transformer-block` (pre-norm, signal propagation at initialization), `sec_init_param`, 1.6 (the residual stream as a workspace).

**New:**
- Hidden-state growth with depth under pre-norm, and the dilution of each layer's contribution.
- Hyper-connections: several residual streams mixed by learned matrices. Unconstrained mixing is unstable; manifold-constrained hyper-connections fix this with doubly stochastic mixing, obtained by Sinkhorn projection.
- Learned aggregation over depth:
  - depth-weighted averaging (DenseFormer);
  - value residuals;
  - Attention Residuals, where each layer attends over earlier layers' outputs instead of summing them with unit weights, as in Kimi K3.
- What these cost in memory traffic, and the kernels that hide it.

**Derivation:**
- Products of unconstrained mixing matrices across depth can grow or vanish geometrically.
- Doubly stochastic matrices are closed under products, have spectral norm at most one, and preserve the mean across streams. This restores the identity-mapping property.

**Notebook:**
1. At R2 scale, compare pre-norm residuals, hyper-connections with and without the doubly stochastic constraint, and Attention Residuals.
2. Plot hidden-state norm against depth and gradient norm per layer.
3. Compare loss and the stable learning-rate range.

**Sources:**
- Hyper-connections: [Hyper-Connections](https://arxiv.org/abs/2409.19606); [mHC](https://arxiv.org/abs/2512.24880).
- Depth aggregation: [Attention Residuals](https://arxiv.org/abs/2603.15031); [Value Residual Learning](https://arxiv.org/abs/2410.17897); [DenseFormer](https://arxiv.org/abs/2402.02622).
- Release: [Kimi K3](https://arxiv.org/html/2607.24653v1).

### 14.8 ◇ Beyond fixed tokens and left-to-right decoding · T1 · [Codex 2.6, 2.8]

**Builds on:** `sec_text-sequence`, `sec_diffusion-discrete` (masked diffusion, with the text-scale example proposed under "Changes outside this part"), 12.3.

**New:**
- Multi-token prediction as a pretraining objective. 12.3 reuses its heads for drafting.
- Byte-level models with learned segmentation: entropy-based patches (BLT) and dynamic chunking (H-Net).
- Diffusion language models: masked diffusion at scale (LLaDA); parallel decoding and its speed–quality trade-off; commercial diffusion LMs.

**Derivation:** Compute per byte of a patch-based model as a function of mean patch length, against a BPE model of equal FLOPs.

**Notebook:**
1. Train a small byte-level LM in minutes, and segment R3 into entropy-based patches.
2. Compare patch lengths with BPE token lengths.
3. Sample from a masked-diffusion LM, and plot quality against the number of denoising steps.

**Sources:**
- Multi-token prediction: [Multi-token prediction](https://arxiv.org/abs/2404.19737).
- Byte-level models: [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) and its [code](https://github.com/facebookresearch/blt); [H-Net](https://arxiv.org/abs/2507.07955).
- Diffusion LMs: [LLaDA](https://arxiv.org/abs/2502.09992) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct); [Mercury](https://arxiv.org/abs/2506.17298).

### 14.9 ◇ Designing under a budget · T0 · [Codex 16.5, 16.6]

**Builds on:** 14.1 (the design table), 5.2, 5.6, 4.7.

**New:**
- From a task distribution, data, compute, hardware and a serving objective to a design.
- Choosing experiments that resolve the remaining uncertainty.

**Notebook:** The capstone. It requires one measured baseline and one rejected design with evidence.

**Sources:** [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook); [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook); [nanochat](https://github.com/karpathy/nanochat).

---

## Changes outside this part

The labels below are cited from Basics, Advanced or Mathematics but defined in parts that will be retired or that come after this one. They were found by scanning every `:numref:` in the surviving parts.

| Label (retiring part) | Cited from | Action |
|---|---|---|
| `chap_nlp_pretrain` (Language Models) | `sec_transformer`, `sec_scaling-laws` | Re-point to `chap_lm_encoders` and `chap_lm_pretraining` |
| `sec_word2vec_data`, `subsec_negative-sampling` (LM) | `sec_mdl-mutual-information`, `sec_scaling-laws` | Defined anew in 2.1 |
| `sec_word2vec_pretraining` (LM) | `sec_pandas` | Re-point to 2.1 |
| `sec_seq2seq`, `sec_machine_translation` (LM, `seq2seq.md`) | `sec_attention-scoring-functions`, `sec_text-sequence` | Either move `seq2seq.md` into `chap_modern_rnn` as the RNN encoder–decoder, or reword both sentences to cite `sec_transformer`. The first keeps Bahdanau-style attention's motivation intact. |
| `sec_fine_tuning` (Image Models) | `sec_parameters`, `sec_read_write` (4 frameworks each) | Keep if the Image Models part keeps `sec_fine_tuning`. Otherwise re-point to 7.1; the JAX, TensorFlow and MXNet tabs would then point to a PyTorch-only section, which is acceptable for a "when and how far to unfreeze" pointer. |
| `sec_training_systems` (Tools) | `chap_performance`, `sec_multi_gpu`, `sec_multi_gpu_concise`, `sec_adamw`, `sec_practice` | Re-point to 6.1, 6.7 or 6.8 by context |
| `sec_hardware_buyers` (Tools) | `sec_hardware`, `chap_performance` | Re-point to 5.6 for datacenter cost, and the sentence in `sec_hardware` to 5.1–5.2 and 5.6. The desktop-GPU and unified-memory buying advice has no home in this part; keep it as a short appendix or drop it. |
| `sec_utils` (Tools) | `chap_deep_rl`, `sec_oo-design` | Keep the utilities appendix; it is API reference, not content |
| `chap_cv`, `sec_fcn`, `sec_transposed_conv` (Image Models) | GAN, diffusion, ViT, padding sections | Unaffected: the Image Models part stays |
| `chap_recsys`, `chap_gp` (attic) | `sec_multi_gpu`, `sec_mdl-distributions` | Outside this plan |

**Other changes:**

1. **Retire** the NLP Pretraining and NLP Applications chapters. Their teaching role is covered by 2.1–2.2. `subword-embedding.md` is superseded by `sec_text-sequence`.
2. **Diffusion language models.** Add a text-scale masked-diffusion example (MDLM/LLaDA) to `sec_diffusion-discrete`. The optional 14.8 builds on it for diffusion LMs at scale (Codex 2.8).
3. **Multimodal models move to the Image Models part.** v3's multimodal chapter had three sections:
   - contrastive image–text encoders (CLIP, SigLIP);
   - vision–language models (projectors, cross-attention, early fusion);
   - speech and audio language models (neural codecs, Whisper, spoken dialogue).

   They are better kept with the image material and extended there. They build on this part: 2.3 (contrastive embeddings), 7.2 (instruction tuning) and 11.4 (vector quantization). The Image Models part should therefore come after this one. Speech and audio do not fit the title "Image Models": either rename that part (for example, "Multimodal Models") or give audio its own chapter there. v3 has the section plans and verified sources.
4. **Scheduler extension** for T2–T4, plus `d2l.launch`.
5. **Advanced forward references** ("the Language Models part") should cite the new labels in the contract table.

## Codex units not carried over

| Codex units | Reason |
|---|---|
| 2.2, 2.3, 2.5 | Covered by `sec_transformer`, `sec_transformer-block`, `sec_attention-at-scale`, `sec_kv-cache` |
| 3.1, 3.3, 3.6, 3.7 | Covered by `sec_moe`, `sec_deltanet`, `sec_hybrids` |
| 3.9 | Folded into 1.5 and each notebook's comparison rules |
| 4.1–4.6 | Covered by `sec_perf_model`, `sec_hardware`, `sec_memory_precision`, `sec_compilation`, `sec_numerics` |
| 5.2–5.4, 5.7 | Covered by `sec_adamw`, `sec_muon`, `sec_scheduler`, `sec_batch_size`, `sec_scaling`, `sec_scaling-laws`, `sec_memory_precision` |
| 6.1–6.4 | Covered by `sec_multi_gpu`, `sec_multi_gpu_concise` |
| 9.6 (derivations) | Covered by `sec_baselines`, `sec_rl_sequences` |
| 13.2, 13.4 | Covered by `sec_attention-at-scale`, `sec_kv-cache`, `sec_hybrids` |
| 14.8 | Absorbed into 13.1 |
| 15.1, 15.8 | Covered by `sec_language-model`; frontiers folded into each chapter |
| 2.8 | Text-scale example moved to `sec_diffusion-discrete`; diffusion LMs at scale in 14.8 |
