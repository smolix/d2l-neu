# Review of the Language Models part plan

Reviewed: `language_models_part_plan.md` (Codex, 28 Oct 2026), against the current book source, `smolix/d2l-neu` (Basics, Advanced, Mathematics for Deep Learning, Tools appendix).

## Verdict

The technical content is sound and well sourced. As a plan for a D2L part, though, it is three things at once: a course syllabus, a research bibliography and a book outline. The result is too large, it re-teaches much of what the book already has, and it specifies notebooks the book's build cannot run. The fix is mostly subtraction. About 35 sections in 7 chapters covers everything the rest of the book has promised this part, plus the genuinely new material.

## 1. Size

| | Sections |
|---|---:|
| Basics (8 chapters) | 58 |
| Advanced (9 chapters) | 72 |
| Mathematics for DL (6 chapters) | 27 |
| **Proposed LM part (16 chapters)** | **130** |

The proposed part alone equals Basics plus Advanced. Existing Advanced sections run about 7–10k words each. The transformers chapter has 7 sections and 52k words; recurrent-modern has 7 sections and 76k. At that density, 130 units would roughly double the book.

There are two separate causes:

1. **Duplication.** About 35 units re-teach material the book already has, and about 10 more partly do (§2).
2. **Granularity.** The other ~85 units are new, but each is about a third of a D2L section: one concept, three references, one experiment. Merging them gives about 35 sections at D2L density.

## 2. Overlap with the existing book

The plan says it "extends the existing Basics and Advanced parts rather than repeating their introductory treatment". It doesn't: the overlap is not introductory. Advanced already covers a lot of the frontier material:

- Muon, WSD with branching, muP coordinate checks and critical batch size.
- Aux-loss-free MoE balancing.
- GQA and attention sinks.
- Gated DeltaNet with the WY form, and hybrid recall bounds.
- The token-MDP view of RL, GRPO's group-baseline bias and RLOO.
- The KL-regularized optimum and the masked-diffusion ELBO.

| Plan units | Already in the book | Disposition |
|---|---|---|
| 1.1 Text as data, 1.2 Tokenization | `text-sequence.md`: byte-level BPE from scratch, GPT-2 pre-tokenization, verified against production tokenizers, fertility/glitch tokens. `language-model.md`: perplexity *and* bits per byte across tokenizers | Keep only unigram/SentencePiece, vocab-size-vs-compute, multilingual fertility and Unicode normalization (absent) |
| 2.2 Objectives and families | `encoders-decoders.md`: MLM and encoder–decoder implemented | Cut; span corruption becomes an exercise |
| 2.3 Contemporary block | `transformer-block.md`, `gpt.md`: RMSNorm, pre/post-norm experiment, SwiGLU, QK-norm | Cut; Attention Residuals goes to the case study |
| 2.4 RoPE and context extension | `positional-information.md`: RoPE derived, ALiBi/NoPE, PI with experiment; YaRN only named | Fold the YaRN code into continued pretraining |
| 2.5 Local/global attention | `attention-at-scale.md` (sliding window), `kv-cache.md` (window + sinks on GPT-2) | Cut; YOCO and causal enc–dec go to the case study |
| 3.1, 3.3 Sparse capacity, balancing | `moe.md`: active-parameter accounting, aux loss vs bias controller at fixed compute, fine-grained/shared experts | Cut |
| 3.6, 3.7 Delta rule, hybrids | `deltanet.md`, `matrix-state.md`, `hybrids.md`: chunked WY form, Gated DeltaNet, matched hybrid comparison | Cut; pretrained hybrid stacks go to the case study |
| 4.1–4.6, 4.8 Hardware chapter | `performance-model.md`, `hardware.md`, `memory-precision.md`, `compilation.md`; Basics `numerics.md` (bf16/fp8/tf32, loss scaling); Tools `hardware.md` | Cut the chapter; 4.7 (networks) moves to parallelism |
| 5.2–5.4, 5.7 Optimizers, schedules, scaling, recompute | `adamw.md`, `muon.md`, `lr-scheduler.md` (WSD branching), `batch-size.md`, `scaling.md` (muP), `scaling-laws.md` (Chinchilla, 6ND, mini study), `gpus-devices-memory.md` (checkpointing) | Cut |
| 6.1–6.4 Collectives, cost models, DDP, overlap | `multiple-gpus.md` (ring all-reduce, α–β model), `multi-gpu-practice.md` (DDP run; FSDP sketched) | Cut; executed FSDP joins tensor parallelism |
| 9.2, 9.6 KL-regularized objective, group/LOO baselines | `regularized.md` (closed form π_ref·e^{r/β}, BT reward model, overoptimization curves), `baselines.md` (LOO, GRPO normalization), `rl-sequences.md` (token MDP, (K−1)/K bias, verifier hacking) | Cut the derivations; keep KL *estimators* on real rollouts |
| 13.1–13.4 Generator, FlashAttention, compile, KV accounting | `kv-cache.md` (cached forward with identical logits, memory formula, prefill/decode), `attention-at-scale.md` (online softmax), `compilation.md` (CUDA graphs) | Cut |
| 14.8 Operating a service | Tools `model-serving.md` | Cut |
| 15.1 Likelihood metrics | `language-model.md`, `softmax-regression.md` (calibration) | Cut |

Math the part should cross-reference rather than re-derive:

- Forward/reverse KL and f-divergences: `mdl-divergences-distances.md`.
- Cross-entropy, perplexity and arithmetic coding: `mdl-information-theory.md`, which already links tokenizers to compression.
- SVD, Eckart–Young and LoRA: `mdl-svd-low-rank.md`.
- Lagrangian duality: `mdl-constrained-optimization-duality.md`.
- Importance sampling: `mdl-bayesian-computation.md`.
- Bootstrap and CIs: `mdl-statistics.md`.

Genuinely missing from the Math part, so they belong here: MinHash/LSH, Hadamard and random rotations, quantization error, the DPO derivation, and paired item-level bootstrap for benchmarks.

### Duplication inside the plan

The plan's closing ownership map ("Chapter 3 owns… Chapter 13 owns…") is good, but its own structure breaks it:

- **Packing and document masks:** 1.8, 5.1, 8.4.
- **Context extension:** 2.4, 5.8, 15.4, with RULER also in 3.9.
- **Windows and sinks:** 2.5 and 13.6, both citing StreamingLLM.
- **Sparse attention:** 3.5 and 13.6, both citing NSA.
- **MLA and state accounting:** 3.4 and 13.4.
- **Multi-token prediction and drafting:** 2.6 and 13.8.
- **Low precision:** 4.5, 5.6, 8.7, 12.x and 13.5.
- **Bradley–Terry:** 9.1 and 15.5.
- **Teacher-generated data:** 1.7, 8.3 and 10.2.
- **Contamination:** 1.5 and 15.7.
- **Reasoning compression:** 10.6 and 11.7.
- **Asynchronous RL:** 9.8 and 14.7.
- **Experimental methodology:** 3.9, 15.6, 15.8 and 16.5.

## 3. What the book has already promised this part

Advanced chapters forward-reference the Language Models part explicitly. The plan should be organized around fulfilling this contract, and it currently never lists it:

| Promise | Where it is made |
|---|---|
| Multi-node training: tensor, pipeline, expert parallelism; network fabrics | `computational-performance/index.md`, `multiple-gpus.md`, `multi-gpu-practice.md` |
| Serving engines: continuous batching, paged KV caches, speculative decoding | `computational-performance/index.md`, `hardware.md`, `kv-cache.md` |
| Quantization as compression for inference | `computational-performance/index.md` |
| The fp8 *training* recipe | `fast-transformer.md` |
| DPO derivation and LM-scale post-training | `reinforcement-learning/index.md`, `rl-sequences.md` |
| RL notation contract ("inherited verbatim") | `rl-sequences.md`, `tab_rl_notation_contract` |
| Data curation, instruction tuning, learning from feedback | `scaling-laws.md` |
| Pretrained hybrid stacks | `hybrids.md` |

Also, `chap_nlp_pretrain` is referenced from `scaling-laws.md`, `encoders-decoders.md` and the preface. Re-point that label when the old part is replaced.

## 4. Fit with the D2L build and house style

The plan's notebooks are written like course projects. The book's notebooks are re-executed by the build and their outputs committed. Per the repo's `CLAUDE.md`:

- **GPU slots.** The scheduler gives 1 GPU slot per 7.5 GiB of VRAM. A notebook takes 1 slot, or at most 2 on one GPU (≈15 GiB), or 2×1 / 2×2 across GPUs. The plan's default tier, "one GPU around 24GB, capable 1–8B models", does not fit this.
- **Seeds and noise.** There is one seeded run per notebook, no multi-seed sweeps, and no conclusion may rest on a difference smaller than run-to-run noise. That rules out many of the plan's headline comparisons as notebooks:
  - proxy-mixture rankings transferring to a larger model (1.6);
  - LoRA rank/module grids (8.6);
  - SFT vs RL vs SFT→RL (11.2);
  - multi-seed reanalysis (15.6);
  - weaker vs mismatched teacher (10.8).
- **Experiments per notebook.** Most plan notebooks bundle 3–5 experiments; 1.5, for example, has hashing, MinHash/LSH, a labeled-pair evaluation, a quality classifier and a proxy-model loss.

Suggested budget for every section:

- **One headline experiment** of ≤ ~15 minutes in one or two slots, on Qwen3-0.6B/1.7B or SmolLM2-135M/360M, or on the book's own small decoder.
- **Larger models (8B) and sweeps** go in exercises or capstone projects. `rl-sequences.md` already has a "Capstone Projects" section to copy.
- **Parallelism chapter:** use 2×1/2×2 slots for correctness, and analytical models plus published numbers for scale.
- **Low precision:** FP8 runs natively on the build's Ada GPUs. MXFP4/NVFP4 need newer hardware, so simulate them.

House-style points to settle in the plan:

- **Framework.** State "PyTorch only" as a deliberate exception to the Advanced PyTorch+JAX policy. The RL chapter is the precedent.
- **Running example.** The plan lists twelve "recurring artifacts". A book needs one spine. I'd use:
  - (a) `d2l.GPT`, extended with GQA and QK-norm (both already exist as pluggable code in `kv-cache.md` and `transformer-block.md`), loading a real Qwen3 checkpoint and matching Hugging Face logits;
  - (b) a small decoder trained from scratch in the pretraining chapter.

  Every later chapter adapts, quantizes or serves one of these two. `scaling-laws.md` already maps Qwen3's config onto `d2l.GPT` constructor arguments, so this is a short step.
- **Prose register.** The plan has ~80 "rather than / does not guarantee / not universal" constructions, roughly one or two per section. An authoring agent will carry them into the text. Put one "evidence and claims" note in the part introduction, and let section specs say what to teach.
- **Frontier coupling.** Kimi K3 and DeepSeek-V4.1-Flash are cited in chapters 2, 3, 5, 9, 10, 12, 13, 14 and 16. These reports will look dated within a year. Confine them to the case-study chapter so the rest ages well.
- **Private sources.** There are 35 links to the ETH Drive decks, with slide numbers. They are fine as authoring inputs, but should not reach a public bibliography or reuse figures without permission. Keep them in the authoring notes, not the section specs.
- **Evaluation placement.** The plan says evaluation is "introduced immediately", yet the chapter comes 15th. Move the core of it to the first chapter, where every later notebook can use it.
- **One small gap.** Prompting and in-context learning as a way of *using* base models (few-shot vs chat prompting) are never treated, and evaluation depends on them. Advanced's `what-attention-computes.md` (induction heads) gives the mechanism. A subsection in 1.1 or 1.2 is enough.

## 5. Proposed structure: 7 chapters, ~35 sections

Bracketed numbers are the plan units each section absorbs.

**1. Working with Pretrained Language Models**

1. From `d2l.GPT` to a real checkpoint [walkthrough, 2.1]. Extend `d2l.GPT` with GQA and QK-norm, load Qwen3-0.6B, match HF logits, cover chat templates, base vs instruct, revision pinning and few-shot prompting.
2. Evaluating language models [15.2, 15.3, 15.5, 15.6]. Scoring conventions, pass@k, a small suite, LLM judges, paired item-level bootstrap.
3. Text embeddings and retrieval [2.7]. Replaces the word2vec/GloVe/BERT chapters.
4. *(Optional)* Diffusion language models [2.8]. Alternatively, add a text example to `discrete-diffusion.md`.

**2. Data**

1. From crawl to corpus [1.1, 1.4]. Extraction, language ID, Unicode normalization, provenance, dataset cards.
2. Filtering, deduplication and decontamination [1.5, 15.7]. Derives MinHash/LSH.
3. Mixtures and synthetic data [1.6, 1.7].
4. Tokenizers at scale [1.2, 1.3]. Unigram, vocabulary size vs compute, multilingual fertility, vocabulary extension. Builds on Basics' BPE.

**3. Pretraining**

1. A real pretraining run [1.8, 5.1, 5.2, 5.9, packing from 8.4]. Streaming shards, document masks, resumable data state, diagnosing spikes.
2. FP8 and microscaled training [4.5, 5.6, 5.7]. Fulfills `fast-transformer.md`'s promise.
3. Continued pretraining and long context [5.8, 2.4, 15.4]. Replay, YaRN in code, RULER-style evaluation.

**4. Parallelism Beyond One Node**

1. Tensor parallelism and sharded state [6.5–6.8]. Includes actually running FSDP, which is only sketched in Advanced.
2. Pipeline parallelism and schedules [7.1–7.3].
3. Sequence and context parallelism [7.4]. Ring attention reuses Advanced's online softmax.
4. Expert parallelism [dispatch from 3.2, 7.5, upcycling from 3.8].
5. Meshes, topology and failures [4.7, 7.6–7.8].

**5. Post-Training**

1. Supervised fine-tuning [8.1–8.4]. Templates, loss masks, instruction data.
2. LoRA, QLoRA and merging [8.5–8.8].
3. Reward models and PPO-RLHF at LM scale [9.1, 9.4, KL estimators from 9.2].
4. Direct preference optimization [9.3].
5. RL with verifiable rewards [9.5, 9.7, 9.9]. GRPO at LM scale, Dr. GRPO/DAPO fixes, reward hacking.
6. Distillation, on- and off-policy [10.1–10.8]. Sequence KD, logit KD, OPD; OPSD as a variant, not its own section.
7. Training reasoning models [11.2, 11.3].
8. Test-time compute [11.1, 11.4–11.7].

**6. Inference and Serving**

1. Quantizing weights [12.1–12.4]. Grids, outliers, GPTQ/AWQ.
2. Below four bits [12.5–12.8]. Rotations, QAT, lossless coding, a brief note on pruning.
3. Shrinking and skipping the KV cache [3.4, 3.5, 13.5, 13.6]. Trainable MLA, sparse attention, KV quantization, eviction.
4. Speculative decoding [2.6, 13.7, 13.8]. Includes MTP heads as drafters.
5. Structured generation [13.9].
6. Continuous batching and scheduling [14.1, 14.2].
7. Paged and prefix-shared KV [14.3, 14.4].
8. Serving at scale [9.8, 14.5–14.7]. Disaggregation, multi-LoRA, rollout systems.

**7. Frontier Models**

1. Reading a technical report: an open dense model end to end [16.1, 16.2]. Uses OLMo or Qwen3.
2. Two frontier designs [16.3–16.5, Attention Residuals from 2.3, Engram from 2.6, pretrained hybrids from 3.7]. Dated, refreshed yearly.
3. Designing under a budget [16.6, 5.5]. A capstone project rather than a notebook.

Count: 3 (+1 optional) + 4 + 3 + 5 + 8 + 8 + 3 = **34–35 sections**.

Chapter 4 could reasonably go to Advanced's Computational Performance chapter instead. But that chapter explicitly hands multi-node work to this part, and the language-model setting is what motivates it, so I'd keep it here.

## 6. Keep from the plan

- The notebook authoring contract (§"Notebook authoring contract"), especially correctness before speed, measured vs projected, and pinned revisions.
- The ownership map. With the structure above it becomes enforceable.
- Model choices: Qwen3 small checkpoints for capability, OLMo/SmolLM for training transparency, and explicit base/instruct/reasoning distinctions.
- One embeddings notebook replacing the word2vec/GloVe sequence.
- The per-section reference lists. Move them to a separate authoring bibliography (e.g., `planning/source-guide.md`) so the outline stays readable.

## 7. The current Language Models part

| Current sections | Disposition |
|---|---|
| word2vec, approx-training, word-embedding-dataset, word2vec-pretraining, glove, similarity-analogy | Attic |
| subword-embedding (BPE) | Drop; superseded by `text-sequence.md` |
| bert, bert-dataset, bert-pretraining | Attic; `encoders-decoders.md` already implements MLM. The fine-tuning-heads idea from `finetuning-bert.md` feeds 1.3 and 5.1 |
| seq2seq | Keep, but move next to attention/encoder–decoders (it is a modern rewrite) |
| sentiment and NLI applications, legacy-attention-lib | Attic or drop |
