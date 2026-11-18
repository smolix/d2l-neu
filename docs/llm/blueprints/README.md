# Language Models part — chapter blueprints

These files expand the chapters of `../language_models_part_plan.md` into per-chapter blueprints detailed enough that an author (human or agent) can write each notebook without making content decisions. A blueprint fixes, for every section:

> **Relation to the plan.** The blueprints were written against an earlier outline of the part and keep its section numbering and some of its wording. `../language_models_part_plan.md` is authoritative: where a blueprint and the plan disagree (structure, models, tiers, labels), the plan wins. The map at the end of this file gives, for every blueprint section, the plan section it feeds. Mentions of "v5", "the Codex plan" or "Codex N.M" refer to earlier drafts that are no longer in the tree (git history up to commit `42ddd3bb`) and are provenance only.

- the teaching arc and derivations;
- the from-scratch code, with line budgets;
- the experiments, with tiers, budgets and expected outcomes;
- the numerical tests, figures and exercises;
- the sources, already checked.

It also fixes the chapter's assignment.

## How to use

1. Read `00_conventions.md`: the ground rules, the chapter and section templates, compute tiers T0–T4, the running examples R1–R3, the section label registry and the reserved `d2l` names.
2. Read `01_interfaces.md`: the canonical cross-chapter contract (shared functions, artifacts and labels). **It overrides any chapter file that disagrees.**
3. Read the chapter blueprint, then write its sections in order. Each section lists what it builds on, what it exports (`#@save`) and which artifacts it commits. A later chapter imports those exports and never re-defines them.

## Index

| Ch | Blueprint | Lines | Sections | Tiers | Assignment |
|---|---|---|---|---|---|
| 1 | `ch01_pretrained.md` — Working with Pretrained LMs | 1330 | 7 | T1 (T0 fallbacks); 1.7 replays cached API responses at T0 | A1 Run, port, and measure |
| 2 | `ch02_encoders.md` — Encoders, Embeddings and Retrieval | 873 | 5 | T1 (2.1 on CPU) | A2 RAG over the book |
| 3 | `ch03_data.md` — Data | 1222 | 6 | T0, T1; T2 (3.4) | A3 Data pipeline with a checkable manifest |
| 4 | `ch04_pretraining.md` — Pretraining | 1207 | 7 (4.7 ◇) | T0–T2; R2 run is a one-off T3 job | A4 Pretrain with packing, document masks, WSD |
| 5 | `ch05_hardware.md` — Hardware for Training and Inference | 1193 | 6 | T0, T2, T3 | A5 Performance model of a chip, a node and a cluster |
| 6 | `ch06_parallelism.md` — Parallelism Beyond One Node | 1272 | 8 | T3; T4 (6.1, 6.7) | A6 TP and PP blocks verified against one GPU |
| 7 | `ch07_finetuning.md` — Fine-Tuning | 1511 | 9 | T1; T2 (7.1); T3 (7.8) | A7 part 1 (SFT) |
| 8 | `ch08_rl.md` — Preference Optimization and RL | 1115 | 7 (8.7 ◇) | T1–T3 | A7 (SFT → DPO → GRPO) |
| 9 | `ch09_reasoning.md` — Distillation and Reasoning | 1284 | 7 | T1, T2; T3 with T2 reduction (9.3) | Mini-assignment: on-policy distillation |
| 10 | `ch10_agents.md` — Agents and Tool Use | 1121 | 5 | T0, T1; T3 with T1 reduction (10.4) | none (revisit once the 10.3 sandbox is confirmed) |
| 11 | `ch11_compression.md` — Model Compression | 1214 | 6 | T1 only | Mini-assignment: compress Qwen3-1.7B |
| 12 | `ch12_inference.md` — Efficient Inference | 811 | 4 | T1; T2 with T1 reduction (12.3) | supplies A8 hooks |
| 13 | `ch13_serving.md` — Serving Systems | 976 | 5 (13.5 ◇) | T1; T2 pair and T3, with a simulator path (13.4) | A8 Inference engine (paged KV, continuous batching) |
| 14 | `ch14_architectures.md` — Advanced Architectures | 1718 | 9 (14.8–14.9 ◇) | T0–T3 | Capstone: design under a budget |
| | **Total** | **≈ 16,850** | **91** | | 8 assignments, 2 mini-assignments, capstone |

◇ marks an optional section. Every T2, T3 or T4 notebook has a reduced configuration flag that runs on the T1 build box. All assignments are optional for readers; "mini" marks the smaller scope only.

Assignment labels are listed in `01_interfaces.md` §4: `sec_lm_a1`–`sec_lm_a8`, `sec_lm_a7_sft`, `sec_lm_a9_mini`, `sec_lm_a11_mini` and `sec_lm_capstone` (listed in the table, unnumbered). Cross-chapter artifacts (`lm-*` DATA_HUB keys, their producers and consumers) are in `01_interfaces.md` §3.

## Verification status (2026-10-29)

- **Labels and names.** A consistency checker (headings against the label registry, label owners, `d2l` names, `assumed:` lines, DATA_HUB keys) passes. The only labels cited across files are deliberate citations (`eq_lm_fsdp_*` from 7.8, `eq_lm_hw_decode_step` from 12 and 13).
- **Leftover assumptions.** Two `assumed:` notes remain. Each marks a real residual risk and has a stated fallback:
  - Chapter 4: whether `generate` and `score_choices` accept a callable model;
  - 11.1: no published Qwen3 4-bit GPTQ checkpoint.

  Decision 4 resolved the other two: Chapter 4's R1 attribute check and 12.1's RoPE-attribute risk.
- **arXiv.** All 190 arXiv IDs cited beyond the earlier plan draft were fetched, and each abstract-page title matches the paper the blueprint names.
- **URLs.** 252 of 255 new URLs return HTTP 200. The GloVe site refuses connections and is marked "not re-verified"; the other two are API base URLs in 1.7 (`http://localhost:8000/v1` and `https://router.huggingface.co/v1`), not pages.
- **Hugging Face IDs.** All 107 model and dataset IDs resolve (the Qwen3.5 helpers, `Qwen/Qwen3-14B-Base`, `Qwen/Qwen3.8-27B` and 1.7's API models were checked on 2026-10-29), except:
  - three that are GitHub repositories;
  - `Qwen/Qwen3-4B-GPTQ-Int4`, which the blueprint already notes does not exist;
  - `MiniMaxAI/MiniMax-01`, which 14.1 cites on purpose as the wrong ID, next to the correct `MiniMaxAI/MiniMax-Text-01`.
- **Other unverified items.** 12 claims stay marked "(unverified)" in the text (8 of them in Ch14: V4 hash-layer semantics, engine hybrid-cache support, and similar; 1.7 adds OpenAI's terms page, which refuses automated fetches). Several numbers are marked "re-read before quoting". Authors must check both kinds against the source while writing.
- **Not done.** No notebook code has been run. Line budgets, timings and tolerances are the blueprint authors' estimates, and some are flagged as provisional.

## Decisions

All 21 decisions are settled; the facts still missing are listed at the end of this section.

**One-way doors**

Decided 2026-10-28 and applied to the blueprints:

1. **R2 tokenizer (3.6).** Byte-level BPE, vocabulary 32,768, Qwen3-style single-digit pre-tokenization, EOS id 0, no BOS. Ids 0–31 are reserved for 32 special tokens: `<|endoftext|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`, then `<|reserved_4|>` … `<|reserved_31|>`, so later chat or tool tokens need no retraining. The merges start at id 32.
2. **R2 production run (4.2).** The model stays at 138M parameters on 3.15B tokens, trained once as a T3 job (1.3–1.7 h on 8×A100). The T2 notebook runs a 1.05B-token short schedule and analyzes the committed checkpoints. The tokenizer, the checkpoints (about 4.8 GB) and the FineWeb-Edu shards (about 8 GB) live in d2l-owned Hugging Face repositories. Their `DATA_HUB` entries pin a commit and a sha1 (`01_interfaces.md` §3).
3. **`d2l.generate` name collision.** (Revised 2026-11-12: no rename. The book's old `generate(step_fn, prefix, num_tokens)` of `sec_decoding` carries no `#@save`, so it is not in the library and nothing collides.) The name `d2l.generate` belongs to 1.2's text-in, text-out sampler, with no dispatch.
4. **Qwen3 support in the book's classes.** Chapter 1 subclasses them instead of editing them: `d2l.Qwen3Attention(CausalAttention)`, `d2l.SwiGLU(FeedForward)`, `d2l.Qwen3GPT(GPT)`. `load_qwen3` returns a `Qwen3GPT`. `Qwen3GPT` defines its own `generate_cached`, because the book's cell has no `#@save`. Chapter 4's `PackedAttention` and `PackedGPT` subclass these classes, and 14.6 uses `d2l.SwiGLU`.

Decided 2026-10-29:

5. **Judge model.** The part's one judge is `Qwen/Qwen3.5-9B` in bf16, text-only and non-thinking (17.9 GB of weights). It serves 1.5, 2.5 and A2, 3.5, 7.3, 7.9 and 8.x. The readers' floor is one 24 GB RTX 4090, so T1 notebooks that load the judge claim a whole GPU on the build box. Where a model does not fit, a smaller Qwen3.5 in bf16 replaces it, never a 4-bit load; 4-bit appears only where quantization is the subject (7.6, Chapter 11, 12.1). In 1.5, Qwen3-4B stays on as the second, older-family judge for the self-preference exercise. `00_conventions.md` records the judge, the T1 change and the no-4-bit rule.

   **Helper models and independence.** Helper roles moved to Qwen3.5 in bf16:
   - 2.5's generator, closed-book probe, query rewriting, HyDE and contextual headers now use Qwen3.5-4B. Part 6 contrasts its 32 KB-per-token KV cache with Qwen3-4B's 147 KB.
   - 2.5's answer check uses Qwen3.8-27B at authoring time, so neither the drafter nor the judge filters the test set.
   - 3.5's rephraser is Qwen3.5-2B, with Qwen3.5-4B as the optional stronger rephraser. 3.3's paraphrases use Qwen3.5-4B, and 3.2's second annotator is the judge.
   - 7.3's Magpie generator stays Qwen3-1.7B on purpose: it must differ from the scorer and from the judge's family. 7.3's selection conclusions rest on IFEval and GSM8K.

   `00_conventions.md` states the rules: when two roles need different models, the same model, or the same tokenizer.

   **Memory budgets.** "≤ 15 GiB" in a budget line is the build-box slot (two notebooks per 4090), not the readers' card. Every claim that a model does not fit "T1" or "a 16 GB GPU" now says what fits a slot and what fits a whole 24 GB GPU (Chapter 1's compute plan, 1.1, 1.2, A1; 4.2's OLMo-2-7B extension; Chapter 8's budget summary and A7; 10.4; 11.1; 11.5; 14.2; the capstone's hardware column). The largest change is 7.6: QLoRA's headline run moves from Qwen3-8B in 15 GiB to `Qwen/Qwen3-14B-Base` on a whole 24 GB GPU, because 8B in bf16 (16.4 GB) leaves room for plain LoRA on 24 GB and no longer makes the case for 4-bit. 14B in bf16 is 29.5 GB; after NF4 with double quantization its static weights are 9.93 GB (≈ 9.25 GiB), and the expected training peak of 13–15 GB is to be measured. Qwen3-8B stays as the fallback that fits a slot.

Decided 2026-10-29 and applied to the blueprints (a sweep by chapter, then a consistency check):

**Scope**

6. **Assignments for Chapters 9 and 10.** Chapter 9 gets a mini-assignment, `sec_lm_a9_mini`: on-policy distillation at 9.2's T1 scale (a 0.6B LoRA student and a 1.7B teacher, starting from A7's SFT checkpoint), graded on 9.2's numerical checks and a written report, not on a gain threshold. Chapter 10 has none until the 10.3 sandbox is confirmed on the build host.
7. **All assignments are optional for readers.** "Mini" marks the smaller scope of Chapters 9 and 11.
8. **Capstone and Engram.** The capstone is listed in the assignment table, unnumbered, and stays hosted by 14.9. Engram stays a sub-block of 14.2.
9. **R3** keeps Mathematics for Deep Learning and Computer Vision.
10. **Chapter 6's multi-node runs.** 6.7's core is two nodes (6 two-node and 2 one-node runs, about 15 GPU-hours instead of about 64); the 4-node pass and one EP=16 MoE run are optional. The 512×4096 global batch (2,097,152 tokens, 4× R2's) belongs to 6.7 only; the other 6.x batches are per-node settings.

   **Models behind an API (new section 1.7, `sec_lm_api`).** The part had no coverage of hosted models. 1.7 teaches the OpenAI-compatible protocol with one client (`d2l.ChatAPI`) against the Hugging Face router (`Qwen/Qwen3.5-397B-A17B`, `openai/gpt-oss-120b`) and a local `transformers serve`: latency and throughput from the client, rate limits and backoff, cost from the `usage` block, reproducibility, what an API cannot give (prompt log-probabilities), and terms of use. Every response is cached, so the build replays them at T0; live calls are optional and capped.

**Licences and review**

11. **Models.** 7.8's headline is `Qwen/Qwen3.8-27B` (2026-08-05, Apache-2.0, dense Gated DeltaNet hybrid), replacing Qwen2.5-72B, which was neither current nor compact; `Qwen/Qwen3-32B` is the fallback. Full fine-tuning of 27B does not fit 8× 40 GB (60.7 GB per GPU) and LoRA fits easily, so the section's point survives. Models larger than a node are fine-tuned through a hosted API (Tinker lists the same model) and sampled through 1.7's client. The ZeRO-3 offload arm, which needed 400 GB of host RAM, is gone. 2.5's authoring-time answer check also uses Qwen3.8-27B (tensor-parallel on a T2 pair, or through an API). Qwen-Scope SAEs are a run-time download with the model card's caution quoted; Qwen3-4B-Base (8.6) is Apache-2.0 on its own entry; the AngelSlim drafter is Apache 2.0 per its licence file.
12. **Licences.** The book's text is CC BY-SA 4.0 and its code MIT-0; book-written data follows suit (`lm-repair30` MIT-0, `lm-injection-suite` CC BY-SA 4.0). Permissive third-party data ships with attribution: FineWeb-Edu derivatives and peS2o (ODC-By), PG-19 and InferenceX (Apache-2.0), Winogrande (CC-BY), PIQA (AFL-3.0), MATH-500 (MIT upstream). Non-commercial or unlicensed data is downloaded at run time and never committed: SciQ (CC-BY-NC-3.0), Math-Shepherd, LiveCodeBench.
13. **Content and legal review.** Educational use of permissively licensed public sources needs no further review. Common Crawl text is never hosted: 3.1 ships a pinned WARC byte range and a manifest that the notebook rebuilds. The rephrased FineWeb-Edu corpus ships as text. 8.7's should-refuse set publishes categories, rubric and aggregate rates only; you review its prompts before freeze. 2.5 keeps 240 human-verified `dev` and `test` items each.

**Infrastructure**

14. **PyTorch 2.11.0** for every notebook, with Python 3.12, the book's pin (superseded 2026-11-12: the part does not move the pin to 2.15.0; its notebooks run in `.venv-lm`, the book's torch 2.11.0/torchvision 0.26.0 plus Transformers and the OpenAI client, `docs/build-system.md` §6.4.2). Engines and toolchains that pin an older `torch` run in their own environments: vLLM 0.32.0 (2.13.0), SGLang 0.5.22 (2.14.1), Unsloth and Axolotl (`<2.15.0`).
15. **Hardware.** `00_conventions.md` has the new tiers:
    - T1: one RTX 4090. The build box (4× RTX 4090, no NVLink, unreliable peer-to-peer on its AMD host) runs no multi-GPU experiment.
    - T2: one A100 40 GB PCIe (1,555 GB/s; 5.1 adds its `chip_specs` row). Two-GPU experiments (TP2, prefill/decode split, NCCL with and without the bridge) run on a **T2 pair** joined by an NVLink bridge.
    - T3: one 8× A100 40 GB node with limited access, and each run happens once. Nothing needs 80 GB GPUs: the 80 GB numbers left in the part are arithmetic or datasheet comparisons. T4: a handful of multi-node runs.
    - Every chapter was swept: T2 plans refit to 40 GB (smaller micro-batches, LoRA where full fine-tuning no longer fits, the 30B MoE runs on a T2 pair), build-box multi-GPU runs moved to the T2 pair or a one-rank path, and interconnect-dependent predictions carry both rows with `topo.txt` deciding.
16. **Units and experiment size.** No GB/GiB convention. Experiments are sized to their claims (`00_conventions.md`). Cuts: 10.3's grid from 540 to 240 episodes; 8.6's T3 runs from 8 to 5; 9.3's RFT arm; 6.7 from about 64 to about 15 GPU-hours; 5.3–5.5's secondary sweeps and 3-node runs; 13.4's DP8·TP1 and DP2·TP4 load sweep; 7.8's offload arm and its 27B full-fine-tuning arm (80 GB only); 6.5's 128k run (80 GB only, now a prediction).

**Smaller technical calls**

17. **LoRA scaling and chat template.** `rslora=False` stays the default (PEFT's); 7.4 reports both scalings and adds a fixed-learning-rate slice (ranks {1, 8, 64}) that tests rsLoRA's actual claim. The part uses `d2l_chatml_v1` (7.2); Chapter 8's fallback model keeps Qwen's own template, and Chapter 9's `<think>` blocks live inside assistant content.
18. **`failure_rate`** defaults to Llama 3's rate: 419 interruptions in 54 days on 2,048 nodes, 3.79 per 1,000 node-days. RSC-1 (6.50) and RSC-2 (2.34) remain options. Llama 3's Table 5 is internally inconsistent (its counts sum to 419 only with 148 faulty-GPU events, 35.3%, but it prints 30.1%); 6.8 notes this, and the rate does not depend on it.
19. **`qk_clip_`** moves to the book's `sec_muon`, which gains a MuonClip subsection (book-side edit below); 4.3 imports it.
20. **Drafters and long context.** 12.2 pairs Qwen3-0.6B with Qwen3-4B (identical tokenizers); the 103M distilled draft belongs to 12.3, because R1 (Qwen3-0.6B) has no smaller Qwen3 sibling. 12.4 falls back to Qwen3-4B, which has published RULER scores, instead of Qwen3-1.7B.
21. **A8's criterion is "batch-invariant".** Against the engine's own batch-1 run in deterministic mode, logits must be bitwise equal; against the fp32 reference generator, token ids must match, with near-ties counted. vLLM's batch-invariant mode covers Ada and A100.

**Still open (facts only you have)**

- The 8× A100 40 GB servers: NVSwitch (SXM) or PCIe pairs? Until the first T3 run commits `topo.txt`, the interconnect-dependent predictions carry both rows.
- Whether the build may spend money on live API calls (1.7, 7.8's hosted fine-tuning), and where the token lives. Until then those cells replay committed transcripts.
- The 10.3 sandbox: run `bwrap --ro-bind / / true` and `unshare -rn true` once on the build host.
- The SHA-256 of 3.1's pinned Common Crawl byte range (one 256 MiB download at authoring time).

## Book-side edits this part requires

These edits fall outside the new chapters:

- **Decision 3 above, the `generate_ids` rename** (withdrawn 2026-11-12: not needed, see decision 3; kept for the record; line numbers as of `main` on 29 Oct 2026):
  - `chapter_recurrent-neural-networks/decoding.md`: the `#@save` definition (line 482) and the two calls (lines 577 and 598), the only calls in the book (`gpt.md`'s `generate` is a method and keeps its name);
  - `docs/rnn-ssm-modernization-apis.md`: the API list (line 7) and the signature (line 62).
- **Decision 4 needs no book edit.** The book's `CausalAttention`, `FeedForward`, `TransformerBlock` and `GPT` stay as they are. `sec_kv-cache`'s `generate_cached` stays unsaved (2026-11-12): `Qwen3GPT` owns its cached loop.
- **Legacy labels to re-point:**
  - `sec_word2vec_pretraining` (cited in `pandas.md`) → `subsec_lm_categorical_embeddings`;
  - `chap_nlp_pretrain` and `chap_nlp_app` (cited in the preface, `scaling-laws.md` and `encoders-decoders.md`) → `chap_lm_encoders`.
- **Labels with no target or owner after retirement:**
  - `sec_bert-pretraining`, cited from appendix exercises in `cloud-instances.md` and `training-systems.md`;
  - `sec_seq2seq` and `sec_machine_translation`, cited from `attention-scoring.md` and `text-sequence.md`.
- **Legacy helpers to keep:** `DATA_HUB['ptb']` and the word2vec data helpers, which `sec_scaling-laws` still uses.
- **`d2l.normalize` inconsistency:** it uses unbiased `torch.std` in one framework tab and population std in another. Chapter 8 uses population std, so the group-advantage fixtures depend on this.
- **α–β communication model:** it appears only in the exercises of `sec_multi_gpu`. 5.4 now states it; consider moving it into that section's text.
- **`sec_hardware_buyers`:** it has no datacenter content. 5.6 owns cost per token and owned versus rented; the buyer's guide could point there.
- **Decision 14 (withdrawn 2026-11-12):** the pin stays `torch==2.11.0`/`torchvision==0.26.0`; no bump and no re-capture. The part's packages live in the `lm` extra and `.venv-lm` (`docs/build-system.md` §6.4.2).
- **Decision 19, MuonClip in `sec_muon`** (`chapter_optimization/muon.md`; lines 809–813 mention MuonClip in one sentence): add a subsection that defines `d2l.qk_clip_(model, stats, tau, alpha=0.5) -> dict` with `#@save`. Contract, as 4.3 relies on it: called in place after an optimizer step; `stats[layer]['max_logit'][head]` is the per-head maximum of |q·k|/√d_h; each head with S > tau has its `W_q` rows scaled by (tau/S)^alpha and its `W_k` rows by (tau/S)^(1−alpha); returns the per-layer scale factors; a no-op that warns and returns `{}` when the model uses QK-norm. Tests: after a clip the batch maximum equals tau (relative 1e-5), a second clip changes nothing, heads at or below tau are untouched bit for bit. The book cannot import 4.3's `attn_logit_stats`, so the subsection needs its own small statistics hook, and it must decide the behaviour under grouped-query attention (4.3's testbed has as many KV heads as query heads).
- **Decision 12, licence file:** the repository has none; add one stating that the text is CC BY-SA 4.0 and the code MIT-0 (`pyproject.toml` already declares MIT-0).

## Files

- `00_conventions.md`: ground rules, templates, tiers, label registry, reserved names, default models.
- `01_interfaces.md`: canonical shared API, resolved duplicates, artifacts, labels, patch list.
- `ch01_…` to `ch14_…`: the chapter blueprints.

## Blueprint section to plan section map

The blueprints in `docs/llm/blueprints/` use their own section numbering. Each row gives one blueprint section, the plan section or sections it feeds, and the plan label; "not in the plan as a section" marks a blueprint section whose content is now a subsection, table, exercise or Further Reading (FR) notebook item of the plan section named. Plan sections and FR notebooks with no blueprint section are listed at the end. Where a blueprint and the plan disagree, the plan wins.

| Blueprint section | Blueprint title | Plan section(s) | Plan title(s) | Plan label(s) | Note |
|---|---|---|---|---|---|
| 1.1 | Pretrained models with Hugging Face | 1.1 | Pretrained Models with Hugging Face | `sec_lm_hf_models` |  |
| 1.2 | From `d2l.GPT` to a released checkpoint | 1.2 | A Released Checkpoint in the Book's Own Code | `sec_lm_d2l_checkpoint` |  |
| 1.3 | Prompting and in-context learning | 1.3 | Prompting and In-Context Learning | `sec_lm_prompting` |  |
| 1.4 | Scoring language models | 1.4 | Scoring Language Models | `sec_lm_scoring` |  |
| 1.5 | Evaluation suites, judges and error bars | 1.5 | Evaluation Suites, Judges and Error Bars | `sec_lm_eval_suites` |  |
| 1.6 | Interpretability | 1.6 | Interpretability | `sec_lm_interpretability` |  |
| 2.1 | Word embeddings in brief | 2.1 | Word Embeddings | `sec_lm_word_embeddings` |  |
| 2.2 | Pretrained encoders: BERT and its descendants | 2.2 | Pretrained Encoders | `sec_lm_bert` |  |
| 2.3 | Text embeddings from contrastive training | 2.3 | Text Embeddings from Contrastive Training | `sec_lm_text_embeddings` |  |
| 2.4 | Retrieval: sparse, dense and approximate search | 2.4 | Sparse, Dense and Approximate Retrieval | `sec_lm_retrieval` |  |
| 2.5 | Retrieval-augmented generation | 2.5 | Retrieval-Augmented Generation | `sec_lm_rag` |  |
| 3.1 | From crawl to documents | 3.1 | Document Extraction from Web Crawls | `sec_lm_crawl` |  |
| 3.2 | Quality filtering | 3.2 | Quality Filtering | `sec_lm_quality_filtering` |  |
| 3.3 | Deduplication and decontamination | 3.3 | Deduplication and Decontamination | `sec_lm_dedup` |  |
| 3.4 | Data mixtures and curricula | 3.4 | Data Mixtures | `sec_lm_data_mixtures` |  |
| 3.5 | Synthetic and rephrased data | 3.5 | Synthetic and Rephrased Data | `sec_lm_synthetic_data` |  |
| 3.6 | Tokenizers at scale | 3.6 | Tokenizers at Scale | `sec_lm_tokenizers` |  |
| 4.1 | The pretraining data path | 4.1 | The Pretraining Data Path | `sec_lm_data_path` |  |
| 4.2 | Training the reference model | 4.2 | The Reference Pretraining Run | `sec_lm_reference_model` | carries blueprint section 4.3 as a subsection |
| 4.3 | Training instabilities | not in the plan as a section (now a subsection of 4.2 and its high-learning-rate arm) | The Reference Pretraining Run | `sec_lm_instabilities` is not a plan label |  |
| 4.4 | Low-precision training | 4.3 | Low-Precision Training | `sec_lm_low_precision` |  |
| 4.5 | Continued pretraining and mid-training | 4.4 | Continued Pretraining and Mid-Training | `sec_lm_continued_pretraining` |  |
| 4.6 | Long-context training | 4.5 | Long-Context Training | `sec_lm_long_context` |  |
| 4.7 | Budgets beyond Chinchilla (optional) | 4.6 | Training Budgets beyond the Compute Optimum | `sec_lm_budgets` | hosts A5 |
| 5.1 | Accelerator design: compute, memory and I/O | 5.1 | Accelerator Design | `sec_lm_accelerator_design` | (tile derivation and GEMM microbenchmark to 5.2) |
| 5.2 | Training, prefill and decode: why their hardware needs differ | 5.3 | Training, Prefill and Decode Workloads | `sec_lm_workload_hardware` | hosts A6a |
| 5.3 | Memory and storage tiers, from SRAM to disk | 5.4 | Memory and Storage Tiers | `sec_lm_memory_tiers` | (checkpoint measurement to 6.9; data loading at scale added) |
| 5.4 | Data paths inside a server | 5.5 | Data Paths inside a Server | `sec_lm_server_paths` |  |
| 5.5 | Networks between servers | 5.6 and 5.7 | Scale-Up Domains and Scale-Out Networks; Collective Communication at Scale | `sec_lm_networks`; `sec_lm_collectives` | fabric half MERGE into 5.6; collective half expanded as 5.7 |
| 5.6 | Scale-up, scale-out and what it costs | 5.6 and 5.8 | Scale-Up Domains and Scale-Out Networks; Energy, Power and Cost | `sec_lm_networks`; `sec_lm_energy_cost` | domains and layout calculator in 5.6; cost in 5.8 (power and energy added); goodput in 6.10 |
| 6.1 | Running jobs across machines | 6.1 | Multi-Node Jobs and Hang Diagnosis | `sec_lm_multinode_jobs` |  |
| 6.2 | Sharded data parallelism at scale | 6.2 | Sharded Data Parallelism at Scale | `sec_lm_fsdp` |  |
| 6.3 | Tensor and sequence parallelism | 6.3 | Tensor and Sequence Parallelism | `sec_lm_tensor_parallel` | hosts A6b with 6.4 |
| 6.4 | Pipeline parallelism | 6.4 | Pipeline Parallelism | `sec_lm_pipeline_parallel` |  |
| 6.5 | Context parallelism | 6.5 | Context Parallelism | `sec_lm_context_parallel` |  |
| 6.6 | Expert parallelism | 6.6 | Expert Parallelism | `sec_lm_expert_parallel` |  |
| 6.7 | Composing parallelism across nodes | 6.7 | Composed Parallelism and Topology | `sec_lm_composing_parallelism` |  |
| 6.8 | Checkpoints, failures and long runs | 6.9 and 6.10 | Checkpoints at Scale; Failures, Elasticity and Goodput | `sec_lm_checkpoints`; `sec_lm_long_runs` | (6.9 also takes blueprint 5.3's checkpoint measurement; 6.10 takes blueprint 5.6's goodput) |
| 7.1 | Full fine-tuning | 7.1 | Full Fine-Tuning | `sec_lm_full_finetuning` |  |
| 7.2 | Supervised instruction tuning | 7.2 | Supervised Instruction Tuning | `sec_lm_sft` | hosts A7 part 1 |
| 7.3 | Instruction data | 7.3 | Instruction Data | `sec_lm_instruction_data` |  |
| 7.4 | LoRA from scratch | 7.4 | Low-Rank Adaptation | `sec_lm_lora` | merged with blueprint section 7.5 or 7.4 |
| 7.5 | Beyond LoRA | not in the plan as a section (now 7.4's taxonomy table and DoRA exercise) | Low-Rank Adaptation | `sec_lm_peft` is not a plan label |  |
| 7.6 | Quantization for fine-tuning: QLoRA | 7.5 | Quantized Fine-Tuning | `sec_lm_qlora` |  |
| 7.7 | Fine-tuning toolchains | not in the plan as a separate section (merged into 7.6) | Fine-Tuning at Scale and Toolchains | `sec_lm_ft_toolchains` is not a plan label |  |
| 7.8 | Fine-tuning large models across GPUs | 7.6 | Fine-Tuning at Scale and Toolchains | `sec_lm_ft_multi_gpu` | merged with blueprint section 7.8 or 7.7 |
| 7.9 | Merging and retaining capabilities | 7.7 | Merging and Retaining Capabilities | `sec_lm_merging` |  |
| 8.1 | Reward models for language | 8.1 | Reward Models | `sec_lm_reward_models` |  |
| 8.2 | RLHF with PPO for language models | 8.2 | RLHF with PPO at Language-Model Scale | `sec_lm_rlhf_ppo` |  |
| 8.3 | Direct preference optimization | 8.3 | Direct Preference Optimization and Its Variants | `sec_lm_dpo` | merged with blueprint section 8.4 or 8.3 |
| 8.4 | The preference-optimization family | not in the plan as a section (now 8.3's variants table and exercise) | Direct Preference Optimization and Its Variants | `sec_lm_po_family` is not a plan label |  |
| 8.5 | RL with verifiable rewards | 8.4 | RL with Verifiable Rewards | `sec_lm_rlvr` |  |
| 8.6 | RL training systems | 8.5 | RL Training Systems | `sec_lm_rl_systems` |  |
| 8.7 | Overoptimization, safety and evaluating post-trained models (optional) | 8.6 and 8.7 | Reward Overoptimization; Safety Evaluation and Red-Teaming | `sec_lm_overoptimization`; `sec_lm_safety` | 8.7 hosts A8 |
| 9.1 | Knowledge distillation for language models | 9.1 | Knowledge Distillation | `sec_lm_distillation` |  |
| 9.2 | On-policy distillation | 9.2 | On-Policy Distillation | `sec_lm_onpolicy_distillation` | hosts the mini-assignment |
| 9.3 | Training reasoning models | 9.3 | Training Reasoning Models | `sec_lm_reasoning_training` |  |
| 9.4 | Verifiers and process supervision | 9.4 | Verifiers and Process Supervision | `sec_lm_verifiers` |  |
| 9.5 | Parallel test-time compute | 9.5 | Parallel Test-Time Compute | `sec_lm_parallel_tts` |  |
| 9.6 | Sequential and adaptive test-time compute | 9.6 | Sequential and Adaptive Test-Time Compute | `sec_lm_sequential_tts` | merged with blueprint section 9.7 or 9.6 |
| 9.7 | Efficient reasoning | not in the plan as a section (now 9.6's length penalties and overthinking, taught generically) | Sequential and Adaptive Test-Time Compute | `sec_lm_efficient_reasoning` is not a plan label |  |
| 10.1 | Tool calling | 10.1 | Tool Calling | `sec_lm_tool_calling` |  |
| 10.2 | Structured outputs and constrained decoding | 10.2 | Structured Outputs and Constrained Decoding | `sec_lm_structured_outputs` |  |
| 10.3 | Agent loops and harnesses | 10.3 | Agent Loops and Harnesses | `sec_lm_agent_loops` | hosts A9 |
| 10.4 | Training agents with reinforcement learning | 10.4 | Training Agents with Reinforcement Learning | `sec_lm_agentic_rl` |  |
| 10.5 | Evaluating and securing agents | 10.5 | Evaluating and Securing Agents | `sec_lm_agent_eval` |  |
| 11.1 | Quantization for inference | 11.1 | Quantization for Inference | `sec_lm_quant_inference` |  |
| 11.2 | Outliers and equivalent rescaling | 11.2 | Activation Outliers and Equivalent Rescaling | `sec_lm_outliers` |  |
| 11.3 | GPTQ and reconstruction-based quantization | 11.3 | Reconstruction-Based Quantization | `sec_lm_gptq` |  |
| 11.4 | Rotations and vector quantization | 11.4 | Rotations and Incoherence | `sec_lm_rotations` | (codebooks to the FR notebook) |
| 11.5 | Quantization-aware training and native low-bit formats | 11.5 | Quantization-Aware Training and Native Low-Bit Formats | `sec_lm_qat` |  |
| 11.6 | Pruning and lossless compression | 11.6 | Pruning and Sparsity | `sec_lm_pruning` | (lossless coding to the FR notebook) |
| 12.1 | A fast decoder on one GPU | 12.1 and 12.2 | A Fast Decoder on One GPU; Decode Attention Kernels | `sec_lm_fast_decoder`; `sec_lm_decode_kernels` | the kernel half is NEW as 12.2 |
| 12.2 | Speculative decoding | 12.4 | Speculative Decoding | `sec_lm_speculative` |  |
| 12.3 | Drafters and multi-token prediction | 12.5 | Drafters and Multi-Token Prediction | `sec_lm_drafters` | (takes blueprint 14.8's MTP paragraph) |
| 12.4 | Compressing the KV cache at inference time | not in the plan as a section: now KV-Cache Quantization; KV-Cache Eviction and Selective Reading of 12.6 and 12.7 | KV-Cache Quantization; KV-Cache Eviction and Selective Reading | `sec_lm_kv_quantization`; `sec_lm_kv_eviction` (`sec_lm_kv_compression` has no section |  |
| 13.1 | Workloads and service objectives | 13.1 | Workloads and Service Objectives | `sec_lm_serving_workloads` |  |
| 13.2 | Continuous batching and chunked prefill | 13.2, 13.5 and 12.3 | Continuous Batching and Chunked Prefill; Scheduling under Overload; Deterministic and Batch-Invariant Inference | `sec_lm_continuous_batching`; `sec_lm_overload_scheduling`; `sec_lm_deterministic_inference` | admission, preemption and fairness are 13.5; batch invariance is 12.3 |
| 13.3 | Paged KV memory and prefix caching | 13.3, 13.4 and 13.6 | Paged KV Memory; Prefix Caching and Shared-Prefix Workloads; KV-Cache Offload, Stores and Prefix-Aware Routing | `sec_lm_paged_kv`; `sec_lm_prefix_caching`; `sec_lm_kv_stores` | 13.3 hosts A10; the offload subsection is 13.6's |
| 13.4 | Serving across GPUs and nodes | 13.7, 13.8 and 13.6 | Tensor and Expert Parallelism for Serving; Prefill and Decode Disaggregation; KV-Cache Offload, Stores and Prefix-Aware Routing | `sec_lm_distributed_serving`; `sec_lm_disaggregation`; `sec_lm_kv_stores` | stores and routing across replicas are 13.6's |
| 13.5 | Serving many models (optional) | 13.9 | Serving Many Adapters | `sec_lm_multi_model_serving` | routing between models is an exercise |
| 14.1 | Reading an architecture | 14.1 | Architecture Configurations | `sec_lm_reading_architectures` |  |
| 14.2 | Mixture-of-experts design at scale | 14.2 | Mixture-of-Experts Design at Scale | `sec_lm_moe_design` |  |
| 14.3 | Latent attention and cross-layer KV sharing | 14.3 | Latent Attention and Cross-Layer KV Sharing | `sec_lm_mla` |  |
| 14.4 | Refinements of softmax attention | 14.4 | Softmax Attention Refinements | `sec_lm_attention_refinements` |  |
| 14.5 | Learned sparse attention | 14.5 | Learned Sparse Attention | `sec_lm_sparse_attention` |  |
| 14.6 | Hybrid models at scale | 14.6 | Hybrid Models at Scale | `sec_lm_hybrids_at_scale` |  |
| 14.7 | The residual stream across depth | 14.7 | The Residual Stream across Depth | `sec_lm_residual_stream` |  |
| 14.8 | Beyond fixed tokens and left-to-right decoding (optional) | not in the plan as a section (multi-token prediction is in 12.5; byte patching and diffusion language models are in the Chapter 14 FR notebook) | Drafters and Multi-Token Prediction; Further Reading: Architectures | `sec_lm_beyond_tokens` is not a plan label |  |
| 14.9 | Designing under a budget (optional) | not in the plan as a section (the budget checklist and the capstone are in the Chapter 14 FR notebook) | Further Reading: Architectures | `sec_lm_design_budget` is not a plan label; `sec_lm_capstone` is kept |  |

| Plan section | Title | Label |
|---|---|---|
| 1.7 | Models behind an API | `sec_lm_api` |
| 2 FR | Further Reading: Encoders and Retrieval | `sec_lm_encoders_fr` |
| 5.2 | Kernels, Tiles and Fusion | `sec_lm_kernels` |
| 6.8 | Profiling and Performance Debugging of Distributed Runs | `sec_lm_distributed_profiling` |
| 8 FR | Further Reading: Post-Training | `sec_lm_rl_fr` |
| 11 FR | Further Reading: Compression | `sec_lm_compression_fr` |
| 12.3 | Deterministic and Batch-Invariant Inference | `sec_lm_deterministic_inference` |
| 12.8 | Inference on CPUs and Devices | `sec_lm_on_device` |
| 14 FR | Further Reading: Architectures | `sec_lm_architectures_fr` |
