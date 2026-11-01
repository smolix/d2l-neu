# Chapter 7: Fine-Tuning — Blueprint

**Label:** `chap_lm_finetuning` · **Arc:** IV. Post-training (7 Fine-Tuning → 8 Preference Optimization and RL → 9 Distillation and Reasoning) · **Sections:** 7.1–7.9 · **Tiers:** T1 (7.2–7.7, 7.9), T2 (7.1), T3 (7.8). Per-section tier is the heavier one; every T2/T3 notebook has a T1 flag.

## Chapter overview

### Draft introduction (for the chapter's `index.md`)

A base model completes text; it does not follow instructions, keep to a format, or stop when it should. Fine-tuning closes that gap, and it is the first place in this book where a model you did not train becomes a model you own. This chapter treats adaptation as an engineering problem with a bill: bytes of memory, hours of GPU time, and capabilities forgotten. We begin with full fine-tuning because it sets the baseline for all three costs (7.1), then make the data path exact: how conversations are serialized, masked and packed (7.2), and where instruction data comes from and how it is selected (7.3). Next we cut the bill. LoRA (7.4) and its relatives (7.5) shrink trainable state; quantizing the frozen weights shrinks the rest, which is where this book first teaches quantization (7.6). Every method is written from scratch, checked against a reference, and then run through the libraries practitioners use (7.7). Finally we scale the same recipes to a 27B-parameter hybrid model on one 8-GPU node, and reach models larger than a node through a hosted fine-tuning API (7.8), and combine or retain what several fine-tunes learned (7.9). The chapter ends by handing a supervised checkpoint to Chapter 8, where preference optimization and reinforcement learning start from it.

### Learning objectives (measurable; refines v5)

After this chapter a reader can:

1. **Decide.** Given a goal (format, style, skill, new knowledge), a dataset size and a GPU budget, choose among prompting (1.3), retrieval (2.5), PEFT and full fine-tuning, and defend the choice with a memory ledger and a measured forgetting number. (7.1, 7.9)
2. **Account.** Predict the peak GPU memory of a fine-tuning run to within 15% from the model config, the method (full, LoRA, QLoRA), the sequence length and the micro-batch, and explain the residual. (7.1, 7.4, 7.6, 7.8)
3. **Serialize.** Write a chat renderer whose training prefixes equal the inference prefixes token for token, and prove that the loss mask touches only the intended positions; show that packing with a block-diagonal mask reproduces unpacked logits. (7.2)
4. **Select data.** Build an instruction mixture with a license audit, a decontamination report and a selection rule, and compare it with a random subset at an equal token budget with a paired bootstrap interval. (7.3)
5. **Implement LoRA.** Implement `LoRALinear` with merge/unmerge, state the rsLoRA argument for the `α/√r` multiplier and test it at a fixed learning rate across ranks (the default stays PEFT's `α/r`), set the A/B learning-rate ratio by the LoRA+ argument, and verify merged = unmerged outputs to a stated tolerance. (7.4)
6. **Choose a PEFT method** by what it changes (initialization, parameterization, where parameters enter, optimizer state) and compare DoRA, prompt tuning and GaLore with LoRA at matched memory. (7.5)
7. **Quantize.** Implement absmax, blockwise and NF4 quantizers; derive the Δ²/12 error and the block-size dependence; explain why NF4 bins have equal probability and why that differs from the MSE-optimal Lloyd–Max grid; account exactly for storage including scales and double quantization; QLoRA-fine-tune a 14B model on a 24 GB GPU, where its bf16 weights alone would not fit. (7.6)
8. **Cross-check libraries.** Reproduce one recipe in two toolchains and explain every difference in the first-step losses and in throughput by pointing at the from-scratch oracle. (7.7)
9. **Scale.** Fine-tune a 27B hybrid (Gated DeltaNet + attention) model across 8 GPUs with FSDP2 + LoRA/QLoRA, report memory per GPU and step time, and predict full fine-tuning's requirement from the same ledger on 40 GB and 80 GB GPUs; for a model larger than the node, say what a hosted fine-tuning API does and what it hides. (7.8)
10. **Merge.** Apply task arithmetic, TIES, DARE and SLERP to two fine-tunes, state which assumption each makes, and show when merging fails. (7.9)

### Prerequisites

Book labels (what is used; what must not be re-taught):

- `sec_parameters` (`subsec_freezing`): `requires_grad=False`, counting parameters and bytes, tied parameters. 7.1 uses these; do not re-teach.
- `sec_custom_layer`: writing an `nn.Module` with parameters and buffers, and a custom `autograd.Function` (the straight-through estimator is the model for the dequantize-matmul function in 7.6).
- `sec_numerics`, `sec_hardware` (`fig_float_formats`, `tab_gpu_specs`): fp32/bf16/fp16/fp8/fp4 layouts, "exponent sets range, mantissa sets precision". 7.6 recaps this in one paragraph and moves to integer and codebook grids.
- `sec_memory_precision`: the 16 bytes/param ledger, `torch.cuda.max_memory_allocated`, activation checkpointing, gradient accumulation. 7.1 extends the ledger; the formula is not re-derived.
- `sec_adamw` (`subsec_adamw-memory`): AdamW update, decoupled decay, 20 bytes/param bf16 pattern, 8-bit optimizers named. 7.6 implements the 8-bit idea.
- `sec_scheduler`: warmup, cosine, WSD. Fine-tuning uses short warmup plus cosine or linear decay; no re-teaching.
- `sec_gpt`, `sec_transformer-block`, `sec_kv-cache`: `d2l.GPT`, `d2l.TransformerBlock`, `d2l.FeedForward` (`W_g`, `W_1`, `W_2`), `d2l.GQAAttention` (`W_q`, `W_k`, `W_v`, `W_o`). Module names fix LoRA target matching.
- `sec_multi_gpu`, `sec_multi_gpu_concise` (`subsec_mgp-fsdp`): DDP, FSDP/ZeRO derivation and the `fully_shard` sketch. 7.8 runs it; it does not re-derive sharding.
- `sec_imitation` (`subsec_il_sft`): SFT as behavior cloning; context is state, next token is action. 7.2 cites it once.
- `sec_mdl-svd-low-rank` (`sec_mdl-svd-modern-dl`, `eq_mdl-lora-ratio`, `fig_mdl-la-lora`): rank-r update, parameter ratio r(m+n)/mn, Eckart–Young ceiling, PiSSA idea. 7.4 starts from `h = Wx + BAx`; it does not re-derive the ratio.
- `sec_read_write`: safetensors, checkpoints. `sec_repro`: seeds.
- `sec_fine_tuning` (CV chapter) is the legacy target of 8 references in Basics; 7.1 is their new home (see 7.1 "Contract").

Earlier sections of this part:

- 1.1/1.2: `d2l.load_hf`, `d2l.load_qwen3` (R1 parity checks only). 1.3: prompting baselines. 1.4: `d2l.score_choices`, `d2l.pass_at_k`. 1.5: `d2l.EvalSuite`, `d2l.paired_bootstrap`, `d2l.judge_pairwise`. 2.5: RAG baseline over R3 for 7.9.
- 3.3 (MinHash dedup) and 3.5 (synthetic data) for 7.3; 4.1 (`d2l.PackedStream`, document ids) for 7.2; 4.2 (`d2l.load_reference_model`, R2) as the *non*-Qwen model in 7.2's cross-tokenizer edge case and 7.9's incompatible-merge example; 4.4 (block scaling, FP8/FP4 formats) for 7.6.
- 6.1 (`d2l.launch`), 6.2 (`sec_lm_fsdp`, FSDP2) for 7.8. 5.1 (`d2l.chip_specs()`), 5.6 (cost model, `lm-hw-prices`), 4.2 (`flops_per_token`, `mfu`), 6.2 (`fsdp_wrap`, `fsdp_train_step`) for 7.8's estimates.

### Section dependency graph

```
7.1 full FT ──► 7.2 SFT ──► 7.3 data ──────────────┐
   │              │                                 │
   │              └──────────────┐                  ▼
   └──► 7.4 LoRA ──► 7.5 beyond ─┼──► 7.7 toolchains ──► 7.9 merging
            │                    │         ▲
            └──► 7.6 QLoRA ──────┴──► 7.8 multi-GPU (also needs 6.2)
7.2 ──► handoff to Chapter 8 (A7 part 1)
```

Reading orders: *NLP with Pretrained Models* course = 7.1–7.4. *LLM* course = 7.2, 7.4, 7.6. *Systems* course = 7.8 (+7.6 for the byte accounting). 7.5 and 7.9 can be skipped without breaking later chapters (11.x needs only 7.6; 8.x needs 7.2 and optionally 7.4 for `d2l.lora_disabled`).

### Running examples and shared artifacts

Used:
- R1 (`d2l.load_qwen3`): parity oracle for LoRA/QLoRA on a model we built (7.4, 7.6).
- R2 (`d2l.load_reference_model`): used in 7.2 (a different tokenizer/template to expose serialization assumptions) and 7.9 (merge across unrelated bases fails).
- R3 (the book as a corpus, `d2l.book_corpus`) and `d2l.book_qa`: the "new knowledge" task in 7.9 (fine-tune on the book vs retrieve from it).
- Models (HF IDs, Apache-2.0, verified 2026-10-28 against the Hub API): `Qwen/Qwen3-0.6B-Base` (596M), `Qwen/Qwen3-1.7B-Base` (1.72B), `Qwen/Qwen3-8B-Base` (8.19B), `Qwen/Qwen3-14B-Base` (14.77B; 7.6), and the post-trained `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B` (post-trained counterparts, used as ceilings; `Qwen3-1.7B` is also the 7.3 Magpie generator); `Qwen/Qwen3.5-2B` (7.3 scorer) and `Qwen/Qwen3.5-9B` (the part's judge). 7.8: `Qwen/Qwen3.8-27B` (27.78B in the file, 26.90B text-only; Apache-2.0; hybrid Gated DeltaNet + attention) as the headline, with `Qwen/Qwen3-32B` (32.76B dense, Apache-2.0) as the fallback, `Qwen/Qwen3-8B-Base` for the full-fine-tuning calibration arm and `Qwen/Qwen3.5-2B-Base` (2.27B in the file, 1.88B text-only; the same architecture family) for the T1 and T2-pair checks.
- Chapter 1 suite as the retention suite.

Produced (`d2l.DATA_HUB` keys; formats in each section's **Exports**):
- `lm-ch07-banking77-runs` (7.1), `lm-sft-qwen3-0p6b` (7.2; the A7 handoff checkpoint), `lm-sft-mix-v1` and `lm-sft-qwen3-0p6b-v1` (7.3), `lm-ch07-lora-sweeps` (7.4), `lm-ch07-quant-errors` (7.6), `lm-ch07-toolchain-logs` (7.7), `lm-ch07-fsdp-27b-logs` (7.8, T3), `lm-ch07-merge-adapters` (7.9).
- Code: `d2l.LoRALinear`, `d2l.quantize_blockwise`, `d2l.quantize_nf4`, `d2l.dequantize` (reserved) plus `d2l.QTensor`, `d2l.QuantLinear`, `d2l.add_lora`, `d2l.lora_disabled`, `d2l.render_chat`, `d2l.sft_loss`, `d2l.chunked_lm_loss` and the others listed per section.

### Compute plan

| § | Tier | Reduced T1 config | Target wall-clock | Peak memory target | Committed artifacts |
|---|---|---|---|---|---|
| 7.1 | T2 | Qwen3-0.6B-Base, seq 512, fp32 master | T2 ≤ 3 h, T1 ≤ 15 min | T2 ≤ 38 GiB (one A100 40 GB), T1 ≤ 15 GiB | run logs, eval JSON, ledger table |
| 7.2 | T1 | (native) | ≤ 15 min | ≤ 15 GiB | SFT checkpoint (A7), token/mask fixtures |
| 7.3 | T1 | (native) | ≤ 15 min (two runs of ≤ 6 min) | ≤ 15 GiB | curated manifest, license/decontam report |
| 7.4 | T1 | (native; Qwen3-0.6B-Base) | ≤ 15 min | ≤ 12 GiB | sweep logs |
| 7.5 | T1 | (native) | ≤ 15 min | ≤ 12 GiB | comparison logs |
| 7.6 | T1 | 14B QLoRA with ≤ 100 steps | ≤ 15 min | ≤ 15 GiB; the 14B arm claims a whole 24 GB GPU | error tables, memory logs |
| 7.7 | T1 | (native) | ≤ 15 min | ≤ 15 GiB | configs, loss traces |
| 7.8 | T3 | one-rank FSDP2 path (LoRA and QLoRA) on `Qwen3.5-2B-Base` on one 4090; the two-rank checks run on a T2 pair | T3 ≤ 2 h; T2 pair ≤ ~1 h (to be measured); T1 ≤ 15 min | T3 per-GPU ≤ 38 GiB (fits 8× 40 GB); T1 ≤ 15 GiB | logs, memory snapshots, `topo.txt`, `ledger.json` |
| 7.9 | T1 | (native) | ≤ 15 min (adapters cached via `DATA_HUB`; training is T1 offline) | ≤ 12 GiB | adapters, merge table |

"Target" is a design budget the author must meet by choosing token counts; it is not a prediction. All "to be measured" magnitudes are filled by the first full run and committed.

## Assignment

v5 attaches **A7 (SFT → DPO → GRPO on Qwen3-0.6B)** to Chapter 8 and lists it as spanning Chapters 7–8. Chapter 7 has no assignment of its own. This blueprint specifies **A7 Part 1 (SFT)**, which Chapter 7 owns and which Chapter 8 consumes. The notebook author places it at the end of the chapter as "Assignment A7, part 1".

### A7 part 1 — Build the SFT checkpoint (T1; 8–12 student hours)

**Label:** `sec_lm_a7_sft` (heading label; Chapter 8's A7 is `sec_lm_a7`).

**Goal.** Starting from `Qwen/Qwen3-0.6B-Base`, produce a supervised-fine-tuned checkpoint with a serialization that Chapter 8 (preference pairs, rollouts) can reuse unchanged. The checkpoint, the data and the mask conventions are the contract.

**Tasks.**
1. *Serialize* (`render_chat`). Implement `render_chat(tokenizer, messages, add_generation_prompt=False) -> (input_ids: list[int], assistant_mask: list[bool])` for the ChatML template `d2l_chatml_v1` (below). Starter code supplies the template string, 14 fixtures (single turn, multi-turn, system prompt, empty assistant answer, assistant message that contains `<|im_end|>` as text, truncation at `max_length`, two consecutive user turns, unicode, a message whose content starts with a newline and one whose content ends in a newline (BPE merges `\n` + `\n` across the role/content boundary when prompt and completion are tokenized separately), a prompt that already contains a BOS-like string, a final user turn with `add_generation_prompt=True`, and three from the 7.2 notebook) and a golden `fixtures.json` with expected `input_ids` and masks.
2. *Mask and loss* (`sft_loss`). Implement token-mean loss normalized by the number of target tokens in the *effective* batch (all micro-batches), not per micro-batch.
3. *Pack* (`pack_examples`, `block_causal_mask`). Pack conversations into rows of `max_len=2048` with best-fit-decreasing, reset positions per conversation and use a block-diagonal causal mask.
4. *Train.* Fine-tune Qwen3-0.6B-Base on `lm-sft-mix-v1` (7.3's curated mixture) for a fixed budget of `TOKEN_BUDGET` supervised tokens (default 8M, to be calibrated to ≤ 15 min on the build box; students may use full FT or LoRA r=64 on all linear layers). Hyperparameters are free except tokenizer, template, mask convention and the budget.
5. *Evaluate and write.* Report validation loss, EOS rate, retention on the Chapter 1 suite (`d2l.EvalSuite(...).run(...).scores(task)`, with `d2l.paired_bootstrap` intervals, `.mean`/`.ci`, against the base model), and a 300-word analysis (questions below).

**Autograder checks (exact tolerances).**

| # | Check | Pass condition |
|---|---|---|
| 1 | Fixtures | `input_ids` equal to golden for all 14 fixtures; `assistant_mask` equal elementwise |
| 2 | Train/inference prefix | For 200 sampled validation conversations and every assistant turn *t*, `render_chat(msgs[:t], add_generation_prompt=True)` is an exact prefix of `render_chat(msgs[:t+1])` |
| 3 | Mask coverage | Supervised tokens = assistant content + `<\|im_end\|>` only; zero supervised tokens in system/user text, role headers or the trailing newline after `<\|im_end\|>` (count equality on all fixtures) |
| 4 | Loss normalization | `sft_loss` on a 4×micro-batch accumulation equals the single large-batch loss; abs diff ≤ 1e-6 in fp32 |
| 5 | Packing equivalence | On a 2-layer random-init Qwen3-shaped model in fp32 (seed 0), per-conversation logits from a packed row equal unpacked logits; max abs diff ≤ 1e-4. An unmasked packed row must differ by > 1e-2 (the test must detect the leak) |
| 6 | Validation loss | Held-out `lm-sft-mix-v1` validation loss (token-mean, assistant tokens) in the band `[ref − 0.10, ref + 0.10]` nats, where `ref` is the median of 3 instructor runs at the same budget (**to be measured**; band width to be set from the spread of the 3 runs, never below 0.05) |
| 7 | Termination | ≥ 98% of 500 generated completions (greedy, `max_new_tokens=512`) end with `<\|im_end\|>` |
| 8 | Retention | No retention task falls below the base model by more than its paired-bootstrap 95% interval half-width (`d2l.paired_bootstrap`) |
| 9 | Artifact | The saved directory loads with `d2l.load_hf` and generates from the template's generation prompt with no extra arguments |

**Written analysis.**
(a) You compute the loss on the `<|im_end|>` token but not the `\n` after it. What goes wrong at inference if you also supervise the `\n`? What goes wrong if you supervise neither? Use your fixtures.
(b) Which of your edge cases changed the number of supervised tokens by more than 5%? Explain with token tables.
(c) Packing changes which tokens an optimizer step sees. Show that, with your mask, it does not change the *objective*, and name one setting in which it would.
(d) What happened to the retention suite, and is the change distinguishable from noise?

**T1 path.** The assignment is T1 natively. `TOKEN_BUDGET` is the lever: the 8M default targets ≤ 15 min of training on the Ada slot; a 2M-token fast path is allowed for autograder checks 6–8 with a wider band (instructor-defined).

### Contract delivered to Chapter 8 (SFT checkpoint and data)

Chapter 8's blueprint may rely on exactly the following. Any change must be agreed with the editor.

1. **Template `d2l_chatml_v1`** (committed as `data/d2l_chatml_v1.jinja` and stored in the checkpoint's `tokenizer_config.json`). ChatML as in Qwen: each message renders as `<|im_start|>{role}\n{content}<|im_end|>\n`; the generation prompt is exactly `<|im_start|>assistant\n`; **no** `<think>` blocks (reasoning formats are Chapter 9's). No default system prompt is injected (Qwen's default is not used).
2. **Special tokens.** `<|im_end|>` (id 151645) is the end-of-turn token and the stop token; `<|endoftext|>` (151643) is `pad` and the document separator when packing. The checkpoint's `generation_config.json` has `eos_token_id=[151645, 151643]`, `pad_token_id=151643`. (Values read from the pinned `Qwen/Qwen3-0.6B` generation config on 2026-10-28; the author re-checks.)
3. **Loss mask.** Supervised = assistant content tokens + the `<|im_end|>` that closes the assistant turn. Not supervised: the role header `<|im_start|>assistant\n`, the newline after `<|im_end|>`, and all system/user tokens. Chapter 8 uses *the same mask* for log-probabilities (DPO: chosen and rejected completions; GRPO: sampled completions).
4. **Data (`lm-sft-mix-v1`, DATA_HUB).** JSONL with `{"id", "source", "messages": [{"role","content"}…], "n_tokens", "split"}` and `manifest.json` (hashes, license audit, decontamination report, sampling seed). Splits: `train` (the curated subset used by the checkpoint), `val` (2,000 conversations), and **`prompt_pool`**: conversations *disjoint* from `train` and `val`, ≥ 20,000 first-user-turn prompts with their reference answers, which Chapter 8 uses for preference pairs and rollouts. (Counts of `train` follow from `TOKEN_BUDGET`.)
5. **Checkpoint (`lm-sft-qwen3-0p6b`, DATA_HUB; produced by 7.2 on `lm-sft-mix-v0`, with `lm-sft-qwen3-0p6b-v1` as the curated-data variant).** Full bf16 weights in safetensors (`Qwen3ForCausalLM` layout, the merged weights if LoRA was used), tokenizer with the template, `train_config.yaml`, `train_log.jsonl` (step, tokens, loss, lr), and `eval.json` (validation loss, EOS rate, retention scores with per-item arrays so Chapter 8 can run paired bootstraps against it). Base revision pinned in `train_config.yaml`.
6. **Utilities Chapter 8 imports from this chapter:** `d2l.render_chat`, `d2l.sft_loss`, `d2l.completion_logprobs(model, input_ids, mask) -> Tensor[B, T]` (log-probs of next tokens at masked positions, zero elsewhere), `d2l.chunked_lm_loss`, `d2l.add_lora`, `d2l.LoRALinear`, `d2l.lora_disabled(model)` (a context manager that turns every adapter off, so the frozen reference policy is "the same weights with adapters off", at no extra memory).
7. **What Chapter 8 must not assume:** that the checkpoint can reason, call tools or use `<think>`; that it was packed (it may or may not have been; behavior is identical by check 5 above); that the SFT run used LoRA.
8. **Baselines Chapter 8 reports against** (measured in Chapter 8 with the shared harness, not asserted here): the base model with few-shot prompting; the SFT checkpoint; `Qwen/Qwen3-0.6B` (post-trained) as the ceiling.

## Teaching sources

v5's five chapter sources, kept, with what to take from each:

- Thinking Machines, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): the experimental protocol for LoRA vs full FT (LoRA on all layers including MLP; learning-rate ratio; capacity limits on large datasets; low-rank is enough for RL). Use for 7.4's sweep design and 7.8; treat its numbers as hypotheses on our model and data (as the Codex plan advises), not prescriptions.
- Sebastian Raschka, [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms): rank/alpha experiments with seed-variance caveats; 7.4 exercises.
- [HF smol course](https://huggingface.co/learn/smol-course/en/unit0/1): the gentlest SFT/LoRA path in TRL; 7.2, 7.7 for students who need a library-first on-ramp.
- [PEFT docs](https://huggingface.co/docs/peft/index): configuration reference for 7.4, 7.5, 7.7.
- Answer.AI, [FSDP-QLoRA](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html): the engineering account of making quantized frozen weights shard; 7.8 and the memory story of 7.6.

Added from the Codex plan (Chapter 8 teaching sources there) and the author's reading:

- [torchtune](https://github.com/pytorch/torchtune) (now at github.com/meta-pytorch/torchtune): readable single-device and distributed recipes. **Its README states it is no longer actively maintained** (checked 2026-10-28); cite it as a reading source and as a frozen, pinned reference (see 7.7).
- [Tülu 3](https://arxiv.org/abs/2411.15124): the complete open SFT recipe with data mixtures; the model for 7.2/7.3 and for A7.
- Stanford [CS336](https://cs336.stanford.edu/) (alignment/SFT assignment): the course precedent for token-level masks and checkable SFT.
- Hugging Face [Alignment Handbook](https://github.com/huggingface/alignment-handbook): recipes as config files (verified 200 on 2026-10-28); 7.7's "configuration as experiment record".
- Hugging Face [chat templating](https://huggingface.co/docs/transformers/chat_templating) (verified 200): the serialization contract for 7.2.
- Hugging Face [bitsandbytes 4-bit blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes) (verified 200): the practitioner's account of NF4/double quantization for 7.6.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (new, verified): the chat template, thinking/non-thinking modes and the base/post-trained pairing the chapter relies on.

## Sections

### 7.1 Full fine-tuning

- **Label:** `sec_lm_full_finetuning` · **Tier:** T2 (reduced T1 config: `Qwen3-0.6B-Base`, seq 512, micro-batch 4, fp32 master weights, activation checkpointing on, flag `CFG='t1'`) · **Budget:** T2 ≤ 3 h on 1× A100 40 GB (PCIe), peak ≤ 38 GiB (usable); T1 ≤ 15 min, peak ≤ 15 GiB · **Codex units:** 8.1.
- **Builds on (book):**
  - `sec_parameters` (`subsec_freezing`, "Counting Parameters, Counting Bytes", tied parameters): `requires_grad=False`, byte ledger. Reused; not re-taught.
  - `sec_memory_precision` (`subsec_mp-anatomy`, `subsec_mp-checkpointing`, `subsec_mp-accumulation`): the 16 bytes/param floor and activation checkpointing. This section extends the ledger to the 20-byte bf16-with-master pattern and adds the logits term; it does not re-derive checkpointing.
  - `sec_adamw` (`subsec_adamw-memory`): the three precision setups table; extended, not repeated.
  - `sec_scheduler`: warmup + cosine, used as is.
  - `sec_read_write`: safetensors save/load of the fine-tuned model.
  - `sec_fine_tuning` (CV chapter): this section is the new target of the 8 `sec_fine_tuning` references in Basics (see "Contract" below).
- **Builds on (this part):** 1.1 (`d2l.load_hf`, `d2l.config_memory`), 1.3 (prompted baseline), 1.4 (`d2l.score_choices`), 1.5 (`d2l.EvalSuite`, `d2l.paired_bootstrap`), 4.1 (`d2l.chunked_lm_loss`, imported, not defined here; `:numref:`sec_lm_data_path``).
- **Exports (`#@save`):**
  - `d2l.ft_memory(config, seq_len, micro_batch, method='full', precision='bf16_master32', optimizer='adamw', lora_r=0, lora_targets=None, quant=None, n_gpus=1, shard='none', grad_ckpt=True, loss='full', act_coef=None) -> dict[str, int]`. Bytes for `weights`, `grads`, `master`, `opt_state`, `adapters`, `activations`, `logits`, `total`. `shard='zero3'` divides weights, grads, master and optimizer state by `n_gpus`. `act_coef` is the per-token-per-layer working-set coefficient, calibrated once from a measured run in this notebook; 7.4, 7.6 and 7.8 extend this function (they add arguments, never change defaults).
  - `d2l.freeze_until(model, k, embeddings=False) -> list[str]`. Freezes the embeddings (optional) and the first `k` transformer blocks; returns the trainable parameter names.
  - `d2l.layerwise_lr_groups(model, base_lr, decay=0.9, weight_decay=0.0) -> list[dict]`. AdamW parameter groups: block `i` of `L` gets `base_lr * decay**(L-1-i)`; norms and biases get no weight decay.
  - (`d2l.chunked_lm_loss` is **not** exported here: it is owned by 4.1 and imported: `d2l.chunked_lm_loss(hidden, weight, labels, chunk=8192, z_loss=0.0, ignore_index=-100) -> (sum_loss, n_valid)`.)
  - `d2l.LastTokenClassifier(backbone, num_labels, pool='last')`. A linear head on the final hidden state of the last non-pad token.
  - Data artifact `DATA_HUB['lm-ch07-banking77-runs']`: JSONL of per-run configs and per-example predictions (for paired bootstrap), plus `ledger.json`.
- **Models and data:**
  - `Qwen/Qwen3-1.7B-Base` (1.72B; T2) and `Qwen/Qwen3-0.6B-Base` (596M; T1), Apache-2.0, pinned by revision. Config read from the Hub on 2026-10-28: 1.7B has `hidden_size=2048`, 28 layers, 16 heads / 8 KV heads, `head_dim=128`, `intermediate_size=6144`, vocabulary 151,936, tied embeddings; 0.6B has `hidden_size=1024`, 28 layers, `intermediate_size=3072`, tied embeddings.
  - Task data: `PolyAI/banking77`, CC-BY-4.0, 10,003 train / 3,080 test, 77 intents (verified 2026-10-28). Carve 1,003 of train as validation (seeded). Warn that two label strings contain `?` and a capital letter (`reverted_card_payment?`, `Refund_not_showing_up`); the generative arm normalizes them.
  - Retention suite (Chapter 1): `allenai/ai2_arc` (CC-BY-SA-4.0), `openai/gsm8k` (MIT), `google/IFEval` (Apache-2.0), `TIGER-Lab/MMLU-Pro` (MIT). Evaluated with `d2l.EvalSuite(['arc_challenge', 'gsm8k', 'mmlu_pro'], n=500, seed=0).run(model, tok)`, which returns a `d2l.EvalResult`; per-item scores come from `.scores(task)` (`np.ndarray` ordered by `item_id`), and comparisons from `.compare(other)` or `d2l.paired_bootstrap` (`BootstrapResult` with `.mean`, `.ci`). IFEval is not a built-in task id in 1.5: it is wrapped as a `d2l.EvalTask` here.
- **Packages:** `torch`, `transformers`, `datasets`, `accelerate` (pins in 7.7's `data/ch07_requirements.txt`); `bitsandbytes` only for the optional 8-bit AdamW arm.

#### Narrative arc

1. **The question.** Opening cell: load `Qwen3-1.7B-Base`, print `d2l.ft_memory(config, 1024, 4, method='full')` as a table (bf16 weights 3.44 GB, grads, fp32 master, `m`, `v`, logits), then try one naive AdamW step in plain bf16 with `lr=1e-5` and print two symptoms: (i) the peak-memory overshoot versus the table (to be measured; the logits term is the surprise at vocabulary 151,936) and (ii) the **fraction of weight elements whose bf16 value did not change** after the step (the update was absorbed by rounding; to be measured, expected large; derivation D2).
2. **When fine-tuning is the right tool.** A decision table (not a section on prompting or RAG): rows = full FT, PEFT (7.4–7.6), few-shot prompting (1.3), retrieval (2.5); columns = changes behavior/format, injects new facts, data needed, GPU memory, forgetting risk, serving cost. Entries cite measurements from this chapter where they exist and mark others "see 7.9". One recap paragraph on the pretrain-then-adapt paradigm: ULMFiT's three ingredients (discriminative learning rates, slanted triangular schedule, gradual unfreezing) and why a large LM often needs fewer of them.
3. **The memory budget of full fine-tuning.**
   - Bytes per parameter for four precision schemes: fp32 everything (16), bf16 weights/grads with fp32 master (`2+2+4+4+4=16`; 20 with an fp32 gradient accumulator, as `sec_adamw` shows), pure bf16 (`2+2+2+2=8` with bf16 `m`,`v`), 8-bit optimizer state (`2+2+4+1+1=10`). Equation `eq_lm_fft_bytes` (derivation D1).
   - The logits term: `B·T·V·4` bytes for fp32 logits plus the same for their gradient; table over `V=151,936`; 4.1's `d2l.chunked_lm_loss` removes it. Cell: measured peak with and without chunking. *Recap (one paragraph, `:numref:`sec_lm_data_path``):* the loss is a sum over tokens, each term depends only on its hidden state `h_t` and the output matrix `W`, so the tokens can be processed in chunks whose `[chunk, V]` logits are recomputed in backward under `torch.utils.checkpoint` and never kept whole; the function returns `(sum_loss, n_valid)` with `ignore_index=-100` positions excluded, so there is no reduction argument: a mean is `sum_loss / n_valid`, and under gradient accumulation the divisor is the supervised-token count of the whole effective batch (7.2). Call site in this notebook: `sum_loss, n_valid = d2l.chunked_lm_loss(h, W, labels); loss = sum_loss / n_valid`.
   - Activation checkpointing on/off; calibrate `act_coef` from two measured configs, validate on a third; report error (objective: within 15%).
   - Fused AdamW (`torch.optim.AdamW(fused=True)`) allocates no per-parameter temporaries; the unfused foreach path spikes (measured; `sec_memory_precision` explains the spike).
4. **Learning rate, schedule, epochs.** Cell sweeps `lr ∈ {5e-6, 1e-5, 2e-5, 5e-5}` (T2; two values at T1) with 3% warmup and cosine decay to 10%, 3 epochs on banking77, reports validation accuracy per epoch and retention. Expected direction: smaller learning rates retain more; the best task accuracy sits in a mid range; overfitting appears by epoch 3 for the larger rates. Magnitudes are to be measured; the Tülu 3 paper's SFT hyperparameter table (the author reads it) is the cited external reference for 7B–70B scale, and the notebook must not transplant its numbers to 1.7B.
5. **How far to unfreeze.** `d2l.freeze_until(model, k)` for `k ∈ {L (head only), L−4, L−14, 0 (all)}` plus embeddings frozen/unfrozen. Cells measure per setting: trainable fraction, peak memory, step time. Two mechanical facts the cells must show: (a) freezing the first `k` blocks removes their gradients *and* optimizer state; (b) backward stops at the lowest trainable block, so activations of the frozen prefix are not saved (the input embedding output has `requires_grad=False`). Discriminative learning rates via `d2l.layerwise_lr_groups` with `decay ∈ {1.0, 0.9, 0.8}`.
6. **Task heads versus text-to-text.** Same backbone, same 1,003-example-to-full-train budgets `n ∈ {64, 256, 1024, 9000}`:
   - *Head arm:* `d2l.LastTokenClassifier` (77-way linear head on the last token's final hidden state; head randomly initialized, learning rate 10× the backbone's).
   - *Generative arm:* prompt `"Message: …\nIntent:"`, target is the normalized label string plus `<|endoftext|>`; evaluate (i) by constrained scoring with `d2l.score_choices` over the 77 label strings (always valid by construction) and (ii) by free greedy generation, counting invalid outputs (not one of the 77).
   - Table `tab_lm_ft_heads`: accuracy with 95% bootstrap interval, invalid-output rate, parameters added, p50 latency per example. Discussion: the head arm adds 77·hidden parameters and needs a new output space; the generative arm reuses the LM head and can change tasks by changing the prompt.
7. **Forgetting, measured.** Evaluate base and each fine-tune on the retention suite; plot retention drop against learning-rate and against unfreezing depth (figure `fig_lm_ft_forgetting`). Summarize "LoRA Learns Less and Forgets Less" as the pointer to 7.4: it reports the learning–forgetting trade-off for full fine-tuning versus LoRA on code and math; this section only measures the full-fine-tuning side.
8. **Contract paragraph.** A box stating where the legacy `sec_fine_tuning` promise ("when fine-tuning helps and how far to unfreeze") is answered: subsections 2 and 5. Basics' references should be re-pointed to `sec_lm_full_finetuning`.

#### Derivations

**D1. The bytes-per-parameter ledger (`eq_lm_fft_bytes`).**
Statement: for `P` parameters, weights in `b_w` bytes, gradients in `b_g`, optional master copy `b_m` (0 if none), optimizer state `s` values of `b_s` bytes each, static memory is `P·(b_w + b_g + b_m + s·b_s)`.
Assumptions: AdamW (`s=2`); no gradient accumulator; no sharding.
Steps:
1. List what must persist across the optimizer step: weights used in forward/backward, their gradients, any higher-precision copy the update is applied to, and the optimizer's moments.
2. fp32 everything: `4+4+0+2·4 = 16`.
3. bf16 compute with fp32 master: `2+2+4+2·4 = 16`; with an fp32 gradient accumulator `+4 = 20` (the `sec_adamw` table).
4. Pure bf16: `2+2+0+2·2 = 8` (but see D2: updates are lost).
5. 8-bit state with blockwise scales (block 2048, fp32 absmax): `s·b_s = 2·(1 + 4/2048)`; total `2+2+4+2.004 = 10.004`.
6. ZeRO-3 over `N` GPUs divides every term by `N` (recap of `subsec_mgp-fsdp`; not re-derived).
Check cell: compute the table for `Qwen3-1.7B-Base` from `config_memory`'s parameter count and compare with `torch.cuda.max_memory_allocated()` for a one-step run with checkpointing on and `loss='chunked'`; residual must be explained (activations, workspaces) and is reported; static terms must match to within 3%.

**D2. bf16 update absorption (`eq_lm_bf16_absorb`).**
Statement: an additive update `Δ` to a bf16 weight `w ≠ 0` changes the stored value only if `|Δ| ≥ ½·ulp(w)`, where `ulp(w) = 2^{⌊log2|w|⌋-7}` (bf16 has 7 explicit mantissa bits). With Adam, `|Δ| ≈ lr` for most elements early in training, so the weights that can move are those with `|w|` below about `256·lr` (the exact cutoff is a binade boundary, so within a factor of 2).
Steps:
1. bf16: 1 sign, 8 exponent, 7 mantissa bits (from `fig_float_formats`). For `|w| ∈ [2^e, 2^{e+1})` adjacent bf16 numbers differ by `2^{e-7}`.
2. Round-to-nearest changes the value iff `|Δ| > 2^{e-8}`.
3. Hence the condition for a given `lr`: `2^e < 256·lr` where `2^e` is the binade floor of `|w|`, so `|w| < 512·lr` is necessary and `|w| < 256·lr` is sufficient.
4. For `lr = 1e-5`: the sufficient bound is `|w| < 2.6e-3` and the necessary bound `|w| < 5.1e-3`. Whether most Qwen weights sit above these is an empirical question the check cell answers (the author reports the measured fraction; a typical projection-weight scale of order 1e-2 would put most elements above the bound).
5. Remedies, each tested in the notebook: fp32 master weights (what the ledger charges 4 bytes for); stochastic rounding (unbiased: `E[round(w+Δ)] = w+Δ`); Kahan-compensated summation (extra bf16 buffer).
Check cell: draw `w` from the empirical weight distribution of one `Qwen3-1.7B-Base` projection, apply `Δ = −lr·sign(g)`, count changed elements; compare the measured fraction with the prediction from steps 2–3 evaluated on the same weights (agreement within 1 percentage point); repeat with stochastic rounding and show the *mean* movement matches `Δ` to within sampling error.

**D3. Chunked loss is exact (recap; the implementation and its proof sketch live in 4.1).**
Statement: `Σ_t CE(softmax(h_t W^T), y_t)` can be computed in chunks of tokens with the same value and gradient as the unchunked loss.
Steps: (1) loss is a sum over tokens; (2) each term depends only on `h_t` and `W`; (3) gradients w.r.t. `W` add across chunks; (4) with `checkpoint`, each chunk's logits are discarded after the forward and recomputed in backward, trading one extra `h_t W^T` matmul for `O(chunk·V)` memory.
Check: values equal to 1e-6 relative in fp32, gradients w.r.t. `h` and `W` to 1e-5; peak memory measured.

#### From-scratch implementation

Budget ≈ 90 lines (≤ 130); the 20-line `chunked_lm_loss` is no longer written here (4.1 owns it).
- `ft_memory` (45 lines): pure arithmetic over the HF config (`hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `tie_word_embeddings`), with parameters counted exactly as in the audit (linear parameters + embeddings + norms).
- `freeze_until`, `layerwise_lr_groups` (20 lines): match names on `model.layers.<i>.` (HF) and `blks.<i>.` (`d2l.GPT`).
- `LastTokenClassifier` (20 lines): gathers the hidden state at the last non-pad index from `attention_mask`; handles left and right padding (a test case).
- Tests: (1) parameter count from `ft_memory` equals `sum(p.numel())` of the loaded model exactly (tied embedding counted once); (2) 4.1's `chunked_lm_loss` vs `F.cross_entropy(..., reduction='sum')` and `n_valid` vs the label count (fp32, seed 0, `V=1000`, 3 chunk sizes, `ignore_index` positions present) ≤ 1e-6 relative (a use-site check of D3, not a re-test of 4.1); (3) classifier pooling equals manual indexing on left- and right-padded batches; (4) `freeze_until(model, k)` leaves exactly the expected names trainable and the optimizer holds no state for frozen parameters (state size from `optimizer.state_dict()`).

#### In production

- **Libraries and pins:** Transformers `Trainer` and TRL `SFTTrainer` (versions in 7.7). Torchtune's `full_finetune_single_device` recipe is described as the readable reference; it is frozen (no longer maintained) and is not executed here.
- **Oracle use:** run the same 20 optimizer steps with `Trainer` (`bf16=True`, `optim='adamw_torch_fused'`, `gradient_checkpointing=True`) and with the from-scratch loop on identical batches; the step-0 loss must agree to 1e-3 relative and the step-20 weights to bf16 tolerance (to be measured). Compare peak memory with `ft_memory`. Document the three differences that cause drift: gradient-accumulation normalization (checked in 7.2), mixed-precision master copy, and the default `weight_decay`.
- **8-bit AdamW arm** (`bitsandbytes.optim.AdamW8bit`): the memory column of the headline table. The from-scratch 8-bit optimizer is in 7.6.

#### Headline experiment

- **Hypothesis.** On banking77 with `Qwen3-1.7B-Base`, (H1) unfreezing only the top half of the blocks matches full fine-tuning's validation accuracy within noise while using less memory and forgetting less; (H2) the generative arm with constrained scoring matches the classification head at ≥ 1,000 examples but is better at 64 examples because it reuses the pretrained LM head (direction only).
- **Setup.** T2: `Qwen3-1.7B-Base`, seq 128, micro-batch 32, `lr` chosen from the sweep on the validation split (one value per arm), 3 epochs, cosine, fp32 master, 3 seeds (data order and head init). T1: 0.6B, 2 seeds, 1 epoch at `n=1024`.
- **Procedure.** Arms: {all, top-14, top-4, head-only} × {head, generative}; sizes `n` as above for `all` only; retention suite after each arm.
- **Metrics.** Test accuracy (3,080 examples), invalid-output rate, peak GiB, tokens/s, retention delta per suite task.
- **Plot spec.** Figure `fig_lm_ft_depth`: x = trainable fraction of parameters (log), y = test accuracy; one marker per arm, seed range as error bars; second panel: x = same, y = retention drop (mean over suite tasks); third panel: peak memory bar chart with the ledger's prediction as a tick on each bar. Table `tab_lm_ft_heads` as above.
- **Expected outcome.** Direction only: accuracy saturates once the top blocks are trainable; memory and forgetting fall monotonically with frozen depth; head-only underperforms. LoRA Learns Less and Forgets Less is the cited external support for the forgetting direction.
- **Noise check.** 3 seeds; pairwise differences reported with `d2l.paired_bootstrap` over the 3,080 test items, resampling within seed; the notebook prints "within noise" when the interval contains 0 and the author must not rank arms inside that interval.
- **T1 reduction.** `CFG='t1'` (0.6B, 2 seeds, arms {all, top-4}).
- **Artifacts committed for T2.** `lm-ch07-banking77-runs` (JSONL per run, per-example predictions, `ledger.json`), the figure PNGs, and the calibrated `act_coef`.

#### Figures and tables

- `fig_lm_ft_ledger`: stacked bars of weights, grads, master, optimizer, logits, activations for four configs (cell: ledger + measured peak overlay).
- `tab_lm_ft_decision`: when to fine-tune, prompt, retrieve (static table authored in Markdown; entries link to measured cells).
- `fig_lm_bf16_absorb`: fraction of weight elements that change vs `lr`, measured vs the D2 prediction; one curve with stochastic rounding.
- `fig_lm_ft_depth`, `tab_lm_ft_heads`, `fig_lm_ft_forgetting` (as above).

#### Pitfalls and misconceptions

- "bf16 training means everything in bf16": without a master copy small updates vanish (D2).
- Reporting only trainable-parameter fractions: optimizer state and the logits term decide what fits, and frozen layers still hold weights and (if reachable by gradient) activations.
- Weight tying: `lm_head` shares the embedding in the 0.6B/1.7B models; freezing "the embeddings" also freezes the output layer (the 8B model is untied). Show with `freeze_until(embeddings=True)`.
- Padding side for last-token pooling; with right padding the "last token" must be selected by `attention_mask`, not index −1.
- Dropout and weight decay on norm/bias parameters during fine-tuning (default `weight_decay=0.01` in `Trainer` applies to everything it is asked to decay).
- Epoch-wise validation inflation from reusing the validation split for choosing the learning rate; the notebook picks `lr` on validation and reports test.

#### Summary (draft bullets)

- Full fine-tuning's cost is dominated by static bytes (16–20 per parameter) and the logits term at large vocabularies; both are predictable from the config.
- Pure-bf16 updates are absorbed by rounding when `lr` is small relative to `|w|/256`; keep an fp32 master copy or round stochastically.
- Unfreezing depth, learning rate and epochs are one trade-off: how much the model may move, against what it forgets.
- A classification head adds an output space; text-to-text reuses the LM head, and constrained scoring removes invalid outputs.
- Forgetting is measured with a fixed retention suite and paired intervals, not by anecdote.

#### Exercises

1. *Check.* A 7B model is fine-tuned with AdamW in the "bf16 weights and grads, fp32 master, fp32 `m`,`v`" pattern. How many GB are static? How many if the optimizer state is 8-bit? Which term would you shrink first on a 40 GB card, and why does shrinking activations not appear in this list? *Sketch:* `7e9·(2+2+4+8)=112 GB`; 8-bit: `7e9·(2+2+4+2)=70 GB`; neither fits 40 GB, so sharding or PEFT is required; activations scale with batch and sequence, which the ledger's static terms exclude.
2. *Check.* You freeze the embeddings and the first 20 of 28 blocks of Qwen3-1.7B-Base (`requires_grad=False`, no `torch.no_grad()`). Does backward still traverse the frozen blocks? What if instead only the *last* 8 blocks are frozen? *Sketch:* no for the frozen prefix (no tensor requiring grad enters it); yes for the frozen suffix, because gradients must flow through it to reach the trainable prefix, so its activations are saved.
3. *Derive.* Show that stochastic rounding makes a bf16 update unbiased, and compute its variance for an update of `Δ = 0.3·ulp(w)`. Then explain why unbiased updates still add noise that fp32 masters avoid. *Sketch:* round up with probability `Δ/ulp`; `E = Δ`; variance `ulp²·p(1−p) = 0.21·ulp²`.
4. *Derive.* For a vocabulary `V`, hidden size `h` and `T` tokens, give the memory of the unchunked fp32 loss path and the chunked path with chunk `c`, and find the `c` at which the chunked loss costs the same as one transformer block's working set at `act_coef`. *Sketch:* unchunked `8·T·V` bytes (logits + grad, fp32) versus `8·c·V`; equate with `act_coef·T·h` per layer.
5. *Measure.* Predict, then measure, how the validation accuracy of the generative arm changes when you drop the leading space in `"Intent: "` (tokenization of the first label token). Report the interval and explain through the tokenizer (`sec_text-sequence`).
6. *Measure.* Predict the memory saved by `loss='chunked'` at `seq_len ∈ {512, 1024, 2048}` with micro-batch 4, then measure; explain any gap between the ledger and the allocator. Use the 8 B/param pure-bf16 scheme on one A100 40 GB (T2) or the 0.6B model on T1: at seq 2048 the unchunked path holds `8,192 × 151,936 × (4 + 4 + 2) B = 12.4 GB` of logit tensors, which on top of the 27.5 GB of the fp32-master scheme would exceed the 38 GiB budget.
7. *Open.* Pick a task of your own with 500 labeled examples and a retention requirement. Choose among head, text-to-text and PEFT, justify with a ledger, run it, and report accuracy and retention with intervals.

#### Further reading

- **Papers:** [ULMFiT](https://arxiv.org/abs/1801.06146) (discriminative rates, slanted triangular schedule, gradual unfreezing); [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (the learning–forgetting trade-off); [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792) (new, verified; which layers transfer); [Fine-Tuning can Distort Pretrained Features](https://arxiv.org/abs/2202.10054) (new, verified; why a randomly initialized head can damage features and why linear-probe-then-fine-tune helps).
- **Tutorials/courses:** [HF LLM Course, fine-tuning chapters](https://huggingface.co/learn/llm-course/chapter1/1); Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html) for the training loop being instrumented.
- **Code:** [torchtune](https://github.com/pytorch/torchtune) (frozen; recipe reading); [TRL](https://github.com/huggingface/trl).

#### Risks and open decisions

- **Retention suite signature** depends on 1.5; fallback: this section ships a 4-task local wrapper with `lm-eval` (`lm_eval` 0.4.13, verified on PyPI) and the same per-item output contract.
- **1.7B full FT at T2** needs ≈ 27.5 GB static (`1,720,574,976 × 16 B`) on one A100 40 GB, which leaves ≈ 13.3 GB of the 38 GiB (40.8 GB) usable for activations, one chunk of logits and workspaces. That fits the headline setup (seq 128, micro-batch 32 = 4,096 tokens per step, checkpointing and `chunked_lm_loss` on) but not an unchunked loss at 4,096 tokens on top of the fp32-master scheme (`4,096 × 151,936 × (4 + 4 + 2) B = 6.2 GB` for fp32 logits, their gradient and the bf16 copy, to be measured). Cells that deliberately use the unchunked path run the 8 B/param pure-bf16 scheme (13.8 GB static) or the 0.6B model; the 8-bit AdamW arm (≈ 10 B/param, 17.2 GB static) is the fallback for the headline arms. If the T2 slot is shared, the 0.6B config is the fallback and the plots use it.
- **T1 memory:** 0.6B at 16 B/param is 9.5 GB before activations; with checkpointing and chunked loss it should fit 15 GiB (to be measured). Fallback: 8-bit AdamW arm, or micro-batch 2.
- **Dataset license:** CC-BY-4.0 requires attribution; the notebook prints the citation.
- **`flash-attn` wheels** may not exist for the book's pinned `torch==2.15.0`; use SDPA everywhere in this section.

### 7.2 Supervised instruction tuning

- **Label:** `sec_lm_sft` · **Tier:** T1 (native; token budget is the config lever, `TOKEN_BUDGET`) · **Budget:** ≤ 15 min wall-clock in total (four ≤ 3 min comparison runs plus one packing benchmark), peak ≤ 15 GiB · **Codex units:** 8.2, 8.4.
- **Builds on (book):**
  - `sec_imitation` (`subsec_il_sft`, `subsec_il_distribution_shift`): SFT is behavior cloning, context = state, next token = action. One recap paragraph; compounding error is cited, not re-derived.
  - `sec_text-sequence`: byte-level BPE, pre-tokenization, special tokens. Not re-taught; this section only shows where the pre-tokenizer merges across a role/content boundary.
  - `sec_gpt`/`sec_kv-cache`: causal attention and masks; `sec_positional-information`: RoPE's relative-offset property is *used* in D2, not re-derived.
  - `sec_attention-at-scale`: FlashAttention and sliding windows, not re-taught; only the `varlen`/document-mask consequence for packing.
- **Builds on (this part):** 1.3 (prompt formats, chat templates as used for inference), 4.1 (`d2l.PackedStream` and its document ids; the pretraining packer that SFT packing differs from), 7.1 (`d2l.chunked_lm_loss`, `d2l.ft_memory`, the memory ledger).
- **Exports (`#@save`):**
  - `d2l.CHATML_V1` (str): the Jinja template `d2l_chatml_v1` of the Assignment contract; also written to `data/d2l_chatml_v1.jinja`.
  - `d2l.render_chat(tokenizer, messages, add_generation_prompt=False, max_length=None) -> (list[int], list[bool])`: token ids and the assistant mask; never calls `apply_chat_template` internally, so that the test can compare the two.
  - `d2l.sft_loss(logits, labels, num_items=None, ignore_index=-100) -> Tensor`: summed token cross entropy divided by `num_items` (default: number of non-ignored targets in the call). Gradient accumulation passes the effective-batch count.
  - `d2l.pack_examples(examples, max_len, strategy='bfd', pad_id=151643) -> list[dict]`: rows with `input_ids`, `labels`, `position_ids` (reset per conversation) and `seq_ids` (conversation index per token, −1 for padding). Strategies `'none'` (pad per example), `'bucket'`, `'bfd'` (best-fit decreasing).
  - `d2l.block_causal_mask(seq_ids) -> BoolTensor[B, 1, T, T]`.
  - `d2l.completion_logprobs(model, input_ids, mask, attention_mask=None, position_ids=None) -> Tensor[B, T]` (the log-probability of each next token at masked positions, 0 elsewhere; reused by Chapter 8).
  - `d2l.sft_collate(rows, pad_id)`.
  - `d2l.sft_train(model, batches, optimizer, scheduler=None, loss_fn=None, grad_accum=1, max_grad_norm=None, log_every=10, callback=None) -> list[dict]` (the trainer Chapter 9 reuses; no class, a function). `batches` is an iterable of dicts with `input_ids`, `labels`, optional `attention_mask`/`position_ids`/`seq_ids`; every `grad_accum` consecutive batches form one effective batch. `loss_fn(model, batch, num_items) -> Tensor` returns the **summed** loss of that batch divided by `num_items` (the supervised-token count of the whole effective batch, computed by `sft_train` before the first backward, per D1); the default `loss_fn` is next-token cross entropy via `sft_loss`/`d2l.chunked_lm_loss` (4.1; returns `(sum_loss, n_valid)`, so the default `loss_fn` uses `sum_loss / num_items`, never `sum_loss / n_valid` of one micro-batch). A distillation loss plugs in by returning `Σ_t KL(teacher_t || student_t) / num_items` over the same mask; extra batch keys (e.g. `teacher_logits`) are passed through untouched. Returns one dict per optimizer step: `{'step','lr','loss','grad_norm','tokens'}`; `callback(step, log)` is called after each step. An `SFTTrainer` class is not provided; TRL's is used only in 'In production'.
  - Artifacts: `DATA_HUB['lm-ch07-sft-fixtures']` (`fixtures.json`, 14 cases: ids and masks); `DATA_HUB['lm-sft-mix-v0']` (the random 40k-conversation baseline subset used in this notebook, JSONL, seed recorded); `DATA_HUB['lm-sft-qwen3-0p6b']` (the checkpoint of this notebook, trained on `lm-sft-mix-v0` at the A7 token budget with the template, mask idiom and `eos` settings of the Assignment contract; this is the key Chapter 8 consumes). The curated mixture `lm-sft-mix-v1` and the checkpoint retrained on it, `lm-sft-qwen3-0p6b-v1`, are produced in 7.3; the contract is identical for both.
- **Models and data:**
  - `Qwen/Qwen3-0.6B-Base` (596M, Apache-2.0; tied embeddings; tokenizer `eos_token=<|endoftext|>`, `add_bos_token=False`, ChatML template shipped in `tokenizer_config.json`, 4,116 characters, **no `{% generation %}` markers**). Reference/ceiling: `Qwen/Qwen3-0.6B` (post-trained; its `generation_config.json` lists `eos_token_id=[151645, 151643]`).
  - `HuggingFaceTB/smol-smoltalk` (Apache-2.0; train 460,341, test 24,229; columns `messages`, `source`; built for models under 1B; used for SmolLM2-135M/360M-Instruct per its card). Cite SmolLM2 (arXiv 2502.02737, verified).
  - Cross-tokenizer edge case: the R2 reference model's tokenizer (`d2l.load_reference_model`; 4.2) renders the same fixtures to show that a template is a property of a (tokenizer, model) pair.
- **Packages:** `transformers` (SDPA path with explicit 4-D masks; `padding_free` needs `flash-attn`, optional), `trl` (production arm), `datasets`; PyTorch `flex_attention` (docs verified 200) as the block-mask fallback when `flash-attn` is unavailable.

#### Narrative arc

1. **The question.** Opening cell renders one conversation three ways and prints the token table: (a) Qwen's shipped template with `add_generation_prompt=True`, (b) the same template rendering the full conversation for training, (c) `d2l_chatml_v1`. Measured symptom printed under the table: with Qwen's shipped template the training render of the *final* assistant turn begins `<|im_start|>assistant\n<think>\n\n</think>\n\n…`, while the default inference prompt ends at `<|im_start|>assistant\n`; earlier assistant turns carry no think block at all. The cell asserts `render(msgs[:t], gen=True)` is a prefix of `render(msgs[:t+1])` and shows it **fails** for the shipped template and passes for ours. (Facts read from the Hub template on 2026-10-28; the author re-prints them from the pinned revision, since a template is data that can change.)
2. **Serialization as part of the model interface.**
   - ChatML anatomy: role header, content, end-of-turn token, newline. Why `<|im_end|>` (id 151645) is the stop token and `<|endoftext|>` (151643) is pad/document separator. Check the embeddings of `<|im_start|>`/`<|im_end|>` in the *base* model against the vocabulary's typical norm (to be measured; if they are untrained the notebook shows it and why SFT must teach them).
   - Why Qwen's template is not suitable for SFT (think-block asymmetry, no generation markers); the fix is a template with no think logic (`CHATML_V1`), keeping reasoning formats for Chapter 9. **Decided (2026-10-29): the whole part uses `d2l_chatml_v1`.** Four reasons, each shown by a cell: (i) Qwen's shipped template renders think blocks only after the last user turn, so the training render of a turn is not an extension of its inference prefix (the failed prefix assertion of the opening cell); (ii) `<|endoftext|>` (151643), `<|im_start|>` (151644) and `<|im_end|>` (151645) already exist in the Qwen3 vocabulary (151,936 rows in the embedding), so adopting the template needs no vocabulary or embedding resize; (iii) the same Jinja string works for the R2 tokenizer, where `<|im_start|>` and `<|im_end|>` have ids 2 and 3 (3.6; the cell renders a fixture with both tokenizers and compares the *strings*, not the ids); (iv) the stop ids belong to the checkpoint, not to the template: the base tokenizer's `eos_token` is `<|endoftext|>`, so the SFT checkpoint must write `eos_token_id=[151645, 151643]` and `pad_token_id=151643` into its `generation_config.json` (Assignment contract, item 2) or generation will not stop at `<|im_end|>`.
   - Special-token insertion: `add_bos_token=False` here, but the notebook shows a model family where `apply_chat_template(tokenize=True)` adds BOS twice if the template also writes it (use the R2 tokenizer or a synthetic one; this is the "double-BOS" pitfall).
   - Tokenization boundaries: tokenize the rendered string at once; compare with prompt-and-completion tokenized separately. Show the cases where ids differ: content starting with `\n`, content starting with a space, and Unicode normalization.
3. **Loss masks.**
   - Completion-only (assistant tokens + `<|im_end|>`), full-sequence, and last-turn-only masks as *objective choices*; none is declared universally better. Instruction Tuning With Loss Over Instructions (new, verified) reports that the effect depends on the prompt-to-completion length ratio; the notebook measures our data's ratio (`tab_lm_sft_ratio`: mean prompt tokens, mean completion tokens per `source`).
   - Mask tables: `tab_lm_sft_mask` prints `token | id | mask | loss weight` for the 14 fixtures (visible token strings, escaped whitespace); assertions count supervised tokens per role. Checks are on ids, never on decoded text.
   - End-of-turn semantics: what is supervised for `<|im_end|>` and what for the trailing `\n` (not supervised; if it were, generation would be taught to continue after the stop token and the model would emit a newline before stopping).
   - The shift convention: the mask marks *target* tokens. The first assistant content token is a supervised target predicted at the position of the header's final `\n`; that position's own target is the content token, so the header tokens are never targets but the last header token is the *input* that produces the first supervised prediction. Diagram `fig_lm_sft_shift`.
4. **Loss normalization across micro-batches and ranks.** Derivation D1; cell compares per-micro-batch mean, global token mean and per-sequence mean on a 4-micro-batch accumulation with ragged supervised-token counts; plot gradient-norm difference. Pointer: the same normalization question reappears for DDP ranks (6.x) and in GRPO's loss aggregation (8.5).
5. **Packing without leakage.**
   - Padding waste: fraction of useful tokens for `none`, `bucket`, `bfd` on the 40k-conversation sample (cell, measured).
   - Packing independent conversations is not the pretraining stream of 4.1: here each conversation must see only itself. Block-diagonal causal mask; position reset; derivation D2 (RoPE relativity means reset is an interface requirement of varlen kernels, not of exactness).
   - Equivalence test on a 2-layer random-init Qwen3-shaped model in fp32 (CPU, seconds): per-conversation logits packed vs unpacked ≤ 1e-4; unmasked packed rows differ by > 1e-2 on the second conversation (leak detector).
   - Kernel path: explicit-mask SDPA versus `flash-attn` varlen (position-id resets, `padding_free`) versus `flex_attention` block masks; measured tokens/s and peak memory on the Ada GPU (`d2l.Benchmark`). The lesson from the Codex plan carried over: a sparse mask does not guarantee speed-up on every kernel path.
   - Truncation: best-fit packing without splitting conversations versus "wrapped" packing that cuts mid-conversation (TRL's `bfd`/`bfd_split`/`wrapped` strategies, documented); Fewer Truncations Improve Language Modeling (new, verified) is the pointer.
6. **Training run and checks on the checkpoint.** Full FT of `Qwen3-0.6B-Base` on `lm-sft-mix-v0` (chunked loss, fp32 master, AdamW, `lr` 1e-5 start point to be tuned in 7.1's protocol, 3% warmup, cosine). After training: `tab_lm_sft_samples` prints three greedy generations from the base and the SFT model on the same prompts; `tab_lm_sft_checks` prints the autograder checks 2–3 and 7 of the Assignment on the checkpoint.
7. **From SFT to what comes next.** One paragraph: SFT imitates demonstrations, so it inherits their errors and cannot exceed them; the exposure-bias discussion of `sec_imitation` returns in 8.x. SFT also teaches *format*; knowledge comes from pretraining (the LIMA hypothesis is discussed in 7.3).

#### Derivations

**D1. Normalization of the SFT loss under accumulation (`eq_lm_sft_norm`).**
Statement: let the effective batch contain `K` micro-batches with `n_k` supervised tokens, `N = Σ n_k`, and token losses `ℓ_{k,t}`. The intended objective is `L = (1/N) Σ_k Σ_t ℓ_{k,t}`. Averaging per-micro-batch token means gives `L' = (1/K) Σ_k (1/n_k) Σ_t ℓ_{k,t}`; `L' = L` iff all `n_k` are equal. Per-sequence means weight a token by `1/(S·ℓ_s)`.
Steps:
1. Write the weight of token `(k,t)` in `L`: `1/N`; in `L'`: `1/(K n_k)`.
2. Weights are equal iff `n_k = N/K` for all `k`.
3. The gradient `∇L' − ∇L = Σ_k (1/(K n_k) − 1/N) Σ_t ∇ℓ_{k,t}`; tokens of short micro-batches are over-weighted by `N/(K n_k)`.
4. Fix: divide each micro-batch's *sum* by `N` (pass `num_items=N`), with `N` known from the labels before the first backward. For DDP, `N` is the all-reduced count over ranks, and the gradient is the sum, not the mean, across ranks.
5. Length bias: per-sequence mean down-weights tokens of long answers by `1/ℓ_s`; per-token mean lets long answers dominate. Neither is neutral (a choice the notebook names, not resolves).
Check cell: two micro-batches with `n=(10, 90)` on a tiny model; compute `∇L` on the concatenated batch, `∇L'` by naive accumulation and by `sft_loss(num_items=N)`; the latter matches `∇L` to 1e-6 (fp32); the naive one differs by a computed ratio.

**D2. Packed attention equals unpacked attention (`eq_lm_pack_equiv`).**
Statement: for a causal transformer with RoPE, packing conversations `c_1..c_S` in one row with the block-diagonal causal mask yields, for each token of `c_s`, the same hidden states and logits as running `c_s` alone, up to floating-point reduction order.
Assumptions: per-token operations outside attention (norms, MLP, embeddings) have no cross-token dependence; no learned absolute positions.
Steps:
1. In layer `ℓ`, attention for query `i` in segment `s` is `Σ_j softmax_j(q_i·k_j + M_{ij}) v_j` with `M_{ij} = −∞` unless `j ≤ i` and `seg(j)=seg(i)`.
2. Terms with `seg(j) ≠ seg(i)` contribute exactly zero weight, so the sum runs over the same keys as in the unpacked run.
3. RoPE: `q_i·k_j = q'_i · R_{j−i} k'_j` depends on the offset `j−i` only (`sec_positional-information`), so an additive constant shift of positions within a segment leaves scores unchanged. Hence position reset is *not* needed for exactness.
4. Induction on layers: if inputs to layer `ℓ` agree token-wise for each segment, outputs agree; the embedding layer is token-wise.
5. Therefore logits agree; the loss on a packed row equals the sum of the per-conversation losses.
6. Why we still reset `position_ids`: HF's padding-free path and FlashAttention varlen infer segment boundaries from position resets; positions also bound the rotation angles (numerical precision) for rows of length 2,048.
Check cell: as described in narrative item 5; also run with and without position reset and show max diff < 1e-5 (supports step 3), and with the mask removed (leak).

**D3. What a packed batch costs and saves (`eq_lm_pack_cost`).** Statement: with row length `T`, mean conversation length `μ`, and fill ratio `f ∈ (0,1]`, useful-token throughput is `f ·` raw throughput; attention FLOPs of a packed row are `Σ_s ℓ_s²` (with a block-aware kernel) versus `T²` (dense masked). Steps: (1) padded batches have fill `f_pad = E[ℓ]/E[max ℓ in batch]`; (2) packed rows have fill `f_pack = Σℓ_s / T`; (3) with explicit-mask SDPA the dense `T×T` score computation is paid regardless, so the speed-up is `f_pack/f_pad` only if the model is MLP-bound (the usual regime at `T=2048, h=1024`); (4) with a varlen kernel the attention term also shrinks by `Σℓ_s²/T²`. Check cell: compute `f_pad`, `f_pack` and the attention-FLOP ratio on the sample, compare with measured tokens/s on each kernel path.

#### From-scratch implementation

Budget ≈ 120 lines.
- `render_chat` (35 lines): iterate messages; build ids by `tokenizer.convert_tokens_to_ids` for specials and `tokenizer.encode(text, add_special_tokens=False)` for text spans; mask is `True` for assistant content and the closing `<|im_end|>`. Truncation by `max_length` cuts from the end and never leaves a half-open assistant turn silently (returns the mask for whatever remains; the fixture documents it).
- `sft_loss` (8 lines) and `completion_logprobs` (15 lines).
- `pack_examples` (35 lines): sort by length descending, place each in the row with least remaining capacity that still fits; reset `position_ids`; build `seq_ids`; pad with `pad_id` and `labels=-100`.
- `block_causal_mask` (10 lines): `(seq_ids[:, :, None] == seq_ids[:, None, :]) & tril`, padding rows attend to themselves to avoid NaNs.
- `sft_collate` (10 lines).
- **Tests.** (1) 14 fixtures vs `fixtures.json`; (2) prefix property for all assistant turns of 200 validation conversations; (3) supervised-token counts by role equal expected; (4) D1 gradient equality (1e-6, fp32, seed 0); (5) D2 packed vs unpacked (≤ 1e-4 fp32), leak detector (> 1e-2); (6) `render_chat(...)` equals `tokenizer.apply_chat_template(msgs, chat_template=CHATML_V1, tokenize=True)` ids for all fixtures (the library as a cross-check of the from-scratch renderer).

#### In production

- **TRL `SFTTrainer`/`SFTConfig`** (pin in 7.7): `assistant_only_loss=True` requires `{% generation %}`/`{% endgeneration %}` in the chat template; TRL documents that for known families (including Qwen3) it patches the template automatically, which means **the template used for training is not the one in `tokenizer_config.json`**. The notebook prints both and diffs the token ids on the fixtures; the from-scratch `render_chat` is the oracle. Also: `completion_only_loss` (default on for prompt–completion datasets), `packing=True` with `packing_strategy` in `{'bfd','bfd_split','wrapped'}`, `padding_free` (FlashAttention 2 only), `loss_type`, `max_length` (default 1,024 in the pinned docs; with `truncation_mode='keep_start'`).
- **Oracle use:** run 20 steps of TRL and of the from-scratch loop on identical batches (`shuffle_dataset=False`, same seed, `packing=False`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=4`) and assert the step-0 loss to 1e-3 relative; then repeat with `packing=True` and show that TRL's per-row loss normalization matches `sft_loss(num_items)` only if the token counts are handled as in D1 (to be measured on the pinned versions; report the finding, whichever way it goes).
- **torchtune** is described (frozen, no longer maintained): its `SFTDataset` and `ChatFormat` abstractions as a design reference; not executed.
- **Axolotl/LLaMA-Factory**: configuration-driven; compared in 7.7.

#### Headline experiment

- **Hypothesis.** Identical data and token budget, different serialization/masking, produce measurably different models: (H1) training on Qwen's shipped template while prompting at inference with the default generation prompt degrades format-following and the first-token loss relative to the matched template; (H2) not supervising `<|im_end|>` leaves the model unable to stop by `<|im_end|>`; (H3) full-sequence versus completion-only loss differs by an amount that depends on the prompt/completion ratio (direction not asserted).
- **Setup.** `Qwen3-0.6B-Base`, `lm-sft-mix-v0` (40,000 conversations from smol-smoltalk, random, seed 0), `TOKEN_BUDGET` tokens (default 2M per variant, to be sized to ≤ 3 min per run), full FT, `lr` 1e-5 (tuned in 7.1), micro-batch of 8k packed tokens, 3 seeds if time allows else 1 (report which). Four variants: A = `d2l_chatml_v1` + completion-only; B = A with full-sequence loss; C = Qwen's shipped template for training rendering, default generation prompt at inference; D = A without the `<|im_end|>` supervision (EOS = `<|endoftext|>`).
- **Procedure.** Train A–D; evaluate on the 2,000-conversation validation split rendered in the **inference** format (teacher-forced assistant tokens after `<|im_start|>assistant\n`), generate 300 completions per variant with greedy decoding.
- **Metrics.** Inference-format validation loss on assistant tokens; EOS-by-`<|im_end|>` rate within 512 new tokens; mean generation length; retention on one cheap task (`gsm8k` 200 items, or the Chapter 1 dev subset).
- **Plot spec.** `fig_lm_sft_variants`: grouped bars, x = variant (A–D), left panel y = inference-format validation loss with seed range, right panel y = EOS rate; annotate each bar with run count.
- **Expected outcome.** Directions only: C worse than A on inference-format loss (it learned to expect a think block); D low on EOS-by-`<|im_end|>`; B vs A "to be measured".
- **Noise check.** 3 seeds where affordable, bootstrap interval over the 2,000 validation conversations for loss and over 300 completions for EOS; no claim about B vs A unless the paired interval excludes 0.
- **T1 reduction.** This section is T1 natively.
- **Artifacts committed.** `lm-ch07-sft-fixtures`, `lm-sft-mix-v0` manifest, `lm-sft-qwen3-0p6b`, `variants.jsonl`.

#### Figures and tables

- `fig_lm_sft_shift`: schematic of ids / labels / mask with the shift (authored SVG from a cell's output).
- `tab_lm_sft_ratio`: prompt and completion lengths by source.
- `tab_lm_sft_mask`: token / id / mask for the 14 fixtures (rendered subset in the text, full in the artifact).
- `fig_lm_sft_fill`: fill ratio for `none`, `bucket`, `bfd` at row lengths 1,024, 2,048, 4,096.
- `tab_lm_sft_kernels`: tokens/s and peak GiB for explicit-mask SDPA, flex-attention block mask, and `flash-attn` varlen (if installable).
- `fig_lm_sft_variants`, `tab_lm_sft_samples`, `tab_lm_sft_checks`.

#### Pitfalls and misconceptions

- "The chat template is just formatting": it is part of the model interface; training and inference must render identical prefixes.
- Using `apply_chat_template` for training with a template that has asymmetric history handling (think blocks), or tokenizing prompt and completion separately and concatenating.
- Supervising the newline after `<|im_end|>`, or the role header.
- Per-micro-batch means under gradient accumulation (D1).
- Packing: forgetting the mask (silent cross-contamination); forgetting that labels at conversation boundaries must be `-100` for the first token of each segment's prediction from the previous segment's last token (the shifted label at a segment's first position belongs to the previous segment).
- Padding with `<|endoftext|>` and treating it as EOS: the label mask must ignore pad, or the model learns to stop on pad.
- TRL's patched template differs from the checkpoint's template; saved checkpoints must carry the template that was *used*.

#### Summary (draft bullets)

- SFT is next-token cross entropy on demonstrations; serialization and masking decide what is learned.
- The training render must be a token-level extension of the inference prompt; verify it with prefix tests, never by reading decoded text.
- Normalize the loss by the effective-batch token count; per-micro-batch means bias the gradient.
- Packing with a block-diagonal mask is exact for RoPE models; whether it is fast depends on the kernel path.
- Libraries patch templates and defaults; the from-scratch renderer is the oracle that finds the differences.

#### Exercises

1. *Check.* A dataset's final assistant turn is `"\n\nSure."`. Tokenizing `prompt` and `completion` separately gives ids that differ from tokenizing the concatenated string. Which token differs and why does the training prefix then differ from the inference prefix? *Sketch:* `"\n"` at the end of `assistant\n` and `"\n\n"` at the start of the completion merge into one `"\n\n"` token when tokenized jointly; separately they are two `"\n"` tokens (check with the Qwen tokenizer's `encode`).
2. *Check.* State which tokens receive loss for a two-turn conversation under (a) completion-only, (b) last-turn-only, (c) full-sequence. Which of these can teach the model to write the user's side? *Sketch:* (a) both assistant turns plus their `<|im_end|>`; (b) only the last; (c) everything including role headers; only (c).
3. *Derive.* Show that, with per-token mean over the effective batch, an example's gradient contribution is proportional to its number of supervised tokens; then give the weight an example has under per-sequence-mean normalization and compute the ratio for a 20-token and a 400-token answer. *Sketch:* weights `1/N` per token vs `1/(S ℓ_s)`; ratio of per-example total weight 400/20 = 20 vs 1.
4. *Derive.* For rows of length `T` filled to fraction `f` and an explicit-mask SDPA path, the attention score work is `T²` per row. Derive the ratio of FLOPs per useful token between a packed row and a padded batch with `B` rows of the same length, as a function of `f_pack`, `f_pad`. *Sketch:* per useful token `T²/(f T)=T/f`; ratio `f_pad/f_pack`.
5. *Measure.* Predict, then measure, the fill ratio of `bfd` at `max_len ∈ {1024, 2048, 4096, 8192}` on your sample; find the knee. Explain it from the length distribution.
6. *Measure.* Replace `completion-only` by `full-sequence` loss on a prompt-heavy source (long prompts, short answers) versus a completion-heavy source; predict which source benefits, then measure with paired intervals (Shi et al. as the hypothesis).
7. *Open.* Write the chat renderer for a model family with a different template (Llama-style or Gemma-style headers) and a tool-call turn; give the fixture set that would catch your own bugs.

#### Further reading

- **Papers:** [InstructGPT](https://arxiv.org/abs/2203.02155); [Tülu 3](https://arxiv.org/abs/2411.15124) (SFT mixture, hyperparameters, loss normalization discussion for accumulation); [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394) (new, verified); [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830) (new, verified); [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027) (new, verified); [SmolLM2](https://arxiv.org/abs/2502.02737) (new, verified; SFT data for sub-1B models).
- **Tutorials:** [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer) (the objective/masking options); [HF chat templating](https://huggingface.co/docs/transformers/chat_templating); [FlexAttention blog](https://pytorch.org/blog/flexattention/) (verified 200).
- **Code:** [TRL](https://github.com/huggingface/trl); [Alignment Handbook](https://github.com/huggingface/alignment-handbook); [torchtune](https://github.com/pytorch/torchtune) (frozen).

#### Risks and open decisions

- **Template drift:** Qwen can edit `tokenizer_config.json`; pin the revision and print the template hash; the committed `CHATML_V1` is independent of upstream.
- **`<|im_start|>`/`<|im_end|>` embeddings in the base model** may be under-trained; if the 0.6B base fails to learn the stop token within the budget, fallback is to initialize both embeddings from the mean embedding plus noise and to report it as part of the recipe (measure first; do not assume).
- **`flash-attn` for torch 2.15:** wheels may be unavailable; fallback `flex_attention`, then explicit-mask SDPA; the kernel table reports whichever run.
- **TRL behavior is version-dependent** (patched templates, default `max_length`, packing strategy names); the oracle test is pinned to the 7.7 versions and the finding is reported, not asserted in advance.
- **Token budget:** the 2M-per-variant default is a guess; the author measures tokens/s on the build box first and sizes the budget to ≤ 3 min per run.

### 7.3 Instruction data

- **Label:** `sec_lm_instruction_data` · **Tier:** T1 (native; scoring of the 80k-conversation pool is an offline T2 precompute whose outputs are committed) · **Budget:** ≤ 15 min: three ≤ 4 min training runs plus ≤ 2 min of CPU/GPU selection and evaluation; peak ≤ 15 GiB. The offline scoring job: ≤ 1 h on 1× A100 40 GB (the 2B scorer and the 0.6B embedder need under 10 GB of weights in bf16; to be measured), not part of the notebook budget. · **Codex units:** 8.3.
- **Builds on (book):** `sec_mdl-statistics` (bootstrap intervals, used through 1.5). No book section is re-taught here; selection, decontamination and licensing are new material.
- **Builds on (this part):**
  - 3.3 (near-duplicate removal): `d2l.near_dedup(texts, threshold=0.8, ngram=5, num_perm=128, seed=0) -> list[int]` (indices to keep), imported; the call appears once.
  - 3.5 (synthetic data: the Magpie and rephrasing recipes at pretraining scale; this section applies them to instructions) and 3.2 (quality filtering; this section's scorers are instruction-specific and implemented here).
  - 1.4 (`d2l.score_choices`: the judge's expected rating is read from digit-token log-probabilities), 1.5 (`d2l.EvalSuite`, `d2l.paired_bootstrap`, `d2l.judge_pairwise`), 2.3 (`Qwen/Qwen3-Embedding-0.6B` usage), 7.2 (`render_chat`, `pack_examples`, `sft_loss`, `chunked_lm_loss`).
- **Exports (`#@save`):**
  - `d2l.ngram_contamination(train_texts, eval_texts, n=8, normalize=True) -> list[dict]`: per training text, `{'max_overlap': float, 'eval_id': int}` where overlap is the fraction of its n-grams found in any eval text (hash set of 64-bit n-gram hashes; normalization lower-cases, strips punctuation and collapses whitespace).
  - `d2l.llm_rating(model, tokenizer, prompts, rubric, scale=(1, 5)) -> Tensor[len(prompts)]`: expected rating `Σ_k k·p_k` from the next-token distribution restricted to the digit tokens of `scale`.
  - `d2l.select_diverse(embeddings, k, scores=None, init=None) -> list[int]`: farthest-first traversal (cosine distance), optionally seeded from the highest-score point.
  - `d2l.budget_subset(lengths, order, budget_tokens, seed=0) -> list[int]`: take items in `order` until the sum of supervised-token lengths reaches the budget (the equal-token rule).
  - `d2l.length_matched_sample(pool_lengths, target_lengths, seed=0) -> list[int]`: random subset whose length histogram (log-spaced bins) matches the target's.
  - `d2l.magpie_sample(model, tokenizer, n, prefix='<|im_start|>user\n', max_new_tokens=256, temperature=1.0, top_p=1.0, seed=0) -> list[str]`: sample user instructions by letting an aligned model complete only the pre-query template.
  - Artifacts: `DATA_HUB['lm-ch07-instr-pool']` (JSONL of the 80k-conversation pool: `id`, `source`, `n_prompt_tokens`, `n_completion_tokens`, `quality`, `complexity`, `emb256` as fp16); `DATA_HUB['lm-ch07-decontam-report']` (per-eval-set overlap counts); `DATA_HUB['lm-ch07-magpie-qwen3-1p7b']` (3,000 instruction–response pairs with generation settings and model revision, as dataset metadata); `DATA_HUB['lm-sft-mix-v1']` (curated mixture: the A7 data of the Assignment contract; format given there); `DATA_HUB['lm-sft-qwen3-0p6b-v1']` (checkpoint retrained on `lm-sft-mix-v1`; same contract as `lm-sft-qwen3-0p6b`).
- **Models and data:**
  - Pool: `HuggingFaceTB/smol-smoltalk` train, 80,000 conversations sampled with seed 1, Apache-2.0 at the dataset level (verified 2026-10-28), `source` column kept.
  - Scorer: `Qwen/Qwen3.5-2B` (`15852e8c16360a2fea060d615a32b45270f8a8fc`; post-trained, non-thinking mode: generation prompt with the empty think block, as Qwen documents; see 7.2 on why we render it explicitly), Apache-2.0, loaded text-only. Embedder: `Qwen/Qwen3-Embedding-0.6B` (595.8M, Apache-2.0; the model card documents user-defined output dimensions 32–1024, so the 256-dimensional truncation used for storage is a documented feature, not a hack). Magpie generator: `Qwen/Qwen3-1.7B`, kept on purpose: it differs from the scorer, and it sits outside the judge's Qwen3.5 family, so the judge does not reward the Magpie arm for its own family's style.
  - Training base: `Qwen/Qwen3-0.6B-Base`.
  - Decontamination sets: `openai/gsm8k` test (MIT), `google/IFEval` (Apache-2.0), `TIGER-Lab/MMLU-Pro` test (MIT), `allenai/ai2_arc` (CC-BY-SA-4.0).
  - **License audit table** (`tab_lm_instr_licenses`, built live from `huggingface.co/api/datasets/<id>` card metadata): `HuggingFaceTB/smol-smoltalk` Apache-2.0; `allenai/tulu-3-sft-mixture` ODC-BY (its card states that different licenses apply to subsets and that some portions are non-commercial); `HuggingFaceH4/ultrachat_200k` MIT; `databricks/databricks-dolly-15k` CC-BY-SA-3.0; `OpenAssistant/oasst1` Apache-2.0; `HuggingFaceH4/no_robots` CC-BY-NC-4.0 (one of the Tülu 3 mixture's sources, per that card's `source_datasets`); `Magpie-Align/Magpie-Pro-300K-Filtered` `llama3`; `GAIR/lima` `other`, gated. All values verified 2026-10-28; the notebook re-reads them. The text states plainly that dataset-level license metadata is not legal advice and that the terms of the model that generated a response are a separate question.
- **Packages:** `datasets`, `transformers`, `numpy`; no extra dependency for selection (farthest-first is NumPy).

#### Narrative arc

1. **The question.** Opening cell loads the 80k-conversation pool and prints, for three subsets of equal supervised-token budget (random, longest-first, top quality score), the *number of examples* each contains and the distribution over `source`. Symptom: equal tokens, very different example counts and source mixes; "more data" is meaningless without a unit. A second print shows 5 random training texts whose 8-grams appear in GSM8K/IFEval/MMLU-Pro/ARC test sets (counts from the committed decontamination report; to be measured).
2. **Where instructions come from.** A taxonomy table `tab_lm_instr_sources`: human-written (Dolly-style, OASST, no_robots), templated (FLAN-style conversions), model-generated from seeds (Self-Instruct, Evol), and model-generated from nothing (Magpie). For each: who writes the prompt, who writes the response, typical failure modes (templated: low diversity; self-instruct: near-duplicates and invalid tasks; model-written responses inherit the teacher's errors and terms). Self-Instruct and Magpie as the two worked recipes.
3. **Magpie in a cell.** Feed `<|im_start|>user\n` to `Qwen3-1.7B`; sample 128 instructions live (seconds); print 10; then generate responses with the same model (non-thinking) for the committed 3,000 pairs; filtering rules (length bounds, repetition ratio, language, refusal strings); print before/after counts. Pointer to the Magpie paper's claim that aligned models "autocomplete" the user turn (cited, not re-measured at scale). Temperature and top-p are data-set metadata (stored with the artifact).
4. **Quality, difficulty, diversity.**
   - *Quality and complexity ratings:* `d2l.llm_rating` with two rubrics; show the rating distribution; calibration check against human-readable anchors (5 hand-written examples per level, in the notebook); position-in-scale bias (the model prefers "4"); check that the rating correlates with response length (Spearman, measured) — the length confound.
   - *Diversity:* embed first user turns; `select_diverse` (farthest-first). Derivation D2. Visualize with a 2-D random projection or PCA (no t-SNE dependency) of selected vs random points.
   - *Combined rule* (Deita-style, new reference): sort by `quality × complexity` descending, then farthest-first among the top 30% to reach the token budget. Name it `C`.
   - Gradient-based selection (LESS, new reference) is described in a half-page box and left to an exercise (it needs per-example gradients; not T1).
5. **Decontamination.** Algorithm (8-gram hash sets), the committed report, which fraction of the pool is flagged for each eval set and which `source`s carry it. Derivation D1: why rephrased contamination evades n-gram matching; the notebook rephrases 50 GSM8K test questions with Qwen3-1.7B, then compares the 8-gram detector with an embedding-similarity detector (cosine ≥ threshold chosen on a held-out set); measured hit rates are printed (Rethinking Benchmark and Contamination, new reference, reports that rephrased samples evade n-gram checks; we measure it on our pool).
6. **The headline comparison** (as below), including the length-matched control.
7. **Building the Assignment mixture.** `lm-sft-mix-v1` = `C`'s selection at the A7 budget with decontaminated, deduplicated, license-audited rows; `val` (2,000) and `prompt_pool` (≥ 20,000, disjoint) drawn from the remaining pool by seeded sample and re-screened against eval sets. The manifest hash is printed.
8. **Fine-tuning versus the data you have:** one paragraph connecting LIMA ("superficial alignment": 1,000 curated examples suffice for style, per the paper) to the 7.2 lesson that SFT teaches format; the notebook does *not* claim a conclusion beyond its own measurements.

#### Derivations

**D1. Detection of paraphrased contamination by n-gram matching (`eq_lm_ngram_detect`).**
Statement: a training text of `L` tokens that preserves each source token independently with probability `ρ` retains at least one n-gram of the source with probability `≈ 1 − exp(−(L−n+1)·ρ^n)`.
Assumptions: independent token preservation; n-gram occurrences approximately Poisson (ignores positive correlation between adjacent n-grams, so the approximation is rough for large `ρ`).
Steps:
1. A specific n-gram survives iff all `n` of its tokens are preserved: probability `ρ^n`.
2. There are `L−n+1` n-gram positions; the expected number of survivors is `μ = (L−n+1)ρ^n`.
3. Poisson approximation: `P(no survivor) ≈ e^{−μ}`.
4. Detection probability `1 − e^{−μ}`. For `L=60`, `n=8`: `ρ=0.9 → μ=22.8`, `ρ=0.7 → μ=3.05`, `ρ=0.5 → μ=0.21` (arithmetic; evaluated by the cell), so a rewrite that preserves half of the tokens is detected about 19% of the time, and the detector's recall falls off a cliff exactly where paraphrases live.
5. Smaller `n` raises recall but also false positives; the false-positive rate for common phrases is measured on the pool by flagging 5-grams in a clean reference set (the Chapter 1 *dev* split that is not used for reporting, 1.5).
Check cell: simulate token-dropout paraphrases of 1,000 GSM8K train questions at `ρ ∈ {0.9, 0.7, 0.5}`; compare the empirical detection rate with the formula (agreement to within a few percentage points at `ρ ≤ 0.7`; the notebook prints the gap and explains the correlation effect at high `ρ`).

**D2. Farthest-first traversal is a 2-approximation for k-center (`eq_lm_kcenter`).**
Statement: let `S_k` be the first `k` points chosen by farthest-first traversal in a metric space and `r_k = max_x d(x, S_k)` the covering radius; let `r*_k` be the optimal `k`-center radius. Then `r_k ≤ 2 r*_k`.
Steps:
1. Let `x_{k+1}` be the point attaining `r_k`; so `d(x_{k+1}, S_k) = r_k`.
2. By the traversal rule, each chosen point was at distance `≥ r_k` from all earlier points (the farthest distance is non-increasing in `k`), so the `k+1` points `S_k ∪ {x_{k+1}}` are pairwise at least `r_k` apart.
3. An optimal solution with `k` centers covers these `k+1` points; by pigeonhole two of them lie within distance `r*_k` of the same center.
4. Triangle inequality: those two points are at distance `≤ 2 r*_k`; hence `r_k ≤ 2 r*_k`. ∎
5. Consequence for data selection: farthest-first maximizes the *minimum* pairwise separation of the selected set (a coverage, not a density, criterion) and therefore over-selects outliers; this is why the notebook applies it only within the high-quality tier and reports the share of selected points that are in the lowest density decile.
Check cell: 12 random points in the plane, `k=3`; compute the optimal radius by exhaustive search (`C(12,3)=220` subsets) and the farthest-first radius; assert `r_ff ≤ 2 r*` for 200 random instances and report the largest observed ratio.

**D3. Expected rating from logits.** Statement: for rubric scale `{1..5}` the rating `R̂ = Σ_k k·p_k`, `p_k = softmax` restricted to the five digit tokens, is a proper expected value and has variance `Σ_k p_k (k − R̂)²` that can be used to drop ambiguous ratings. Steps: (1) restrict logits to the five digit tokens; (2) renormalize; (3) mean and variance; (4) report the share of the mass outside the digit tokens as a validity check (`> 5%` flagged). Check cell: on the 25 anchor examples the mass outside the digits and the Spearman correlation of `R̂` with the anchor level are printed.

#### From-scratch implementation

Budget ≈ 110 lines.
- `ngram_contamination` (25 lines): normalization; rolling hash of token n-grams (Python hash of tuple, seeded via `PYTHONHASHSEED`-independent `hashlib.blake2b(digest_size=8)` for reproducibility).
- `llm_rating` (20 lines): build rubric prompt with `render_chat`, `add_generation_prompt=True`, nonthinking prefix; forward; read logits at the last position at the digit token ids; softmax; expected value; batched.
- `select_diverse` (20 lines): farthest-first with a running min-distance vector; `O(nk)`; cosine on normalized embeddings.
- `budget_subset`, `length_matched_sample` (25 lines).
- `magpie_sample` (20 lines): `model.generate` on the prefix with `stop` at `<|im_end|>`.
- **Tests.** (1) contamination: a planted 8-gram in 3 of 1,000 texts is found, with zero false positives on a shuffled-token control; (2) `llm_rating` outputs lie in `[1,5]`, and are invariant to batch size to 1e-3; (3) D2 check; (4) `budget_subset` returns a total within one item of the budget; (5) `length_matched_sample` histogram L1 distance to the target `< 0.05`; (6) end-to-end determinism: the same seed gives byte-identical selection manifests (hash equality).

#### In production

- **Libraries:** `datasets` (map/filter with fingerprinting), `datatrove` or `text-dedup` for scale-out dedup (pointers to 3.3), `lm-evaluation-harness` decontamination utilities (n-gram based; the notebook compares their flags with `ngram_contamination` on the same eval sets as an oracle test: identical sets of flagged items at the same `n`, differences explained by normalization).
- **Judge-based selection in practice:** Tülu 3 and SmolTalk rely on filtering plus LLM classification; the section's scorer is a small stand-in whose agreement with a larger judge (the part's judge, `Qwen/Qwen3.5-9B` in bf16, 500 items, optional) is reported as a Spearman correlation.
- **Data cards as metadata:** the manifest records dataset revision, license, generator model/revision, sampling parameters (for Magpie), decontamination `n` and thresholds.

#### Headline experiment

- **Hypothesis.** At an equal supervised-token budget, SFT on a curated subset (`C`: quality × complexity, farthest-first diversity) beats a random subset (`R`) on instruction following, and part of the gap is explained by length (a length-matched random subset `R_len` closes some of it).
- **Setup.** `Qwen3-0.6B-Base`; budget `B` = 3M supervised tokens (to be sized so a run takes ≤ 4 min; the author measures tokens/s first); full FT, `lr` 1e-5, cosine, packed rows of 2,048 tokens with block mask (7.2), identical optimizer steps; 3 seeds per arm if time allows, else 1 seed plus 3 evaluation resamples (and says so). Arms: `R`, `R_len`, `C`; optional `M` (Magpie-Qwen3-1.7B subset at the same budget) behind `with_magpie=True`.
- **Procedure.** Select, decontaminate (all arms), train, evaluate on IFEval (541 prompts; prompt-level strict accuracy), GSM8K test (1,319 items, flexible numeric extraction as in 1.4), and 200 pairwise preferences against arm `R` judged by `d2l.judge_pairwise` (judge: the part's `Qwen/Qwen3.5-9B` in bf16, loaded after the trained model is freed; position-consistent preference only; judge agreement between orderings is reported as its own metric; because the scorer and the judge are both Qwen3.5 models, conclusions about the selection arms rest on IFEval and GSM8K, with the pairwise rate as supporting evidence).
- **Metrics.** IFEval prompt-level strict accuracy; GSM8K accuracy; judge win rate with tie handling; mean response length; number of training examples; training-set decontamination flag rate (must be 0 for the arms).
- **Plot spec.** `fig_lm_instr_arms`: x = arm; panel 1 IFEval accuracy with 95% bootstrap interval (items) and seed dots; panel 2 judge win rate vs `R` with interval; panel 3 mean response length and example count (twin axis).
- **Expected outcome.** Directions only: `C` ≥ `R` on IFEval; `R_len` between `R` and `C` on any metric correlated with length; GSM8K differences within noise (the pool contains little math). The Deita and LIMA papers report that small selected sets can match larger ones; the notebook states its own measured gaps with intervals.
- **Noise check.** Seeds (`≥ 2` if possible), item-level bootstrap from 1.5, paired by prompt across arms; the notebook refuses to rank arms whose interval contains 0, and prints "within noise".
- **T1 reduction.** Native T1; `B` is the lever.
- **Artifacts committed.** `lm-ch07-instr-pool`, `lm-ch07-decontam-report`, `lm-ch07-magpie-qwen3-1p7b`, `lm-sft-mix-v1`, `lm-sft-qwen3-0p6b-v1`, `arms.jsonl` (per-item outputs and scores).

#### Figures and tables

- `tab_lm_instr_sources` (taxonomy), `tab_lm_instr_licenses` (live license audit).
- `fig_lm_instr_rating`: rating distributions for quality and complexity, and their relation to response length.
- `fig_lm_instr_select`: 2-D projection of random vs farthest-first points.
- `fig_lm_ngram_detect`: detection probability vs token preservation `ρ`, simulation vs `eq_lm_ngram_detect`.
- `tab_lm_instr_contam`: flagged counts by eval set and by source.
- `fig_lm_instr_arms`.

#### Pitfalls and misconceptions

- "Equal data" without a unit: equalize supervised tokens and report example counts.
- Quality scores that are mostly a proxy for length; always include the length-matched control.
- Decontaminating the *training prompts* only; also the responses, and also the held-out `val`/`prompt_pool`.
- N-gram decontamination gives false comfort against paraphrased leakage (D1).
- Selecting on test-like data: do not use the retention/test sets to tune thresholds.
- Treating a dataset-level license as covering every row; model-generated rows carry their generator's terms.
- Magpie and self-distillation transfer the teacher's style and errors, not new knowledge.

#### Summary (draft bullets)

- Instruction data is defined by who writes the prompt, who writes the response, and under what license.
- Compare data at equal supervised tokens, and control for length.
- Selection by quality, complexity and diversity is cheap with a small judge and an embedder; coverage by farthest-first has a 2-approximation guarantee.
- Decontamination by n-grams catches copies, not paraphrases.
- The result of this section is the mixture and checkpoint that Chapter 8 starts from.

#### Exercises

1. *Check.* Two subsets each hold 3M supervised tokens: one has 6,000 conversations, the other 12,000. Name two confounds besides "quality" that could explain a difference in IFEval score. *Sketch:* answer length, source mix (format-following data share), and number of optimizer steps if batches are by examples.
2. *Check.* An 8-gram decontamination pass flags 0 training examples against GSM8K test. Does that show the model never saw GSM8K-like questions? *Sketch:* no; paraphrases, translations and different number substitutions evade it (D1); also pretraining data is untouched by this pass.
3. *Derive.* Extend D1 to account for a rewrite that deletes each token independently with probability `1−ρ` and also inserts random tokens at rate `q` between tokens. *Sketch:* an n-gram survives iff its `n` tokens are kept and none of the `n−1` gaps receives an insertion: `ρ^n (1−q)^{n−1}`.
4. *Derive.* For farthest-first traversal show `r_k` is non-increasing in `k` and that the selected set's minimum pairwise distance is `≥ r_{k−1}`. *Sketch:* distance to a growing set can only shrink; each new point attained the previous radius.
5. *Measure.* Predict the Spearman correlation between the quality rating and response length on the pool, then measure; repeat with the complexity rating. What does it imply for `C` vs `R_len`?
6. *Measure.* Change the diversity step: drop farthest-first and take the top-score items only; predict the change in source mix and IFEval, then measure with intervals.
7. *Open.* Build a 2,000-example domain instruction set for the book (R3) with Magpie-style prompts conditioned on book passages; decontaminate against `d2l.book_qa`'s dev split and measure leakage.

#### Further reading

- **Papers:** [Self-Instruct](https://arxiv.org/abs/2212.10560); [LIMA](https://arxiv.org/abs/2305.11206); [Magpie](https://arxiv.org/abs/2406.08464); [Tülu 3](https://arxiv.org/abs/2411.15124) (SFT mixture construction and decontamination); [Deita](https://arxiv.org/abs/2312.15685) (new, verified; complexity, quality, diversity selection); [LESS](https://arxiv.org/abs/2402.04333) (new, verified; gradient-based targeted selection); [Rethinking Benchmark and Contamination](https://arxiv.org/abs/2311.04850) (new, verified; rephrased samples evade n-gram detection).
- **Datasets:** [Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture); [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk).
- **Code:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (decontamination utilities; verified 200); [Alignment Handbook](https://github.com/huggingface/alignment-handbook).

#### Risks and open decisions

- **Judge quality:** a 9B judge is still noisy for pairwise preference; mitigated by two-order consistency and by reporting agreement; fallback is IFEval and GSM8K only, with the judge as a cached T2 job.
- **Pool scoring cost:** 80k conversations through the 2B scorer and a 0.6B embedder is a T2 job on one A100 40 GB (to be measured); fallback pool of 30k with a larger relative budget.
- **Licenses:** smol-smoltalk's rows are model-generated (per its card, derived from SmolTalk components); the audit table states the dataset-level license and the caveat. If the editor wants a stricter commercial-use stance, switch the pool to `HuggingFaceH4/ultrachat_200k` (MIT at dataset level, model-generated rows).
- **Magpie arm:** outputs of `Qwen/Qwen3-1.7B` (Apache-2.0 model) are used with its terms; stored with model revision.
- **Source-mix confound:** `C` may concentrate in one source; the notebook prints the mix and offers a source-stratified variant as an exercise.

### 7.4 LoRA from scratch

- **Label:** `sec_lm_lora` · **Tier:** T1 (native; the full rank × target × learning-rate grid is a committed artifact and the notebook re-runs a slice) · **Budget:** ≤ 15 min (19 short runs of ≈ 0.5–1 min: the 14 of the main slice plus the 5 new runs of the fixed-learning-rate slice; the author sizes the token slice so the average run is ≤ 45 s, measured first), peak ≤ 12 GiB on `Qwen3-0.6B-Base` · **Codex units:** 8.5 (LoRA computation, gradients, initialization, merging) and 8.6 (rank, targets, scaling, learning-rate interaction).
- **Builds on (book):**
  - `sec_mdl-svd-low-rank` (`sec_mdl-svd-modern-dl`, `eq_mdl-lora-ratio`, `fig_mdl-la-lora`): the picture `h = Wx + BAx`, the parameter ratio `r(m+n)/mn`, the Eckart–Young ceiling, PiSSA in one sentence. Recap in one paragraph; no re-derivation.
  - `sec_parameters` (`subsec_freezing`): freezing is the first line of `add_lora`.
  - `sec_custom_layer`: a parameterized `nn.Module` whose state is a small number of registered parameters.
  - `sec_init_param`: variance of `Ax` for Kaiming-uniform `A`; the recap is one paragraph.
  - `sec_adamw`: Adam's per-element normalization is the premise of D1's Adam case.
- **Builds on (this part):** 7.1 (`d2l.ft_memory` with `method='lora'`, `d2l.layerwise_lr_groups`, `d2l.chunked_lm_loss`), 7.2 (`render_chat`, `pack_examples`, `sft_loss`, `lm-sft-mix-v0`), 1.5 (`d2l.paired_bootstrap`), 1.2 (R1 parity target).
- **Exports (`#@save`, reserved name first):**
  - `d2l.LoRALinear(base, r=8, alpha=16.0, dropout=0.0, rslora=False, adapter_dtype=torch.float32)`; `nn.Module`; `base` is a frozen `nn.Linear` (or `d2l.QuantLinear` from 7.6).
    - Parameters: `lora_A` of shape `[r, in_features]`, initialized `U(−1/√in, 1/√in)` (Kaiming-uniform with `a=√5`, the PEFT default); `lora_B` of shape `[out_features, r]`, zeros. Both `adapter_dtype` (default fp32), `requires_grad=True`; the base weight and bias have `requires_grad=False`.
    - Interface aliases required by Chapters 8 and 12: `A` and `B` are properties returning `lora_A` and `lora_B`; `scale` is a property returning `scaling`; `base` and the boolean `enabled` (default `True`; `False` means the base output, so Chapter 8's reference log-probabilities need no second model copy) are plain attributes; `merge()` and `unmerge()` as below.
    - `scaling` property: `alpha / r`, or `alpha / sqrt(r)` if `rslora`. The default `rslora=False` is PEFT's default (`use_rslora=False`), so adapters, results and the PEFT oracle (test 4) agree without arguments; the grid below reports both multipliers.
    - `forward(x)`: `y = base(x) + scaling · dropout(x.to(A.dtype)) @ A.T @ B.T` cast to `y`'s dtype. If `enabled` is `False` or the module is merged, returns `base(x)`.
    - `delta_weight() -> Tensor[out, in]`: `scaling · B @ A` in fp32.
    - `merge(keep_original=False)`: for a floating `base`, `W ← (W.float() + delta_weight()).to(W.dtype)`; sets `merged=True`. With `keep_original=True` it stores a copy of `W` so that `unmerge()` is bit-exact. For a quantized base it raises `NotImplementedError` and points to `merged_weight()`.
    - `unmerge()`: if a copy exists restore it, else `W ← (W.float() − delta_weight()).to(W.dtype)`; sets `merged=False`.
    - `merged_weight(dtype=torch.bfloat16)`: dequantized (if needed) base plus delta, without modifying the module (the export path for QLoRA adapters, 7.6).
    - Guarantees tested in the notebook: B=0 at init so `forward` equals `base(x)` exactly; fp32 base: merged vs unmerged output max relative error ≤ 1e-5 of `‖y‖∞`; `unmerge(merge(W))` max abs error ≤ 4 ulp(fp32) of `W` (bit-exact with `keep_original`); bf16 base: merged vs unmerged output relative error ≤ 3e-2 (rounding of merged weights, `eq_lm_bf16_absorb`; tolerance to be calibrated on the first run).
  - `d2l.apply_lora = d2l.add_lora` (the name Chapter 8 uses; both names are exported and identical).
  - `d2l.add_lora(model, target_modules='all', r=8, alpha=16.0, dropout=0.0, rslora=False, exclude=('lm_head',), adapter_dtype=torch.float32) -> list[str]`: replaces matching `nn.Linear`/`QuantLinear` modules by `LoRALinear` in place, freezes every other parameter, and returns the replaced module names. `target_modules` is `'attn'`, `'mlp'`, `'all'`, or an iterable of name suffixes. Alias table `d2l.LORA_TARGETS`: HF Qwen3 `q_proj k_proj v_proj o_proj | gate_proj up_proj down_proj`; R1 `d2l.GPT`/`GQAAttention`/`FeedForward` `W_q W_k W_v W_o W_qkv | W_g W_1 W_2`.
  - `d2l.lora_param_groups(model, lr, lr_ratio=1.0, weight_decay=0.0) -> list[dict]`: `lora_A` at `lr`, `lora_B` at `lr_ratio·lr` (LoRA+); no decay on adapters by default.
  - `d2l.merge_lora(model, keep_original=False)`, `d2l.unmerge_lora(model)`.
  - `d2l.lora_state_dict(model) -> dict[str, Tensor]`: only `lora_A`/`lora_B` (keys `<module path>.lora_A`, `.lora_B`) plus a `__meta__` entry `{'r', 'alpha', 'rslora', 'targets', 'base': <model id and revision>}`; `d2l.load_lora_state_dict(model, sd, strict=True)`.
  - `d2l.lora_disabled(model)`: context manager that sets `enabled=False` on every `LoRALinear` and restores it on exit; the frozen reference policy for Chapter 8.
  - Artifact `DATA_HUB['lm-lora-r1-demo']` (for Chapter 13's multi-LoRA serving): a small adapter for R1 (`Qwen/Qwen3-0.6B`, the model `d2l.load_qwen3` loads): `r=8`, `alpha=16`, all-linear (≈ 5.0M parameters, ≈ 10 MB in bf16 safetensors), trained for 150 steps on `lm-sft-mix-v0`; stored in the `d2l.lora_state_dict` format (`.safetensors` plus `meta.json` with base id/revision, `r`, `alpha`, targets), loadable with `d2l.load_lora_state_dict` and by PEFT.
  - Artifact: `DATA_HUB['lm-ch07-lora-sweeps']`: JSONL of the full grid (rank, target, scaling rule, learning rate, `lr_ratio`, seed, final validation loss, per-100-step training loss), with the fixed-learning-rate slice rows tagged `slice='fixed_lr'`.
- **Models and data:** `Qwen/Qwen3-0.6B-Base` (T1 workhorse; 28 layers, `hidden_size` 1024, 16 heads / 8 KV heads, `head_dim` 128, `intermediate_size` 3072); `lm-sft-mix-v0` from 7.2 (smol-smoltalk subset, Apache-2.0) with a 1.5M-supervised-token training slice and the 2,000-conversation validation split; R1 (`d2l.load_qwen3('Qwen/Qwen3-0.6B')`) for one parity test. Parameter counts from the config (arithmetic, printed by the notebook): attention-only `r=16` ≈ 4.59M, all-linear `r=16` ≈ 10.09M, `r=64` ≈ 40.37M, against 596M total.
- **Packages:** `peft` (oracle), `transformers`, `torch`; versions in 7.7.

#### Narrative arc

1. **The question.** Opening cell: take `d2l.load_hf('Qwen/Qwen3-0.6B-Base')`, call `add_lora(model, 'all', r=16)`, print trainable parameters (10.09M of 606M, 1.7%), then run **one** training step at the *same* learning rate for `r ∈ {1, 4, 16, 64}` with `scaling = α/r` and print the change of the logits (relative to the base) after the step. Symptom: under `α/r` the one-step logit change is nearly independent of rank; under `scaling = 1` it grows with `r`; under `α/√r` it also grows (Adam; see D1). A reader who remembers rsLoRA expects the opposite for `α/√r`; that is the question the section answers.
2. **The layer.** Recap (one paragraph, `:numref:sec_mdl-svd-low-rank`): `W' = W + (α/r)BA`. Then the cell-by-cell construction of `LoRALinear`: parameters and init (why `B=0`: the adapted model starts exactly at the base; why `A` random: with `B=0` and `A=0` both gradients vanish), forward in three lines, `delta_weight`, `merge/unmerge`. Gradients derived on the page: `∂L/∂B = γ·g (Ax)ᵀ`, `∂L/∂A = γ·Bᵀ g xᵀ` (so `A` receives no gradient at step 0), checked against autograd on a 4×3 example.
3. **Where to attach adapters.** `d2l.add_lora` and the alias table; the table of per-module parameter counts for Qwen3-0.6B (`q: r(1024+2048)`, `k,v: r(1024+1024)`, `o: r(2048+1024)`, `gate,up: r(1024+3072)`, `down: r(3072+1024)`); targets as a *capacity* decision. Reports from LoRA Without Regret (applying LoRA to all weight matrices, MLP especially, outperforms attention-only even at matched parameter count) and from QLoRA (all layers) are stated as hypotheses H3 and tested.
4. **Initialization and scale.**
   - `A`: Kaiming-uniform (variance `1/(3·in)`), Gaussian alternative; `B`: zero. The scale-invariance remark: scaling `A` by `c` and `B`'s learning rate by `1/c` leaves the function path unchanged under Adam/SGD up to reparameterization (the "init scales and multipliers invariances" of LoRA Without Regret; the notebook verifies one case numerically).
   - D1: why the multiplier is `α/r` (Adam) and `α/√r` (SGD); the one-step experiment `fig_lm_lora_onestep`; a measured early-training curve comparison across ranks.
   - The rsLoRA test at a fixed learning rate. rsLoRA's claim (arXiv 2312.03732) is about one learning rate shared by all ranks: with `α/r` the gain from a higher rank stalls, with `α/√r` it does not. A per-rank tuned learning rate (the main grid) can hide that, because the optimum moves with `r`. The notebook therefore runs a small slice at one fixed learning rate (the rank-4 optimum under `α/r`), ranks {1, 8, 64} × both multipliers, one seed (`fig_lm_lora_fixedlr`); the default stays `α/r` (PEFT's `use_rslora=False`) and the grid reports both.
   - D2: LoRA+ and the two learning rates.
5. **Learning rates relative to full fine-tuning.** LoRA Without Regret reports that the optimal LoRA learning rate is about 10× the full-FT optimum, nearly rank-independent under `α/r` (optimal LR changing by less than 2× from rank 4 to 512, per the post), and that LoRA is less tolerant of large batch sizes than full FT. The notebook tests: LR sweep for full FT and for LoRA (`r=16`, all-linear), plot the U-shaped curves `fig_lm_lora_lr`; then a two-point batch-size check (8k vs 32k tokens per step; to be measured).
6. **Merging, unmerging and the bf16 trap.** `merge`/`unmerge` unit tests; the surprising number: for a bf16 base, how much of `ΔW` survives merging (D3). The notebook prints `‖ΔW_bf16 − ΔW‖/‖ΔW‖` per module type and the logit KL between the adapter model and its merged bf16 version; the merged model is *evaluated separately* from the adapter model in every table from here on.
7. **Memory accounting.** `d2l.ft_memory(..., method='lora', lora_r=16, lora_targets='all')` against measured peak for batch sizes 1, 2, 4 at seq 1,024 (calibrates the LoRA activation term; adapters at fp32 cost `4+4+4+4 = 16` bytes per adapter parameter; the frozen base costs 2 bytes per parameter and, because weight gradients are not needed, its layers do not save their inputs). Table `tab_lm_lora_memory`: full FT vs LoRA (attn / all, r = 8 / 64).
8. **Rank sweeps and capacity** (headline experiment below). One paragraph on what capacity means: LoRA with `r=1` already has millions of parameters; LoRA Without Regret's information-theoretic count of bits per training token is cited as the argument for why RL needs little capacity (8.5 returns to it); LoRA Learns Less and Forgets Less (10–100× higher rank of full-FT updates) is the counter-pointer for large-data regimes.
9. **Adapters as artifacts.** `lora_state_dict` size on disk vs the base; loading into PEFT's `PeftModel` (names mapping) as the interoperability test.

#### Derivations

**D1. Rank scaling under SGD and under Adam (`eq_lm_lora_scale`).**
Statement: with `B=0`, `A` having iid entries of variance `σ²` independent of `r`, multiplier `γ`, and inputs `x, x'` of `A`: after one optimizer step on a single example `(x, g)` (`g = ∂L/∂y`), the change of the adapter output at input `x'` has magnitude
`‖Δy‖ = η γ² r σ² |⟨x,x'⟩| ‖g‖` for SGD with learning rate `η`, and `‖Δy‖ ≈ η γ r c √d_out` for Adam, whose first step is `ΔB = −η·sign(∂L/∂B)` (here `c = E[sign(u)·u']` with `u = a·x`, `u' = a·x'`, a constant that does not depend on `r`).
Hence the rank-independent multiplier is `γ ∝ 1/√r` for SGD (rsLoRA's result) and `γ ∝ 1/r` for Adam (the standard `α/r`).
Assumptions: `B=0` at the step; single step; random `A` independent of `x`; sums over `r` rank-one terms concentrate (law of large numbers in `r`).
Steps:
1. Output `y_ad = γ B A x`; with `B=0`, `∂L/∂B = γ g (Ax)ᵀ`; `∂L/∂A = γ Bᵀ g xᵀ = 0`.
2. SGD: `B₁ = −η γ g (Ax)ᵀ`. Then `Δy(x') = γ B₁ A x' = −η γ² g · (Ax)ᵀ(Ax')`.
3. `(Ax)ᵀ(Ax') = Σ_{i=1}^r (a_i·x)(a_i·x')`, a sum of `r` iid terms with mean `σ² ⟨x,x'⟩`, so it concentrates at `r σ² ⟨x,x'⟩`. Hence `‖Δy‖ = η γ² r σ² |⟨x,x'⟩| ‖g‖`.
4. Rank independence requires `γ² r = Θ(1)`, i.e. `γ = Θ(1/√r)`. With `γ = α/r` the change falls as `1/r`; with `γ = 1` it grows as `r`.
5. Adam: the first update is `B₁ = −η·sign(γ g (Ax)ᵀ) = −η·sign(g_j)·sign((Ax)_k)` (independent of `γ`'s magnitude for `γ>0`). Then `Δy_j(x') = −η γ sign(g_j) Σ_k sign((Ax)_k)(Ax')_k`. The sum has `r` terms of expectation `E[sign(u)u']`, a constant times `σ ρ ‖x'‖` for correlated Gaussian projections; hence `Δy ∝ η γ r`.
6. Rank independence for Adam requires `γ r = Θ(1)`: `γ = α/r`. This is the "1/r prefactor makes the optimal LR approximately independent of rank, and the learning curves coincide at the start" observation of LoRA Without Regret.
7. Scope. The statement is about the first step. After the first steps `A` is also updated and `B` is no longer zero; rsLoRA's paper proves its `1/√r` claim for the stable-learning condition over training under SGD-type updates and reports improved high-rank fine-tuning empirically; the experiment below decides which multiplier produces rank-stable optimal learning rates for AdamW on our data. The notebook states this tension openly and tests it at a fixed learning rate in the headline experiment (H5).
Check cell (CPU, seconds, numerics already verified by the blueprint author in NumPy, `d_in=1024`, `d_out=512`, `r ∈ {1,4,16,64,256}`, 20 draws): the one-step `‖Δy‖` ratio between `r=256` and `r=1` is ≈ 200 for `γ=1` (both optimizers), ≈ 0 (collapses) for `α/r` under SGD and ≈ 0.8 under Adam, and ≈ 0.8 for `α/√r` under SGD but ≈ 17 under Adam. The notebook prints its own run; to be re-measured with the final seeds, expected pattern as stated.

**D2. LoRA+: the A-induced and B-induced feature updates (`eq_lm_loraplus`).**
Statement: with `B=0` initially, two SGD steps on the same example `(x, g)` and learning rates `η_A, η_B`, the output change at `x'` splits exactly into `c_B = −η_B γ² ⟨Ax, Ax'⟩ g` (from `B`) and `c_A = −η_A η_B² γ⁴ ‖Ax‖² ‖g‖² ⟨x,x'⟩ g` (from `A`), so
`‖c_A‖/‖c_B‖ = η_A η_B γ² ‖g‖² ⟨x,x'⟩ ‖Ax‖² / ⟨Ax,Ax'⟩ ≈ η_A η_B γ² ‖g‖² ‖x‖²` for `x' ≈ x`.
Steps:
1. Step 1: `B₁ = −η_B γ g (Ax)ᵀ` (`A` unchanged because `B₀=0`); `c_B` is `γ B₁ A x'`.
2. Step 2 gradient on `A`: `γ B₁ᵀ g xᵀ`; so `ΔA = −η_A γ B₁ᵀ g xᵀ`.
3. A-induced output change: `γ B₁ ΔA x' = −η_A γ² B₁B₁ᵀ g ⟨x,x'⟩`, and `B₁B₁ᵀ = η_B² γ² ‖Ax‖² g gᵀ`.
4. Substitute and divide.
5. Interpretation: the two contributions are balanced when `η_A η_B ≈ 1/(γ² ‖g‖² ‖x‖²)`. With activations `x` of width `n` (so `‖x‖² = Θ(n)`) and `η_B` fixed by requiring a Θ(1) B-induced change, `η_A = Θ(1/n)`: the ratio `η_B/η_A = Θ(n)`. This is the LoRA+ scaling argument (large-width, SGD-type); the paper's practical recommendation is a fixed ratio (powers of two up to about 16) for Adam. The notebook sweeps `lr_ratio ∈ {1, 4, 16}` and reports; the paper reports improvements of 1–2% and up to ≈ 2× faster fine-tuning (abstract), which the section treats as a hypothesis (H4).
Check cell: random `A, x, x', g` in NumPy; compute `c_A`, `c_B` by explicit two-step SGD and compare with the closed forms (`np.allclose`) and the ratio formula (verified by the blueprint author; relative difference 0).

**D3. Merging into a lower-precision weight (`eq_lm_merge_err`).**
Statement: merging `ΔW = γBA` into a bf16 weight `W` stores `fl_bf16(W+ΔW)`; the effective merged delta is `fl(W+ΔW) − W`, which equals `ΔW` only up to a rounding of size `½ulp(W)`; elements of `ΔW` below `½ulp(W_ij)` are lost in expectation-free fashion (round-to-nearest).
Steps: (1) from `eq_lm_bf16_absorb` an element of `ΔW` is lost if `|ΔW_ij| < 2^{e−8}` where `2^e` is the binade of `W_ij`; (2) for LoRA after short training `|ΔW_ij|` is of order `lr·steps·(…)` ≪ `|W_ij|/256`, so most of the delta rounds to 0 or to one ulp; (3) the relative error `‖Δ_eff − ΔW‖_F/‖ΔW‖_F` can be of order 1; (4) consequences: merge in fp32 and cast once, or serve unmerged; (5) the quantized case (7.6): dequantize → add → requantize is lossy at the quantization error of the base, not only the rounding of bf16.
Check cell: for each module type compute `‖Δ_eff − ΔW‖_F/‖ΔW‖_F` for the trained adapter, the share of `ΔW` elements below `½ulp`, and the logit KL of merged-bf16 vs adapter model on 200 validation prompts (to be measured).

#### From-scratch implementation

Budget ≈ 140 lines (≤ 150).
- `LoRALinear` (65 lines) with the contract above, including `merged_weight`.
- `add_lora` + `LORA_TARGETS` (30 lines): suffix matching on `named_modules()`, parent lookup via `rsplit('.', 1)`, `setattr`.
- `lora_param_groups`, `merge_lora`, `unmerge_lora`, `lora_state_dict`, `load_lora_state_dict`, `lora_disabled` (45 lines).
- **Numerical tests.**
  1. `B=0` ⇒ `LoRALinear(x) == base(x)` bitwise (fp32, seed 0, `x ∈ ℝ^{4×1024}`).
  2. Gradient formulas vs autograd, 4×3 example, `atol=1e-6`.
  3. fp32 merged vs unmerged: relative error ≤ 1e-5 of `‖y‖∞` after setting `B` to random values; `unmerge(merge(W))` max abs error ≤ 4 ulp; bit-exact with `keep_original=True`.
  4. PEFT oracle: copy `A, B` into `peft.LoraConfig(r=16, lora_alpha=32, target_modules='all-linear', use_rslora={False, True}, lora_dropout=0)` and compare logits on 3 prompts (fp32, 1e-5 relative); with `rslora=True` this checks the `α/√r` convention.
  5. R1 parity: `d2l.load_qwen3('Qwen/Qwen3-0.6B')` with the same adapter weights gives logits equal to the HF-based model within bf16 tolerance (to be calibrated; argmax agreement ≥ 99.9%).
  6. `lora_disabled(model)` reproduces the base logits exactly; trainable parameter count equals `r·(m+n)` summed over targets; base `.grad is None` after backward.
  7. `lora_state_dict` round trip.

#### In production

- **PEFT:** `LoraConfig(r, lora_alpha, lora_dropout, target_modules='all-linear', use_rslora, init_lora_weights=True, bias='none')`, `get_peft_model`, `PeftModel.from_pretrained`, `merge_and_unload()`; `peft.optimizers.create_loraplus_optimizer` for LoRA+ (the notebook confirms the name in the pinned release). The from-scratch layer is the oracle (tests 4–5); each PEFT feature is introduced with the line of `LoRALinear` it corresponds to.
- **TRL:** `SFTTrainer(peft_config=…)` for the toolchain comparison (7.7).
- **Adapter serving:** multi-adapter serving (vLLM LoRA support) is a pointer to 13.5; this section only fixes the file format.

#### Headline experiment

- **Hypothesis.** (H1) With `γ = α/r` and AdamW, the best learning rate varies by less than 2× across `r ∈ {1, 8, 64}` and the first-50-step loss curves nearly coincide; (H2) with `γ = α/√r` the best learning rate decreases with `r`; (H3) all-linear LoRA beats attention-only at *matched trainable parameters*; (H4) `lr_ratio=16` helps or is neutral at `r=16`; and the best LoRA learning rate is roughly 10× full fine-tuning's; (H5) at one fixed learning rate (the rank-4 optimum under `α/r`), validation loss under `α/r` stops improving as the rank grows from 1 to 64, while under `α/√r` it keeps improving over at least part of that range, as rsLoRA reports. D1 adds a caveat: under Adam the step-one output change under `α/√r` grows with `r`, so at rank 64 the shared learning rate may overshoot; the direction is therefore measured, not assumed.
- **Setup.** `Qwen3-0.6B-Base`, `lm-sft-mix-v0` slice of 1.5M supervised tokens, rows of 1,024 tokens, 8 rows per step (≈ 8k tokens, ≈ 180 steps; to be set by measured throughput), 3% warmup, linear decay, fp32 adapters, bf16 base, dropout 0, `α=32` (as LoRA Without Regret), seeds {0,1,2} for adapter init (data order fixed). Grid (full run, committed): rank {1, 4, 16, 64} × target {attn, mlp, all} × scaling {`α/r`, `α/√r`} × learning rate `{3e-5·3^k, k=0..5}`; plus full FT at `{3e-6·3^k, k=0..3}`. T1 slice (live): rank {4, 64} × target {attn, all} × scaling {`α/r`} × 3 learning rates, plus full FT at 2 learning rates and `lr_ratio ∈ {1, 16}` at `r=16`. Fixed-learning-rate slice (live, added for H5): all-linear, rank {1, 8, 64} × scaling {`α/r`, `α/√r`} at `lr★`, the learning rate with the lowest validation loss at `r=4`, `α/r` (read from the main slice's three learning rates, chosen to bracket the grid optimum), one seed; the `r=64`, `α/r` run is shared with the main slice, so 5 runs are new. No `r=256` run. The committed grid already holds ranks {1, 4, 16, 64} × both multipliers at six learning rates, so the same comparison is read off at every grid learning rate as a robustness check at no extra cost.
- **Metrics.** Validation loss (token-mean, assistant tokens); per-10-step training loss; peak GiB; tokens/s.
- **Plot spec.** `fig_lm_lora_lr`: x = learning rate (log), y = validation loss; one curve per rank, panels for `α/r` and `α/√r`, full-FT curve in black, vertical ticks at each curve's argmin. `fig_lm_lora_early`: x = step (first 50), y = training loss, one line per rank. `fig_lm_lora_targets`: x = trainable parameters (log), y = best validation loss over learning rates, one line per target set. `fig_lm_lora_fixedlr`: x = rank (log₂, ticks 1, 8, 64), y = validation loss at `lr★`; one line per multiplier (`α/r`, `α/√r`), with the `r=4`, `α/r` three-seed mean ± 2 standard deviations as a grey band for scale; a second panel repeats it for each learning rate of the committed grid (small multiples). `tab_lm_lora_memory`.
- **Expected outcome.** Direction/magnitude from sources only: LoRA Without Regret reports learning curves identical early in training across ranks for `α/r`, the optimal LR for LoRA ≈ 10× full FT's, attention-only underperforming even at matched parameters; rsLoRA reports (at one fixed learning rate) that `α/r` stalls high ranks and `α/√r` fixes it; LoRA+ reports 1–2% gains. Our measurements either support or contradict each; the notebook prints "agrees"/"disagrees" with intervals and does not smooth over disagreement.
- **Noise check.** 3 adapter-init seeds on every configuration of the T1 slice; difference between two configurations is reported only if it exceeds twice the across-seed standard deviation of the validation loss; the argmin learning rate is reported with the grid spacing (factor 3) as its resolution, so a "2×" claim is not made from a grid with ×3 spacing (the notebook uses a finer ×1.5 grid around the minimum for the H1 test). The fixed-learning-rate slice has one seed per point, so H5 is judged against the three-seed spread measured at `r=4`, `α/r` in the main slice: a difference between two ranks or two multipliers counts only if it exceeds twice that standard deviation, and the committed grid's three seeds per point back the reading.
- **T1 reduction.** Native T1 slice; full grid precomputed (`lm-ch07-lora-sweeps`).
- **Artifacts committed.** `lm-ch07-lora-sweeps`, figures, `peak_memory.json`.

#### Figures and tables

- `fig_lm_lora_onestep`: one-step output change vs rank for `γ ∈ {1, α/r, α/√r}` under SGD and Adam (cell: D1 check).
- `fig_lm_lora_lr`, `fig_lm_lora_early`, `fig_lm_lora_targets`, `fig_lm_lora_fixedlr` (above).
- `tab_lm_lora_params`: per-module adapter parameter counts for Qwen3-0.6B, 1.7B and 8B (from configs).
- `tab_lm_lora_memory`: predicted vs measured peak memory.
- `fig_lm_lora_merge`: fraction of `ΔW` surviving a bf16 merge by module type (D3).

#### Pitfalls and misconceptions

- "`α` is a free learning rate": it is a multiplier interacting with the optimizer; under Adam, doubling `α` at `B=0` changes the first step's output change by 2× but not the update size of `B`.
- Changing `r` while keeping `α/r` fixed (α ∝ r) silently changes the effective scale.
- Comparing ranks at one learning rate (the minimum moves).
- Quoting trainable-parameter fractions as memory: adapter state is fp32 and the activation term is unchanged by freezing.
- Merging into bf16 and concluding "LoRA doesn't work": D3.
- Dropout on LoRA input only helps in the small-data regime; with `dropout>0` the merged and unmerged paths agree only in `eval()` mode.
- Target-module names differ between HF, R1 and other repos; `add_lora` returns the list of replaced names for a reason.
- LoRA on the embedding/`lm_head` requires a different layer (tied weights); not covered.

#### Summary (draft bullets)

- LoRA replaces a frozen `W` by `W + γBA` with `B=0`; training touches only `A`, `B`.
- The right multiplier depends on the optimizer: `1/r` equalizes Adam's first-step output change across ranks, `1/√r` equalizes SGD's; the experiment decides what matters over a whole run.
- `A` and `B` have different effective learning rates; LoRA+ sets a ratio.
- Apply LoRA to MLP layers as well as attention; adapter learning rates are about an order of magnitude above full fine-tuning's (as reported; measured here).
- Merge in high precision or serve unmerged; evaluate the merged artifact separately.

#### Exercises

1. *Check.* How many trainable parameters does all-linear LoRA with `r=8` add to a layer with `W_q: 2048×1024`, `W_k, W_v: 1024×1024`, `W_o: 1024×2048`, and MLP `1024→3072` (×2) and `3072→1024`? *Sketch:* `8·[(2048+1024) + 2(1024+1024) + (1024+2048) + 2(1024+3072) + (3072+1024)] = 8·[3072+4096+3072+8192+4096] = 8·22,528 = 180,224`.
2. *Check.* At the first step, why does `A` receive exactly zero gradient, and what happens to learning if `A` and `B` are both initialized to zero? *Sketch:* `∂L/∂A ∝ Bᵀ g xᵀ = 0` when `B=0`; with `A=0` as well, `∂L/∂B ∝ g (Ax)ᵀ = 0`: both stay zero forever.
3. *Derive.* Extend D1 (SGD case) to `T` identical steps and show `A`'s contribution first appears at step 2 with size proportional to `η_A η_B²`. *Sketch:* the formula of D2.
4. *Derive.* Show that scaling `A → cA` and `lr_B → lr_B/c`, `α → α/c` leaves the Adam first-step function change unchanged (a parametrization invariance), by tracking `γ B A x`. *Sketch:* `γ→γ/c`, `A→cA` cancel in the product at init; the Adam step for `B` does not depend on the gradient scale.
5. *Measure.* Predict the optimal learning-rate ratio LoRA/full FT for your setup, then measure with a ×1.5 grid; compare with "10×" and state the interval of your estimate from the 3 seeds.
6. *Measure.* Predict at which batch size (8k, 32k, 128k tokens per step) LoRA's gap to full FT first exceeds the seed noise; measure at constant epochs.
7. *Open.* Implement LoRA for the embedding matrix or a fused `W_qkv` layer (R1 `d2l.GPT`), define what the rank-`r` update is, test merge/unmerge, and decide how `lora_disabled` should treat it.

#### Further reading

- **Papers:** [LoRA](https://arxiv.org/abs/2106.09685); [rsLoRA](https://arxiv.org/abs/2312.03732) (proves `1/√r`; stalls at high rank with `1/r`); [LoRA+](https://arxiv.org/abs/2402.12354) (large-width argument; 1–2% and up to ≈ 2× in the abstract); [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (rank of full-FT updates is 10–100× typical LoRA ranks; less forgetting); [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228) (new, verified; "intruder dimensions").
- **Tutorials:** Thinking Machines, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (read it for the 10×, `1/r`, all-layers and batch-size findings; checked 2026-10-28); Raschka's practical tips; [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora) (verified 200).
- **Code:** [PEFT](https://github.com/huggingface/peft); [Unsloth](https://github.com/unslothai/unsloth) (fused LoRA kernels; 7.7).

#### Risks and open decisions

- **Optimizer-dependent scaling (decided 2026-10-29).** The default is `rslora=False` (`α/r`), which follows PEFT's `use_rslora=False`; the grid reports both multipliers. D1 shows `α/√r` is not rank-invariant for Adam at step one, and rsLoRA's claim is made at a fixed learning rate, so the fixed-learning-rate slice (H5) tests that claim directly; any recommendation printed by the notebook comes from the measured slice and grid, not from either paper. If `lr★` makes `α/√r` diverge at rank 64, the slice reports the divergence as the finding and the committed grid supplies the lower learning rates.
- **Grid cost.** The full grid is 4×3×2×6×3 seeds ≈ 430 runs of ≈ 40 s; a T1 offline job of about 5 h (to be measured). Fallback: drop the `α/√r` arm to the fixed-learning-rate column of H5 (ranks {1, 4, 16, 64} at `lr★`, three seeds).
- **bf16 base with fp32 adapters** costs casts; fallback is bf16 adapters with fp32 optimizer master (not recommended; noted in pitfalls).
- **PEFT API drift** (`create_loraplus_optimizer`, `use_rslora`): confirm against the 7.7 pin.
- **R1 module names** (`W_q`… vs `q_proj`): the alias table covers both; if 1.2's `load_qwen3` renames layers, the alias table is the only place to change.

### 7.5 Beyond LoRA

- **Label:** `sec_lm_peft` · **Tier:** T1 (native) · **Budget:** ≤ 15 min (six ≈ 1.5-min runs plus CPU analysis), peak ≤ 12 GiB on `Qwen3-0.6B-Base` · **Codex units:** 8.5 (adapters, soft/prefix tuning), 8.6 (DoRA, rank-stabilized variants as mechanism probes).
- **Builds on (book):** `sec_mdl-svd-low-rank` (Eckart–Young, randomized SVD of Halko et al. as the fast path for PiSSA; the PiSSA sentence of `sec_mdl-svd-modern-dl`); `sec_adamw` (`subsec_adamw-memory`: what optimizer state costs); `sec_memory_precision` (peak-memory measurement). Nothing is re-derived.
- **Builds on (this part):** 7.4 (`d2l.LoRALinear`, `d2l.add_lora`, `d2l.lora_param_groups`, `d2l.ft_memory`), 7.1 (`d2l.ft_memory` extension points, D2 on bf16 absorption, which GaLore reopens), 7.2 (data path), 1.5 (paired bootstrap).
- **Organizing principle (the section's table `tab_lm_peft_taxonomy`):** each method changes exactly one thing relative to LoRA, and the notebook says which.

| Axis | What changes | Methods | Trainable per layer (Qwen3-0.6B, formula) | Mergeable? |
|---|---|---|---|---|
| Initialization | Where `(A, B)` start | PiSSA (SVD init), LoftQ (7.6) | `r(m+n)`, same as LoRA | yes |
| Parameterization | How `ΔW` is built from parameters | DoRA (magnitude × direction), VeRA (shared frozen `A,B`, trained vectors) | DoRA `r(m+n)+m`; VeRA `r + m` per target | yes (DoRA, VeRA) |
| Where parameters enter | Which activation they touch | Bottleneck adapters, prefix tuning, prompt tuning, IA³ | adapters `2(2hb+h+b)` per layer; prefix `2·p·d_kv` per layer; prompt `p·h` total; IA³ `d_k + d_v + d_ff` per layer | IA³ yes; others no |
| Optimizer state | What the optimizer stores | GaLore (low-rank gradient projection) | all weights trainable; state `r·n` per matrix | n/a (full-rank weights) |

- **Exports (`#@save`):**
  - `d2l.pissa_init(lora: LoRALinear, niter: int | None = 4) -> None`: sets `W_res = W − B₀A₀`, `B₀ = U_r S_r^{1/2}`, `A₀ = S_r^{1/2} V_rᵀ` using `torch.svd_lowrank(W, q=r+8, niter=niter)` (exact `torch.linalg.svd` when `niter is None`); writes `W_res` into the base weight and the factors into the adapter; sets `scaling` so that `scaling·B₀A₀` equals the truncated SVD (the module is created with `alpha = r`, so `γ = 1`, as PEFT recommends for PiSSA).
  - `d2l.DoRALinear(base, r=8, alpha=16.0, dropout=0.0, detach_norm=True)`: `LoRALinear` subclass with a magnitude vector `m ∈ ℝ^{out}` (init `‖W‖` per output row) and output `y = (m/‖V‖_row) ⊙ (W+γBA)x` computed without materializing `V` (via `‖V_i‖² = ‖W_i‖² + 2γ W_i·(BA)_i + γ²‖(BA)_i‖²`, the cross term from `(WᵀB)` of shape `[in, r]`); `detach_norm=True` treats `‖V‖` as constant in backward (the memory shortcut of the DoRA paper); `merge()` folds `m/‖V‖` into the weight.
  - `d2l.PromptTuningLM(model, num_virtual_tokens=32, init='vocab')`: prepends `p` learned embeddings to the input embeddings; the loss mask and attention mask are extended by `p` leading positions; only the `p×h` embedding matrix trains; `init='vocab'` samples rows of the embedding table.
  - `d2l.GaLoreAdamW(params, lr, rank=128, update_proj_gap=200, scale=0.25, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, master_fp32=True)`: per-matrix gradient projection `R = PᵀG` (or `G Q`, whichever side is shorter), AdamW moments in the projected space, update back-projected and scaled; `P` refreshed from the SVD of the current gradient every `update_proj_gap` steps; non-matrix parameters use plain AdamW.
  - `d2l.peft_param_table(config, r=16, bottleneck=64, num_virtual_tokens=32) -> DataFrame`: trainable parameters and optimizer-state bytes (fp32 Adam) per method, from a config (used for `tab_lm_peft_params`).
  - Artifacts: `DATA_HUB['lm-ch07-peft-runs']` (JSONL, per-run configs, loss curves, peak memory).
- **Models and data:** `Qwen/Qwen3-0.6B-Base`; `lm-sft-mix-v0` (1.5M-token slice, 2,000-conversation validation split, as in 7.4); retention: `allenai/ai2_arc` ARC-Challenge test subset of 300 items (CC-BY-SA-4.0) through the Chapter 1 suite. Config facts (Hub, 2026-10-28): 28 layers, `hidden_size` 1024, 8 KV heads × 128 = 1,024 KV width, `intermediate_size` 3072.
- **Packages:** `peft` (all of `use_dora`, `init_lora_weights='pissa_niter_4'`, `VeraConfig`, `IA3Config`, `PromptTuningConfig`, `PrefixTuningConfig` exist in the pinned release; confirm names in 7.7's pin), `adapters` 1.3.0 (AdapterHub; verified on PyPI 2026-10-28) for bottleneck adapters, `galore-torch` (last PyPI release March 2024; confirm that the Transformers pin still exposes `optim='galore_adamw'`, else the from-scratch optimizer is the only arm).

#### Narrative arc

1. **The question.** Opening cell prints `peft_param_table` for Qwen3-0.6B: trainable parameters and fp32-Adam state bytes for LoRA r=16, DoRA r=16, VeRA r=256, IA³, bottleneck adapters (64), prefix tuning (p=32), prompt tuning (p=32), full AdamW, and GaLore r=128. Symptom: trainable fractions range over four orders of magnitude, but *optimizer state* is not the whole memory story: a second column prints the measured peak memory of one step of each (to be measured). The question: which of these changes buy memory, which buy quality, and which buy neither at this scale?
2. **Change the initialization: PiSSA.** Construction in a cell; function-preserving init (`W_res + B₀A₀ = W`); gradient scale comparison (D1); the quantization link: quantizing `W_res` rather than `W` (measured preview, finished in 7.6 with LoftQ).
3. **Change the parameterization: DoRA, VeRA.** DoRA from the weight-decomposition analysis: measure `ΔM` (change of row magnitudes) and `ΔD` (change of row directions) for a full-FT checkpoint (7.1/7.2 run, loaded from `DATA_HUB`) and for a LoRA run (7.4), reproducing the DoRA paper's diagnostic (it reports different magnitude/direction behavior for FT and LoRA; we report ours, with `d2l.paired_bootstrap` over layers). Then the gradient identity (D2), the unmaterialized norm, and `detach_norm`. VeRA in 25 lines of the notebook (no export): shared frozen `A, B` from a seed, trainable `λ_d ∈ ℝ^r`, `λ_b ∈ ℝ^m`; parameter counts.
4. **Change where parameters enter.** One cell per mechanism, each ≤ 20 lines on the 0.6B model through PEFT or `adapters`: bottleneck adapter (Houlsby: `h + f(hW↓)W↑`, cannot be merged, adds latency, measured per-token delta); prefix tuning (learned key/value prefixes per layer; prefix and prompt tuning also *consume sequence length*, measured attention cost); prompt tuning (`PromptTuningLM`, `p×h` parameters; the Prompt Tuning paper reports that the gap to full fine-tuning closes as model size grows, which at 0.6B is a prediction we can test: expected to underperform); IA³ (rescales `K`, `V` and the FFN hidden activations by learned vectors initialized at 1; about 143k parameters at 0.6B by the table).
5. **Change what the optimizer stores: GaLore.** The memory formula (D3), the from-scratch optimizer, the rank-growth property (rank of `W_K − W_0` can exceed `r` after several subspace switches, unlike LoRA), and the bf16 problem: GaLore trains *full-rank weights*, so the 7.1 D2 absorption issue is back; the notebook runs both `master_fp32=True` (memory higher than LoRA's) and a per-layer-update bf16 variant (matched memory, quality to be measured).
6. **Matched memory.** Procedure: measure peak memory `M` of 7.4's LoRA r=16 all-linear run at the section's batch; for each method choose its size knob (DoRA `r`, adapters bottleneck, prompt length, GaLore rank) to land within ±10% of `M` using `ft_memory` as the predictor and a measured confirmation; if a method cannot reach `M` (GaLore with fp32 master) it is reported at its own memory and marked. Table `tab_lm_peft_matched`.
7. **A decision rule.** One page: if you need merge-ability and serving simplicity, LoRA/DoRA; if you need thousands of per-user adapters, VeRA/IA³-class budgets; if the data distribution is tiny and the model huge, prompt/prefix tuning; if you need full-rank updates under a memory cap, GaLore. The rule is stated as conditional on the measurements here, not as general truth.

#### Derivations

**D1. PiSSA: exact initialization, minimal residual and gradient scale (`eq_lm_pissa`).**
Statement: for `W = UΣVᵀ`, setting `B₀ = U_r Σ_r^{1/2}`, `A₀ = Σ_r^{1/2} V_rᵀ`, `W_res = W − B₀A₀` gives a model whose initial function equals the base's, with `‖W_res‖_F² = Σ_{i>r} σ_i²` minimal over all rank-`r` choices (Eckart–Young, `sec_mdl-svd-low-rank`), and with initial gradients `∂L/∂B = G V_r Σ_r^{1/2}`, `∂L/∂A = Σ_r^{1/2} U_rᵀ G` (`γ=1`), where `G = ∂L/∂W`.
Steps:
1. Function: `W_res + B₀A₀ = W` by construction.
2. Residual: Eckart–Young says the best rank-`r` approximation error in Frobenius norm is `√Σ_{i>r}σ_i²`, attained by the truncated SVD; so `W_res` has the smallest energy among all splits `W = W_res' + (rank-r)`.
3. Gradients: `∂L/∂B = γ G Aᵀ`, `∂L/∂A = γ Bᵀ G` for `y = W_res x + γ B A x`; substitute `A₀, B₀`.
4. Compare with LoRA at step 0: `B=0`, `∂L/∂A = 0` and `∂L/∂B = γ G Aᵀ` with random `A` (entries of variance `1/(3·in)`): the PiSSA gradient is non-zero for *both* factors from step 0 and is aligned with the principal subspace of `W`.
5. Capture ratio: define `κ(P) = ‖P_Gᵀ G‖_F²/‖G‖_F²`, the fraction of gradient energy inside the adapter's row-space; LoRA-random has `E κ = r/in` (random `r`-dimensional subspace); PiSSA has `κ = ‖G V_r‖_F²/‖G‖_F²`. Whether `κ` is larger for PiSSA is an empirical question about `G`'s alignment with `W`'s principal directions.
Check cell: on every `q_proj` and `down_proj` of 3 layers of Qwen3-0.6B: (i) reconstruction `‖W_res + B₀A₀ − W‖_F/‖W‖_F ≤ 1e-6` (fp32); (ii) `‖W_res‖_F²/‖W‖_F²` equals the tail energy from the exact SVD to 1e-5, and the randomized (`niter=4`) value is within a stated tolerance of the exact; (iii) `κ` for random `A` vs PiSSA from the gradient on one validation batch (measured).

**D2. DoRA: the direction gradient is orthogonal to the direction (`eq_lm_dora_grad`).**
Statement: for one output row `v ∈ ℝ^n` (`v = w_i + γ(BA)_i`), `w = m·v/‖v‖`; with `g = ∂L/∂w`: `∂L/∂m = v̂ᵀ g` and `∂L/∂v = (m/‖v‖)(g − (v̂ᵀg) v̂)` where `v̂ = v/‖v‖`. Hence the update of the direction parameters is orthogonal to the current direction (changes direction only), and the magnitude changes only through `m`.
Steps: (1) `∂w/∂m = v̂`; (2) `∂w/∂v = (m/‖v‖)(I − v̂v̂ᵀ)`; (3) chain rule with `g`; (4) `v̂ᵀ(I − v̂v̂ᵀ) = 0`, so a first-order step in `v` cannot change `‖v‖`; (5) with `detach_norm=True` the projection term is dropped: `∂L/∂v ≈ (m/‖v‖) g`, a cheaper estimate whose cosine with the exact gradient is `‖g − (v̂ᵀg)v̂‖/‖g‖`-dependent, near 1 when `g` is nearly orthogonal to `v̂`.
Check cell: autograd vs formula for random `v, g` (1e-6); `detach_norm` vs exact on real gradients: cosine and relative norm error per module type (to be measured); memory saving of the detached path (to be measured).

**D3. GaLore memory and rank (`eq_lm_galore`).**
Statement: for an `m×n` weight with `m ≤ n` and projection rank `r`, AdamW states cost `2·m·n` values; GaLore stores `2·r·n` plus the projector `m·r`: `r(2n+m)`. The per-step update is confined to `span(P)`, so `rank(ΔW_step) ≤ r`; after `K` projector refreshes `rank(W_K − W_0) ≤ K·r`.
Steps: (1) `R = PᵀG ∈ ℝ^{r×n}`; Adam moments have the shape of `R`; (2) update `ΔW = −η·α·P·Adam(R)`, rank ≤ `r`; (3) summing updates under different projectors gives rank ≤ `K·r` (subadditivity); (4) memory: `m·n·2 → r(2n+m)` values; for `m=1024, n=3072, r=128`: `2·1024·3072 = 6.29M` vs `128·(6144+1024) = 0.92M`, a factor 6.9 (arithmetic, evaluated by the cell); (5) nothing here removes the full-rank *gradient* (needed to form `R`) or the full-rank weights; per-layer updates during backward (the GaLore paper's trick) free each gradient immediately.
Check cell: toy 64×128 weight, `r=4`, three refreshes: numeric rank of `W_K − W_0` ≤ 12; LoRA's ≤ 4; measured optimizer-state bytes equal the formula exactly.

#### From-scratch implementation

Budget ≈ 145 lines (≤ 150).
- `pissa_init` (20 lines), `DoRALinear` (45 lines), `PromptTuningLM` (25 lines), `GaLoreAdamW` (45 lines), `peft_param_table` (10 lines). VeRA and adapters live in notebook cells outside the budget (≤ 25 lines each, not exported).
- **Tests.** (1) PiSSA: D1 (i)–(ii); initial logits equal the base's to 1e-4 (fp32). (2) DoRA: at init (`B=0`, `m=‖W‖`) `DoRALinear(x) == base(x)` to 1e-6; D2 identities; `merge()` output equality to 1e-5 (fp32); PEFT `use_dora=True` oracle with copied weights (the section records whether PEFT's norm axis and `detach` convention agree). (3) Prompt tuning: output shape and loss mask extension; prepended positions never contribute to loss; gradient reaches only the prompt embedding. (4) GaLore: with `rank = min(m,n)` and `update_proj_gap=1` the update equals AdamW's up to the `scale` factor (sanity), state bytes equal D3, rank-growth check; compare one step against `galore_torch.GaLoreAdamW` on a 2-layer MLP (oracle; if the package is unavailable skip and say so).

#### In production

- **PEFT:** `LoraConfig(use_dora=True)`, `LoraConfig(init_lora_weights='pissa_niter_4')`, `VeraConfig`, `IA3Config`, `PromptTuningConfig`, `PrefixTuningConfig`; `get_peft_model(...).print_trainable_parameters()` as the cross-check for `peft_param_table` (counts must agree exactly).
- **GaLore:** Transformers `TrainingArguments(optim='galore_adamw', optim_target_modules=[...])` and the `galore-torch` package, or the from-scratch optimizer; per-layer variants (`galore_adamw_layerwise`) as documented in the pinned Transformers release (confirm availability).
- **Adapters library** for Houlsby/Pfeiffer bottleneck adapters (the PEFT library does not implement them).
- **Oracle use:** each from-scratch module is checked against the PEFT counterpart with identical weights; disagreements are findings (e.g., DoRA norm axis, prompt-tuning initialization), reported, not hidden.

#### Headline experiment

- **Hypothesis.** At matched peak memory on a 0.6B model and a ≈ 1.5M-token SFT budget: (H1) DoRA ≈ LoRA (within seed noise) on validation loss; (H2) prompt tuning is clearly worse than LoRA (the Prompt Tuning paper's size trend predicts a gap at this scale); (H3) GaLore with an fp32 master reaches lower validation loss than LoRA but at higher memory, and the bf16 per-layer variant loses part of that advantage (7.1 D2); (H4) PiSSA's init converges faster over the first 50 steps than LoRA's at the same learning rate, with the caveat that the optimizer-dependent scale of 7.4 D1 applies to it too.
- **Setup.** `Qwen3-0.6B-Base`; `lm-sft-mix-v0` slice (1.5M tokens, rows of 1,024, 8 rows per step); learning rates from a 3-point grid per method around its documented default (LoRA/DoRA/PiSSA: the 7.4 optimum; prompt tuning: `{3e-2, 1e-1, 3e-1}`; GaLore: `{1e-5, 3e-5, 1e-4}` with `scale=0.25`), 3 seeds. Runs: LoRA r=16 (baseline), DoRA r=16, PiSSA r=16, prompt tuning p=32, GaLore r=128 (fp32 master), GaLore r=128 (bf16, per-layer update), full FT (reference; 7.4's best).
- **Procedure.** Tune the learning rate per method on the first 2 seeds' validation loss; report seed 3 plus all seeds' final loss as the primary; confirm memory with `torch.cuda.max_memory_allocated`; evaluate ARC-Challenge retention (300 items).
- **Metrics.** Validation loss; first-50-step training loss; peak GiB; step time; trainable parameters; retention delta vs base.
- **Plot spec.** `fig_lm_peft_frontier`: x = peak memory (GiB, linear), y = validation loss; one marker per method with seed range bars; the full-FT point marked; dashed vertical line at LoRA's memory `M`. `fig_lm_peft_early`: first 50 steps, LoRA vs PiSSA vs DoRA. `tab_lm_peft_matched`.
- **Expected outcome.** Directions from sources: DoRA's paper reports consistent gains over LoRA on its benchmarks; PiSSA's reports faster convergence; GaLore's reports performance near full-rank training with up to 65.5% less optimizer-state memory (abstract). Our run reports agreement or disagreement per hypothesis with intervals; absolute numbers to be measured.
- **Noise check.** 3 seeds; differences reported only beyond 2× the across-seed standard deviation; for H1, the interval for DoRA − LoRA from seed pairs; "within noise" printed when applicable. The learning-rate grid is coarse (×3), so a method that is best at the grid edge is flagged as unconverged.
- **T1 reduction.** Native T1.
- **Artifacts committed.** `lm-ch07-peft-runs` and the figures.

#### Figures and tables

- `tab_lm_peft_taxonomy` (above); `tab_lm_peft_params` (`peft_param_table` output); `tab_lm_peft_matched`.
- `fig_lm_peft_dm`: `ΔM` vs `ΔD` per layer for full FT, LoRA and DoRA (scatter; one panel each).
- `fig_lm_peft_frontier`, `fig_lm_peft_early`.
- `fig_lm_galore_rank`: numeric rank of `W_K − W_0` vs number of subspace refreshes for GaLore and LoRA.

#### Pitfalls and misconceptions

- "Fewer trainable parameters ⇒ proportionally less memory": activations, the frozen base and (for GaLore) full gradients remain.
- DoRA's norm axis and the `detach_norm` shortcut differ between papers' notation and libraries' tensors; compare numerically.
- Prompt and prefix tuning lengthen the sequence (attention cost and KV cache) and need learning rates two to three orders above LoRA's.
- GaLore with bf16 weights hits update absorption; its published memory numbers rely on per-layer updates and 8-bit states.
- PiSSA modifies the base weights (`W_res`): saving only the adapter is not enough to reconstruct the model.
- Reporting a "winner" from one seed at one learning rate per method.

#### Summary (draft bullets)

- PEFT methods differ in what they change: initialization, parameterization, where parameters enter, or what the optimizer stores.
- PiSSA starts from the principal components and an exactly preserved function; DoRA separates magnitude from direction, whose gradient is orthogonal to the current direction.
- Prompt-like methods lengthen the sequence and tend to need large models; GaLore keeps full-rank weights and shrinks optimizer state only.
- Compare at matched peak memory, with seeds, and expect most differences between LoRA variants to be small.

#### Exercises

1. *Check.* IA³ adds learned vectors on `K`, `V` and the FFN hidden activations. Count its parameters for Qwen3-0.6B and Qwen3-8B. *Sketch:* 0.6B: `28·(1024+1024+3072) = 143,360`; 8B: `36·(1024+1024+12288) = 516,096` (KV width `8·128 = 1,024`).
2. *Check.* Why can a bottleneck adapter not be merged into `W`, while DoRA can? *Sketch:* the adapter applies a nonlinearity between the two projections; DoRA's output is linear in `x` for fixed parameters.
3. *Derive.* Show that the DoRA magnitude vector at initialization makes the layer equal to the base layer, and compute the extra parameters `m` for Qwen3-0.6B all-linear. *Sketch:* `m_i = ‖W_i‖`, `B=0` ⇒ `V = W` ⇒ `w = ‖W_i‖·W_i/‖W_i‖`; extra `28·(2048+1024+1024+1024+3072+3072+1024) = 28·12,288 = 344,064`.
4. *Derive.* Prove that after `K` GaLore refreshes the numeric rank of `W_K − W_0` is at most `K·r`, and find a case where it equals `K·r`. *Sketch:* sum of `K` matrices of rank ≤ `r`; equality for generic orthogonal projector subspaces.
5. *Measure.* Predict the prompt length `p` at which prompt tuning matches LoRA r=16 on your validation loss, then sweep `p ∈ {8, 32, 128}`; include the attention-cost increase.
6. *Measure.* Predict whether DoRA's advantage (if any) is larger at `r=4` or `r=64`; measure with 3 seeds.
7. *Open.* Design a PEFT method for one weight type (embedding, norm, router) not covered here, state what it changes in the taxonomy, and test merge-ability.

#### Further reading

- **Papers:** [DoRA](https://arxiv.org/abs/2402.09353); [PiSSA](https://arxiv.org/abs/2404.02948); [VeRA](https://arxiv.org/abs/2310.11454); [Adapters](https://arxiv.org/abs/1902.00751); [Prefix-Tuning](https://arxiv.org/abs/2101.00190); [Prompt Tuning](https://arxiv.org/abs/2104.08691); [IA³](https://arxiv.org/abs/2205.05638); [GaLore](https://arxiv.org/abs/2403.03507).
- **Tutorials:** [PEFT docs](https://huggingface.co/docs/peft/index); [HF smol course](https://huggingface.co/learn/smol-course/en/unit0/1) (LoRA unit).
- **Code:** [PEFT](https://github.com/huggingface/peft); `galore-torch` (PyPI) and the GaLore repository linked from the paper; [AdapterHub `adapters`](https://pypi.org/project/adapters/) (PyPI, verified).

#### Risks and open decisions

- **Matched memory is not achievable for every method** (GaLore with fp32 master); the section reports such methods at their own memory and says so.
- **Library API drift** (`galore-torch` unmaintained since 2024; Transformers may have changed its GaLore hooks): fallback = from-scratch optimizer only.
- **Prompt tuning at 0.6B** may be unstable; if it diverges at all three learning rates, report that as the result.
- **DoRA speed:** the unmaterialized norm adds a `WᵀB` matmul; a PEFT-native DoRA is faster in some versions; speed claims are measured, not asserted.
- **Scope pressure:** the section covers eight methods; VeRA and bottleneck adapters are demonstrated, not tuned, to keep T1 time.

### 7.6 Quantization for fine-tuning: QLoRA

- **Label:** `sec_lm_qlora` · **Tier:** T1 (native; the 14B run is a ≤ 100-step demonstration on a whole 24 GB GPU, the quality comparisons run on 1.7B) · **Budget:** ≤ 15 min in total (quantizer study on CPU/GPU ≈ 3 min; 14B QLoRA ≤ 100 steps ≈ 3 min, to be measured; 1.7B comparison arms ≈ 3 × 2 min), peak ≤ 15 GiB except the 14B arm, which claims a whole 24 GB GPU (expected ≈ 13–15 GB, to be measured) · **Codex units:** 8.7 (quantized adaptation), 12.1 only for the "what compression is" framing (the broader taxonomy is Chapter 11's).
- **Scope.** This is where the book first teaches quantization, and only what fine-tuning needs: storing a *frozen* weight matrix in 4–8 bits and using it in a matmul. Activations, calibration data, outlier handling, GPTQ/AWQ, rotations, speed and KV compression belong to Chapter 11/12; the section ends with a hand-off table (below).
- **Builds on (book):**
  - `sec_numerics` ("Below 16 Bits") and `sec_hardware` (`fig_float_formats`, `tab_gpu_specs`): fp8 has two layouts and fp4 has sixteen values, with per-block scaling. One recap paragraph; the format ladder is not re-drawn. 4.4 (block scaling, FP8/NVFP4 training formats) supplies the *scale-format* vocabulary; this section reuses it for weights.
  - `sec_memory_precision`/`sec_adamw` (`subsec_adamw-memory`): 8-bit optimizer state named there as `Dettmers et al. 2022`; implemented here.
  - `sec_custom_layer` ("Custom Gradients", the straight-through estimator): the model for a custom `autograd.Function`. Here the function is *not* a straight-through estimator: the weight is frozen, and the function exists so that backward re-dequantizes instead of saving the dequantized weight.
  - `sec_perf_model` (roofline): dequantization is a memory-traffic story (D5).
  - `sec_mdl-svd-low-rank` (Eckart–Young): LoftQ's second half-step.
- **Builds on (this part):** 7.4 (`d2l.LoRALinear`, `d2l.add_lora`, `d2l.merge_lora`, `d2l.ft_memory`), 7.5 (`d2l.pissa_init`, as the "absorb the residual" cousin of LoftQ), 7.1 (`d2l.chunked_lm_loss`), 7.2 (data), 4.4 (block-scaled formats; assumed only as vocabulary), 1.5 (`d2l.paired_bootstrap`).
- **Exports (`#@save`; reserved names bold):** the contract is in the next subsection; summary:
  - **`d2l.quantize_blockwise(w, bits=4, block_size=64, symmetric=True, scale_dtype=torch.float32) -> QTensor`** (per editor's interface note: `bits` is the second positional argument and defaults to 4; the returned `QTensor` has fields `codes`, `scales` (absmax-derived scale per block), `zeros` (`None` if symmetric), `block_size`, `shape`, `bits`, plus the extra fields of the layout table; `dequantize` and `quantize_nf4` take/return the same type; no tuple form exists)
  - **`d2l.quantize_nf4(w, block_size=64, double_quant=False, scale_dtype=torch.float32) -> QTensor`**
  - **`d2l.dequantize(q, dtype=torch.bfloat16) -> Tensor`**
  - `d2l.QTensor` (dataclass), `d2l.NF4_CODEBOOK` (16 fp32 values), `d2l.quantize_codebook(w, codebook, block_size=64, scale_dtype=torch.float32, double_quant=False) -> QTensor`.
  - `d2l.QuantLinear(q: QTensor, bias=None, compute_dtype=torch.bfloat16)` and `QuantLinear.from_linear(linear, kind='nf4', **kw)`; `d2l.quantize_model(model, kind='nf4', block_size=64, double_quant=True, skip=('lm_head', 'embed_tokens')) -> dict` (counts and bytes saved).
  - `d2l.loftq_init(lora: LoRALinear, kind='nf4', block_size=64, iters=5) -> list[float]` (returns the error after each iteration).
  - `d2l.quant_error_report(w, q, x=None) -> dict`: relative Frobenius error, SQNR (dB), max abs error and the guarantee it should satisfy (G1–G3), optional output error on activations `x`.
  - `d2l.ft_memory` gains the `quant` argument values `'nf4'`, `'nf4_dq'`, `'int8'` (7.1's function; bits per parameter from the QTensor accounting below).
  - Artifacts: `DATA_HUB['lm-ch07-quant-errors']` (per-tensor error table for Qwen3-0.6B/1.7B), `DATA_HUB['lm-ch07-qlora-runs']` (logs).
- **Models and data:** `Qwen/Qwen3-8B-Base` (8.19B; 36 layers, `hidden_size` 4096, 32 heads / 8 KV heads, `head_dim` 128, `intermediate_size` 12288, **untied** embeddings; Apache-2.0) as the fallback that fits a 15 GiB slot; `Qwen/Qwen3-14B-Base` (`0b0bd3732e2c374d483664439ea334928b65f304`; 14.77B; 40 layers, `hidden_size` 5120, 40 heads / 8 KV heads, `head_dim` 128, `intermediate_size` 17408, **untied** embeddings; Apache-2.0) for the headline run on a whole 24 GB GPU; `Qwen/Qwen3-1.7B-Base` for the BF16-LoRA vs QLoRA vs LoftQ comparison; `Qwen/Qwen3-0.6B-Base` for weight studies and unit tests; `lm-sft-mix-v0` (1.5M-token slice; validation split of 2,000 conversations). Config values read from the Hub on 2026-10-28; computed from them (arithmetic printed by the notebook): 8B linear parameters 6.95B, embedding + `lm_head` 1.24B (both kept in bf16), LoRA r=16 all-linear 43.6M parameters, 1.7B linear parameters 1.41B.
- **Packages:** `bitsandbytes` (0.50.2 on PyPI at 2026-10-28; oracle and paged/8-bit optimizers; its metadata requires `torch<3,>=2.4`, so the book's `torch==2.15.0` is allowed; that the wheel runs on Ada and Ampere is confirmed in 7.7's environment), `torchao` (0.18.0; `NF4Tensor` as a second, pure-PyTorch oracle; confirm in the pinned release), `peft`, `transformers`.

#### The QTensor layout contract (read by Chapters 11 and 12)

All quantizers return a `QTensor` for a weight `w` of shape `[*lead, n]` (for `nn.Linear` weights `[out, in]`, so `n = in_features`, the reduction dimension of `y = x Wᵀ`).

| Field | Shape and dtype | Meaning |
|---|---|---|
| `codes` | `bits=8`: `int8` (symmetric) or `uint8` (asymmetric), shape `[*lead, n]`. `bits=4`: `uint8`, shape `[*lead, n/2]` | Packed along the last axis. **Two 4-bit codes per byte; the even-indexed element (index `2i`) is in the high nibble, the odd-indexed (`2i+1`) in the low nibble** (the bitsandbytes order, read from its CUDA kernel source on 2026-10-28), so that the first element of a block is the most significant nibble of its first byte. |
| `scales` | `[*lead, n/block_size]`, `scale_dtype` (default `float32`; `float16`/`bfloat16` allowed) | One scale per contiguous block of `block_size` elements **along the last axis**; blocks never cross rows. `n % block_size == 0` is required (every Qwen3 projection satisfies it for `block_size ∈ {16, 32, 64, 128, 256}`: input widths 1024, 2048, 3072, 4096, 6144, 12288); otherwise `ValueError`. A block with `absmax = 0` stores scale 0 and codes 0. |
| `zeros` | like `scales`, `uint8`; `None` if `symmetric` | Asymmetric (affine) grids only. |
| `codebook` | `float32[2**bits]` or `None` | `None` for integer grids; the 16 sorted NF4 values for `kind='nf4'`; arbitrary for `quantize_codebook`. Code `c` decodes to `codebook[c]`; code order = sorted value order. |
| double-quant state | `scales_q: uint8 [*lead, n/block_size]`, `scales_absmax: float32 [⌈(numel/block_size)/256⌉]`, `scales_offset: float32 []`; `scales` is `None` | The flattened scale vector minus its mean (`scales_offset`) is quantized in blocks of **256** with a symmetric int8 absmax code. Dequantizing the scales reverses it. (This differs from bitsandbytes, whose second-level code is its dynamic 8-bit map; so equality with bnb holds without double quantization, and within a stated tolerance with it.) |
| metadata | `kind ∈ {'int','nf4','codebook'}`, `bits`, `block_size`, `symmetric`, `shape`, `scale_dtype` | |
| derived | `nbytes`, `bits_per_param` | Exact storage in bytes: `codes.numel()·itemsize + scales (or scales_q + scales_absmax + offset) + zeros`; `bits_per_param = 8·nbytes/numel`. No padding is hidden: the notebook prints the table below from `nbytes`. |

| Scheme (per weight element) | Bits |
|---|---|
| int8, per-tensor | `8 + ε` |
| int8, block 64, fp32 scale | `8 + 32/64 = 8.5` |
| NF4, block 64, fp32 scale | `4 + 32/64 = 4.5` |
| NF4, block 64, double quant (block 256, fp32 second-level scale, fp32 mean) | `4 + 8/64 + 32/(64·256) = 4.127` |
| int4, block 128, fp16 scale (the GPTQ/AWQ-style grouping used for inference) | `4 + 16/128 = 4.125` |
| fp4 (E2M1) block 32 with an 8-bit scale (MXFP4-like) | `4 + 8/32 = 4.25` |

**Round-trip guarantees (tested in the notebook, with the tolerances stated; `a` is a block's absmax, `qmax = 2^{b−1}−1`).**
- **G1 (integer, symmetric).** With fp32 scales: `|w − ŵ| ≤ a/(2 qmax)` elementwise (plus fp32 rounding, test tolerance `1e-6·a`); the absmax element and every zero are reproduced exactly. With fp16/bf16 scales `ŝ`: `|w − ŵ| ≤ ŝ/2 + a·2^{−11}` (fp16) or `ŝ/2 + a·2^{−8}` (bf16), because rounding the scale can push `|w|/ŝ` slightly past `qmax`, where codes are clipped.
- **G2 (NF4).** `|w − ŵ| ≤ g_max·a/2` with `g_max = 0.30381` (the gap between −1 and −0.6962 in the normalized grid; verified from the grid values), i.e. `≤ 0.1519·a`; the absmax element and zero are exact (±1 and 0 are in the grid).
- **G3 (double quantization).** With stage-1 scale error `δ_j = |a_j − ŝ_j| ≤ max_k |a_k − offset|/(2·127)` over its 256-block: `|w − ŵ| ≤ δ_j + ŝ_j·g_max/2`.
- **G4 (distributional).** For `w` iid `N(0,σ²)` blocks of `B` elements, the mean squared error of blockwise absmax rounding to `b` bits is `(1 − 1/B)·E[a_B²]/(12 qmax²)`, with `E[a_B²]` from the integral in D1. Check range: within 3% of measured for `B ≥ 64` (author's NumPy pre-check: ratios 0.985, 0.997, 0.997 for `B = 64, 256, 1024` at 4 and 8 bits; 0.94 at `B = 16` without the `(1 − 1/B)` factor).
- **Hand-off to Chapter 11/12.** 11.x imports `quantize_blockwise`/`quantize_nf4`/`dequantize` and extends them (activation quantization, calibration, GPTQ). 12.x's int4 decoder reads `codes` (packed along `in_features`, even element in the high nibble) and group scales `[out, in/block]`; a kernel can therefore process one output row as a contiguous byte stream with one scale per 64 (or 128) elements. If 11/12 need a different nibble order (e.g., the Marlin or torchao layout), they re-pack with a documented function; this section does not.

#### Narrative arc

1. **The question.** Opening cell: load `Qwen3-14B-Base` in bf16 only to count bytes (meta device; no weights): 29.5 GB of weights, which cannot train on a 24 GB card even with LoRA (adapters add about 1 GB but the base alone exceeds the card). Qwen3-8B, at 16.4 GB, leaves room for bf16 LoRA on a 24 GB card (to be measured), so it no longer makes the point. Then the same count after NF4 + double quantization of the linear layers (computed by `ft_memory`: 6.82 GB for 13.21B linear parameters at 4.127 bits, plus 3.11 GB for embeddings and `lm_head` in bf16: 9.93 GB ≈ 9.25 GiB). Symptom to print: the single transformer-block weight tensor `down_proj` (5120×17408) in bf16 vs NF4, and the peak memory of a forward pass through one dequantized block (to be measured).
2. **A quantizer is (codebook, scale granularity, scale format).** One recap paragraph on formats from `sec_hardware`/4.4, then the unification: int4 = uniform codebook; fp4 (E2M1) = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}; NF4 = quantile codebook; per-tensor, per-channel, per-block; fp32/fp16/e8m0/e4m3 scales. Cells: absmax per-tensor int8 (a 12-line quantizer); per-block int8/int4; affine grids with scale and zero point (`symmetric=False`).
3. **Measuring error.** The error report: relative Frobenius error, SQNR, max error vs G1–G3, output error `‖(W−Ŵ)X‖/‖WX‖` on 256 real activation vectors captured from the model on validation text. Table `tab_lm_quant_error` over (tensor type × layer {0, 13, 27}) × (per-tensor int8, block-64 int8, per-tensor int4, block-{16, 32, 64, 128, 256} int4, NF4 block 64, NF4 + DQ, fp4 block 32) on Qwen3-0.6B; ≈ 100 numbers, plus a whole-model perplexity column (all linear layers replaced by dequantized weights).
4. **Why per-tensor fails and blocks help.** Derivation D1; kurtosis and outlier maps of real weights; the one-outlier example (a single entry 40× the standard deviation forces `Δ` so large that the bulk rounds to zero); block-size sweep with storage (`fig_lm_quant_blocksize`: x = bits per parameter, y = relative error, one curve per scheme).
5. **NF4.** Construction from normal quantiles (offset 0.9677083, asymmetric 8+7, exact zero, normalized to [−1,1]); QQ-plot of normalized real weights against the normal; derivation D2 (equal-probability bins versus Lloyd–Max); the measured comparison of grids on Gaussian data and on real weights: int4, fp4, NF4, Lloyd–Max, and a data-fitted k-means grid.
6. **Double quantization.** Accounting D3; implementation; the measured extra error; the 8B savings in GB.
7. **Dequantize on the fly inside the matmul.** `QuantLinear`'s custom `autograd.Function`: forward `y = x Ŵᵀ + b`; backward `dx = dy Ŵ` with `Ŵ` re-dequantized, `dW = None`; the peak-memory difference between the naive module (autograd saves `Ŵ` per layer) and the custom function (measured on 8 stacked blocks; to be measured). Roofline reading (D5): decode-like shapes are bandwidth-bound and dequantization adds traffic; training shapes are compute-bound and the overhead is a fraction of step time (measured: QLoRA step time vs bf16 LoRA at 1.7B).
8. **Gradients through frozen quantized weights.** The adapted layer is `y = x Ŵᵀ + γ x Aᵀ Bᵀ`; the adapter sees the *quantized* network; the quantization error `E = W − Ŵ` is a fixed perturbation. LoftQ (D4) initializes `BA ≈ E`'s best rank-`r` approximation, iterating quantize/SVD; compared with PiSSA's residual idea (7.5) and with the default `B=0`.
9. **QLoRA on a 14B model in 24 GB.** Recipe: NF4 + DQ for linear layers, bf16 embeddings and `lm_head`, LoRA r=16 (all-linear, fp32 adapters), `chunked_lm_loss`, gradient checkpointing, micro-batch 1–2 at 512–1,024 tokens, paged 8-bit AdamW from bitsandbytes for the adapter state (and the from-scratch 8-bit optimizer on the 1.7B arm). The notebook prints the predicted vs measured peak per sequence length (512, 1,024, 2,048; to be measured), step time, tokens/s, and the validation loss before/after 100 steps. The QLoRA paper's appendix hyperparameters (LoRA r=64, α=16, learning rate 2e-4 for ≤ 13B models; the author reads the table) are shown next to the notebook's own choices.
10. **8-bit and paged optimizers.** Adam state at 8 bits: block-wise quantization of `m` (signed) and `√v` (unsigned) with block 2,048; failure of naive linear quantization of `v` (small entries round to 0 and the update explodes), shown on a toy problem; fix by quantizing `√v` and flooring at one code; bitsandbytes' dynamic map as the production answer. Paged optimizers: unified-memory allocation of the state with eviction under memory pressure; demonstrated with `bnb.optim.PagedAdamW8bit` on an artificial memory spike (not from scratch).
11. **QLoRA vs BF16 LoRA, with and without LoftQ** (headline experiment). Merge/export: `merged_weight()` returns dequantized + delta in bf16; the notebook evaluates (a) quantized base + adapter, (b) merged bf16 export, (c) merge then re-quantize, and reports all three with the same data (the standard QLoRA deployment trap).
12. **Hand-off.** Table of what this section does *not* cover and where it goes: activation quantization and outliers (11.2), error-aware rounding (11.3), rotations (11.4), QAT with straight-through estimators (11.5), int4 kernels and speed (12.x).

#### Derivations

**D1. Uniform quantization error `Δ²/12` and blockwise absmax (`eq_lm_quant_mse`).**
Statement: for a quantizer `Q(x) = Δ·round(x/Δ)` and a signal whose density is approximately constant over each cell, the error `e = x − Q(x)` is uniform on `[−Δ/2, Δ/2]` and `E e² = Δ²/12`. For absmax block quantization with `b` bits, `qmax = 2^{b−1} − 1`, block size `B` and iid `N(0,σ²)` entries, `E e² = (1 − 1/B)·E[a_B²]/(12 qmax²)` with `E[a_B²] = ∫₀^∞ 2t (1 − erf(t/√2/σ)^B) dt` (in units of `σ`), which grows like `2σ² ln B` to leading order.
Assumptions: high-resolution regime (`Δ ≪ σ` within a cell; marginal for `b=4`, tested); absmax element reproduced exactly.
Steps:
1. Cell geometry: `x` in cell `[kΔ − Δ/2, kΔ + Δ/2)` maps to `kΔ`; `e = x − kΔ`.
2. If the density is constant on the cell, `e ~ U(−Δ/2, Δ/2)`.
3. `E e² = (1/Δ)∫_{−Δ/2}^{Δ/2} e² de = Δ²/12`.
4. Bit-width: a range `[−a, a]` with `2^b` levels has `Δ = 2a/2^b` ⇒ `E e² = a²/(3·4^b)`; each extra bit divides the MSE by 4 (6.02 dB).
5. Block absmax: `Δ_j = a_j/qmax`; given the block, the conditional MSE is `Δ_j²/12`; averaging over blocks gives `E[a_B²]/(12 qmax²)`.
6. The block's own absmax element has zero error (it sits at `±qmax·Δ_j`), so one of `B` elements contributes 0: the factor `(1 − 1/B)`.
7. Extreme-value statistics: `P(a_B ≤ t) = erf(t/(σ√2))^B`, so `E[a_B²] = ∫ 2t(1 − P(a_B ≤ t)) dt`; for `σ=1`: 4.549 (`B=16`), 6.914 (64), 9.404 (256), 11.958 (1024) (author's quadrature; the cell recomputes). Hence MSE `∝ ln B`: bigger blocks cost accuracy logarithmically, and cost fewer scale bits `s/B` linearly.
8. Outlier failure of per-tensor absmax: one entry `M ≫ σ` gives `Δ = M/qmax`; if `Δ/2 > ~3σ` every bulk element rounds to 0 and MSE → `σ²` (SQNR → 0 dB). For `M = 40σ`, `b=8`: `Δ/2 = 0.157σ`, fine; for `b=4`: `Δ/2 = 2.86σ`, nearly all zero.
Check cell: (i) Monte Carlo of uniform rounding `E e² ≈ Δ²/12` to 1%; (ii) `G4` vs measured for `B ∈ {16, 64, 256, 1024}` and `b ∈ {4, 8}`; (iii) synthetic tensor with one `40σ` outlier: per-tensor vs block-64 error at 4 and 8 bits (steps 8).

**D2. NF4: equal-probability bins versus Lloyd–Max (`eq_lm_nf4`).**
Statement: for `X ~ N(0,1)` and `k = 2^b` codes, (a) the quantizer with cell boundaries `Φ^{−1}(i/k)` gives each code probability `1/k` and therefore maximizes the code entropy at `log₂ k` bits; (b) the MSE-optimal quantizer (Lloyd–Max) has levels at conditional means `c_i = (φ(a_i) − φ(a_{i+1}))/(Φ(a_{i+1}) − Φ(a_i))` and boundaries at midpoints `a_i = (c_{i−1}+c_i)/2`, with strictly smaller MSE and non-uniform cell probabilities; (c) NF4 takes the equal-probability idea, then (i) forces an exact zero by using 8 positive and 7 negative quantiles, and (ii) rescales to `[−1, 1]` so it can be applied after absmax normalization.
Steps:
1. Entropy of a discrete code `H = −Σ p_i log₂ p_i ≤ log₂ k`, equality iff `p_i = 1/k` (Jensen / Gibbs); equal-probability cells attain it.
2. Boundaries `a_i = Φ^{−1}(i/k)` give `P(cell_i) = 1/k`.
3. Reconstruction at the conditional mean of each equal-mass cell gives MSE `1 − Σ_i c_i²/k` (for unit variance).
4. Lloyd–Max: for fixed levels the best boundaries are midpoints; for fixed boundaries the best levels are conditional means; iterating decreases MSE monotonically; the fixed point satisfies both.
5. Gaussian, `k=16` (author's NumPy pre-check, stdlib `NormalDist`): Lloyd–Max MSE `0.0095σ²`, code entropy 3.77 bits; equal-probability cells with centroid levels MSE `0.0222σ²`, entropy 4.00 bits. So the equal-probability grid is information-optimal and about 2.3× worse in MSE.
6. NF4 grid (offset `δ = 1 − ½(1/32 + 1/30) = 0.9677083`): positive levels `Φ^{−1}(linspace(δ, 0.5, 9)[:-1])`, negative levels `−Φ^{−1}(linspace(δ, 0.5, 8)[:-1])`, plus 0, divided by the max magnitude; values `−1, −0.6962, −0.5251, −0.3949, −0.2844, −0.1848, −0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1` (matches the bitsandbytes lookup table in `csrc/kernels.cu` to all printed digits).
7. After absmax normalization the data are *not* `N(0,1)`: a block of 64 has `E[a] ≈ 2.59σ` while NF4's normalization implicitly assumes `a ≈ 1.85σ` (the outermost quantile at `δ`). The relevant comparison is blockwise: Gaussian data, `B = 64` (author's pre-check, MSE in units of `σ²`): int4 `0.0116`, NF4 `0.0084`, a k-means grid fitted to the normalized samples `0.0076`. So NF4 beats uniform int4 by about 27% and a fitted grid beats NF4 by about another 10%; NF4 is a good data-independent default, not an optimum.
Check cell: compute Lloyd–Max by iteration (200 iterations) and the equal-probability grid; print MSE and entropy; verify the centroid formula by Monte Carlo; reproduce the 16 NF4 values and compare with `NF4_CODEBOOK` and with `bitsandbytes`' table (`bitsandbytes.functional.get_4bit_type('nf4')` if available; exact to 1e-7); then the blockwise comparison on Gaussian and on real Qwen3 weights.

**D3. Double quantization accounting and error (`eq_lm_dq`).**
Statement: with first-level block `B`, second-level block `B₂`, `s₁` bits per first-level scale and `s₂` bits per second-level scale (plus a per-tensor offset), the cost per parameter falls from `b + s₁/B` to `b + 8/B + s₂/(B·B₂)`. For `b=4, B=64, s₁=32, B₂=256, s₂=32`: `4.5 → 4.127`, a saving of `0.373` bits/param; the error obeys G3.
Steps: (1) count scale bits per parameter; (2) second level stores 8 bits per first-level scale (`8/B` per parameter) plus one fp32 absmax per `B₂` scales (`32/(B·B₂)`); (3) subtract: `32/64 − 8/64 − 32/16384 = 0.5 − 0.125 − 0.00195 = 0.373`; (4) for 6.95B linear parameters (Qwen3-8B), `0.373·6.95e9/8 = 0.324 GB`; (5) error: `w = a·n`, `ŵ = ŝ·c(n)`, so `w − ŵ = (a − ŝ) n + ŝ (n − c(n))`, and `|n| ≤ 1`, giving G3; (6) why subtract the mean: scales are positive, a symmetric code would waste half its range.
Check cell: `bits_per_param` from `QTensor.nbytes` equals 4.127 within 1e-3; G3 bound holds on all blocks of a real tensor; extra reconstruction error relative to no-DQ reported (to be measured).

**D4. LoftQ: alternating quantization and low-rank fit (`eq_lm_loftq`).**
Statement: LoftQ seeks `Q` (quantized) and rank-`r` `BA` minimizing `‖W − Q − BA‖_F`. Alternating: `Q_t = quant(W − B_{t−1}A_{t−1})`, `B_tA_t = SVD_r(W − dequant(Q_t))` (with `B_0A_0 = 0`).
Steps: (1) for fixed `Q`, the best rank-`r` `BA` is the truncated SVD of the residual (Eckart–Young), so the second half-step is optimal; (2) for fixed `BA`, nearest-code rounding is optimal for fixed scales and grid, but the absmax scales are recomputed from `W − BA`, so the first half-step is not guaranteed optimal; (3) therefore monotone decrease is guaranteed for step (1) only, and the notebook checks monotonicity of the full sequence empirically and reports exceptions; (4) with `B=0` (standard QLoRA) the initial error `‖W − Q‖_F` is the plain quantization error; LoftQ's initial error is `‖W − Q_t − B_tA_t‖_F ≤ ‖W − Q_t‖_F` (since the truncated SVD can only reduce the residual); (5) adapter scaling: with `γ = α/r`, set the stored factors to `B_t, A_t/γ`-split so that `γ·BA` equals the SVD factor product.
Check cell: errors per iteration for 20 weights; non-increasing with the exceptions counted; initial adapted-model logit KL vs the full-precision model for `B=0`, PiSSA-style (`W_res` quantized), and LoftQ with `iters ∈ {1, 5}` (to be measured; direction: LoftQ ≤ others).

**D5. What on-the-fly dequantization costs (`eq_lm_dequant_cost`).**
Statement: for an `m×n` weight and a batch of `T` tokens, bf16 matmul moves `2mn + 2T(m+n)` bytes and does `2mnT` FLOPs; a naive dequantize-then-matmul at 4.127 bits moves `0.516mn` (read codes+scales) `+ 2mn` (write `Ŵ`) `+ 2mn` (read `Ŵ`) `+ 2T(m+n)`, i.e. about `4.5mn` bytes of weight traffic; a fused kernel moves `0.516mn`.
Steps: (1) count bytes per stage; (2) arithmetic intensity `2T/bytes_per_weight`: for bf16 `T` FLOP/B, for naive QLoRA `2T/4.5 ≈ 0.44T`, for fused `3.9T`; (3) compare with the ridge point of the GPU (`sec_perf_model`, `tab_gpu_specs`: ≈ 165 FLOP/B for the RTX 4090 in bf16): training with `T = 1,024` tokens per micro-batch is compute-bound in all cases (intensity ≥ 450 FLOP/B), so the overhead is the transient write/read of `Ŵ` as a fraction of the matmul time, not a bandwidth wall; decode with `T=1` is bandwidth-bound, so time follows weight bytes: the naive path moves `4.5mn` against bf16's `2mn` (about 2.25× slower), while a fused kernel moves `0.516mn` (about 3.9× fewer bytes than bf16; a bound, not a promised speed-up); (4) the transient dequantized weight (`100.7 MB` for `down_proj` in the 8B model: `4096·12288·2 B`) is the workspace to budget in `ft_memory`.
Check cell: time `dequantize(q)` alone and the matmul alone for `down_proj` shapes of 1.7B and 8B (`d2l.Benchmark`); compare with the byte prediction using `d2l.chip_specs()`'s bandwidth for the build box's GPU (column `mem_bw_tbs`, 5.1; one-line fallback: the vendor datasheet value typed in with its citation if the build box's GPU is not in the table); report the ratio of measured to predicted (to be measured).

#### From-scratch implementation

Budget: core ≈ 150 lines (exported); optional extras ≈ 45 lines (LoftQ 20; 8-bit AdamW 25 in a notebook cell, not exported). Total ≈ 195, over the ~150 guideline; the notebook author may move `quantize_codebook`'s double-quant branch and the 8-bit optimizer into collapsible "optional" cells.
- `QTensor` (25 lines: dataclass, `nbytes`, `bits_per_param`, `_pack4`, `_unpack4`).
- `quantize_blockwise` (30 lines): reshape to `[*lead, n/B, B]`; `a = abs(w).amax(-1)`; `s = a/qmax` (asymmetric: `s = (max − min)/(2^b − 1)`, `z = round(−min/s)`); codes `round(w/s).clamp(−qmax, qmax)` (computed in fp32, scales cast to `scale_dtype` *before* division so the dequantizer and quantizer agree); per-tensor mode when `block_size=None`; pack 4-bit.
- `quantize_codebook` / `quantize_nf4` (30 lines): normalize by block absmax; nearest codebook entry by `argmin |x − c|` computed through midpoints and `torch.bucketize` (ties go to the lower index; identical to bitsandbytes' decision tree); optional double quantization (flatten scales, subtract mean, int8 blocks of 256).
- `NF4_CODEBOOK`: built from `statistics.NormalDist().inv_cdf` at import (the notebook shows the 6-line construction and asserts equality with the hard-coded 16 values).
- `dequantize` (20 lines): unpack nibbles, gather from codebook or scale integers, multiply by scales (dequantize scales first when double-quantized), reshape, cast; computed in fp32 then cast to `dtype`.
- `QuantLinear` + `_QuantMatmul(torch.autograd.Function)` + `from_linear` + `quantize_model` (45 lines): the custom function saves `codes`, scales and `x`'s shape only; backward re-dequantizes.
- `loftq_init` (20 lines) and `quant_error_report` (15 lines).
- **Numerical tests (reference, tolerance, seed 0).**
  1. Pack/unpack round trip on random 4-bit codes: exact.
  2. G1–G3 bounds hold on every block of a random Gaussian tensor `[256, 1024]` and of three real Qwen3 weights; zero and absmax elements exact.
  3. D1 `E e² = Δ²/12` to 1%; G4 within 3% for `B ≥ 64`.
  4. bitsandbytes oracle: `bnb.functional.quantize_4bit(w, blocksize=64, quant_type='nf4', compress_statistics=False)` then `dequantize_4bit`: ≥ 99.99% of codes identical, and dequantized values bit-identical on identical codes (fp32 scales, bf16 input cast to fp32 first); with `double_quant=True` agreement of dequantized weights within a relative Frobenius difference of 5e-3 (second-level code differs; tolerance to be calibrated); the nibble order assertion is a one-line byte comparison.
  5. torchao `NF4Tensor` oracle (pure PyTorch; `block_size=64`, `scaler_block_size=256`), same tolerances as (4) with double quantization (to be confirmed in the pinned release).
  6. `QuantLinear` forward equals `F.linear(x, dequantize(q), b)` exactly; backward `dx` equals autograd of the dense dequantized layer to 1e-5 (fp32); weight gradient is `None`; frozen quantized buffers unchanged after 10 optimizer steps (byte equality of `codes`); adapter gradients finite.
  7. Memory: 8 stacked `QuantLinear` blocks: custom function's peak < naive peak (assert the ordering, report the ratio).
  8. LoftQ: first-iteration error ≤ plain quantization error on 20 weights.

#### In production

- **bitsandbytes via Transformers/PEFT:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`; `prepare_model_for_kbit_training`; `LoraConfig(target_modules='all-linear')`; `optim='paged_adamw_8bit'`. The from-scratch `QuantLinear` + `LoRALinear` model is the oracle: identical adapter weights and the same quantized base (from bnb's `Params4bit` codes when comparing) must give logits within 1e-2 relative (bf16; to be calibrated); the section prints the largest discrepancy and where it comes from (different dequantization order, compute dtype of the matmul, double-quantization code).
- **PEFT quantization guide** (verified 200) for LoftQ (`init_lora_weights='loftq'`, `LoftQConfig`): same from-scratch comparison on 20 weights.
- **Other libraries:** `torchao` (NF4 tensors; quantized training paths; torchtune's QLoRA), FSDP-QLoRA packing (7.8), Unsloth's pre-quantized checkpoints (7.7).
- **Export:** adapters + original base; or merged bf16 (`merged_weight`); not "merge into NF4".

#### Headline experiment

- **Hypothesis.** (H1) NF4 with double quantization is within noise of BF16 LoRA on validation loss for 1.7B on this SFT task while cutting base-weight memory by ≈ 3.5× in the linear layers; (H2) NF4 gives lower weight and output error than int4 at equal block size and storage, and the validation-loss gap between QLoRA-int4 and QLoRA-NF4 follows it in direction; (H3) LoftQ initialization reduces the *initial* adapted-model loss gap to the full-precision model and may speed up early training, with a smaller or no advantage at the end of training; (H4) a 14B QLoRA run fits a 24 GB GPU at sequence 512–1,024 (whether it also fits a 15 GiB slot is measured).
- **Setup.** (a) 14B: `Qwen3-14B-Base` on a whole 24 GB GPU, NF4+DQ, LoRA r=16 all-linear, `chunked_lm_loss`, gradient checkpointing, micro-batch 1–2, ≤ 100 steps, paged 8-bit AdamW; record `torch.cuda.max_memory_allocated` and `max_memory_reserved`. (b) 1.7B, four arms, identical data order, steps (≈ 180 steps of 8k tokens), seeds {0,1,2}, `lr` from 7.4's optimum for LoRA (same for all arms; one ×3 check): BF16 LoRA; QLoRA NF4+DQ (`B=0`); QLoRA NF4+DQ + LoftQ (`iters=5`); QLoRA int4-absmax block-64 (a worse quantizer on purpose).
- **Procedure.** Quantize once per arm; measure weight-space error on all linear layers (relative Frobenius; per module type); initial validation loss; train; final validation loss; evaluate each arm in the three deployments (quantized+adapter, merged bf16, merge-then-requantize).
- **Metrics.** Validation loss at step 0, at 50, final; peak memory (GiB); step time; error tables; merged-model loss delta.
- **Plot spec.** `fig_lm_qlora_arms`: x = step (0–180), y = validation loss (evaluated every 20 steps), one line per arm with 3-seed bands; `fig_lm_quant_blocksize` (x = bits per parameter, y = relative error); `fig_lm_qlora_memory`: stacked bars (weights, adapters, optimizer, activations, logits) predicted vs measured for 1.7B BF16 LoRA, 1.7B QLoRA, 8B QLoRA at sequence 512/1,024/2,048.
- **Expected outcome.** From sources, directionally: the QLoRA paper reports that 4-bit NF4 with double quantization recovers 16-bit LoRA fine-tuning performance on its benchmarks and that NF4 beats FP4 and int4 in their zero-shot comparisons, and the k-bit scaling-laws paper reports 4-bit as the usual bit-precision optimum for zero-shot accuracy at a given memory budget; LoftQ's paper reports gains over QLoRA especially at 2-bit and mixed precision. Our magnitudes are to be measured; H3's size is the least certain.
- **Noise check.** 3 seeds (adapter init only; quantization is deterministic); bands drawn; for the arm comparisons the paired difference (same seed, same data order) with the seed standard deviation; no claim of difference unless it exceeds two standard deviations. For H4, a single deterministic run suffices (memory is deterministic up to allocator fragmentation; report both allocated and reserved).
- **T1 reduction.** Native; the 14B arm is the heavy cell and claims a whole 24 GB GPU; if it does not fit, fall back to `Qwen3-8B-Base`, whose static weights (5.65 GiB) also fit a 15 GiB slot, and report the largest fitting model with its measured peak (the 14B headline supersedes the claim in v5's notebook list, "an 8B model within 15 GiB", which the 8B fallback is expected to meet; to be measured).
- **Artifacts committed.** `lm-ch07-quant-errors`, `lm-ch07-qlora-runs`, memory snapshots (JSON of per-stage allocated bytes).

#### Figures and tables

- `tab_lm_quant_bits`: bits per parameter for each scheme (above; generated from `QTensor.nbytes`).
- `tab_lm_quant_error`: error table (tensor type × layer × scheme).
- `fig_lm_quant_blocksize`; `fig_lm_quant_qq` (normalized real weights vs normal quantiles, per tensor type); `fig_lm_quant_grids` (the 16 levels of int4, fp4, NF4, Lloyd–Max, k-means on one axis, with the Gaussian density).
- `fig_lm_nf4_entropy`: bar chart of code probabilities for equal-probability vs Lloyd–Max cells.
- `fig_lm_qlora_arms`, `fig_lm_qlora_memory`, `tab_lm_qlora_deploy` (three deployments).
- `tab_lm_quant_handoff` (what Chapters 11–12 pick up).

#### Pitfalls and misconceptions

- "4-bit means 4 bits per parameter": scales cost 0.125–0.5 bits; the notebook prints `bits_per_param` from actual bytes.
- NF4 is "optimal": it is entropy-optimal for Gaussian data and not MSE-optimal; after block normalization a fitted grid does slightly better (D2).
- Dequantizing in the model's forward with plain autograd saves the dequantized weights for backward and erases the memory benefit (D5, step 7 of the narrative).
- Embeddings and `lm_head` are left unquantized by default; at 8B they are 2.49 GB (bf16), 41% of the quantized model's weights.
- Evaluating the adapter on the quantized base is not the same as evaluating the merged bf16 export; report both.
- Scale dtype matters: fp16 scales saturate for tensors with absmax above 65,504 (not weights, but relevant for activations in 11.2) and bf16 scales cost accuracy (G1).
- Nibble order and block axis: another library may pack along a different axis or order; assert on bytes.
- bitsandbytes' double quantization uses a dynamic 8-bit code; ours is linear int8, so DQ results agree only within tolerance.
- LoftQ's initialization changes the adapter's starting point and thus the learning-rate sensitivity (revisit 7.4's scaling discussion).

#### Summary (draft bullets)

- Quantization stores a weight as codes from a small codebook plus per-block scales; int4, fp4 and NF4 differ only in the codebook.
- Uniform rounding error is `Δ²/12`; per-block absmax scales cap `Δ` by the block's largest element, so smaller blocks cost scale bits and buy accuracy; outliers ruin per-tensor scales.
- NF4's bins hold equal probability under a normal distribution; Lloyd–Max minimizes MSE instead; blockwise normalization changes which is better.
- Double quantization cuts scale overhead from 0.5 to 0.127 bits per parameter.
- QLoRA keeps the frozen base in 4 bits, dequantizes transiently inside a custom backward-aware matmul, and trains full-precision adapters; a 14B model, whose bf16 weights (29.5 GB) exceed a 24 GB card, trains on one.
- Evaluate quantized+adapter and merged exports separately.

#### Exercises

1. *Check.* How many bytes does a 4096×12288 `down_proj` occupy in bf16, NF4 (block 64, fp32 scales) and NF4+DQ? What is its dequantized transient size? *Sketch:* `50,331,648` params; bf16 `100.66 MB`; NF4 `50.33M·4.5/8 = 28.31 MB`; NF4+DQ `50.33M·4.127/8 = 25.96 MB`; transient bf16 `100.66 MB`.
2. *Check.* A tensor row has 63 values with `|x| ≤ 1` and one value 50. For block 64 int8 per-block absmax, what are `Δ` and the error of a typical small value? What if the block is split into 16-element blocks? *Sketch:* `Δ = 50/127 = 0.394`, so values below 0.197 round to 0 (error up to 0.197); with block 16 only that one 16-block has the large `Δ`.
3. *Derive.* Show that the SQNR of a `b`-bit uniform quantizer on a full-scale sine wave is `6.02b + 1.76` dB, and explain what changes for a Gaussian signal clipped at `a = 4σ`. *Sketch:* sine power `a²/2`, noise `Δ²/12 = a²/(3·4^b)`, ratio `1.5·4^b`; Gaussian: signal `σ²`, noise `a²/(3·4^b)` with `a=4σ`, ratio `4^b·3/16`, i.e. `6.02b − 7.27` dB.
4. *Derive.* Prove the 2-way split `w − ŵ = (a − ŝ)n + ŝ(n − c(n))` and use it to bound the double-quantization error, then find the second-level block size `B₂` at which the extra `8/B` bits and the error bound trade off for a given weight scale spread. *Sketch:* triangle inequality; the second-level error scales with the spread of scales inside a `B₂`-block; larger `B₂` lowers bits (`32/(B·B₂)`) but raises spread.
5. *Measure.* Predict the relative Frobenius error of NF4 vs int4 at block sizes 32, 64, 128 on `gate_proj` of layer 13 and the share that is due to the scales being stored in fp16 rather than fp32; then measure.
6. *Measure.* Make the 8-bit optimizer store `v` linearly in 8 bits (not `√v`); predict the step at which the loss diverges on a toy problem; measure and explain through the zero code.
7. *Open.* Replace the NF4 codebook by the k-means grid fitted to normalized blocks of one layer type; measure the error and whether `fine-tuning` quality changes; discuss why shipping per-layer codebooks complicates kernels.

#### Further reading

- **Papers:** [QLoRA](https://arxiv.org/abs/2305.14314) (NF4, double quantization, paged optimizers; appendix for hyperparameters); [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861); [LoftQ](https://arxiv.org/abs/2310.08659); [The case for 4-bit precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720); [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) (new, verified; affine grids, scale/zero point, granularity); [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) (new, verified; the original affine integer quantization). Lloyd (1982) and Max (1960) for the optimal scalar quantizer: cited by title (`Least squares quantization in PCM`; `Quantizing for minimum distortion`), no stable open landing page (unverified).
- **Tutorials:** Maarten Grootendorst, [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization); Hugging Face [bitsandbytes 4-bit blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes) (verified 200); [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) (verified 200).
- **Code:** [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes); [QLoRA reference code](https://github.com/artidoro/qlora) (verified 200); [torchao](https://github.com/pytorch/ao).

#### Risks and open decisions

- **The 14B run on 24 GB is the least certain claim.** The static weights (≈ 9.25 GiB) leave about 14 GiB of the card for adapters (1.0 GB at fp32 with Adam for 64.2M LoRA parameters), activations and logits; likely feasible at sequence ≤ 1,024 with micro-batch 1 and checkpointing, possibly 2,048; the measured peak decides, including whether the run also fits a 15 GiB slot. Fallback: `Qwen3-8B-Base` (static weights ≈ 5.7 GiB) and a note in the artifact.
- **bitsandbytes on the Ada box and the pinned `torch==2.15.0`:** if the wheel is unavailable, the oracle tests (4) and the paged optimizer demonstration are skipped and replaced by the torchao oracle (5) and a documented CPU-offload analogue.
- **Double quantization mismatch with bnb:** by design (linear int8 vs dynamic code); the contract states the tolerance; if the editor prefers bit-exactness, implement bnb's dynamic map (≈ 25 more lines).
- **Layout choice (last-axis blocks, even-high nibble)** is the contract for Chapters 11/12; any change after those blueprints are written must be negotiated through the editor.
- **fp16 scales** are in the table for int4 g128 only; the default is fp32 for bnb parity; Chapter 11 owns the inference-oriented choices.
- **Line budget** exceeds 150 (see above); an editorial cut would move LoftQ and the 8-bit optimizer to exercises.

### 7.7 Fine-tuning toolchains

- **Label:** `sec_lm_ft_toolchains` · **Tier:** T1 (live arms: the from-scratch oracle and PEFT+TRL; the Unsloth, torchtune, Axolotl and LLaMA-Factory arms are produced by committed scripts in pinned side environments and analyzed from committed logs) · **Budget:** notebook ≤ 15 min (oracle and PEFT+TRL runs of 20 steps ≈ 2 min each, the ablation table ≈ 8 runs of ≈ 40 s), peak ≤ 12 GiB; the side-environment jobs are T1 offline jobs of ≈ 5 min each · **Codex units:** new (v5); touches 8.2/8.5 material only through the shared recipe.
- **Builds on (book):** `sec_compilation` (`torch.compile` as one of the throughput levers), `sec_perf_model` (`d2l.Benchmark`, tokens/s and bytes), `sec_read_write`/`sec_repro` (checkpoints, seeds, determinism). Nothing re-taught.
- **Builds on (this part):** 7.4 (`LoRALinear`, `add_lora`, `lora_param_groups`; the oracle), 7.2 (`render_chat`, `sft_loss`, `pack_examples`, `block_causal_mask`), 7.1 (`chunked_lm_loss`), 7.6 (`QuantLinear` for the one QLoRA row), 1.1 (`d2l.load_hf` and revision pinning).
- **Recipe R7 (the single recipe every toolchain must express):** `Qwen/Qwen3-0.6B-Base`; LoRA `r=16`, `α=32`, dropout 0, all-linear targets; AdamW `β=(0.9, 0.999)`, `ε=1e-8`, weight decay 0, **no gradient clipping** unless stated; learning rate = the 7.4 optimum for `r=16` all-linear; 3% linear warmup then linear decay to 0; global batch = 8 rows of 1,024 tokens, one micro-batch (no accumulation) in the main comparison and 4 micro-batches of 2 rows in the accumulation test; bf16 base, fp32 adapters; loss = token mean over supervised tokens of the effective batch; data = the first 20 batches of `lm-sft-mix-v0` pre-tokenized with `render_chat` (identical `input_ids`/`labels` for all toolchains); seed 0.
- **Exports (`#@save`):**
  - `d2l.fixed_batches(tokenizer, conversations, n_batches=20, rows_per_batch=8, max_len=1024, seed=0) -> list[dict]`: deterministic padded batches with `input_ids`, `attention_mask`, `labels` (−100 on masked and pad), shared by all arms.
  - `d2l.copy_lora_weights(src, dst, name_map='peft') -> int`: copies `lora_A`/`lora_B` from a model patched with `add_lora` into a PEFT model (`...lora_A.default.weight`, `...lora_B.default.weight`; also an `'unsloth'` map, which is PEFT's) and returns the number of tensors copied; asserts equal shapes.
  - `d2l.compare_loss_traces(a, b, rtol=1e-2, atol=0.0) -> DataFrame`: per-step relative difference, the first step exceeding tolerance, and a verdict.
  - `d2l.defaults_table(objs: dict[str, object], keys: Sequence[str]) -> DataFrame`: reads the named attributes (e.g. `max_grad_norm`, `weight_decay`, `adam_epsilon`, `warmup_ratio`, `lr_scheduler_type`, `bf16`, `gradient_checkpointing`, `packing`) from configuration objects of different libraries, with `—` where absent; the source of `tab_lm_toolchain_defaults`.
  - `d2l.throughput(step_fn, tokens_per_step, warmup=3, repeats=5) -> dict`: median tokens/s, min, max, peak allocated and reserved GiB, using `d2l.Benchmark` semantics (synchronize before timing).
  - Artifacts: `DATA_HUB['lm-ch07-toolchain-configs']` (the YAML/py files: `trl_sft.py`, `unsloth_sft.py`, `torchtune_lora.yaml`, `axolotl.yml`, `llamafactory.yaml`); `DATA_HUB['lm-ch07-toolchain-logs']` (JSONL traces and throughput per arm, with the full `pip freeze`); `data/ch07_requirements.txt` and the per-toolchain pin files.
- **Models and data:** `Qwen/Qwen3-0.6B-Base` (Apache-2.0); `lm-sft-mix-v0` (7.2; Apache-2.0 source).
- **Packages and candidate pins** (the latest PyPI releases on 2026-10-28, listed so the author starts from known-existing versions; they must be resolved together on the build box and frozen into `data/ch07_requirements.txt`): `transformers 5.21.0`, `trl 1.16.0`, `peft 0.21.2`, `accelerate 1.15.0`, `datasets 5.1.0`, `bitsandbytes 0.50.2`, `liger-kernel 0.8.5`, `flash-attn 2.8.3.post2` (optional), `unsloth 2026.10.10` (own environment), `axolotl 0.21.1` (own environment, Python ≥ 3.12), `llamafactory 0.9.5` (own environment), `torchtune 0.6.1` (own environment; the last release, April 2025), `lm-eval 0.4.13`, `mergekit 0.1.5` (7.9), `torchao 0.18.0`. Every notebook runs on `torch==2.15.0` and Python 3.12 (the book's pin), so the packages above are resolved together against 2.15.0 on the build box and frozen. Libraries with their own `torch` pins get their own environments, each recorded in the logs it produces. PyPI metadata read on 2026-10-29: `unsloth 2026.10.10` requires `torch<2.15.0,>=2.4.0` and `axolotl 0.21.1` requires `torch<2.15.0,>=2.13.0`, so neither can join the notebook environment; `llamafactory 0.9.5` (`torch>=2.4.0`) and `torchtune 0.6.1` (`torchdata==0.11.0`) are kept in their own environments too, whether or not they would resolve; `peft 0.21.2`, `accelerate 1.15.0`, `liger-kernel 0.8.5`, `mergekit 0.1.5` and `bitsandbytes 0.50.2` (`torch<3,>=2.4`) state no upper bound below 2.15.0, and `trl 1.16.0`, `transformers 5.21.0` and `lm-eval 0.4.13` state no `torch` requirement.

#### Narrative arc

1. **The question.** Opening cell: run R7 for 20 steps in the from-scratch loop and in `SFTTrainer` with library defaults on identical batches and print the two loss traces with `compare_loss_traces`. Symptom (to be measured): the traces differ from step 0 or step 1 even though the recipe "is the same"; print the first divergence step and the relative difference. The section is the bisection of that difference.
2. **One recipe, five dialects.** Print R7 side by side: `trl_sft.py` (`SFTConfig`, `LoraConfig`), `unsloth_sft.py`, `torchtune_lora.yaml`, `axolotl.yml`, `llamafactory.yaml` (from the artifact; trimmed to the lines that express the recipe). Configuration as experiment record: what must be in a record (library versions, model revision, data hash, seed, effective batch, schedule, every non-default knob) and what tools exist for it (the YAML itself; `trainer_state.json`; `trainer.args` dumps). A "run card" JSON produced by the notebook (`d2l.defaults_table` + `pip freeze`).
3. **The difference ledger.** `tab_lm_toolchain_defaults`: the same ten knobs across the available config objects: gradient clipping (`max_grad_norm`), weight decay, Adam `ε`, scheduler type and warmup definition (ratio vs steps; off-by-one at the end of warmup), `bf16`/autocast versus pure-bf16 parameters, gradient-checkpointing implementation, loss reduction under accumulation (per-microbatch mean vs `num_items_in_batch`), padding side, dropout in LoRA, LoRA `α` convention, whether adapters are fp32. Values are extracted programmatically from the pinned libraries (never typed by hand), so the table is a test as well as an exhibit.
4. **Bisecting to agreement.** The procedure and the expected narrative: (a) copy the oracle's LoRA weights into the PEFT model (`copy_lora_weights`), so initialization cannot differ; (b) compare the step-0 loss (no update has happened yet): any difference is a *forward* difference (masks, dropout, dtype, attention backend); (c) align the optimizer and schedule knobs one at a time, re-running 5 steps after each and logging the change of step-5 loss; (d) align loss normalization under accumulation (reusing 7.2 D1); (e) show agreement over 20 steps within a stated tolerance (to be calibrated; target `rtol=1e-2` per step in bf16, or the seed-to-seed noise band if larger). `tab_lm_toolchain_bisect`: one row per knob, with the measured effect size.
5. **A second toolchain.** The same bisection against Unsloth from its committed trace: its patches (fused RMSNorm/RoPE/cross-entropy, LoRA kernels, its own `train_on_responses_only` masking helper) are listed; the first divergence is attributed (e.g. different loss reduction or masking of the end-of-turn token; to be measured); the section never asserts a cause without an ablation that removes it.
6. **Configuration-driven wrappers.** Axolotl and LLaMA-Factory take YAML and generate the above objects; the notebook loads their *resolved* configs from the committed logs (they print the effective `TrainingArguments`), checks the ten knobs against R7, and reports any default that moved. One smoke run per wrapper is committed (20 steps; loss trace compared with the same tolerance).
7. **torchtune.** Its config format and recipes (`lora_finetune_single_device`, QLoRA, full-finetune) are read as design documents; its README states it is no longer actively maintained and its last PyPI release is April 2025, so the arm is analysis of a committed log from a frozen environment, if that environment runs on the build box, and config-only otherwise. The section uses it to show *why* a hackable recipe format was worth having and where each knob of R7 lives in it.
8. **Where throughput comes from.** Ablation ladder on the live PEFT+TRL arm: (i) defaults; (ii) `attn_implementation='sdpa'` vs `'flash_attention_2'` (if installable); (iii) `use_liger_kernel=True` (fused linear cross-entropy and fused norms; the Liger paper reports on average 20% higher throughput and 60% lower memory against Hugging Face implementations; we measure); (iv) `packing=True`/`padding_free` (7.2 D3); (v) gradient checkpointing on/off; (vi) `torch_compile=True`. Then the from-scratch loop (7.1/7.2 kernels: `chunked_lm_loss`) as a reference point and Unsloth from its log. `fig_lm_toolchain_waterfall`. Each factor is explained in bytes or FLOPs (fused CE avoids materializing logits: `4·T·V` bytes; packing raises the useful-token fraction `f`; checkpointing costs one extra forward). Differences smaller than the repeat range are not interpreted.
9. **Choosing and trusting a toolchain.** A decision paragraph: library for convenience and ecosystem; from-scratch for debugging; Unsloth for single-GPU speed; Axolotl/LLaMA-Factory for configuration-first teams; torchtune as a reference design. The trust rule: before using any library for a real run, reproduce the first steps of a small oracle run.

#### Derivations

There is no new theorem; this section's "derivation" is an accounting identity used to attribute effects. **D1. Effective batch and schedule bookkeeping (`eq_lm_effbatch`).**
Statement: for micro-batch rows `b`, accumulation steps `k`, data-parallel ranks `p`, sequence length `T`, the effective batch is `E = b·k·p` rows and `N_tok = Σ supervised tokens` over those rows; the optimizer's `s`-th update uses learning rate `lr_s` from the schedule evaluated at step index `s` (0- or 1-based, library-specific); total optimizer steps are `⌊n_rows/E⌋` (drop-last) or `⌈n_rows/E⌉`.
Steps:
1. Tokens per update `N_tok` depend on the mask; the loss normalizer must be `N_tok` of the *effective* batch (7.2 D1).
2. A warmup of `w` steps given as a ratio `ρ` is `⌈ρ·S⌉` in some libraries and `⌊ρ·S⌋` in others (S = total steps); a linear decay can end at 0 on the last or the next-to-last step; an off-by-one at either end changes the learning rate by `lr/S` per step: negligible individually, visible in a 20-step comparison (`S=20`: 5% per step).
3. Count steps in each library from its logs and compare with `⌊n_rows/E⌋`.
4. The step-`s` loss difference between two runs with a learning-rate difference `δ_s` is `≈ g·Δθ = δ_s‖g‖²` to first order, which is the check that attributes a difference to schedule vs loss normalization: normalization errors scale all of the gradient, schedule errors scale the step.
Check cell: from the oracle and PEFT traces compute `lr_s` and `‖g_s‖`; verify that the predicted loss difference from the schedule discrepancy alone matches the measured difference to within a stated tolerance when the schedule is the only knob left unaligned (to be measured).

#### From-scratch implementation

Budget ≈ 70 lines (the oracle itself is 7.4's code plus ≈ 40 lines of loop).
- The oracle training loop (40 lines): fixed batches, `add_lora`, `lora_param_groups`, linear warmup/decay scheduler written explicitly, token-sum loss divided by the effective-batch count, optional clipping, trace logging of `(step, lr, loss, grad_norm)`.
- `fixed_batches` (15 lines), `copy_lora_weights` (10 lines), `compare_loss_traces` (10 lines), `defaults_table` (10 lines), `throughput` (10 lines).
- **Tests.** (1) `fixed_batches` determinism: equal hashes across two calls; (2) `copy_lora_weights` returns `2·(number of adapted linears)` and, after copying, PEFT and oracle logits on batch 0 agree to 1e-3 relative (bf16 base, to be calibrated); (3) oracle reproducibility: two runs with the same seed give identical traces (bitwise on CPU for a 2-layer model in fp32; to bf16 tolerance on GPU, to be measured); (4) `compare_loss_traces` flags an injected 3% scale error at the right step; (5) `defaults_table` reads `SFTConfig().max_grad_norm` etc. without error on the pinned versions.

#### In production

- **PEFT + TRL.** `SFTConfig(output_dir, per_device_train_batch_size, gradient_accumulation_steps, learning_rate, lr_scheduler_type='linear', warmup_ratio, max_steps, bf16=True, gradient_checkpointing, packing, padding_free, assistant_only_loss | completion_only_loss, max_length, use_liger_kernel, seed, report_to='none')`, `LoraConfig(...)`, `SFTTrainer(model, args, train_dataset, peft_config)`; the TRL docs on PEFT integration and Liger integration (both verified 200 on 2026-10-28) are the references. For fixed pre-tokenized batches the notebook passes a dataset with `input_ids` and `completion_mask` (to be confirmed for the pinned TRL; fallback: a custom collator returning the shared `labels`). The oracle supplies the expected losses.
- **Unsloth.** `FastLanguageModel.from_pretrained(max_seq_length, load_in_4bit=False, dtype=torch.bfloat16)`, `get_peft_model(r, lora_alpha, target_modules, use_gradient_checkpointing='unsloth')`, then TRL's `SFTTrainer`; the committed script prints the resolved configuration. Run in its own environment.
- **Axolotl.** YAML-driven (`base_model`, `adapter: lora`, `lora_r`, `lora_alpha`, `lora_target_linear: true`, `datasets`, `sequence_len`, `micro_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `lr_scheduler`, `warmup_steps`, `sample_packing`, `special_tokens`); committed resolved config.
- **LLaMA-Factory.** YAML-driven (`stage: sft`, `finetuning_type: lora`, `lora_rank`, `lora_alpha`, `lora_target: all`, `template: qwen3`/custom, `cutoff_len`, `per_device_train_batch_size`, `learning_rate`, `lr_scheduler_type`, `warmup_ratio`). The template name must be checked token-for-token with `render_chat` (7.2): a mismatched `template` is the commonest silent difference.
- **torchtune.** `tune run lora_finetune_single_device --config qwen3/0.6B_lora_single_device` (config names to be confirmed in the frozen release); no longer maintained, so used for reading and for the one committed log.
- **How the from-scratch code is the oracle:** every library arm is judged against the oracle's `(step, loss)` trace and against the identical first-batch logits; the verdicts are printed, not asserted in advance.

#### Headline experiment

- **Hypothesis.** (H1) With library defaults, at least two knobs differ from R7 (among gradient clipping, scheduler/warmup definition, loss normalization), and aligning them reproduces the oracle's first 20 losses within tolerance; (H2) toolchain throughput differences are explained by a small number of mechanisms (fused cross-entropy, attention backend, packing), measured on the ablation ladder; (H3) the library arms reach the same step-20 loss as the oracle within the across-seed noise band.
- **Setup.** `Qwen3-0.6B-Base`, R7, batches from `fixed_batches` (20 batches); arms: oracle, PEFT+TRL (defaults), PEFT+TRL (aligned), Unsloth (aligned, logged), Axolotl and LLaMA-Factory (logged smoke runs), torchtune (logged or config-only). Throughput ladder on the PEFT+TRL arm with 3 warmup + 5 timed repeats of 20 steps. One QLoRA row (NF4+DQ via bitsandbytes vs `QuantLinear`) at the same recipe for the memory column.
- **Procedure.** As in the arc, items 4–8.
- **Metrics.** Per-step loss and relative difference to the oracle; first divergence step; step-20 loss; tokens/s (median, min, max over repeats); peak allocated and reserved GiB.
- **Plot spec.** `fig_lm_toolchain_traces`: x = step (0–19), y = loss; oracle in black, each arm a colored line, inset: relative difference to the oracle on a log axis; defaults and aligned arms of PEFT+TRL shown in two panels. `fig_lm_toolchain_waterfall`: horizontal bars of tokens/s after each cumulative toggle, with range bars; second panel peak memory.
- **Expected outcome.** No magnitudes asserted. Direction: defaults diverge, aligned runs agree, throughput ranks Unsloth/Liger arms above plain TRL on this single-GPU, short-sequence setting, to be tested rather than assumed; the Liger paper's averages (20% throughput, 60% memory) are the cited prior.
- **Noise check.** The seed-to-seed noise of the oracle (3 seeds of adapter init, dropout 0 so only `A` init varies, plus bf16 nondeterminism) defines the tolerance band for loss agreement; throughput repeats define the band for speed claims. Anything inside the band is reported as "within noise".
- **T1 reduction.** Native T1; side-environment arms use committed logs.
- **Artifacts committed.** `lm-ch07-toolchain-configs`, `lm-ch07-toolchain-logs`, run cards, pin files.

#### Figures and tables

- `tab_lm_toolchain_defaults`: ten knobs × configuration objects (generated).
- `tab_lm_toolchain_bisect`: one row per aligned knob with its effect on the step-5 and step-20 loss.
- `fig_lm_toolchain_traces`, `fig_lm_toolchain_waterfall`.
- `tab_lm_toolchain_roles`: when to use which toolchain (authored).

#### Pitfalls and misconceptions

- "Same hyperparameters" across libraries: defaults differ and some knobs mean different things (warmup ratio vs steps; `α` and `r` conventions; which tokens count).
- Comparing a library's final loss with another's without matching masks: different supervised-token sets give different token-mean losses.
- Benchmarking throughput with different sequence handling (padded vs packed) and calling it a kernel win.
- Assuming a maintained library: torchtune is no longer actively maintained; check release dates and pin.
- Unsloth's patches change numerics slightly (fused ops); agreement is to tolerance, not bitwise.
- Trusting a chat template by name (`template: qwen3`): verify ids against `render_chat`.
- Saving only the adapter without recording the base revision and the exact library versions.

#### Summary (draft bullets)

- A fine-tuning run is the recipe plus every default; two libraries running "the same recipe" usually are not.
- Reproduce the first steps against a small from-scratch oracle before trusting a toolchain, and bisect differences one knob at a time.
- Configuration files are experiment records only if they pin the resolved values, versions and data hashes.
- Throughput differences come from a few identifiable mechanisms: fused loss and norm kernels, attention backend, packing, checkpointing, compilation.
- Choose by task: convenience and ecosystem (TRL/PEFT), single-GPU speed (Unsloth), configuration-first operation (Axolotl, LLaMA-Factory), reading and teaching (torchtune).

#### Exercises

1. *Check.* HF `TrainingArguments` clips gradient norm at 1.0 by default; the oracle does not clip. After the first update, which quantity differs between the two runs if the unclipped gradient norm was 3.0? *Sketch:* the update direction is the same under SGD but scaled by 1/3; under Adam the first step is `sign`-like and nearly unaffected, but later steps differ through `m`, `v`: explain why the effect depends on the optimizer.
2. *Check.* A run has 1,000 rows, `b=2`, `k=4`, `p=1`. How many optimizer steps with drop-last? How many tokens per update if every row is 1,024 tokens but 40% are masked? *Sketch:* `E=8`, 125 steps; `8·1024·0.6 = 4,915` supervised tokens per update.
3. *Derive.* Show that dividing each micro-batch loss by `k` (instead of by the effective-batch token count) rescales a token's gradient weight by `N_tok/(k·n_j)` for a micro-batch with `n_j` supervised tokens, and state when the two coincide. *Sketch:* 7.2 D1.
4. *Derive.* Compute the logits memory saved per micro-batch by a fused linear cross-entropy for `T=8,192`, `V=151,936` in fp32, and the extra recomputation in FLOPs. *Sketch:* `4·T·V = 4.98 GB` for logits (plus the same for gradients if not fused) versus a chunk; recomputation `2·T·h·V` FLOPs for the recomputed matmul at `h=1,024` is `2.5 TFLOP`.
5. *Measure.* Predict which of the ladder's factors gives the largest throughput gain at sequence 1,024 and micro-batch 8, rank the six factors, then measure; compare your rank with the ladder.
6. *Measure.* Change the LoRA dropout from 0 to 0.05 in two libraries and predict whether the traces still agree step-wise; measure and explain (the random-number streams differ).
7. *Open.* Write the run card for one of your own fine-tuning runs (versions, hashes, resolved config, seed) and a script that re-runs the first five steps and compares them with the logged ones.

#### Further reading

- **Papers:** [LLaMA-Factory](https://arxiv.org/abs/2403.13372); [Liger Kernel](https://arxiv.org/abs/2410.10989) (new, verified; where fused-kernel throughput and memory gains come from); [Tülu 3](https://arxiv.org/abs/2411.15124) (a full training recipe with its configs).
- **Tutorials:** [PEFT docs](https://huggingface.co/docs/peft/index); [TRL docs](https://huggingface.co/docs/trl/index), [TRL PEFT integration](https://huggingface.co/docs/trl/peft_integration), [TRL Liger integration](https://huggingface.co/docs/trl/liger_kernel_integration) (verified 200); [Unsloth docs](https://docs.unsloth.ai) (verified 200); [Axolotl docs](https://docs.axolotl.ai) (verified 200); [LLaMA-Factory docs](https://llamafactory.readthedocs.io) (verified 200); [Hugging Face smol course](https://huggingface.co/learn/smol-course/en/unit0/1); [The future of torchtune](https://github.com/meta-pytorch/torchtune/issues/2883) (verified 200).
- **Code:** [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [torchtune](https://github.com/pytorch/torchtune), [Unsloth](https://github.com/unslothai/unsloth), [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (all verified 200 on 2026-10-28).

#### Risks and open decisions

- **Environments (decided 2026-10-29).** The book environment is `torch==2.15.0` on Python 3.12 and runs the from-scratch oracle and PEFT+TRL. Unsloth and Axolotl cap `torch` below 2.15.0 (PyPI metadata, 2026-10-29), so the arms for them, LLaMA-Factory and torchtune run in separate environments from committed scripts, and their committed logs record the `torch` version they used. If PEFT+TRL fails to resolve against 2.15.0 at build time, that is a finding for the editor, not a reason to move the pin.
- **torchtune is no longer maintained** (README banner and issue 2883, checked 2026-10-28; last PyPI release April 2025). v5 lists it among the three main toolchains; this blueprint demotes it to a frozen reference (config reading + at most one committed log). Alternative: replace it by Axolotl as the third live-analyzed arm (Axolotl 0.21.1 released 2026-10-27, active). Editor's call.
- **Unsloth's licensing/installability on the build box** (GPU-only; its own `unsloth-zoo` pins) is unverified; fallback: skip the arm and keep Axolotl/LLaMA-Factory.
- **TRL API drift** (a 1.x release series exists; argument names such as `completion_mask`, `assistant_only_loss`, `padding_free` may differ): the pinned docs are the reference and the section prints the versions it ran.
- **Throughput claims are hardware-specific** (Ada GPU): the ladder is a measurement on this box, labeled as such, not a ranking of libraries.
- **Run-to-run determinism:** bf16 kernels are not bitwise reproducible across libraries; the tolerance band is measured, not guessed.

### 7.8 Fine-tuning large models across GPUs

- **Label:** `sec_lm_ft_multi_gpu` · **Tier:** T3 (reduced configs, flag `CFG`: `'t1'` runs the same script through `d2l.launch(..., gpus_per_node=1)` on **one** 4090 with `Qwen3.5-2B-Base` LoRA and QLoRA, i.e. `fully_shard` over a one-rank mesh, so the sharding factor is checked by arithmetic only; `'t2pair'` runs the same script on two ranks of a T2 pair, which is where the two-rank correctness checks and the measured sharding factor live; no multi-GPU run uses the build box) · **Budget:** T3 ≤ 2 h on 8× A100, planned to fit 8× 40 GB (model loading and checkpoint reads are the long pole, to be measured; the timed steps are minutes), per-GPU peak ≤ 38 GiB; T2 pair ≤ ~1 h (to be measured); T1 ≤ 15 min, ≤ 15 GiB on one rank · **Codex units:** new (v5).
- **Builds on (book):** `sec_multi_gpu`/`sec_multi_gpu_concise` (`subsec_mgp-fsdp`, `fig_fsdp_lifecycle`, `tab_collectives`, the `fully_shard` sketch): sharding derivation and the collectives are *used*, not re-derived. `sec_memory_precision`/`sec_adamw`: the 16-byte ledger and 8-bit states. `sec_hardware`/`tab_gpu_specs`: ridge points. `sec_perf_model`: MFU.
- **Builds on (this part):**
  - 6.1 (`d2l.launch(script, nodes=1, gpus_per_node=8, **env)` and its log collection), 6.2 (`sec_lm_fsdp`: FSDP2 mechanics, `fully_shard`, `reshard_after_forward`, mixed precision, prefetching), 6.8 (checkpoint/restart of long runs: sharded checkpoints are used here only briefly).
  - 7.4 (`LoRALinear`, `add_lora`, `lora_state_dict`), 7.6 (`QuantLinear`, `quantize_model`, the QTensor layout, bits per parameter), 7.1 (`ft_memory`, `chunked_lm_loss`), 7.7 (the single-device oracle loop and `compare_loss_traces`).
  - 5.1 (`d2l.chip_specs()`: the A100 40GB PCIe row that 5.1 adds, with `p16_tflops` 312, `mem_bw_tbs` 1.555, `mem_gb` 40 and a scale-up domain of two GPUs over the NVLink bridge, and, for datasheet comparisons only, the A100 80GB SXM row (2.039, 80, 600); each with `source_url`), 5.6 (cost model) and Chapter 5's `lm-hw-prices` artifact (price per GPU-hour is an *input* from `lm-hw-prices`, not a number in this section), 4.2 (`d2l.flops_per_token`, `d2l.mfu`), 6.2 (`d2l.fsdp_wrap`, `d2l.fsdp_train_step`: the training script wraps with `fsdp_wrap` and steps with `fsdp_train_step` instead of re-implementing FSDP2 setup). 6.2's `ModelShape` describes dense decoders only; the hybrid layer bookkeeping of this section lives in `ft_memory` (below).
  - 1.7 (`sec_lm_api`: `d2l.ChatAPI(base_url, model, ..., mode='replay')` with its response cache, `d2l.chat_batch`, `d2l.api_cost(usage, price)` with `d2l.PricePoint`), for the models-larger-than-a-node part.
- **Exports (`#@save`):**
  - `d2l.ft_memory(...)` gains arguments `shard='fsdp'`, `reshard_after_forward=True`, `adapter_shard=True`, `offload=None | 'optimizer' | 'params'` and returns per-GPU bytes including the all-gather working set `gather_ws` (7.1's function; defaults unchanged). It also reads hybrid configs: for `model_type='qwen3_5'` it uses `text_config.layer_types`, `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim` and `attn_output_gate`, and counts a Gated DeltaNet layer as `h·(2·K + V) + h·V + 2·h·n_v + V·h` projection parameters (`K = n_k·d_k`, `V = n_v·d_v`), `(2K + V)·kernel` convolution parameters, `2·n_v + d_v` state and norm parameters; a full-attention layer with the doubled query projection of the output gate. Both layer types are priced separately, so `gather_ws` uses the largest.
  - `d2l.min_gpus(config, method, hbm_gb=40.0, headroom_gb=8.0, **kw) -> int`: smallest `N` with per-GPU `total ≤ hbm − headroom` under `ft_memory(..., n_gpus=N, shard='fsdp')`, where `total` is the shard, the gathered working set and the checkpoint-boundary activations (the layer's internals, NCCL buffers and the allocator overhead are the headroom; a design parameter recalibrated from the measured residual). Called with `hbm_gb=80.0` for the 80 GB column.
  - `d2l.train_flops(config, tokens, method='lora', grad_ckpt=True) -> float`: a thin wrapper over 4.2's `d2l.flops_per_token(config)` (which supplies the per-token matmul and attention FLOPs; the method multipliers below are applied to its forward count; if 4.2 reports the 6P training convention, divide by 3 first; one-line check in the notebook). FLOPs per run: forward `2P`, recompute `2P` if checkpointed, input gradients `2P`, weight gradients `2P` for trainable weights only (adapters negligible); i.e. LoRA `4P` or `6P` per token, full fine-tuning `6P` or `8P`. `P` counts matmul parameters only (the 27B model's 25,621,954,560: layer projections plus the output head; the embedding lookup is not a matmul). 4.2's `flops_per_token` reads dense-decoder attributes and its attention term `12·L·d·T` would treat all 64 layers as full attention, so for the hybrid model `train_flops` passes `T=None` (matmul-only) and takes the text config, not the multimodal wrapper.
  - `d2l.fsdp_comm_time(config, n_gpus, bus_gbps, method='lora', gathers=2, quant=None) -> dict`: seconds for all-gather (`gathers·(N−1)/N·W_bytes/bus`) and reduce-scatter (trainable-parameter gradient bytes), with `W_bytes` the stored weight bytes (4-bit codes for `quant='nf4_dq'`; embeddings, head and non-linear tensors stay bf16).
  - `d2l.parse_launch_logs(path) -> DataFrame`: per-rank, per-step loss, step time, tokens/s, allocated/reserved GiB.
  - The `%%writefile train_fsdp.py` script text is notebook content (≈ 120 lines), not an export.
  - Artifacts: `DATA_HUB['lm-ch07-fsdp-27b-logs']` (per-step per-rank JSONL for each arm, `memory_snapshot_rank0.pickle`, `topo.txt` from `nvidia-smi topo -m`, `env.json` with versions, GPU memory per device and installed kernel packages, NCCL profile summary), `DATA_HUB['lm-ch07-fsdp-ledger']` (`ledger.json`: predicted vs measured per-GPU components).
- **Models and data:**
  - Headline: `Qwen/Qwen3.8-27B` (revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; Apache-2.0 per the Hub API; not gated; read on 2026-10-29). Chosen because it is the newest dense Qwen release (2026-08-05; the same `qwen3_5` layout and parameter count as `Qwen/Qwen3.5-27B`), compact enough that the full-FT versus LoRA contrast happens inside one node, and offered by the hosted fine-tuning API of item 13 and by the Hugging Face router used in 1.7. Config (`text_config`): 64 layers, `hidden_size` 5,120, `intermediate_size` 17,408, vocabulary 248,320 (padded), **untied** embeddings, bf16 (`mamba_ssm_dtype` fp32). Layer pattern `layer_types`: `[linear_attention ×3, full_attention] ×16`, i.e. 48 Gated DeltaNet layers and 16 full-attention layers (`full_attention_interval=4`). Full-attention layers: 24 query heads / 4 KV heads, `head_dim` 256, an output gate (`attn_output_gate`, so `q_proj` is 5,120 → 12,288), partial rotary factor 0.25, `rope_theta` 1e7. Gated DeltaNet layers: 16 key heads and 48 value heads of dimension 128 (`linear_key_head_dim`, `linear_value_head_dim`), convolution kernel 4; projections `in_proj_qkv` 5,120 → 10,240, `in_proj_z` 5,120 → 6,144, `in_proj_a` and `in_proj_b` 5,120 → 48, `out_proj` 6,144 → 5,120, plus `conv1d` (10,240 × 1 × 4), `A_log` (48, fp32), `dt_bias` (48) and a 128-dimensional gated norm (fp32). 11 safetensors shards, 55,562,872,800 bytes. Parameter counts read from the safetensors headers: 27,781,427,952 in the file (the Hub API agrees), of which the vision tower is 460,730,096 and the multi-token-prediction head 424,699,392; **text-only 26,895,998,464** (layers 24,353,196,544: Gated DeltaNet layer 383,273,184, full-attention layer 372,255,232; embedding 1,271,398,400 and output head 1,271,398,400; final norm 5,120), which is what `AutoModelForCausalLM` builds (`Qwen3_5ForCausalLM`; the vision tower and the MTP head are skipped, as `00_conventions.md` says for Qwen3.5). The matmul parameters (projections of the 64 layers, 24,350,556,160, plus the head) are 25,621,954,560. A decoder layer's linear projections hold `99.99%` of its parameters (the `conv1d`, state and norm tensors are 51,424 of the Gated DeltaNet layer's 383,273,184).
  - Fallback: `Qwen/Qwen3-32B` (revision `9216db5781bf21249d130ec9da846c4624c16137` on `main` at 2026-10-29; 32,762,123,264 parameters; dense, standard attention; Apache-2.0; 64 layers, `hidden_size` 5,120, `intermediate_size` 25,600; 64 heads / 8 KV heads, `head_dim` 128; vocabulary 151,936; untied). Used if the Gated DeltaNet kernels misbehave under FSDP2 on A100 (Risks).
  - Full-fine-tuning calibration model: `Qwen/Qwen3-8B-Base` (`49e3418fbbbca6ecbdf9608b4d22e5a407081db4` on `main` at 2026-10-29; 8,190,735,360; Apache-2.0; dense, untied; see 7.6).
  - Small models for the T1 and T2-pair checks: `Qwen/Qwen3.5-2B-Base` (`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`; Apache-2.0; 24 layers of the same 3:1 pattern, `hidden_size` 2,048, `intermediate_size` 6,144, 16 value heads, 8 query / 2 KV heads of `head_dim` 256, **tied** embeddings; 2,274,069,824 in the file, text-only 1,881,825,088 with the tied embedding counted once). It exercises the same Gated DeltaNet code path, the fp32 scalar parameters and the tied-embedding sharding case on one 24 GB GPU.
  - Data: `lm-sft-mix-v1` (7.3; Apache-2.0 sources), the first 40 global batches of 16 rows × 1,024 tokens (packing off in the timed runs to keep the comparison simple; see the Gated DeltaNet note in Risks).
  - Datasheet facts (NVIDIA A100 datasheet, <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf>, verified 200 on 2026-10-29): bf16 tensor 312 TFLOPS dense (624 with sparsity) for every A100 form factor; A100 40GB PCIe: HBM2 40 GB at 1,555 GB/s, 250 W; A100 80GB SXM: HBM2e 80 GB at 2,039 GB/s, 400 W; NVLink 600 GB/s on SXM boards (HGX, NVSwitch); the PCIe cards join **two** GPUs through an NVLink bridge at 600 GB/s and use PCIe Gen4 (64 GB/s) for everything else; no FP8. **The T3 node has 40 GB per GPU; its topology is recorded (`env.json`, `topo.txt`) and decides which row of the communication estimate applies.**
- **Packages:** `torch` (FSDP2 `torch.distributed.fsdp.fully_shard`, DCP), `transformers`, `peft`, `bitsandbytes`, `accelerate` (production arm), `flash-linear-attention` (PyPI 0.6.0 on 2026-10-29, imported as `fla`; requires `transformers>=5.6.2,<6`) and optionally `causal-conv1d` (1.7.0) and `kernels` (0.18.0) for the Gated DeltaNet layers; pins in 7.7. No DeepSpeed.

#### Narrative arc

1. **The question.** Opening cell (CPU, meta device): build `Qwen3.8-27B` text-only on the meta device from its config, count 26,895,998,464 parameters (the file holds 27,781,427,952; the difference is the vision tower and the MTP head, which the cell lists by key prefix), and print the per-GPU bytes for four schemes on 8 GPUs (from `ft_memory`: shard, gathered working set, boundary activations at micro-batch 2 and sequence 1,024) with `min_gpus` at 40 GB and at 80 GB per GPU. Full FT at 16 B/param: `53.8 + 5.6 + 1.3 = 60.7 GB` per GPU, so **impossible on 8× 40 GB** (`min_gpus` = 18, three nodes) and feasible on 8× 80 GB (`min_gpus` = 7); full FT with 8-bit `m`,`v` and an fp32 master (10 B/param): `40.6 GB`, still impossible at 40 GB (`min_gpus` = 11, two nodes), feasible at 80 GB (5); LoRA on bf16 weights: `13.1 GB` (`min_gpus` = 3 at 40 GB, 1 at 80 GB); QLoRA: `7.1 GB` (1 and 1). Symptom: the headline "27B on one node" is a statement about *which method* (and, for full fine-tuning, about the node's GPU memory), not about the node: on 8× 40 GB only the adapter methods fit, and the ledger shows the answer changing with 80 GB GPUs.
2. **Why plain data parallelism fails and what FSDP changes.** One paragraph recap of `subsec_mgp-fsdp`: shard parameters, gradients and optimizer state across ranks; all-gather each layer's weights just in time; reduce-scatter gradients. Two consequences specific to fine-tuning: (a) with a **frozen** base only trainable parameters need reduce-scatter and optimizer state, so LoRA's communication is dominated by the forward/backward all-gathers of the *frozen* weights; (b) with a **quantized** base the all-gathers move 4-bit codes (plus the bf16 embeddings and head, which are not quantized).
3. **FSDP1 vs FSDP2 for LoRA.** FSDP1's flat parameter groups required a uniform `requires_grad` within a unit, which is why Answer.AI's FSDP-QLoRA work needed engineering; FSDP2's per-parameter sharding (DTensor) lets frozen base weights and trainable adapters live in the same wrapped module (6.2; the author confirms the statement against the pinned PyTorch docs and the mixed-dtype handling, see Risks). Wrap policy: `fully_shard` on each decoder layer (Gated DeltaNet and full-attention layers alike: their gathered sizes differ by 3%, see D1), then on the root; `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`; adapters fp32.
4. **The hybrid layer under FSDP.** What changes when a model has Gated DeltaNet layers, and what does not. (a) *Parameters:* 99.99% of a layer is projections, so the ledger is dominated by the same matmul weights as in a dense model; the layer types differ by 3% in bytes (0.7665 GB against 0.7445 GB in bf16), which matters for the gathered working set, not for the shard. (b) *Kernels:* transformers resolves the delta-rule and convolution functions in three tiers (read from `integrations/hub_kernels.py` and `models/qwen3_5/modeling_qwen3_5.py` on `main`, 2026-10-29; the author re-reads them at the pinned release): Hugging Face Hub kernels (`kernels-community/fla`, listed first in the function's docstring and used if requested through the `kernels` package and `USE_HUB_KERNELS`), then the `flash-linear-attention` package (`fla`), then a pure-PyTorch reference (`torch_chunk_gated_delta_rule`). The reference path is correct but slow (the library logs a warning saying so) and raises `ValueError` if sequences are packed (`cu_seqlens`), so packing needs `fla`. The causal convolution has a plain `F.conv1d` fallback that is cheap, so `causal-conv1d` is an optimization. The cell prints which tier is active and times one layer's forward and backward in each tier on a T1 GPU. (c) *Kernels see local tensors:* the Triton kernels run after the all-gather on ordinary tensors, so they are not DTensor-aware; the interaction with FSDP2 that can go wrong is dtype, not sharding (D1 step 3 and Risks). (d) *Scalars:* `A_log` and the gated-norm weight are stored in fp32 next to bf16 projections; `MixedPrecisionPolicy(param_dtype=bf16)` would cast them too, and the model code takes `A_log.float()` afterwards, so the cast loses precision relative to the one-rank reference unless handled (the T1/T2-pair oracle comparison detects it).
5. **The memory ledger per GPU (D1).** The five components, the checkpoint-boundary activations, and the working set of all-gathered layers (`≈ (1 + prefetch) × layer bytes`) plus the embedding or head matrix while it is gathered (2.54 GB, 3.3× the largest layer); a table for `N ∈ {2, 4, 8}`: bf16 LoRA `34.0, 20.1, 13.1 GB`, QLoRA `14.4, 9.5, 7.1 GB`; the claim "QLoRA 27B fits on one 40 GB GPU" (`24.2 GB`, T2 exercise) and why it still helps on eight (traffic, D3).
6. **Sharded versus replicated adapters.** 116,727,808 adapter parameters at `r=16` on every linear layer (the five Gated DeltaNet projections, including the two 48-row ones, and the seven attention and MLP matrices): fp32 weights + gradients + Adam = 1.87 GB if replicated, 0.23 GB if sharded; communication difference (all-reduce vs reduce-scatter + all-gather of small tensors). Measure both on the `Qwen3.5-2B-Base` T2-pair arm (16,819,200 adapter parameters); report the 27B ledger arithmetic.
7. **QLoRA under FSDP.** What must hold for sharding quantized weights: uniform dtype per parameter group, so the packed codes are *viewed* as bf16 (4 codes per element; PEFT's `bnb_4bit_quant_storage_dtype='bfloat16'` does this, as its FSDP guide documents) and re-viewed as `uint8` after the all-gather; `QuantLinear` gets a `shard-safe` mode doing this view; dequantization happens after the gather, per layer; scales shard with the codes (the double-quant state is replicated, ≈ 0.2% of the weights, to be checked). Embeddings, head and the non-linear tensors stay bf16: 5.09 GB of the 17.65 GB stored bytes, which is why the stored weights shrink by `3.05×`, not by `3.9×`. Fallback arm: PEFT + bitsandbytes + Accelerate FSDP config.
8. **Communication time (D3).** Compute the all-gather and reduce-scatter time per step for the three methods from `fsdp_comm_time` for the bandwidth of the row that `topo.txt` selects (NVSwitch SXM, or PCIe with NVLink-bridged pairs); measure the realized `busbw` of an `all_gather` of one layer-sized message on the node (reuse 6.1's microbenchmark), then compare the predicted exposed communication with the profiler's fraction of non-overlapped NCCL time.
9. **Compute accounting (D2).** `train_flops` for LoRA vs full FT; MFU from measurements; why LoRA at 27B saves memory far more than time (`6P` vs `8P` with checkpointing); which FLOPs the counters do not see (the delta-rule kernels).
10. **Full fine-tuning on one node: what actually fits.** A table of schemes for three models (`Qwen3-8B-Base`, `Qwen3.8-27B`, `Qwen3-32B`) on 8 GPUs of 40 GB and of 80 GB. For the 27B model: 16 B/param needs ≥ 18 GPUs at 40 GB (three nodes) and ≥ 7 at 80 GB; 10 B/param with fp32 master and 8-bit `m`,`v` ≥ 11 and ≥ 5; 6 B/param bf16-only with 8-bit state ≥ 7 and ≥ 3 (`27.1 GB` per GPU on eight, so it fits 8× 40 GB by the ledger, with the quality risk of 7.1 D2); ZeRO-3 with optimizer-state offload (GPU holds bf16 weights and gradients, 4 B/param = 13.4 GB per GPU on eight; the CPU holds 12 B/param = 323 GB for the 27B model; needs that much host RAM and PCIe bandwidth; ZeRO-Infinity adds NVMe) is named and costed by the ledger, not run. The measured full-FT arm is `Qwen3-8B-Base` at 16 B/param on 8× 40 GB (A3), which tests the 16 B/param ledger and the gradient traffic; the 27B full-FT row is arithmetic.
11. **Running it.** `d2l.launch('train_fsdp.py', nodes=1, gpus_per_node=8, MODEL_ID=..., METHOD=..., ...)`; what the log collection returns; how the notebook parses it (`parse_launch_logs`); T1 (`gpus_per_node=1`) vs T2 pair (2) vs T3 (8) differences. The training script's anatomy (meta-device init, rank-0 load and broadcast or sharded loading, wrap with `d2l.fsdp_wrap` (6.2), optimizer with `lora_param_groups`, step via `d2l.fsdp_train_step`, gradient checkpointing, loss with `chunked_lm_loss`, timing with CUDA events, `torch.cuda.max_memory_allocated`, sharded and adapter-only checkpoints).
12. **Checkpoints and the adapter.** Adapter-only gather to rank 0 (`full_tensor()` per adapter parameter; 467 MB in fp32 for the 27B model at `r=16`, 233 MB in bf16; the notebook reports the file size); distributed (DCP) checkpoint for resuming (pointer to 6.8); merging into the 27B bf16 model requires 53.8 GB of host RAM for the text-only weights and is done out of band (documented, not run).
13. **Models larger than a node: a hosted fine-tuning API.** Past the node (hundreds of billions of parameters, mixture of experts) the section's recipe changes owner: the provider runs the sharded training, the reader supplies data, a loss and a learning rate, and gets an adapter back. The pattern, which does not depend on the provider: (i) tokenize and mask locally with `d2l.render_chat` (7.2), so the loss mask is the reader's; (ii) send batches; (iii) the service runs forward-backward and the optimizer step on a LoRA adapter of the requested rank; (iv) save a sampler checkpoint; (v) sample through an OpenAI-compatible endpoint with 1.7's client and its response cache, or download the adapter and merge it into the base model. Token usage comes back with every response; the spend is `d2l.api_cost(usage, price)` with a `PricePoint` the reader types in from the provider's page, so the section hardcodes no price. What the reader gives up: precision, sharding, kernels, the optimizer details and the memory ledger; what remains: data, mask, rank, learning rate (the LoRA Without Regret rule of about 10× the full-FT rate, 7.4) and evaluation (Chapter 1's suite). One instance, verified on 2026-10-29: Tinker (Thinking Machines). Its SDK cheatsheet shows `create_lora_training_client(base_model=..., rank=...)`, `forward_backward`, `optim_step` and `save_weights_for_sampler`; its models page lists models with a training price, up to 975B total parameters (`thinkingmachines/Inkling`), among them `Qwen/Qwen3.5-397B-A17B` and `moonshotai/Kimi-K2.6`, and prices per million tokens (not copied here); a checkpoint can be sampled through an OpenAI-compatible endpoint (beta) by passing its checkpoint path as the model name, or merged into a Hub base model with `weights.build_hf_model`. That page lists `Qwen/Qwen3.8-27B`, the section's own headline model, so the hosted run can fine-tune the same base the T3 run shards; model lists change, so the notebook states the pattern and the date as well. These cells need an account and a key and are not executed in the build: a recorded transcript of the calls is committed and replayed at T0, like the other API-backed cells. Together AI documents the same workflow (<https://docs.together.ai/docs/fine-tuning/overview>, verified 200; its supported models were not checked).

#### Derivations

**D1. Per-GPU memory under FSDP with a frozen base (`eq_lm_ft_fsdp_mem`).**
This specializes Chapter 6's general FSDP memory derivation :eqref:`eq_lm_fsdp_mem` (6.2) to a frozen quantized base plus adapters; the general sharding terms are cited, not re-derived.
Statement: with `N` ranks, `P_lin` linear parameters (stored at `b_w` bytes each: 2 for bf16, `0.516` for NF4+DQ), `P_rest` all other text-only parameters in bf16 (embedding, head, norms, convolution and state tensors), `P_a` adapter parameters (fp32 weights, gradients and Adam: 16 B each), `L` layers with `W_ℓ` gathered bytes per layer, `W_emb` the embedding or head matrix while it is gathered, activations `A` (checkpoint boundaries `L·B·T·h·2` bytes plus one layer's working set) and logits `Λ` (`0` with chunked loss): per-GPU memory is
`M = (b_w P_lin + 2 P_rest)/N + 16 P_a/N_a + k_g·W_max + W_emb + A + Λ + M_nccl`, with `N_a = N` for sharded adapters and `1` for replicated ones, `k_g` the number of gathered layers alive at once (`1 + prefetch depth`) and `W_max` the largest layer's gathered bytes. Full fine-tuning replaces the adapter and frozen-weight terms by `16·P/N` (fp32 master, gradient, `m`, `v`) and adds one layer's unsharded gradient (`+ W_max`) to the transient.
Steps:
1. Persistent shards: each rank holds `1/N` of every parameter (FSDP2 shards dim 0), so frozen bf16 weights cost `2P/N`; for the text-only 27B model `P = 26,895,998,464`, i.e. `53.79 GB` in bf16 and `6.72 GB` per GPU on eight.
2. Adapters: trainable, hence weights, gradients and Adam states, each fp32 (`4+4+8 = 16 B`); sharded over `N` or replicated.
3. Transient: with `reshard_after_forward=True`, a layer's full weights exist only during its forward and backward, plus prefetched layers: `k_g·W_max`. The two layer types differ: Gated DeltaNet `383,273,184` parameters (`0.7665 GB`, 48 layers), full attention `372,255,232` (`0.7445 GB`, 16 layers), so `W_max = 0.7665 GB`; the MLP (`3·5,120·17,408 = 267,386,880`) is about 70% of either. The embedding and the head are `1,271,398,400` parameters each (`2.543 GB`, 3.3× a layer) and are the largest gathered units. The fp32 `A_log` and gated-norm tensors are gathered as bf16 under the policy above (51,424 parameters per layer, negligible in bytes, not in numerics).
4. Activations: with checkpointing only layer-boundary tensors persist: `L·B·T·h·2 = 64·2·1024·5120·2 = 1.34 GB` for micro-batch 2; plus one layer's internals (calibrated `act_coef`, 7.1; for Gated DeltaNet layers the internals depend on the kernel tier, so the coefficient is calibrated for both layer types and the tier is recorded).
5. Plug in for `N=8`: bf16 base shard `6.72 GB`; NF4+DQ shard `2.21 GB` (stored bytes `17.65 GB` = `24,350,556,160·0.516 B = 12.56 GB` of codes and scales + `5.09 GB` bf16 rest); adapters sharded `0.23 GB` (replicated `1.87 GB`); gathered working set `3·0.7665 = 2.30 GB` plus `2.54 GB` for the head, `4.84 GB` for bf16; for NF4+DQ `3·0.198 = 0.59 GB` of gathered codes (0.198 GB per layer) plus the dequantized transient of the largest linear (`17,408·5,120·2 = 0.178 GB`) plus `2.54 GB`, `3.31 GB`; boundary activations `1.34 GB`. Sum, bf16 LoRA: `6.72 + 0.23 + 4.84 + 1.34 = 13.14 GB`; QLoRA: `2.21 + 0.23 + 3.31 + 1.34 = 7.10 GB` (both before the layer's internals, the NCCL buffers and the allocator overhead; the measured peak decides, with the residual reported). Full FT at 16 B/param: `53.79 + 5.61 + 1.34 = 60.74 GB` (the working set includes the gradient transient).
6. `min_gpus` with `headroom_gb = 8`: at 40 GB the budget is 32 GB, and bf16 LoRA needs `N ≥ 3` (`N = 2` gives 34.0 GB, `N = 3` gives 24.7 GB); QLoRA fits `N = 1` (24.2 GB); full FT at 16 B/param needs `N = 18` (30.9 GB). At 80 GB the budget is 72 GB: LoRA and QLoRA `N = 1`, full FT `N = 7` (68.4 GB).
Check cell: `ft_memory` vs `torch.cuda.max_memory_allocated` per component group (weights from `named_parameters` bytes; adapters; activations by memory-snapshot categories) on `Qwen3.5-2B-Base` at `N = 1` (T1) and `N = 2` (T2 pair), at `N = 8` for the 27B model and for `Qwen3-8B-Base` (T3). Agreement to 15% on the total and the residual explained is the learning objective's target; a larger miss is reported as a finding.

**D2. FLOPs per token: LoRA versus full fine-tuning (`eq_lm_ft_flops`).**
Statement: for a transformer with `P` weight parameters in matmuls and tokens `D`, ignoring attention-score FLOPs, the delta-rule recurrence and the adapters (`≪ P`): full fine-tuning costs `6PD` (forward `2P`, input gradient `2P`, weight gradient `2P`), `8PD` with activation checkpointing; LoRA costs `4PD`, `6PD` with checkpointing (no weight gradient for frozen layers).
Steps: (1) a matmul `Y = XW` with `X ∈ ℝ^{T×n}`, `W ∈ ℝ^{n×m}` costs `2Tnm`; (2) backward needs `dX = dY Wᵀ` (`2Tnm`) always, and `dW = Xᵀ dY` (`2Tnm`) only if `W` trains; (3) checkpointing reruns the forward (`+2Tnm`); (4) ratio LoRA/full: `4/6 = 0.67` without and `6/8 = 0.75` with checkpointing; (5) therefore, "LoRA is cheaper" is mostly a *memory* statement: at 27B with checkpointing it saves 25% of FLOPs. For one step of `D = 16·1,024 = 16,384` tokens and `P = 25,621,954,560`: LoRA with checkpointing `6PD = 2.52 PFLOP`, full FT with checkpointing `8PD = 3.36 PFLOP`. Not counted: the 16 full-attention layers' score FLOPs and the delta-rule recurrence of the 48 Gated DeltaNet layers, which run in Triton kernels that `FlopCounterMode` does not see; the notebook labels the resulting MFU "matmul-only" (a lower bound), because `d2l.mfu` is called with `T=None` for the hybrid model (Exports).
Check cell: count FLOPs by `torch.utils.flop_counter.FlopCounterMode` on one decoder layer's projections and MLP (bf16, small shapes, the kernels bypassed) for LoRA vs full FT with and without checkpointing and verify the 4:6:6:8 pattern to within a few percent (adapter FLOPs and attention scores explain the residual).

**D3. Communication per step and QLoRA's traffic advantage (`eq_lm_ft_fsdp_comm`).**
This builds on Chapter 6's general all-gather/reduce-scatter cost :eqref:`eq_lm_fsdp_comm` (6.2); only the stored-weight bytes (4-bit codes) and trainable-gradient bytes are new.
Statement: with `N` ranks, ring all-gather over `B_bus` bytes/s of bus bandwidth, `g` gathers of the stored weight bytes `W_s` per step, the all-gather time is `T_ag = g·((N−1)/N)·W_s/B_bus`; gradient reduce-scatter of trainable bytes `G_t`: `T_rs = ((N−1)/N)·G_t/B_bus`.
Steps:
1. Ring all-gather: each rank sends and receives `(N−1)/N` of the total (`sec_multi_gpu`, `subsec_mg-ring`).
2. Gathers per step: with `reshard_after_forward=True` and checkpointing, `g = 2` (forward, then backward-with-recompute); `g = 1` if layers stay gathered between forward and backward (`reshard_after_forward=False`, memory cost `+W_s`).
3. Qwen3.8-27B bf16: `W_s = 53.79 GB`; `T_ag = 2·(7/8)·53.79/B_bus`: `94.1 GB` per step per rank, i.e. `0.31 s` at `300 GB/s` (half of the 600 GB/s NVLink datasheet figure of an NVSwitch node, counted per direction) and `0.47 s` at `200 GB/s` realized bus bandwidth (an illustrative realized value; the measured `busbw` replaces it).
4. NF4+DQ: `W_s = 17.65 GB` (codes and scales; embeddings, head and non-linear tensors in bf16): `30.9 GB` per step per rank, `0.10 s` at `300 GB/s`: a `3.05×` reduction.
5. LoRA reduce-scatter: `G_t = 4 P_a = 0.47 GB`: negligible. Full fine-tuning adds `(7/8)·2P = 47.1 GB` for bf16 gradients (twice that with `reduce_dtype=torch.float32`).
6. Compare with compute: a step on 16k tokens costs `6·P·D = 2.52 PFLOP` for LoRA with checkpointing (D2), `≈ 3.4 s` at an assumed 30% MFU on 8×312 TFLOPS (illustrative; measured MFU replaces it): the bf16 all-gather is `≈ 9–14%` of compute time (300 and 200 GB/s) and the NF4 all-gather `≈ 3–5%`, overlappable by prefetching; exposed communication is what the profiler measures.
7. PCIe case: if the node has no NVSwitch fabric (PCIe A100s with NVLink bridges between pairs), the bridge joins two GPUs only; a ring over eight must cross PCIe, so `B_bus` is bounded by PCIe Gen4 (datasheet 64 GB/s bidirectional per GPU, i.e. 32 GB/s each way, a ceiling that host topology can lower): `T_ag = 2.94 s` for bf16 (`87%` of the assumed compute time) and `0.97 s` for NF4 (`29%`), `9.4×` larger than at 300 GB/s, so the step is communication-bound and QLoRA's `3.05×` traffic reduction becomes a *speed* argument.
Check cell: `fsdp_comm_time` for the three methods; measured `busbw` from a layer-sized `all_gather` loop (`torch.distributed`, 20 iterations); the predicted vs profiled exposed NCCL time per step (to be measured); on the T2 pair (two ranks joined by the NVLink bridge) the same comparison for `Qwen3.5-2B-Base`, which measures the bridge row of the table, not the 8-GPU one.

#### From-scratch implementation

Budget ≈ 90 lines of exported helpers + the ≈ 120-line training script (a notebook cell, `%%writefile`).
- `ft_memory` extension (35 lines, including the hybrid-layer counts), `min_gpus` (10), `train_flops` (10, on top of `flops_per_token`), `fsdp_comm_time` (15), `parse_launch_logs` (15).
- **The single-GPU reference** is 7.7's oracle loop (the same model, data and recipe on `world_size=1`); the sharded run must reproduce its first steps.
- **Tests.** (1) `train_flops` pattern 4:6:6:8 (D2) on one layer via `FlopCounterMode` within 5%; (2) `ft_memory` parameter count equals `sum(numel)` on the meta device exactly for `Qwen3.8-27B` (26,895,998,464, text-only), `Qwen3.5-2B-Base` (1,881,825,088, tied embedding counted once) and `Qwen3-8B-Base` (8,190,735,360); (3) correctness: on T1, FSDP2 over a one-rank mesh with LoRA on `Qwen3.5-2B-Base` vs the plain loop on identical batches, 10 steps, loss relative difference ≤ 1e-3 per step (the wrapper path, mixed-precision policy and the fp32 scalar handling; tolerance to be calibrated); on the T2 pair, 2-rank FSDP2 vs the 1-rank loop, 10 steps, loss relative difference ≤ 1e-2 per step (bf16; tolerance to be calibrated); (4) adapter-only checkpoint round trip: save from rank 0, load into a fresh single-GPU model, logits on one batch equal to the sharded model's within bf16 tolerance (T1 with one rank, T2 pair with two); (5) QLoRA shard-safe view: `view(bf16)` then `view(uint8)` round trip on a `QTensor` is bit-exact (T1); all-gathered then dequantized weight equals the single-device `dequantize` exactly (T2 pair, 2 ranks); (6) Gated DeltaNet tiers: on one layer with random inputs, the `fla` (or Hub kernel) forward and gradients agree with the pure-PyTorch reference within a bf16 tolerance to be calibrated, and the notebook prints which tier transformers resolved (T1).

#### In production

- **PyTorch FSDP2** (`fully_shard`, `MixedPrecisionPolicy`, `CPUOffloadPolicy`, `torch.distributed.checkpoint`): the notebook's script *is* the production path at this scale; the from-scratch pieces are the adapters, the quantizer and the ledger.
- **PEFT + Accelerate + TRL (FSDP).** The PEFT FSDP guide (verified 200) fine-tunes a 70B model with LoRA on 8×80 GB GPUs with `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP`, `fsdp_sharding_strategy: FULL_SHARD`, `fsdp_cpu_ram_efficient_loading: true`, `mixed_precision: bf16`, `fsdp_state_dict_type: SHARDED_STATE_DICT`, and for QLoRA `--bnb_4bit_quant_storage_dtype bfloat16` (packs four 4-bit values per bf16 element). The notebook reproduces one short run with this stack on the T2 pair (`Qwen3.5-2B-Base`, same data, 2 ranks) and compares per-GPU memory and step time with the from-scratch script's (to be measured); the ledger is the oracle for both. The claim (a toolchain's agreement with the ledger) does not need the T3 node.
- **DeepSpeed ZeRO-3** (`zero_optimization: stage 3`, `offload_optimizer: cpu`) and ZeRO-Infinity (NVMe): cited and costed in item 10, not run (the offload arm needs 323 GB of host RAM for the 27B model).
- **torchtune** (frozen) distributed recipes `lora_finetune_distributed`, `full_finetune_distributed`, `qlora_finetune_distributed` are read as references; not executed.
- **Unsloth/Axolotl/LLaMA-Factory** have multi-GPU modes; pointers only (7.7).
- **Hosted fine-tuning:** the pattern of item 13; one verified provider, no recipe.

#### Headline experiment

- **Hypothesis.** (H1) A bf16-LoRA fine-tune of Qwen3.8-27B fits on 8×A100-40GB with per-GPU peak within 15% of the `ft_memory` prediction (13.1 GB plus the layer's internals), and a QLoRA run has about one-third of the weight-shard memory (`2.21` vs `6.72 GB`, arithmetic); (H2) QLoRA's step time is within the dequantization overhead of bf16 LoRA, and is *lower* if the run is communication-bound (the PCIe row of `tab_lm_ft_fsdp_comm`); (H3) full fine-tuning at 16 B/param follows the same ledger: `Qwen3-8B-Base` on 8×40 GB peaks within 15% of its prediction (`19.8 GB`), and its extra gradient traffic matches D3 step 5; the 27B full-FT row of `tab_lm_ft_fsdp_fits` (60.7 GB, not fitting 40 GB) is arithmetic; (H4) exposed communication is a small fraction of step time on an NVSwitch node and a large one on PCIe-pair nodes (a measured, not assumed, statement for whichever node `topo.txt` shows).
- **Setup (T3).** Arms (global batch 16 rows × 1,024 tokens, micro-batch 2 per GPU, gradient checkpointing, `chunked_lm_loss`, 3% warmup, `lr` as 7.4's LoRA optimum for the LoRA arms and 7.1's full-FT value for the full-FT arms): (A1) Qwen3.8-27B, bf16 LoRA `r=16` all-linear, FSDP2, 8×40 GB; (A2) Qwen3.8-27B QLoRA (NF4+DQ, `r=16`), FSDP2 with shard-safe codes; (A3) Qwen3-8B-Base full FT at 16 B/param (fp32 sharded parameters, bf16 compute, fp32 reduce), FSDP2. 40 steps each (10 warm-up, 30 timed where the step budget allows). The DeepSpeed-offload arm and the PEFT/Accelerate arm of the earlier plan are not run here (offload needs ≥ 323 GB of host RAM and supports no stated claim; the toolchain cross-check moved to the T2 pair, see In production).
- **Procedure.** Record `topo.txt` and the GPU memory per device first; they fix which arms run and which row of `tab_lm_ft_fsdp_comm` applies. Print the Gated DeltaNet tier in use. Launch via `d2l.launch`. Per step and per rank: loss, step time (CUDA events), tokens/s, `max_memory_allocated`/`reserved`; a memory snapshot at step 5 on rank 0; a 5-step profiler window for NCCL overlap. After A1/A2: save the adapter, reload and check logits.
- **Metrics.** Peak GiB per GPU (allocated and reserved) vs ledger; median step time (IQR over timed steps); tokens/s per GPU and cluster MFU from `d2l.mfu(tokens_per_s, config, peak_flops, n_gpus=N)` (4.2; `peak_flops` from `chip_specs()['p16_tflops']`, 312 for every A100), plus hardware-FLOPs utilization `train_flops / (step_time · N · peak)` where the method multipliers matter; exposed-communication fraction; for A3, the estimated 27B full-FT requirement (GPUs for the node memory at hand, GPU-hours per 100M tokens at the measured utilization, price from Chapter 5's `lm-hw-prices` artifact).
- **Plot spec.** `fig_lm_ft_fsdp_memory`: grouped bars per arm (weights, adapters/optimizer, gathered working set, activations, other), predicted vs measured, with a 40 GB line; `fig_lm_ft_fsdp_step`: per-step time across the 30 timed steps for each arm (box plot per arm) with MFU annotated; `fig_lm_ft_fsdp_timeline`: profiler timeline summary, compute and NCCL bars for one step of A1 and A2; `tab_lm_ft_fsdp_fits`: the what-fits table for 8B, 27B and 32B across schemes (computed, with measured rows marked).
- **Expected outcome.** Directional: A1 and A2 fit on 40 GB; A2's weight shard ≈ 3.05× smaller than A1's (arithmetic from the QTensor accounting); A3 fits at about 20 GB per GPU; full FT of the 27B model does not fit 8×40 GB (arithmetic, `tab_lm_ft_fsdp_fits`) and, by the ledger, fits 8×80 GB at about 61 GB. Step-time and MFU values: to be measured. Source context: Answer.AI reports a 70B QLoRA fine-tune on two 24 GB GPUs; the PEFT guide documents a 70B LoRA recipe on 8×80 GB GPUs.
- **Noise check.** Step times: median and IQR over ≥ 15 timed steps; arms are compared only when their IQRs do not overlap; single run per arm (T3 budget), so no seed variance is claimed for losses (losses are reported as sanity, not compared across arms); memory numbers are deterministic up to allocator fragmentation, so allocated and reserved are both reported. T1 and T2 pair: 3 repeats of the one-rank and two-rank runs.
- **T1 reduction.** `CFG='t1'`: `Qwen3.5-2B-Base`, one rank on one 4090, A1/A2-analogs and the correctness checks that need one rank (tests 1, 2, 3 (one-rank part), 4 (one rank), 5 (first half) and 6); `CFG='t2pair'`: two ranks, the two-rank parts of tests 3–5, the measured sharding factor at `N ∈ {1, 2}` and the PEFT/Accelerate cross-check. Figures are produced from T2-pair logs, or from T1 logs, only if T3 artifacts are absent.
- **Artifacts committed for T3.** `lm-ch07-fsdp-27b-logs`, `lm-ch07-fsdp-ledger`, the figures' data; the analysis cells run on them (T3 notebooks are re-executed per release, 6.x convention).

#### Figures and tables

- `tab_lm_ft_fsdp_fits`: schemes × {`Qwen3-8B-Base`, `Qwen3.8-27B`, `Qwen3-32B`}: bytes per parameter, per-GPU total at `N=8`, `min_gpus` at 40 GB and at 80 GB, offload requirement, verdict (computed from the ledger; measured rows flagged). For the 27B model: 16 B/param `60.7 GB`, 18 and 7 GPUs; 10 B/param `40.6 GB`, 11 and 5; 6 B/param `27.1 GB`, 7 and 3; bf16 LoRA `13.1 GB`, 3 and 1; QLoRA `7.1 GB`, 1 and 1.
- `fig_lm_ft_fsdp_memory`, `fig_lm_ft_fsdp_step`, `fig_lm_ft_fsdp_timeline`.
- `tab_lm_ft_fsdp_comm` (7.8's own; distinct from 6.2's `tab_lm_fsdp_comm`): predicted all-gather/reduce-scatter seconds for LoRA, QLoRA, full FT at the measured `busbw`, with both the NVSwitch (300 GB/s) and the PCIe-pair (≤ 32 GB/s) rows printed and the one `topo.txt` selects marked.
- `fig_lm_ft_fsdp_flops`: FLOPs per token for LoRA/full FT with and without checkpointing (D2) with the measured counter values.

#### Pitfalls and misconceptions

- "LoRA means little communication": the frozen weights are all-gathered every step; only the gradient traffic is small.
- "QLoRA is for memory only": under FSDP it also cuts the all-gather traffic by about 3×, less than the 4-bit ratio because the embeddings and head stay in bf16.
- "27B fits on one node" is a statement about the method and the GPU: full FT at 16 B/param does not fit 8×40 GB and, by the ledger, does fit 8×80 GB.
- Every rank loading the full checkpoint into CPU RAM (`8 × 54 GB`): use meta-device initialization with rank-0 loading or sharded loading.
- Mixed dtypes and `requires_grad` inside one FSDP unit (FSDP1 flat parameters; FSDP2 per-parameter sharding relaxes this, quantized codes still need a uniform storage dtype in some versions).
- A mixed-precision policy casts every floating parameter: the fp32 `A_log` and gated-norm weights of the Gated DeltaNet layers become bf16 in the gathered copy unless the script handles them, and the loss then differs slightly from the one-rank reference.
- The pure-PyTorch Gated DeltaNet path is correct, slow and memory-hungry, and rejects packed sequences; a run that is "working" with a warning in the log has not used the kernels.
- The two adapter matrices of a 48-row projection (`in_proj_a`, `in_proj_b`) are tiny and cost almost nothing, but they are a quantized-base edge case (block size 64 over a 5,120 × 48 tensor); keep them in the shape test.
- `reshard_after_forward=False` saves a gather per step but keeps every wrapped layer's full weights resident from its forward until its backward, i.e. about `W_s` extra per rank (53.8 GB for the 27B model in bf16: not viable on 40 GB; fine for the T1 models).
- Embedding and `lm_head` at vocabulary 248k are 2.54 GB each in bf16 and are gathered like any layer (3.3× the largest decoder layer); logits memory returns if the loss is not chunked.
- Trusting loss values across arms with different masks or batch composition; the arms here use identical batches.
- Estimating cost from a single `nvidia-smi` reading: use `max_memory_allocated` and the snapshot, and the topology file.
- Merging a LoRA into a 27B model needs the full bf16 weights in one place (≈ 54 GB of host RAM).
- Treating a hosted fine-tuning API's model list as stable: it changes, and a model you validated locally may not be offered.

#### Summary (draft bullets)

- "Fine-tune a 27B model on one node" works for LoRA and QLoRA on 8×40 GB; full fine-tuning at 16 B/param needs about three such nodes, or 8×80 GB GPUs, or offload.
- FSDP shards frozen weights, adapters, gradients and optimizer states; the per-GPU ledger predicts the peak, and the residual is activations and workspaces.
- LoRA saves about a quarter of the FLOPs per token, not most of them; its saving is in optimizer and gradient memory.
- Communication under FSDP is dominated by all-gathering the frozen base; NF4 storage shrinks it by about 3×.
- A hybrid model changes the kernels, not the ledger: check which Gated DeltaNet implementation is running.
- Measure on the actual node: the topology and the GPU memory decide whether the run fits and whether it is compute- or communication-bound; past the node, a hosted API trades the ledger for a service.

#### Exercises

1. *Check.* Qwen3.8-27B (text-only, bf16): how many bytes per GPU does the base shard take on 8 GPUs, and how many on 16? What does QLoRA take on 8? *Sketch:* `26.896e9·2/8 = 6.72 GB`; `3.36 GB` on 16; `17.65/8 = 2.21 GB`.
2. *Check.* A step gathers each layer's weights twice. Which flag changes this to once, and what does it cost in memory? *Sketch:* `reshard_after_forward=False`; keeps all layers' gathered weights until backward consumes them (up to the full model's bf16 weights for the wrapped layers; check `k_g` in D1).
3. *Derive.* Derive the `4PD`/`6PD`/`8PD` FLOP counts and the minimum token count at which LoRA's saved weight-gradient FLOPs exceed the adapter's extra forward/backward FLOPs for rank `r` on a `d×d` layer. *Sketch:* adapter FLOPs per token `≈ 6·r·2d` (fwd+bwd) vs saved `2d²` per layer; saving dominates when `r < d/6`, i.e. always for typical `r`.
4. *Derive.* For ring all-gather over `N` ranks with `α–β` costs, give the time for a message of `n` bytes per shard, and the message size at which the latency term is 5% of the total. *Sketch:* `(N−1)(α + n/B)`; latency fraction `α B/(n + αB)`; solve for `n` given the node's measured `α` (from 6.1).
5. *Measure.* Predict the exposed-communication fraction of A1 at `B_bus = 200 GB/s` using D3 and the measured compute time, then measure with the profiler; explain the gap.
6. *Measure.* Predict and measure the single-GPU QLoRA fit for Qwen3.8-27B on one A100 40 GB (T2): per-GPU peak and step time (the ledger gives `24.2 GB` at micro-batch 2 and sequence 1,024), and the point at which longer sequences break it. State which Gated DeltaNet tier you used.
7. *Open.* Design the cheapest cluster configuration (GPU count, GPU memory, offload, method) to fully fine-tune a 27B dense model on 200M tokens in under 24 h, justified with the ledger, D2, D3 and a cost table from 5.6 (`lm-hw-prices`). Hint: `8PD = 4.1e19` FLOP is about 122 GPU-hours at an assumed 30% MFU, so memory, not time, sets the GPU count.
8. *Measure (needs an account).* Repeat A1's recipe (same data, `r=16`) on a hosted LoRA fine-tuning service with a model larger than 27B; list what the service lets you set and what it hides, and compare its adapter with A1's on the Chapter 1 suite through 1.7's client with `d2l.paired_bootstrap`. Do not compare losses across different tokenizers.

#### Further reading

- **Papers:** [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) (offload to CPU and NVMe); [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) (new, verified; the design FSDP2 builds on); [QLoRA](https://arxiv.org/abs/2305.14314) (paged optimizers and the 65B/48 GB claim); [ZeRO](https://arxiv.org/abs/1910.02054) (in the book's references); [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) (new, verified; the layer type of the 27B model).
- **Tutorials:** Answer.AI, [You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) (FSDP-QLoRA); PEFT, [FSDP guide](https://huggingface.co/docs/peft/accelerate/fsdp) (new, verified 200); Hugging Face Accelerate, [FSDP guide](https://huggingface.co/docs/accelerate/usage_guides/fsdp) (verified 200); PyTorch [`fully_shard` documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) (verified 200 through redirect); Tinker, [models and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/), [SDK cheatsheet](https://tinker-docs.thinkingmachines.ai/tinker/sdk-cheatsheet/), [OpenAI-compatible inference](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/) and [merging into a Hugging Face model](https://tinker-docs.thinkingmachines.ai/cookbook/deployment/export-hf/) (all verified 200 on 2026-10-29; one hosted instance, not an endorsement); [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) (verified 200).
- **Code:** [torchtune](https://github.com/pytorch/torchtune) (distributed LoRA/QLoRA recipes; frozen); [PEFT](https://github.com/huggingface/peft) (`examples/sft` FSDP scripts); [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) (verified 200); [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (the `fla` kernels; verified 200); transformers' [`modeling_qwen3_5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) and [`hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py) (verified 200; where the kernel tiers are chosen).

#### Risks and open decisions

- **Hardware: node topology.** The T3 node has 8× A100 40 GB; its interconnect (NVSwitch SXM, or PCIe cards with NVLink bridges between pairs) is not known in advance, and `topo.txt` decides which row of `tab_lm_ft_fsdp_comm` applies; the communication estimate and the verdict on H4 change by roughly an order of magnitude between the rows, and the notebook prints both. It must not assume 600 GB/s all-to-all NVLink.
- **Gated DeltaNet kernels under FSDP2 on A100 (the headline's main risk).** The Triton kernels (`fla` or the Hub kernel) are the only practical path (the pure-PyTorch reference is slow, memory-hungry and rejects packing), they compile and autotune per shape on first use, and the interaction with FSDP2 is through dtype handling (the fp32 `A_log` and norm weights under `MixedPrecisionPolicy`), not through sharding. Checks, in order of cost: T1 (kernel vs reference on one layer, test 6), T1 (one-rank FSDP2 vs plain loop, test 3), T2 pair (two-rank vs one-rank, test 3), and only then T3. **Fallback:** `Qwen/Qwen3-32B` (dense, standard attention, Apache-2.0) for A1 and A2 if any of these fails, if gradients contain NaNs in the Gated DeltaNet layers, or if the first steps are far off the ledger because the reference path was silently used. Ledger numbers for the fallback, same conventions (8 GPUs, micro-batch 2, sequence 1,024, `r=16` all-linear = 134,217,728 adapter parameters): bf16 LoRA `14.3 GB` per GPU (`min_gpus` 3 at 40 GB, 2 at 80 GB), full FT at 16 B/param `72.3 GB` (`min_gpus` 21 at 40 GB, 9 at 80 GB, so it does not fit eight 80 GB GPUs), 10 B/param `47.8 GB` (14 and 6); the T3 table keeps both models.
- **Kernel packages on torch 2.15.0.** `flash-linear-attention` 0.6.0 states no `torch` requirement and `transformers>=5.6.2,<6`; `causal-conv1d` 1.7.0 is optional and may have no wheel for torch 2.15.0 (it builds from source; the `F.conv1d` fallback makes it unnecessary); the Hub-kernel path downloads kernels at run time and needs network access. The notebook records the installed versions in `env.json`.
- **Host RAM.** Loading the 53.8 GB text-only weights (11 shards, 55.6 GB in the file) needs rank-0 or sharded loading and at least one shard of host RAM per rank; verify the node's RAM before scheduling. There is no offload arm any more, so the earlier ≥ 400 GB requirement is gone.
- **Third-party hosted API.** The hosted-fine-tuning cells need an account, a key and a service that keeps offering the model; the committed transcript replays at T0, and the section states the pattern, so a provider change edits one paragraph (the facts above were verified on 2026-10-29).
- **FSDP2 + frozen/mixed dtype specifics** (quantized `uint8` parameters in sharded groups; `requires_grad` mixing; fp32 scalars next to bf16 weights) are version-dependent; fallback: the PEFT/Accelerate/bitsandbytes stack with `bnb_4bit_quant_storage_dtype=bfloat16`, which the PEFT guide documents.
- **T2 pair availability.** The two-rank checks need a pair joined by an NVLink bridge; if none is free, they are skipped, the one-rank path and the arithmetic stand, and the report says so (no multi-GPU run falls back to the build box).
- **Time budget.** Two 27B loads plus an 8B load can exceed 2 h; loading is timed apart from the steps.
- **Cost numbers** come from 5.6's cost model over `lm-hw-prices`; this section prints no currency figure of its own.

### 7.9 Merging and retaining capabilities

- **Label:** `sec_lm_merging` · **Tier:** T1 (native; the evaluated adapters are trained by an offline T1 script of ≈ 12 min and cached; the notebook trains one 150-step demo adapter live) · **Budget:** ≤ 15 min: merges (seconds each), ≈ 8 evaluated variants × ≈ 1 min (GSM8K 200 items and SQL 200 items by greedy generation, ARC-Challenge and MMLU-Pro 500-item subsets by log-likelihood scoring), the book-QA comparison ≈ 2 min; peak ≤ 12 GiB. Config flag `CFG='t2'` switches the base to `Qwen3-1.7B-Base` · **Codex units:** 8.8.
- **Builds on (book):** `sec_parameters`/`sec_read_write` (state dicts, safetensors), `sec_gpt` (`d2l.GPT`, trained from scratch for the ancestry demo), `sec_mdl-svd-low-rank` (rank of a sum of low-rank matrices), `sec_mdl-statistics` (bootstrap, via 1.5). Not re-taught.
- **Builds on (this part):** 7.4 (`LoRALinear`, `add_lora`/`apply_lora`, `merge_lora`, `lora_state_dict`, `lora_disabled`), 7.2 (`sft_train`, `render_chat`, `sft_loss`), 7.3 (`lm-sft-mix-v0`, replay data), 7.1 (forgetting protocol, retention suite), 1.4 (`d2l.score_choices`), 1.5 (`d2l.EvalSuite`, `d2l.paired_bootstrap`), 2.4–2.5 (`d2l.BookRetriever.search(q, k) -> list[d2l.Hit]`, `d2l.book_corpus`, `d2l.book_qa(split)` items with `question`, `reference`, `key_facts`, `gold_spans`; `d2l.rag_judge(judge, tokenizer, item, result, context_texts)['correct']` as the scorer, using the A2 judge), 7.6 (merging into a quantized base is lossy).
- **Exports (`#@save`):** all operate on state dicts (`dict[str, Tensor]`), fp32 arithmetic, result cast back to the base dtype once.
  - `d2l.task_vector(sd_ft, sd_base, keys=None) -> dict`: `τ = θ_ft − θ_base` for floating tensors whose shapes match (asserts identical key sets, else `ValueError` naming the first mismatch).
  - `d2l.merge_task_arithmetic(sd_base, tvs, lambdas) -> dict`: `θ₀ + Σ λ_t τ_t`.
  - `d2l.merge_ties(sd_base, tvs, density=0.2, lam=1.0) -> dict`: per tensor and per task keep the top-`density` entries by magnitude (mergekit convention; the TIES paper trims over the whole flattened vector), elect each entry's sign as the sign of the sum of trimmed values, average only the entries that agree with the elected sign (disjoint mean), scale by `lam`.
  - `d2l.dare(tv, p=0.9, seed=0, rescale=True) -> dict`: drop each entry with probability `p`, rescale survivors by `1/(1−p)`; composes with `merge_task_arithmetic` or `merge_ties`.
  - `d2l.slerp(a, b, t, lerp_thresh=0.9995) -> Tensor` (flattened spherical interpolation, falls back to linear interpolation when `|cos| > lerp_thresh`) and `d2l.merge_slerp(sd_a, sd_b, t=0.5, sd_base=None) -> dict` (per tensor; if `sd_base` is given, interpolates the task vectors, else the weights).
  - `d2l.compose_lora(adapters, mode='concat', weights=None) -> dict`: `'concat'` stacks `B=[B₁ … B_k]`, `A=[A₁; …; A_k]` (rank `Σ r_i`, exact sum of `γ_i B_iA_i` when the `γ`s are folded into `B`), `'avg_ab'` averages `A` and `B` separately (the naive method), `'avg_delta'` returns the rank-`Σr` factors of the weighted ΔW sum (same as concat with weights).
  - `d2l.interference_stats(tvs, density=0.2) -> DataFrame`: per module type, pairwise cosine of task vectors, sign-conflict rate among jointly non-zero trimmed entries, energy fraction kept by trimming.
  - Artifacts: `DATA_HUB['lm-ch07-merge-adapters']` (adapters: GSM8K ×3 seeds, SQL ×1, knowledge adapter on the book, replay arms at 0/10/30%, one shared-`A` pair; `r=16`, all-linear, `lora_state_dict` format) and `DATA_HUB['lm-ch07-merge-evals']` (per-item scores of every variant).
- **Models and data:** `Qwen/Qwen3-0.6B-Base` (T1; Apache-2.0); optional `Qwen/Qwen3-1.7B-Base`. Task M: `openai/gsm8k` train (7,473; MIT; fields `question`, `answer` with `#### n`), evaluated on test items by final-number match. Task S: `b-mc2/sql-create-context` (78,577 rows; CC-BY-4.0; fields `question`, `context`, `answer`), seeded 90/10 split, normalized exact match on 200 held-out items (caveat printed: schemas recur across the split). General suite (no overlap with M or S): `allenai/ai2_arc` ARC-Challenge (CC-BY-SA-4.0) and `TIGER-Lab/MMLU-Pro` (MIT), 500-item subsets each, scored with `d2l.score_choices`. Replay data: `lm-sft-mix-v0`. Knowledge task: R3 (`d2l.book_corpus`, `d2l.book_qa(split='dev')`). Ancestry demo: tiny `d2l.GPT` models on a small text corpus (T0).
- **Packages:** `peft` (oracle for adapter composition: `add_weighted_adapter` with `combination_type` `'cat'`, `'ties'`, `'dare_ties'`, `'svd'`; confirm names in the 7.7 pin), `mergekit 0.1.5` (PyPI 2026-10-20; oracle for TIES/DARE/SLERP configs; run on CPU from YAML).

#### Narrative arc

1. **The question.** Opening cell: load the cached M-adapter, merge it, and print GSM8K accuracy (200 items), then the ARC-C/MMLU-Pro scores. Then the SQL adapter. Symptom (to be measured): each adapter raises its own task and lowers the other task or the general suite; and "one model that does both" is the goal. Three candidate routes are named: train jointly (needs both datasets and a new run), serve two adapters, merge weights.
2. **Forgetting and replay.** Recap of 7.1's retention protocol. Replay arms (0%, 10%, 30% of each batch drawn from `lm-sft-mix-v0`, same optimizer steps; cached adapters): table of task score vs general score; also WiSE-FT-style weight interpolation `θ(α) = (1−α)θ₀ + αθ_ft` for `α ∈ {0, .25, .5, .75, 1}`, giving the Pareto curve of task vs retention (`fig_lm_merge_pareto`). Two other retention levers named, not run: lower learning rate/fewer steps (7.1) and LoRA's implicit restriction (7.4; LoRA Learns Less and Forgets Less).
3. **Task vectors and the linearization argument (D1).** `τ = θ_ft − θ₀`; adding, scaling, negating; the additivity diagnostic on logits (`fig_lm_merge_additivity`); Task Arithmetic in the Tangent Space (new, verified) as the explanation of *when* it works (weight disentanglement).
4. **Interference and TIES (D2).** `interference_stats` on the real task vectors: cosines, sign conflicts, energy kept by top-20% trimming; implementation of the three TIES steps; the comparison of trimming to a random mask.
5. **DARE (D2).** Random drop and rescale; unbiasedness and variance; `p ∈ {0.5, 0.9, 0.99}` sweep on output error and on task scores; composition `DARE + TA`, `DARE + TIES`.
6. **SLERP and soups (D3).** Why SLERP of two fine-tunes' weights equals LERP; where it differs (task vectors); a model soup of three GSM8K adapters (uniform average of ΔW) against the individual seeds.
7. **Composing adapters (D4).** Concatenation is exactly the sum; naive averaging of `A` and `B` is not; shared-`A` training makes `B`-averaging exact; Lorahub-style learned mixing described (exercise), `peft.add_weighted_adapter` as the oracle.
8. **Merging fails: ancestry.** T0 demo: two tiny `d2l.GPT` models trained from different seeds on the same data versus two fine-tunes from one init; loss along `θ(α) = (1−α)θ_a + αθ_b`. Different-ancestry merges show a barrier (permutation symmetry); same-ancestry merges do not. Also the shape/key mismatch check of `task_vector` on a different architecture (the R2 reference model).
9. **New knowledge: fine-tune or retrieve (2.5).** Four conditions on 100 dev questions of `d2l.book_qa`: base without context; base + `BookRetriever.search(q, 5)` top-5 passages (`Hit.text` as `context_texts`); base + LoRA trained on the book chunks (3 epochs, cached adapter; no context at test time); both. Metric: `d2l.rag_judge(judge, tokenizer, item, result, context_texts)['correct']` (the A2 judge from 2.5) averaged over the 100 items, with `context_texts=[]` for the no-context conditions and the paired bootstrap over items for differences (open question: which judge model is used here, the A2 default or a larger cached one; the choice is recorded in the artifact); accompanying check of hallucination-style failures (answers with unsupported specifics) by a 20-item manual rubric printed in the notebook. Reading: Ovadia et al. (RAG vs fine-tuning for knowledge injection) and Gekhman et al. (fine-tuning on unfamiliar facts and hallucination) are the cited priors.
10. **Decision table** `tab_lm_merge_choose`: joint training, replay, adapter serving (13.5), merging, retrieval: cost, retention, composability, failure modes.

#### Derivations

**D1. Additivity of task vectors under linearization (`eq_lm_taskvec`).**
Statement: if `f(x; θ)` is replaced by its first-order expansion `f_lin(x;θ) = f(x;θ₀) + J(x)(θ−θ₀)`, then `f_lin(θ₀ + Σ_t λ_t τ_t) − f(θ₀) = Σ_t λ_t [f_lin(θ₀+τ_t) − f(θ₀)]` exactly; for the true `f` the discrepancy is the second-order term `Σ_{s≠t} λ_sλ_t τ_sᵀ H τ_t` (+ diagonal terms common to the single-task models).
Steps: (1) linearity of `J τ` in `τ`; (2) expand `f` to second order, `f(θ₀+τ) = f₀ + Jτ + ½τᵀHτ`; (3) for `τ = λ_Mτ_M + λ_Sτ_S` the quadratic form contains `λ_Mλ_S τ_Mᵀ H τ_S`, absent from the sum of single-task changes; (4) so merging is additive up to the *cross* curvature; small when the task vectors act on disjoint function directions.
Check cell: on 100 prompts compute `ρ = ‖f(θ_{M+S}) − f(θ_M) − f(θ_S) + f(θ₀)‖ / ‖f(θ_M) − f(θ₀)‖` on the final-token logits for `λ=1`; report with a bootstrap over prompts, for full task vectors and for LoRA task vectors (to be measured; `ρ ≪ 1` supports additivity).

**D2. Trimming energy, TIES elections and DARE's unbiasedness (`eq_lm_dare`).**
Statement: (a) for `τ_i ~ N(0,1)` the top fraction `q` of entries by magnitude carries energy `2zφ(z) + q` with `P(|τ|>z)=q`: `q = 0.2` gives `z = 1.2816` and `≈ 65%` of the energy; heavier tails keep more. (b) DARE: `τ̃_i = m_i τ_i/(1−p)`, `m_i ~ Bernoulli(1−p)` independent, gives `E τ̃_i = τ_i` and `Var τ̃_i = τ_i² p/(1−p)`; for a linear layer the output change `Σ_i x_iτ̃_i` has mean `Σ x_iτ_i` and variance `(p/(1−p)) Σ x_i²τ_i²`.
Steps: (a) `E[τ² 1{|τ|>z}] = 2[zφ(z) + (1−Φ(z))]`; (b) `E m = 1−p`; `E m² = 1−p`, so `Var(mτ/(1−p)) = τ²(1−p)(1−(1−p))/(1−p)² = τ² p/(1−p)`; (c) relative output noise `√(p/(1−p))·‖x∘τ‖₂ / |Σ x_iτ_i|` shrinks when many coordinates contribute coherently (large effective width), explaining why very high `p` can work for small, redundant deltas; (d) TIES: trimming first removes small entries that would be summed into conflicting signs; the sign election and disjoint mean prevent cancellation between tasks (the mean is over agreeing entries only).
Check cell: (a) `NormalDist` arithmetic and Monte Carlo; trimmed energy on real task vectors; (b) mask sampling: empirical mean and variance of `τ̃` and of the layer output against the formulas (relative error < 2% with 10⁴ masks); (c) sign-conflict rate vs trim density from `interference_stats`.

**D3. SLERP versus LERP (`eq_lm_slerp`).**
Statement: `slerp(a,b;t) = [sin((1−t)Ω)a + sin(tΩ)b]/sinΩ`, `Ω = arccos(â·b̂)`. For `‖a‖ = ‖b‖` it has norm `‖a‖` for all `t`; LERP has norm `‖a‖√((1−t)²+t²+2t(1−t)cosΩ)`, minimal at `t=½`: `‖a‖cos(Ω/2)`. For small `Ω`, `slerp = (1−t)a + tb + O(Ω²)` with coefficients `(1−t)[1+t(2−t)Ω²/6]` on `a` and `t[1+(1−t²)Ω²/6]` on `b`.
Steps: (1) norm of the combination via `a·b = ‖a‖²cosΩ`; (2) identity `sin((1−t)Ω)² + sin(tΩ)² + 2 sin((1−t)Ω) sin(tΩ) cosΩ = sin²Ω`; (3) series `sin x ≈ x − x³/6`; (4) two fine-tunes of one base differ from each other by `‖τ_a−τ_b‖ ≪ ‖θ‖`, so `Ω ≪ 1` and SLERP of *weights* equals LERP to `O(Ω²)`; (5) task vectors of different tasks can be nearly orthogonal (`Ω ≈ 90°`), where LERP at `t=½` shrinks the norm to `0.71‖τ‖` and SLERP does not.
Check cell: per-tensor `Ω` between the M and S fine-tuned weights (expected small; measured), `‖slerp − lerp‖/‖lerp‖` against the bf16 resolution `2^{−8}`; for task vectors the norm ratios of LERP and SLERP; mergekit's `slerp` as an oracle on 5 tensors (agreement ≤ 1e-6 fp32).

**D4. Composing low-rank adapters (`eq_lm_lora_compose`).**
Statement: `Σ_i γ_iB_iA_i = [γ₁B₁ … γ_kB_k][A₁; …; A_k]` exactly (rank `Σr_i`); averaging factors gives `B̄Ā = (1/k²) Σ_{i,j} B_iA_j`, which contains `k` diagonal terms with weight `1/k²` (not `1/k`) and `k(k−1)` cross terms `B_iA_j`; if all adapters share one frozen `A`, then `B̄A = (1/k)Σ B_iA` exactly.
Steps: block-matrix multiplication; expand the product of sums; independent random `A_i` make cross terms noise-like. Check cell: build adapters M and S, compare `‖ΔW_concat − (γ_MB_MA_M + γ_SB_SA_S)‖` (exact to 1e-6), the diagonal vs cross energy of `avg_ab`, the shared-`A` pair's exactness, and PEFT `add_weighted_adapter(combination_type='cat')` as oracle.

#### From-scratch implementation

Budget ≈ 110 lines: `task_vector`, `merge_task_arithmetic` (20), `merge_ties` (25), `dare` (10), `slerp`/`merge_slerp` (20), `compose_lora` (20), `interference_stats` (15).
- **Tests (seed 0).** (1) `task_vector` + `merge_task_arithmetic(λ=1)` recovers `θ_ft` exactly (fp32); (2) TIES on two hand-built vectors with known signs gives the hand-computed result (a 6-entry table in the notebook); (3) DARE statistics (D2); (4) `slerp` limits: `t=0,1`, antipodal and colinear fallbacks; norm preservation; mergekit oracle; (5) `compose_lora('concat')` exactness; (6) PEFT oracle for `'cat'` and `'ties'` (adapter-level) with tolerance 1e-5 fp32; (7) key/shape mismatch raises.

#### In production

- **mergekit** (YAML: `merge_method: task_arithmetic | ties | dare_ties | dare_linear | slerp`, per-tensor `density`/`weight`/`t` schedules, `dtype`, `base_model`): the notebook writes the equivalent YAML for each from-scratch merge, runs it on CPU for the 0.6B model and compares tensors (identical up to the documented conventions: per-tensor density, mass-based sign election); the from-scratch result is the oracle. MergeKit (new, verified) is the paper.
- **PEFT** `add_weighted_adapter(adapters, weights, adapter_name, combination_type)`, `merge_and_unload()`; **adapter serving** is the alternative to merging (13.5).
- Merging into a quantized base: dequantize, merge, and re-quantize is lossy (7.6); recommend merging in bf16 first.

#### Headline experiment

- **Hypothesis.** (H1) Merging the M and S adapters by task arithmetic retains most of each task's gain with less loss on the general suite than either adapter alone, and the best `λ` is between 0.5 and 1; (H2) TIES and DARE+TIES beat plain task arithmetic on the *worse* of the two tasks, by a margin that can be inside noise; (H3) SLERP of weights equals averaging (D3) and halves both tasks' gains, as does LERP; (H4) replay at 10–30% reduces general-suite forgetting at a small task cost; (H5) for new knowledge, retrieval beats the fine-tuned model without context (the priors of Ovadia et al. and Gekhman et al.), with the combination best or equal.
- **Setup.** Base `Qwen3-0.6B-Base`; cached adapters (`r=16`, all-linear, α=32, `lr` from 7.4's optimum, 2 epochs for M, 1 epoch for S); variants: base; M; S; joint-trained (cached, both datasets mixed 1:1); weight average of the two merged models (soup form); task arithmetic `λ ∈ {0.3, 0.5, 0.7, 1.0}`; TIES (`density 0.2`, `λ=1`); DARE+TA (`p=0.9`); DARE+TIES; SLERP (weights); SLERP (task vectors); concat adapter; `avg_ab` adapter. Evaluate each on GSM8K (200), SQL (200), ARC-C (500), MMLU-Pro (500).
- **Procedure.** Merge in fp32, cast to bf16, evaluate with the same prompts and decoding (greedy, 256 new tokens for GSM8K, 128 for SQL). DARE with 3 seeds for the mask.
- **Metrics.** Task accuracies; general-suite mean; "retention" (general score / base general score); additivity ratio `ρ`; interference statistics.
- **Plot spec.** `fig_lm_merge_scatter`: x = GSM8K accuracy, y = SQL accuracy, one marker per variant (merge methods colored by family, joint training as a star, base as a cross), marker size = general-suite score; error bars from item bootstrap on both axes. `fig_lm_merge_pareto`: interpolation curves of task vs general score. `fig_lm_merge_ancestry`: loss vs `α` for same-ancestry and different-ancestry pairs (T0). `tab_lm_merge_qa`: the four knowledge conditions.
- **Expected outcome.** Directions only, from sources: Task Arithmetic and TIES report that merging task vectors can give multi-task models, with TIES reducing interference; DARE reports that most delta weights can be dropped and rescaled; Model Soups reports that averaging fine-tunes of one init can improve accuracy. Magnitudes to be measured; a merge that is worse than both singles on its own task is reported plainly.
- **Noise check.** Evaluation noise from item bootstrap (`d2l.paired_bootstrap`, paired across variants on the same items); training noise from the three GSM8K seeds (soup experiment) and the three DARE masks; a variant is called better only if the paired interval excludes 0 and the difference exceeds the seed spread; with 200 items per task the notebook prints the interval half-width (several points) so readers see how coarse the comparison is.
- **T1 reduction.** Native; 1.7B is the `t2` flag.
- **Artifacts committed.** `lm-ch07-merge-adapters`, `lm-ch07-merge-evals`.

#### Figures and tables

- `fig_lm_merge_scatter`, `fig_lm_merge_pareto`, `fig_lm_merge_additivity` (histogram of `ρ`), `fig_lm_merge_ancestry`.
- `tab_lm_merge_interference`: cosine, sign conflicts, trimmed energy by module type.
- `tab_lm_merge_replay`: replay ratio × (task, general).
- `tab_lm_merge_qa`: four knowledge conditions with intervals; `tab_lm_merge_choose`.

#### Pitfalls and misconceptions

- Merging models of different ancestry (different init or tokenizer): meaningless without alignment (permutation symmetry); `task_vector` refuses mismatched keys and shapes.
- Averaging LoRA `A` and `B` separately (D4); merge the product or concatenate.
- Using `λ=1/k` by habit: task arithmetic *adds* vectors; the right scale is tuned on a validation set that is not the report set.
- Tuning merge hyperparameters on the evaluation items (200-item evals overfit quickly).
- SLERP is not "better averaging" for fine-tunes of one base (D3).
- Merging in bf16 loses small deltas (7.4 D3); merge in fp32.
- Judging knowledge injection by training-loss decrease; test on held-out questions with and without retrieval.
- Forgetting measured only on the training-task format.

#### Summary (draft bullets)

- Forgetting is a trade-off to be measured: replay and weight interpolation move along it.
- Task vectors add approximately when the model is close to its linearization; the cross curvature measures interference.
- TIES removes sign conflicts after trimming; DARE drops and rescales so that the delta is unbiased.
- SLERP of nearby weights is linear interpolation; it matters for far-apart vectors.
- Adapters compose exactly by concatenation, not by averaging factors.
- New facts are better supplied by retrieval than injected by fine-tuning, unless the knowledge must be available without context.

#### Exercises

1. *Check.* Two adapters of rank 16 are concatenated. What rank does the result have and how many parameters per `d×d` layer? *Sketch:* rank 32; `32·2d` parameters.
2. *Check.* Why can't you merge a fine-tune of `Qwen3-0.6B-Base` with one of the R2 reference model? *Sketch:* different shapes/tokenizer/ancestry; `task_vector` raises.
3. *Derive.* Prove that for `t = ½` and unit vectors at angle `Ω`, LERP has norm `cos(Ω/2)`; find the angle at which the norm falls to 0.9. *Sketch:* `‖(a+b)/2‖² = (2+2cosΩ)/4 = cos²(Ω/2)`; `cos(Ω/2)=0.9 ⇒ Ω ≈ 51.7°`.
4. *Derive.* Show that DARE with rescaling is unbiased but that without rescaling the expected output is scaled by `1−p`; give the variance for `p=0.9`. *Sketch:* `E[mτ] = (1−p)τ`; variance factor `p/(1−p) = 9`.
5. *Measure.* Predict the `λ` that maximizes the mean of the two task accuracies for task arithmetic, sweep `λ` on a validation split, and report the test value with an interval.
6. *Measure.* Predict whether a soup of 3 GSM8K seeds beats the best single seed; measure with the paired bootstrap and the seed spread.
7. *Open.* Implement LoraHub-style composition: learn mixture weights over 4 adapters with a gradient-free optimizer on 20 examples of a new task; compare with a trained adapter.

#### Further reading

- **Papers:** [Task Arithmetic](https://arxiv.org/abs/2212.04089); [TIES](https://arxiv.org/abs/2306.01708); [DARE](https://arxiv.org/abs/2311.03099); [Model Soups](https://arxiv.org/abs/2203.05482); [Task Arithmetic in the Tangent Space](https://arxiv.org/abs/2305.12827) (new, verified); [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903) (new, verified; weight interpolation); [LoraHub](https://arxiv.org/abs/2307.13269) (new, verified); [Arcee's MergeKit](https://arxiv.org/abs/2403.13257) (new, verified); [Fine-Tuning or Retrieval?](https://arxiv.org/abs/2312.05934) (new, verified); [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904) (new, verified).
- **Tutorials:** [PEFT docs](https://huggingface.co/docs/peft/index) (adapter merging guide); Hugging Face smol course.
- **Code:** [mergekit](https://github.com/arcee-ai/mergekit); [PEFT](https://github.com/huggingface/peft).

#### Risks and open decisions

- **Eval size.** 200-item evals make method differences hard to resolve; fallback: cache larger T2 evals as artifacts.
- **Judge choice for book QA.** Scoring uses 2.5's `d2l.rag_judge` with the A2 judge (Qwen3.5-9B, bf16); on 100 items its errors are not negligible, so the 20-item manual rubric doubles as a judge-agreement check; open: switch to a larger cached judge if agreement is low.
- **Tasks M and S are too different to interfere strongly**; if TIES/DARE show no gain, say so and keep the interference diagnostics; alternative pair (GSM8K and a second math-style task) as an exercise.
- **mergekit on CPU with the pinned environment** may need its own env; fallback: the from-scratch merges only.
- **SQL split leakage** (shared schemas): reported as a caveat; a schema-disjoint split is the stricter alternative.
