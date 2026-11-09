# Efficient Inference
:label:`chap_lm_inference`

The reader has a compressed model from Chapter 11 and, from :numref:`sec_kv-cache` and :numref:`sec_hardware`, a bound on how fast one GPU can decode it; the plain generation loop reaches a small fraction of that bound. This chapter takes one request, or a handful, from the eager loop to the bound and then past its per-token form: the kernels that read the weights and the cache at full bandwidth, the reduction orders that make outputs reproducible, the verification trick that checks several tokens per weight read, and the cache compressions that a trained model tolerates. It ends on a laptop, where the same bound holds with a tenth of the bandwidth.

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "Chapter 12 Efficient Inference").
Blueprint: docs/llm/blueprints/ch12_inference.md.
The plan is a specification, not book text; this comment is replaced
when the chapter is written.

**Label:** `chap_lm_inference` (8 sections; T0 to T2, every T2 section has a reduced T1 configuration).
**Hook:** The reader has a compressed model from Chapter 11 and, from `sec_kv-cache` and `sec_hardware`, a bound on how fast one GPU can decode it; the plain generation loop reaches a small fraction of that bound. This chapter takes one request, or a handful, from the eager loop to the bound and then past its per-token form: the kernels that read the weights and the cache at full bandwidth, the reduction orders that make outputs reproducible, the verification trick that checks several tokens per weight read, and the cache compressions that a trained model tolerates. It ends on a laptop, where the same bound holds with a tenth of the bandwidth.
**Route:** Where do the milliseconds of a decode step go, and what removes each cost (12.1)? Why does attention over a long cache leave most of the GPU idle, and how is the cache read in parallel (12.2)? Why does the same request return different tokens in different batches, and what fixes the reduction order (12.3)? Can a cheap model propose tokens that the target verifies without changing its output (12.4)? What makes a good drafter, and what does a prediction head trained with the model add (12.5)? How many bits do keys and values need, and why do they differ (12.6)? Which cache entries can be dropped or skipped, and how is that judged beyond one needle (12.7)? What changes when the memory system is a laptop's (12.8)?
**Out of scope:** Scheduling across many requests, paged memory, prefix caching and everything that happens between two engine steps belongs to Chapter 13. Architectural changes that need pretraining (latent attention, learned sparse attention, hybrids) are Chapter 14; this chapter changes nothing before training, and 12.5's multi-token head is only used, not trained from scratch. Weight quantization is Chapter 11. How to run vLLM, SGLang, llama.cpp or Ollama as a user, and which to pick, stays in `sec_model_serving` and `subsec_model-serving-local`, which this chapter links.
**Writer notes:** The running model is the Qwen3-0.6B checkpoint loaded into the book's own code (`d2l.load_qwen3`, a `d2l.Qwen3GPT` with QK-norm, explicit `head_dim` 128, RoPE base `1e6`, interleaved RoPE after 1.2's load-time permutation; module names from `01_interfaces.md`: `token_emb`, `blks[i].attention` with `W_q, W_k, W_v, W_o, q_norm, k_norm`, `blks[i].ffn` with `W_g, W_1, W_2`, `norm1`, `norm2`); Qwen3-4B is the target of 12.4 and the long-context model of 12.6 and 12.7 (8.05 GB in BF16, 144 KiB of KV per token); Qwen3.5-2B appears only for its shipped prediction head in 12.5. The chapter's shared code is `d2l.FastGenerator` (12.1), whose normative API (`alloc/free/reset`, `forward/advance/rollback`, `prefill/decode_step/verify/commit`, `generate(prompts, max_new_tokens, *, sampling, eos_id, speculator, sync_every, return_stats)`, `warmup`, per-slot `pos` and `kv_len`, the `KVCache` protocol with `append/view/copy_slot/copy_cols/reset/nbytes`, and `KVPolicy.make_cache`) every later section and Chapter 13 build on without touching model code; 12.2 supplies its decode attention kernel, 12.3 its deterministic mode (`d2l.det_mode`, owned here and imported by 13.2), 12.4 the `Speculator` protocol, 12.6 and 12.7 cache backends and policies. The byte accounting of the chapter is `d2l.decode_ledger(config, weights, group_size, quantize_head, kv_len, batch, kv_bits)` (12.1), cross-checked against `d2l.config_memory` (1.1) within 1%; every speed is plotted against the bound recomputed from that ledger, so the remaining gap is always attributable. The decode step-time model is `d2l.ServingModel(cfg, chip).decode_step(b, L)` from the hardware chapter (5.3), and `d2l.chip_specs()` supplies `mem_bw_tbs` and `p16_tflops`; there is no `d2l.decode_step_time`. Timing follows `sec_perf_model`: `d2l.Benchmark`, warm-up, synchronization, interleaved repetitions with min and max reported, and a rule that discards a repetition during which `nvidia-smi` showed another process on the shared build-box GPU. Assignment A10 (the inference engine, hosted by 13.3) takes its hooks from this chapter: `FastGenerator.forward` with per-row positions (12.1), the paged decode kernel's reference (12.2), `det_mode` for the batch-invariance checks (12.3), and the `Speculator` and `d2l.spec_greedy_check` (12.4); the single-stream numbers of 12.1 are the baseline row of the A10 report. PyTorch only, said once in 12.1; the Triton kernels of 12.2 and 12.3 are the only cells allowed over 30 lines, because a kernel is one object. The chapter's prose quotes published speedups only with the model, hardware and batch they were measured on, and states its own numbers as measured.
**Teaching sources:**
- PyTorch, [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/): the template for 12.1's sequence of optimizations (eager, compiled, int8, int4, speculative) and its bandwidth-utilization argument; its speculative section is reproduced qualitatively in 12.4.
- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html): the compute, bandwidth and overhead taxonomy that 12.1 applies to a decode step.
- PyTorch, [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/): the split-KV idea of 12.2 with its component-level measurements.
- Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): the batch-invariance argument and kernels of 12.3, and its section on on-policy reinforcement learning.
- Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/): the map of the chapter (quantization, speculation, cache tricks).
- [ICML 2024 tutorial, Towards Efficient Generative LLM Serving](https://icml.cc/virtual/2024/tutorial/35229) and the [EMNLP 2025 tutorial on efficient inference](https://aclanthology.org/2025.emnlp-tutorials.1/): the algorithm-to-system framing and the taxonomy of drafters and cache methods (12.4 to 12.7).
- [Stanford CS336, Language Modeling from Scratch](https://cs336.stanford.edu/) (its inference lecture and the Triton kernel of its systems assignment) and [CMU 11-868 LLM Systems](https://llmsystem.github.io/llmsystem2026spring/) (generation and speculative decoding, GPU acceleration, attention sinks): the two courses whose inference material the chapter covers at book depth.
- [MIT 6.5940](https://hanlab.mit.edu/courses/2026-fall-65940): the lab that deploys a quantized model on a laptop, the model for 12.8.
- Alex Smola, [Efficiency in LLMs](https://alex.smola.org/posts/45-mlss-efficiency/): the weight-against-cache accounting and the maximal-coupling picture of speculative sampling (12.4).
**Further Reading:** none.
-->

```toc
:maxdepth: 2

fast-decoder
decode-kernels
deterministic-inference
speculative
drafters
kv-quantization
kv-eviction
on-device
```
