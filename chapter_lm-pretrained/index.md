# Working with Pretrained Language Models
:label:`chap_lm_pretrained`

The reader has trained the GPT of :numref:`sec_gpt` on a few hundred megabytes of text and has seen released models answer questions that model cannot. This chapter loads such a model, runs the same tensors in the book's own code, scores it under explicit conventions, opens it, and reaches the models that no local machine can hold.

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "Chapter 1 Working with Pretrained Language Models").
Blueprint: docs/llm/blueprints/ch01_pretrained.md.
The plan is a specification, not book text; this comment is replaced
when the chapter is written.

**Label:** `chap_lm_pretrained`. Seven sections, T0 to T1. The chapter turns released checkpoints into instruments and builds the scoring and comparison tools that every later notebook of the part imports.
**Hook:** The reader has trained the GPT of `sec_gpt` on a few hundred megabytes of text and has seen released models answer questions that model cannot. This chapter loads such a model, runs the same tensors in the book's own code, scores it under explicit conventions, opens it, and reaches the models that no local machine can hold.
**Route:** What is in a released checkpoint, and how much memory and speed follow from its configuration file (1.1)? Do its tensors run the same in the book's code, layer by layer (1.2)? How much of a score does the prompt decide (1.3)? Can two scoring conventions rank two models differently (1.4)? When does a measured difference survive a paired test, and what does a judge add (1.5)? How is a claim about what a model computes tested rather than read off (1.6)? What can a hosted model be made to measure, and what does it withhold (1.7)?
**Out of scope:** No fine-tuning (`chap_lm_finetuning`), no serving beyond one local endpoint (`chap_lm_serving`), no architecture beyond reading a configuration (`chap_lm_architectures`); usage, pinning, licences and buying stay in the Tools appendix (`sec_software_ecosystem`, `sec_model_serving`, `sec_hardware_buyers`), which this chapter links and never absorbs.
**Writer notes:**
- Running models (Table 5): the current small Qwen generation is Qwen3.5 (0.8B, 2B, 4B; Apache-2.0); the book's own code runs the Qwen3 dense ladder (0.6B, 1.7B, 4B, 8B), chosen because Qwen3 is the last pure-transformer Qwen ladder; Gemma 4 E2B is the one deliberate second family (1.1 comparison, exercise elsewhere); the hosted models of 1.7 are DeepSeek-V4.1-Flash-1017 and Qwen3.8-27B, reached only through the API; the judge is Qwen3.5-4B in bf16 (not the Qwen3.5-9B judge of blueprint ch01 Decision 5, which needs a 24 GB card). Every memory claim says what fits a 15 GiB scheduler slot; no T1 notebook claims a whole 24 GB card.
- Shared artifacts the chapter produces: the Qwen3-0.6B checkpoint loaded into the book's code (1.2; `d2l.Qwen3GPT`, `d2l.load_qwen3`), the harness (1.4 `d2l.score_choices`, `d2l.pass_at_k`, `d2l.extract_answer`, `d2l.bits_per_byte`; 1.5 `d2l.EvalSuite`, `d2l.paired_bootstrap`, `d2l.judge_pairwise`), the client `d2l.ChatAPI` (1.7), `d2l.load_hf`, `d2l.config_memory`, `d2l.config_params` (1.1), `d2l.fewshot_prompt` (1.3), `d2l.intervene`, `d2l.TopKSAE` (1.6). Signatures are fixed by blueprint `01_interfaces.md` §1 and are not renegotiated by the author; `d2l.generate` is the batched text-in, text-out sampler of 1.2 and the book's old `generate(step_fn, ...)` of `sec_decoding` is renamed `d2l.generate_ids` (blueprint README Decision 3, a book-side edit that must land first).
- Book-side edits this chapter depends on: the `generate_ids` rename; `sec_kv-cache`'s `generate_cached` cell marked `#@save` (required, so `d2l.Qwen3GPT` overrides only `forward_cached`); `data/lm_revisions.json` with pinned commit hashes for every model id, re-resolved on the authoring date.
- Assignment A1 (run, port and measure; instructor-facing, not book text): logits of the port within a stated bf16 tolerance; the harness reproduces one published score (Qwen3-0.6B-Base on MMLU, 5-shot) within its interval. Starter package, autograder tolerances and the T1 path are in blueprint ch01 "Assignment A1"; keep them, with the judge and the local endpoint moved down to Qwen3.5-4B.
- Compute: 1.1 to 1.6 are T1 with T0 fallbacks on committed artifacts; 1.7 makes live paid API calls through OpenRouter (or a similar aggregator) under a fixed per-build budget and records the responses, so the notebook also replays them at T0. Live API calls need a token held outside the repository (read from an environment variable, never committed).
- Consistency points: the RL notation table is `tab_rl_notation` (owner `sec_mdp`; use exactly this label); benchmark names live in a per-edition data file (`data/lm_benchmarks.json`), never in prose; the chapter uses `enable_thinking=False` for every Qwen3.5 helper role and says once that Qwen3 post-trained checkpoints default to thinking on; every comparison of two numbers carries a paired interval from 1.5 or, before 1.5, a short local bootstrap that the text names as provisional.
**Teaching sources:**
- [CS336, Language Modeling from Scratch](https://cs336.stanford.edu/): the evaluation lectures and their habit of reporting what was held fixed; the model for 1.4's convention table and 1.5's protocol.
- [CS224N](https://web.stanford.edu/class/cs224n/): assignment A4 (LLM benchmarking and evaluation) is the course-side counterpart of 1.4 and 1.5; the interpretability guest lecture pairs with 1.6.
- [CMU 11-711 Advanced NLP](https://cmu-l3.github.io/anlp-spring2026) and [Berkeley CS288](https://cal-cs288.github.io/sp26): both give a full lecture to experimental design and human annotation; 1.5's reproduce-with-error-bars protocol is written to serve their reproduction assignments.
- [Hugging Face LLM Course, chapter 2](https://huggingface.co/learn/llm-course/chapter2/1): the Transformers API tour; assign before 1.1 for readers new to the Auto classes.
- [NeurIPS 2024 tutorial on evaluating LLMs](https://neurips.cc/virtual/2024/tutorial/99524) and [ICML 2024 tutorial on LM evaluation](https://icml.cc/virtual/2024/tutorial/35227): the evaluation landscape behind 1.5's lifecycle paragraph.
- Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/): a small development suite built from real failures; 1.5's development-versus-reporting distinction.
- Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): batch-dependent kernels as the source of run-to-run drift; the basis of 1.4's determinism controls and 1.7's "temperature 0 is not determinism".
- [ARENA](https://github.com/ARENA-education/ARENA_materials) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens): the interpretability curriculum and library 1.6 follows in spirit; 1.6 implements the operations with plain hooks so the code is not tied to a library's model support.
- [openai-python](https://github.com/openai/openai-python): the client `d2l.ChatAPI` wraps in 1.7.
**Further Reading:** none.
-->

```toc
:maxdepth: 2

hf-models
d2l-checkpoint
prompting
scoring
eval-suites
interpretability
api
```
