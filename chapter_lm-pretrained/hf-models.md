# Pretrained Models with Hugging Face
:label:`sec_lm_hf_models`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "1.1 Pretrained Models with Hugging Face · T1").
Blueprint: docs/llm/blueprints/ch01_pretrained.md section 1.1.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A released model is a configuration file plus tensors, and its memory and speed follow from the configuration.
**Opening:** The reader has `d2l.GPT` and its GPT-2 weights. Three lines load Qwen3.5-2B and it answers a question the book's model cannot. The section reads the repository until every field of the configuration maps to a section of this book and to a number of bytes, and it closes on the question of whether those same tensors would run in the book's own code.
**Teaches:**
- Repository anatomy: the model card, `config.json`, sharded safetensors with their index, the tokenizer files, `generation_config.json`; a revision is a commit hash, and a pinned load passes it to both model and tokenizer.
- The Auto classes, `dtype` and `device_map`, `generate` and its sampling fields mapped onto the sampler of `sec_decoding`; one chat template serves base, instruct and thinking variants, and the rendered token sequence is printed so the text cannot drift from the file.
- Qwen3.5's configuration read field by field: Gated DeltaNet layers with one full-attention layer in four (`sec_deltanet`, `sec_hybrids`, read as fields only), gated attention and partial RoPE, a multi-token-prediction head, a vision tower; Qwen3.6 and Qwen3.8 are the same architecture at larger scale, said in one sentence.
- Gemma 4 E2B as the second family: per-layer embeddings, sliding windows interleaved with global layers, text-only loading; the same derivation applied to a different configuration.
- Pinning, licences and `trust_remote_code` in two sentences with a link to `sec_software_ecosystem`.
**Derivation:** Bytes to run a model from its configuration: weights (parameters times bytes per element, counted by group so the sum reconciles to the safetensors header), KV cache per token for full-attention layers (2 × layers × KV heads × head dimension × bytes), the fixed recurrent state of linear-attention layers (independent of context), the activation workspace of one forward pass and the logits term. Smallest numeric case: one full-attention layer with 2 KV heads of head dimension 256 in bf16 caches 2 × 2 × 256 × 2 = 2,048 bytes per token; a hybrid with one such layer in four and a 19.5 MB fixed state crosses an all-attention stack of the same width at a context the reader solves for.
**Notebook:**
1. Load Qwen3.5-2B pinned to its commit with `d2l.load_hf`; read the safetensors header by HTTP range request without downloading tensors (`d2l.safetensors_header`); reconcile the header's element count to `d2l.config_params(by_group=True)` (language model, vision tower, MTP head) exactly.
2. Derive the memory terms with `d2l.config_memory`; prefill contexts of 256, 1,024, 4,096 and 16,384 tokens from a committed passage and compare the cache walker's bytes and the allocator's growth with the formula.
3. Time 64 greedy decode steps at each context with `d2l.Benchmark` (3 warm-ups, 10 repeats, median and interquartile band) and plot tokens per second against the bandwidth ceiling of `eq_hw_decode_bound` with the card's datasheet bandwidth.
4. Repeat steps 1 to 3 for Gemma 4 E2B (text-only); the per-layer embedding tables enter the weight term, the sliding layers cache at most the window.
5. Claim that survives noise: measured memory is within 10% of the derived total at every context for both models (the allocator's block rounding is the gap); measured decode speed is within a factor of 2 of the bandwidth ceiling, with the gap's size stated as measured and the launch overhead counted with the profiler in one decode step.
6. Reused pieces: `d2l.Benchmark`, `d2l.sample_next` as the semantic reference for Transformers' sampling arguments; the KV byte formula `eq_kv-cache-bytes` recapped in one paragraph.
**Figures:** `fig_lm_hub_repo` (what a repository holds and which file each loader reads); `fig_lm_memory_vs_config` (plot: cache bytes and decode tokens per second against context for both models, formula lines and the all-attention counterfactual).
**Restate / forward:** `subsec_hw-two-workloads` (decode reads every weight once per token, so its speed is a bandwidth ratio); `eq_kv-cache-bytes` (the KV cache grows linearly in context); `sec_software_ecosystem` (pinned revisions, licences, why `trust_remote_code` is off). Forward: 14.1 reads whole families this way.
**Exercises:** 6 to 8. Load-bearing: derive Gemma 4 E2B's KV bytes per token from its configuration and the context at which its cache equals the per-layer embedding tables (derive); measure decode tokens per second against batch size 1 to 32 and locate the batched-decode crossover (measure); check the parameter count of Qwen3-0.6B's post-trained file against its base file and explain the duplicate head (check); open: at what context length does a hybrid's cache plus state equal an all-attention stack's, and what does that say about long-context serving.
**Writer notes:**
- Library exports with fixed signatures (`01_interfaces.md`): `d2l.load_hf(model_id, revision=None, dtype=torch.bfloat16, device_map='auto') -> (model, tokenizer)` always passes `revision` to both objects, sets `trust_remote_code=False`, `use_safetensors=True`, warns if `hf_device_map` spreads modules over devices (silent CPU offload is the performance trap); `d2l.config_params(config, include_vision=False, include_mtp=False, by_group=False)` raises `NotImplementedError('use d2l.config_stats (14.1)')` on MoE configs; `d2l.config_memory(config, seq_len, batch_size, dtype_bytes=2, *, state_bytes=4, ...)` returns the keys `params, weights, kv_cache, recurrent_state, activations, logits, total` and reads `layer_types` (`full_attention`, `sliding_attention` with `sliding_window`, `linear_attention`); `d2l.HF_REVISIONS` loads `data/lm_revisions.json`.
- Gemma 4 adds two configuration shapes `config_memory` must handle before authoring: per-layer embedding tables (weights, not activations) and sliding layers; the blueprint's `config_memory` was written for Qwen3.5 only, so extend it and add a test per model. Numbers on the Gemma 4 cards are secondary in `sota.md`; read sizes and windows off the card at authoring.
- Silent breakages: `generation_config.json` may set sampling (Qwen3.5-2B has none, so defaults are greedy); the state dtype of the live cache (the config's `mamba_ssm_dtype` is float32) must be read off the cache, not assumed; `torch_dtype` versus `dtype` across Transformers versions; `model.parameters()` deduplicates tied weights and `state_dict()` does not; the MTP weights are loaded but unused by `generate` (12.5 uses them).
- Transformers version drift: Qwen3.5 modeling code is recent (`transformers >= 5.2`); the cache-walker cell is written against `cache.get_recurrent_state` and `get_conv_state` with a generic fallback that sums every tensor found by traversal. Print the bound kernel (`chunk_gated_delta_rule` fast versus the torch fallback) and run either way.
- Do not re-teach: BPE (`sec_text-sequence`; print the two vocabularies' sizes and tokens per character only), the sampler (`sec_decoding`), safetensors as a format (`sec_read_write`), the decode bound (`sec_hardware`), the DeltaNet recurrence (`chap_modern_rnn`). The section makes the one forward reference to 14.1 and no field-by-field forward references.
- Anecdotes stay anecdotes: prompts compared across thinking modes are labelled as such; no accuracy claim from them.
- Committed artifacts: `data/lm_revisions.json`; the Qwen3.5-2B header (`lm-qwen35-2b-header.json`, about 100 KB) so the reconciliation runs offline; the measured table behind the figure.
- Confirm at authoring the exact Gemma 4 E2B id to pin (`google/gemma-4-E2B` or `-it`; the instruct variant serves the chat comparison).
**Sources:**
- Papers: [Transformers: State-of-the-Art Natural Language Processing](https://arxiv.org/abs/1910.03771) (the library's design); [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (Table 1 for the dense architectures that 1.2 ports; the thinking switch).
- Code and documentation: [Transformers documentation](https://huggingface.co/docs/transformers/index) and its [quick tour](https://huggingface.co/docs/transformers/quicktour); [generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) (the sampling fields mapped onto `sec_decoding`); [model cards](https://huggingface.co/docs/hub/model-cards); [safetensors](https://huggingface.co/docs/safetensors/index) (header format, read by range request); [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (counting launches in one decode step); model cards [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) (sizes, windows, per-layer embeddings: read at authoring); [Qwen3.8 README](https://github.com/QwenLM/Qwen3.8) (the larger models of the same architecture).
- Courses, tutorials and blogs: [Hugging Face LLM Course, chapter 2](https://huggingface.co/learn/llm-course/chapter2/1) (the Auto-class tour); [CS336](https://cs336.stanford.edu/) (resource accounting lecture: FLOPs, memory, arithmetic intensity).
-->

```{.python .input #hf-models-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

<!-- Placeholder prose (stage 2): the section text is written after the runs. -->
A model id names a repository whose files change over time; a revision names
one commit of it. `HF_REVISIONS` pins every model this chapter loads to the
commit it was written against, and later sections add their own entries to
it. The commits were resolved with the Hub API on 2026-11-12.

```{.python .input #hf-models-revisions}
%%tab pytorch
#@save
HF_REVISIONS = {
    'Qwen/Qwen3-0.6B': 'c1899de289a04d12100db370d81485cdf75e47ca',
    'Qwen/Qwen3-0.6B-Base': 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
    'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
    'Qwen/Qwen3-1.7B-Base': 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1',
    'Qwen/Qwen3-4B': '1cfa9a7208912126459214e8b04321603b3df60c',
    'Qwen/Qwen3-8B': 'b968826d9c46dd6066d109eabc6255188de91218',
    'Qwen/Qwen3.5-0.8B': '2fc06364715b967f1860aea9cf38778875588b17',
    'Qwen/Qwen3.5-0.8B-Base': 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68',
    'Qwen/Qwen3.5-2B': '15852e8c16360a2fea060d615a32b45270f8a8fc',
    'Qwen/Qwen3.5-4B': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    'google/gemma-4-E2B-it': '4adf15748de9e1a0097b6274cee9c15836c41c37'}
```
