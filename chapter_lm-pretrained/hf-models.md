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

```{.python .input #hf-models-imports-extra}
%%tab pytorch
import copy
import json
import transformers
from huggingface_hub import HfApi, file_exists
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig)
```

<!-- Placeholder (stage 2): a shard's header is read by two range requests. -->

```{.python .input #hf-models-safetensors-header}
%%tab pytorch
#@save
def safetensors_header(model_id, revision=None, timeout=30):
    """Shape and dtype of every tensor of a Hub checkpoint, read by HTTP
    range requests without fetching tensor data (in production:
    `huggingface_hub.get_safetensors_metadata`)."""
    import urllib.error
    import urllib.request
    import huggingface_hub
    revision = revision or d2l.HF_REVISIONS.get(model_id)
    if revision is None:
        revision = huggingface_hub.HfApi().model_info(model_id).sha
        print(f'unpinned: {model_id}@{revision}')
    token = huggingface_hub.get_token()  # for gated repositories

    class HubRedirect(urllib.request.HTTPRedirectHandler):
        """Keep the token on redirects within the Hub, not to the CDN."""
        def redirect_request(self, request, *args):
            new = super().redirect_request(request, *args)
            auth = request.unredirected_hdrs.get('Authorization')
            if new and auth and new.host == request.host:
                new.add_unredirected_header('Authorization', auth)
            return new

    opener = urllib.request.build_opener(HubRedirect)

    def fetch(filename, start=None, stop=None):  # bytes start, ..., stop - 1
        request = urllib.request.Request(huggingface_hub.hf_hub_url(
            model_id, filename, revision=revision))
        if start is not None:
            request.add_header('Range', f'bytes={start}-{stop - 1}')
        if token:  # unredirected: dropped when a redirect leaves the Hub
            request.add_unredirected_header('Authorization', f'Bearer {token}')
        with opener.open(request, timeout=timeout) as response:
            if start is None or response.status == 206:
                return response.read()
            return response.read(stop)[start:]  # the server ignored the range

    try:  # a sharded checkpoint lists its shards in an index
        index = json.loads(fetch('model.safetensors.index.json'))
        shards = sorted(set(index['weight_map'].values()))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        shards = ['model.safetensors']
    tensors = {}
    for shard in shards:
        n = int.from_bytes(fetch(shard, 0, 8), 'little')  # header length
        header = json.loads(fetch(shard, 8, 8 + n))
        header.pop('__metadata__', None)
        tensors.update({name: (tuple(t['shape']), t['dtype'])
                        for name, t in header.items()})
    return tensors
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

<!-- Placeholder prose (stage 2): the section text is written after the runs. -->
The files this part of the book records itself (measured tables, reference
logits, cached API responses) are published once under a versioned name and
never change; `DATA_HUB` pins each to its SHA-1 under `LM_DATA_URL`.

```{.python .input #hf-models-lm-data-url}
%%tab pytorch
#@save
LM_DATA_URL = 'https://d2l-data.smola.org/lm/'
```

<!-- Placeholder (stage 2): the pinned load, its text model and its report. -->

```{.python .input #hf-models-load-hf}
%%tab pytorch
#@save
_TEXT_KEY_MAPPING = {  # checkpoint key renames for a multimodal file's text
    'gemma4_text': {r'^model\.language_model\.': 'model.'}}

#@save
def load_hf(model_id, revision=None, dtype=torch.bfloat16, device_map='auto'):
    """A pinned causal language model from the Hugging Face Hub, text part
    only, in eval mode, and its tokenizer; prints a load report. The
    tokenizer's end token joins the stop tokens of `generate`."""
    import collections
    import warnings
    import transformers
    if revision is None:
        revision = d2l.HF_REVISIONS.get(model_id)
    if revision is None:
        import huggingface_hub
        revision = huggingface_hub.HfApi().model_info(model_id).sha
        print(f'unpinned: {model_id}@{revision}')
    if device_map == 'auto':  # one GPU: everything on it; none: the CPU
        n_gpus = torch.cuda.device_count()
        device_map = ('cpu' if n_gpus == 0 else {'': 0} if n_gpus == 1
                      else 'auto')
    pin = dict(revision=revision, trust_remote_code=False)
    config = transformers.AutoConfig.from_pretrained(model_id, **pin)
    text = config.get_text_config()  # the language model's configuration
    renames = {}
    if text is not config and text.model_type in _TEXT_KEY_MAPPING:
        renames['key_mapping'] = _TEXT_KEY_MAPPING[text.model_type]
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, config=text, dtype=dtype, device_map=device_map,
        use_safetensors=True, output_loading_info=True, **pin, **renames)
    if info['missing_keys']:  # these modules would stay randomly initialized
        del model  # a kept traceback would otherwise hold its memory
        raise RuntimeError(f'no weights for {sorted(info["missing_keys"])}')
    model.revision = revision
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **pin)
    # Without generation_config.json, `generate` stops only at the
    # configuration's end token, which need not be the one ending a chat turn
    stop = model.generation_config.eos_token_id
    stop = ([] if stop is None else [stop] if isinstance(stop, int)
            else list(stop))
    if tokenizer.eos_token_id not in [*stop, None]:
        stop.append(tokenizer.eos_token_id)
        model.generation_config.eos_token_id = stop
    unexpected = collections.Counter(  # keys in the file the model ignores
        '.'.join(key.split('.')[:2]) for key in info['unexpected_keys'])
    print(f'{model_id}@{revision[:12]}: {type(model).__name__}, 0 missing '
          f'keys, unexpected keys: {dict(unexpected) or "none"}, stop tokens '
          f'{stop}')
    devices = set(getattr(model, 'hf_device_map', {}).values())
    if len(devices) > 1:
        warnings.warn(f'{model_id} is spread over {sorted(map(str, devices))}')
    return model.eval(), tokenizer
```

<!-- Placeholder (stage 2): parameters by group from the configuration. -->

```{.python .input #hf-models-config-params}
%%tab pytorch
#@save
def _hf_text_config(config):
    """A configuration and its text part, as plain dicts."""
    c = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
    return c, c.get('text_config') or c

#@save
def _hf_layers(t):
    """Kind, head dimension, KV heads and KV sharing of every layer."""
    L, H = t['num_hidden_layers'], t['num_attention_heads']
    first_shared = L - (t.get('num_kv_shared_layers') or 0)
    overrides = {int(i): o
                 for i, o in (t.get('per_layer_config') or {}).items()}
    layers = []
    for i, kind in enumerate(t.get('layer_types') or ['full_attention'] * L):
        d_h = t.get('head_dim') or t['hidden_size'] // H
        H_kv = t.get('num_key_value_heads') or H
        if kind == 'full_attention':  # Gemma 4's global layers are wider
            d_h = t.get('global_head_dim') or d_h
            if t.get('attention_k_eq_v'):
                H_kv = t.get('num_global_key_value_heads') or H_kv
        o = overrides.get(i, {})
        layers.append((kind, o.get('head_dim', d_h),
                       o.get('num_key_value_heads', H_kv), i >= first_shared))
    return layers

#@save
def config_params(config, include_vision=False, include_mtp=False,
                  by_group=False):
    """Parameters of a dense or hybrid model (Qwen3, Qwen3.5, Gemma 4),
    counted from its configuration; other layouts raise."""
    c, t = _hf_text_config(config)
    if (t.get('num_experts') or t.get('n_routed_experts')
            or t.get('enable_moe_block')):
        raise NotImplementedError('use d2l.config_stats (14.1)')
    known = ('qwen3', 'qwen3_5_text', 'gemma4_text')  # QK norms, no biases
    if (t['model_type'] not in known or t.get('attention_bias')
            or t.get('mlp_bias')):
        raise NotImplementedError(f'{t["model_type"]} layout, not one of '
                                  f'{known} without biases')
    d, f, H, V = (t['hidden_size'], t['intermediate_size'],
                  t['num_attention_heads'], t['vocab_size'])
    gemma = t['model_type'] == 'gemma4_text'
    gate = 2 if (t.get('attn_output_gate')  # gated attention output
                 or t['model_type'] == 'qwen3_5_text') else 1
    g = dict.fromkeys(['embed', 'per_layer_embed', 'attention',
                       'linear_attention', 'mlp', 'per_layer_proj', 'norm',
                       'lm_head', 'vision', 'mtp'], 0)
    g['embed'] = V * d
    g['lm_head'] = 0 if t.get('tie_word_embeddings', True) else V * d

    def attention(kind, d_h, H_kv, shared):  # Q (+ gate), K, V, O, QK norms
        n_kv = (0 if shared else
                1 if t.get('attention_k_eq_v') and kind == 'full_attention'
                else 2)
        norms = d_h * (1 if shared else 2)
        return gate * d * H * d_h + n_kv * d * H_kv * d_h + H * d_h * d + norms

    for kind, d_h, H_kv, shared in _hf_layers(t):
        if kind == 'linear_attention':  # Gated DeltaNet
            H_v, d_v = t['linear_num_value_heads'], t['linear_value_head_dim']
            K = t['linear_num_key_heads'] * t['linear_key_head_dim']
            U, C = H_v * d_v, 2 * K + H_v * d_v  # C: convolution channels
            g['linear_attention'] += (d * C + 2 * d * U + 2 * d * H_v + 2 * H_v
                                      + C * t['linear_conv_kernel_dim'] + d_v)
        else:
            g['attention'] += attention(kind, d_h, H_kv, shared)
        wide = 2 if shared and t.get('use_double_wide_mlp') else 1
        g['mlp'] += 3 * d * f * wide
        g['norm'] += (4 if gemma else 2) * d
    g['norm'] += d  # final norm
    p, L = t.get('hidden_size_per_layer_input') or 0, t['num_hidden_layers']
    if p:  # Gemma 4: per-layer embeddings, their projections and norms
        g['per_layer_embed'] = t['vocab_size_per_layer_input'] * L * p
        g['per_layer_proj'] = d * L * p + L * 2 * d * p
        g['norm'] += p + L * d
    if include_mtp and t.get('mtp_num_hidden_layers'):  # Qwen3.5
        block = attention('full_attention', t.get('head_dim') or d // H,
                          t['num_key_value_heads'], False) + 3 * d * f + 2 * d
        g['mtp'] = t['mtp_num_hidden_layers'] * block + 2 * d * d + 3 * d
    v = c.get('vision_config')
    if include_vision and v:
        if (v['model_type'] not in ('qwen3_5', 'qwen3_5_vision')
                or v.get('deepstack_visual_indexes')):
            raise NotImplementedError(f'vision tower of {v["model_type"]}')
        D, I, m = v['hidden_size'], v['intermediate_size'], (
            v['spatial_merge_size'] ** 2)
        patch = (v['in_channels'] * v['temporal_patch_size']
                 * v['patch_size'] ** 2)
        block = 4 * D * D + 2 * D * I + 9 * D + I  # with biases, LayerNorms
        g['vision'] = ((patch + 1 + v['num_position_embeddings']) * D
                       + v['depth'] * block + 2 * D
                       + (D * m + 1) * (D * m + v['out_hidden_size']))
    return g if by_group else sum(g.values())
```

<!-- Placeholder (stage 2): bytes of weights, caches, state and logits. -->

```{.python .input #hf-models-config-memory}
%%tab pytorch
#@save
def config_memory(config, seq_len, batch_size, dtype_bytes=2, *,
                  state_bytes=4, conv_bytes=None, conv_taps=None,
                  window_cached=None, include_vision=False, include_mtp=False,
                  logits_to_keep=1):
    """Bytes to run a model on `batch_size` sequences of `seq_len` tokens.

    The defaults store what Transformers 5.22.2 stores: a sliding layer keeps
    `sliding_window - 1` positions, a linear-attention layer all
    `linear_conv_kernel_dim` convolution inputs. `activations` estimates one
    layer's tensors with fused attention; `logits_to_keep` of None or 0 keeps
    the logits of every position."""
    import warnings
    c, t = _hf_text_config(config)
    n, B, b = seq_len, batch_size, dtype_bytes
    if n > t.get('max_position_embeddings', n):
        warnings.warn(f'seq_len {n} exceeds max_position_embeddings '
                      f'{t["max_position_embeddings"]}')
    if window_cached is None and t.get('sliding_window'):
        window_cached = t['sliding_window'] - 1
    conv_bytes = b if conv_bytes is None else conv_bytes
    if conv_taps is None:
        conv_taps = t.get('linear_conv_kernel_dim')
    kv = state = width = 0  # per sequence; width: Q, K, V entries per token
    for kind, d_h, H_kv, shared in _hf_layers(t):
        if kind == 'linear_attention':  # fixed matrix state + conv inputs
            H_v, d_v = t['linear_num_value_heads'], t['linear_value_head_dim']
            C = 2 * t['linear_num_key_heads'] * t['linear_key_head_dim'] + (
                H_v * d_v)
            state += (H_v * t['linear_key_head_dim'] * d_v * state_bytes
                      + C * conv_taps * conv_bytes)
            continue
        width = max(width, (t['num_attention_heads'] + 2 * H_kv) * d_h)
        if not shared:  # a KV-sharing layer reads an earlier layer's cache
            kept = min(n, window_cached) if kind == 'sliding_attention' else n
            kv += 2 * H_kv * d_h * b * kept
    f = t['intermediate_size'] * (2 if t.get('use_double_wide_mlp')
                                  and t.get('num_kv_shared_layers') else 1)
    params = config_params(config, include_vision, include_mtp)
    n_out = min(logits_to_keep, n) if logits_to_keep else n
    memory = {'params': params, 'weights': params * b,
              'kv_cache': B * kv, 'recurrent_state': B * state,
              'activations': B * n * b * (2 * t['hidden_size'] + 3 * f
                                          + width),
              'logits': B * n_out * t['vocab_size'] * b}
    memory['total'] = sum(memory.values()) - params
    return memory
```
