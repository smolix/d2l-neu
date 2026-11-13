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
import accelerate
import collections
import copy
import functools
import gc
import hashlib
import huggingface_hub
import importlib.util
import itertools
import json
import os
import pandas as pd
import re
import transformers
from huggingface_hub import HfApi, file_exists
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig)
```

<!-- Placeholder (prose, opening): the studied models, the question, and the library versions that produced the outputs. -->

```{.python .input #hf-models-settings}
%%tab pytorch
QWEN, GEMMA = 'Qwen/Qwen3.5-2B', 'google/gemma-4-E2B-it'
QWEN3 = 'Qwen/Qwen3-0.6B'  # its tokenizer only
GPU = torch.cuda.is_available()
# On a GPU the studied model runs in bf16; without one, its smallest sibling
# runs in fp32 and the measurements of the last part are skipped
RUN_MODEL, RUN_REV, RUN_DTYPE = (
    (QWEN, '15852e8c16360a2fea060d615a32b45270f8a8fc', torch.bfloat16) if GPU
    else ('Qwen/Qwen3.5-0.8B', '2fc06364715b967f1860aea9cf38778875588b17',
          torch.float32))
QUESTION = ('Why does a language model keep a key-value cache when it '
            'generates text?')
transformers.logging.set_verbosity_error()  # no notices or progress bars
transformers.logging.disable_progress_bar()
huggingface_hub.utils.disable_progress_bars()
huggingface_hub.utils.logging.set_verbosity_error()
for module in (torch, transformers, huggingface_hub, accelerate):
    print(f'{module.__name__} {module.__version__}')
```

<!-- Placeholder (prose, opening): GPT-2 with its released weights continues the question with more text. -->

```{.python .input #hf-models-gpt2-question}
%%tab pytorch
gpt2, enc = d2l.load_gpt2().to(d2l.try_gpu()), d2l.gpt2_tokenizer()
out = gpt2.generate(enc.encode(f'Q: {QUESTION}\nA:'), 32, top_k=1)
d2l.print_wrapped(enc.decode(out))
del gpt2
```

<!-- Placeholder (prose, opening): three lines load a current open model at one commit, and it answers. -->

```{.python .input #hf-models-three-lines}
%%tab pytorch
tokenizer = AutoTokenizer.from_pretrained(RUN_MODEL, revision=RUN_REV)
model = AutoModelForCausalLM.from_pretrained(
    RUN_MODEL, revision=RUN_REV, dtype=RUN_DTYPE, device_map=d2l.try_gpu())
prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': QUESTION}], add_generation_prompt=True,
    enable_thinking=False, return_dict=True,
    return_tensors='pt').to(model.device)
reply = model.generate(**prompt, max_new_tokens=64, do_sample=False)
n_prompt = prompt['input_ids'].shape[1]
d2l.print_wrapped(tokenizer.decode(reply[0, n_prompt:],
                                   skip_special_tokens=True))
print(f'{type(model).__name__}: {model.num_parameters():,} parameters')
```

<!-- Placeholder (prose, ### Repository Anatomy): the files of the repository at its pinned commit. -->

```{.python .input #hf-models-repo-files}
%%tab pytorch
info = HfApi().model_info(QWEN, revision=d2l.HF_REVISIONS[QWEN],
                          files_metadata=True)
for file in info.siblings:
    print(f'{file.size / 1e6:10,.2f} MB  {file.rfilename}')
print(f'safetensors: {info.safetensors.total:,} elements')
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

<!-- Placeholder (prose, ### Repository Anatomy): elements by top-level prefix and by dtype, read from the header alone. -->

```{.python .input #hf-models-header-read}
%%tab pytorch
def top_prefix(name):  # model.language_model, model.visual or mtp
    return '.'.join(name.split('.')[:2 if name.startswith('model.') else 1])

header = safetensors_header(QWEN)
elements = collections.Counter()
for name, (shape, dtype) in header.items():
    elements[top_prefix(name)] += math.prod(shape)
dtypes = collections.Counter(dtype for _, dtype in header.values())
print(f'{len(header)} tensors {dict(dtypes)}, '
      f'{sum(elements.values()):,} elements')
for prefix, n in elements.items():
    print(f'{n:15,}  {prefix}')
assert sum(elements.values()) == info.safetensors.total
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

<!-- Placeholder (prose, ### Auto Classes and the Text Model): the pinned commit, the device, and the parameters left on disk. -->

```{.python .input #hf-models-pinned-check}
%%tab pytorch
assert RUN_REV == HF_REVISIONS[RUN_MODEL]
# A pinned load reads its files from a cache folder named after the commit
folder = os.path.dirname(huggingface_hub.hf_hub_download(
    RUN_MODEL, 'config.json', revision=RUN_REV))
in_file = sum(math.prod(shape)
              for shape, _ in safetensors_header(RUN_MODEL).values())
loaded = model.num_parameters()
print(f'{folder}\non {sorted({str(p.device) for p in model.parameters()})}')
print(f'{loaded:,} parameters loaded, {in_file:,} elements in the file, '
      f'{in_file - loaded:,} left on disk')
```

<!-- Placeholder (prose, ### Tokenizers and Chat Templates): vocabulary sizes and tokens per character on three texts. -->

```{.python .input #hf-models-tokenizers}
%%tab pytorch
CHINESE = ('语言模型每次只生成一个词元。为了预测下一个词元，它要用到前面所有词元的'
           '信息。键值缓存把每一层已经算好的键和值保存下来，生成新词元时只需计算这'
           '一个位置，而不必从头处理整段文本。上下文越长，缓存占用的显存就越多，所以'
           '长文本推理常常受限于内存，而不是算力。')
fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', '../data',
                     '090b5e7e70c295757f55df93cb0a180b9691891a')
book = open(fname).read()
texts = {'The Time Machine': book[:20000],
         'd2l/torch.py': open(d2l.__file__).read()[:20000],
         'Chinese': CHINESE}

def per_char(encode):  # tokens per character of each text
    return {name: len(encode(text)) / len(text)
            for name, text in texts.items()}

table = {'GPT-2': {'vocabulary': enc.n_vocab, 'embedding rows': enc.n_vocab,
                   **per_char(enc.encode_ordinary)}}
tokenizers = {}
for m in (QWEN3, QWEN, GEMMA):
    tokenizers[m] = AutoTokenizer.from_pretrained(m, revision=HF_REVISIONS[m])
    rows = AutoConfig.from_pretrained(
        m, revision=HF_REVISIONS[m]).get_text_config().vocab_size
    table[m] = {'vocabulary': len(tokenizers[m]), 'embedding rows': rows,
                **per_char(functools.partial(tokenizers[m].encode,
                                             add_special_tokens=False))}
table = pd.DataFrame(table).T.astype({'vocabulary': int,
                                     'embedding rows': int})
print(table.round(2).to_string())
```

<!-- Placeholder (prose, ### Tokenizers and Chat Templates): the prompt the model of the opening read, and the token ids of its tail. -->

```{.python .input #hf-models-chat-template}
%%tab pytorch
text = tokenizers[QWEN].apply_chat_template(
    [{'role': 'user', 'content': QUESTION}], add_generation_prompt=True,
    enable_thinking=False, tokenize=False)
print(text)
tail = tokenizers[QWEN].encode(text, add_special_tokens=False)[-9:]
print(tail, tokenizers[QWEN].convert_ids_to_tokens(tail))
```

<!-- Placeholder (prose, ### Generation Arguments and Defaults): greedy generate against a four-line argmax loop. -->

```{.python .input #hf-models-greedy-check}
%%tab pytorch
def argmax_loop(ids, steps):  # greedy decoding by hand, without a cache
    for _ in range(steps):
        logits = model(ids, logits_to_keep=1, use_cache=False).logits
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return ids

with torch.inference_mode():
    ref = model.generate(**prompt, max_new_tokens=32, do_sample=False,
                         use_cache=False)
    ours = argmax_loop(prompt['input_ids'], ref.shape[1] - n_prompt)
    print('identical:', torch.equal(ours, ref))
    assert torch.equal(ours, ref)
    for name, f in (
            ('argmax loop without a cache', lambda: argmax_loop(
                prompt['input_ids'], 32)),
            ('generate with a cache', lambda: model.generate(
                **prompt, min_new_tokens=32, max_new_tokens=32,
                do_sample=False))):
        print(f'{name}: {d2l.Benchmark(f, 0, 1).time:.2f} s for 32 tokens')
```

<!-- Placeholder (prose, ### Generation Arguments and Defaults): the sampling defaults each repository ships. -->

```{.python .input #hf-models-generation-defaults}
%%tab pytorch
for m in (QWEN, GEMMA):
    rev = HF_REVISIONS[m]
    if file_exists(m, 'generation_config.json', revision=rev):
        source = 'generation_config.json'
        gen = GenerationConfig.from_pretrained(m, revision=rev)
    else:  # Transformers derives one from the model configuration
        source = 'config.json, library defaults'
        gen = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(m, revision=rev).get_text_config())
    print(f'{m} ({source}): do_sample={gen.do_sample}, temperature='
          f'{gen.temperature}, top_k={gen.top_k}, top_p={gen.top_p}, '
          f'stop tokens {gen.eos_token_id}')
```

<!-- Placeholder (prose, ### The Two Configurations): the fields of both configurations, the layer pattern, and the layers that keep memory of their own. -->

```{.python .input #hf-models-config-fields}
%%tab pytorch
configs = {m: AutoConfig.from_pretrained(m, revision=HF_REVISIONS[m])
           for m in (QWEN, GEMMA)}
text_configs = {m: c.get_text_config().to_dict() for m, c in configs.items()}
FIELDS = (
    'hidden_size', 'num_hidden_layers', 'sliding_window',
    'num_attention_heads', 'num_key_value_heads', 'head_dim',
    'attn_output_gate',
    'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim',
    'linear_value_head_dim', 'linear_conv_kernel_dim', 'mamba_ssm_dtype',
    'num_kv_shared_layers', 'use_double_wide_mlp',
    'hidden_size_per_layer_input', 'intermediate_size', 'hidden_act',
    'hidden_activation', 'vocab_size', 'tie_word_embeddings',
    'final_logit_softcapping', 'mtp_num_hidden_layers',
    'max_position_embeddings')
print(pd.DataFrame({m: {f: t.get(f, '–') for f in FIELDS}
                    for m, t in text_configs.items()}).to_string())

def layer_pattern(types):  # the shortest run-length pattern that repeats
    runs = [(kind, len(list(group)))
            for kind, group in itertools.groupby(types)]
    k = next(k for k in range(1, len(runs) + 1)
             if runs == runs[:k] * (len(runs) // k))
    return runs[:k], len(runs) // k

for m, t in text_configs.items():
    pattern, repeats = layer_pattern(t['layer_types'])
    L = t['num_hidden_layers']
    own = L - (t.get('num_kv_shared_layers') or 0)
    print(f'{m}: {pattern} x {repeats}; {own} of {L} layers keep a cache '
          f'or state of their own; rope_parameters {t["rope_parameters"]}')
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

<!-- Placeholder (prose, ### Parameter Counts): the configuration reproduces each bucket of the file to the element. -->

```{.python .input #hf-models-reconcile}
%%tab pytorch
groups = config_params(configs[QWEN], include_vision=True, include_mtp=True,
                       by_group=True)
derived = {'model.language_model': sum(n for g, n in groups.items()
                                       if g not in ('vision', 'mtp')),
           'model.visual': groups['vision'], 'mtp': groups['mtp']}
buckets = pd.DataFrame({'from config': derived, 'from header': elements})
buckets.loc['total'] = buckets.sum()
print(buckets.map('{:,}'.format).to_string())
assert derived == dict(elements)
```

<!-- Placeholder (prose, ### The Gemma 4 Configuration): head widths, the per-layer embedding table, and what the file stores beyond the model. -->

```{.python .input #hf-models-gemma-config}
%%tab pytorch
t = text_configs[GEMMA]
overrides = {int(i): o for i, o in t['per_layer_config'].items()}
head_dims = collections.Counter(  # global layers override the head width
    (kind, overrides.get(i, {}).get('head_dim', t['head_dim']))
    for i, kind in enumerate(t['layer_types']))
print(f'layers by kind and head width {dict(head_dims)}; per-layer '
      f'embeddings {t["vocab_size_per_layer_input"]:,} x '
      f'{t["num_hidden_layers"]} x {t["hidden_size_per_layer_input"]}')
gemma_groups = config_params(configs[GEMMA], by_group=True)
print({g: f'{n:,}' for g, n in gemma_groups.items() if n})
gemma_header, first_shared = safetensors_header(GEMMA), (
    t['num_hidden_layers'] - t['num_kv_shared_layers'])

def unused(name):  # K/V of the KV-sharing layers: stored, never built
    m = re.match(r'model\.language_model\.layers\.(\d+)\.self_attn\.'
                 r'(k_proj|v_proj|k_norm)\.', name)
    return m is not None and int(m.group(1)) >= first_shared

def numel(keep):  # elements of the header's tensors whose name passes keep
    return sum(math.prod(shape) for name, (shape, _) in gemma_header.items()
               if keep(name))

text_total = numel(lambda name: name.startswith('model.language_model.'))
residual = text_total - sum(gemma_groups.values())
scalars = numel(lambda name: name.endswith('.layer_scalar'))  # buffers
print(f'text tensors in the file {text_total:,}, from the configuration '
      f'{sum(gemma_groups.values()):,}: {residual:,} = {numel(unused):,} '
      f'unused K/V + {scalars} layer scalars')
assert residual == numel(unused) + scalars
print(f'parameters read per token: '
      f'{sum(gemma_groups.values()) - gemma_groups["per_layer_embed"]:,}')
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

<!-- Placeholder (prose, ### Weights, Caches and State): the terms of eq_lm_hf_memory for both models, the all-attention counterfactual, and what fits a 15 GiB slot. -->

```{.python .input #hf-models-memory-table}
%%tab pytorch
GiB, BUDGET = 2**30, 15 * 2**30  # a 15 GiB slot of a 24 GB card
TERMS = ('weights', 'kv_cache', 'recurrent_state', 'activations', 'logits',
         'total')

def all_attention(config):
    """The text configuration with every layer keeping its own keys and
    values; with window_cached=n, sliding layers keep every position too."""
    t = config.get_text_config().to_dict()
    t['layer_types'] = ['full_attention' if kind == 'linear_attention'
                        else kind for kind in t['layer_types']]
    t['num_kv_shared_layers'] = 0
    return t

rows = []
for m, c in configs.items():
    for n in (1024, 4096, 16384, 65536,
              c.get_text_config().max_position_embeddings):
        mem = config_memory(c, n, 1)
        kv_all = config_memory(all_attention(c), n, 1, window_cached=n)
        rows.append({'model': m, 'n': n, **{k: mem[k] / GiB for k in TERMS},
                     'all-attention kv': kv_all['kv_cache'] / GiB,
                     'fits 15 GiB': mem['total'] <= BUDGET})
print(pd.DataFrame(rows).round(3).to_string(index=False))
for m, c in configs.items():
    a, b = config_memory(c, 4096, 1), config_memory(c, 4097, 1)
    slope = b['kv_cache'] - a['kv_cache']  # bytes per token beyond windows
    fixed = a['kv_cache'] + a['recurrent_state'] - 4096 * slope
    slope_all = config_memory(all_attention(c), 1, 1,
                              window_cached=1)['kv_cache']
    batch = (BUDGET - a['weights']) // (a['total'] - a['weights'])
    context = max(n for n in (2**k for k in range(8, 19))
                  if n <= c.get_text_config().max_position_embeddings
                  and config_memory(c, n, 1)['total'] <= BUDGET)
    print(f'{m}: {slope:,} B/token + {fixed:,} B against {slope_all:,} '
          f'B/token: smaller beyond {fixed / (slope_all - slope):.0f} '
          f'tokens, {(slope * 16384 + fixed) / (slope_all * 16384):.2f} of it '
          f'at 16,384; within 15 GiB: batch {batch} at 4,096 tokens, '
          f'{context:,} tokens at batch 1')
logits = config_memory(configs[QWEN], 16384, 1, logits_to_keep=None)
print(f'logits of all 16,384 positions: {logits["logits"] / 1e9:.1f} GB')
```

<!-- Placeholder (prose, ### Decode Throughput): the bandwidth the card streams, next to its datasheet value. -->

```{.python .input #hf-models-bandwidth}
%%tab pytorch
BETA_DATASHEET = 1.0e12  # bytes/s, RTX 4090
if GPU:
    x = torch.randn(256 * 1024 * 1024, device=d2l.try_gpu())  # 1 GB fp32
    beta = 2 * x.nbytes / d2l.Benchmark(lambda: x * 2.0).time
    del x
    print(f'measured {beta / 1e12:.2f} TB/s '
          f'(datasheet {BETA_DATASHEET / 1e12:.1f} TB/s)')
```

<!-- Placeholder (prose, ### Measured Memory): the measuring functions (cache walker, prefill, decode timing, one step's kernels, bytes read per step). -->

```{.python .input #hf-models-measure-fns}
%%tab pytorch
MiB = 2**20
MATMUL = re.compile('gemm|gemv|xmma|cutlass|matmul|splitK', re.I)

def cache_bytes(cache):
    """Bytes of the floating-point tensors reachable from a cache object,
    by dtype: logical (every distinct view) and resident (every underlying
    storage once). Integer tensors are bookkeeping, not cache."""
    logical, resident = collections.Counter(), collections.Counter()
    views, storages, visited, todo = set(), set(), set(), [cache]
    while todo:
        obj = todo.pop()
        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
            if (obj.data_ptr(), obj.shape) not in views:
                views.add((obj.data_ptr(), obj.shape))
                logical[obj.dtype] += obj.nbytes
            storage = obj.untyped_storage()
            if storage.data_ptr() not in storages:
                storages.add(storage.data_ptr())
                resident[obj.dtype] += storage.nbytes()
        elif id(obj) not in visited:
            visited.add(id(obj))
            if isinstance(obj, dict):
                todo.extend(obj.values())
            elif isinstance(obj, (list, tuple)):
                todo.extend(obj)
            elif type(obj).__module__.startswith('transformers.'):
                todo.extend(vars(obj).values())  # cache and layer objects
    return logical, resident

def prefill(model, ids, n, keep=1):
    """Prefill the first n tokens: the output, the allocated bytes before,
    and the growth and the peak above them."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        out = model(ids[:, :n], logits_to_keep=keep, use_cache=True)
    torch.cuda.synchronize()
    return (out, before, torch.cuda.memory_allocated() - before,
            torch.cuda.max_memory_allocated() - before)

def decode_ms(model, cache, token, steps=64, warmup=3, repeats=10):
    """Median, first and third quartile of milliseconds per greedy decode
    step; every repeat starts from a fresh copy of the prefilled cache."""
    def loop(c, token):
        for _ in range(steps):
            logits = model(token, past_key_values=c, logits_to_keep=1).logits
            token = logits[:, -1].argmax(-1, keepdim=True)  # on the device

    times = []
    with torch.inference_mode():
        for _ in range(warmup + repeats):
            c = copy.deepcopy(cache)  # outside the timer
            seconds = d2l.Benchmark(lambda: loop(c, token), 0, 1).time
            times.append(1e3 * seconds / steps)
    return np.percentile(times[warmup:], [50, 25, 75])

def step_kernels(model, cache, token, steps=10):
    """Device milliseconds per kernel of one decode step, and the number of
    kernels the step launches."""
    c = copy.deepcopy(cache)

    def step():
        model(token, past_key_values=c, logits_to_keep=1)

    with torch.inference_mode():
        step()  # warm-up
        kernels = d2l.kernel_times(step, steps)
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
    return kernels, sum(e.device_type == torch.autograd.DeviceType.CUDA
                        for e in prof.events())

def bytes_read(config, n, b=2):
    """Bytes a decode step reads at context n: the weights except the
    gathered per-layer table, every cache a layer attends to (a KV-sharing
    layer re-reads its source's), and the state, read and written."""
    t = config.get_text_config().to_dict()
    t['num_kv_shared_layers'] = 0
    mem = config_memory(t, n, 1, b)
    groups = config_params(config, by_group=True)
    return (b * (sum(groups.values()) - groups['per_layer_embed'])
            + mem['kv_cache'] + 2 * mem['recurrent_state'])
```

<!-- Placeholder (prose, ### Measured Memory): Qwen3.5-2B at four contexts; the walker, the allocator and the kept logits against the formula. -->

```{.python .input #hf-models-run-qwen}
%%tab pytorch
def close(measured, derived):  # allocator rounding: 5% or 2 MiB
    return abs(measured - derived) <= max(0.05 * derived, 2 * MiB)

def measure(model, config, ids, beta, contexts=(256, 1024, 4096, 16384)):
    """Memory and decode time at each context: the rows, and the kernels of
    one decode step at 1,024 tokens."""
    b, t = model.dtype.itemsize, config.get_text_config()
    rows, kernels = [], None
    for n in contexts:
        pinned = config_memory(config, n, 1, b, window_cached=n)
        need = pinned['total'] - pinned['weights'] + (  # unfused attention
            t.num_attention_heads * n * n * b)
        torch.cuda.empty_cache()  # cached blocks count as free below
        if need > torch.cuda.mem_get_info()[0]:
            print(f'{n} tokens skipped: {need / GiB:.1f} GiB not free')
            continue
        out, before, growth, peak = prefill(model, ids, n)
        cache, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
        logical, resident = cache_bytes(cache)
        mem = config_memory(config, n, 1, b)
        assert sum(logical.values()) == mem['kv_cache'] + mem[
            'recurrent_state']  # C3
        if n == 1024:
            print('live cache, bytes by dtype:',
                  {str(k): v for k, v in logical.items()})
        wall = decode_ms(model, cache, token)
        step_ms, launches = step_kernels(model, cache, token)
        kernels = step_ms if n == 1024 else kernels
        logits_bytes = out.logits.nbytes
        del out
        with torch.inference_mode():  # sliding layers drop prompt keys here
            step = model(token, past_key_values=cache, logits_to_keep=1)
        after = config_memory(config, n + 1, 1, b,
                              window_cached=getattr(t, 'sliding_window', None))
        R = bytes_read(config, n, b)
        rows.append(dict(
            model=model.name_or_path, n=n,
            walker_logical=sum(logical.values()),
            walker_resident=sum(resident.values()),
            formula=mem['kv_cache'] + mem['recurrent_state'], growth=growth,
            growth_derived=(pinned['kv_cache'] + pinned['recurrent_state']
                            + logits_bytes),
            growth_after_step=torch.cuda.memory_allocated() - before,
            growth_after_step_derived=(after['kv_cache'] + after[
                'recurrent_state'] + step.logits.nbytes),
            peak=peak, act_estimate=mem['activations'],
            wall_ms_median=wall[0], wall_ms_q1=wall[1], wall_ms_q3=wall[2],
            busy_ms=sum(step_ms.values()),
            matmul_ms=sum(ms for name, ms in step_ms.items()
                          if MATMUL.search(name)),
            launches=launches,
            r_over_beta_datasheet_ms=1e3 * R / BETA_DATASHEET,
            r_over_beta_measured_ms=1e3 * R / beta))
        del cache, step
    return rows, kernels

def show(rows):
    df = pd.DataFrame(rows)
    print(pd.DataFrame({
        'n': df.n, 'walker MiB': df.walker_logical / MiB,
        'formula MiB': df.formula / MiB,
        'growth/derived': df.growth / df.growth_derived,
        'after step/derived': (df.growth_after_step
                               / df.growth_after_step_derived),
        'peak/Act': df.peak / df.act_estimate,
        'wall ms': df.wall_ms_median, 'q1': df.wall_ms_q1,
        'q3': df.wall_ms_q3, 'busy ms': df.busy_ms,
        'matmul ms': df.matmul_ms, 'R/beta sheet': df.r_over_beta_datasheet_ms,
        'R/beta meas.': df.r_over_beta_measured_ms,
        'launches': df.launches}).round(2).to_string(index=False))

if GPU:
    print('Gated DeltaNet kernels:', 'flash-linear-attention'
          if importlib.util.find_spec('fla') else 'PyTorch reference code')
    ids = tokenizer(book, return_tensors='pt').input_ids.to(model.device)
    qwen_rows, qwen_kernels = measure(model, configs[QWEN], ids, beta)
    show(qwen_rows)
    for row in qwen_rows:  # C4, right after the prefill
        assert close(row['growth'], row['growth_derived'])
    out, _, growth, _ = prefill(model, ids, 4096, keep=0)  # every logit
    mem = config_memory(configs[QWEN], 4096, 1, logits_to_keep=0)
    extra = growth - mem['kv_cache'] - mem['recurrent_state']
    del out
    print(f'logits of all 4,096 positions: growth beyond the cache '
          f'{extra:,} B, derived {mem["logits"]:,} B')
    assert abs(extra - mem['logits']) <= 0.05 * mem['logits']  # C6
```

<!-- Placeholder (prose, ### Auto Classes and the Text Model): Gemma 4 through load_hf, its load report, and its repository's sampling defaults. -->

```{.python .input #hf-models-load-gemma}
%%tab pytorch
def chat(tok, content):  # one user turn, rendered for generation
    return tok.apply_chat_template(
        [{'role': 'user', 'content': content}], add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors='pt')

if GPU:
    del model, prompt, reply, ref, ours
    gc.collect()
    torch.cuda.empty_cache()
    gemma, gemma_tok = load_hf(GEMMA)
    assert gemma.num_parameters() == config_params(configs[GEMMA])  # C2
    print(f'{gemma.num_parameters():,} parameters, as config_params says')
    sea = chat(gemma_tok, 'Write one sentence about the sea.').to(
        gemma.device)
    torch.manual_seed(0)
    n_sea = sea['input_ids'].shape[1]
    replies = [gemma_tok.decode(  # the repository's sampling defaults
        gemma.generate(**sea, max_new_tokens=16)[0, n_sea:],
        skip_special_tokens=True) for _ in range(2)]
    print(replies, 'differ:', replies[0] != replies[1])
    question = chat(gemma_tok, QUESTION).to(gemma.device)
    greedy = gemma.generate(**question, max_new_tokens=64, do_sample=False,
                            top_k=None, top_p=None)
    d2l.print_wrapped(gemma_tok.decode(
        greedy[0, question['input_ids'].shape[1]:], skip_special_tokens=True))
```

<!-- Placeholder (prose, ### Measured Memory): the same measurements for Gemma 4, compared after its first decode step. -->

```{.python .input #hf-models-run-gemma}
%%tab pytorch
if GPU:
    gemma_ids = gemma_tok(book, return_tensors='pt').input_ids.to(
        gemma.device)
    gemma_rows, gemma_kernels = measure(gemma, configs[GEMMA], gemma_ids,
                                        beta)
    show(gemma_rows)
    print('resident/logical:', [round(r['walker_resident']
                                      / r['walker_logical'], 2)
                                for r in gemma_rows])
    for row in gemma_rows:  # C4, after the first decode step
        assert close(row['growth_after_step'],
                     row['growth_after_step_derived'])
```

<!-- Placeholder (prose, ### Decode Throughput): where a decode step's time goes, and matrix-product time against bytes read. -->

```{.python .input #hf-models-profile-compare}
%%tab pytorch
if GPU:
    at = {r['model']: r for r in qwen_rows + gemma_rows if r['n'] == 1024}
    for m, kernels in ((QWEN, qwen_kernels), (GEMMA, gemma_kernels)):
        print(m)
        d2l.show_profile(kernels, at[m]['wall_ms_median'])
        host_us = 1e3 * (at[m]['wall_ms_median'] - at[m]['busy_ms'])
        print(f'host time per launch: {host_us / at[m]["launches"]:.1f} us, '
              f'device busy under half the step: '
              f'{at[m]["busy_ms"] < at[m]["wall_ms_median"] / 2}')
    held = config_params(configs[GEMMA]) / config_params(configs[QWEN])
    read = (bytes_read(configs[GEMMA], 1024)
            / bytes_read(configs[QWEN], 1024))
    ratio = at[GEMMA]['matmul_ms'] / at[QWEN]['matmul_ms']
    print(f'matrix products Gemma/Qwen: {ratio:.2f} (bytes read {read:.2f}, '
          f'bytes held {held:.2f}); closer to bytes read: '
          f'{ratio < math.sqrt(read * held)}')
    for m, c in configs.items():
        groups = config_params(c, by_group=True)
        weights_ms = 2e3 * (sum(groups.values()) - groups['per_layer_embed']
                            ) / beta
        print(f'{m}: matrix products {at[m]["matmul_ms"]:.2f} ms, weights '
              f'read at measured bandwidth {weights_ms:.2f} ms, within a '
              f'factor of 2: {at[m]["matmul_ms"] <= 2 * weights_ms}')
```

<!-- Placeholder (prose, none): the measured rows are written for upload as lm-hf-memory-speed-v1. -->

```{.python .input #hf-models-save-table}
%%tab pytorch
if GPU:  # the candidate file; a download never writes to this path
    measured = {'rows': qwen_rows + gemma_rows, 'beta_measured': beta,
                'transformers': transformers.__version__}
    path = os.path.join('..', 'data', 'lm', 'rerun',
                        'lm-hf-memory-speed.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(measured, f, indent=1)
    with open(path, 'rb') as f:
        print(path, hashlib.sha1(f.read()).hexdigest())
```

<!-- Placeholder (prose, ### Decode Throughput): the plot below; caption sentence from spec fig_lm_memory_vs_config. -->

```{.python .input #hf-models-plot}
%%tab pytorch
if not GPU:  # the rows of the run behind the published figure
    with open(d2l.download('lm-hf-memory-speed-v1',
                           os.path.join('..', 'data', 'lm'))) as f:
        measured = json.load(f)
names = {QWEN: 'Qwen3.5', GEMMA: 'Gemma 4'}
mem_series, time_series = [], []  # (x, y, format, label) per curve
for i, (m, c) in enumerate(configs.items()):
    rows = [r for r in measured['rows'] if r['model'] == m]
    n_run = [r['n'] for r in rows]
    ns = [2**k for k in range(
        8, c.get_text_config().max_position_embeddings.bit_length())]
    formula = [config_memory(c, n, 1) for n in ns]
    mem_series += [
        (ns, [(f['kv_cache'] + f['recurrent_state']) / MiB for f in formula],
         f'C{i}-', f'{names[m]} formula'),
        (ns, [config_memory(all_attention(c), n, 1, window_cached=n)[
            'kv_cache'] / MiB for n in ns], f'C{i}--',
         f'{names[m]} all attention'),
        (n_run, [r['walker_logical'] / MiB for r in rows], f'C{i}o',
         f'{names[m]} measured')]
    time_series += [
        (n_run, [r[k] for r in rows], f'C{i}{fmt}', f'{names[m]} {label}')
        for k, fmt, label in (('wall_ms_median', '-', 'wall'),
                              ('busy_ms', '--', 'device busy'),
                              ('r_over_beta_measured_ms', ':', 'bytes/β'))]
_, axes = d2l.plt.subplots(1, 2, figsize=(7.5, 3.2))
with d2l.plt.rc_context({'legend.fontsize': 'x-small'}):
    for ax, series, ylabel, yscale, loc in (
            (axes[0], mem_series, 'cache + state (MiB)', 'log', 'upper left'),
            (axes[1], time_series, 'ms per decode step', 'linear', None)):
        X, Y, fmts, legend = zip(*series)
        d2l.plot(list(X), list(Y), 'context (tokens)', ylabel, list(legend),
                 xscale='log', yscale=yscale, fmts=fmts, axes=ax,
                 legend_loc=loc)
for i, m in enumerate(configs):  # interquartile band of the wall time
    rows = [r for r in measured['rows'] if r['model'] == m]
    axes[1].fill_between([r['n'] for r in rows],
                         [r['wall_ms_q1'] for r in rows],
                         [r['wall_ms_q3'] for r in rows],
                         color=f'C{i}', alpha=0.2)
```
