# A Released Checkpoint: Hub Files and Sampler Plumbing (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_d2l_checkpoint` uses and does not show:

- `_qwen3_files`, the Hub download behind `load_qwen3` (configuration,
  generation configuration, one file or the shards of an index);
- the argument plumbing of `generate`: `_lm_stopping` (defaults of
  `max_new_tokens`, the end tokens and the stop strings), `_lm_prompt_ids`
  (chat template or BOS, then token ids), `_lm_generators` (one random
  stream per call or per prompt), `_lm_bos_ids` and `_lm_unwrap` (the model
  inside a `torch.compile` or PEFT wrapper; also used by 18.4 and 18.5);
- the registry entry of the fp64 logit reference of assignment A1
  (`lm-qwen3-0p6b-logit-ref-v1`), which no notebook of the book reads.

Every block is tagged `%%tab pytorch` and imports third-party modules
(`huggingface_hub`) inside its functions.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_d2l_checkpoint`

```{.python .input #d2l-checkpoint-lib-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-qwen3-0p6b-logit-ref-v1'] = (
    d2l.LM_DATA_URL + 'lm-qwen3-0p6b-logit-ref-v1.pt',
    '2fa5449638a8a8924bc6b35384dc996c23513cb6')
```

```{.python .input #d2l-checkpoint-lib-files}
%%tab pytorch
#@save
def _qwen3_files(model_id, revision):
    """Local paths of config.json, generation_config.json (None if the
    repository has none) and the safetensors files of a Hub checkpoint at a
    revision, downloaded into the Hugging Face cache."""
    import huggingface_hub
    # the Hub answered that the file is absent; LocalEntryNotFoundError
    # (offline, not cached) is a sibling subclass and must propagate
    from huggingface_hub.errors import RemoteEntryNotFoundError

    def get(filename, optional=False):
        try:
            return huggingface_hub.hf_hub_download(model_id, filename,
                                                   revision=revision)
        except RemoteEntryNotFoundError:
            if optional:
                return None
            raise

    index = get('model.safetensors.index.json', optional=True)
    if index is None:
        weights = [get('model.safetensors')]
    else:  # a sharded checkpoint lists its shards in the index
        with open(index) as fh:
            shards = sorted(set(json.load(fh)['weight_map'].values()))
        weights = [get(shard) for shard in shards]
    return (get('config.json'), get('generation_config.json', optional=True),
            weights)
```

```{.python .input #d2l-checkpoint-lib-generate}
%%tab pytorch
#@save
def _lm_unwrap(model):
    """The model inside a torch.compile or PEFT wrapper (which pass every
    keyword argument through to it), or model itself."""
    while True:
        inner = getattr(model, '_orig_mod', None)
        if inner is None and hasattr(model, 'get_base_model'):
            inner = model.get_base_model()
        if inner is None or inner is model:
            return model
        model = inner

#@save
def _lm_bos_ids(tokenizer):
    """The special tokens the tokenizer puts before a text (a BOS, or
    none). Those it appends after a text, such as an EOS, are left out:
    a prompt continues where its text ends."""
    with_sp, plain = (tokenizer(['a'], add_special_tokens=s)['input_ids'][0]
                      for s in (True, False))
    return next((with_sp[:k] for k in range(len(with_sp) - len(plain) + 1)
                 if with_sp[k:k + len(plain)] == plain), [])

#@save
def _lm_stopping(model, tokenizer, max_new_tokens=None, eos_token_ids=None,
                 stop=None):
    """generate's stopping arguments with their defaults: max_new_tokens
    from the model's generation_config (else 256); the end tokens, by
    default the tokenizer's and the generation_config's, as a set of ids
    (one id may be given bare); stop strings as a list (one string is one
    stop string, not its characters)."""
    config = getattr(model, 'generation_config', None)
    if max_new_tokens is None:
        max_new_tokens = getattr(config, 'max_new_tokens', None) or 256
    if eos_token_ids is None:
        eos = getattr(config, 'eos_token_id', None)
        eos_token_ids = [tokenizer.eos_token_id,
                         *(eos if isinstance(eos, list) else [eos])]
    if isinstance(eos_token_ids, (int, np.integer)):
        eos_token_ids = [eos_token_ids]
    eos_ids = {int(i) for i in eos_token_ids if i is not None}
    if isinstance(stop, str):
        stop = [stop]
    return max_new_tokens, eos_ids, stop

#@save
def _lm_prompt_ids(tokenizer, prompts, chat=False, enable_thinking=None):
    """Token ids of each prompt of generate. With chat=True a prompt is
    rendered by the chat template (a string is one user message; the
    template adds no BOS); otherwise the tokenizer's BOS, if it has one,
    goes before the text, and never an EOS after it."""
    if chat:
        opts = {} if enable_thinking is None else {
            'enable_thinking': enable_thinking}
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': x}] if isinstance(x, str) else x,
            tokenize=False, add_generation_prompt=True, **opts)
                   for x in prompts]
    head = [] if chat else _lm_bos_ids(tokenizer)
    ids = [head + tokenizer(x, add_special_tokens=False)['input_ids']
           for x in prompts]
    if not all(ids):  # Qwen3's tokenizer adds no BOS to predict from
        raise ValueError('an empty prompt has no position to predict from')
    return ids

#@save
def _lm_generators(seed, num_prompts):
    """generate's random streams, as a function rows -> the generator
    argument of sample_logits for those prompts. seed is an int or a
    torch.Generator (one stream for the whole call) or one int per prompt
    (one stream each, so that a prompt's draws do not depend on the
    others)."""
    if isinstance(seed, torch.Generator):
        return lambda rows: seed
    if isinstance(seed, (int, np.integer)):
        shared = torch.Generator().manual_seed(int(seed))
        return lambda rows: shared
    assert len(seed) == num_prompts, 'one seed per prompt'
    streams = [torch.Generator().manual_seed(int(s)) for s in seed]
    return lambda rows: [streams[i] for i in rows]
```
