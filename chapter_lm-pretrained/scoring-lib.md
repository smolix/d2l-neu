# Scoring: Tokenization Plumbing and Measured Results (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_scoring` uses and explains in a sentence but does not
show:

- the tokenization behind `score_choices` and `bits_per_byte`:
  `_lm_encode` (any tokenizer, no special tokens), `_lm_boundary_id` (the
  document-boundary token), `_lm_max_len` and `_lm_ctx_len` (the model's
  input limit), `_lm_continuations` (prompt and choice tokenized apart,
  the whitespace move, the merge warning, the left cut) and
  `_lm_components` (the `ScoreComponents` of every pair);
- the branches the section's models never take: `_lm_cont_logits` (the
  model's own logits, for a soft-capped Transformers head or a d2l GPT)
  and `_lm_unscored_head` (a tokenizer without a boundary token);
- the POSIX parts of `run_python_tests`: `_run_limits` (CPU-time and
  address-space limits set in the child) and `_kill_session` (the child
  and every process it started);
- the chapter's measured-result plumbing: `download_keyed` (one folder per
  `DATA_HUB` key), `lm_rerun` (the `D2L_RERUN` switch), `lm_registered`,
  `lm_artifact` and `lm_save_rerun` (registered files against this run's
  regenerated ones), `read_jsonl`, `write_jsonl`, `lm_free` and `lm_peak`.

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble (`transformers`, `warnings`, `gc`, `signal`) inside its functions:
a module-level import would join the `_preamble` shard and move the
freshness fingerprint of every notebook in the book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_scoring`

```{.python .input #scoring-lib-tokens}
%%tab pytorch
#@save
def _lm_encode(tokenizer, texts):
    """Token ids of each text without special tokens: a Transformers
    tokenizer is called with add_special_tokens=False, any other one (such
    as tiktoken's) through its plain encode."""
    hf = sys.modules.get('transformers')
    texts = [str(t) for t in texts]  # also numpy arrays, pandas Series
    if not texts:
        return []
    if hf is not None and isinstance(tokenizer, hf.PreTrainedTokenizerBase):
        return tokenizer(texts, add_special_tokens=False)['input_ids']
    return [list(tokenizer.encode(t)) for t in texts]

#@save
def _lm_boundary_id(tokenizer):
    """The token that separates documents in pretraining: <|endoftext|> if
    the tokenizer has it, else its EOS, else None. Qwen3: 151643, Qwen3.5:
    248044, for base and post-trained checkpoints alike."""
    conv = getattr(tokenizer, 'convert_tokens_to_ids', None)
    if conv is not None:
        i = conv('<|endoftext|>')
        if i is not None and i != getattr(tokenizer, 'unk_token_id', None):
            return int(i)
    elif '<|endoftext|>' in getattr(tokenizer, 'special_tokens_set', ()):
        return tokenizer.encode_single_token('<|endoftext|>')  # tiktoken
    return getattr(tokenizer, 'eos_token_id', None)

#@save
def _lm_max_len(model):
    """The longest input the model takes, or None if it does not say."""
    model = d2l._lm_unwrap(model)
    cfg = getattr(model, 'config', None)
    if cfg is not None and hasattr(cfg, 'get_text_config'):
        cfg = cfg.get_text_config()
    return (getattr(cfg, 'max_position_embeddings', None) or
            getattr(model, 'max_len', None))

#@save
def _lm_ctx_len(model, ctx_len):
    """ctx_len, capped (with a warning) at the model's longest input."""
    import warnings
    max_len = _lm_max_len(model)
    if max_len and ctx_len > max_len:
        warnings.warn(f'bits_per_byte: ctx_len {ctx_len} capped at the '
                      f"model's maximum input length {max_len}")
        return max_len
    return ctx_len
```

```{.python .input #scoring-lib-continuations}
%%tab pytorch
#@save
def _lm_continuations(tokenizer, prompts, choices, max_len=None):
    """Every (prompt, choice) pair of score_choices, as (rows, inputs,
    conts, texts, shape): rows[r] = (i, j) names prompt i and its choice j,
    inputs[r] the token ids the model reads, conts[r] the continuation's
    ids it scores, texts[r] the choice text without one leading
    whitespace character (the harness's length), shape (N, C_max).
    choices holds N lists (ragged) or one list shared by all prompts.
    Trailing whitespace of a prompt moves to the front of its choices; the
    continuation is the suffix of encode(prompt + choice) beyond
    encode(prompt), with a warning when the prompt's last token merges into
    the choice's; an empty prompt is preceded by the document-boundary
    token; a pair longer than max_len loses prompt tokens from the left,
    with a warning naming the item; an empty continuation raises."""
    import warnings
    prompts, choices = list(prompts), list(choices)  # also arrays, Series
    if len(choices) and isinstance(choices[0], str):  # one shared list
        choices = [choices] * len(prompts)
    choices = [list(cs) for cs in choices]
    if len(choices) != len(prompts):
        raise ValueError(f'{len(prompts)} prompts, {len(choices)} choices')
    heads = [p.rstrip() for p in prompts]
    rows = [(i, j) for i, cs in enumerate(choices) for j in range(len(cs))]
    conts = [prompts[i][len(heads[i]):] + choices[i][j] for i, j in rows]
    texts = [c[1:] if c[:1].isspace() else c for c in conts]
    ctx_ids = _lm_encode(tokenizer, heads)
    full = _lm_encode(tokenizer, [heads[i] + c
                                  for (i, _), c in zip(rows, conts)])
    cut_items, inputs, cont_ids = set(), [], []
    merged = []  # pairs whose prompt tokens are not a prefix of the whole
    for (i, j), f in zip(rows, full):
        ctx, cont = ctx_ids[i], f[len(ctx_ids[i]):]
        if not cont:
            raise ValueError(f'item {i}, choice {j}: empty continuation')
        if f[:len(ctx)] != ctx:
            merged.append((i, j))
        if not ctx:
            b = _lm_boundary_id(tokenizer)
            if b is None:
                raise ValueError(f'item {i}: empty prompt and no boundary '
                                 'token to put before it')
            ctx = [b]
        cut = len(ctx) + len(cont) - 1 - (max_len or float('inf'))
        if cut > 0:  # cut the prompt from the left
            if cut >= len(ctx):
                raise ValueError(f'item {i}, choice {j}: the continuation '
                                 f'does not fit the context of {max_len}')
            if i not in cut_items:
                warnings.warn(f'score_choices: prompt of item {i} cut by '
                              f'{cut} tokens to fit the context of {max_len}')
                cut_items.add(i)
            ctx = ctx[cut:]
        inputs.append((ctx + cont)[:-1])  # the last token predicts nothing
        cont_ids.append(cont)
    if merged:
        warnings.warn(
            f'score_choices: in {len(merged)} of {len(rows)} prompt-choice '
            f'pairs (first: item {merged[0][0]}, choice {merged[0][1]}) the '
            "prompt's last token merges with the choice's first, so the "
            'scored tokens do not spell prompt + choice; end the prompts '
            'differently (for example without a newline)')
    shape = (len(prompts), max(map(len, choices), default=0))
    return rows, inputs, cont_ids, texts, shape

#@save
def _lm_components(rows, ll, conts, texts, shape):
    """ScoreComponents of shape (N, C_max) from the pairs of
    _lm_continuations and their summed log-likelihoods ll; padded columns
    hold -inf (ll) and 0."""
    comps = [torch.zeros(shape, dtype=dt)
             for dt in (torch.float32, torch.long, torch.long, torch.long)]
    comps[0].fill_(-math.inf)
    for r, (i, j) in enumerate(rows):
        for t, v in zip(comps, (ll[r].item(), len(conts[r]), len(texts[r]),
                                len(texts[r].encode('utf-8')))):
            t[i, j] = v
    return d2l.ScoreComponents(*comps)

#@save
def _lm_cont_logits(model, inputs, n_cont, pad=0):
    """The model's own logits, in its dtype, at the positions that predict
    the last n_cont[k] tokens of each row inputs[k] + [next token],
    concatenated over the rows: a Transformers model (one whose output
    layer is not a plain linear layer, such as a soft-capped one) in a
    left-padded batch with logits_to_keep; a d2l.Qwen3GPT right-padded,
    with logits_to_keep from the earliest continuation; any other module
    (a d2l.GPT) right-padded, whole (exact under the causal mask)."""
    inner = d2l._lm_unwrap(model)
    hf = sys.modules.get('transformers')
    hf = hf is not None and isinstance(inner, hf.PreTrainedModel)
    device = next(model.parameters()).device
    T = max(map(len, inputs))
    if hf:
        X = torch.tensor([[pad] * (T - len(x)) + x for x in inputs],
                         device=device)
        mask = torch.tensor([[0] * (T - len(x)) + [1] * len(x)
                             for x in inputs], device=device)
        pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
        z = model(input_ids=X, attention_mask=mask, position_ids=pos,
                  use_cache=False, logits_to_keep=max(n_cont)).logits
        return torch.cat([z[k, z.shape[1] - m:]
                          for k, m in enumerate(n_cont)])
    X = torch.tensor([x + [0] * (T - len(x)) for x in inputs], device=device)
    starts = [len(x) - m for x, m in zip(inputs, n_cont)]
    qwen3 = isinstance(inner, d2l.Qwen3GPT)
    off = min(starts) if qwen3 else 0
    out = model(X, logits_to_keep=T - off) if qwen3 else model(X)
    z = getattr(out, 'logits', out)
    return torch.cat([z[k, s - off:s - off + m]
                      for k, (s, m) in enumerate(zip(starts, n_cont))])

#@save
def _lm_unscored_head(tokenizer, text, ids):
    """For a tokenizer without a boundary token: the text's first token has
    no context, so it is not scored, nor any next token up to the end of
    the character the first one starts. Returns how many of ids go
    unscored and the UTF-8 bytes they spell."""
    first = 1
    while first < len(ids):
        head = tokenizer.decode(ids[:first])
        if text.startswith(head) or '�' not in head:
            break
        first += 1
    return first, len(tokenizer.decode(ids[:first]).encode('utf-8'))
```

```{.python .input #scoring-lib-run-tests}
%%tab pytorch
#@save
def _run_limits(timeout):
    """Python source that, run first in the child, limits its CPU time to
    timeout + 1 seconds (rounded up) and its address space to 1 GiB; empty
    off POSIX, where only the wall-clock timeout applies."""
    if os.name != 'posix':
        return ''
    cpu = math.ceil(timeout) + 1
    return ('import resource\n'
            f'for r, v in ((resource.RLIMIT_CPU, {cpu}), '
            '(resource.RLIMIT_AS, 1 << 30)):\n'
            '    try:\n        resource.setrlimit(r, (v, v))\n'
            '    except (ValueError, OSError):\n        pass\n')

#@save
def _kill_session(proc):
    """Kill a child started with start_new_session=True and every process
    it started (on POSIX; elsewhere the child alone), then reap it."""
    import signal
    if os.name == 'posix':
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    else:
        proc.kill()
    proc.wait()
```

```{.python .input #scoring-lib-artifacts}
%%tab pytorch
#@save
def download_keyed(key, root=os.path.join('..', 'data', 'lm')):
    """Download DATA_HUB[key] into root/key/ and return its local path.
    Files from different datasets often share a name (Hugging Face's
    'test-00000-of-00001.parquet'), and d2l.download keeps one file per
    name and folder, so each key gets a folder of its own. The file's sha1
    is checked after a fresh download too."""
    path = d2l.download(key, os.path.join(root, key))
    sha1 = d2l.DATA_HUB[key][1]
    if sha1:
        with open(path, 'rb') as f:
            got = hashlib.sha1(f.read()).hexdigest()
        assert got == sha1, f'{path}: sha1 {got}, DATA_HUB pins {sha1}'
    return path

#@save
def lm_rerun(artifacts):
    """The measured results this run regenerates, from the environment
    variable D2L_RERUN: '1' names all of artifacts, 'a,b' the ones it
    lists; unset, empty or '0', none (a normal build)."""
    names = os.environ.get('D2L_RERUN', '0').split(',')
    rerun = set(artifacts if names == ['1'] else names) - {'0', ''}
    if not rerun <= set(artifacts):
        raise ValueError(f'D2L_RERUN names one of {tuple(artifacts)}, or 1')
    return rerun

#@save
def lm_registered(key):
    """Local path of the registered artifact DATA_HUB[key], or None while
    the key is not registered."""
    return d2l.download_keyed(key) if key in d2l.DATA_HUB else None

#@save
def lm_artifact(key, rerun_name=None,
                rerun_dir=os.path.join('..', 'data', 'lm', 'rerun')):
    """Where a measured result is read from: rerun_dir/rerun_name, the file
    this run regenerated, when rerun_name is given; else the registered
    artifact DATA_HUB[key], which must exist."""
    if rerun_name is not None:
        return os.path.join(rerun_dir, rerun_name)
    path = lm_registered(key)
    if path is None:
        raise KeyError(f'{key} is not registered: regenerate it with '
                       'D2L_RERUN, upload it and register its sha1')
    return path

#@save
def lm_save_rerun(name, write,
                  rerun_dir=os.path.join('..', 'data', 'lm', 'rerun')):
    """Call write(path) for a regenerated artifact in rerun_dir and print
    its path and sha1; the author uploads it under the next -vN key."""
    os.makedirs(rerun_dir, exist_ok=True)
    path = os.path.join(rerun_dir, name)
    write(path)
    with open(path, 'rb') as f:
        print(f'wrote {path}, sha1 {hashlib.sha1(f.read()).hexdigest()}')
    return path

#@save
def read_jsonl(path):
    """A file of JSON lines as a DataFrame, one row per line, values with
    the types they were written with."""
    with open(path, encoding='utf-8') as f:
        return pd.DataFrame([json.loads(line) for line in f])

#@save
def write_jsonl(path, records):
    """Write records (dicts) as JSON lines."""
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n'
                     for r in records)

#@save
def lm_free():
    """Return the memory of deleted models to the GPU."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

#@save
def lm_peak():
    """Peak GPU memory since the last call, as text: what tensors held, and
    what the allocator reserved from the card (what nvidia-smi shows, less
    the CUDA context)."""
    if not torch.cuda.is_available():
        return 'no GPU'
    gib = [f() / 2**30 for f in (torch.cuda.max_memory_allocated,
                                 torch.cuda.max_memory_reserved)]
    torch.cuda.reset_peak_memory_stats()
    return f'peak {gib[0]:.1f} GiB ({gib[1]:.1f} GiB reserved)'
```
