# Prompting: Label Tokens and Task Data (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_prompting` uses and explains in a sentence but does not
show:

- `label_token_ids` (each label word's first token after a prompt, with
  the three checks: the prompt's end merges with a word, two words share a
  first token, a word is several tokens) and its separator rule
  `_label_sep` (a space, none after whitespace or next to a wide
  character);
- the two branches of `label_logprobs` that the section's 1.7B models never
  take: `_label_output_layer` (whether the label logits can be read from
  the output layer's rows) and `_label_logprobs_logits` (a soft-capped
  Transformers head, or a d2l GPT, which takes no attention mask);
- `clean_ag_news`, the cleanup of the HTML and Windows-1252 escapes left in
  AG News titles.

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble (`transformers`, `unicodedata`, `html`, `string`) inside
its functions: a module-level import would join the `_preamble` shard and
move the freshness fingerprint of every notebook in the book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_prompting`

```{.python .input #prompting-lib-label-ids}
%%tab pytorch
#@save
def _label_sep(prompt, word):
    """A space between a prompt and a label word, except after whitespace
    and next to a wide (Chinese, Japanese or Korean) character."""
    import unicodedata
    wide = any(unicodedata.east_asian_width(c) in ('W', 'F')
               for c in prompt[-1:] + word[:1])
    return '' if wide or prompt[-1:].isspace() else ' '

#@save
def label_token_ids(tokenizer, prompts, label_words, sep=None):
    """The prompts' token ids, and per prompt the id of each label word's
    first token as the next token, the word following the prompt after
    sep (by default _label_sep's choice). Raises if (a) the prompt's end
    merges with a word, (b) two words share a first token, (c) a word is
    several tokens, checked in this order."""
    enc = lambda texts: tokenizer(texts, add_special_tokens=False)[
        'input_ids']
    ids, tails = enc(prompts), [[] for _ in prompts]
    seps = [[_label_sep(p, w) if sep is None else sep for w in label_words]
            for p in prompts]
    for j, w in enumerate(label_words):
        full = enc([p + s[j] + w for p, s in zip(prompts, seps)])
        for i, (p, h, f) in enumerate(zip(prompts, ids, full)):
            if f[:len(h)] != h:  # (a) the prompt ends inside a token
                raise ValueError(
                    f'prompt {i} ends inside a token ({p[-12:]!r} merges '
                    f'with {w!r}): strip trailing whitespace from the '
                    'prompt, the space belongs to the label')
            if len(f) == len(h):
                raise ValueError(f'label word {w!r} adds no token')
            tails[i].append(f[len(h):])
    for t in tails:  # (b) two words, one first token
        first = {}
        for w, s in zip(label_words, t):
            if s[0] in first:
                raise ValueError(
                    f'label words {first[s[0]]!r} and {w!r} share the '
                    f'first token {tokenizer.decode(s[:1])!r} ({s[0]})')
            first[s[0]] = w
    for t, seps_i in zip(tails, seps):  # (c) a word of several tokens
        for w, s, sp in zip(label_words, t, seps_i):
            if len(s) > 1:
                raise ValueError(
                    f'label word {w!r} after {sp!r} is {len(s)} tokens '
                    f'{[tokenizer.decode([u]) for u in s]}: use a word '
                    'that is one token there, or another sep')
    return ids, [[s[0] for s in t] for t in tails]
```

```{.python .input #prompting-lib-logits}
%%tab pytorch
#@save
def _label_output_layer(model):
    """The output layer of a Transformers model when its label logits are
    the last hidden state times the label words' rows (a plain linear
    layer, no final logit soft-capping); None otherwise, as for a d2l
    GPT."""
    inner = d2l._lm_unwrap(model)
    hf = sys.modules.get('transformers')  # loaded if model is one of its
    if hf is None or not isinstance(inner, hf.PreTrainedModel):
        return None
    W = inner.get_output_embeddings()
    capped = getattr(inner.config.get_text_config(),
                     'final_logit_softcapping', None) is not None
    return W if isinstance(W, nn.Linear) and not capped else None

#@save
def _label_logprobs_logits(model, tokenizer, ids, labels, batch_size):
    """label_logprobs from the model's own logits at the last position:
    a Transformers model in left-padded batches (logits_to_keep=1), a d2l
    GPT, which takes no attention mask, in batches of prompts of equal
    length."""
    inner = d2l._lm_unwrap(model)
    hf = sys.modules.get('transformers')
    hf = hf is not None and isinstance(inner, hf.PreTrainedModel)
    if hf:  # one padded group, in input order
        groups = [list(range(len(ids)))]
        pad = getattr(tokenizer, 'pad_token_id', None) or 0
    else:
        groups = [[i for i in range(len(ids)) if len(ids[i]) == n]
                  for n in sorted({len(r) for r in ids})]
    device = next(model.parameters()).device
    out = torch.empty(len(ids), len(labels[0]))
    for rows in (g[j:j + batch_size] for g in groups
                 for j in range(0, len(g), batch_size)):
        lab = torch.tensor([labels[i] for i in rows], device=device)
        with torch.inference_mode():
            if hf:
                T = max(len(ids[i]) for i in rows)
                X = torch.tensor([[pad] * (T - len(ids[i])) + ids[i]
                                  for i in rows], device=device)
                mask = torch.tensor([[0] * (T - len(ids[i])) +
                                     [1] * len(ids[i]) for i in rows],
                                    device=device)
                pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
                z = model(input_ids=X, attention_mask=mask, position_ids=pos,
                          use_cache=False, logits_to_keep=1).logits[:, -1]
            else:
                X = torch.tensor([ids[i] for i in rows], device=device)
                z = (model(X, logits_to_keep=1) if isinstance(
                    inner, d2l.Qwen3GPT) else model(X))[:, -1]
            lp = z.gather(-1, lab).float().log_softmax(-1).cpu()
        out[rows] = lp  # out is an ordinary tensor the caller may modify
    return out
```

```{.python .input #prompting-lib-ag-news}
%%tab pytorch
#@save
def clean_ag_news(titles):
    """AG News titles with the escapes left in them resolved: HTML
    entities and tags, '#39;'-style codes without the '&', Windows-1252
    codes 145 to 151, backslashes. Returns the cleaned titles (an object
    array) and, per rule, the number of titles it changed; raises if an
    escape is left that no rule handles."""
    import html
    cp1252 = {'145': "'", '146': "'", '147': '"', '148': '"', '151': '-'}
    codes = {37, 38, 133, 160, 163, 225, 233, 234, 8217, 8482}  # in AG News

    def quote(m):  # an apostrophe or closing quote takes the space before
        return ('' if m[2] in ('145', '146', '148') else m[1]) + cp1252[m[2]]

    def numeric(m):  # '#233;' is 'é'; 128 to 159 are Windows-1252 codes
        n = int(m[2])
        if n not in codes:  # left for the check at the end
            return m[0]
        c = bytes([n]).decode('cp1252') if 128 <= n < 160 else chr(n)
        return ('' if c.isalnum() or c in "&'’" else m[1]) + c
    rules = [  # in this order
        ("#39; is '", lambda s: re.sub(r' ?#39;', "'", s)),
        ('#36; is $', lambda s: s.replace('#36;', '$')),
        ('#145; to #151;', lambda s: re.sub(r'( ?)#(14[5-8]|151);', quote,
                                             s)),
        ('other #n;', lambda s: re.sub(r'( ?)#(\d+);', numeric, s)),
        ('&entities;', html.unescape),
        ('<tags>', lambda s: re.sub(r'<[^<>]*>', '', s)),
        ('\\ and spaces', lambda s: ' '.join(s.replace('\\', ' ').split()))]
    changed = {}
    for name, rule in rules:
        new = [rule(s) for s in titles]
        changed[name] = sum(a != b for a, b in zip(titles, new))
        titles = new
    left = sorted({m for s in titles
                   for m in re.findall(r'#\d+;|&\w+;|\\', s)})
    assert not left, f'no cleanup rule for {left}'
    return np.array(titles, dtype=object), changed
```
