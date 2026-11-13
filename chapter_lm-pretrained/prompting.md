# Prompting and In-Context Learning
:label:`sec_lm_prompting`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "1.3 Prompting and In-Context Learning · T1").
Blueprint: docs/llm/blueprints/ch01_pretrained.md section 1.3.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** The prompt is tokens, and few-shot accuracy depends on how the demonstrations are laid out.
**Opening:** The reader asks a base checkpoint to classify a news headline and gets a continuation instead of a label. Four solved examples placed before the query lift accuracy by tens of points, and the rendered template shows why: the model continues a pattern of tokens. The section measures what in that pattern matters and closes on what a few-shot score therefore measures.
**Teaches:**
- A chat template as token ids: roles, special tokens, the generation prefix and the thinking switch, printed as the rendered tail so the text cannot drift from the template file; `d2l.fewshot_prompt` renders plain, question-answer and chat layouts.
- Zero- and few-shot formats; scoring by restricted next-token logits over the label words (`d2l.label_logprobs`), the special case that 1.4 generalizes.
- Sensitivity to the number, order and label correctness of demonstrations, with contextual calibration as the one correction worked (`d2l.calibrate_labels`).
- In-context learning against a weight update: the same labelled examples as a probe on frozen features, at equal labelled data and unequal cost.
**Derivation:** none.
**Notebook:**
1. Render the chat template of Qwen3-1.7B for one conversation and print the token ids with the special tokens marked; render the same demonstrations as plain text for the base checkpoint.
2. On 200 stratified test items of a four-class news task, measure accuracy of Qwen3-1.7B-Base and Qwen3-1.7B against the number of demonstrations (0, 1, 2, 4, 8, 16) with 3 demonstration seeds.
3. At 8 demonstrations, measure accuracy over four orders of the same demonstrations per seed; then over a fraction of deliberately wrong labels (0, 0.25, 0.5, 0.75, 1.0) with natural and with abstract label words.
4. Fit a logistic probe on the frozen base model's mean-pooled features with the same labelled examples and add it to the plot.
5. Claim that survives noise: order alone moves accuracy by several points (the spread over orders is comparable to the spread over demonstration sets); shuffled labels cost less than removing the demonstrations altogether for natural label words. Each claim needs a paired bootstrap interval over items that excludes zero and an effect larger than the across-seed standard deviation; 200 items give about plus or minus 7 points unpaired, so the text interprets only paired differences.
6. Reused pieces: `d2l.load_hf`, `d2l.generate` for free-generation spot checks, a 6-line local bootstrap (the shared `d2l.paired_bootstrap` arrives in 1.5 and the text says so).
**Figures:** `fig_lm_chat_template` (the rendered conversation as a token sequence with roles, special tokens and the generation prefix labelled); `fig_lm_icl_sensitivity` (plot: accuracy against demonstration count for base, instruct and the probe; a strip plot of accuracy over orders at k = 8).
**Restate / forward:** `sec_what-attention-computes` (an induction head copies what followed an earlier occurrence of the current token; the repeated-sequence probe is the behavioural signature of in-context copying, and `tab_induction-evidence` says what a behavioural result can claim). Forward: 1.4 (what a restricted-logit score does not measure).
**Exercises:** 5 to 7. Load-bearing: measure the spread of accuracy over 20 orders of the same 8 demonstrations and compare it with the spread over 20 demonstration sets (measure); derive why subtracting the content-free log-probabilities shifts the model's label prior and when it cannot help (derive); open: design a prompt format whose few-shot accuracy for the base model is within the order spread of the instruct model's, and report the cost per query of each.
**Writer notes:**
- Exports: `d2l.fewshot_prompt(demos, query, template='plain', *, label_map=None, system=None)` returning a string or a message list; `d2l.label_logprobs(model, tokenizer, prompts, label_words, *, batch_size=8, chat=False, enable_thinking=False)` returning log-probabilities renormalized over the label set, which asserts the label words' first tokens are distinct and single after the format's separator (the leading-space convention: encode `separator + word` and take the last token of the difference); `d2l.calibrate_labels(logprobs, content_free_logprobs)`.
- Silent breakages: leading-space tokenization of label words (`" Sports"` versus `"Sports"`) when a format changes; chat-templating a base model or omitting the template for an instruct model and attributing the change to ability; Qwen3 post-trained defaults to thinking on and Qwen3.5 to off, read from the pinned template file; test items or near-duplicates leaking into the demonstration pool.
- Data: `fancyzhx/ag_news` (train 120,000 for demonstrations, test 7,600; licence listed as unknown on the Hub, so the notebook downloads it at run time and commits only item indices and predictions); the fallback with a clear licence is ARC-style multiple choice (`allenai/ai2_arc`, CC BY-SA 4.0), with the same code.
- The notebook reduces the blueprint's grid: four orders instead of twelve (twenty orders is the exercise), three seeds, 200 items; the committed per-item grid `lm-fewshot-grid.csv` makes the figure cells seconds at T0.
- What a few-shot score measures: with natural labels much of the gain is format and label-space information rather than the input-label mapping (Min et al.); the section says this and does not call it "learning the task". Restricted-logit accuracy is not free-generation accuracy; 1.4 owns that distinction.
- Cost: few-shot is not free; the probe comparison is at equal labelled data, not equal cost, and the text says which.
- Do not re-teach the induction circuit; restate it in two sentences with the label.
- Decide at authoring, with the editor, whether the clearly licensed ARC fallback is the default task.
**Sources:**
- Papers: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (few-shot as a capability of scale); [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837) (label correctness matters less than format and label space); [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (the mechanistic account restated from `sec_what-attention-computes`); [Calibrate Before Use](https://arxiv.org/abs/2102.09690) (contextual calibration); [Fantastically Ordered Prompts and Where to Find Them](https://arxiv.org/abs/2104.08786) (order sensitivity).
- Code and documentation: Transformers, [chat templating](https://huggingface.co/docs/transformers/chat_templating) (the rendered template and `add_generation_prompt`); [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (its few-shot construction options as the oracle).
- Courses, tutorials and blogs: [CS224N](https://web.stanford.edu/class/cs224n/) (prompting and efficient adaptation lecture); [Hugging Face LLM Course, chapter 2](https://huggingface.co/learn/llm-course/chapter2/1) (tokenizers and templates).
-->

```{.python .input #prompting-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #prompting-imports-extra}
%%tab pytorch
import gc
import hashlib
import html
import itertools
import json
import os
import re
import string
import sys
import time
import pandas as pd
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
```

<!-- Placeholder (prose, opening): the two checkpoints, the label words, and the invented headlines of the examples. -->

```{.python .input #prompting-setup}
%%tab pytorch
SMALL = not torch.cuda.is_available()  # no GPU: the 0.6B pair in fp32
RERUN = os.environ.get('D2L_RERUN', '0') == '1' and not SMALL
BASE, INST = (('Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3-0.6B') if SMALL
              else ('Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3-1.7B'))
NATURAL = ['World', 'Sports', 'Business', 'Tech']  # AG News's 4 classes
ABSTRACT = ['foo', 'bar', 'baz', 'zip']
FMT = 'Headline: {x}\nTopic: {y}'
ASK = 'What is the topic of this news headline?\n{x}\n'
SYSTEM = ('Name the topic of each news headline: World, Sports, Business '
          'or Tech.')
QUERY = 'Local team wins cup final'  # invented, as are all examples below
DEMOS = [('Leaders meet to settle a border dispute', 'World'),
         ('Striker signs a new three-year contract', 'Sports'),
         ('Chip maker shares climb after strong earnings', 'Business'),
         ('New phone battery lasts a whole week', 'Tech')]
HEADLINES = [QUERY, 'Parliament votes on a new trade treaty',
             'Oil prices fall as supply grows',
             'Researchers build a faster quantum chip']
transformers.logging.set_verbosity_error()  # no notices or progress bars
transformers.logging.disable_progress_bar()
```

<!-- Placeholder (prose, opening): a base checkpoint and its post-trained sibling, pinned. -->

```{.python .input #prompting-load-models}
%%tab pytorch
dtype = torch.float32 if SMALL else torch.bfloat16
base, tok = d2l.load_hf(BASE, dtype=dtype)
inst, tok_i = d2l.load_hf(INST, dtype=dtype)
for name, model in ((BASE, base), (INST, inst)):
    print(f'{name}: {model.num_parameters():,} parameters in {model.dtype}')
if SMALL:
    print('No GPU: the 0.6B pair in fp32 stands in for the 1.7B models, so '
          'the outputs below differ from those quoted in the text.')
```

<!-- Placeholder (prose, opening): asked plainly, the base model continues the text; four solved headlines change what follows. -->

```{.python .input #prompting-opening-generate}
%%tab pytorch
for prompt in (ASK.format(x=QUERY),
               d2l.fewshot_prompt(DEMOS, QUERY, fmt=FMT)):
    reply = d2l.generate(base, tok, [prompt], max_new_tokens=12)[0]
    print(f'{prompt}<continuation>{reply.text!r}\n')
```

<!-- Placeholder (prose, ### Chat Templates): one conversation rendered by the template, then as token ids with the special tokens marked. -->

```{.python .input #prompting-chat-template}
%%tab pytorch
conversation = [
    {'role': 'system', 'content': 'Name the topic of the headline.'},
    {'role': 'user', 'content': 'Headline: ' + DEMOS[1][0]},
    {'role': 'assistant', 'content': DEMOS[1][1]},
    {'role': 'user', 'content': 'Headline: ' + QUERY}]
chat_text = tok_i.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=True,
    enable_thinking=False)
print(chat_text)
start = tok_i.convert_tokens_to_ids('<|im_start|>')
special = set(tok_i.added_tokens_decoder)  # <|im_start|>, <think>, ...
line = []
for i in tok_i(chat_text, add_special_tokens=False)['input_ids']:
    if i == start and line:  # one line per message
        print(' '.join(line))
        line = []
    t = tok_i.decode([i]).replace('\n', '\\n')
    line.append(f'[{i}:{t}]' if i in special else f'{i}:{t}')
print(' '.join(line))
```

<!-- Placeholder (prose, ### Chat Templates): the generation prefix, and whether each family thinks by default (the table below). -->

```{.python .input #prompting-template-tails}
%%tab pytorch
rows = []
for name in ('Qwen/Qwen3-1.7B', 'Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3.5-2B',
             'Qwen/Qwen3.5-4B'):
    t = AutoTokenizer.from_pretrained(name, revision=d2l.HF_REVISIONS[name])
    render = lambda **kw: t.apply_chat_template(
        [{'role': 'user', 'content': QUERY}], tokenize=False, **kw)
    body = render(add_generation_prompt=False)
    unset, off = (render(add_generation_prompt=True, **kw)[len(body):]
                  for kw in ({}, {'enable_thinking': False}))
    rows.append({'model': name, 'tail, enable_thinking unset': repr(unset),
                 'tail, enable_thinking=False': repr(off),
                 'thinks by default': 'no' if '</think>' in unset else 'yes'})
pd.DataFrame(rows)
```

<!-- Placeholder (stage 2): read a label from the next token, over the label words only. -->

```{.python .input #prompting-label-logprobs}
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
def _label_token_ids(tokenizer, prompts, label_words, sep=None):
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

#@save
def label_logprobs(model, tokenizer, prompts, label_words, *, batch_size=8,
                   chat=False, enable_thinking=False, sep=None):
    """Log-probabilities (len(prompts), len(label_words)), float32 on the
    CPU, of each label word's first token as the next token, normalized
    over the label tokens only. A word follows its prompt after sep, by
    default a space, but none after whitespace or next to Chinese text.
    With chat=True a prompt is a message list (a string is one user
    message), rendered with the generation prefix. Transformers models
    get left-padded batches; a d2l GPT, which takes no attention mask,
    gets batches of prompts of equal length."""
    if chat:
        opts = {} if enable_thinking is None else {
            'enable_thinking': enable_thinking}
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': m}] if isinstance(m, str) else m,
            tokenize=False, add_generation_prompt=True, **opts)
                   for m in prompts]
    ids, labels = _label_token_ids(tokenizer, prompts, label_words, sep)
    if not chat:  # as d2l.generate: a plain prompt gets the BOS, if any
        head = d2l._lm_bos_ids(tokenizer)  # and never an EOS
        ids = [head + r for r in ids]
    if not all(ids):
        raise ValueError('an empty prompt has no position to predict from')
    hf = sys.modules.get('transformers')
    hf = hf is not None and isinstance(d2l._lm_unwrap(model),
                                       hf.PreTrainedModel)
    if hf:  # one padded group, in input order
        groups = [list(range(len(ids)))]
        pad = getattr(tokenizer, 'pad_token_id', None) or 0
    else:
        groups = [[i for i in range(len(ids)) if len(ids[i]) == n]
                  for n in sorted({len(r) for r in ids})]
    device = next(model.parameters()).device
    out = torch.empty(len(ids), len(label_words))
    for rows in (g[j:j + batch_size] for g in groups
                 for j in range(0, len(g), batch_size)):
        with torch.inference_mode():
            if hf:  # logits_to_keep=1 and no cache: no full-vocabulary
                T = max(len(ids[i]) for i in rows)  # logits per position
                X = torch.tensor([[pad] * (T - len(ids[i])) + ids[i]
                                  for i in rows], device=device)
                mask = torch.tensor([[0] * (T - len(ids[i])) +
                                     [1] * len(ids[i]) for i in rows],
                                    device=device)
                pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
                z = model(input_ids=X, attention_mask=mask,
                          position_ids=pos, use_cache=False,
                          logits_to_keep=1).logits[:, -1]
            else:
                X = torch.tensor([ids[i] for i in rows], device=device)
                z = (model(X, logits_to_keep=1) if isinstance(
                    d2l._lm_unwrap(model), d2l.Qwen3GPT) else model(X))[:, -1]
            lab = torch.tensor([labels[i] for i in rows], device=z.device)
            lp = z.gather(-1, lab).float().log_softmax(-1).cpu()
        out[rows] = lp  # out is an ordinary tensor the caller may modify
    return out
```

<!-- Placeholder (prose, ### Reading a Label from the Next Token): the reader against the full softmax, and batched against one prompt at a time. -->

```{.python .input #prompting-reader-check}
%%tab pytorch
prompts = [d2l.fewshot_prompt(DEMOS, h, fmt=FMT) for h in HEADLINES]
_, label_ids = d2l._label_token_ids(tok, prompts, NATURAL)
one = d2l.label_logprobs(base, tok, prompts, NATURAL, batch_size=1)
for prompt, ids, lp in zip(prompts, label_ids, one):
    x = tok(prompt, return_tensors='pt').input_ids.to(base.device)
    with torch.inference_mode():  # every position's logits, then the last
        z = base(input_ids=x, use_cache=False).logits[0, -1].float()
    ref = z.log_softmax(-1)[ids].cpu()
    ref = ref - ref.logsumexp(-1)  # renormalized over the label words
    # fp32: rounding only; bf16: two ulps of the largest label logit
    tol = 1e-4 if SMALL else 2 * 2.0 ** (
        math.floor(math.log2(z[ids].abs().max().item())) - 7)
    err = (lp - ref).abs().max().item()
    print(f'reader vs full softmax: {err:.1e} nats (tolerance {tol:.1e})')
    assert err <= tol
batched = d2l.label_logprobs(base, tok, prompts, NATURAL, batch_size=4)
err = (batched - one).abs().max().item()
same = torch.equal(batched.argmax(-1), one.argmax(-1))
print(f'batched vs one at a time: {err:.1e} nats, same argmax: {same}')
assert err <= 0.25 and same
```

<!-- Placeholder (prose, ### Reading a Label from the Next Token): one word, two token ids; a prompt ending in a space; a label of three tokens. -->

```{.python .input #prompting-label-tokens}
%%tab pytorch
_, (after_colon,) = d2l._label_token_ids(tok, ['Topic:'], NATURAL)
_, (after_newline,) = d2l._label_token_ids(tok_i, [chat_text], NATURAL)
d2l._label_token_ids(tok, ['Topic:'], ABSTRACT)  # raises unless 1 token
print(pd.DataFrame({
    'word': NATURAL,
    'after "Topic:"': [f'{i} {tok.decode([i])!r}' for i in after_colon],
    'after a newline': [f'{i} {tok.decode([i])!r}' for i in after_newline]
    }).to_string(index=False))
assert not set(after_colon) & set(after_newline)

def label_mass(texts, ids):  # probability of the label tokens, full softmax
    with torch.inference_mode():
        return np.mean([base(input_ids=tok(t, return_tensors='pt').input_ids
                             .to(base.device), use_cache=False)
                        .logits[0, -1].float().softmax(-1)[ids].sum().item()
                        for t in texts])

print(f'label mass after "Topic:": {label_mass(prompts, after_colon):.3f}; '
      f'after "Topic: ", read at the bare words: '
      f'{label_mass([p + " " for p in prompts], after_newline):.3f}')
for texts, words in (([p + ' ' for p in prompts], NATURAL),
                     (prompts, ['World', 'Sports', 'Business', 'Sci/Tech'])):
    try:
        d2l.label_logprobs(base, tok, texts, words)
    except ValueError as e:
        print('ValueError:', e)
```

<!-- Placeholder (prose, ### Reading a Label from the Next Token): one zero-shot item, three pairings of checkpoint and layout. -->

```{.python .input #prompting-template-mismatch}
%%tab pytorch
chat = d2l.fewshot_prompt([], QUERY, 'chat_multi', system=SYSTEM)
plain = d2l.fewshot_prompt([], QUERY, fmt=FMT, system=SYSTEM)
runs = {'instruct, chat template': (inst, tok_i, chat, True),
        'instruct, plain text': (inst, tok_i, plain, False),
        'base, chat template': (base, tok, chat, True)}
probs = {name: d2l.label_logprobs(m, t, [p], NATURAL, chat=c)[0].exp()
         for name, (m, t, p, c) in runs.items()}
pd.DataFrame(probs, index=NATURAL).T.round(3)
```

<!-- Placeholder (stage 2): one set of demonstrations, four layouts. -->

```{.python .input #prompting-fewshot-prompt}
%%tab pytorch
#@save
def fewshot_prompt(demos, query, template='plain', *, fmt=None,
                   label_map=None, system=None):
    """Demonstrations demos, a list of (x, y), and a query x as a prompt.
    'plain' (fmt, default 'Input: {x}\\nOutput: {y}') and 'qa'
    ('Q: {x}\\nA: {y}') give a string whose blocks are separated by a blank
    line and which ends after the query's label field, without a space;
    'chat_single' gives the same text as one user message, 'chat_multi'
    one user and one assistant message per demonstration. label_map maps
    each y to its label word."""
    if template not in ('plain', 'qa', 'chat_single', 'chat_multi'):
        raise ValueError(f'unknown template: {template!r}')
    if fmt is not None:
        import string
        if template not in ('plain', 'chat_single'):
            raise ValueError(f'template {template!r} takes no fmt')
        parts = list(string.Formatter().parse(fmt))
        fields = [f for _, f, _, _ in parts if f is not None]
        tail = parts[-1][0] if parts and parts[-1][1] is None else ''
        if (set(fields) != {'x', 'y'} or fields.count('y') > 1 or
                fields[-1] != 'y' or tail.strip()):
            raise ValueError('fmt needs the fields {x} and {y}, with {y} '
                             f'once and last: {fmt!r}')

    def word(y):
        if label_map is not None:
            if y not in label_map:
                raise KeyError(f'label {y!r} is not in label_map')
            y = label_map[y]
        if not isinstance(y, str):
            raise TypeError(f'label {y!r} is not a string: pass label_map')
        return y
    demos = [(x, word(y)) for x, y in demos]
    head = [] if system is None else [{'role': 'system', 'content': system}]
    if template == 'chat_multi':
        return head + [{'role': r, 'content': c} for x, y in demos
                       for r, c in (('user', x), ('assistant', y))] + [
                           {'role': 'user', 'content': query}]
    fmt = 'Q: {x}\nA: {y}' if template == 'qa' else (
        fmt or 'Input: {x}\nOutput: {y}')
    text = '\n\n'.join([fmt.format(x=x, y=y) for x, y in demos] +
                       [fmt.format(x=query, y='').rstrip()])
    if template == 'chat_single':
        return head + [{'role': 'user', 'content': text}]
    return text if system is None else system + '\n\n' + text
```

<!-- Placeholder (prose, ### Demonstrations and Formats): two demonstrations in the four layouts, the chat ones as the template renders them. -->

```{.python .input #prompting-formats}
%%tab pytorch
layouts = {
    'plain': d2l.fewshot_prompt(DEMOS[:2], QUERY, fmt=FMT),
    'qa': d2l.fewshot_prompt(DEMOS[:2], QUERY, 'qa'),
    'chat_multi': d2l.fewshot_prompt(DEMOS[:2], QUERY, 'chat_multi',
                                     system=SYSTEM),
    'chat_single': d2l.fewshot_prompt(DEMOS[:2], QUERY, 'chat_single',
                                      fmt=FMT)}
for name, prompt in layouts.items():
    if not isinstance(prompt, str):  # a message list
        prompt = tok_i.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    print(f'--- {name}\n{prompt}')
```

<!-- Placeholder (stage 2): the task data and the registered few-shot grid. -->

```{.python .input #prompting-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['ag-news-train'] = (
    'https://huggingface.co/datasets/sh0416/ag_news/resolve/'
    '70e3fa1915be9a8daebec5e840f20df9a8e18793/train.jsonl',
    '4b8aa80c8d0e1bea19ec0924b720317a6a61b040')
d2l.DATA_HUB['ag-news-test'] = (
    'https://huggingface.co/datasets/sh0416/ag_news/resolve/'
    '70e3fa1915be9a8daebec5e840f20df9a8e18793/test.jsonl',
    '6c874e265fe5ff35861f27eaf3e681dda147ba12')
```

<!-- Placeholder (prose, ### Number of Demonstrations): the task data, its cleanup rules (HTML and Windows-1252 escapes left in the titles), and the key that finds near-duplicates. -->

```{.python .input #prompting-data-clean}
%%tab pytorch
def read_jsonl(key):
    path = d2l.download(key, os.path.join('..', 'data', 'lm', key))
    with open(path, encoding='utf-8') as f:
        return pd.DataFrame([json.loads(line) for line in f])

CP1252 = {'145': "'", '146': "'", '147': '"', '148': '"', '151': '-'}

def cp1252(m):  # an apostrophe or closing quote takes the space before it
    return ('' if m[2] in ('145', '146', '148') else m[1]) + CP1252[m[2]]

CODES = {37, 38, 133, 160, 163, 225, 233, 234, 8217, 8482}  # in AG News

def numeric(m):  # '#233;' is 'é'; 128 to 159 are Windows-1252 codes
    n = int(m[2])
    if n not in CODES:  # left for the check at the end of clean()
        return m[0]
    c = bytes([n]).decode('cp1252') if 128 <= n < 160 else chr(n)
    return ('' if c.isalnum() or c in "&'’" else m[1]) + c

CLEANUP = [  # in this order
    ("#39; is '", lambda s: re.sub(r' ?#39;', "'", s)),
    ('#36; is $', lambda s: s.replace('#36;', '$')),
    ('#145; to #151;', lambda s: re.sub(r'( ?)#(14[5-8]|151);', cp1252, s)),
    ('other #n;', lambda s: re.sub(r'( ?)#(\d+);', numeric, s)),
    ('&entities;', html.unescape),
    ('<tags>', lambda s: re.sub(r'<[^<>]*>', '', s)),
    ('\\ and spaces', lambda s: ' '.join(s.replace('\\', ' ').split()))]

def clean(titles):
    changed = {}
    for name, rule in CLEANUP:
        new = [rule(s) for s in titles]
        changed[name] = sum(a != b for a, b in zip(titles, new))
        titles = new
    left = sorted({m for s in titles
                   for m in re.findall(r'#\d+;|&\w+;|\\', s)})
    assert not left, f'no cleanup rule for {left}'
    return np.array(titles, dtype=object), changed

def dedup_key(title):  # no source tag such as '(AP)', no punctuation
    title = re.sub(r'\s*\([^()]*\)\s*$', '', title).lower()
    title = title.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(title.split())
```

<!-- Placeholder (prose, ### Number of Demonstrations): the test items and the demonstration pool without test duplicates; counts only, no article text. -->

```{.python .input #prompting-data}
%%tab pytorch
N_PER_CLASS = 50
train, test = read_jsonl('ag-news-train'), read_jsonl('ag-news-test')
TRAIN_TITLES, changed_train = clean(list(train.title))
TEST_TITLES, changed_test = clean(list(test.title))
TRAIN_Y, TEST_Y = train.label.to_numpy() - 1, test.label.to_numpy() - 1
ITEMS = np.sort(test.groupby('label').head(N_PER_CLASS).index.to_numpy())
GOLD = TEST_Y[ITEMS]
assert np.bincount(GOLD).tolist() == [N_PER_CLASS] * 4
test_keys = {dedup_key(t) for t in TEST_TITLES}
in_pool = np.array([dedup_key(t) not in test_keys for t in TRAIN_TITLES])
POOL = [np.flatnonzero(in_pool & (TRAIN_Y == c)) for c in range(4)]
print(f'{len(ITEMS)} test items, {N_PER_CLASS} per class; pool per class '
      f'{[len(p) for p in POOL]}; {(~in_pool).sum()} train titles '
      'removed as duplicates of test titles')
print(pd.DataFrame({'train': changed_train, 'test': changed_test}).T
      .to_string())
```

<!-- Placeholder (prose, ### Number of Demonstrations): the declared draws of demonstrations, their orders, wrong labels and relabelings, each from a declared seed. -->

```{.python .input #prompting-draws}
%%tab pytorch
DRAWS = range(5)
DRAW = {}  # train rows of each draw: blocks of one title per class
for s in DRAWS:
    rng = np.random.default_rng(s)
    picks = [rng.choice(POOL[c], 64, replace=False) for c in range(4)]
    DRAW[s] = np.array([picks[c][b] for b in range(64)
                        for c in rng.permutation(4)])
ORDERS = {s: [np.arange(8)] + [np.random.default_rng(100 + 10 * s + o)
                               .permutation(8) for o in (1, 2, 3)]
          for s in DRAWS}
NOISE = {}  # per draw: a wrong label per position, and the order of use
for s in DRAWS:
    rng = np.random.default_rng(1000 + s)
    wrong = (TRAIN_Y[DRAW[s][:8]] + rng.integers(1, 4, size=8)) % 4
    NOISE[s] = wrong, rng.permutation(8)
DERANGEMENTS = [p for p in itertools.permutations(range(4))
                if all(i != j for i, j in enumerate(p))]
SIGMA = {s: np.array(DERANGEMENTS[np.random.default_rng(2000 + s)
                                  .integers(len(DERANGEMENTS))])
         for s in DRAWS}
```

<!-- Placeholder (prose, ### Number of Demonstrations): the conditions of the grid, each a change of one factor from the reference condition. -->

```{.python .input #prompting-grid-def}
%%tab pytorch
MODELS, WORDS = ['base', 'instruct'], ['natural', 'abstract']
K_BASE, K_INST = [0, 1, 2, 4, 8, 16, 32, 64], [0, 1, 2, 4, 8, 16]
COND = dict(model='base', layout='plain', k=8, words='natural', rho=0.0,
            derange=False, order=0, instr=0)  # the reference condition
CONTENT_FREE = ['N/A', '', '[MASK]']  # in place of a headline

def conditions():
    """Every (condition, draw) of the grid; without demonstrations a
    condition has the single draw -1."""
    out = []
    def add(**kw):
        for s in ([-1] if kw.get('k', 8) == 0 else DRAWS):
            out.append(dict(COND, **kw, seed=s))
    for model, ks in zip(MODELS, (K_BASE, K_INST)):
        for k in ks:
            add(model=model, k=k)
        add(model=model, k=0, instr=1)  # an instruction names the labels
        for k in ks if model == 'instruct' else [8]:  # the chat layout
            add(model=model, layout='chat', k=k)
        for order in (1, 2, 3):
            add(model=model, order=order)
        for words, rhos in (('natural', (0.25, 0.5, 0.75, 1.0)),
                            ('abstract', (0.0, 0.25, 0.5, 0.75, 1.0))):
            for rho in rhos:
                add(model=model, words=words, rho=rho)
            add(model=model, words=words, derange=True)
    return out
```

<!-- Placeholder (prose, ### Number of Demonstrations): a condition's demonstrations and prompts. -->

```{.python .input #prompting-grid-prompts}
%%tab pytorch
def demos(cond):
    """The (title, label index) demonstrations of one condition."""
    s, k = cond['seed'], cond['k']
    if k == 0:
        return []
    rows = DRAW[s][:k][ORDERS[s][cond['order']]] if k == 8 else DRAW[s][:k]
    y = TRAIN_Y[rows].copy()
    wrong, positions = NOISE[s]
    hit = positions[:round(8 * cond['rho'])]  # nested in rho
    assert not len(hit) or (k == 8 and cond['order'] == 0)
    y[hit] = wrong[hit]
    return list(zip(TRAIN_TITLES[rows],
                    SIGMA[s][y] if cond['derange'] else y))

def build_prompts(cond, titles):
    """One condition's prompts for the titles: strings, or message lists
    in the chat layout."""
    words = dict(enumerate(NATURAL if cond['words'] == 'natural'
                           else ABSTRACT))
    if cond['layout'] == 'ask':
        return [ASK.format(x=x) for x in titles]
    ds = demos(cond)
    if cond['layout'] == 'chat':
        return [d2l.fewshot_prompt(ds, x, 'chat_multi', label_map=words,
                                   system=SYSTEM) for x in titles]
    system = SYSTEM if cond['instr'] else None
    return [d2l.fewshot_prompt(ds, x, fmt=FMT, label_map=words,
                               system=system) for x in titles]
```

<!-- Placeholder (prose, ### Number of Demonstrations): every condition scored on the same items, and the prompt tokens the grid costs. -->

```{.python .input #prompting-grid-size}
%%tab pytorch
def n_tokens(tokenizer, prompts):  # per prompt, as the reader sees it
    texts = [p if isinstance(p, str) else tokenizer.apply_chat_template(
        p, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for p in prompts]
    return [len(i) for i in
            tokenizer(texts, add_special_tokens=False)['input_ids']]

GRID, TITLES = conditions(), list(TEST_TITLES[ITEMS]) + CONTENT_FREE
TOKENS = {}
for model, tokenizer in zip(MODELS, (tok, tok_i)):
    conds = [c for c in GRID if c['model'] == model]
    TOKENS[model] = sum(sum(n_tokens(tokenizer, build_prompts(c, TITLES)))
                        for c in conds)
    print(f'{model}: {len(conds)} (condition, draw) pairs, '
          f'{len(conds) * len(TITLES):,} prompts, {TOKENS[model]:,} tokens')
assert [sum(c['model'] == m for c in GRID) for m in MODELS] == [112, 123]
```

<!-- Placeholder (prose, ### Number of Demonstrations): one row per condition, draw and item: the four label log-probabilities and the prediction. -->

```{.python .input #prompting-grid-score}
%%tab pytorch
COLS = ['model', 'exp', 'layout', 'k', 'words', 'rho', 'perm', 'order',
        'instr', 'seed', 'item', 'lp0', 'lp1', 'lp2', 'lp3', 'pred', 'valid']
IDS = list(ITEMS) + [-1, -2, -3]  # test rows, then the content-free inputs

def head(cond, exp='icl'):  # the condition columns of a grid row
    perm = ''.join(map(str, SIGMA[cond['seed']])) if cond['derange'] else ''
    return [cond['model'], exp, cond['layout'], cond['k'], cond['words'],
            cond['rho'], perm, cond['order'], cond['instr'], cond['seed']]

def score(cond):
    """Grid rows of one condition: its test items and content-free inputs,
    in fixed batches (bf16 results depend on the batch)."""
    model, tokenizer = (base, tok) if cond['model'] == 'base' else (
        inst, tok_i)
    words = NATURAL if cond['words'] == 'natural' else ABSTRACT
    lp = d2l.label_logprobs(
        model, tokenizer, build_prompts(cond, TITLES), words,
        batch_size=32 if cond['k'] <= 16 else 8, chat=cond['layout'] == 'chat')
    return [head(cond) + [i, *r.tolist(), int(r.argmax()), 1]
            for i, r in zip(IDS, lp)]

def read_grid(path):
    return pd.read_csv(path, dtype={'perm': str}).fillna({'perm': ''})

def matched(new, old, exp):  # rows of both files, keyed by condition, item
    return new[new.exp == exp].merge(old[old.exp == exp], on=COLS[:11],
                                     suffixes=('', '_old'))
```

<!-- Placeholder (prose, ### Number of Demonstrations): free generation, read by a declared rule. -->

```{.python .input #prompting-grid-gen}
%%tab pytorch
GEN = [dict(COND, k=0, layout='ask', seed=-1), dict(COND, k=0, seed=-1)] + [
    dict(COND, k=4, seed=s) for s in DRAWS]  # free generation, base model

def answer(text):
    """A free generation's label index by the declared rule, else -1."""
    line = next((s for s in text.split('\n') if s.strip()), '')
    word = line.strip(string.whitespace + string.punctuation).lower()
    return next((j for j, w in enumerate(NATURAL) if w.lower() == word), -1)

def generations(cond):  # greedy, 8 tokens, the test items only
    out = d2l.generate(base, tok, build_prompts(cond, TITLES[:len(ITEMS)]),
                       max_new_tokens=8, batch_size=32)
    return [head(cond, 'gen') + [i, None, None, None, None, p, int(p >= 0)]
            for i, p in zip(ITEMS, (answer(g.text) for g in out))]
```

<!-- Placeholder (prose, ### Number of Demonstrations): the registered grid, one condition re-scored live, and (D2L_RERUN=1 only) the whole grid regenerated. -->

```{.python .input #prompting-grid-run}
%%tab pytorch
RERUN_PATH = os.path.join('..', 'data', 'lm', 'rerun', 'lm-fewshot-grid.csv')
registered = None
if not RERUN or 'lm-fewshot-grid-v1' in d2l.DATA_HUB:
    registered = read_grid(d2l.download('lm-fewshot-grid-v1',
                                        os.path.join('..', 'data', 'lm')))
if RERUN:  # regenerate the grid (probe rows: prompting-probe)
    torch.cuda.reset_peak_memory_stats()
    tic = time.time()
    rows = [r for c in GRID for r in score(c)]
    scored = time.time() - tic  # the label read-out alone
    rows += [r for c in GEN for r in generations(c)]
    wall = time.time() - tic
    os.makedirs(os.path.dirname(RERUN_PATH), exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(RERUN_PATH, index=False,
                                            float_format='%.6g')
    grid = read_grid(RERUN_PATH)
    tokens = sum(TOKENS.values())
    print(f'{len(grid):,} rows in {wall / 60:.1f} min; {tokens:,} prompt '
          f'tokens scored in {scored / 60:.1f} min ({tokens / scored:,.0f} '
          f'tokens/s); peak {torch.cuda.max_memory_allocated() / 2**30:.1f}'
          ' GiB')
    for exp in ('icl', 'gen') if registered is not None else ():
        m = matched(grid, registered, exp)
        print(f'{exp}: {(m.pred == m.pred_old).sum():,} of {len(m):,} '
              'predictions as registered')
else:
    grid = registered
    print(f'lm-fewshot-grid-v1: {len(grid):,} rows')
    if not SMALL:  # one declared condition, re-scored live
        m = matched(pd.DataFrame(score(dict(COND, seed=0)), columns=COLS),
                    registered, 'icl')
        m = m[m.item >= 0]
        print(f'base, plain, k = 8, draw 0, re-scored: '
              f'{(m.pred == m.pred_old).sum()} of {len(m)} predictions as '
              'registered')
        margin = lambda cols: np.diff(np.sort(m[cols].to_numpy(float), 1)[
            :, -2:], axis=1)[:, 0]  # top label minus the runner-up
        lp = ['lp0', 'lp1', 'lp2', 'lp3']
        for i, a, b, same in zip(m.item, margin(lp),
                                 margin([c + '_old' for c in lp]),
                                 m.pred == m.pred_old):
            if not same:
                print(f'  item {i}: margin {a:.3f} live, {b:.3f} registered')
del inst  # not needed below
gc.collect()
torch.cuda.empty_cache()
```

<!-- Placeholder (prose, ### Number of Demonstrations): reading one condition back from the grid. -->

```{.python .input #prompting-grid-select}
%%tab pytorch
def grid_values(cols, items=ITEMS, **cond):
    """Columns cols of one condition, (draws, items, cols); a condition
    without demonstrations has one draw, repeated for every draw."""
    c = {**COND, 'exp': 'icl', **cond}
    keep = (grid.perm != '') == c.pop('derange')
    for key, v in c.items():
        keep &= grid[key] == v
    r = grid[keep].set_index(['seed', 'item'])
    seeds = sorted(r.index.unique('seed'))
    v = r.loc[pd.MultiIndex.from_product([seeds, list(items)]), cols]
    v = v.to_numpy(float).reshape(len(seeds), len(items), len(cols))
    return v.repeat(len(DRAWS), 0) if seeds == [-1] else v

def pred(**cond):  # predicted label indices, (draws, items)
    return grid_values(['pred'], **cond)[..., 0].astype(int)

def correct(**cond):  # 1.0 where the prediction is the gold label
    return (pred(**cond) == GOLD).astype(float)
```

<!-- Placeholder (prose, ### Number of Demonstrations): a provisional paired bootstrap over items (d2l.paired_bootstrap follows in sec_lm_eval_suites), and a permutation null for order. -->

```{.python .input #prompting-paired-ci}
%%tab pytorch
def paired_ci(a, b=0, n_boot=2000, seed=0):
    """Mean of a - b over items, with its 95% bootstrap interval."""
    d = np.asarray(a, dtype=float) - b
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi

def order_null(correct, n_perm=2000, seed=0):
    """SD of accuracy over orders (correct: draws x orders x items), mean
    over draws, and its p-value when each item's order labels are
    permuted within a draw."""
    stat = lambda c: c.mean(-1).std(-1, ddof=1).mean()
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permuted(correct, axis=1))
                     for _ in range(n_perm)])
    return stat(correct), (1 + (null >= stat(correct)).sum()) / (1 + n_perm)

def compare(a, b):
    """a - b for (draws, items) arrays: the difference with its paired
    interval over items, and in how many draws it has the same sign
    ('n/a' when neither array varies over draws: no demonstrations)."""
    diff, lo, hi = paired_ci(a.mean(0), b.mean(0))
    same = (np.sign(a.mean(1) - b.mean(1)) == np.sign(diff)).sum()
    fixed = all((v == v[0]).all() for v in (a, b))
    return {'diff': diff, 'lo': lo, 'hi': hi,
            'draws': 'n/a' if fixed else f'{same}/{len(a)}'}

def show(r):  # a difference with its interval, as one string
    s = f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
    return s if r['draws'] == 'n/a' else f"{s} in {r['draws']} draws"
```

<!-- Placeholder (prose, ### Number of Demonstrations): accuracy against the number of demonstrations, three curves, each against k = 0 and, beyond k = 8, against k = 8; then free generation. -->

```{.python .input #prompting-k-table}
%%tab pytorch
curves = {'base, plain': dict(model='base'),
          'instruct, chat': dict(model='instruct', layout='chat'),
          'instruct, plain': dict(model='instruct')}
rows = []
for name, c in curves.items():
    zero, eight = correct(**dict(c, k=0)), correct(**dict(c, k=8))
    for k in K_BASE if name == 'base, plain' else K_INST:
        acc = correct(**dict(c, k=k))
        rows.append({'curve': name, 'k': k, 'accuracy': acc.mean(),
                     **(compare(acc, zero) if k else {}),
                     **({'vs k = 8': show(compare(acc, eight))} if k > 8
                        else {})})
print(pd.DataFrame(rows).round(3).fillna('').to_string(index=False))
gain = lambda m: correct(model=m, k=4) - correct(model=m, k=0)
print('gain from k = 0 to 4, plain, instruct minus base:',
      show(compare(gain('instruct'), gain('base'))))
gen = {'question': pred(exp='gen', layout='ask', k=0),
       'plain, k = 0': pred(exp='gen', k=0),
       'plain, k = 4': pred(exp='gen', k=4)}
rows = []
for name, p in gen.items():
    row = {'prompt': name, 'valid label': (p >= 0).mean(),
           'exact match': (p == GOLD).mean()}
    for what, f in (('valid', lambda p: (p >= 0) * 1.0),
                    ('exact', lambda p: (p == GOLD) * 1.0)):
        if name != 'question':
            row[f'{what} vs question'] = show(compare(f(p),
                                                  f(gen['question'])))
    rows.append(row)
print(pd.DataFrame(rows).round(3).fillna('').to_string(index=False))
```

<!-- Placeholder (prose, ### Number of Demonstrations): the template has to match the checkpoint (the table below). At k = 8 the chat layout carries SYSTEM, which names the labels, and the plain layout does not, so the chat-versus-plain rows measure template plus instruction; the k = 0 rows with and without the instruction separate the two. -->

```{.python .input #prompting-format-table}
%%tab pytorch
rows = []
for model in MODELS:
    pairs = {'chat, k = 8': (dict(layout='chat'), {}),
             'plain + instruction, k = 0': (dict(k=0, instr=1), dict(k=0))}
    if model == 'instruct':
        pairs['chat, k = 0'] = (dict(layout='chat', k=0), dict(k=0))
    for name, (c, ref) in pairs.items():
        a, b = correct(model=model, **c), correct(model=model, **ref)
        rows.append({'model': model, 'prompt': name, 'accuracy': a.mean(),
                     'plain, same k': b.mean(), **compare(a, b)})
pd.DataFrame(rows).round(3)
```

<!-- Placeholder (prose, ### Order): accuracy over four orders of the same demonstrations, the flip rate, and the permutation null. -->

```{.python .input #prompting-order}
%%tab pytorch
ORDER_ACC = {}
rng = np.random.default_rng(0)
boot = rng.integers(0, len(DRAWS), (2000, len(DRAWS)))  # draws resampled
for model in MODELS:
    C = np.stack([correct(model=model, order=o) for o in range(4)], 1)
    P = np.stack([pred(model=model, order=o) for o in range(4)], 1)
    acc = ORDER_ACC[model] = C.mean(-1)  # (draws, orders)
    table = pd.DataFrame(acc, columns=[f'order {o}' for o in range(4)])
    table['SD over orders'] = acc.std(1, ddof=1)
    table['flip rate'] = np.mean([(P[:, a] != P[:, b]).mean(-1) for a, b in
                                  itertools.combinations(range(4), 2)], 0)
    table['p (null)'] = [order_null(C[s:s + 1])[1] for s in range(len(C))]
    sd, p = order_null(C)  # pooled over draws
    table.loc['pooled'] = [*acc.mean(0), sd, table['flip rate'].mean(), p]
    print(f'{model}:\n{table.round(3).to_string()}')
    order_sd, draw_sd = acc.std(1, ddof=1), acc[:, 0]
    for name, v, b in (('orders', order_sd.mean(), order_sd[boot].mean(1)),
                       ('draws', draw_sd.std(ddof=1),
                        draw_sd[boot].std(1, ddof=1))):
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f'  SD over {name}: {v:.3f} [{lo:.3f}, {hi:.3f}]')
```

<!-- Placeholder (prose, ### Order): the plot below; caption sentence from spec fig_lm_icl_order. -->

```{.python .input #prompting-order-plot}
%%tab pytorch
x = [i + np.repeat(np.linspace(-0.3, 0.3, len(DRAWS)), 4) for i in (0, 1)]
d2l.plot(x, [ORDER_ACC[m].ravel() for m in MODELS], 'draws of each model',
         'accuracy at k = 8', legend=MODELS, fmts=('o', 'o'), figsize=(4, 3))
ax = d2l.plt.gca()
ax.set_xticks([0, 1], MODELS)
for i, m in enumerate(MODELS):  # draw means, then the model's mean
    ax.hlines(ORDER_ACC[m].mean(1), x[i][::4] - 0.05, x[i][::4] + 0.05,
              color=f'C{i}')
    ax.axhline(ORDER_ACC[m].mean(), color=f'C{i}', linestyle='--', lw=0.8)
```

<!-- Placeholder (prose, ### Label Correctness and Label Words): noisy and consistently relabeled demonstrations, natural and abstract words. -->

```{.python .input #prompting-labels}
%%tab pytorch
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SIGMA_Y = np.stack([SIGMA[s][GOLD] for s in DRAWS])  # demonstrated labels
LABEL_ACC, rows, effects = {}, [], []
for model in MODELS:
    for words in WORDS:
        c = dict(model=model, words=words)
        sigma = (pred(**c, derange=True) == SIGMA_Y) * 1.0
        LABEL_ACC[model, words] = [correct(**c, rho=r).mean() for r in RHOS]
        LABEL_ACC[model, words, 'sigma'] = sigma.mean()
        rows.append({'model': model, 'words': words,
                     **dict(zip(map(str, RHOS), LABEL_ACC[model, words])),
                     'deranged vs y': correct(**c, derange=True).mean(),
                     'deranged vs sigma(y)': sigma.mean()})
        if words == 'natural':  # accuracies, each with its interval
            for name, v in (('y', correct(**c, derange=True)),
                            ('sigma(y)', sigma)):
                d, lo, hi = paired_ci(v.mean(0))
                effects.append({'model': model, 'effect': 'natural, '
                                f'deranged: accuracy against {name}',
                                'diff': d, 'lo': lo, 'hi': hi})
        else:  # the control: renaming abstract classes changes nothing
            effects.append({'model': model, 'effect': 'abstract: sigma(y) '
                            'deranged minus rho = 0', **compare(
                                sigma, correct(**c, rho=0.0))})
    drop = lambda w: (correct(model=model, words=w, rho=0.0) -
                      correct(model=model, words=w, rho=1.0))
    effects += [
        {'model': model, 'effect': 'drop rho 0 to 1, abstract minus natural',
         **compare(drop('abstract'), drop('natural'))},
        {'model': model, 'effect': 'natural: rho = 0.75 minus k = 0',
         **compare(correct(model=model, rho=0.75),
                   correct(model=model, k=0))}]
print(pd.DataFrame(rows).round(3).to_string(index=False))
pd.DataFrame(effects).round(3).fillna('')
```

<!-- Placeholder (prose, ### Label Correctness and Label Words): the plot below; caption sentence from spec fig_lm_icl_labels. -->

```{.python .input #prompting-labels-plot}
%%tab pytorch
keys = [(m, w) for m in MODELS for w in WORDS]
d2l.plot(RHOS, [LABEL_ACC[key] for key in keys], 'fraction of wrong labels',
         'accuracy', legend=[f'{m}, {w}' for m, w in keys], ylim=(0, 1),
         fmts=('C0-o', 'C0--s', 'C1-o', 'C1--s'), figsize=(4.5, 3))
ax = d2l.plt.gca()
ax.axhline(0.25, color='gray', linestyle=':')  # chance
for i, (m, w) in enumerate(keys):  # derangement, scored against sigma(y)
    ax.plot([1.04], [LABEL_ACC[m, w, 'sigma']], 'os'[i % 2], mfc='none',
            color=f'C{i // 2}', markersize=8)
```

<!-- Placeholder (stage 2): contextual calibration divides out the prompt's label prior. -->

```{.python .input #prompting-calibrate}
%%tab pytorch
#@save
def calibrate_labels(logprobs, content_free_logprobs):
    """Contextual calibration: subtract the mean log-probabilities (M, L)
    of content-free inputs from logprobs (N, L) and renormalize over the
    L labels; one content-free input may be given as an (L,) row. The
    mean is taken in log space; Zhao et al. (2021) average the
    probabilities instead."""
    logprobs = torch.as_tensor(logprobs)
    cf = torch.as_tensor(content_free_logprobs)
    if cf.ndim not in (1, 2) or cf.shape[-1] != logprobs.shape[-1]:
        raise ValueError(f'content-free log-probabilities {tuple(cf.shape)}'
                         f' are not (M, {logprobs.shape[-1]}) or '
                         f'({logprobs.shape[-1]},)')
    return torch.log_softmax(logprobs - cf.reshape(-1, cf.shape[-1]).mean(0),
                             dim=-1)
```

<!-- Placeholder (prose, ### Contextual Calibration): accuracy and the share of the most-predicted label, before and after calibration, each change with its paired interval. -->

```{.python .input #prompting-calibrate-grid}
%%tab pytorch
LP, rows = ['lp0', 'lp1', 'lp2', 'lp3'], []
n = len(ITEMS)
boot = np.random.default_rng(0).integers(0, n, (2000, n))  # items resampled
share = lambda p: (p[..., None] == np.arange(4)).mean(-2).max(-1)  # per draw
for model in MODELS:
    for k in (0, 1, 4):
        v = torch.tensor(grid_values(LP, IDS, model=model, k=k))
        before = v[:, :n].argmax(-1).numpy()  # (draws, items)
        after = torch.stack([d2l.calibrate_labels(d[:n], d[n:])
                             for d in v]).argmax(-1).numpy()
        change = share(after) - share(before)  # (draws,)
        lo, hi = np.percentile((share(after[:, boot]) -
                                share(before[:, boot])).mean(0), [2.5, 97.5])
        same = (np.sign(change) == np.sign(change.mean())).sum()
        draws = 'n/a' if k == 0 else f'{same}/{len(change)}'
        rows.append({'model': model, 'k': k,
                     'accuracy before': (before == GOLD).mean(),
                     'accuracy after': (after == GOLD).mean(),
                     **compare((after == GOLD) * 1.0, (before == GOLD) * 1.0),
                     'top share before': share(before).mean(),
                     'top share after': share(after).mean(),
                     'top share change': show({
                         'diff': change.mean(), 'lo': lo, 'hi': hi,
                         'draws': draws})})
pd.DataFrame(rows).round(3)
```

<!-- Placeholder (prose, ### Copying in Context): a random token sequence repeated once; the loss on each copy against the uniform bound. -->

```{.python .input #prompting-induction}
%%tab pytorch
V = tok.vocab_size  # ordinary tokens; the ids above are special or padding
assert V == min(tok.added_tokens_decoder)
seq = torch.randint(0, V, (32, 64), generator=torch.Generator().manual_seed(0))
X = torch.cat([seq, seq], 1)[:8 if SMALL else 32]
losses = []
for chunk in X.split(8):  # the full logits of 8 sequences at a time
    chunk = chunk.to(base.device)
    with torch.inference_mode():
        logits = base(input_ids=chunk, use_cache=False).logits[:, :-1]
        losses.append(F.cross_entropy(logits.float().transpose(1, 2),
                                      chunk[:, 1:], reduction='none').cpu())
loss = torch.cat(losses)  # loss[:, j]: predicting token j + 1
first, second = loss[:, :63].mean(1), loss[:, 64:].mean(1)  # without 64
print(f'{len(X)} sequences of 2 x 64 tokens; bound ln {V:,} = '
      f'{math.log(V):.2f} nats')
for name, v in (('first copy', first), ('second copy', second)):
    print(f'{name}: {v.mean():.2f} nats per token (SD over sequences '
          f'{v.std():.2f})')
```

<!-- Placeholder (prose, ### A Probe on Frozen Features): the probe, a logistic regression on the base model's mean-pooled features of the headline alone. -->

```{.python .input #prompting-probe-def}
%%tab pytorch
def features(model, tokenizer, titles, batch_size=64):
    """Mean-pooled last hidden states of the titles alone, (n, d) fp32."""
    out = []
    for i in range(0, len(titles), batch_size):
        enc = tokenizer(list(titles[i:i + batch_size]), padding=True,
                        return_tensors='pt').to(model.device)
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            h = model.model(**enc, position_ids=pos,
                            use_cache=False).last_hidden_state.float()
        m = enc.attention_mask.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu())
    return torch.cat(out).numpy()

def probe_accuracy(X_train, y_train, X_test, y_test):
    """Accuracy and log-probabilities of a logistic-regression probe on
    standardized features (L2, C = 1) fit to the labelled examples."""
    scale = StandardScaler().fit(X_train)
    probe = LogisticRegression(C=1.0, max_iter=2000).fit(
        scale.transform(X_train), y_train)
    logp = probe.predict_log_proba(scale.transform(X_test))
    assert logp.shape[1] == 4, 'every class needs a labelled example'
    return (logp.argmax(1) == y_test).mean(), logp
```

<!-- Placeholder (prose, ### A Probe on Frozen Features): one fit on the 16 demonstrations of draw 0, and (D2L_RERUN=1 only) every draw and size. -->

```{.python .input #prompting-probe}
%%tab pytorch
K_PROBE = [4, 8, 16, 32, 64, 256]
if RERUN:  # every draw and size; completes the regenerated grid
    test_x, rows = features(base, tok, TEST_TITLES[ITEMS]), []
    for s in DRAWS:
        train_x = features(base, tok, TRAIN_TITLES[DRAW[s]])
        for k in K_PROBE:
            _, logp = probe_accuracy(train_x[:k], TRAIN_Y[DRAW[s][:k]],
                                     test_x, GOLD)
            cond = dict(COND, layout='none', k=k, seed=s)
            rows += [head(cond, 'probe') + [i, *r, int(r.argmax()), 1]
                     for i, r in zip(ITEMS, logp)]
    pd.concat([grid, pd.DataFrame(rows, columns=COLS)]).to_csv(
        RERUN_PATH, index=False, float_format='%.6g')
    grid = read_grid(RERUN_PATH)
    with open(RERUN_PATH, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    print(f'{RERUN_PATH}: {len(grid):,} rows, sha1 {sha1}')
    if registered is not None:
        m = matched(grid, registered, 'probe')
        print(f'probe: {(m.pred == m.pred_old).sum():,} of {len(m):,} '
              'predictions as registered')
elif not SMALL:  # one fit, live: the 16 demonstrations of draw 0
    train_x = features(base, tok, TRAIN_TITLES[DRAW[0]])  # batched as above
    acc, logp = probe_accuracy(train_x[:16], TRAIN_Y[DRAW[0][:16]],
                               features(base, tok, TEST_TITLES[ITEMS]), GOLD)
    same = (logp.argmax(1) == pred(exp='probe', layout='none', k=16)[0]).sum()
    print(f'probe, draw 0, 16 examples: accuracy {acc:.3f}; {same} of '
          f'{len(ITEMS)} predictions as registered')
else:
    print('probe, draw 0, 16 examples: accuracy '
          f'{correct(exp="probe", layout="none", k=16)[0].mean():.3f} '
          '(the registered 1.7B run; without a GPU the probe is not fit)')
```

<!-- Placeholder (prose, ### A Probe on Frozen Features): the plot below; caption sentence from spec fig_lm_icl_k; ICL minus probe at equal labelled data. -->

```{.python .input #prompting-icl-vs-probe-plot}
%%tab pytorch
series = {'base, ICL (plain)': (K_BASE, dict(model='base')),
          'instruct, ICL (chat)': (K_INST, dict(model='instruct',
                                                layout='chat')),
          'instruct, ICL (plain)': (K_INST, dict(model='instruct')),
          'base, probe': (K_PROBE, dict(exp='probe', layout='none'))}
acc = [np.array([correct(**c, k=k).mean(1) for k in ks])  # (k, draws)
       for ks, c in series.values()]
at = lambda ks: np.log2(np.maximum(ks, 0.5))  # a log2 axis, 0 at the left
d2l.plot([at(ks) for ks, _ in series.values()] * 3,
         [a.mean(1) for a in acc] + [a.min(1) for a in acc] +
         [a.max(1) for a in acc], 'labelled examples', 'accuracy',
         legend=list(series), fmts=[f'C{i}{f}' for f in ('-o', ':', ':')
                                    for i in range(4)], figsize=(5, 3.2))
ticks = sorted(set(K_BASE + K_PROBE))
d2l.plt.gca().set_xticks(at(ticks), [str(k) for k in ticks])
rows = [{'examples': k, **compare(correct(k=k), correct(
    exp='probe', layout='none', k=k))} for k in K_PROBE[:-1]]
print('ICL (base, plain) minus probe, at equal labelled data')
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f'probe at 256: '
      f'{correct(exp="probe", layout="none", k=256).mean():.3f}')
```

<!-- Placeholder (prose, ### A Probe on Frozen Features): the cost per query, in prompt tokens and key-value cache. -->

```{.python .input #prompting-cost}
%%tab pytorch
c = base.config  # K and V of every layer, per token, in bf16 as served
kv_bytes = (c.num_hidden_layers * 2 * c.num_key_value_heads * c.head_dim *
            torch.bfloat16.itemsize)
probe = np.mean(n_tokens(tok, list(TEST_TITLES[ITEMS])))  # the headline
rows = []
for k in K_BASE:
    cond = dict(COND, k=k, seed=-1 if k == 0 else 0)
    t = np.mean(n_tokens(tok, build_prompts(cond, TEST_TITLES[ITEMS])))
    rows.append({'k': k, 'tokens per query': t, 'vs probe': t / probe,
                 'KV cache (MB)': t * kv_bytes / 1e6})
rows = pd.DataFrame(rows)
rows.insert(2, 'vs k = 0', rows['tokens per query'] / rows.iloc[0, 1])
print(f'probe: {probe:.1f} tokens per query; KV cache {kv_bytes:,} bytes '
      'per token in bf16')
rows.round(1)
```
