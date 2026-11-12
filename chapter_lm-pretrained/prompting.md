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
import html
import json
import os
import re
import string
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
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
        head = _lm_bos_ids(tokenizer)  # and never an EOS
        ids = [head + r for r in ids]
    if not all(ids):
        raise ValueError('an empty prompt has no position to predict from')
    hf = sys.modules.get('transformers')
    hf = hf is not None and isinstance(_lm_unwrap(model), hf.PreTrainedModel)
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
                    _lm_unwrap(model), d2l.Qwen3GPT) else model(X))[:, -1]
            lab = torch.tensor([labels[i] for i in rows], device=z.device)
            lp = z.gather(-1, lab).float().log_softmax(-1).cpu()
        out[rows] = lp  # out is an ordinary tensor the caller may modify
    return out
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
