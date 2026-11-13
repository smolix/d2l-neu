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
import itertools
import json
import os
import re
import string
import textwrap
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
INSTRUCTION = 'Name the topic of each news headline: {}, {}, {} or {}.'
SYSTEM = INSTRUCTION.format(*NATURAL)  # World, Sports, Business or Tech
SYSTEMS = [None, SYSTEM, INSTRUCTION.format(*NATURAL[::-1])]  # by 'instr'
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

def print_table(rows, digits=3):  # a list of dicts, or a DataFrame
    print(pd.DataFrame(rows).to_string(
        index=False, na_rep='', float_format=f'{{:.{digits}f}}'.format))
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
    print('No GPU: the 0.6B pair in fp32 stands in for the 1.7B models,\n'
          'so the outputs below differ from those quoted in the text.')
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
    if i == start and line:  # one paragraph per message
        print(textwrap.fill(' '.join(line), 79))
        line = []
    t = tok_i.decode([i]).replace('\n', '\\n')
    line.append(f'[{i}:{t}]' if i in special else f'{i}:{t}')
print(textwrap.fill(' '.join(line), 79))
```

<!-- Placeholder (prose, ### Chat Templates): the generation prefix, and whether each template opens a think block by default (the table below). The column is a property of the template file: Qwen3-1.7B-Base ships the post-trained template, but the base model was never trained to think. -->

```{.python .input #prompting-template-tails}
%%tab pytorch
PREFIX = '<|im_start|>assistant\n'  # every template's tail starts so
rows = []
for name in ('Qwen/Qwen3-1.7B', 'Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3.5-2B',
             'Qwen/Qwen3.5-4B'):
    t = AutoTokenizer.from_pretrained(name, revision=d2l.HF_REVISIONS[name])
    render = lambda **kw: t.apply_chat_template(
        [{'role': 'user', 'content': QUERY}], tokenize=False, **kw)
    body = render(add_generation_prompt=False)
    unset, off = (render(add_generation_prompt=True, **kw)[len(body):]
                  for kw in ({}, {'enable_thinking': False}))
    assert unset.startswith(PREFIX) and off.startswith(PREFIX)
    rows.append({'model': name, 'unset': repr(unset[len(PREFIX):]),
                 'enable_thinking=False': repr(off[len(PREFIX):]),
                 'thinks by default': 'no' if '</think>' in unset else 'yes'})
print(f'tail after the user turn: {PREFIX!r}, then')
print_table(rows)
```

<!-- Placeholder (stage 2): read a label from the next token, over the label words only. -->

```{.python .input #prompting-label-logprobs}
%%tab pytorch
#@save
def label_logprobs(model, tokenizer, prompts, label_words, *, batch_size=8,
                   chat=False, enable_thinking=False, sep=None):
    """Log-probabilities (len(prompts), len(label_words)), float32 on the
    CPU, of each label word's first token as the next token, normalized
    over the label tokens only. A word follows its prompt after sep, by
    default a space, but none after whitespace or next to Chinese text.
    With chat=True a prompt is a message list (a string is one user
    message), rendered with the generation prefix. The label logits are
    computed in float32 from the last hidden state and the label words'
    rows of the output layer (bf16 logits of size 20 to 40 are 1/8 to 1/4
    apart and tie)."""
    if chat:
        opts = {} if enable_thinking is None else {
            'enable_thinking': enable_thinking}
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': m}] if isinstance(m, str) else m,
            tokenize=False, add_generation_prompt=True, **opts)
                   for m in prompts]
    ids, labels = d2l.label_token_ids(tokenizer, prompts, label_words, sep)
    if not chat:  # as d2l.generate: a plain prompt gets the BOS, if any
        ids = [d2l._lm_bos_ids(tokenizer) + r for r in ids]
    if not all(ids):
        raise ValueError('an empty prompt has no position to predict from')
    W = d2l._label_output_layer(model)
    if W is None:  # a d2l GPT, or a soft-capped output layer
        return d2l._label_logprobs_logits(model, tokenizer, ids, labels,
                                          batch_size)
    decoder = d2l._lm_unwrap(model).get_decoder()
    pad = getattr(tokenizer, 'pad_token_id', None) or 0
    device = next(model.parameters()).device
    out = torch.empty(len(ids), len(label_words))
    for j in range(0, len(ids), batch_size):  # left-padded, in input order
        rows, lab = ids[j:j + batch_size], labels[j:j + batch_size]
        T = max(len(r) for r in rows)
        X = torch.tensor([[pad] * (T - len(r)) + r for r in rows],
                         device=device)
        mask = torch.tensor([[0] * (T - len(r)) + [1] * len(r)
                             for r in rows], device=device)
        pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
        with torch.inference_mode():
            h = decoder(input_ids=X, attention_mask=mask, position_ids=pos,
                        use_cache=False).last_hidden_state[:, -1]
            lab = torch.tensor(lab, device=device)  # (batch, labels)
            z = torch.einsum('bd,bld->bl', h.float(), W.weight[lab].float())
            if W.bias is not None:
                z = z + W.bias[lab].float()
            lp = z.log_softmax(-1).cpu()
        out[j:j + batch_size] = lp  # out is an ordinary tensor
    return out
```

<!-- Placeholder (prose, ### Reading a Label from the Next Token): the reader against the full softmax, and batched against one prompt at a time. -->

```{.python .input #prompting-reader-check}
%%tab pytorch
def renorm(z, ids):  # the full softmax, renormalized over the labels
    lp = z.float().log_softmax(-1)[ids]
    return (lp - lp.logsumexp(-1)).cpu()

prompts = [d2l.fewshot_prompt(DEMOS, h, fmt=FMT) for h in HEADLINES]
_, label_ids = d2l.label_token_ids(tok, prompts, NATURAL)
one = d2l.label_logprobs(base, tok, prompts, NATURAL, batch_size=1)
W = base.lm_head.weight.float()  # the whole output layer, in fp32
for prompt, ids, lp in zip(prompts, label_ids, one):
    x = tok(prompt, return_tensors='pt').input_ids.to(base.device)
    with torch.inference_mode():  # every logit at the last position
        h = base.model(input_ids=x, use_cache=False).last_hidden_state[0, -1]
        ref = renorm(h.float() @ W.T, ids)
        ref16 = renorm(base.lm_head(h), ids)  # the model's own logits
    err, err16 = ((lp - r).abs().max().item() for r in (ref, ref16))
    print(f'reader vs full softmax: {err:.1e} nats (from its own logits: '
          f'{err16:.1e})')
    assert err <= 1e-4
del W
batched = d2l.label_logprobs(base, tok, prompts, NATURAL, batch_size=4)
screen = 1e-4 if SMALL else 0.25  # a mask or position bug costs whole nats
for e in (batched - one).abs().max(-1).values.tolist():
    print(f'batched vs one at a time: {e:.1e} nats (screen {screen:.1e})')
    assert e <= screen
assert torch.equal(batched.argmax(-1), one.argmax(-1))  # same predictions
```

<!-- Placeholder (prose, ### Reading a Label from the Next Token): label mass, the full-softmax probability of the label tokens, which the renormalized read-out hides. -->

```{.python .input #prompting-label-mass-fn}
%%tab pytorch
def label_mass(model, tokenizer, texts, ids):
    """Full-softmax probability of the label tokens ids[i] as the next
    token after texts[i], one text at a time."""
    mass = []
    for text, i in zip(texts, ids):
        x = tokenizer(text, return_tensors='pt').input_ids.to(model.device)
        with torch.inference_mode():
            z = model(input_ids=x, use_cache=False,
                      logits_to_keep=1).logits[0, -1]
        mass.append(z.float().softmax(-1)[i].sum().item())
    return np.array(mass)
```

<!-- Placeholder (prose, ### Reading a Label from the Next Token): one word, two token ids; a prompt ending in a space; a label of three tokens. -->

```{.python .input #prompting-label-tokens}
%%tab pytorch
_, (after_colon,) = d2l.label_token_ids(tok, ['Topic:'], NATURAL)
_, (after_newline,) = d2l.label_token_ids(tok_i, [chat_text], NATURAL)
d2l.label_token_ids(tok, ['Topic:'], ABSTRACT)  # raises unless 1 token
print(pd.DataFrame({
    'word': NATURAL,
    'after "Topic:"': [f'{i} {tok.decode([i])!r}' for i in after_colon],
    'after a newline': [f'{i} {tok.decode([i])!r}' for i in after_newline]
    }).to_string(index=False))
assert not set(after_colon) & set(after_newline)
spaced = [p + ' ' for p in prompts]  # the space typed into the prompt
mass = [label_mass(base, tok, texts, [ids] * len(texts)).mean()
        for texts, ids in ((prompts, after_colon), (spaced, after_newline))]
print(f'label mass after "Topic:": {mass[0]:.3f}; after "Topic: ", read '
      f'at the bare words: {mass[1]:.3f}')
for texts, words in ((spaced, NATURAL),
                     (prompts, ['World', 'Sports', 'Business', 'Sci/Tech'])):
    try:
        d2l.label_logprobs(base, tok, texts, words)
    except ValueError as e:
        print(textwrap.fill(f'ValueError: {e}', 79))
```

<!-- Placeholder (prose, ### Reading a Label from the Next Token): one zero-shot item, four pairings of checkpoint and layout, with the full-softmax probability of the four label tokens (label mass). The base model never saw the empty think block that enable_thinking=False appends, and after it the label mass is near zero, so its renormalized distribution means nothing. Without the think block (the template's default end) the label mass on this zero-shot item rises only from 0 to about 0.02 and the top label is still wrong: say only that. The evidence that the base model reads the chat layout without the think block is the 8-shot row "chat, no think block" of the format table below (about 0.77, as good as plain text), not this item. -->

```{.python .input #prompting-template-mismatch}
%%tab pytorch
chat = d2l.fewshot_prompt([], QUERY, 'chat_multi', system=SYSTEM)
plain = d2l.fewshot_prompt([], QUERY, fmt=FMT, system=SYSTEM)
render = lambda t, **kw: t.apply_chat_template(  # thinking off: empty block
    chat, tokenize=False, add_generation_prompt=True, **kw)
runs = {'instruct, chat template': (inst, tok_i, render(
            tok_i, enable_thinking=False)),
        'instruct, plain text': (inst, tok_i, plain),
        'base, chat template': (base, tok, render(tok, enable_thinking=False)),
        'base, chat template, no think block': (base, tok, render(tok))}
rows = {}
for name, (m, t, text) in runs.items():  # text: the rendered prompt
    _, ids = d2l.label_token_ids(t, [text], NATURAL)
    p = d2l.label_logprobs(m, t, [text], NATURAL)[0].exp()
    rows[name] = [*p.tolist(), label_mass(m, t, [text], ids)[0]]
table = pd.DataFrame(rows, index=NATURAL + ['label mass']).T
print(table.to_string(float_format='{:.3f}'.format))
```

<!-- Placeholder (prose, ### Reading a Label from the Next Token): the instruct model's own greedy replies to the four invented headlines in its chat layout, with SYSTEM and without an instruction line: the World answer is the model's, not the reader's, and without a line the model answers with no label word at all. -->

```{.python .input #prompting-instruct-replies}
%%tab pytorch
for system in (SYSTEM, None):
    chats = [d2l.fewshot_prompt([], h, 'chat_multi', system=system)
             for h in HEADLINES]
    replies = d2l.generate(inst, tok_i, chats, max_new_tokens=6, chat=True,
                           enable_thinking=False)
    print('SYSTEM line' if system else 'no instruction line')
    for h, r in zip(HEADLINES, replies):
        print(f'  {h:40} -> {r.text!r}')
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
d2l.DATA_HUB['lm-fewshot-grid-v2'] = (
    d2l.LM_DATA_URL + 'lm-fewshot-grid-v2.csv',
    '0c2a1c87635bbfcdca9cedb479723901b3d5cbd5')
```

<!-- Placeholder (prose, ### Number of Demonstrations): the task data, its cleanup rules (HTML and Windows-1252 escapes left in the titles; d2l.clean_ag_news, in the library, not shown), and the key that finds near-duplicates. -->

```{.python .input #prompting-data-clean}
%%tab pytorch
def read_jsonl(key):
    path = d2l.download(key, os.path.join('..', 'data', 'lm', key))
    with open(path, encoding='utf-8') as f:
        return pd.DataFrame([json.loads(line) for line in f])

def dedup_key(title):  # no source tag such as '(AP)', no punctuation
    title = re.sub(r'\s*\([^()]*\)\s*$', '', title).lower()
    title = title.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(title.split())
```

<!-- Placeholder (prose, ### Number of Demonstrations): the test items (the first 50 test rows of each class in file order, a fixed stratified subset, not a random sample) and the demonstration pool without test duplicates; counts only, no article text. -->

```{.python .input #prompting-data}
%%tab pytorch
N_PER_CLASS = 50
train, test = read_jsonl('ag-news-train'), read_jsonl('ag-news-test')
TRAIN_TITLES, changed_train = d2l.clean_ag_news(list(train.title))
TEST_TITLES, changed_test = d2l.clean_ag_news(list(test.title))
TRAIN_Y, TEST_Y = train.label.to_numpy() - 1, test.label.to_numpy() - 1
ITEMS = np.sort(test.groupby('label').head(N_PER_CLASS).index.to_numpy())
GOLD = TEST_Y[ITEMS]
assert np.bincount(GOLD).tolist() == [N_PER_CLASS] * 4
test_keys = {dedup_key(t) for t in TEST_TITLES}
in_pool = np.array([dedup_key(t) not in test_keys for t in TRAIN_TITLES])
POOL = [np.flatnonzero(in_pool & (TRAIN_Y == c)) for c in range(4)]
print(f'{len(ITEMS)} test items, the first {N_PER_CLASS} of each class in '
      'file order')
print(f'demonstration pool per class: {[len(p) for p in POOL]}')
print(f'{(~in_pool).sum()} train titles removed as duplicates of test '
      'titles; titles changed by each cleanup rule:')
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
INSTR = {0: 'none', 1: 'SYSTEM', 2: 'SYSTEM, labels reversed'}  # SYSTEMS
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
        for instr in (1, 2):  # an instruction names the labels
            add(model=model, k=0, instr=instr)
        for k in ks if model == 'instruct' else [8]:  # the chat layout
            add(model=model, layout='chat', k=k, instr=1)
        if model == 'instruct':  # no instruction, or its labels reversed
            for instr, k in itertools.product((0, 2), (0, 1, 4, 8)):
                add(model=model, layout='chat', k=k, instr=instr)
        if model == 'base':  # the template's default end: no think block
            add(model=model, layout='chat_default', instr=1)
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
    ds, system = demos(cond), SYSTEMS[cond['instr']]
    if cond['layout'].startswith('chat'):
        chats = [d2l.fewshot_prompt(ds, x, 'chat_multi', label_map=words,
                                    system=system) for x in titles]
        if cond['layout'] == 'chat':  # rendered by the reader, no thinking
            return chats
        return [tok.apply_chat_template(  # base only: the default end
            m, tokenize=False, add_generation_prompt=True) for m in chats]
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
assert [sum(c['model'] == m for c in GRID) for m in MODELS] == [118, 156]
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
GRID_KEY = 'lm-fewshot-grid-v2'
registered = read_grid(d2l.download(GRID_KEY,
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
    for exp in ('icl', 'gen'):
        m = matched(grid, registered, exp)
        print(f'{exp}: {(m.pred == m.pred_old).sum():,} of {len(m):,} '
              'predictions as registered')
else:
    grid = registered
    print(f'{GRID_KEY}: {len(grid):,} rows')
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

<!-- Placeholder (prose, ### Number of Demonstrations): accuracy against the number of demonstrations, three curves, each against k = 0 and, beyond k = 8, against k = 8; then free generation. The instruct chat curve carries SYSTEM; its zero-shot point, and so its gain from 0 to 4 demonstrations, depends on that line (prompting-instruction-lines), so the text quotes the gain as the range over the three lines and does not call the instruct model "at chance" zero-shot. -->

```{.python .input #prompting-k-table}
%%tab pytorch
curves = {'base, plain': dict(model='base'),
          'instruct, chat': dict(model='instruct', layout='chat', instr=1),
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
print_table(rows)
gain = lambda m: correct(model=m, k=4) - correct(model=m, k=0)
print('gain from k = 0 to 4, plain, instruct minus base:',
      show(compare(gain('instruct'), gain('base'))))
gen = {'question': pred(exp='gen', layout='ask', k=0),
       'plain, k = 0': pred(exp='gen', k=0),
       'plain, k = 4': pred(exp='gen', k=4)}
rows = []
for what, f in (('valid label', lambda p: (p >= 0) * 1.0),
                ('exact match', lambda p: (p == GOLD) * 1.0)):
    for name, p in gen.items():
        rows.append({'prompt': name, 'rate of': what, 'rate': f(p).mean(),
                     'vs question': '' if name == 'question' else show(
                         compare(f(p), f(gen['question'])))})
print_table(rows)
```

<!-- Placeholder (prose, ### Number of Demonstrations): the template has to match the checkpoint (the table below). At k = 8 the chat layout carries SYSTEM, which names the labels, and the plain layout does not, so the chat-versus-plain rows measure template plus instruction; the k = 0 rows with and without the instruction separate the two. The base row "chat, no think block" uses the template's default end (enable_thinking unset), which separates the template from the empty think block. The instruct rows at k = 0 hold for this SYSTEM line only: the next table shows that the line's wording and label order move them. -->

```{.python .input #prompting-format-table}
%%tab pytorch
rows = []
for model in MODELS:
    pairs = {'chat, k = 8': (dict(layout='chat', instr=1), {}),
             'plain + instruction, k = 0': (dict(k=0, instr=1), dict(k=0))}
    if model == 'base':  # the same template without the empty think block
        pairs['chat, no think block, k = 8'] = (
            dict(layout='chat_default', instr=1), {})
    if model == 'instruct':
        pairs['chat, k = 0'] = (dict(layout='chat', k=0, instr=1), dict(k=0))
    for name, (c, ref) in pairs.items():
        a, b = correct(model=model, **c), correct(model=model, **ref)
        rows.append({'model': model, 'prompt': name, 'accuracy': a.mean(),
                     'plain, same k': b.mean(), **compare(a, b)})
print_table(rows)
```

<!-- Placeholder (prose, ### Number of Demonstrations): the instruction line is part of the prompt too. The instruct model in its chat layout without an instruction, with SYSTEM, and with SYSTEM's label list reversed (k = 0, 1, 4, 8), then the plain layout at k = 0 with each line for both models: the zero-shot accuracy and the most-predicted label move with the line's wording and order, and the curves meet only by k = 8. -->

```{.python .input #prompting-instruction-lines}
%%tab pytorch
def top(p):  # the most-predicted label and its share, over draws and items
    counts = np.bincount(p.ravel(), minlength=4)
    return NATURAL[counts.argmax()], counts.max() / counts.sum()

rows = []
for instr, line in INSTR.items():
    c = dict(model='instruct', layout='chat', instr=instr)
    for k in (0, 1, 4, 8):
        label, top_share = top(pred(**c, k=k))
        rows.append({'instruction': line, 'k': k,
                     'accuracy': correct(**c, k=k).mean(), 'top label': label,
                     'top share': top_share, **(compare(
                         correct(**c, k=k), correct(**c, k=0)) if k else {})})
print('instruct, chat layout; each k against k = 0 with the same line')
print_table(rows)
rows = []  # each line against none
for model, (instr, line) in itertools.product(MODELS, INSTR.items()):
    label, top_share = top(pred(model=model, k=0, instr=instr))
    acc = correct(model=model, k=0, instr=instr)
    rows.append({'model': model, 'plain, k = 0, instruction': line,
                 'accuracy': acc.mean(), 'top label': label,
                 'top share': top_share, **(compare(acc, correct(
                     model=model, k=0)) if instr else {})})
print('\nplain layout, k = 0; each line against none')
print_table(pd.DataFrame(rows).drop(columns='draws'))
```

<!-- Placeholder (prose, ### Number of Demonstrations): how much probability the readings above rest on: the label mass over the 200 test items, without and with an instruction line, at k = 0 and with demonstrations (draw 0). -->

```{.python .input #prompting-label-mass}
%%tab pytorch
MASS = [('base', 'plain', 0, 0), ('base', 'plain', 0, 1),
        ('base', 'plain', 4, 0), ('instruct', 'plain', 0, 0),
        ('instruct', 'plain', 0, 1), ('instruct', 'chat', 0, 0),
        ('instruct', 'chat', 0, 1), ('instruct', 'chat', 0, 2),
        ('instruct', 'chat', 4, 0), ('instruct', 'chat', 4, 1),
        ('instruct', 'chat', 8, 0)]
rows = []
for model, layout, k, instr in MASS:  # (model, layout, k, line)
    cond = dict(model=model, layout=layout, k=k, instr=instr,
                seed=0 if k else -1)  # k > 0: draw 0 only
    m, t = (base, tok) if model == 'base' else (inst, tok_i)
    texts = build_prompts(dict(COND, **cond), TEST_TITLES[ITEMS])
    if layout == 'chat':  # rendered as the reader renders it
        texts = [t.apply_chat_template(x, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False) for x in texts]
    _, ids = d2l.label_token_ids(t, texts, NATURAL)
    mass = label_mass(m, t, texts, ids)
    row = {'model': model, 'layout': layout, 'k': k,
           'instruction': INSTR[instr], 'mean mass': mass.mean(),
           'median mass': np.median(mass)}
    if not SMALL:  # the grid's accuracy for the same condition
        row['accuracy'] = correct(**cond).mean()
    rows.append(row)
print_table(rows)
del inst  # not needed below
gc.collect()
torch.cuda.empty_cache()
```

<!-- Placeholder (prose, ### Order): accuracy over four orders of the same demonstrations, the flip rate, and the permutation null. -->

```{.python .input #prompting-order}
%%tab pytorch
ORDER_ACC = {}
p_text = lambda p: '<0.001' if p < 0.001 else f'{p:.3f}'  # p >= 1/2001
for model in MODELS:
    C = np.stack([correct(model=model, order=o) for o in range(4)], 1)
    P = np.stack([pred(model=model, order=o) for o in range(4)], 1)
    acc = ORDER_ACC[model] = C.mean(-1)  # (draws, orders)
    table = pd.DataFrame(acc, columns=[f'order {o}' for o in range(4)])
    table['SD over orders'] = acc.std(1, ddof=1)
    table['flip rate'] = np.mean([(P[:, a] != P[:, b]).mean(-1) for a, b in
                                  itertools.combinations(range(4), 2)], 0)
    p_draws = [order_null(C[s:s + 1])[1] for s in range(len(C))]
    sd, p = order_null(C)  # pooled over draws
    table.loc['pooled'] = [*acc.mean(0), sd, table['flip rate'].mean()]
    table['p (null)'] = [p_text(v) for v in p_draws + [p]]
    print(f"{model}:\n{table.to_string(float_format='{:.3f}'.format)}")
    print(f'  SD over orders, mean over draws: {acc.std(1, ddof=1).mean():.3f}'
          f'; SD over draws, order 0: {acc[:, 0].std(ddof=1):.3f}')
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
    effects += [{'model': model, 'effect': f'drop rho 0 to 1, {w}',
                 **compare(drop(w), np.zeros_like(drop(w)))} for w in WORDS]
    effects += [
        {'model': model, 'effect': 'drop rho 0 to 1, abstract minus natural',
         **compare(drop('abstract'), drop('natural'))},
        {'model': model, 'effect': 'natural: rho = 0.75 minus k = 0',
         **compare(correct(model=model, rho=0.75),
                   correct(model=model, k=0))}]
print_table(rows)
print()
print_table(effects)
```

<!-- Placeholder (prose, ### Label Correctness and Label Words): the plot below; caption sentence from spec fig_lm_icl_labels. -->

```{.python .input #prompting-labels-plot}
%%tab pytorch
keys = [(m, w) for m in MODELS for w in WORDS]
names = [f'{m}, {w}' for m, w in keys]
d2l.plot(RHOS, [LABEL_ACC[key] for key in keys], 'fraction of wrong labels',
         'accuracy', legend=names, xlim=(-0.05, 1.12), ylim=(0, 1),
         fmts=('C0-o', 'C0--s', 'C1-o', 'C1--s'), figsize=(4.5, 3))
ax = d2l.plt.gca()
ax.legend(names, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.axhline(0.25, color='gray', linestyle=':')  # chance
for i, (m, w) in enumerate(keys):  # derangement, scored against sigma(y)
    ax.plot([1.06], [LABEL_ACC[m, w, 'sigma']], 'os'[i % 2], mfc='none',
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
LP, n = ['lp0', 'lp1', 'lp2', 'lp3'], len(ITEMS)
item_boot = np.random.default_rng(0).integers(0, n, (2000, n))
draw_top_share = lambda p: (p[..., None] == np.arange(4)).mean(-2).max(-1)
CAL, acc_rows, share_rows = {}, [], []  # CAL: (before, after) predictions
for model, k in itertools.product(MODELS, (0, 1, 4)):
    v = torch.tensor(grid_values(LP, IDS, model=model, k=k))
    before = v[:, :n].argmax(-1).numpy()  # (draws, items)
    after = torch.stack([d2l.calibrate_labels(d[:n], d[n:])
                         for d in v]).argmax(-1).numpy()
    CAL[model, k] = before, after
    acc_rows.append({'model': model, 'k': k,
                     'before': (before == GOLD).mean(),
                     'after': (after == GOLD).mean(),
                     **compare((after == GOLD) * 1.0, (before == GOLD) * 1.0)})
    change = draw_top_share(after) - draw_top_share(before)  # per draw
    lo, hi = np.percentile((draw_top_share(after[:, item_boot]) -
                            draw_top_share(before[:, item_boot])).mean(0),
                           [2.5, 97.5])
    same = (np.sign(change) == np.sign(change.mean())).sum()
    share_rows.append({'model': model, 'k': k,
                       'before': draw_top_share(before).mean(),
                       'after': draw_top_share(after).mean(),
                       'change': show({'diff': change.mean(), 'lo': lo,
                                       'hi': hi, 'draws': 'n/a' if k == 0
                                       else f'{same}/{len(change)}'})})
print('accuracy, before and after calibration')
print_table(acc_rows)
print('\nshare of the most-predicted label, before and after calibration')
print_table(share_rows)
```

<!-- Placeholder (prose, ### Contextual Calibration): at k = 1, draw by draw: the one demonstrated label, the content-free prior's probability of it, and what calibration does to accuracy and to the share of that label. -->

```{.python .input #prompting-calibrate-draws}
%%tab pytorch
rows = []
for model in MODELS:
    v = torch.tensor(grid_values(LP, IDS, model=model, k=1))
    before, after = CAL[model, 1]
    for s in DRAWS:
        y = TRAIN_Y[DRAW[s][0]]  # the one demonstrated label
        prior = v[s, n:].mean(0).softmax(-1)  # as calibrate_labels uses it
        rows.append({'model': model, 'draw': s, 'demo': NATURAL[y],
                     'prior of demo': prior[y].item(),
                     'acc. before': (before[s] == GOLD).mean(),
                     'acc. after': (after[s] == GOLD).mean(),
                     'share before': (before[s] == y).mean(),
                     'share after': (after[s] == y).mean()})
print('k = 1, by draw; share: of the predictions, the demonstrated label')
print_table(rows)
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
          f'{correct(exp="probe", layout="none", k=16)[0].mean():.3f}\n'
          '(the registered 1.7B run; without a GPU the probe is not fit)')
```

<!-- Placeholder (prose, ### A Probe on Frozen Features): the plot below; caption sentence from spec fig_lm_icl_k (the bands are the range over draws for base ICL and the probe; the instruct-chat k = 0 point depends on the instruction line); ICL minus probe at equal labelled data. -->

```{.python .input #prompting-icl-vs-probe-plot}
%%tab pytorch
series = {'base, ICL (plain)': (K_BASE, dict(model='base')),
          'instruct, ICL (chat)': (K_INST, dict(model='instruct',
                                                layout='chat', instr=1)),
          'instruct, ICL (plain)': (K_INST, dict(model='instruct')),
          'base, probe': (K_PROBE, dict(exp='probe', layout='none'))}
acc = [np.array([correct(**c, k=k).mean(1) for k in ks])  # (k, draws)
       for ks, c in series.values()]
at = lambda ks: np.log2(np.maximum(ks, 0.5))  # a log2 axis, 0 at the left
d2l.plot([at(ks) for ks, _ in series.values()], [a.mean(1) for a in acc],
         'labelled examples', 'accuracy', legend=list(series),
         ylim=(0.25, 0.85), fmts=[f'C{i}-o' for i in range(4)],
         figsize=(5, 3.2))
ticks = sorted(set(K_BASE + K_PROBE))
ax = d2l.plt.gca()
for i in (0, 3):  # the range over draws: base ICL and the probe
    ks = list(series.values())[i][0]
    ax.fill_between(at(ks), acc[i].min(1), acc[i].max(1), color=f'C{i}',
                    alpha=0.15, lw=0)
ax.set_xticks(at(ticks), [str(k) for k in ticks])
ax.legend(list(series), loc='lower center', bbox_to_anchor=(0.5, 1.0),
          ncol=2)  # above the axes, clear of the curves
rows = [{'examples': k, **compare(correct(k=k), correct(
    exp='probe', layout='none', k=k))} for k in K_PROBE[:-1]]
print('ICL (base, plain) minus probe, at equal labelled data')
print_table(rows)
print(f'probe at 256: '
      f'{correct(exp="probe", layout="none", k=256).mean():.3f}')
```

<!-- Placeholder (prose, ### A Probe on Frozen Features): the cost per query, in prompt tokens and key-value cache, without prefix caching. The demonstrations are a prefix shared by every query, so a server that caches it computes it once and a query then costs about its own tokens; the table's ratios (about 30 at 16 shots) are the uncached cost. -->

```{.python .input #prompting-cost}
%%tab pytorch
cfg = base.config  # K and V of every layer, per token, in bf16 as served
kv_bytes = (cfg.num_hidden_layers * 2 * cfg.num_key_value_heads *
            cfg.head_dim * torch.bfloat16.itemsize)
probe_tokens = np.mean(n_tokens(tok, list(TEST_TITLES[ITEMS])))  # headline
rows = []
for k in K_BASE:
    cond = dict(COND, k=k, seed=-1 if k == 0 else 0)
    t = np.mean(n_tokens(tok, build_prompts(cond, TEST_TITLES[ITEMS])))
    rows.append({'k': k, 'tokens per query': t, 'vs probe': t / probe_tokens,
                 'KV cache (MB)': t * kv_bytes / 1e6})
table = pd.DataFrame(rows)
table.insert(2, 'vs k = 0', table['tokens per query'] / table.iloc[0, 1])
print(f'probe: {probe_tokens:.1f} tokens per query; KV cache {kv_bytes:,} '
      'bytes per token in bf16')
print_table(table, digits=1)
```
