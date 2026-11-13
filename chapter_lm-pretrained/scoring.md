# Scoring Language Models
:label:`sec_lm_scoring`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "1.4 Scoring Language Models · T1").
Blueprint: docs/llm/blueprints/ch01_pretrained.md section 1.4.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A score is a convention, and two conventions can rank two models differently.
**Opening:** One multiple-choice item scored by the log-probability of each answer's text, by the log-probability of its letter, and by a generated answer gives three verdicts for the same model. The section makes each convention explicit, builds the scorers every later chapter imports, and closes on how many items it takes before a difference means anything.
**Teaches:**
- Log-likelihood scoring: the cloze format against the letter format, length normalization by token, character and byte (`d2l.score_choices` with `norm` in `sum`, `token`, `char`, `byte`, one forward pass yielding every convention through `d2l.ScoreComponents`), and the harness's continuation-splitting convention.
- Generative answers and extraction (`d2l.extract_answer` for GSM8K-style and boxed answers), execution checks (`d2l.run_python_tests`, not a security boundary), and why scoring only the answer option mischaracterizes a model meant to reason before answering.
- pass@k from a common sample pool (`d2l.pass_at_k`); majority voting as a different selection policy.
- Bits per byte as the tokenizer-free likelihood score (`d2l.bits_per_byte` with sliding windows); chrF and BLEU restated in one sentence as string-overlap conventions for generated text.
- Determinism controls: batch-dependent kernels change bf16 near-ties, so the batch size is part of the protocol and the disagreement rate between batch sizes 1 and 8 is reported beside every flip table.
**Derivation:** The unbiased pass@k estimator from n samples with c correct, 1 - C(n - c, k) / C(n, k), its product form for stable evaluation, and its variance when each sample succeeds independently with probability p. Smallest numeric case: n = 4, c = 1, k = 2 gives 1 - C(3, 2) / C(4, 2) = 1 - 3/6 = 0.5, against the biased 1 - (1 - 1/4)^2 = 0.4375.
**Notebook:**
1. Score Qwen3-0.6B and Qwen3.5-0.8B on 500 seeded ARC-Challenge items under cloze-sum, cloze-token, cloze-char and letter conventions from one forward pass per format; compare `d2l.score_choices` against committed lm-evaluation-harness per-item choices on 200 items (at most 2 of 200 differ per norm).
2. Score the same items generatively: free generation, `d2l.extract_answer`, exact match; add 100 GSM8K problems with an 8-sample pool for pass@k at k = 1, 2, 4, 8 and compare with majority voting.
3. Plot accuracy per convention per model with item-bootstrap bars and the paired difference per convention with its interval.
4. Measure the batch-size disagreement rate (1 versus 8) in bf16 on the 500 items and print it beside the flip table.
5. Claim that survives noise: at least one convention flips the ranking of the two checkpoints by more than the paired interval, or, if none does, the section reports the convention dependence of the margins and of the reliability diagram instead; the text is written after the grid runs.
6. Reused pieces: `d2l.generate`, `d2l.sample_logits` (1.2), `d2l.label_logprobs` (1.3) as the special case; a local 8-line bootstrap until 1.5.
**Figures:** `fig_lm_scoring_conventions` (the same item rendered under cloze, letter and generative conventions with the scored tokens marked); `fig_lm_ranking_flip` (plot: accuracy per convention per model with bars and the paired difference with its interval).
**Restate / forward:** `subsec_bits-per-byte` and `sec_mdl-coding-compression` (per-token perplexity is not comparable across tokenizers; bits per byte is, and `eq_mdl-bpc` is the conversion); `sec_seq2seq` (chrF and BLEU score n-gram overlap with a reference and nothing else); `subsec_deterministic_arithmetic` (kernel nondeterminism makes bitwise reproduction a setting, not a default); `sec_softmax` (calibration of the choice probabilities). Forward: 1.5 (the shared bootstrap and suites).
**Exercises:** 6 to 8. Load-bearing: derive the n at which the variance of the pass@k estimator halves for fixed p and k (derive); measure the effect of length normalization on HellaSwag against ARC and explain the difference by the choices' length spread (measure); implement minimum-Bayes-risk decoding over the 8-sample pool with chrF as the utility and compare with majority voting (open); check that `d2l.pass_at_k` returns 0.4 for n = 5, c = 2, k = 1 and 1.0 whenever n - c < k (check).
**Writer notes:**
- Exports with fixed contracts (`01_interfaces.md`): `d2l.score_choices(model, tokenizer, prompts, choices, norm='token', *, batch_size=8, return_components=False)` (log-softmax in fp32 whatever the model dtype; prompt and choice tokenized separately and the continuation taken as the suffix; trailing prompt whitespace moved to the choice; no BOS; an empty continuation raises); `d2l.pass_at_k(n, c, k)` vectorized over integer arrays, exact 0.0 for c = 0 and 1.0 for n - c < k; `d2l.pass_at_k_variance(n, p, k)`; `d2l.extract_answer(text, kind='gsm8k'|'boxed', strict=False)`; `d2l.bits_per_byte(model, tokenizer, texts, ctx_len=2048, stride=None, batch_size=4)` scoring every token once; `d2l.format_mc(question, options, fmt='cloze'|'letter', *, shots=())`. 8.5 composes `extract_answer` with its own `canonical_answer`; 3.4 imports `bits_per_byte` and adds `bpb_from_nll`.
- Reference implementation and tolerance: a committed fixture `lm_eval_reference_arc.json` from a pinned lm-evaluation-harness version (per-item loglikelihoods and `acc`, `acc_norm` for Qwen3-0.6B-Base on 200 items); `acc_norm`'s length definition and the continuation rule are read from that version and the fixture is regenerated if it changes.
- Scale: two checkpoints (Qwen3-0.6B and Qwen3.5-0.8B) and 500 items instead of four checkpoints and the full splits; the committed per-item scores keep the figure cells at T0.
- Sandbox: `d2l.run_python_tests` runs model-written code in a `python -I` subprocess with resource limits; the build must run that cell in a container or commit outcomes only with `RERUN=False`.
- Pitfalls to state: pass@k estimated as 1 - (1 - c/n)^k or from n = k samples; extractors crediting the last number mentioned; label-word leakage in letter-format demonstrations; scoring a reasoning model's answer option without its reasoning.
- Data: `allenai/ai2_arc` ARC-Challenge test (CC BY-SA 4.0); `openai/gsm8k` (MIT), 4-shot with the first four train problems; `Rowan/hellaswag` for the exercise (licence read from the card); `Salesforce/wikitext` test for bits per byte (not redistributed). Committed: `lm-arc-scores.parquet`, `lm-gsm8k-samples.jsonl`.
- Do not re-teach: perplexity and bits per byte (defined in `sec_language-model`), arithmetic coding (`sec_mdl-coding-compression`), chrF and BLEU (`sec_seq2seq`), seeds and determinism (`sec_repro`).
- If no flip occurs on the chosen pair, add Qwen3-1.7B-Base as a third checkpoint (one extra column, no new idea).
**Sources:**
- Papers: [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (pass@k and its unbiased estimator); [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) (harness design choices and their consequences); [Leveraging Large Language Models for Multiple Choice Question Answering](https://arxiv.org/abs/2210.12353) (cloze against letter formats); [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) (multi-metric reporting and calibration).
- Code and documentation: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (the fixture's source; task conventions); [LightEval](https://github.com/huggingface/lighteval) (a second scorer implementation to read); [human-eval](https://github.com/openai/human-eval) (the execution protocol); [IFEval](https://arxiv.org/abs/2311.07911) (verifiable instruction constraints, an exercise); Transformers, [perplexity guide](https://huggingface.co/docs/transformers/en/perplexity) (fixed-length and strided evaluation).
- Courses, tutorials and blogs: Hugging Face, [What's going on with the Open LLM Leaderboard?](https://huggingface.co/blog/open-llm-leaderboard-mmlu) (one benchmark, three implementations, three rankings); Thinking Machines, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (batch-dependent kernels); [CS336](https://cs336.stanford.edu/) (evaluation lecture).
-->

```{.python .input #scoring-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #scoring-imports-extras}
%%tab pytorch
import collections
import json
import os
import re
import string
import subprocess
import sys
import tempfile
import time
import pandas as pd
```

<!-- Placeholder (stage 2): the datasets, pinned to commits; the registered artifacts follow at capture. -->

```{.python .input #scoring-data-hub}
%%tab pytorch
#@save
# Several of these files share a name: fetch them with d2l.download_keyed
d2l.DATA_HUB['arc-challenge-test'] = (
    'https://huggingface.co/datasets/allenai/ai2_arc/resolve/'
    '210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/'
    'test-00000-of-00001.parquet',
    '029435570dd5b680c95603f2f51a93218fa30cb2')
d2l.DATA_HUB['arc-challenge-train'] = (
    'https://huggingface.co/datasets/allenai/ai2_arc/resolve/'
    '210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/'
    'train-00000-of-00001.parquet',
    '43a863338c7852a03fc5ab79f7c80b88e01a192d')
d2l.DATA_HUB['gsm8k-train'] = (
    'https://huggingface.co/datasets/openai/gsm8k/resolve/'
    '740312add88f781978c0658806c59bc2815b9866/main/'
    'train-00000-of-00001.parquet',
    '2328586d74d5426d030f29ac85a485f58514efbd')
d2l.DATA_HUB['gsm8k-test'] = (
    'https://huggingface.co/datasets/openai/gsm8k/resolve/'
    '740312add88f781978c0658806c59bc2815b9866/main/'
    'test-00000-of-00001.parquet',
    'b0c27c2f1cb845ba83b135c7e2575070b33ca967')
d2l.DATA_HUB['wikitext-2-raw-test'] = (
    'https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
    'b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/'
    'test-00000-of-00001.parquet',
    'ab23bd60c20d2e169a8dc2d139ab6e7bf8b74e91')
```

```{.python .input #scoring-download}
%%tab pytorch
#@save
def download_keyed(key, root=os.path.join('..', 'data', 'lm')):
    """Download DATA_HUB[key] into root/key/ and return its local path.
    Files from different datasets often share a name (Hugging Face's
    'test-00000-of-00001.parquet'), and d2l.download keeps one file per
    name and folder, so each key gets a folder of its own."""
    return d2l.download(key, os.path.join(root, key))
```

<!-- Placeholder (stage 2): one item, two formats. -->

```{.python .input #scoring-format-mc}
%%tab pytorch
#@save
def format_mc(question, options, fmt='cloze', *, shots=()):
    """A multiple-choice item as (prompt, choices). 'cloze': the prompt
    'Question: {question}\\nAnswer:' and each option after a space;
    'letter': 'Question: {question}', a line 'A. {option}' per option and
    'Answer:', with the letters ' A', ' B', ... as choices (OLMES's
    multiple-choice formulation); 'mmlu': the same without 'Question: ',
    as the MMLU paper and lm-evaluation-harness's mmlu tasks write it. Each
    shot, a tuple (question, options, answer_index), is rendered the same
    way with its answer and put before the item, separated by a blank line.
    Callers that want a header (such as MMLU's subject line) prepend it
    themselves."""
    if fmt not in ('cloze', 'letter', 'mmlu'):
        raise ValueError(f'unknown fmt: {fmt!r}')

    def render(q, opts):
        opts = [str(o) for o in opts]  # also numpy arrays, pandas Series
        if not opts:
            raise ValueError(f'no options for {q!r}')
        if fmt == 'cloze':
            return f'Question: {q}\nAnswer:', [' ' + o for o in opts]
        if len(opts) > 26:
            raise ValueError(f'{len(opts)} options, but 26 letters')
        letters = [chr(ord('A') + i) for i in range(len(opts))]
        lines = [q if fmt == 'mmlu' else f'Question: {q}'] + [
            f'{a}. {o}' for a, o in zip(letters, opts)]
        return '\n'.join(lines + ['Answer:']), [' ' + a for a in letters]
    blocks = []
    for q, opts, a in shots:
        prompt, choices = render(q, opts)
        if not 0 <= a < len(choices):
            raise ValueError(f'answer index {a} for {len(choices)} options')
        blocks.append(prompt + choices[a])
    prompt, choices = render(question, options)
    return '\n\n'.join(blocks + [prompt]), choices
```

<!-- Placeholder (stage 2): the log-probability of each choice, from one forward pass per batch. -->

```{.python .input #scoring-score-choices}
%%tab pytorch
#@save
ScoreComponents = collections.namedtuple(  # each (N, C_max), padded with
    'ScoreComponents', 'll n_tokens n_chars n_bytes')  # -inf (ll) and 0

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
def score_choices(model, tokenizer, prompts, choices, norm='token', *,
                  batch_size=8, return_components=False):
    """Log-likelihood scores (N, C_max) of each choice after its prompt,
    -inf in padded columns; the prediction is argmax(-1). choices holds N
    lists (ragged) or one list shared by all prompts. norm divides the
    summed log-probability ll: 'sum' by 1, 'token' by the continuation's
    tokens, 'char' and 'byte' by the length and UTF-8 length of the choice
    without one leading whitespace character (lm-evaluation-harness's
    acc_norm and acc_bytes). Trailing whitespace of a prompt moves to the
    front of its choices; the continuation is the suffix of
    encode(prompt + choice) beyond encode(prompt) (a warning says when the
    prompt's last token merges into the choice's, which this rule scores as
    a different text), with no BOS (Qwen has none; for a tokenizer that
    needs one, start each prompt with tokenizer.bos_token); an empty
    prompt is preceded by the document-boundary token. With
    return_components=True, returns ScoreComponents, from which every norm
    follows. Log-softmax is taken in fp32 over the continuation positions
    only. In bf16 an item's score depends on which items share its batch,
    so a different call over the same items (another batch size or item
    set) agrees only up to that batch drift, not bit for bit."""
    import warnings
    if norm not in ('sum', 'token', 'char', 'byte'):
        raise ValueError(f'unknown norm: {norm!r}')
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
    max_len, cut_items, inputs, cont_ids = _lm_max_len(model), set(), [], []
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
    hf = sys.modules.get('transformers')
    hf = hf is not None and isinstance(d2l._lm_unwrap(model),
                                       hf.PreTrainedModel)
    qwen3 = isinstance(d2l._lm_unwrap(model), d2l.Qwen3GPT)
    pad = (getattr(tokenizer, 'pad_token_id', None) or 0) if hf else 0
    device = next(model.parameters()).device
    order = sorted(range(len(rows)), key=lambda r: (-len(inputs[r]), r))
    ll = torch.empty(len(rows), dtype=torch.float64)
    for s in range(0, len(order), batch_size):
        b = order[s:s + batch_size]
        T = max(len(inputs[r]) for r in b)
        m = [len(cont_ids[r]) for r in b]
        with torch.inference_mode():
            if hf:  # left padding: the continuations end every row
                X = torch.tensor([[pad] * (T - len(inputs[r])) + inputs[r]
                                  for r in b], device=device)
                mask = (torch.arange(T, device=device) >= torch.tensor(
                    [T - len(inputs[r]) for r in b], device=device)[:, None]
                        ).long()
                pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
                z = model(input_ids=X, attention_mask=mask, position_ids=pos,
                          use_cache=False, logits_to_keep=max(m)).logits
                Z = torch.cat([z[k, z.shape[1] - mk:]
                               for k, mk in enumerate(m)])
            else:  # right padding, exact under the causal mask
                X = torch.tensor([inputs[r] + [pad] * (T - len(inputs[r]))
                                  for r in b], device=device)
                starts = [len(inputs[r]) - mk for r, mk in zip(b, m)]
                off = min(starts) if qwen3 else 0
                out = model(X, logits_to_keep=T - off) if qwen3 else model(X)
                z = getattr(out, 'logits', out)
                Z = torch.cat([z[k, st - off:st - off + mk]
                               for k, (st, mk) in enumerate(zip(starts, m))])
            y = torch.tensor([t for r in b for t in cont_ids[r]],
                             device=Z.device)
            lp = torch.cat([  # fp32 log-softmax, at most 4,096 rows at once
                Z[i:i + 4096].float().log_softmax(-1).gather(
                    -1, y[i:i + 4096, None])[:, 0]
                for i in range(0, len(y), 4096)])
            sums = torch.stack([v.double().sum() for v in lp.split(m)])
        ll[b] = sums.cpu()
    C = max(map(len, choices), default=0)
    comps = [torch.zeros(len(prompts), C, dtype=dt)
             for dt in (torch.float32, torch.long, torch.long, torch.long)]
    comps[0].fill_(-math.inf)
    for r, (i, j) in enumerate(rows):
        for t, v in zip(comps, (ll[r].item(), len(cont_ids[r]),
                                len(texts[r]),
                                len(texts[r].encode('utf-8')))):
            t[i, j] = v
    comps = ScoreComponents(*comps)
    if return_components:
        return comps
    return comps.ll / {'sum': 1, 'token': comps.n_tokens,
                       'char': comps.n_chars, 'byte': comps.n_bytes}[norm]
```

<!-- Placeholder (stage 2): bits per byte of running text, in sliding windows. -->

```{.python .input #scoring-bits-per-byte}
%%tab pytorch
#@save
def bits_per_byte(model, tokenizer, texts, ctx_len=2048, stride=None,
                  batch_size=4):
    """Bits per UTF-8 byte of texts: total negative log-likelihood in nats
    over ln 2 times the total bytes. Each text is put in Unicode NFC form
    (as Qwen's tokenizers do themselves, so the bytes counted are those
    scored), tokenized whole (no BOS) and preceded by one unscored
    document-boundary token (Qwen3: 151643, Qwen3.5: 248044; see
    _lm_boundary_id); without one, its first token, and any more up to the
    end of the character it starts, are unscored and their bytes leave the
    denominator. Windows of ctx_len input tokens (capped at the model's
    maximum input length, with a warning) advance by stride (default
    ctx_len // 2) and score every token once, the first window ctx_len
    tokens and each later one stride tokens, each with at least
    ctx_len - stride + 1 tokens of context; texts are scored independently.
    A Transformers model with a linear output layer runs its decoder on
    batch_size windows and projects 1,024 positions at a time; any other
    model runs one window at a time."""
    import unicodedata
    import warnings
    texts = [texts] if isinstance(texts, str) else list(texts)
    texts = [unicodedata.normalize('NFC', str(t)) for t in texts]
    max_len = _lm_max_len(model)
    if max_len and ctx_len > max_len:
        warnings.warn(f'bits_per_byte: ctx_len {ctx_len} capped at the '
                      f"model's maximum input length {max_len}")
        ctx_len = max_len
    stride = ctx_len // 2 if stride is None else stride
    if ctx_len < 1 or not 1 <= stride <= ctx_len:
        raise ValueError(f'need 1 <= stride <= ctx_len, got {stride}, '
                         f'{ctx_len}')
    b, windows, n_bytes = _lm_boundary_id(tokenizer), [], 0
    for text, ids in zip(texts, _lm_encode(tokenizer, texts)):
        if not ids:
            continue
        n_bytes += len(text.encode('utf-8'))
        first = 1  # unscored tokens: the boundary, or the text's first ones
        if b is None:  # the first token has no context; if it ends inside
            while first < len(ids):  # a character, so do the next ones
                head = tokenizer.decode(ids[:first])
                if text.startswith(head) or '\ufffd' not in head:
                    break
                first += 1
            n_bytes -= len(tokenizer.decode(ids[:first]).encode('utf-8'))
        seq = ids if b is None else [b] + ids
        done = first - 1  # targets seq[first:done + 1] are scored
        while done < len(seq) - 1:  # inputs seq[start:end] predict
            end = min(max(ctx_len, done + 1) if done == first - 1
                      else done + stride, len(seq) - 1)
            start = max(end - ctx_len, 0)  # seq[start + 1:end + 1]
            windows.append((seq[start:end], seq[done + 1:end + 1]))
            done = end
    if not n_bytes:
        raise ValueError('no text to score')
    inner = d2l._lm_unwrap(model)
    hf = sys.modules.get('transformers')
    hf = hf is not None and isinstance(inner, hf.PreTrainedModel)
    head = inner.get_output_embeddings() if hf else None
    cfg = inner.config.get_text_config() if hf else None
    split = (hf and isinstance(head, nn.Linear) and
             getattr(cfg, 'final_logit_softcapping', None) is None)
    device, nll = next(model.parameters()).device, 0.0

    def add_nll(H, y, project):  # H: (positions, d) or logits; y: targets
        total = 0.0
        for i in range(0, len(y), 1024):
            z = project(H[i:i + 1024]).float()
            total += F.cross_entropy(z, y[i:i + 1024], reduction='sum').item()
        return total
    with torch.inference_mode():
        for s in range(0, len(windows), batch_size if split else 1):
            w = windows[s:s + (batch_size if split else 1)]
            T = max(len(x) for x, _ in w)
            X = torch.tensor([x + [0] * (T - len(x)) for x, _ in w],
                             device=device)
            y = torch.tensor([t for _, ys in w for t in ys], device=device)
            if split:  # decoder states, then the head in chunks
                mask = (torch.arange(T, device=device) < torch.tensor(
                    [len(x) for x, _ in w], device=device)[:, None]).long()
                dec = (inner.get_decoder() if hasattr(inner, 'get_decoder')
                       else inner.model)
                H = dec(input_ids=X, attention_mask=mask,
                        use_cache=False).last_hidden_state
                H = torch.cat([H[k, len(x) - len(ys):len(x)]
                               for k, (x, ys) in enumerate(w)])
                nll += add_nll(H, y, head)
            else:  # one window: logits for its scored positions
                n = len(y)
                if hf:
                    z = model(input_ids=X, use_cache=False,
                              logits_to_keep=n).logits[0]
                elif isinstance(inner, d2l.Qwen3GPT):
                    z = model(X, logits_to_keep=n)[0]
                else:
                    out = model(X)
                    z = getattr(out, 'logits', out)[0, -n:]
                nll += add_nll(z, y, lambda v: v)
    return nll / (math.log(2) * n_bytes)
```

<!-- Placeholder (stage 2): the parser is part of the metric. -->

```{.python .input #scoring-extract-answer}
%%tab pytorch
#@save
def extract_answer(text, kind='gsm8k', strict=False):
    """The answer a generated text commits to, as a string, or None.
    'gsm8k': the first number after the last '####', else after the last
    'the answer is' (any case), else, unless strict, the last number in the
    text; commas and '$' are removed and the number is normalized ('1,000'
    and '1000.00' give '1000', '-0.50' and '−0.50' give '-0.5'); a hyphen
    right after a letter, digit or point is not a sign ('10-15' gives
    '15'). 'boxed': the content
    of the last \\boxed{...} whose braces match, stripped."""
    if kind == 'boxed':
        found, i = None, text.find('\\boxed{')
        while i != -1:
            depth = 0
            for j in range(i + 6, len(text)):  # from the opening brace
                depth += {'{': 1, '}': -1}.get(text[j], 0)
                if depth == 0:
                    found = text[i + 7:j].strip()
                    break
            i = text.find('\\boxed{', i + 1)
        return found or None
    if kind != 'gsm8k':
        raise ValueError(f'unknown kind: {kind!r}')
    from decimal import Decimal
    number = re.compile(r'(?:(?<![\w.])[-\u2212])?\$?'  # sign, not hyphen
                        r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)')

    def normalize(s):
        d = Decimal(s.translate({ord(','): None, ord('$'): None,
                                 0x2212: '-'}))
        if d == d.to_integral_value():
            return str(int(d))
        return format(d.normalize(), 'f')
    for marker in ('####', 'the answer is'):
        ends = [k.end() for k in re.finditer(re.escape(marker), text, re.I)]
        m = number.search(text, ends[-1]) if ends else None
        if m:
            return normalize(m.group())
    found = None if strict else number.findall(text)
    return normalize(found[-1]) if found else None
```

<!-- Placeholder (stage 2): twelve strings, expected answers. -->

```{.python .input #scoring-extract-check}
%%tab pytorch
cases = [  # (text, kind, expected)
    ('so she has 18 left.\n#### 18', 'gsm8k', '18'),
    ('#### 1,234', 'gsm8k', '1234'),
    ('The total is $1,000.', 'gsm8k', '1000'),
    ('The temperature drops to -7 degrees.\n#### -7', 'gsm8k', '-7'),
    ('He pays 3.50 each, 2 times: 7.00.', 'gsm8k', '7'),
    ('The answer is 42.', 'gsm8k', '42'),
    ('the answer is 12, not 15 as first thought', 'gsm8k', '12'),
    ('First 5 apples, then 3 more, so 8 apples in all', 'gsm8k', '8'),
    ('I cannot tell from the question.', 'gsm8k', None),
    (r'So \boxed{B}, or perhaps \boxed{C}.', 'boxed', 'C'),
    (r'\boxed{ \frac{1}{2} }', 'boxed', r'\frac{1}{2}'),
    (r'The letter is B.', 'boxed', None)]
for text, kind, want in cases:
    got = d2l.extract_answer(text, kind)
    assert got == want, (text, got, want)
    print(f'{kind:5s} {got!s:12s} {text[:48]!r}')
```

<!-- Placeholder (stage 2): pass@k from a pool of n samples, and its variance. -->

```{.python .input #scoring-pass-at-k}
%%tab pytorch
#@save
def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples of which c are correct: the
    probability that k of them, drawn without replacement, include a
    correct one, 1 - C(n - c, k) / C(n, k), in the product form
    1 - prod_{i = n - c + 1}^{n} (1 - k / i). Integer arrays broadcast."""
    n, c, k = np.broadcast_arrays(*(np.asarray(v) for v in (n, c, k)))
    if not all(np.issubdtype(v.dtype, np.integer) for v in (n, c, k)):
        raise TypeError('n, c and k must be integers')
    for bad, msg in ((c < 0, 'c < 0'), (c > n, 'c > n'), (k < 1, 'k < 1'),
                     (k > n, 'k > n')):
        if np.any(bad):
            raise ValueError(f'pass_at_k needs 0 <= c <= n, 1 <= k <= n: '
                             f'{msg}')
    out = np.empty(n.shape)
    for idx in np.ndindex(n.shape):
        ni, ci, ki = int(n[idx]), int(c[idx]), int(k[idx])
        out[idx] = 1.0 if ni - ci < ki else 1.0 - np.prod(
            1.0 - ki / np.arange(ni - ci + 1, ni + 1))
    return float(out) if out.ndim == 0 else out

#@save
def pass_at_k_variance(n, p, k):
    """Variance of pass_at_k(n, c, k) when c ~ Binomial(n, p): the sum over
    c of Binomial(c; n, p) (U(n, c, k) - (1 - (1 - p)^k))^2."""
    if not 0 <= p <= 1 or not 1 <= k <= n:
        raise ValueError(f'need 0 <= p <= 1 and 1 <= k <= n: {n}, {p}, {k}')
    if p in (0, 1):
        return 0.0
    c = np.arange(n + 1)
    log_pmf = (math.lgamma(n + 1) - np.array([math.lgamma(i + 1) + math.lgamma(
        n - i + 1) for i in c]) + c * math.log(p) + (n - c) * math.log1p(-p))
    U = pass_at_k(np.full(n + 1, n), c, np.full(n + 1, k))
    return float(np.sum(np.exp(log_pmf) * (U - (1 - (1 - p) ** k)) ** 2))
```

<!-- Placeholder (stage 2): the smallest case and the exact moments. -->

```{.python .input #scoring-passk-check}
%%tab pytorch
print(f'pass@2 from n=4, c=1: {d2l.pass_at_k(4, 1, 2):.4f}; '
      f'plug-in: {1 - (1 - 1 / 4) ** 2:.4f}')
n, p, k = 20, 0.3, 5
mean = 1 - (1 - p) ** k
var = d2l.pass_at_k_variance(n, p, k)
c = np.arange(n + 1)
pmf = np.array([math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in c])
plug = np.sum(pmf * (1 - (1 - c / n) ** k))
print(f'n={n}, p={p}, k={k}: mean {mean:.5f}, variance {var:.6f} '
      f'(sd {var ** 0.5:.4f}), plug-in mean {plug:.5f} '
      f'(bias {plug - mean:+.4f})')
```

<!-- Placeholder (stage 2): run a program against its tests, with limits. -->

```{.python .input #scoring-run-tests}
%%tab pytorch
#@save
def run_python_tests(program, test_code, timeout=5.0):
    """True iff program followed by test_code runs to its end and the
    process exits with code 0, in a fresh `python -I` process, in an empty
    temporary directory, with stdin closed, within timeout seconds of
    wall-clock time and, on POSIX, within limits on CPU time and address
    space (1 GiB). Exiting early (sys.exit, os._exit, a unittest.main() of
    the program's own) fails, as in human-eval; on POSIX, any process the
    code started is killed when the call returns. This is not a sandbox:
    the code runs as the caller, can read and write the caller's files and
    reach the network. Run code you did not write in a container."""
    import secrets
    import signal
    import subprocess
    import tempfile
    token = secrets.token_hex(16)  # written only once test_code has run
    limits = ''
    if os.name == 'posix':  # set in the child, before the program runs
        cpu = math.ceil(timeout) + 1
        limits = (
            'import resource\n'
            'for r, v in ((resource.RLIMIT_CPU, %d), '
            '(resource.RLIMIT_AS, 1 << 30)):\n'
            '    try:\n        resource.setrlimit(r, (v, v))\n'
            '    except (ValueError, OSError):\n        pass\n' % cpu)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'main.py'), 'w') as f:
            f.write(program + '\n' + test_code + '\n')
        done = os.path.join(tmp, 'done')  # absolute: the code may chdir
        proc = subprocess.Popen(
            [sys.executable, '-I', '-c', limits + 'import runpy\n'
             "runpy.run_path('main.py', run_name='__main__')\n"
             f'open({done!r}, "w").write({token!r})'],
            cwd=tmp, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=os.name == 'posix')
        try:
            if proc.wait(timeout=timeout) != 0:
                return False
            with open(done) as f:
                return f.read() == token
        except (subprocess.TimeoutExpired, OSError):
            return False
        finally:
            if os.name == 'posix':  # the process and any it started
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                proc.kill()
            proc.wait()
```

<!-- Placeholder (stage 2): four hand-written programs, one verdict each. -->

```{.python .input #scoring-run-tests-check}
%%tab pytorch
add = 'def add(a, b):\n    return a + b'
tests = 'assert add(2, 3) == 5\nassert add(-1, 1) == 0'
programs = {'correct': add,
            'wrong': 'def add(a, b):\n    return a - b',
            'infinite loop': 'def add(a, b):\n    while True:\n        pass',
            'syntax error': 'def add(a, b)\n    return a + b'}
timeout = 5.0
for name, program in programs.items():
    tic = time.time()
    ok = d2l.run_python_tests(program, tests, timeout=timeout)
    elapsed = time.time() - tic
    print(f'{name:14s} passed={ok!s:5s} {elapsed:.2f} s')
    assert ok == (name == 'correct') and elapsed < timeout + 1
```
