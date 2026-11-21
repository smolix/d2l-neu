# Synthetic and Rephrased Data
:label:`sec_lm_synthetic_data`

<!-- For maintainers. Plan: docs/llm/language_models_part_plan.md, "3.5
Synthetic and Rephrased Data"; blueprint docs/llm/blueprints/ch03_data.md,
section 3.5. Rights: the rephrased corpus is FineWeb-Edu text (ODC-By)
rewritten by an Apache-2.0 model and ships as text with its attribution;
Nemotron-CC-v2 (NVIDIA Data Agreement for Model Training) is cited, never
shipped. Every prose sentence that quotes an arm result says "at T1 scale
(8.1M-parameter proxies, 2^24 tokens)" (author ruling: 20.5 runs its proxy
comparison at T1). The judge runs with thinking on and no thinking budget
(author ruling R3, 2026-11-20); open for the author: max_tokens 12,000
leaves 4 of 60 replies without a verdict. -->

The previous sections took text as the web offers it and decided what to
keep, how to remove its copies and how often to read each kind. At the end
of :numref:`sec_lm_data_mixtures` the supply ran short: a small source read
many times is worth far less than as many fresh sources of the same size.
A language model offers a way out that would have sounded odd a few years
ago. It can write more text.

Here is the smallest version of the idea, which we run later in this
section. A model of two billion parameters reads a paragraph from an
educational web page about a mouthwash for oral pain and rewrites it in the
style of an encyclopedia. The rewrite is cleaner than its source, it keeps
every fact, and it costs the model a few seconds. If one paragraph can be
rewritten, so can a million, and a pretraining run could read each source
once as written and once rephrased instead of twice as written. Maini and
colleagues (2024) did this with the C4 web crawl and report that their
models reached the same perplexity about three times faster. Whole crawls
have since been rewritten the same way.

Generated text has costs, and they are hard to see in any single example. A
rewrite can change a number or add a claim and still read perfectly well.
Samples from one prompt resemble each other far more than documents written
by different people. And a model that learns from a model's text learns the
generator's habits along with its content. This section measures these
costs. It builds a rephrasing pipeline and checks what comes out, then
trains small models on an equal budget of real text, rewritten text and
mixtures of the two, and evaluates each on both kinds of text. Two questions
close it: what happens when models learn from models generation after
generation, and who may train on generated text at all.

```{.python .input #synthetic-data-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #synthetic-data-extras}
%%tab pytorch
import collections
import gzip
import hashlib
import io
import concurrent.futures
import itertools
import json
import os
import pathlib
import re
import shutil
import tarfile
import textwrap
import time
import pandas as pd
from scipy import sparse
```

The rewrites shown in the text, about 150 of them, are generated live by
Qwen3.5-2B in bf16 on a GPU in a few minutes. Everything larger was computed once and registered: a
corpus of 4.7 million tokens of source chunks with their rewrites, the
verdicts of a judge model on sixty of them, and the fifteen training runs of
the comparison. A normal build reads these files and repeats the start of
one training run as a check; setting `D2L_RERUN` to a name in `ARTIFACTS`
regenerates that file instead. The runs use the wider proxies of
:numref:`sec_lm_data_mixtures` at that section's smaller scale, T1, so every
result of the comparison is stated at T1 scale (8.1M-parameter proxies,
$2^{24}$ tokens).

```{.python .input #synthetic-data-config}
%%tab pytorch
REPHRASER = 'Qwen/Qwen3.5-2B'  # writes the rephrasings (non-thinking)
JUDGE = d2l.EDU_LABELLER_LOCAL['model']  # checks them: Qwen3.8-27B (vLLM)
JUDGE_MAX_TOKENS = 12_000  # the judge reasons first: no thinking budget
JUDGE_PER_SERVER = 1  # requests in flight per judge server: repeatable
PROXY_SMALL = d2l.ProxyCfg(num_hiddens=192, num_heads=3, num_blks=4,
                           warmup_frac=0.1)  # 26 of 256 steps: see below
TOKENS, U, VAL_U = 2**24, 2**21, 2**19  # per run; S1 and S2; validation
SEEDS, EPOCHS = (0, 1, 2), 8  # data-order seeds; arm A's passes over S1
CHUNK_TOKENS = 400            # the bound of chunk_doc
LEN_RATIO = (0.5, 1.5)  # a rewrite outside is dropped (cut short, padded)
FLAGS = dict(copy_rate=0.8, num_fidelity=1.0)  # past these: flagged, kept
SAMPLING = dict(temperature=0.7, top_p=0.9, seed=0)  # every rephrasing
GPU = torch.cuda.is_available()
SMALL = not GPU  # the CPU path: fewer, shorter live generations
DTYPE = torch.bfloat16 if GPU else torch.float32
N_LIVE = 4 if SMALL else 60          # live chunks per style
LIVE_MAX_SRC = 100 if SMALL else CHUNK_TOKENS  # their longest source
N_SAMPLES = 3 if SMALL else 9        # samples of one prompt (opening)
OPEN_MAX_SRC = 200                   # the opening chunk's longest source
NEW_TOKENS = 192 if SMALL else 512   # max_new_tokens of a live rephrase
SEED_TOKENS = 96 if SMALL else 300   # max_new_tokens from a seed
MIN_WORDS = 50                       # the seed comparison's shortest output
LIVE_BATCH, REGEN_BATCH, N_JUDGE = 32, 96, 60  # 96: a part in < 30 min
REGEN_TOKENS = 512                   # max_new_tokens of the corpus
EVAL_AT, LOG_EVERY, LIVE_TOKENS = (0.25, 0.5, 0.75, 1.0), 4, 2**20
ARTIFACTS = {  # name: (registered key, file a regeneration writes)
    'reph-s1': ('lm-rephrase-fwedu-5m-v1', 'reph-s1.jsonl.gz'),
    'reph-s2': ('lm-rephrase-fwedu-5m-v1', 'reph-s2.jsonl.gz'),
    'reph-val': ('lm-rephrase-fwedu-5m-v1', 'reph-val.jsonl.gz'),
    'reph-judge': ('lm-data-synth-results-v1', 'reph-judge.json'),
    'synth-arms': ('lm-data-synth-results-v1', 'synth-arms.json')}
RERUN = d2l.lm_rerun(ARTIFACTS)
assert GPU or not RERUN, 'the regenerations generate text and train on a GPU'
assert GPU or not os.environ.get('CUDA_VISIBLE_DEVICES'), (
    'a GPU slot without CUDA: the small CPU path must not be captured')
RERUN_DIR = os.path.join('..', 'data', 'lm', 'rerun')
with torch.device('meta'):
    N_PARAMS = d2l.num_params(d2l._proxy_model(PROXY_SMALL))
TIER = (f'at T1 scale ({N_PARAMS / 1e6:.1f}M-parameter proxies, '
        f'2^{round(math.log2(TOKENS))} tokens per run)')
TAG = f'[{TIER}]'

def say(*args, sep=' '):
    """print, each line wrapped to the page's 100 columns."""
    print('\n'.join(l if len(l) <= 100 else textwrap.fill(
        l, 100, subsequent_indent='    ', break_long_words=False,
        break_on_hyphens=False) for l in sep.join(map(str, args)).split('\n')))

say(f'proxy comparison {TIER.replace(" scale", "")}; GPU={GPU}, '
    f'SMALL={SMALL}, regenerating: {", ".join(sorted(RERUN)) or "none"}')
PINS = {REPHRASER: d2l.HF_REVISIONS[REPHRASER],
        JUDGE: d2l.EDU_LABELLER_LOCAL['revision']}
for m, rev in PINS.items():
    say(f'  {m}@{rev[:8]}')
```

Two files are registered. The corpus is FineWeb-Edu text (ODC-By) rewritten
by an Apache-2.0 model, so it ships as text with the attribution that ODC-By
asks for, and every printed example carries it. The results file holds
the judge's verdicts, the logs of the fifteen runs and three Cosmopedia
rows.

```{.python .input #synthetic-data-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-rephrase-fwedu-5m-v1'] = (  # FineWeb-Edu chunks and their
    # wiki-style rewrites by Qwen3.5-2B, 4.7M tokens (20.5); ODC-By
    d2l.LM_DATA_URL + 'lm-rephrase-fwedu-5m-v1.tar',
    '1b8598668e0959eeef5f4b0045cc7ed6574880e1')
d2l.DATA_HUB['lm-data-synth-results-v1'] = (  # the judge's verdicts, the 15
    # proxy runs of the five arms and three Cosmopedia rows (20.5)
    d2l.LM_DATA_URL + 'lm-data-synth-results-v1.json',
    'd27e4094b3936e83efc89b6ccc69760f538c49fd')

TAR_KEY, TAR_FILE = 'lm-rephrase-fwedu-5m-v1', 'lm-rephrase-fwedu-5m-v1.tar'
RES_KEY = 'lm-data-synth-results-v1'
UPSTREAM = ('lm-mix-web-20m-v1', 'lm-mix-val-1m-v1', 'lm-mix-runs-t1-v1',
            'lm-r2-tokenizer-v1')
for name, keys in (('this section', (TAR_KEY, RES_KEY)),
                   ('upstream', UPSTREAM)):
    say(f'{name}: registered', ', '.join(
      k for k in keys if k in d2l.DATA_HUB) or 'none', '| pending:',
        ', '.join(k for k in keys if k not in d2l.DATA_HUB) or 'none')
RESULTS = None  # the registered results: judge verdicts, runs, Cosmopedia

def write_json(path, obj):
    with open(path, 'w') as f:  # strict JSON: a missing value is null
        json.dump(obj, f, sort_keys=True, default=float, allow_nan=False)

if RES_KEY in d2l.DATA_HUB:
    with open(d2l.lm_artifact(RES_KEY)) as f:
        RESULTS = json.load(f)
```

## Kinds of Synthetic Data
:label:`subsec_lm_synth_kinds`

Text written by a model enters training in four ways. A model can write new
documents from a seed, such as a topic, an outline or a web page: a textbook
chapter, a story, an exercise with its solution. It can rewrite existing
documents, as in that example. It can write demonstrations for post-training, the
requests and answers that teach a pretrained model to follow instructions.
And it can label data: the educational scores of
:numref:`sec_lm_quality_filtering` were assigned by Qwen3.8-27B, so the
classifier of that section learned from synthetic labels. This section
treats the first two kinds as pretraining data.

The two differ in what they can add. A rewrite that does its job changes the
form of a text but not its content, since the facts are those of the source.
A generation from a seed can add content, which is both its promise and its
risk, because the content comes from the generator's memory, errors
included.

Fluency proves nothing here; a fluent paragraph can carry a wrong number.
Generated data must be judged by its effect: how much a model trained with it
improves on held-out data, compared with a model trained without it at the
same cost. The best-known demonstration that generated text can help is
phi-1 (Gunasekar and colleagues, 2023), a code model of 1.3 billion
parameters trained on about 6 billion tokens of filtered web code and about 1
billion tokens of textbooks and exercises written by GPT-3.5. It reached
50.6% pass@1 on HumanEval, a strong score for its size. The result is real
and also narrow. It concerns short Python functions of the kind HumanEval
tests, the paper itself examines whether its generated exercises
resemble the benchmark, and it does not show that generated textbooks teach
a general language model more per token than good real text does. That
question needs a controlled comparison at equal cost, which we run below on
a small scale.

## Rephrasing the Web
:label:`subsec_lm_synth_rephrase`

Maini and colleagues (2024) call their recipe WRAP, for web rephrase
augmented pretraining. An off-the-shelf instruction-tuned model rewrites
chunks of web text in a few styles, among them "like Wikipedia" and as
questions and answers, and the language model trains on real and rephrased
text in equal parts. On C4, a crawl with plenty of noise left in it, they
report pretraining about three times faster, and at equal compute a
perplexity more than 10% lower on average over subsets of the Pile. They
explain the gain in two parts: the rewrites are cleaner than the web, and
their styles are closer to the text that evaluations use. The second part
makes a prediction that we can test, and we return to it when we evaluate.

Rephrasing has since reached the scale of whole crawls. Nemotron-CC (Su and
colleagues, 2024) sorts Common Crawl with quality classifiers, rewrites
low-quality documents into cleaner text and turns high-quality ones into
question-answer pairs and other forms. Its current release, Nemotron-CC-v2,
adds crawls from 2024 and 2025, and by its dataset card about 2.5 trillion of
its tokens were written by Mistral-NeMo-12B-Instruct and Qwen3-30B-A3B; the
card also lists DeepSeek and other Qwen models among its generators. It is
released under the NVIDIA Data Agreement for Model Training, so this book
cites it and ships none of it. FinePhrase (Niklaus and colleagues, 2026)
rewrites FineWeb-Edu, the source we use, into 486 billion tokens of FAQs,
tutorials, tables and mathematical word problems, generated by
SmolLM2-1.7B-Instruct and released under ODC-By. Its authors compare
prompts, generators and sources over more than a trillion generated tokens.
They find that generators larger than about a billion parameters add nothing
and report generation costs up to thirty times lower than earlier synthetic
datasets. That finding is why a rephraser of two billion parameters is a
reasonable choice here and not only a cheap one.

Our pipeline is the same in miniature.
Rephrasing has four steps: cut a document into chunks, wrap each chunk in a
prompt, generate, and filter the replies. The two prompts follow two of the
styles of Maini and colleagues, an encyclopedia passage and a conversation of
questions and answers. Both end with the same instruction, to preserve every
fact, name and number and to add nothing. Both also forbid Markdown. Small
instruction-tuned models like to answer with bold headings and bullet lists,
and an encyclopedia article has neither.

Chunks hold at most 400 tokens and end where a sentence ends, so the model
rewrites a passage it can keep in view and never half a sentence.
`rephrase_docs` sends batches of prompts of similar length through
Transformers' `generate` and records for each reply whether the model ended
its turn or ran out of tokens. A reply that ran out stops mid-sentence and is
dropped. One more detail matters for a Qwen model. Its chat template, not its
weights, decides whether the model first reasons in a `<think>` block
(:numref:`sec_lm_hf_models`). Reasoning text has no place in a pretraining
corpus, so the template is called with `enable_thinking=False`, and the
function asserts that no reply contains a thinking tag.

```{.python .input #synthetic-data-rephrase-defs}
%%tab pytorch
#@save
REPHRASE_PROMPTS = {  # style -> user prompt; {text} is the source chunk
    'wiki': (
        'Rewrite the following text as a clear, well-organized passage in '
        'the style of an encyclopedia article, in plain prose paragraphs '
        'without Markdown, headings or lists. Preserve every fact, name and '
        'number; add nothing that the text does not say. Reply with the '
        'passage only.\n\nText:\n{text}'),
    'qa': (
        'Rewrite the following text as a conversation of questions and '
        'answers in plain text without Markdown: each question on its own '
        'line starting with "Question:", each answer on its own line '
        'starting with "Answer:". Preserve every fact, name and number; add '
        'nothing that the text does not say. Reply with the conversation '
        'only.\n\nText:\n{text}')}

#@save
def _token_pieces(tokenizer):
    """The text of every token id decoded on its own, cached on the
    tokenizer (a byte of a multi-byte character decodes to U+FFFD)."""
    pieces = getattr(tokenizer, '_d2l_pieces', None)
    if pieces is None:
        pieces = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
        tokenizer._d2l_pieces = pieces
    return pieces

#@save
def chunk_doc(token_ids, tokenizer, max_tokens=400):
    """Split a document's token ids into chunks of at most max_tokens ids:
    sentences end at a line break or at ., ? or ! (closing quotes and
    brackets allowed) before white space; each chunk takes whole sentences
    while they fit, and a sentence longer than max_tokens is cut into
    pieces of max_tokens. The chunks concatenate to the document."""
    ids, pieces = [int(t) for t in token_ids], d2l._token_pieces(tokenizer)
    ends = []  # positions after which a sentence ends
    for k, t in enumerate(ids):
        p = pieces[t]
        after = pieces[ids[k + 1]] if k + 1 < len(ids) else ' '
        core = p.rstrip().rstrip('"\')]”’»')
        if k + 1 == len(ids) or '\n' in p or (core[-1:] in ('.', '?', '!')
                                              and (p[-1:].isspace() or
                                                   after[:1].isspace())):
            ends.append(k + 1)
    chunks, cur, start = [], [], 0
    for end in ends:
        sentence, start = ids[start:end], end
        if len(cur) + len(sentence) <= max_tokens:
            cur += sentence
            continue
        if cur:
            chunks.append(cur)
        while len(sentence) > max_tokens:  # an over-long sentence
            chunks.append(sentence[:max_tokens])
            sentence = sentence[max_tokens:]
        cur = sentence
    return chunks + [cur] if cur else chunks

#@save
def _synth_generate(model, tokenizer, prompts, max_new_tokens, batch,
                    temperature, top_p, seed, progress=None):
    """Generate a chat reply to every prompt (non-thinking template) in
    batches of similar length with left padding; batch k samples under the
    seed (seed, k). Returns (text, n_out, finish) per prompt in input
    order: finish is 'stop' when the model ended its turn and 'length'
    when max_new_tokens ran out first. progress: seconds between lines."""
    import transformers
    chat = [tokenizer.apply_chat_template(
        [{'role': 'user', 'content': p}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for p in prompts]
    ids = tokenizer(chat, add_special_tokens=False)['input_ids']
    stop = model.generation_config.eos_token_id
    stop = [stop] if isinstance(stop, int) else list(stop)
    pad = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
           else stop[0])
    sample = temperature > 0
    cfg = transformers.GenerationConfig(  # nothing from the model's config
        max_new_tokens=max_new_tokens, do_sample=sample,
        temperature=temperature if sample else None,
        top_p=top_p if sample else None, top_k=0 if sample else None,
        eos_token_id=stop, pad_token_id=pad)
    device = next(model.parameters()).device
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
    out, t0, shown, n_new = [None] * len(ids), time.time(), time.time(), 0
    for k, a in enumerate(range(0, len(order), batch)):
        rows = order[a:a + batch]
        T = max(len(ids[i]) for i in rows)
        X = torch.tensor([[pad] * (T - len(ids[i])) + ids[i] for i in rows],
                         device=device)
        mask = torch.tensor([[0] * (T - len(ids[i])) + [1] * len(ids[i])
                             for i in rows], device=device)
        torch.manual_seed(int(np.random.SeedSequence(
            [seed, k]).generate_state(1)[0]))
        with torch.inference_mode():
            G = model.generate(input_ids=X, attention_mask=mask,
                               generation_config=cfg)
        for i, new in zip(rows, G[:, T:].tolist()):
            n = next((j for j, t in enumerate(new) if t in stop), None)
            new = new if n is None else new[:n]
            out[i] = (tokenizer.decode(new).strip(), len(new),
                      'length' if n is None else 'stop')
            n_new += len(new)
        if progress and time.time() - shown > progress:
            shown = time.time()
            print(f'  {a + len(rows):,} of {len(ids):,} prompts, '
                  f'{n_new / (shown - t0):,.0f} new tokens/s')
    return out

#@save
class _SynthRows(list):
    """The rows rephrase_docs keeps; .dropped holds the rows it dropped."""
    dropped = ()

#@save
def rephrase_docs(chunks, style, model_id='Qwen/Qwen3.5-2B',
                  max_new_tokens=512, batch=32, temperature=0.7, top_p=0.9,
                  seed=0, backend='hf', *, model=None, progress=None):
    """Rewrite each text of chunks in a style: a key of REPHRASE_PROMPTS,
    a prompt template with {text}, or a list of either, one per chunk.
    Non-thinking chat generation by model_id, or by model=(model,
    tokenizer) loaded once by the caller; backend='vllm' generates with
    vLLM instead (model: a vllm.LLM; request i samples under seed + i).
    Returns dict(index, src, out, n_out, finish) rows in
    input order; rows that ran out of tokens (finish 'length') are dropped
    and kept in .dropped. A reply with thinking text fails an assertion."""
    styles = style if isinstance(style, (list, tuple)) else [style] * len(
        chunks)
    prompts = [d2l.REPHRASE_PROMPTS.get(s, s).format(text=c)
               for s, c in zip(styles, chunks)]
    if backend == 'hf':
        m, tok = model or d2l.load_hf(model_id)
        outs = d2l._synth_generate(m, tok, prompts, max_new_tokens, batch,
                                   temperature, top_p, seed, progress)
    elif backend == 'vllm':  # the serving path, if vLLM is installed
        from vllm import LLM, SamplingParams
        rev = d2l.HF_REVISIONS.get(model_id)
        llm = model or LLM(model_id, revision=rev, tokenizer_revision=rev,
                           dtype='bfloat16', seed=seed,
                           gpu_memory_utilization=0.5)  # a shared card
        replies = llm.chat(
            [[{'role': 'user', 'content': p}] for p in prompts],
            [SamplingParams(temperature=temperature, top_p=top_p,
                            max_tokens=max_new_tokens, seed=seed + i)
             for i in range(len(prompts))],
            chat_template_kwargs=dict(enable_thinking=False))
        outs = [(r.outputs[0].text.strip(), len(r.outputs[0].token_ids),
                 'length' if r.outputs[0].finish_reason == 'length'
                 else 'stop') for r in replies]
    else:
        raise ValueError(f'unknown backend {backend!r}')
    rows = d2l._SynthRows()
    rows.dropped = []
    for i, (c, (text, n, finish)) in enumerate(zip(chunks, outs)):
        assert '<think>' not in text and '</think>' not in text, (
            f'chunk {i}: thinking text in the reply')
        (rows.dropped if finish == 'length' else rows).append(dict(
            index=i, src=c, out=text, n_out=n, finish=finish))
    return rows
```

How can we tell a good rewrite from a bad one without reading it? Three
numbers are cheap. The *copy rate* is the Jaccard similarity of the two
texts' sets of character 5-grams: 1 for a copy, near 0 for a text that shares
no phrasing with its source. A copy adds nothing, so a high copy rate is a
warning. *Number fidelity* is the share of the source's digit strings that
reappear in the rewrite, a crude test of whether the facts survived. The
*length ratio* divides the length of the rewrite by that of the source, in
characters, and catches replies that were cut short or padded. For the
diversity of a set of texts we count the share of distinct word 4-grams
among all their 4-grams: 1 when no four-word phrase repeats, small when the
texts say the same thing in the same words.

```{.python .input #synthetic-data-metric-defs}
%%tab pytorch
#@save
def synth_metrics(src, out):
    """Fidelity of a rewrite out of src: copy_rate, the Jaccard similarity
    of their sets of character 5-grams (1 for a copy); num_fidelity, the
    share of src's distinct digit strings (7, 1,300, 3.14) that are digit
    strings of out (1 if src has none); len_ratio, len(out) / len(src) in
    characters."""
    grams = lambda t: {t[i:i + 5] for i in range(len(t) - 4)}
    nums = lambda t: set(re.findall(r'\d+(?:[.,]\d+)*', t))
    a, b, s = grams(src), grams(out), nums(src)
    return dict(copy_rate=len(a & b) / len(a | b) if a | b else float(
        src == out), num_fidelity=len(s & nums(out)) / len(s) if s else 1.0,
                len_ratio=len(out) / max(len(src), 1))

#@save
def distinct_n(texts, n=4):
    """The share of distinct word n-grams among all word n-grams of texts
    (NFC, lowercase, \\w+ words, as d2l.shingles); 0.0 without n-grams."""
    grams = [g for t in texts for g in d2l.shingles(t, n)]
    return len(set(grams)) / len(grams) if grams else 0.0
```

Now the example of the opening. The source is the first chunk of the
training split S1 with at least three numbers and at most 200 tokens. The
splits are cut from the FineWeb-Edu web pool of
:numref:`sec_lm_data_mixtures`: S1 and S2 hold $2^{21}$ tokens of chunks
each for training, and a third split holds $2^{19}$ tokens of chunks from
the pool's separate validation documents. After one rewrite we sample the
same prompt nine more times at temperature 0.7, the setting of every rewrite
in this section.

```{.python .input #synthetic-data-opening}
%%tab pytorch
ATTRIBUTION = ('Source text: FineWeb-Edu (HuggingFaceFW/fineweb-edu, '
               f'sample-10BT, ODC-By 1.0); rewrites by {REPHRASER} '
               '(Apache-2.0)')
KEYS = d2l.MIX_CONFIGS['t1']['keys']
WEB = d2l.load_pool(KEYS['web-20m'])
VAL = d2l.load_pool(KEYS['web-val-1m'], 'web-val-1m')
TOK = d2l.r2_tokenizer()
EOS = WEB.meta['eos_id']
for P in (WEB, VAL):  # the reference tokenizer made both pools
    assert P.meta['tokenizer_sha256'] == d2l.R2_TOKENIZER_SHA256

def documents(pool):
    """(doc_id, token ids without the EOS) of every document of a pool."""
    root = pathlib.Path(pool.ids.filename).parent
    with gzip.open(root / pool.meta['docs'], 'rt') as f:
        ids = [json.loads(line)['doc_id'] for line in f]
    ends = np.flatnonzero(np.asarray(pool.ids) == EOS)
    assert len(ends) == len(ids) == pool.meta['n_docs']
    return [(i, pool.ids[a:b]) for i, a, b in zip(ids, np.r_[0, ends[:-1]
                                                               + 1], ends)]

def take_chunks(docs, budgets):
    """Consecutive runs of the documents' chunks, in order, one run per
    (split, budget): a run ends with the first chunk at which its chunks,
    each with its EOS, hold more than budget tokens, so a pool of them
    has budget // seq_len windows."""
    runs, cur, n = {}, [], 0
    for doc_id, ids in docs:
        for c in d2l.chunk_doc(ids, TOK, CHUNK_TOKENS):
            split, budget = budgets[len(runs)]
            cur.append(dict(chunk_id=f'{split}-{len(cur):05d}',
                            doc_id=doc_id, ids=c))
            n += len(c) + 1
            if n > budget:
                runs[split], cur, n = cur, [], 0
                if len(runs) == len(budgets):
                    return runs
    raise ValueError(f'fewer than {budgets} tokens in the pool')

DOCS = documents(WEB)
SPLITS = {**take_chunks(DOCS, [('S1', U), ('S2', U)]),
          **take_chunks(documents(VAL), [('val', VAL_U)])}
S1, S2, SV = SPLITS.values()
N_DOCS = {k: len({c['doc_id'] for c in v}) for k, v in SPLITS.items()}
print(pd.DataFrame([dict(split=k, pool='web val' if k == 'val' else 'web',
                         documents=N_DOCS[k], chunks=len(v),
                         tokens=sum(len(c['ids']) + 1 for c in v))
                    for k, v in SPLITS.items()]).to_string(index=False))

def side_by_side(a, b, width=46, head=('source', 'rewrite')):
    """Two texts in two columns, lines broken at width characters."""
    A, B = (textwrap.wrap(t.replace('\n', ' '), width) for t in (a, b))
    print(f'{head[0]:{width}} | {head[1]}')
    for x, y in itertools.zip_longest(A, B, fillvalue=''):
        print(f'{x:{width}} | {y}')

numbers = lambda t: set(re.findall(r'\d+(?:[.,]\d+)*', t))
OPEN = next(c for c in S1 if len(c['ids']) <= OPEN_MAX_SRC and len(
    numbers(TOK.decode(c['ids']))) >= 3)
OPEN_SRC = TOK.decode(OPEN['ids'])
GEN_MODEL = d2l.load_hf(REPHRASER, dtype=DTYPE)
one = d2l.rephrase_docs([OPEN_SRC], 'wiki', model=GEN_MODEL,
                        max_new_tokens=NEW_TOKENS, **SAMPLING)
OPEN_ROW = (list(one) + one.dropped)[0]
say(f'\n{OPEN["chunk_id"]} ({len(OPEN["ids"])} tokens), finish '
    f'{OPEN_ROW["finish"]}:')
side_by_side(OPEN_SRC, OPEN_ROW['out'])
say({k: round(v, 3) for k, v in d2l.synth_metrics(
  OPEN_SRC, OPEN_ROW['out']).items()})
many = d2l.rephrase_docs([OPEN_SRC] * N_SAMPLES, 'wiki', model=GEN_MODEL,
                         max_new_tokens=NEW_TOKENS, batch=N_SAMPLES,
                         **SAMPLING)
SAMPLES = [r['out'] for r in sorted(list(many) + many.dropped,
                                    key=lambda r: r['index'])]
say(f'\n{N_SAMPLES} samples of the same prompt at temperature '
    f'{SAMPLING["temperature"]}: distinct 4-grams '
    f'{d2l.distinct_n(SAMPLES, 4):.3f} (one sample alone: '
    f'{d2l.distinct_n(SAMPLES[:1], 4):.3f}); their first words:')
for s in SAMPLES[:3]:
    say('  ', textwrap.shorten(s, 100))
say(ATTRIBUTION)
```

The rewrite is what one hopes for. It keeps the fourfold effectiveness, the
48 of 50 patients and the sixteen complaints about the taste, and it reads
more like an encyclopedia than its source does. Yet its number fidelity is
0.67, because the model wrote "approximately 30 minutes" as "approximately
thirty minutes" and a check on digit strings counts that as a lost number.
The cheap check is wrong here, in the harmless direction. We will meet
rewrites where it is wrong in the other one.

The nine samples show the second cost. They differ in a word here and there,
but the three printed open with the same ten words, and fewer than half of
the nine samples' 4-grams are distinct. Temperature 0.7 gives the model
enough freedom to vary a word and not enough to vary a text. An
instruction-tuned model has one preferred answer to a prompt, and sampling
explores the neighbourhood of that answer.

Before generating at scale, we check the chunker on twenty documents of the
pool.

```{.python .input #synthetic-data-chunk-check}
%%tab pytorch
rng = np.random.default_rng(0)
lengths, full = [], 0
for i in rng.choice(len(DOCS), 20, replace=False):
    ids = DOCS[i][1].tolist()
    chunks = d2l.chunk_doc(ids, TOK, CHUNK_TOKENS)
    assert sum(chunks, []) == ids  # the chunks are the document
    assert TOK.decode(sum(chunks, [])) == TOK.decode(ids)
    lengths += [len(c) for c in chunks]
    full += sum(len(c) == CHUNK_TOKENS for c in chunks)
assert max(lengths) <= CHUNK_TOKENS
assert all(len(c['ids']) <= CHUNK_TOKENS for v in SPLITS.values() for c in v)
say(f'20 pool documents (seed 0) -> {len(lengths)} chunks, each the '
    f'document\'s tokens in order; median {np.median(lengths):.0f} tokens, '
    f'longest {max(lengths)}; {full} of exactly {CHUNK_TOKENS} (a cut '
    'inside an over-long sentence, or a sentence that fit exactly)')
```

Every document is recovered exactly from its chunks, so chunking loses
nothing, and the median chunk holds 382 of the 400 tokens allowed.

Now sixty chunks of S1 are rewritten in each style, in batches of 32 with up
to 512 new tokens. Two rules decide which replies are kept: a reply that ran
out of tokens is dropped, and so is one whose length ratio lies outside
$[0.5, 1.5]$, which catches refusals, replies cut short and padded ones. Copy
rate and number fidelity only flag a reply. They are reported and not used
to select, because we have just seen number fidelity misjudge a good rewrite,
and a corpus selected by a misjudging metric inherits its errors. Every
threshold is a parameter of the pipeline, recorded in its manifest, not a
claim about what makes a good rewrite.

```{.python .input #synthetic-data-live-120}
%%tab pytorch
pool = [c for c in S1 if len(c['ids']) <= LIVE_MAX_SRC]
LIVE = [pool[i] for i in sorted(np.random.default_rng(0).choice(
    len(pool), N_LIVE, replace=False))]
texts, rows = [TOK.decode(c['ids']) for c in LIVE], []
for style in d2l.REPHRASE_PROMPTS:
    t0 = time.time()
    got = d2l.rephrase_docs(texts, style, model=GEN_MODEL, batch=LIVE_BATCH,
                            max_new_tokens=NEW_TOKENS, **SAMPLING)
    say(f'{style}: {len(texts)} chunks in {time.time() - t0:.0f} s, '
        f'{len(got.dropped)} dropped (out of tokens)')
    rows += [dict(style=style, chunk_id=LIVE[r['index']]['chunk_id'],
                  finish=r['finish'], n_out=r['n_out'], src=r['src'],
                  out=r['out'], **d2l.synth_metrics(r['src'], r['out']))
             for r in list(got) + got.dropped]
say(d2l.lm_peak(), '| no thinking text in any of the', len(rows),
    'replies (asserted)')

def drop_reason(df):
    """Why a row is dropped: 'length' (out of tokens), 'len_ratio' (its
    length ratio is outside LEN_RATIO) or '' (kept)."""
    inside = df.len_ratio.between(*LEN_RATIO)
    return pd.Series(np.where(df.finish == 'length', 'length', np.where(
        inside, '', 'len_ratio')), index=df.index)

def flags(df):
    """The flags of FLAGS as boolean columns (flagged rows are kept)."""
    return pd.DataFrame(dict(
        copy=df.copy_rate > FLAGS['copy_rate'],
        numbers=df.num_fidelity < FLAGS['num_fidelity']), index=df.index)

LIVE_ROWS = pd.DataFrame(rows)
LIVE_ROWS['drop'] = drop_reason(LIVE_ROWS)
KEPT = LIVE_ROWS[LIVE_ROWS['drop'] == '']
assert (KEPT.out != '').all()
F_LIVE = flags(KEPT)
STYLE_TABLE = pd.DataFrame([dict(
    style=s, prompt_sha256=hashlib.sha256(
        d2l.REPHRASE_PROMPTS[s].encode()).hexdigest()[:12],
    kept=int((g['drop'] == '').sum()),
    out_of_tokens=int((g['drop'] == 'length').sum()),
    len_ratio_out=int((g['drop'] == 'len_ratio').sum()),
    len_ratio=KEPT.loc[g.index[g['drop'] == '']].len_ratio.mean(),
    copy_rate=KEPT.loc[g.index[g['drop'] == '']].copy_rate.mean(),
    all_numbers=(KEPT.loc[g.index[g['drop'] == '']].num_fidelity == 1
                 ).mean(),
    flagged=F_LIVE.loc[g.index[g['drop'] == '']].any(axis=1).mean())
    for s, g in LIVE_ROWS.groupby('style', sort=False)])
print(STYLE_TABLE.round(3).to_string(index=False))
fig, axes = d2l.plt.subplots(1, 2, figsize=(7, 2.6))
for ax, col, thr, lab in zip(axes, ('copy_rate', 'num_fidelity'),
                             (FLAGS['copy_rate'], FLAGS['num_fidelity']),
                             ('copy rate', 'number fidelity')):
    for style, g in KEPT.groupby('style', sort=False):
        ax.hist(g[col], bins=np.linspace(0, 1, 21), alpha=0.6, label=style)
    ax.axvline(thr, color='black', linestyle=':', linewidth=1)
    ax.set_xlabel(lab)
axes[0].set_ylabel('rewrites')
axes[0].legend()
fig.tight_layout()
pick = {'worst number fidelity': KEPT.num_fidelity.idxmin(),
        'highest copy rate': KEPT.copy_rate.idxmax(),
        'lowest copy rate': KEPT.copy_rate.idxmin()}
rest = [i for i in KEPT.index if i not in pick.values()]
for k, i in enumerate(np.random.default_rng(1).choice(
        rest, min(2, len(rest)), replace=False)):
    pick[f'random {k + 1}'] = i
for name, i in pick.items():
    r = KEPT.loc[i]
    say(f'\n{name}: {r.chunk_id}, {r["style"]}, copy {r.copy_rate:.2f}, '
        f'numbers {r.num_fidelity:.2f}, length ratio {r.len_ratio:.2f}')
    side_by_side(textwrap.shorten(r.src, 420), textwrap.shorten(r.out, 420))
say(ATTRIBUTION)
```

The two styles behave differently. The encyclopedia style is well behaved:
all but about one of the 60 replies are kept, and the rewrites are about as
long as their sources. The conversations are longer than their sources.
About a quarter of them ran out of the 512 new tokens before they ended, and
with those that fell outside the length range, only a little over half
survive. These counts come from one sampled run, and on other hardware they
move by a few. A style that expands its source needs a
larger budget than one that restates it, and no prompt can change that.

The histograms show what the flags see. Copy rates spread from about 0.2
to 1. The near-copies above 0.8 include chunks with little to rewrite, such as
the one-sentence medical disclaimer printed above. Number fidelity is 1 for
about three in five of the encyclopedia rewrites and for about half of the
conversations.
The rewrite with the lowest copy rate turns a blog post about ghee into an
encyclopedia entry and drops its notice about affiliate links, which is what
we want from a rewrite. It also scores 0 on number fidelity, and only
reading it can tell whether a fact was lost. Cheap checks are alarms, not
verdicts.

### Generating from Seeds
:label:`subsec_lm_synth_seeds`

Rewriting is bound to its source. Generation from a seed is its freer
relative: the source only supplies a topic, and the model writes a new text
about it. Cosmopedia, over 25 billion tokens written by Mixtral-8x7B-Instruct
and released under Apache-2.0, is the best-documented example. Its prompts
turn three knobs: a seed text (a web page, a course outline, a textbook
passage), a format (a textbook, a blog post, a story) and an audience (young
children, college students, researchers). Its dataset card states the
reason: varying the audience and the style of the prompt raises the
diversity of the output. We print three of its rows and then test the claim
on a small scale. From the opening's chunk as seed we generate nine texts with
one plain prompt and nine with three formats times three audiences, and
compare both with the nine rephrasings of the opening. Each output is cut to
the length of the shortest, because the share of distinct n-grams falls as
texts grow longer. Diversity is measured twice, by distinct 4-grams and by
the mean Jaccard similarity of the word 5-gram sets of all pairs, which is
the measure of near-duplication of :numref:`sec_lm_dedup`.

```{.python .input #synthetic-data-seeds}
%%tab pytorch
COSMO_FILES = {  # first file of each configuration, at the pinned revision
    'web_samples_v2': 'data/web_samples_v2/train-00000-of-00118.parquet',
    'stanford': 'data/stanford/train-00000-of-00013.parquet',
    'stories': 'data/stories/train-00000-of-00043.parquet'}
COSMO_ROWS = None if RESULTS is None else RESULTS['cosmopedia']
if 'synth-arms' in RERUN:  # registered with the results of the runs
    _, repo, rev, _ = d2l.LM_DATA_SOURCES['cosmopedia']
    COSMO_ROWS = [dict(config=c, row=0, **{
        col: d2l.lm_hub_rows(repo, path, rev, 0, 1, col)[0][0]
        for col in ('format', 'audience', 'seed_data', 'text_token_length')},
                       text=d2l.lm_hub_rows(repo, path, rev, 0, 1)[0][0][:200])
                  for c, path in COSMO_FILES.items()]
if COSMO_ROWS is None:
    say(f'Cosmopedia rows: registered with {RES_KEY} (pending)')
else:
    for r in COSMO_ROWS:
        say(f'cosmopedia/{r["config"]} row {r["row"]}: format '
            f'{r["format"]!r}, audience {r["audience"]!r}, seed data '
            f'{r["seed_data"]!r}, {r["text_token_length"]} tokens:')
        say('  ', repr(r['text'][:200]))
    say('Cosmopedia (HuggingFaceTB/cosmopedia, Apache-2.0)')
FORMATS = ('textbook section', 'blog post', 'short story')
AUDIENCES = ('young children', 'high-school students', 'professionals')
COSMO_STYLE = ('Write a {fmt} for {aud} about the topic of the extract '
               'below. Use your own words and examples; do not copy the '
               'extract.\n\nExtract:\n{{text}}')
PLAIN = ('Write a text about the topic of the extract below. Use your own '
         'words and examples; do not copy the extract.\n\nExtract:\n{text}')
styled = [COSMO_STYLE.format(fmt=f, aud=a) for f in FORMATS
          for a in AUDIENCES]
outs = {}
for name, style in (('one plain prompt', [PLAIN] * 9),
                    ('3 formats x 3 audiences', styled)):
    got = d2l.rephrase_docs([OPEN_SRC] * 9, style, model=GEN_MODEL, batch=9,
                            max_new_tokens=SEED_TOKENS, **SAMPLING)
    outs[name] = [r['out'] for r in sorted(list(got) + got.dropped,
                                           key=lambda r: r['index'])]
outs['rephrasings (opening)'] = SAMPLES
K = min(len(d2l.shingles(t, 1)) for v in outs.values() for t in v)
assert K >= MIN_WORDS, f'an output of {K} words: a refusal or an empty reply'
cut = {k: [' '.join(d2l.shingles(t, 1)[:K]) for t in v]
       for k, v in outs.items()}

def mean_jaccard(texts):
    """Mean Jaccard similarity of the word 5-gram sets of all pairs."""
    sets = [set(d2l.shingles(t, 5)) for t in texts]
    return np.mean([d2l.jaccard_exact(a, b)
                    for a, b in itertools.combinations(sets, 2)])

DIVERSITY = pd.DataFrame([dict(setting=k, outputs=len(v),
                               distinct_4=d2l.distinct_n(v, 4),
                               jaccard_5=mean_jaccard(v))
                          for k, v in cut.items()])
say(f'\none seed ({OPEN["chunk_id"]}), every output cut to its first {K} '
    f'words, temperature {SAMPLING["temperature"]}:')
print(DIVERSITY.round(3).to_string(index=False))
gain = DIVERSITY.distinct_4[1] - DIVERSITY.distinct_4[0]
say('conditioning on format and audience', 'raises distinct 4-grams by '
    f'{gain:.3f}' if gain >= 0.02 else 'lowers distinct 4-grams by '
    f'{-gain:.3f}' if gain <= -0.02 else
    f'leaves distinct 4-grams about equal ({gain:+.3f})')
for k in ('one plain prompt', '3 formats x 3 audiences'):
    say(f'{k}, first output:', textwrap.shorten(outs[k][0], 160))
```

The prompt knobs work. Cut to 70 words each, the nine conditioned texts
share almost no 4-gram (0.995 distinct, against about 0.76 for the plain
prompt), and the mean Jaccard similarity of their pairs is zero to three
decimals. All three settings sample at the same temperature, so the gain
comes from the prompt alone. The plain prompt already varies more than the
rephrasings (about 0.44), because a request to write about a topic leaves
the model room that a request to rewrite does not. That room is the point of
generation and also its risk. The first conditioned output is a textbook
section for young children about "the magic of mouthwash for stomach pain";
the mouthwash of the source treated pain in the mouth. Diversity from the
prompt is diversity of form. Whether the content stays true is a separate
question, and the generator answers it from its own memory.

### Checking Generated Text
:label:`subsec_lm_synth_checks`

Sixty live rewrites are too few to train on. The registered corpus holds the
encyclopedia-style rewrites of every chunk of S1, S2 and the validation
split, generated once with the same prompt, model and sampling settings (the
next cell shows how). We load it, verify that its files are what this
notebook would write, and ask it the questions that matter at scale. How many
rewrites were dropped, and why? How often are numbers kept? Do the rewrites
repeat each other more than their sources do? Generators are known to fall
back on the same phrasings, and the MinHash of :numref:`sec_lm_dedup` finds
pairs of chunks whose shingle sets have Jaccard similarity above 0.8; we use
the bands and rows that suit that threshold for 128 hash values,
$(b, r) = (9, 13)$. And did the generator reproduce benchmark items it saw in
its own training? A rewrite could bring in contamination that its source
never had, so we search sources and rewrites alike for 13-gram hits in the
index of GSM8K, ARC-Challenge and MMLU built in the same section.

```{.python .input #synthetic-data-corpus-checks}
%%tab pytorch
GEN_ID = dict(  # how the corpus is generated; every row records it
    model_id=REPHRASER, model_revision=d2l.HF_REVISIONS[REPHRASER],
    style='wiki', prompt_sha256=hashlib.sha256(
        d2l.REPHRASE_PROMPTS['wiki'].encode()).hexdigest(),
    max_new_tokens=REGEN_TOKENS, chunk_tokens=CHUNK_TOKENS,
    batch=REGEN_BATCH, **SAMPLING)
NAMES = {('S1', 'src'): 'train_src_S1', ('S1', 'reph'): 'train_reph_S1',
         ('S2', 'src'): 'train_src_S2', ('S2', 'reph'): 'train_reph_S2',
         ('val', 'src'): 'val_src', ('val', 'reph'): 'val_reph'}

def open_corpus(tar):
    """The extracted corpus: its folder, meta.json and rephrase.jsonl;
    extracted afresh whenever the archive's sha1 changed."""
    root, stamp = pathlib.Path(tar).with_suffix(''), None
    with open(tar, 'rb') as f:
        sha1 = hashlib.file_digest(f, 'sha1').hexdigest()
    if (root / '.extracted').exists():
        stamp = (root / '.extracted').read_text()
    if stamp != sha1:  # never extracted, or another archive at this path
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir()
        d2l.extract(str(tar), str(root))
        (root / '.extracted').write_text(sha1)
    with open(root / 'meta.json') as f:
        meta = json.load(f)
    with open(root / 'rephrase.jsonl', encoding='utf-8') as f:
        rows = pd.DataFrame([json.loads(line) for line in f])
    return root, meta, rows

def chunk_tokens(texts):
    """The token file of chunks: each chunk's ids, then the EOS."""
    return np.array([t for ids in d2l._doc_token_ids(TOK, texts)
                     for t in ids + [EOS]], dtype=np.uint16)

def corpus_report(root, meta, rows):
    """Checks and summaries of a rephrased corpus (asserts its files)."""
    kept = rows[rows.kept]
    for k, v in GEN_ID.items():  # made by this notebook's generation
        assert (rows[k] == v).all(), f'the corpus has another {k}'
    assert not rows.out_text.str.contains('<think>|</think>').any()
    assert (rows['drop'] == drop_reason(rows)).all()  # the length rules
    assert (rows.kept == (rows['drop'] == '')).all()
    assert (kept.out_text != '').all()
    table = []
    for (split, kind), name in NAMES.items():
        P = d2l.load_pool(root, name)
        ids = np.asarray(P.ids)
        if kind == 'src':  # the chunks of this notebook's pools, exactly
            want = np.array([t for c in SPLITS[split] for t in c['ids']
                             + [EOS]], dtype=np.uint16)
        else:  # the kept rewrites, tokenized as text
            want = chunk_tokens(kept[kept.split == split].out_text.tolist())
        assert np.array_equal(ids, want), name
        assert int((ids < 32).sum()) == int((ids == EOS).sum()) == P.meta[
            'n_docs']  # no special id inside a chunk
        table.append(dict(file=name, chunks=P.meta['n_docs'],
                          tokens=P.meta['n_tokens'],
                          bytes=P.meta['n_bytes_nfc'],
                          bytes_per_token=round(P.meta['n_bytes_nfc']
                                                / P.meta['n_tokens'], 2)))
    print(pd.DataFrame(table).to_string(index=False))
    for split in SPLITS:
        src, reph = (meta[f'{NAMES[split, k]}.bin']['n_tokens']
                     for k in ('src', 'reph'))
        say(f'{split}: rewrites / sources = {reph / src:.3f} in tokens')
    say(f'kept and dropped of {len(rows):,} chunks (length: out of '
        f'tokens; len_ratio: outside {LEN_RATIO}):')
    print(pd.crosstab(rows['split'], rows['drop'].replace('', 'kept'))
          .reindex(list(SPLITS)).to_string())
    fl = flags(kept)
    print(pd.DataFrame([dict(
        split=s, kept=len(g), copy_rate=g.copy_rate.mean(),
        all_numbers=(g.num_fidelity == 1).mean(),
        len_ratio_median=g.len_ratio.median(), **{
            f'flag_{k}': fl.loc[g.index, k].mean() for k in fl})
        for s, g in kept.groupby('split', sort=False)]).round(3).to_string(
            index=False))
    br = d2l.best_br(128, 0.8)
    assert br == (9, 13)
    dup = {}
    for kind, texts in (('sources', kept.src_text), ('rewrites',
                                                     kept.out_text)):
        sh = [d2l.shingles(t, 5) for t in texts]
        sigs = [d2l.minhash_signature(s) for s in sh]
        pairs, _ = d2l.lsh_candidates(sigs, *br)
        near = [(i, j) for i, j in pairs
                if d2l.jaccard_exact(sh[i], sh[j]) >= 0.8]
        dup[kind] = (len({i for p in near for i in p}) / len(texts),
                     len(near))
    say('near-duplicates (MinHash at (b, r) = (9, 13), verified Jaccard '
        '>= 0.8): ' + '; '.join(f'{k} {r:.2%} of chunks ({n} pairs)'
                                for k, (r, n) in dup.items()))
    index = d2l.NgramIndex(13)
    for it in d2l.benchmark_items():
        index.add(it['text'], it['benchmark'])
    hits = {k: [index.query(t) for t in texts] for k, texts in (
        ('sources', kept.src_text), ('rewrites', kept.out_text))}
    new = [h for h, s in zip(hits['rewrites'], hits['sources'])
           if h and not s]
    say('13-gram hits on GSM8K, ARC-Challenge, MMLU: ' + ', '.join(
      f'{k} {sum(map(bool, v))} chunks' for k, v in hits.items())
        + f'; rewrites with a hit their source lacks: {len(new)} '
        f'{collections.Counter(b for h in new for b, _ in h)}')
    return dict(dup=dup, hits={k: sum(map(bool, v))
                               for k, v in hits.items()})

CORPUS = None  # (folder, meta, rows) of the rephrased corpus
if TAR_KEY in d2l.DATA_HUB:
    CORPUS = open_corpus(d2l.download_keyed(TAR_KEY))
    REPORT = corpus_report(*CORPUS)
    say(ATTRIBUTION)
else:
    say(f'{TAR_KEY} is not registered yet: the regeneration below writes '
        'it')
```

Of the 14,305 chunks, 13,809 rewrites are kept: 77 ran out of tokens and 419
fell outside the length range. In characters the median rewrite is as long
as its source, yet the rewrites take about 9% fewer tokens. A token
of a rewrite covers about 4.8 bytes against 4.5 for the source, a sign of
plainer, more common words, which the tokenizer encodes in larger pieces.
About 70% of the kept rewrites keep every number, in all three splits alike,
and about a quarter have a copy rate above 0.8.

The generator does not repeat itself across chunks. Two pairs of sources are
near-duplicates and no pair of rewrites is: at temperature 0.7 with a
different source each time, the rewrites stay as distinct as their sources.
The repetition of the opening came from nine samples of one prompt, not from
one sample each of many prompts. One source chunk hits a benchmark 13-gram
and no rewrite does, so the generator added no contamination to this corpus.
That has to be checked for every corpus, not assumed.

The corpus was generated by setting `D2L_RERUN` to `reph-s1`, `reph-s2` and
`reph-val`, at batch 96 on one RTX 4090: about twenty minutes for each
training split and six for validation. S1 holds the chunks of the first
documents of the pool up to $2^{21}$ tokens, S2 the next $2^{21}$, and the
validation chunks come from the pool's validation documents. Each part
writes its own rows, and the last one assembles token files, rows and
dataset card into one archive.

```{.python .input #synthetic-data-regen-rephrase}
%%tab pytorch
PARTS = {'reph-s1': 'S1', 'reph-s2': 'S2', 'reph-val': 'val'}
for name in (n for n in PARTS if n in RERUN):
    split = PARTS[name]
    texts, t0 = [TOK.decode(c['ids']) for c in SPLITS[split]], time.time()
    got = d2l.rephrase_docs(texts, 'wiki', model=GEN_MODEL,
                            batch=REGEN_BATCH, max_new_tokens=REGEN_TOKENS,
                            progress=60, **SAMPLING)
    secs = time.time() - t0
    part = sorted(list(got) + got.dropped, key=lambda r: r['index'])
    recs = [dict(chunk_id=SPLITS[split][r['index']]['chunk_id'], split=split,
                 src_doc_id=SPLITS[split][r['index']]['doc_id'],
                 n_src=len(SPLITS[split][r['index']]['ids']),
                 n_out=r['n_out'], finish=r['finish'], src_text=r['src'],
                 out_text=r['out'], **d2l.synth_metrics(r['src'], r['out']),
                 **GEN_ID) for r in part]
    for rec, why in zip(recs, drop_reason(pd.DataFrame(recs))):
        rec.update(drop=str(why), kept=str(why) == '')

    def write_part(path, recs=recs):
        with gzip.GzipFile(path, 'wb', mtime=0) as f:
            f.write(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False)
                            + '\n' for r in recs).encode('utf-8'))

    d2l.lm_save_rerun(ARTIFACTS[name][1], write_part)
    say(f'{name}: {len(texts):,} chunks in {secs / 60:.1f} min, '
        f'{sum(r["n_out"] for r in part) / secs:,.0f} new tokens/s, '
        f'{len(got.dropped)} dropped; {d2l.lm_peak()}')
del GEN_MODEL
d2l.lm_free()

def read_part(name):
    with gzip.open(os.path.join(RERUN_DIR, ARTIFACTS[name][1]), 'rt',
                   encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_tar(path):
    """The archive: six token files, meta.json, rephrase.jsonl, README.md;
    byte-identical for identical contents."""
    rows = [r for name in PARTS for r in read_part(name)]
    for r in rows:  # all three parts come from this generation
        assert all(r[k] == v for k, v in GEN_ID.items()), r['chunk_id']
    members, meta = {}, {}
    for (split, kind), name in NAMES.items():
        part = [r for r in rows if r['split'] == split]
        assert len(part) == len(SPLITS[split]), (split, len(part))
        if kind == 'src':
            for r, c in zip(part, SPLITS[split]):
                assert r['chunk_id'] == c['chunk_id']
            ids = np.array([t for c in SPLITS[split] for t in c['ids']
                            + [EOS]], dtype=np.uint16)
            texts = [r['src_text'] for r in part]
        else:
            texts = [r['out_text'] for r in part if r['kept']]
            ids = chunk_tokens(texts)
        data = ids.astype('<u2').tobytes()
        members[f'{name}.bin'] = data
        meta[f'{name}.bin'] = dict(
            split=split, kind=kind, n_tokens=len(ids), n_docs=len(texts),
            n_bytes_nfc=sum(len(t.encode('utf-8')) for t in texts),
            dtype='uint16', eos_id=EOS, vocab_size=len(TOK),
            tokenizer_sha256=d2l.R2_TOKENIZER_SHA256,
            sha256=hashlib.sha256(data).hexdigest(),
            source=WEB.meta['source'], revision=WEB.meta['revision'],
            license='odc-by', generator=None if kind == 'src' else
            f'{REPHRASER}@{d2l.HF_REVISIONS[REPHRASER]}')
    members['meta.json'] = json.dumps(meta, indent=1, sort_keys=True).encode()
    members['rephrase.jsonl'] = ''.join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n'
        for r in rows).encode('utf-8')
    members['README.md'] = (
        f'# {TAR_KEY}\n\nFineWeb-Edu chunks (HuggingFaceFW/fineweb-edu, '
        f'sample-10BT @ {WEB.meta["revision"][:12]}, ODC-By 1.0) and their '
        f'wiki-style rewrites by {REPHRASER} @ '
        f'{d2l.HF_REVISIONS[REPHRASER][:12]} (Apache-2.0), generated for '
        'Dive into Deep Learning, sec_lm_synthetic_data. Rewrites are '
        'synthetic text; their rights follow the source and the '
        'generator. As the FineWeb-Edu card states, the use of the source '
        "is also subject to Common Crawl's Terms of Use "
        '(https://commoncrawl.org/terms-of-use). Rewrites that ran out of '
        f'tokens or whose length ratio is outside {LEN_RATIO} are kept in '
        'rephrase.jsonl (kept: false, drop: the reason) and left out of '
        'the token files.\n').encode()
    with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as t:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            t.addfile(info, io.BytesIO(data))

TAR_BUILT = bool(RERUN & set(PARTS)) and all(os.path.exists(os.path.join(
    RERUN_DIR, ARTIFACTS[n][1])) for n in PARTS)
if TAR_BUILT:
    tar = d2l.lm_save_rerun(TAR_FILE, write_tar)
    CORPUS = open_corpus(tar)
    REPORT = corpus_report(*CORPUS)
elif RERUN & set(PARTS):
    say('the archive is assembled once all three parts are written')
```

Number fidelity catches lost numbers and nothing else. A rewrite can keep
every digit and still change a name or add a claim that its source never
made, and no string check sees that. A second model can, at a price. Our
judge is Qwen3.8-27B, the model that labelled documents in
:numref:`sec_lm_quality_filtering`, in a 4-bit build served by vLLM, one
server per GPU. It is a much larger model than the rephraser and a
different one, so it does not grade its own writing; it is of the same
family, though, which is worth keeping in mind. For sixty pairs of S1 it
reads source and rewrite and answers one question: does the rewrite state a
claim that the source does not support? The answer is a line of JSON that
also quotes the first such claim, so that a reader can check every verdict.

The judge reasons before it answers, and we let it. Its replies are greedy,
limited to 12,000 new tokens within a context of 16,384, with no separate
budget for the reasoning. The verdicts are registered; setting `D2L_RERUN`
to `reph-judge` regenerates them, with one judge server per GPU.

```{.python .input #synthetic-data-judge}
%%tab pytorch
SYNTH_JUDGE_V2 = (
    'Below are a source text and a rewrite of it.\n\nSource:\n{src}\n\n'
    'Rewrite:\n{out}\n\nDoes the rewrite state any claim (a fact, name, '
    'number, date or relation) that the source does not support? Answer '
    'with one line of JSON and nothing else: {{"answer": "Yes" or "No", '
    '"claim": "<the first unsupported claim, quoted from the rewrite, or '
    'empty>"}}')
JUDGE_SHA = hashlib.sha256(SYNTH_JUDGE_V2.encode()).hexdigest()
JUDGE_PROTOCOL = dict(thinking=True, temperature=0, seed=0,
                      max_tokens=JUDGE_MAX_TOKENS, thinking_budget=None,
                      per_server=JUDGE_PER_SERVER,
                      vllm=d2l.EDU_LABELLER_LOCAL['vllm'])

def judge_pairs(rows):
    """The N_JUDGE kept (source, rewrite) pairs of S1 the judge reads."""
    s1 = rows[(rows.split == 'S1') & rows.kept].reset_index(drop=True)
    return s1.iloc[np.sort(np.random.default_rng(0).choice(
        len(s1), min(N_JUDGE, len(s1)), replace=False))]

def judge_id(pairs):
    """What verdicts on these pairs record: judge, prompt, protocol, pairs."""
    return dict(model=JUDGE, revision=PINS[JUDGE], prompt_sha256=JUDGE_SHA,
                protocol=JUDGE_PROTOCOL, chunk_ids=pairs.chunk_id.tolist(),
                pairs_sha256=d2l.docs_hash(list(zip(
                    pairs.chunk_id, pairs.src_text + '\0' + pairs.out_text))))

def judged(part, pairs):
    """Whether a judge part holds verdicts on exactly these pairs."""
    return part is not None and all(part.get(k) == v
                                    for k, v in judge_id(pairs).items())

S1_PART = os.path.join(RERUN_DIR, ARTIFACTS['reph-s1'][1])
if CORPUS is not None and (TAR_BUILT or not RERUN & set(PARTS)):
    PAIRS = judge_pairs(CORPUS[2])  # the registered or just-built corpus
elif os.path.exists(S1_PART):  # S1 regenerated, the archive not yet built
    PAIRS = judge_pairs(pd.DataFrame(read_part('reph-s1')))
else:
    PAIRS = None
JUDGE_PART = None

def judge_claims(urls, pairs):
    """The judge's answer ('Yes', 'No', or None without a verdict) and the
    claim it names for each (source, rewrite) pair: chat completions with
    thinking on, greedy, at most JUDGE_MAX_TOKENS new tokens and no
    thinking budget. Server k judges pairs k, k + len(urls), ... one at a
    time, so a pair's reply does not depend on the others'."""
    def one(i, url):
        src, out = pairs[i]
        d = d2l._vllm_post(url, '/chat/completions', dict(
            model=JUDGE, temperature=0, seed=0, max_tokens=JUDGE_MAX_TOKENS,
            messages=[{'role': 'user', 'content': SYNTH_JUDGE_V2.format(
                src=src, out=out)}],
            chat_template_kwargs={'enable_thinking': True}), timeout=3600.0)
        c = d['choices'][0]
        text = (c['message'].get('content') or '').split('</think>')[-1]
        try:
            v = json.loads(re.findall(r'\{[^{}]*\}', text)[-1])
            assert v['answer'] in ('Yes', 'No')
        except (IndexError, ValueError, KeyError, TypeError, AssertionError):
            v = dict(answer=None, claim=text[-300:])
        return dict(answer=v['answer'], claim=str(v.get('claim') or ''),
                    finish=c['finish_reason'],
                    tokens=d['usage']['completion_tokens'])

    def serve(k):
        return [(i, one(i, urls[k])) for i in range(k, len(pairs),
                                                    len(urls))]

    with concurrent.futures.ThreadPoolExecutor(len(urls)) as ex:
        got = dict(r for rs in ex.map(serve, range(len(urls))) for r in rs)
    return pd.DataFrame([got[i] for i in range(len(pairs))])

if 'reph-judge' in RERUN:
    assert PAIRS is not None, 'the judge reads the S1 rewrites: reph-s1 first'
    servers, t0 = d2l.edu_label_servers(), time.time()
    try:  # one judge server per GPU of this run
        with concurrent.futures.ThreadPoolExecutor(len(servers)) as ex:
            urls = list(ex.map(lambda s: s.start(), servers))
        say(f'{len(urls)} judge servers ready in {time.time() - t0:.0f} s')
        t0 = time.time()
        got = judge_claims(urls, list(zip(PAIRS.src_text, PAIRS.out_text)))
    finally:
        for s in servers:
            s.stop()
    say(f'judged {len(PAIRS)} pairs in {time.time() - t0:.0f} s; reply '
        f'tokens median {got.tokens.median():.0f}, largest '
        f'{got.tokens.max()}; cut off {(got.finish == "length").sum()}, '
        f'without a verdict {got.answer.isna().sum()}')
    JUDGE_PART = dict(**judge_id(PAIRS), **{  # None for a missing answer
        k: json.loads(got[k].to_json(orient='values')) for k in (
            'answer', 'claim', 'finish', 'tokens')})
    d2l.lm_save_rerun(ARTIFACTS['reph-judge'][1],
                      lambda p: write_json(p, JUDGE_PART))
elif PAIRS is not None:  # a regeneration prefers its own newer file
    found = [('registered results', RESULTS and RESULTS['judge'])]
    path = os.path.join(RERUN_DIR, ARTIFACTS['reph-judge'][1])
    if os.path.exists(path):
        with open(path) as f:
            found.insert(len(found) if not RERUN else 0,
                         ('rerun file', json.load(f)))
    for where, part in found:
        if judged(part, PAIRS):
            JUDGE_PART = part
            break
        if part is not None:
            say(f'the verdicts of the {where} judged other pairs: ignored')
    assert RERUN or RESULTS is None or JUDGE_PART is RESULTS['judge'], (
        'the registered verdicts do not belong to the registered corpus')
if JUDGE_PART is None:
    say(f'judge verdicts: registered with {RES_KEY} (pending)')
else:
    V = PAIRS.assign(answer=JUDGE_PART['answer'], claim=JUDGE_PART['claim'],
                     tokens=JUDGE_PART['tokens']).reset_index(drop=True)
    ok = V.answer.notna().to_numpy()  # the judge gave a verdict
    assert ok.sum() >= len(V) // 2, 'most replies without a verdict'
    say(f'{JUDGE}, thinking, greedy, up to {JUDGE_MAX_TOKENS:,} new tokens '
        f'on {len(V)} S1 pairs: reply tokens median '
        f'{V.tokens.median():,.0f}, largest {V.tokens.max():,}; '
        f'{(~ok).sum()} replies reached the limit before a verdict and are '
        'left out')
    V = V[ok]
    says, lost = (V.answer == 'Yes').to_numpy(), (V.num_fidelity < 1
                                                  ).to_numpy()
    k, n = int(says.sum()), len(says)
    lo, hi = d2l.wilson_ci(k, n)
    say(f'the rewrite adds a claim in {k} of {n} ({k / n:.0%}, Wilson 95% '
        f'[{lo:.0%}, {hi:.0%}])')
    print(pd.crosstab(pd.Series(says, name='judge: adds a claim'),
                      pd.Series(lost, name='a number lost')).to_string())
    say(f'agreement with the number flags: {(says == lost).mean():.0%}')
    named = V[says & ~lost]
    say(f'claims the judge names where every number was kept '
        f'({len(named)} pairs; up to three):')
    for _, r in named.head(3).iterrows():
        say(f'  {r.chunk_id}:', textwrap.shorten(r.claim, 150))
```

The judge needs that room. Its median reply is about 2,800 tokens, most of
them reasoning, so a budget of a thousand reasoning tokens would have
stopped most of them part way through the argument. Even at 12,000 tokens, 4 of the 60
replies ended without a verdict. They are left out, and the rate is computed on
the other 56.

By the judge, the rewrite adds a claim in 39 of 56 pairs: 70%, with a 95%
interval from 57% to 80%. Many of the added claims are small. One rewrite
calls the CMS Web Guide "an article", another says that something "had been
known for years". Others are not small, such as a sentence about the United
States supporting a revolt in Syria and another in Bahrain. The judge reads
strictly, and no person has checked its verdicts, so 70% is the rate that one
strict reader with one prompt reports, not a measured error rate. By that
reading, a rephraser of two billion parameters, told to add nothing, adds
something most of the time.

The judge and the number check agree on only half of the pairs. In 25 pairs
every number was kept and the judge still found an added claim; in 3, a
number was lost and the judge found nothing to object to. These are not
contradictions. The number check asks what was lost, the judge what was
added, and the lost numbers include harmless rewritings such as "thirty".
Neither check is the truth. A pipeline at scale would run the cheap one on
every row and the expensive one on a sample, and publish both rates with the
data.

## Equal Budgets and Fair Accounting
:label:`subsec_lm_synth_arms`

Now the question of the opening: what does rephrased text do to a model
trained on it? To answer it we fix the budget and vary only the data. Every
run trains the same proxy model for $2^{24}$ tokens, and five arms fill those
tokens differently:

- **A**, real text repeated: S1 read eight times;
- **B**, S1 and its rewrites, each read about four times, which is the WRAP
  recipe applied to a fixed source;
- **C**, the rewrites of S1 alone, read for the whole budget (they are
  shorter than S1, so almost nine times);
- **D**, fresh real text: $2^{24}$ tokens of the pool read once, the
  reference for what more real data would give;
- **E**, S1 and the rewrites of the other split S2, each read about four
  times: new information, but in rephrased form.

Every pool, D's included, consists of chunks with an end-of-text token after
each, so all arms see text cut the same way.

Before any run, the law of repeated data of
:numref:`subsec_lm_mix_temperature` limits what B can gain over A. For $U$
unique tokens read for $E$ epochs it counts

$$
D'(U, E) = U + U R^* \bigl(1 - e^{-(E - 1)/R^*}\bigr), \qquad R^* \approx 15.4,
$$

effective tokens: repeats count almost fully for a few epochs and less and
less after that. If the rewrites teach nothing beyond their source, B is
worth A's eight epochs of $U$. If they were as good as fresh text, B would be
worth two sources of size $U$ read four times each. So

$$
D'(U, 8) \le D'_B \le D'(2U, 4).
$$
:eqlabel:`eq_lm_synth_bounds`

The lower bound assumes that a varied surface never hurts. The upper bound
says that a rewrite can be no better than a new document of the same size,
since it adds no facts. Both use a constant that Muennighoff and colleagues
fitted in another setting :cite:`Muennighoff.Rush.Barak.ea.2023`, so they
are a reference line, not a prediction.

```{.python .input #synthetic-data-bounds}
%%tab pytorch
lo, hi = float(d2l.effective_tokens(1, 8)), 2 * float(
    d2l.effective_tokens(1, 4))
assert abs(lo - 6.62) < 0.01 and abs(hi - 7.45) < 0.01
say(f"D'(U, 8) = {lo:.3f} U <= D'_B <= D'(2U, 4) = {hi:.3f} U: B can gain "
    f'at most {hi / lo - 1:.1%} effective tokens over A')
if CORPUS is not None:
    m = CORPUS[1]
    u_s, u_r = (m[f'{NAMES["S1", k]}.bin']['n_tokens'] for k in ('src',
                                                                  'reph'))
    lo_m = float(d2l.effective_tokens(u_s, TOKENS / u_s))
    hi_m = float(d2l.effective_tokens(u_s, TOKENS / 2 / u_s)
                 + d2l.effective_tokens(u_r, TOKENS / 2 / u_r))
    say(f'measured: S1 {u_s:,} tokens, its rewrites {u_r:,}; in units of '
        f"S1: D'_A = {lo_m / u_s:.2f}, the upper bound for B "
        f'{hi_m / u_s:.2f} ({hi_m / lo_m - 1:.1%})')
```

Eight epochs of $U$ are worth $6.62\,U$ and the upper bound is $7.45\,U$, so
B can gain at most 12.5% in effective tokens over A; with the measured size
of the rewrites, 8% shorter than S1, the margin is 11.9%. How large a
difference in loss is that? Arm D will give a scale. It holds about a fifth
more effective tokens than A, and B's best case is a little over half of
that. Three seeds per arm may well not resolve a gain of this size, and we
say so before training rather than after. A design with a smaller source
read more often leaves B far more room at the same cost, and exercise
:exref:`ex_lm_synth-room` works it out.

### The Five Arms
:label:`subsec_lm_synth_build`

The arms become inputs of `train_proxy_lm`: a dictionary of pools and their
weights. The training stream reads each pool's windows in epochs, one after
another, each a fresh permutation (:numref:`sec_lm_data_mixtures`), which is
the reading that the law assumes. Rewriting changes length, so arms with
equal tokens differ slightly in epochs. The table prints for each pool its
unique tokens, its windows per epoch, the epochs read and the effective
tokens.

```{.python .input #synthetic-data-arms-build}
%%tab pytorch
ARM_DESC = {'A': 'real S1, repeated', 'B': 'S1 + rewrites of S1',
            'C': 'rewrites of S1 only', 'D': 'fresh real text',
            'E': 'S1 + rewrites of S2'}
ROOT = CORPUS[0]
POOL = {n: d2l.load_pool(ROOT, n) for n in (
    'train_src_S1', 'train_reph_S1', 'train_reph_S2', 'val_reph')}
S1P, R1P, R2P = (POOL[n] for n in ('train_src_S1', 'train_reph_S1',
                                   'train_reph_S2'))

def chunked(docs, n):
    """The first n tokens of the documents' chunks, each chunk followed by
    its EOS, as in S1's pool and the rewrites' pools."""
    out, total = [], 0
    for _, ids in docs:
        for c in d2l.chunk_doc(ids, TOK, CHUNK_TOKENS):
            out.append(np.array(c + [EOS], dtype=np.uint16))
            total += len(c) + 1
            if total >= n:
                return np.concatenate(out)[:n]
    raise ValueError(f'fewer than {n} tokens in the pool')

FRESH_IDS = chunked(DOCS, TOKENS + 1)  # one epoch, EOS placed as in A-C, E
assert np.array_equal(FRESH_IDS[:len(S1P.ids)], S1P.ids)  # D starts with S1
FRESH = d2l.TokenPool(FRESH_IDS, dict(WEB.meta, n_tokens=TOKENS + 1))
ARM_POOLS = {'A': {'real': S1P}, 'B': {'real': S1P, 'reph': R1P},
             'C': {'reph': R1P}, 'D': {'real': FRESH},
             'E': {'real': S1P, 'reph': R2P}}
ARM_WEIGHTS = {a: {d: 1 / len(p) for d in p} for a, p in ARM_POOLS.items()}
cfg, rows, MAX_EPOCHS = PROXY_SMALL, [], {}
for a, pools in ARM_POOLS.items():
    s = d2l._ProxyStream(pools, ARM_WEIGHTS[a], TOKENS, cfg.seq_len,
                         cfg.batch, 0, None, max_epochs=10**6)
    assert s.steps * cfg.batch * cfg.seq_len == TOKENS  # equal budgets
    MAX_EPOCHS[a] = math.ceil(max(s.epochs.values()) - 1e-9)
    assert MAX_EPOCHS[a] <= 2 * EPOCHS, f'{a}: a pool is far too short'
    for d, P in pools.items():
        n = len(P.ids)
        rows.append(dict(arm=a, arm_is=ARM_DESC[a], pool=d,
                         unique_tokens=n, windows=s.windows[s.names.index(d)],
                         epochs=s.epochs[d],
                         effective=float(d2l.effective_tokens(n, s.epochs[
                             d]))))
ARM_BUILD = pd.DataFrame(rows)
steps = TOKENS // (cfg.batch * cfg.seq_len)
say(f'{TAG} {TOKENS:,} trained tokens per arm ({steps} steps of '
    f'{cfg.batch} x {cfg.seq_len}):')
print(ARM_BUILD.round(3).to_string(index=False))
assert abs(ARM_BUILD.epochs[0] - EPOCHS) < 0.01  # A reads S1 eight times
```

All arms train on 16,777,216 tokens. A reads S1 exactly eight times, C reads
the shorter rewrites 8.7 times, and B and E read rewrites 4.4 times beside
four passes over S1. Counted by the law, B and E hold about 15.5 million
effective tokens against A's 13.9 million. That is the upper bound of
:eqref:`eq_lm_synth_bounds`, since the table counts rewrites as if they were
fresh text.

Each arm trains with three data-order seeds. The recipe is that of
:numref:`sec_lm_data_mixtures`, with its peak learning rate and a cosine
schedule, and with two changes. The model is the wider proxy of 8.1 million
parameters, 6.3 million of them in its embedding table. And the warmup lasts
10% of the steps instead of 2%, because at the learning rate chosen in that
section the wider model trains less steadily with only five warmup steps.
The warmup was fixed before any arm was compared, all five arms share it,
and exercise :exref:`ex_lm_synth-warmup` measures what it changes. Each run is
evaluated in bits per byte on two sets, the held-out real documents of the
pool and the rewrites of the same documents, at a quarter, half, three
quarters and the end of training. A normal build reads the registered runs
and retrains the first million tokens of arm A as a live check; training all
fifteen runs (`D2L_RERUN=synth-arms`) takes ten to fifteen minutes on one
GPU.

```{.python .input #synthetic-data-arms-run}
%%tab pytorch
with open(d2l.lm_artifact('lm-mix-runs-t1-v1')) as f:
    LR = json.load(f)['lr_choice']['chosen']
CFG = PROXY_SMALL._replace(lr=LR)
EVAL_SETS = {'real': VAL, 'reph': POOL['val_reph']}
BYTE_LEN = d2l.token_byte_lengths(TOK)
CONFIG = json.loads(json.dumps(dict(
    proxy=CFG._asdict(), tokens=TOKENS, U=U, val_U=VAL_U, seeds=SEEDS,
    eval_at=EVAL_AT, log_every=LOG_EVERY, weights=ARM_WEIGHTS,
    max_epochs=MAX_EPOCHS, lr_from='lm-mix-runs-t1-v1',
    fresh='chunks of the web pool, EOS per chunk')))
CORPUS_SHA = {k: v['sha256'] for k, v in CORPUS[1].items()}

def run_arm(arm, seed, **kw):
    return d2l.train_proxy_lm(
        ARM_POOLS[arm], ARM_WEIGHTS[arm], TOKENS, CFG, seed=seed,
        eval_sets=EVAL_SETS, log_every=LOG_EVERY,
        max_epochs=MAX_EPOCHS[arm], eval_at=EVAL_AT, byte_lengths=BYTE_LEN,
        **kw)

if 'synth-arms' in RERUN:
    assert judged(JUDGE_PART, judge_pairs(CORPUS[2])), (
        'the results hold the verdicts on this corpus: reph-judge first')
    assert COSMO_ROWS is not None
    runs, t0 = [], time.time()
    for arm in ARM_POOLS:
        for seed in SEEDS:
            out = run_arm(arm, seed)
            runs.append(dict(run_id=f'{arm}-s{seed}', arm=arm, seed=seed,
                             **out))
            say(f'{arm}-s{seed}: real {out["final"]["real"]["bpb"]:.4f}, '
                f'rephrased {out["final"]["reph"]["bpb"]:.4f} bits per '
                f'byte, {out["throughput"]:,.0f} tokens/s, '
                f'{(time.time() - t0) / 60:.1f} min in all')
    res = dict(schema='d2l-synth-results/1', tier=TIER, config=CONFIG,
               corpus=CORPUS_SHA, pools=dict(web=WEB.meta['sha256'],
                                             val=VAL.meta['sha256']),
               runs=runs, judge=JUDGE_PART, cosmopedia=COSMO_ROWS,
               environment=dict(torch=torch.__version__, device=(
                   torch.cuda.get_device_name() if torch.cuda.is_available()
                   else 'cpu')))
    d2l.lm_save_rerun(ARTIFACTS['synth-arms'][1],
                      lambda p: write_json(p, res))
RES_PATH = d2l.lm_artifact(RES_KEY, ARTIFACTS['synth-arms'][1]
                           if 'synth-arms' in RERUN else None)
with open(RES_PATH) as f:
    RESULTS = json.load(f)
assert RESULTS['config'] == CONFIG, 'the runs had another configuration'
assert RESULTS['corpus'] == CORPUS_SHA and RESULTS['pools'] == dict(
    web=WEB.meta['sha256'], val=VAL.meta['sha256'])
assert judged(RESULTS['judge'], judge_pairs(CORPUS[2]))
REC = {r['run_id']: r for r in RESULTS['runs']}
assert list(REC) == [f'{a}-s{s}' for a in ARM_POOLS for s in SEEDS]
if GPU and not RERUN:
    live = run_arm('A', 0, stop_tokens=LIVE_TOKENS)
    logged = {s: l for s, _, l, _ in REC['A-s0']['log']}
    diff = max(abs(l - logged[s]) for s, _, l, _ in live['log'])
    say(TAG, f'arm A, seed 0, steps {LOG_EVERY}-{live["log"][-1][0]}: '
        f'largest difference from the registered losses {diff:.4f} nats '
        '(bf16; tolerance 0.02)')
    assert diff <= 0.02
else:
    say('live check skipped:', 'no GPU' if not GPU else
        'this run regenerates')
```

### Real and Rephrased Validation
:label:`subsec_lm_synth_results`

The live check matches the registered losses to within 0.001 nats. A
difference between two arms counts only if all three seed-paired
differences have the same sign and their mean exceeds twice the pooled
standard deviation of the seeds, estimated from all fifteen runs with ten
degrees of freedom. Otherwise we write "no difference detected at this
scale". The plot below shows the three runs of each arm and their mean on
both validation sets, with a band of two standard deviations around arm A,
and in its third panel the real-validation curves of A, B and D with the
boundaries of A's epochs.

```{.python .input #synthetic-data-arms-analysis}
%%tab pytorch
ARMS = list(ARM_POOLS)
BPB = {v: {a: np.array([REC[f'{a}-s{s}']['final'][v]['bpb'] for s in SEEDS])
           for a in ARMS} for v in EVAL_SETS}
SIGMA = {v: math.sqrt(np.mean([b[a].var(ddof=1) for a in ARMS]))
         for v, b in BPB.items()}  # pooled within-arm seed SD, 10 df

def verdict(x, y, v):
    """Seed-paired y - x on validation set v, and whether it is real."""
    d = BPB[v][y] - BPB[v][x]
    real = len(set(np.sign(d))) == 1 and abs(d.mean()) > 2 * SIGMA[v]
    return d, real

rows = []  # (x, y, validation set, claim, the sign of y - x it claims)
for x, y, v, claim, sign in (
        ('A', 'B', 'real', 'C3: B beats A', -1),
        ('C', 'B', 'real', 'C3b: B beats C', -1),
        ('B', 'E', 'real', 'C5: E beats B', -1),
        ('A', 'C', 'real', 'C4: C loses to A', 1),
        ('A', 'C', 'reph', 'C4: C wins on rewrites', -1)):
    d, real = verdict(x, y, v)
    rows.append(dict(claim=claim, difference=f'{y} - {x}', val=v,
                     diffs=np.round(d, 4).tolist(), mean=d.mean(),
                     real=real, holds=real and np.sign(d.mean()) == sign))
VERDICTS = pd.DataFrame(rows)
say(TAG, 'pooled seed SD of bits per byte: real '
    f'{SIGMA["real"]:.4f}, rephrased {SIGMA["reph"]:.4f} (10 df)')
print(VERDICTS.round(4).to_string(index=False))
for _, r in VERDICTS.iterrows():
    say(f'  {r.claim}: ' + ('holds' if r.holds else
                            'no difference detected at this scale'
                            if not r.real else 'the opposite holds'))
best = min(ARMS, key=lambda a: BPB['real'][a].mean())
say('lowest real-validation bits per byte: arm', best, '(D, fresh text, '
    'is the reference bound)' if best == 'D' else
    '(not D: a bug to look for, not a result)')
assert best == 'D' or RERUN, 'D is not best on real validation: a bug'
fig, axes = d2l.plt.subplots(1, 3, figsize=(11, 3))
for ax, v, name in zip(axes, EVAL_SETS, ('real validation',
                                         'rephrased validation')):
    m = [BPB[v][a].mean() for a in ARMS]
    d2l.plot_intervals(ARMS, {'mean': (m, m, m)}, 'arm',
                       f'{name}, bits per byte', axes=ax,
                       reference=BPB[v]['A'].mean(), marks={'seeds': (
                           '.', {'mean': (np.repeat(np.arange(len(ARMS)),
                                                    len(SEEDS)),
                                          np.concatenate([BPB[v][a] for a
                                                          in ARMS]))})})
    ax.axhspan(BPB[v]['A'].mean() - 2 * SIGMA[v],
               BPB[v]['A'].mean() + 2 * SIGMA[v], color='0.85', zorder=0)
curve = lambda a: np.mean([[e['domains']['real']['bpb'] for e in REC[
    f'{a}-s{s}']['evals']] for s in SEEDS], 0)
x = [e['tokens'] for e in REC['A-s0']['evals']]
d2l.plot(x, [curve(a) for a in 'ABD'], 'tokens',
         'real validation, bits per byte', legend=[
             f'{a}: {ARM_DESC[a]}' for a in 'ABD'], axes=axes[2])
per_step = CFG.batch * CFG.seq_len
for s in REC['A-s0']['epoch_starts']['real'][1:]:  # A's repeats begin
    axes[2].axvline(s * per_step, color='black', linestyle=':', linewidth=1)
fig.tight_layout()
ARM_TABLE = pd.DataFrame([dict(  # tokens in millions
    arm=a, unique_M=ARM_BUILD[ARM_BUILD.arm == a].unique_tokens.sum() / 1e6,
    effective_M=ARM_BUILD[ARM_BUILD.arm == a].effective.sum() / 1e6,
    real=BPB['real'][a].mean(), real_seeds=np.round(BPB['real'][a], 4)
    .tolist(), reph=BPB['reph'][a].mean(),
    reph_seeds=np.round(BPB['reph'][a], 4).tolist()) for a in ARMS])
say(TAG, 'bits per byte (mean and the three seeds; arms as above):')
print(ARM_TABLE.round(4).to_string(index=False))
```

The seeds agree closely, to about 0.007 bits per byte on both sets, so
differences above 0.014 are resolved. Read the first panel first. At T1
scale (8.1M-parameter proxies, $2^{24}$ tokens), fresh text is best, as it
must be: arm D is lower than A by about 0.03. Arm B is not lower than A at
all. The mean difference is 0.002, and the seeds disagree on its sign: no
difference detected at this scale. The rewrites of the same chunks added
nothing measurable. A null result was what the bound made likely, but even
its best case, a little over half of D's gain, would have been near the edge
of what three seeds resolve, and we did not see it.

The rewrites alone are a different matter. At T1 scale (8.1M-parameter
proxies, $2^{24}$ tokens), arm C loses to A on real validation by about 0.07
bits per byte, and on rephrased validation it wins by about 0.06. The second
panel shows the reversal: on rewrites every arm beats A, those that read
rewrites (B, C and E) by more than fresh text does, and C most of all. Recall from
:numref:`sec_environment-and-distribution-shift` that a model trained on
samples of one distribution and tested on another is subject to
distribution shift. Training on a generator's output moves the training
distribution toward the generator's: a model that has read the rephraser's
style finds that style easy, and a model that has read nothing else is worse
at the web. Had we evaluated only on rewrites, C would be the best arm in
the table. This is the second half of WRAP's explanation at work: style
that matches the evaluation helps the evaluation, whatever it does for
other text.

Mixing repairs most of the damage. At T1 scale (8.1M-parameter proxies,
$2^{24}$ tokens), B beats C on real validation by about 0.06, in every seed.
Arm E, new information in rephrased form, ends below B in the table but by
less than the threshold: no difference detected at this scale.

Did rephrasing behave as WRAP reports? On the kind of text where WRAP
measured, it did. Arm B gains on rephrased validation, text in a style other
than the source's, where it beats A by about 0.05 bits per byte, more than
fresh text does. WRAP's perplexity result is likewise reported on text other
than its C4 source, subsets of the Pile, and its own explanation, that the
rewrites' styles are closer to the evaluation text, predicts this pattern.
On held-out text of the source distribution, where WRAP made no claim, the
rewrites of the same chunks added nothing measurable. There was also little
to gain at our scale: the bound allows 12% more effective tokens, and the
curves of A and B in the third panel lie on top of each other from the first
evaluation to the last, while D pulls away. Whether rewrites help more on a
noisier source than FineWeb-Edu, the first half of WRAP's explanation, this
comparison does not test. What carries over to any scale is the protocol: equal tokens, both
validation sets, a noise threshold fixed in advance and a bound computed
before the runs. A claim that synthetic data helps should come with all
four.

## Recursive Training and Collapse
:label:`subsec_lm_synth_collapse`

So far a generator trained on human text wrote data for a model that also
read human text. As generated text spreads over the web, future crawls will
contain the output of earlier models, and future models will learn from it.
Shumailov and colleagues (2023) asked what happens when this repeats, each
model trained on the output of its predecessor, and called the result the
curse of recursion. The rare events of the original distribution vanish
first, and the models converge to something narrower than the data they
started from.

The smallest model of the process is a Gaussian. Generation 0 is the true
distribution $\mathcal{N}(\mu, \sigma^2)$. Fit $t$ draws $n$ points from fit
$t - 1$ and estimates their mean $\hat\mu_t$ and variance $\hat\sigma_t^2$
by maximum likelihood. In *replace* mode it uses only those $n$ points; in
*accumulate* mode it uses all points drawn so far. With $q = (n - 1)/n$, the
two modes give

$$
\begin{aligned}
\textrm{replace:}\quad & \mathbb{E}[\hat\sigma_T^2] = q^T \sigma^2, \qquad \operatorname{Var}(\hat\mu_T) = \sigma^2 (1 - q^T),\\
\textrm{accumulate:}\quad & \operatorname{Var}(\hat\mu_T) \approx \frac{\sigma^2}{n} \sum_{t=1}^{T} \frac{1}{t^2} < \frac{\pi^2}{6}\,\frac{\sigma^2}{n}.
\end{aligned}
$$
:eqlabel:`eq_lm_synth_collapse`

The derivation takes three steps. First, the maximum-likelihood variance of
$n$ points is biased low by the factor $q$,
$\mathbb{E}[\hat\sigma_t^2 \mid \hat\sigma_{t-1}^2] = q\, \hat\sigma_{t-1}^2$,
and the factors multiply over the generations. Second, the mean takes a
random step at each generation,
$\hat\mu_t = \hat\mu_{t-1} + \hat\sigma_{t-1} z_t / \sqrt{n}$ with $z_t$
standard normal and independent of the past, so its variance is the sum of
the step variances, $\sum_{t=1}^{T} q^{t-1} \sigma^2 / n = \sigma^2 (1 - q^T)$.
Third, under accumulation the $n$ new points are only a share $1/t$ of the
$nt$ points behind fit $t$, so the step shrinks to
$\hat\sigma_{t-1} z_t / (t \sqrt{n})$; with $\hat\sigma_{t-1} \approx \sigma$
the step variances $\sigma^2 / (n t^2)$ have a finite sum.

The formulas describe two different fates. Under replacement the fitted
spread shrinks by 1% per generation for $n = 100$ while the mean wanders, so
after many generations the model is a narrow distribution around a point
drawn at random from the original one. Under accumulation the mean stays
within the error of the first fit, and the spread stays near the truth. The
simulator below fits by maximum likelihood exactly as described.

```{.python .input #synthetic-data-collapse-def}
%%tab pytorch
#@save
def collapse_sim(n, generations, mode='replace', reps=1000, seed=0, mu=0.0,
                 sigma=1.0):
    """The Gaussian refit: generation 0 is N(mu, sigma^2); fit t (t >= 1)
    fits (mu, sigma^2) by maximum likelihood to the n points just sampled
    from fit t - 1 ('replace') or to all points sampled so far
    ('accumulate'). Returns, per generation 0..generations over reps
    repetitions, the mean of mu_t, its variance var_mu, the mean of
    sigma_t^2 and the Monte-Carlo standard errors se_var_mu, se_sigma2."""
    if mode not in ('replace', 'accumulate'):
        raise ValueError(f'unknown mode {mode!r}')
    rng = np.random.default_rng(seed)
    M, S = np.zeros((generations + 1, reps)), np.zeros((generations + 1,
                                                        reps))
    M[0], S[0] = mu, sigma**2
    total = np.zeros(reps), np.zeros(reps), 0  # sum, sum of squares, count
    for t in range(1, generations + 1):
        x = rng.normal(M[t - 1][:, None], np.sqrt(S[t - 1])[:, None],
                       (reps, n))
        s1, s2, k = x.sum(1), (x**2).sum(1), n
        if mode == 'accumulate':
            total = total[0] + s1, total[1] + s2, total[2] + n
            s1, s2, k = total
        M[t] = s1 / k
        S[t] = np.maximum(s2 / k - M[t]**2, 0)
    dev = M - M.mean(1, keepdims=True)
    var = (dev**2).mean(1) * reps / (reps - 1)
    m4 = (dev**4).mean(1)
    return dict(mean=M.mean(1), var_mu=var, mean_sigma2=S.mean(1),
                se_var_mu=np.sqrt(np.maximum(m4 - var**2, 0) / reps),
                se_sigma2=S.std(1, ddof=1) / math.sqrt(reps))
```

We check the closed forms at $n = 100$ over 20 generations and 1,000
repetitions, and then move to text. The discrete toy takes 200,000 words of
the web pool and fits a bigram model over the 20,000 most frequent words
plus a token for all others. Each generation samples a corpus of the same
size from the fitted model and refits, on that corpus alone (replace) or on
all corpora so far (accumulate). We track the word types that the fitted
model can still emit, and what happens to the rarest tenth of the original
types: how many survive and how much probability they hold.

```{.python .input #synthetic-data-collapse}
%%tab pytorch
n, T, reps = 100, 20, 1000
SIM = {mode: d2l.collapse_sim(n, T, mode, reps, seed=0)
       for mode in ('replace', 'accumulate')}
q, t = (n - 1) / n, np.arange(T + 1)
r, acc = SIM['replace'], SIM['accumulate']
z_s = abs(r['mean_sigma2'][T] - q**T) / r['se_sigma2'][T]
z_m = abs(r['var_mu'][T] - (1 - q**T)) / r['se_var_mu'][T]
partial = np.r_[0, np.cumsum(1 / np.arange(1, T + 1)**2)] / n
bound = math.pi**2 / 6 / n
assert z_s < 3 and z_m < 3
assert acc['var_mu'][T] < bound + 3 * acc['se_var_mu'][T]
say(f'replace, T = {T}: E[sigma^2] {r["mean_sigma2"][T]:.4f} (closed form '
    f'q^T = {q**T:.4f}, {z_s:.1f} SE), Var(mu) {r["var_mu"][T]:.4f} '
    f'(1 - q^T = {1 - q**T:.4f}, {z_m:.1f} SE)')
say(f'accumulate, T = {T}: E[sigma^2] {acc["mean_sigma2"][T]:.4f}, '
    f'Var(mu) {acc["var_mu"][T]:.5f} (sum of 1/t^2 over n: '
    f'{partial[T]:.5f}; the bound pi^2 / (6n) = {bound:.5f})')

def corpus_words(n_words):
    words = []
    for _, ids in DOCS:
        words += d2l.shingles(TOK.decode(ids), 1)
        if len(words) >= n_words:
            return words[:n_words]

WORDS = corpus_words(200_000)
counts = collections.Counter(WORDS)
VOCAB = {w: i for i, (w, _) in enumerate(counts.most_common(20_000))}
V = len(VOCAB) + 1  # the last id is <unk>
seq0 = np.array([VOCAB.get(w, V - 1) for w in WORDS])
freq0 = np.bincount(seq0, minlength=V)[:V - 1]
TAIL = np.argsort(freq0, kind='stable')[:len(freq0) // 10]  # rarest 10%

def bigram_counts(seq):
    return sparse.csr_matrix((np.ones(len(seq) - 1), (seq[:-1], seq[1:])),
                             shape=(V, V))

def sample_bigram(C, uni, size, rng, length=200):
    """size words from the bigram counts C: chains of length words, each
    started from the unigram counts uni (also used after a dead end)."""
    C = C.tocsr()
    cum, start = np.cumsum(C.data), np.r_[0.0, np.cumsum(C.data)][C.indptr]
    rows = np.asarray(C.sum(1)).ravel()
    p_uni = uni / uni.sum()
    w = rng.choice(V, size // length, p=p_uni)
    out = [w]
    for _ in range(length - 1):
        dead = rows[w] == 0
        j = np.searchsorted(cum, start[w[~dead]] + rng.random(
            (~dead).sum()) * rows[w[~dead]], side='right')
        w = w.copy()
        w[~dead] = C.indices[j]
        w[dead] = rng.choice(V, dead.sum(), p=p_uni)
        out.append(w)
    return np.stack(out, 1).ravel()

TOY = {}
for mode in ('replace', 'accumulate'):
    rng, seq = np.random.default_rng(0), seq0
    C, uni = bigram_counts(seq0), np.bincount(seq0, minlength=V).astype(float)
    stats = []
    for g in range(11):  # the model fitted at generation g
        stats.append(dict(generation=g, types=int((uni[:-1] > 0).sum()),
                          tail_alive=(uni[TAIL] > 0).mean(),
                          tail_mass=uni[TAIL].sum() / uni.sum()))
        seq = sample_bigram(C, uni, len(seq0), rng)
        c, u = bigram_counts(seq), np.bincount(seq, minlength=V)
        C, uni = (c, u.astype(float)) if mode == 'replace' else (C + c,
                                                                 uni + u)
    TOY[mode] = pd.DataFrame(stats)
say(f'bigram toy: {len(WORDS):,} words, {V - 1:,} types plus <unk>; the '
    f'rarest 10% ({len(TAIL):,} types) hold '
    f'{np.isin(seq0, TAIL).mean():.2%} of the words. Per generation: types '
    'the model can still emit, the share of the rarest types still among '
    'them, and their probability mass:')
print(pd.concat({m: d.set_index('generation') for m, d in TOY.items()},
                axis=1).iloc[[0, 1, 2, 5, 10]].round(5).to_string())
fig, axes = d2l.plt.subplots(1, 2, figsize=(10, 3.6))
g = t[1:]  # the fitted generations (Var(mu) is 0 at generation 0)
d2l.plot(g, [r['mean_sigma2'][1:], q**g, acc['mean_sigma2'][1:],
             r['var_mu'][1:], 1 - q**g, acc['var_mu'][1:]], 'generation',
         'variance', fmts=('C0-', 'C0:', 'C1-', 'C2-', 'C2:', 'C3-'),
         yscale='log', axes=axes[0])
axes[0].axhline(bound, color='black', linestyle='--', linewidth=0.8)
axes[0].legend([r'$E[\hat\sigma^2]$, replace', '$q^t$',
                r'$E[\hat\sigma^2]$, accumulate', r'Var$(\hat\mu)$, replace',
                '$1 - q^t$', r'Var$(\hat\mu)$, accumulate',
                r'$\pi^2 / (6n)$'], loc='upper center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.22))
d2l.plot(TOY['replace'].generation, [TOY[m].types for m in TOY],
         'generation', 'word types the model emits', legend=list(TOY),
         axes=axes[1])
fig.tight_layout()
```

The closed forms hold. After 20 generations of replacement the mean fitted
variance is 0.80 against $q^{20} = 0.82$, and the variance of the mean is
0.17 against 0.18, both within two Monte-Carlo standard errors. Under
accumulation the variance of the mean is 0.0157, below the bound
$\pi^2/(6n) \approx 0.0164$, and the fitted variance stays at 0.99.

The bigram model loses words. Under replacement it can emit 16,001 of its
20,000 types after one generation, 10,238 after five and 7,655 after ten;
under accumulation all 20,000 remain. The rarest tenth are 2,000 words that
occur exactly once, 1% of the 200,000. A word seen once appears in a
resampled corpus of the same size about once on average, and with
probability about $e^{-1}$ not at all. A type that is not drawn gets count
zero, and the refitted model can never emit it again. The survival after one
generation, 62%, is close to $1 - e^{-1} \approx 0.63$, and after ten
generations 17% of these words remain. Curiously, their total probability
does not shrink: it stays near 1%, because the survivors inherit the counts
of the lost. The tail is lost type by type, not as a falling share.

The toy shows a mechanism under a stated protocol, in which every
generation learns only from the previous model's output, with no fresh data
and no filter. Real pipelines keep real data, as accumulation does, and they
filter what they generate. Gerstgrasser and colleagues (2024) prove for
linear models that accumulating data keeps the test error bounded however
many generations pass, and find the same in experiments with language
models. The collapse is therefore not a forecast for the field. It is a
forecast for one practice, replacing real data with generated data again
and again, which nobody should follow.

## Licences and Disclosure
:label:`subsec_lm_synth_rights`

Who may train on generated text? The question has two halves. A rewrite does
not shed the licence of its source. Our rewrites of FineWeb-Edu remain
FineWeb-Edu text under ODC-By and, as the FineWeb-Edu card says, also subject
to Common Crawl's terms of use. And the generator has terms of its own.
Qwen3.5-2B is released under Apache-2.0, which places no condition on its
outputs, so our corpus carries the source's attribution and the generator's
name. Other licences do set conditions. The Llama 3 licence, for example,
asks that a model trained with the help of Llama outputs carry "Llama" at
the start of its name, and the card of Nemotron-CC-v2 warns that models
trained on it may be subject to the terms of the Qwen and DeepSeek models
that wrote parts of it. Hosted models add a second document. As
:numref:`sec_lm_api` found, the licence of the weights can permit training
on outputs while the terms of the service that ran them forbid it, so both
are read before an API model writes a single training example. These
statements describe licences and are not legal advice; the legal status of
model outputs is unsettled in several jurisdictions.

The practical rule is to record everything needed to answer the question
later. The manifest stage `rephrase` holds the generator and its revision,
the hash of the prompt, the sampling settings and seed, the documents and
chunks of each split, the flags and the drops with their reasons, and hashes
of the input and output texts. The dataset card states the synthetic share
in one line.

```{.python .input #synthetic-data-manifest}
%%tab pytorch
_, meta, rows = CORPUS
MAN = d2l.CorpusManifest(TAR_KEY)
read = {c['doc_id'] for v in SPLITS.values() for c in v}  # source documents
MAN.add_source('fineweb-edu', 'hf://datasets/HuggingFaceFW/fineweb-edu/'
               'sample/10BT', revision=WEB.meta['revision'],
               license='odc-by', n_docs=len(read),
               bytes=sum(meta[f'{NAMES[s, "src"]}.bin']['n_bytes_nfc']
                         for s in SPLITS))
kept = rows[rows.kept]
fl = flags(kept)
gen = json.loads(rows[list(GEN_ID)].iloc[0].to_json())  # as recorded
assert all((rows[k] == v).all() for k, v in gen.items())  # one generation
MAN.add_stage(
    'rephrase', dict(generator=f'{gen["model_id"]}@{gen["model_revision"]}',
                     **{k: gen[k] for k in GEN_ID if k not in (
                         'model_id', 'model_revision')},
                     splits={s: dict(documents=N_DOCS[s],
                                     chunks=len(SPLITS[s])) for s in SPLITS},
                     flags={k: int(fl[k].sum()) for k in fl},
                     flag_thresholds=FLAGS, drop_len_ratio=LEN_RATIO,
                     order='length'),
    n_in=len(rows), n_out=len(kept),
    drops={k: int((rows['drop'] == k).sum()) for k in ('length',
                                                        'len_ratio')},
    in_hash=d2l.docs_hash(list(zip(rows.chunk_id, rows.src_text))),
    out_hash=d2l.docs_hash(list(zip(kept.chunk_id, kept.out_text))),
    impl='d2l.rephrase_docs', tokens_out=sum(
        meta[f'{NAMES[s, "reph"]}.bin']['n_tokens'] for s in SPLITS))
print(json.dumps(MAN.stages[-1], indent=1, sort_keys=True))
synth = sum(m['n_tokens'] for m in meta.values() if m['kind'] == 'reph')
share = synth / sum(m['n_tokens'] for m in meta.values())
print(MAN.to_card())  # a Markdown table: printed unwrapped
say(f'Contains synthetic text: {share:.0%} of its tokens were written by '
  f'{REPHRASER} from the FineWeb-Edu sources above.')
```

Of the archive's tokens, 48% were written by the model, the rewrites of the
three splits beside their sources. The card carries that number so that
anyone who trains on the data knows it. The same checks return in
:numref:`sec_lm_instruction_data`, where a model writes the instructions and
answers for post-training and generated text becomes the main ingredient
rather than a supplement. For pretraining one question stays open at our
scale: whether a stronger rephraser, or a style that adds structure rather
than polish, turns the null result of arm B on real text into a gain.

## Summary

Language models can rewrite existing documents or write new ones from a
seed, and both kinds of text now enter pretraining at the scale of whole
crawls. Rewrites have to be checked. Cheap string measures catch lost
numbers and near-copies. A strict judge model finds the added claims they
miss, and by its reading about 70% of the rewrites of a
two-billion-parameter model add one. The two checks see different failures,
losses and additions. At one temperature, conditioning the prompt on format
and audience raised distinct 4-grams from about 0.76 to 0.995. At T1 scale
(8.1M-parameter proxies, $2^{24}$ tokens), rewrites mixed with their own
source changed nothing measurable on real text, within the 12% that the law
of repeated data allowed, and helped on rewritten text. Rewrites alone made
the model better on rewrites and worse on real text, so a claim about
synthetic data needs both validation sets. Training each generation only on
the previous model's output loses variance and rare words, and accumulating
real data prevents it. The rights to generated text follow both its source
and its generator.

## Exercises

1. **Two flagged rewrites.** One rewrite has copy rate 0.95 and number
   fidelity 1.0; another has copy rate 0.05 and number fidelity 0.5.
    1. Is either of them good training data? What would you check before
       deciding?
    1. Write a rewrite of the opening's chunk that scores copy rate below
       0.3 and number fidelity 1.0 and is still wrong. Which of this
       section's checks would catch it?
   :exlabel:`ex_lm_synth-flags`

1. [code] **The corpus keeps its promises.** Using only the registered
   archive, check that no rewrite contains a thinking tag, that every row
   whose `finish` is `length` or whose length ratio lies outside
   $[0.5, 1.5]$ is marked as dropped and is absent from the token files, and
   that the token files decode to the kept texts. Which of these properties
   would a change of chat template break first?
   :exlabel:`ex_lm_synth-corpus`

1. **The rephrasing budget.** A generator of $N_g$ parameters rewrites a
   chunk of $L$ tokens into about $L$ tokens, and a student of $N_s$
   parameters trains on the rewrite.
    1. Count the forward FLOPs of the generation (the prefill of the prompt
       and the decoding of the reply, at about $2N_g$ per token each) and the
       training FLOPs of one pass over the rewrite (about $6N_s$ per token).
       At which ratio $N_g / N_s$ do the two costs match?
    1. Evaluate the ratio for this section (Qwen3.5-2B, about 1.9 billion
       parameters in its language model, and the 8.1-million-parameter
       proxy) and for a 1.7-billion-parameter generator writing data for a
       student of 8 billion parameters trained for several epochs.
    1. Decoding is limited by memory bandwidth, not arithmetic
       (:numref:`sec_kv-cache`). Does that make FLOPs an underestimate or an
       overestimate of the generation's cost in GPU time?
   :exlabel:`ex_lm_synth-budget`

1. **How fast replacement collapses.** In replace mode with $n$ points per
   generation:
    1. Show that $\log \hat\sigma_t^2$ is a random walk whose steps have
       negative mean, and conclude that
       $P(\hat\sigma_T^2 < \epsilon \sigma^2) \to 1$ as $T \to \infty$ for
       every $\epsilon > 0$.
    1. Find the number of generations after which
       $\mathbb{E}[\hat\sigma_T^2] = \sigma^2/2$ for $n = 100$, and compare
       it with the median of $\hat\sigma_T^2$ in a simulation (extend
       `collapse_sim` to return it). Why do the two differ?
    1. Show that $\operatorname{Var}(\hat\mu_T) \to \sigma^2$. What is the
       limiting model, and how much does it know about the original
       distribution?
   :exlabel:`ex_lm_synth-collapse-rate`

1. [code] **Temperature and diversity.** Rewrite the 60 live chunks in the
   encyclopedia style at temperatures 0.3, 0.7 and 1.0. Before running,
   predict how distinct 4-grams across the rewrites, the near-duplicate rate
   among them and number fidelity will move. Measure all three. Is there a
   temperature that raises diversity without lowering fidelity, and how does
   the gain compare with the prompt conditioning of
   :numref:`subsec_lm_synth_seeds`?
   :exlabel:`ex_lm_synth-temperature`

1. [code] **Omissions.** The judge asks only whether a rewrite adds a claim.
   Write a second prompt that asks whether the rewrite leaves out a fact of
   the source, run it on the same 60 pairs, and tabulate it against the
   number check and against the first verdict. Do omissions or additions
   explain more of the lost numbers?
   :exlabel:`ex_lm_synth-omissions`

1. [code] **A stronger rephraser.** Rewrite S1 with `Qwen/Qwen3.5-4B` instead
   of Qwen3.5-2B, keeping prompt, sampling and filters. Before running,
   predict whether the copy rate, the number fidelity and the difference
   between arms B and A will change. Then measure the fidelity statistics and
   train arm B on the new rewrites with the three seeds. Compare with the
   original B by the section's rule and say what a null result would mean
   here.
   :exlabel:`ex_lm_synth-stronger`

1. [code] **Room to gain.** Arm B of this section could gain at most 11.9% in
   effective tokens over A. Keep the budget of $2^{24}$ tokens, but let A read
   a source of $U = 2^{19}$ tokens 32 times and B read that source and its
   rewrites 16 times each.
    1. Compute both sides of :eqref:`eq_lm_synth_bounds` for this design. How
       much more room does B have?
    1. Run A, B and D at this size with three seeds each, and apply
       the section's rule on both validation sets. Before running, predict
       which of the differences it will resolve.
    1. A smaller source read more often also costs A more. Which of the two
       designs answers the question of the opening better, and why?
   :exlabel:`ex_lm_synth-room`

1. [code] **Warmup and seed spread.** The arms warm up over 10% of their
   steps; the proxies of :numref:`sec_lm_data_mixtures` warm up over 2%.
   Train arm A with both warmups at the same peak learning rate, three seeds
   each, and compare the standard deviation of the final real-validation
   bits per byte. Plot the training losses of the first 40 steps. Where does
   the difference come from, and how small a difference between two arms
   could the section's rule resolve under each warmup?
   :exlabel:`ex_lm_synth-warmup`

1. [code] **Fresh data against collapse.** Extend `collapse_sim` so that a
   fraction $\phi$ of the $n$ points of each generation is drawn from the
   true distribution and the rest from the previous fit, in replace mode.
   Find the smallest $\phi$ for which $\mathbb{E}[\hat\sigma_T^2]$ stays
   above $0.9\,\sigma^2$ for all $T$, first by simulation and then from the
   recursion for the expected variance. Do the same for the bigram toy, and
   find the fraction at which the rarest tenth of the types stops
   disappearing. Are the two fractions alike, and should they be?
   :exlabel:`ex_lm_synth-fresh`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §20.5]{.kicker}

Synthetic and Rephrased Data<br>**rewriting · checking · generating from seeds · equal budgets · collapse · licences**
:::
:::

::: {.slide title="A 2B Model Rewrites a Page in Seconds. Train on a Million?"}
[The problem]{.kicker}

| source (FineWeb-Edu) | rewrite (Qwen3.5-2B) |
|---|---|
| ... approximately 30 minutes before and after meals ... | ... approximately thirty minutes before and after each meal ... |
| Relief of oral pain was reported in 48 of the 50 patients. | Reports of relief of oral pain were recorded in 48 out of the 50 patients. |

- Every fact kept, yet number fidelity 0.67: "30" became "thirty".
- Nine samples of the same prompt: fewer than half of their 4-grams distinct.

::: {.d2l-note .warn}
Fluent is not faithful, and many samples are not many documents.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- build a rephrasing pipeline: chunk, prompt, generate, filter;
- check rewrites cheaply (copy rate, numbers, length) and with a judge;
- compare training data at equal tokens and bound the gain in advance;
- derive model collapse in a Gaussian refit;
- say who may train on generated text.

**Running example:** FineWeb-Edu chunks (S1, S2: $2^{21}$ tokens each),
Qwen3.5-2B as rephraser, 8.1M-parameter proxies trained on $2^{24}$ tokens.
:::

::: {.slide title="Synthesis Comes After Mixing and Before Cutting"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| step | section |
|---|---|
| get text | Document Extraction from Web Crawls |
| choose text | Quality Filtering |
| | Deduplication and Decontamination |
| mix text | Data Mixtures |
| | **Synthetic and Rephrased Data** |
| cut text | Tokenizers at Scale |
:::

::: {.col .narrow}
- **Here:** generated and rewritten documents as pretraining data.
- **Skips:** post-training data written by models.
:::
:::
:::

::: {.slide title="A Rewrite Changes the Form; a Generation Can Change the Content"}
[Four kinds]{.kicker}

| kind | adds | example |
|---|---|---|
| generate from a seed | new content, from the generator's memory | Cosmopedia, phi-1 |
| rewrite a document | form, not facts | WRAP, Nemotron-CC, FinePhrase |
| demonstrations | instruction following | post-training sets |
| labels | a signal to train on | educational scores |

::: {.d2l-note}
Judge generated data by its effect on held-out data at equal cost, never by
fluency.
:::
:::

::: {.slide title="Rephrasing Has Reached the Scale of Whole Crawls"}
[Rephrased corpora]{.kicker}

| corpus | generator | generated tokens | licence |
|---|---|---|---|
| Nemotron-CC-v2 (rewrites, QA) | Mistral-NeMo-12B, Qwen3-30B-A3B | about 2.5T | NVIDIA Data Agreement |
| FinePhrase (rewrites of FineWeb-Edu) | SmolLM2-1.7B-Instruct | 486B | ODC-By |
| Cosmopedia (from seeds) | Mixtral-8x7B-Instruct | 25B | Apache-2.0 |

- WRAP: about 3× faster pretraining on C4; perplexity over 10% lower at equal compute.
- FinePhrase: generators above about 1B parameters add nothing.
:::

::: {.slide title="Four Steps, Two Drop Rules, Two Flags"}
[The pipeline]{.kicker}

chunk (≤ 400 tokens, at sentence ends) → prompt → generate → filter

| rule | effect |
|---|---|
| reply ran out of tokens | dropped |
| length ratio outside $[0.5, 1.5]$ | dropped |
| copy rate > 0.8 (5-gram Jaccard) | flagged, kept |
| a digit string lost | flagged, kept |

::: {.d2l-note}
Prompt: "Preserve every fact, name and number; add nothing." Chat template
with thinking off; any `<think>` in a reply fails an assertion.
:::
:::

::: {.slide title="Style Sets the Length: Conversations Run Out of Tokens"}
[Live rewrites, 60 chunks per style, one sampled run]{.kicker}

| | encyclopedia | questions and answers |
|---|---|---|
| kept | 59 | 34 |
| ran out of 512 tokens | 0 | 16 |
| length ratio outside range | 1 | 10 |
| every number kept | 63% | 50% |
| mean copy rate | 0.53 | 0.47 |

Live counts move by a few on other hardware. Registered corpus, 14,305
chunks: 13,809 kept, about 70% keep every number, 9% fewer tokens for the
same characters.
:::

::: {.slide title="Check: Which of These Rewrites Is Good Data?"}
[Check]{.kicker}

Rewrite 1: copy rate 0.95, number fidelity 1.0.<br>
Rewrite 2: copy rate 0.05, number fidelity 0.5.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Neither, as far as the numbers say. The first is nearly a copy
and adds nothing; the second shares no phrasing and lost half its numbers,
or only rewrote them as words. The flags raise questions; reading answers
them.
:::
:::
:::

::: {.slide title="A Strict 27B Judge Finds an Added Claim in 70% of Rewrites"}
[The judge]{.kicker}

Qwen3.8-27B (4-bit), thinking on, no thinking budget, 12,000 new tokens;
median reply about 2,800 tokens; 56 verdicts of 60.

| | number lost | every number kept |
|---|---|---|
| judge: adds a claim | 14 | 25 |
| judge: no added claim | 3 | 14 |

39 of 56 add a claim (95% interval 57% to 80%); agreement with the number
check: 50%.

::: {.d2l-note}
The number check sees losses, the judge sees additions. Run both. No
person has checked the verdicts: 70% is one strict reader's rate.
:::
:::

::: {.slide title="At One Temperature, Format and Audience Raise Diversity"}
[Generating from seeds]{.kicker}

Nine outputs from one seed, temperature 0.7, each cut to 70 words:

| prompt | distinct 4-grams | mean 5-gram Jaccard |
|---|---|---|
| rewrite, same prompt | 0.44 | 0.19 |
| "write about the topic" | 0.76 | 0.05 |
| 3 formats × 3 audiences | 0.995 | 0.00 |

::: {.d2l-note .warn}
Diversity of form, not of truth: the children's version moved the pain from
the mouth to the stomach.
:::
:::

::: {.slide title="Five Arms Fill the Same 2²⁴ Tokens Differently"}
[Equal budgets]{.kicker}

| arm | data | epochs |
|---|---|---|
| A | real S1 | 8 |
| B | S1 + rewrites of S1 | 4 + 4.4 |
| C | rewrites of S1 | 8.7 |
| D | fresh real text | 1 |
| E | S1 + rewrites of S2 | 4 + 4.4 |

Three data-order seeds per arm; bits per byte on real and on rephrased
validation text.
:::

::: {.slide title="The Repeated-Data Law Caps B's Gain at 12.5%"}
[Bound before running]{.kicker}

Effective tokens of $U$ unique tokens read $E$ times:

$$D'(U, E) = U + U R^* \bigl(1 - e^{-(E-1)/R^*}\bigr), \quad R^* \approx 15.4$$

$$D'(U, 8) \le D'_B \le D'(2U, 4)$$

- Lower: rewrites teach nothing new. Upper: rewrites as good as fresh text.
- $6.62\,U \le D'_B \le 7.45\,U$; with the measured rewrite size, 11.9%.
:::

::: {.slide title="Check: What Should Arm B Gain over A?"}
[Check]{.kicker}

Fresh text (D) holds about 21% more effective tokens than A and is lower by
about 0.03 bits per byte. Seed noise: $2\sigma \approx 0.014$. What do you
expect for B?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** At most a little over half of D's gain, near the edge of what
three seeds resolve; a null result is likely. Measured: $+0.002$, seeds
disagree on the sign.
:::
:::
:::

::: {.slide title="Rewrites Alone Win on Rewrites and Lose on the Web"}
[Results at T1 scale: 8.1M parameters, 2²⁴ tokens]{.kicker}

Bits per byte; arm A, then each arm's difference from A:

| arm | real validation | rephrased validation |
|---|---|---|
| A real S1, repeated | 1.92 | 1.80 |
| B + rewrites of S1 | 0.00 | −0.05 |
| C rewrites only | **+0.07** | **−0.06** |
| D fresh real text | −0.03 | −0.03 |
| E + rewrites of S2 | −0.01 | −0.05 |

Resolved: $2\sigma \approx 0.014$, all three seeds one sign. B − A and
E − B: no difference detected. B − C on real: −0.06.
:::

::: {.slide title="Training on a Generator's Output Shifts the Distribution"}
[Reading the result]{.kicker}

- A model that read the rephraser's style finds that style easy.
- Evaluating only on rewrites would make C the best arm.
- B gains on rewritten text (−0.05) and nothing on real text; WRAP, too,
  measured its gain on text other than its source.
- That is the pattern WRAP's "style matches the evaluation" predicts.

::: {.d2l-note}
Report: equal tokens, both validation sets, a noise threshold fixed in
advance, a bound computed before the runs.
:::
:::

::: {.slide title="Replacing Data with Samples Shrinks the Variance Geometrically"}
[Gaussian refit]{.kicker}

Fit $t$ draws $n$ points from fit $t-1$; maximum likelihood; $q = (n-1)/n$.

| | replace | accumulate |
|---|---|---|
| $\mathbb{E}[\hat\sigma_T^2]$ | $q^T\sigma^2$ | $\approx \sigma^2$ |
| $\operatorname{Var}(\hat\mu_T)$ | $\sigma^2(1 - q^T)$ | $< \frac{\pi^2}{6}\frac{\sigma^2}{n}$ |
| $n = 100$, $T = 20$, simulated | 0.80 and 0.17 | 0.99 and 0.0157 |

::: {.d2l-note}
Replace: a narrow distribution around a random point. Accumulate: the
mean's steps shrink as $1/t$.
:::
:::

::: {.slide title="A Bigram Model Loses Its Rare Words Type by Type"}
[Discrete toy: 200,000 words, 20,000 types]{.kicker}

| generation | types (replace) | rarest 10% alive | their mass | types (accumulate) |
|---|---|---|---|---|
| 0 | 20,000 | 100% | 1.0% | 20,000 |
| 1 | 16,001 | 62% | 1.0% | 20,000 |
| 5 | 10,238 | 27% | 1.0% | 20,000 |
| 10 | 7,655 | 17% | 1.1% | 20,000 |

::: {.d2l-note}
The rarest words occur once; a singleton misses a resample with probability
about $e^{-1}$, and a type with count 0 never returns.
:::
:::

::: {.slide title="Check: How Many Generations Halve the Variance?"}
[Check]{.kicker}

Replace mode, $n = 100$. After how many generations is
$\mathbb{E}[\hat\sigma_T^2] = \sigma^2/2$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $q^T = 1/2$, so $T = \ln 2 / \ln(100/99) \approx 69$. Slow per
step, certain in the end: $\hat\sigma_T^2 \to 0$ almost surely.
:::
:::
:::

::: {.slide title="Rights Follow Both the Source and the Generator"}
[Licences and disclosure]{.kicker}

- Source: a rewrite keeps the source's licence (FineWeb-Edu: ODC-By, plus
  Common Crawl's terms).
- Generator: Qwen3.5 is Apache-2.0; Llama 3's licence asks for "Llama" in
  the name of a model trained with its outputs.
- Hosted models: read the weights' licence and the service's terms.
- Record generator, revision, prompt hash, seed, drops; state the synthetic
  share (here 48%).

Descriptive, not legal advice.
:::

::: {.slide title="Recap"}
[Recap]{.kicker}

- Cheap checks: 70% of rewrites keep every number; "thirty" counts as lost.
- A strict 27B judge: 70% of rewrites add a claim; agreement with numbers 50%.
- Prompt conditioning at one temperature: distinct 4-grams 0.76 → 0.995.
- At T1 scale: B = A within noise (bound 12.5%); C loses 0.07 on real, wins
  0.06 on rewrites.
- Replace collapses ($q^T$), accumulate stays bounded; rare words go first.

**Next:** cutting text into tokens at scale.
:::
