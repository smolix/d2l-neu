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
import functools
import gc
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import huggingface_hub
import pandas as pd
import transformers
```

<!-- Placeholder (stage 2): the datasets, pinned to commits, the two fixed inputs, and the four measured results of this section's regeneration run. -->

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
# Fixed inputs: lm-evaluation-harness's scores on 200 ARC items
# (tools/oneshot/make_lm_eval_arc_fixture.py) and the held-out texts
# (tools/oneshot/make_lm_heldout_texts.py)
d2l.DATA_HUB['lm-eval-reference-arc-v1'] = (
    d2l.LM_DATA_URL + 'lm-eval-reference-arc-v1.json',
    'c0bdeb8593b0ba75406025d06e8cbea9016a2c71')
d2l.DATA_HUB['lm-heldout-texts-v1'] = (
    d2l.LM_DATA_URL + 'lm-heldout-texts-v1.json',
    '11b0ed4fead96910fba1078b2d9f5ee5b6a59b69')
# This section's measured results, written by a run with D2L_RERUN=1
d2l.DATA_HUB['lm-bpb-v1'] = (
    d2l.LM_DATA_URL + 'lm-bpb-v1.csv',
    '7c2ceeeda10b3f4ca6774821ed426c896442c3d3')
d2l.DATA_HUB['lm-gsm8k-samples-v1'] = (
    d2l.LM_DATA_URL + 'lm-gsm8k-samples-v1.jsonl',
    '11b561b6fb07d19016f5a00b4a5dcf35d3dbbb07')
d2l.DATA_HUB['lm-arc-generations-v1'] = (
    d2l.LM_DATA_URL + 'lm-arc-generations-v1.jsonl',
    '0b58a8a81e87794c5a564bd958142ede4288d195')
d2l.DATA_HUB['lm-arc-scores-v1'] = (
    d2l.LM_DATA_URL + 'lm-arc-scores-v1.parquet',
    'e4e1ea4b45a7a5288ae3b914459e60228b5e2931')
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

<!-- Placeholder (stage 2): the five checkpoints and the switches of the build (GPU, regeneration). -->

```{.python .input #scoring-setup}
%%tab pytorch
BASES = ['Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3.5-0.8B-Base']
PAIR = ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3.5-0.8B']  # the pre-declared pair
MODELS = [BASES[0], PAIR[0], BASES[1], PAIR[1], 'Qwen/Qwen3-1.7B']
GPU = torch.cuda.is_available()
# A job given a GPU (CUDA_VISIBLE_DEVICES set) must not fall back to the CPU
assert GPU or not os.environ.get('CUDA_VISIBLE_DEVICES'), (
    'a GPU is assigned, but PyTorch sees none')
RERUN = os.environ.get('D2L_RERUN', '0') == '1'  # regenerate the artifacts
assert GPU or not RERUN, 'regenerating the artifacts needs a GPU'
RERUN_DIR = os.path.join('..', 'data', 'lm', 'rerun')
# The regeneration's declared cuts, in order: D2L_RERUN_CUT=k applies the
# first k, chosen before the run from the times of one batch per job
# (tools/oneshot/time_scoring_rerun.py)
CUT = int(os.environ.get('D2L_RERUN_CUT', '0'))
assert 0 <= CUT <= 3 and (RERUN or CUT == 0), 'D2L_RERUN_CUT is 0 to 3'
N_REASON = 100 if CUT >= 1 else 200  # items of the reasoning arm
GSM_MAX_NEW = 320 if CUT >= 2 else 384  # token budget of a GSM8K sample
FP32_RUNS = CUT < 3  # the fp32 batch-size re-scorings
torch.backends.cuda.matmul.allow_tf32 = False  # fp32 means IEEE fp32
torch.backends.cudnn.allow_tf32 = False
ieee_fp32 = lambda: not (torch.backends.cuda.matmul.allow_tf32 or
                         torch.backends.cudnn.allow_tf32)
transformers.logging.set_verbosity_error()  # no notices or progress bars
transformers.logging.disable_progress_bar()
huggingface_hub.utils.disable_progress_bars()
print(f'GPU={GPU}, RERUN={RERUN}' + (f', cuts {CUT}' if CUT else ''))
```

<!-- Placeholder (stage 2): where the measured results live: registered files, or this run's regenerated ones. -->

```{.python .input #scoring-artifacts}
%%tab pytorch
def registered(key):
    """Local path of a registered artifact, or None before its upload."""
    return d2l.download_keyed(key) if key in d2l.DATA_HUB else None

def artifact(key, name):
    """The registered artifact; in the regeneration run before its first
    upload, the file name that this run wrote to RERUN_DIR."""
    path = registered(key)
    if path is None:
        assert RERUN, f'{key} is not registered: run with D2L_RERUN=1'
        path = os.path.join(RERUN_DIR, name)
    return path

def save_rerun(name, write):
    """write(path) a regenerated artifact to RERUN_DIR; print its sha1."""
    path = os.path.join(RERUN_DIR, name)
    os.makedirs(RERUN_DIR, exist_ok=True)
    write(path)
    with open(path, 'rb') as f:
        print(f'wrote {path}, sha1 {hashlib.sha1(f.read()).hexdigest()}')

def read_jsonl(path):  # one record per line, types as written
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f])

def write_jsonl(path, records):
    with open(path, 'w') as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + '\n'
                     for r in records)

def free():
    """Return the memory of deleted models to the GPU."""
    gc.collect()
    torch.cuda.empty_cache()

def peak():
    """Peak GPU memory since the last call, as text."""
    if not torch.cuda.is_available():
        return 'no GPU'
    gib = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return f'peak {gib:.1f} GiB'
```

<!-- Placeholder (stage 2): 500 seeded ARC-Challenge items, five balanced demonstrations, 100 seeded GSM8K problems. -->

```{.python .input #scoring-data}
%%tab pytorch
def arc_items(df):
    """(id, question, options, gold index) per row; the index also reads
    the items labelled 1 to 4 instead of A to D."""
    return [(r.id, r.question, list(r.choices['text']),
             list(r.choices['label']).index(r.answerKey))
            for r in df.itertuples()]

def subset(n_total, n, seed=0):  # prefix-stable: subset(N, m) starts
    return np.random.default_rng(seed).permutation(n_total)[:n]  # (N, n)

arc_test, arc_train = (arc_items(pd.read_parquet(d2l.download_keyed(
    f'arc-challenge-{split}'))) for split in ('test', 'train'))
ARC = [arc_test[i] for i in subset(len(arc_test), 500)]
ARC_IDS, GOLD = [it[0] for it in ARC], np.array([it[3] for it in ARC])
SHOTS, shot_ids = [], []  # in train order, gold letters A, B, C, D, A
for letter in 'ABCDA':
    it = next(it for it in arc_train if len(it[2]) == 4 and
              'ABCD'[it[3]] == letter and it[0] not in shot_ids)
    shot_ids.append(it[0])
    SHOTS.append(it[1:])
gsm_train = pd.read_parquet(d2l.download_keyed('gsm8k-train'))
gsm_test = pd.read_parquet(d2l.download_keyed('gsm8k-test'))
GSM_IDS = subset(len(gsm_test), 100)  # row indices of the test file
print(f'ARC-Challenge: {len(ARC)} of {len(arc_test):,} test items; options '
      f'per item {dict(collections.Counter(len(it[2]) for it in ARC))}')
print('demonstrations:', ', '.join(f'{i} ({"ABCD"[s[2]]})'
                                   for i, s in zip(shot_ids, SHOTS)))
print(f'GSM8K: {len(GSM_IDS)} of {len(gsm_test):,} test problems')
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

<!-- Placeholder (stage 2): one invented item in both formats, and the tokens each choice adds to its prompt. -->

```{.python .input #scoring-format-print}
%%tab pytorch
TOK = {m: transformers.AutoTokenizer.from_pretrained(
    m, revision=d2l.HF_REVISIONS[m]) for m in BASES}
tok = TOK[BASES[0]]
question = 'Which gas do plants take in from the air to make their food?'
options = ['oxygen', 'carbon dioxide', 'nitrogen', 'water vapor']
for fmt in ('cloze', 'letter'):
    prompt, choices = d2l.format_mc(question, options, fmt)
    n = len(tok(prompt, add_special_tokens=False).input_ids)
    print(prompt)
    for c in choices:  # the continuation: tokens beyond the prompt's
        cont = tok(prompt + c, add_special_tokens=False).input_ids[n:]
        print(f'    {c!r:17} ids {cont}, pieces '
              f'{[tok.decode(i) for i in cont]}')
```

<!-- Placeholder (stage 2): every prompt and choice of the 500 items, and a count of token merges across the boundary. -->

```{.python .input #scoring-format-merges}
%%tab pytorch
FORMATS = {  # (prompt, choices) of the 500 items in each scored format
    'cloze': [d2l.format_mc(q, o) for _, q, o, _ in ARC],
    'uncond': [('Answer:', d2l.format_mc(q, o)[1]) for _, q, o, _ in ARC],
    'letter0': [d2l.format_mc(q, o, 'letter') for _, q, o, _ in ARC],
    'letter5': [d2l.format_mc(q, o, 'letter', shots=SHOTS)
                for _, q, o, _ in ARC]}
merges = collections.defaultdict(dict)
for m, t in TOK.items():
    for fmt, items in FORMATS.items():
        heads = t([p for p, cs in items for _ in cs],
                  add_special_tokens=False).input_ids
        whole = t([p + c for p, cs in items for c in cs],
                  add_special_tokens=False).input_ids
        merges[fmt][m] = sum(w[:len(h)] != h for h, w in zip(heads, whole))
    assert all(len(t(' ' + a, add_special_tokens=False).input_ids) == 1
               for a in 'ABCDEFGHIJ')  # ' A' to ' J' are single tokens
print(f'prompt-choice pairs per format: '
      f'{sum(len(cs) for _, cs in FORMATS["cloze"]):,}; merges:')
print(pd.DataFrame(merges))
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

<!-- Placeholder (stage 2): three references for the scorer: a hand loop, one item at a time, and label_logprobs. -->

```{.python .input #scoring-check-def}
%%tab pytorch
def hand_loop(model, tok, prompt, choice):
    """log P(choice | prompt) from the log-softmax of the whole sequence."""
    n = len(tok(prompt, add_special_tokens=False).input_ids)
    ids = tok(prompt + choice, add_special_tokens=False).input_ids
    with torch.inference_mode():
        z = model(torch.tensor([ids], device=model.device)).logits[0]
    lp = z.float().log_softmax(-1)
    return sum(lp[t - 1, ids[t]].item() for t in range(n, len(ids)))

def scorer_errors(model, tok, items, letter_prompts):
    """Largest differences between score_choices and three references."""
    prompts, choices = zip(*items)
    hand = torch.tensor([[hand_loop(model, tok, p, c) for c in cs]
                         for p, cs in items])
    total = d2l.score_choices(model, tok, prompts, choices, 'sum',
                              batch_size=1)  # as the loop: no batch drift
    one = d2l.score_choices(model, tok, prompts, choices, batch_size=1)
    eight = d2l.score_choices(model, tok, prompts, choices, batch_size=8)
    letters = d2l.score_choices(model, tok, letter_prompts,  # one prompt
                                [' A', ' B', ' C', ' D'], 'sum',  # at a
                                batch_size=1)  # time: no batch drift
    labels = d2l.label_logprobs(model, tok, letter_prompts, list('ABCD'),
                                batch_size=1)
    return [('sum vs hand loop', (total - hand).abs().max().item(), 1e-4),
            ('batch 8 vs 1', (eight - one).abs().max().item(), 1e-4),
            ('letters vs label_logprobs',
             (letters.log_softmax(-1) - labels).abs().max().item(), 1e-5)]
```

<!-- Placeholder (stage 2): the checks in fp32 on the first eight four-option items, for both base checkpoints. -->

```{.python .input #scoring-check-scorer}
%%tab pytorch
assert ieee_fp32()  # the fp32 checks assume IEEE fp32 matrix products
first8 = [i for i, it in enumerate(ARC) if len(it[2]) == 4][:8]
items = [FORMATS['cloze'][i] for i in first8]
letter_prompts = [FORMATS['letter0'][i][0] for i in first8]
model, tok = d2l.load_hf(BASES[1], dtype=torch.float32)
rows = [(BASES[1], *r)
        for r in scorer_errors(model, tok, items, letter_prompts)]
del model
free()
base32, tok32 = d2l.load_hf(BASES[0], dtype=torch.float32)  # kept below
rows += [(BASES[0], *r)
         for r in scorer_errors(base32, tok32, items, letter_prompts)]
print(pd.DataFrame(rows, columns=['model', 'check', 'max error',
                                  'tolerance']).to_string(index=False))
assert all(err < tol for *_, err, tol in rows)
```

<!-- Placeholder (stage 2): the harness's per-item scores of the first 200 test items, and a comparison with ours. -->

```{.python .input #scoring-harness-compare}
%%tab pytorch
with open(d2l.download_keyed('lm-eval-reference-arc-v1')) as f:
    ref = json.load(f)
assert ref['dataset_revision'] == re.search(  # the pinned ARC commit
    r'/resolve/([0-9a-f]{40})/', d2l.DATA_HUB['arc-challenge-test'][0])[1]
assert ref['model'] == BASES[0]
assert ref['model_revision'] == d2l.HF_REVISIONS[BASES[0]]
first200 = arc_test[:200]  # in dataset order, as the harness ran them
assert [it[0] for it in first200] == [r['id'] for r in ref['items_data']]
prompts, choices = zip(*[d2l.format_mc(q, o) for _, q, o, _ in first200])

def against_harness(c, setting):
    """Largest |ll| difference and the predictions that differ, per norm."""
    diff, n_differ = 0.0, {}
    for i, r in enumerate(ref['items_data']):
        diff = max([diff] + [abs(c.ll[i, j].item() - v)
                             for j, v in enumerate(r['ll'])])
    for norm, key in (('sum', 'pred_acc'), ('char', 'pred_acc_norm')):
        s = c.ll / (1 if norm == 'sum' else c.n_chars)
        pred = s.argmax(-1).tolist()
        bad = [(r['id'], p, r[key], (s[i, p] - s[i, r[key]]).item())
               for i, (r, p) in enumerate(zip(ref['items_data'], pred))
               if p != r[key]]
        n_differ[norm] = len(bad)
        for item, p, h, gap in bad:
            print(f'    {setting}, {norm}: {item} ours {"ABCDE"[p]}, '
                  f'harness {"ABCDE"[h]}, margin {gap:.2e}')
    print(f'{setting}: max |ll difference| {diff:.1e}; predictions that '
          f'differ: {n_differ}')
    return diff, n_differ
```

<!-- Placeholder (stage 2): the scorer against lm-evaluation-harness, in fp32 and (on a GPU) in bf16. -->

```{.python .input #scoring-harness-fixture}
%%tab pytorch
assert ieee_fp32()
diff, n_differ = against_harness(d2l.score_choices(
    base32, tok32, prompts, choices, batch_size=1, return_components=True),
    'fp32, batch 1')
assert diff < 2e-3 and max(n_differ.values()) <= 2
if GPU:
    model, _ = d2l.load_hf(BASES[0])  # bf16
    against_harness(d2l.score_choices(model, tok32, prompts, choices,
                                      return_components=True),
                    'bf16, batch 8')
    del model
    free()
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

<!-- Placeholder (stage 2): the texts: wikitext-2 test in articles, and the two held-out texts. -->

```{.python .input #scoring-texts}
%%tab pytorch
with open(d2l.download_keyed('lm-heldout-texts-v1')) as f:
    HELD = {t['name']: unicodedata.normalize('NFC', t['text'])
            for t in json.load(f)['texts']}
title = re.compile(r' = [^=].* = ')  # a top-level heading starts an article;
WIKI, prev = [], ''  # it follows an empty row (text rows can match too)
for line in pd.read_parquet(d2l.download_keyed('wikitext-2-raw-test')).text:
    if prev == '' and title.fullmatch(line.rstrip('\n')):
        WIKI.append('')
    if WIKI:
        WIKI[-1] += line
    prev = line
WIKI = [unicodedata.normalize('NFC', a) for a in WIKI]
assert len(WIKI) == 60  # the articles of wikitext-2's test split
BPB_TEXTS = {'wikitext-2': WIKI, 'held-out prose': [HELD['prose']],
             'held-out code': [HELD['code']]}
for name, texts in BPB_TEXTS.items():
    print(f'{name}: {len(texts)} text(s), '
          f'{sum(len(t.encode("utf-8")) for t in texts):,} bytes')
```

<!-- Placeholder (stage 2): bits per byte against full logits, one window at a time, with every position counted. -->

```{.python .input #scoring-bpb-check}
%%tab pytorch
def naive_nll(model, tok, text, ctx_len, stride):
    """Total NLL in nats from full logits, window by window as
    bits_per_byte places them, and how often each position is scored."""
    seq = [_lm_boundary_id(tok)] + tok(text, add_special_tokens=False
                                       ).input_ids
    nll, times, done = 0.0, np.zeros(len(seq), dtype=int), 0
    while done < len(seq) - 1:
        end = min(ctx_len if done == 0 else done + stride, len(seq) - 1)
        start = max(end - ctx_len, 0)  # seq[start:end] predicts the next
        with torch.inference_mode():
            z = model(torch.tensor([seq[start:end]], device=model.device))
        lp = z.logits[0].float().log_softmax(-1).cpu()
        t = torch.arange(done + 1, end + 1)  # the targets scored here
        nll -= lp[t - 1 - start, torch.tensor(seq)[t]].sum().item()
        times[t] += 1
        done = end
    return nll, times

assert ieee_fp32()
ids = tok32(HELD['prose'], add_special_tokens=False).input_ids
for n, ctx_len, stride in ((300, 512, 512), (3000, 512, 256)):
    text = tok32.decode(ids[:n])
    bpb = d2l.bits_per_byte(base32, tok32, text, ctx_len, stride)
    nll, times = naive_nll(base32, tok32, text, ctx_len, stride)
    full = nll / (math.log(2) * len(text.encode('utf-8')))
    assert times[0] == 0 and (times[1:] == 1).all()  # the reference scores
    # each token once, so matching it to 1e-6 tests bits_per_byte's windows
    print(f'{len(times) - 1:,} tokens, ctx {ctx_len}, stride {stride}: '
          f'bpb {bpb:.6f}, full logits {full:.6f}, relative difference '
          f'{abs(bpb - full) / full:.1e}')
    assert abs(bpb - full) / full < 1e-6
del base32
free()
```

<!-- Placeholder (stage 2): one row of the bits-per-byte table, and its agreement with the registered values. -->

```{.python .input #scoring-bpb-def}
%%tab pytorch
def bpb_row(model_id, model, tok, text, ctx_len, stride):
    """Tokens, bytes, token perplexity and bits per byte of BPB_TEXTS[text]."""
    texts = BPB_TEXTS[text]
    T = sum(map(len, tok(texts, add_special_tokens=False).input_ids))
    B = sum(len(t.encode('utf-8')) for t in texts)
    bpb = d2l.bits_per_byte(model, tok, texts, ctx_len, stride)
    nll = bpb * math.log(2) * B  # nats
    vocab = d2l._lm_unwrap(model).config.get_text_config().vocab_size
    return dict(model=model_id, vocab=vocab,  # rows of the output layer
                boundary=_lm_boundary_id(tok), text=text, ctx=ctx_len,
                stride=stride, T=T, B=B, nll=nll, bpb=bpb,
                ppl_tok=math.exp(nll / T))

def bpb_agreement(rows):
    path = registered('lm-bpb-v1')
    if path is None:
        return print('lm-bpb-v1 is not registered yet')
    both = pd.DataFrame(rows).merge(pd.read_csv(path), suffixes=('', '_v1'),
                                    on=['model', 'text', 'ctx', 'stride'])
    print(f'{len(both)} of {len(rows)} rows in lm-bpb-v1; max |bpb '
          f'difference| {(both.bpb - both.bpb_v1).abs().max():.1e}')

def bpb_table(rows):
    """The registered table lm-bpb-v1; before its upload, this run's rows."""
    path = registered('lm-bpb-v1')
    if path is None and not rows:
        path = artifact('lm-bpb-v1', 'lm-bpb.csv')
    return pd.DataFrame(rows) if path is None else pd.read_csv(path)
```

<!-- Placeholder (stage 2): two base models, three texts: tokens per byte, token perplexity, bits per byte. -->

```{.python .input #scoring-bpb-models}
%%tab pytorch
print('boundary token:', {m: _lm_boundary_id(t) for m, t in TOK.items()})
bpb_rows = []
if GPU:
    for m in BASES:
        model, tok = d2l.load_hf(m)
        bpb_rows += [bpb_row(m, model, tok, t, 2048, 1024) for t in BPB_TEXTS]
        del model
        free()
    bpb_agreement(bpb_rows)
bpb = bpb_table(bpb_rows)
main = bpb[(bpb.ctx == 2048) & (bpb.stride == 1024)].assign(
    tpb=lambda d: d['T'] / d['B'])  # tokens per byte
print(main.set_index(['text', 'model'])[['T', 'B', 'tpb', 'ppl_tok', 'bpb']]
      .round(3).to_string())
tpb = main.pivot(index='text', columns='model', values='tpb')
print('tokens per byte, Qwen3 over Qwen3.5:',
      (tpb[BASES[0]] / tpb[BASES[1]]).round(3).to_dict())
```

<!-- Placeholder (stage 2): wikitext-2 under four windowings of Qwen3-0.6B-Base. -->

```{.python .input #scoring-bpb-windows}
%%tab pytorch
WINDOWS = [(512, 512), (512, 256), (2048, 2048)]  # (ctx, stride)
if GPU:
    model, tok = d2l.load_hf(BASES[0])
    rows = [bpb_row(BASES[0], model, tok, 'wikitext-2', c, s)
            for c, s in WINDOWS]
    del model
    free()
    bpb_agreement(rows)
    bpb_rows += rows
    if RERUN:
        save_rerun('lm-bpb.csv', lambda path: pd.DataFrame(bpb_rows).to_csv(
            path, index=False, float_format='%.10g'))
bpb = bpb_table(bpb_rows)
w = bpb[(bpb.model == BASES[0]) & (bpb.text == 'wikitext-2')]
print(w.assign(step=np.where(w.stride == w.ctx, 'stride = ctx',
                             'stride = ctx/2'))
      .pivot(index='ctx', columns='step', values='bpb').round(4))
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

<!-- Placeholder (stage 2): 4-shot GSM8K prompts as a conversation, and three ways to read a number off a solution. -->

```{.python .input #scoring-gsm8k-prompts}
%%tab pytorch
GSM_DEMOS = [(q, re.sub(r'<<[^>]*>>', '', a))  # calculator notes removed
             for q, a in zip(gsm_train.question[:4], gsm_train.answer[:4])]
GSM_PROMPTS = [d2l.fewshot_prompt(GSM_DEMOS, q, template='chat_multi')
               for q in gsm_test.question.iloc[GSM_IDS]]
GSM_GOLD = [d2l.extract_answer(a) for a in gsm_test.answer.iloc[GSM_IDS]]
GSM_SAMPLER = dict(chat=True, enable_thinking=False, strategy='sample',
                   temperature=0.7, p=0.8, k=20, max_new_tokens=GSM_MAX_NEW)

def hash_only(text):
    """The number after the last '####', GSM8K's answer line, or None."""
    found = re.findall(r'####\s*(\$?-?[\d,]*\.?\d+)', text)
    return d2l.extract_answer('#### ' + found[-1]) if found else None

EXTRACTORS = {'#### only': hash_only,
              'strict': lambda text: d2l.extract_answer(text, strict=True),
              'last number': d2l.extract_answer}
tok = transformers.AutoTokenizer.from_pretrained(
    PAIR[0], revision=d2l.HF_REVISIONS[PAIR[0]])
print(tok.apply_chat_template(GSM_PROMPTS[0], tokenize=False,
                              add_generation_prompt=True,
                              enable_thinking=False))
```

<!-- Placeholder (stage 2): eight samples per problem from each model of the pair (regenerated only under RERUN). -->

```{.python .input #scoring-gsm8k-pool}
%%tab pytorch
def gsm_records(model_id, gens, j):
    return [dict(model=model_id, revision=d2l.HF_REVISIONS[model_id],
                 problem_id=int(i), sample_idx=j, seed=j,
                 settings=GSM_SAMPLER, text=g.text,
                 n_tokens=len(g.token_ids), finish_reason=g.finish_reason,
                 **{'answer_' + k: f(g.text) for k, f in
                    zip(['hash', 'strict', 'loose'], EXTRACTORS.values())},
                 gold=gold, correct=EXTRACTORS['strict'](g.text) == gold)
            for i, g, gold in zip(GSM_IDS, gens, GSM_GOLD)]

if RERUN:
    records, tic = [], time.time()
    for m in PAIR:
        model, tok = d2l.load_hf(m)
        for j in range(8):  # sample j of every problem, with seed j
            records += gsm_records(m, d2l.generate(
                model, tok, GSM_PROMPTS, seed=j, batch_size=50,
                **GSM_SAMPLER), j)
        del model
        free()
    print(f'{len(records):,} samples in {(time.time() - tic) / 60:.1f} '
          f'min, {peak()}')
    save_rerun('lm-gsm8k-samples.jsonl',
               lambda path: write_jsonl(path, records))
    path = registered('lm-gsm8k-samples-v1')
    if path is not None:
        both = pd.DataFrame(records).merge(
            read_jsonl(path), on=['model', 'problem_id', 'sample_idx'])
        print(f'correctness agrees with lm-gsm8k-samples-v1 on '
              f'{(both.correct_x == both.correct_y).sum()} of {len(both)}')
pool = read_jsonl(artifact('lm-gsm8k-samples-v1', 'lm-gsm8k-samples.jsonl'))
print(pd.DataFrame({
    'sample-0 accuracy': pool[pool.sample_idx == 0].groupby('model').correct
    .mean(), 'truncated': (pool.finish_reason == 'length').groupby(
        pool.model).mean(), 'mean tokens': pool.groupby('model').n_tokens
    .mean()}).round(3))
```

<!-- Placeholder (stage 2): the same 1,600 generations read by three extractors. -->

```{.python .input #scoring-extractors}
%%tab pytorch
answers = pd.DataFrame({name: pool.text.map(f)
                        for name, f in EXTRACTORS.items()})
right = answers.apply(lambda col: col == pool.gold)
print(right.groupby(pool.model).mean().round(3))
differ = answers.nunique(axis=1, dropna=False) > 1
print(f'extractors disagree on {differ.sum()} of {len(pool):,} generations; '
      f'the last-number fallback alone credits '
      f'{(right["last number"] & ~right["strict"]).sum()}')
```

<!-- Placeholder (stage 2): a provisional bootstrap, until d2l.paired_bootstrap in the next section. -->

```{.python .input #scoring-boot}
%%tab pytorch
@functools.cache  # the same resamples for every statistic over n units
def boot_counts(n, n_boot=10_000, seed=0):
    """(n_boot, n) resampling counts: how often each unit is drawn."""
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n),
                                                   size=n_boot)

def boot(a, b=None, n_boot=10_000, seed=0):
    """Mean of a, or of a - b paired by unit, and its 95% percentile
    interval over n_boot resamples of the units."""
    d = np.asarray(a, float) - (0 if b is None else np.asarray(b, float))
    means = boot_counts(len(d), n_boot, seed) @ d / len(d)
    return d.mean(), *np.percentile(means, [2.5, 97.5])
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

<!-- Placeholder (stage 2): pass@k, its plug-in estimate and majority voting on the pool, with intervals over problems; the plot below. -->

```{.python .input #scoring-passk-pool}
%%tab pytorch
K = [1, 2, 4, 8]

def majority(answers):
    """The most frequent parsed answer, ties to the first one sampled."""
    parsed = [a for a in answers if a is not None]
    counts = collections.Counter(parsed)
    return max(parsed, key=counts.__getitem__) if parsed else None

est, rows = {}, []
for m in PAIR:
    g = pool[pool.model == m].sort_values('sample_idx').groupby('problem_id')
    c = g.correct.sum().to_numpy()  # correct samples of 8, per problem
    maj = (g.answer_strict.agg(majority) == g.gold.first()).to_numpy(float)
    for k in K:
        est[m, k] = boot(d2l.pass_at_k(8, c, k))
        est[m, k, 'plug-in'] = np.mean(1 - (1 - c / 8) ** k)
    est[m, 'maj'] = boot(maj)
    rows.append({'model': m, **{f'pass@{k}': est[m, k] for k in K},
                 'maj@8': est[m, 'maj'],
                 'maj@8 - pass@1': boot(maj, d2l.pass_at_k(8, c, 1))})
show = lambda e: f'{e[0]:.3f} [{e[1]:.3f}, {e[2]:.3f}]'
print(pd.DataFrame(rows).set_index('model').map(show).T.to_string())
print('plug-in:', {m: [round(float(est[m, k, 'plug-in']), 3) for k in K]
                   for m in PAIR})
d2l.plot([K, K, [8], [8]], [[est[m, k, 'plug-in'] for k in K] for m in PAIR]
         + [[est[m, 'maj'][0]] for m in PAIR],
         fmts=('C0--', 'C1--', 'C0D', 'C1D'), figsize=(4.5, 3))
d2l.plot_intervals(K, {m.removeprefix('Qwen/'): tuple(np.array(
    [est[m, k] for k in K]).T) for m in PAIR}, 'k', 'pass@k', kind='band',
    xscale='log2', axes=d2l.plt.gca())
d2l.plt.gca().autoscale(axis='y')  # d2l.plot fixed the range: fit the bands
```

<!-- Placeholder (stage 2): how much of the band's width comes from sampling and how much from the choice of problems. -->

```{.python .input #scoring-passk-variance-split}
%%tab pytorch
rows = []
for m in PAIR:
    c = pool[pool.model == m].groupby('problem_id').correct.sum().to_numpy()
    M, p = len(c), c / 8
    for k in (1, 4):
        total = d2l.pass_at_k(8, c, k).var(ddof=1) / M
        within = (p * (1 - p) / 7 if k == 1 else  # unbiased at k = 1
                  np.array([d2l.pass_at_k_variance(8, q, k) for q in p])
                  ).mean() / M
        rows.append(dict(model=m, k=k, total=total, within=within,
                         between=total - within, within_share=within / total,
                         within_estimate='exact' if k == 1 else 'plug-in'))
print(pd.DataFrame(rows).to_string(index=False, float_format='%.2e'))
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

<!-- Placeholder (stage 2): the two generative arms on ARC: what the chat turn asks, each arm's settings, and how a reply is read. -->

```{.python .input #scoring-arc-arms}
%%tab pytorch
ASK = 'Give the letter of the correct option as \\boxed{X}.'
ARC_CHAT = [d2l.format_mc(q, o, 'letter')[0].removesuffix('Answer:') + ASK
            for _, q, o, _ in ARC]
ARMS = {  # arm: (models, number of subset items, settings of generate)
    'direct': ([PAIR[0], PAIR[1], 'Qwen/Qwen3-1.7B'], 500, dict(
        strategy='greedy', enable_thinking=False, max_new_tokens=128)),
    'reasoning': ([PAIR[0], 'Qwen/Qwen3-1.7B'], N_REASON, dict(
        strategy='sample', temperature=0.6, p=0.95, k=20, seed=0,
        enable_thinking=True, max_new_tokens=1024))}

def boxed_letter(text, n_options, finished=True):
    """The option a finished reply commits to in its last \\boxed{} after
    any thinking, or None; a reply cut at its token budget commits to none."""
    if not finished:
        return None
    s = d2l.extract_answer(text.split('</think>')[-1], 'boxed') or ''
    m = re.match(r'(?:\\text\{)?\(?([A-Z])(?![A-Za-z])', s)
    return m[1] if m and ord(m[1]) - ord('A') < n_options else None
```

<!-- Placeholder (stage 2): the generated answers (regenerated only under RERUN): accuracy, parsing and length per model and arm. -->

```{.python .input #scoring-arc-generate}
%%tab pytorch
if RERUN:
    records, tic = [], time.time()
    for m in dict.fromkeys(ARMS['direct'][0] + ARMS['reasoning'][0]):
        model, tok = d2l.load_hf(m)
        for arm, (models, n, settings) in ARMS.items():
            if m in models:
                out = d2l.generate(model, tok, ARC_CHAT[:n], chat=True,
                                   batch_size=50, **settings)
                letters = [boxed_letter(g.text, len(it[2]),
                                        g.finish_reason != 'length')
                           for it, g in zip(ARC, out)]
                records += [dict(
                    model=m, revision=d2l.HF_REVISIONS[m], arm=arm,
                    item_id=it[0], settings=settings,
                    seed=settings.get('seed'), text=g.text,
                    n_tokens=len(g.token_ids), finish_reason=g.finish_reason,
                    letter=a, correct=a == 'ABCDE'[it[3]])
                    for it, g, a in zip(ARC, out, letters)]
        del model
        free()
    print(f'{len(records):,} replies in {(time.time() - tic) / 60:.1f} min, '
          f'{peak()}')
    save_rerun('lm-arc-generations.jsonl',
               lambda path: write_jsonl(path, records))
arc_gen = read_jsonl(artifact('lm-arc-generations-v1',
                              'lm-arc-generations.jsonl'))
n_opts = dict(zip(ARC_IDS, (len(it[2]) for it in ARC)))
at_cap = arc_gen.finish_reason == 'length'
boxed = np.array([boxed_letter(t, n_opts[i]) is not None  # as if finished
                  for t, i in zip(arc_gen.text, arc_gen.item_id)])
summary = arc_gen.assign(  # direct replies cut after a boxed letter
    invalid=arc_gen.letter.isna() & ~at_cap, truncated=at_cap,
    boxed_then_cut=(at_cap & boxed).astype(float).where(
        arc_gen.arm == 'direct')).groupby(['model', 'arm'])
print(pd.DataFrame({
    'accuracy': summary.correct.mean(),
    'among parsed': arc_gen[arc_gen.letter.notna()].groupby(
        ['model', 'arm']).correct.mean(),
    'invalid': summary.invalid.mean(),
    'truncated': summary.truncated.mean(),
    'boxed, then cut': summary.boxed_then_cut.mean(),
    'mean tokens': summary.n_tokens.mean()}).round(3).to_string())
```

<!-- Placeholder (stage 2): the per-choice scores of the grid as a long table, and the conventions read from it. -->

```{.python .input #scoring-grid-def}
%%tab pytorch
def score_long(model, tok, model_id, fmt, batch_size=8):
    """Components of every choice of the 500 items in format fmt, one row
    per (item, choice), as lm-arc-scores-v1 stores them."""
    prompts, choices = zip(*FORMATS[fmt])
    c = d2l.score_choices(model, tok, prompts, choices,
                          batch_size=batch_size, return_components=True)
    i, j = np.nonzero(c.n_tokens.numpy())
    return pd.DataFrame({
        'model': model_id, 'revision': d2l.HF_REVISIONS[model_id],
        'fmt': fmt, 'item_id': np.array(ARC_IDS)[i], 'choice': j,
        'gold': GOLD[i],
        **{f: v.numpy()[i, j] for f, v in c._asdict().items()}})

def matrix(arc, model, fmt, col='ll'):
    """(items, 5) array of one column, in subset order; nan beyond an
    item's options."""
    d = arc[(arc.model == model) & (arc.fmt == fmt)]
    return d.pivot(index='item_id', columns='choice', values=col).reindex(
        index=ARC_IDS, columns=range(5)).to_numpy(float)
```

<!-- Placeholder (stage 2): the eight conventions, and how each turns the stored components into a prediction. -->

```{.python .input #scoring-conventions}
%%tab pytorch
CONVENTIONS = ['cloze-sum', 'cloze-token', 'cloze-char', 'cloze-PMI',
               'letter-0', 'letter-5', 'gen-direct', 'gen-reasoning']
CLOZE = {'cloze-sum': None, 'cloze-token': 'n_tokens',
         'cloze-char': 'n_chars', 'cloze-byte': 'n_bytes'}

def convention_scores(arc, model, conv, col='ll'):
    """Scores (items, 5) of a likelihood convention, nan beyond options."""
    if conv in CLOZE:
        s = matrix(arc, model, 'cloze', col)
        return s if CLOZE[conv] is None else s / matrix(arc, model, 'cloze',
                                                        CLOZE[conv])
    if conv == 'cloze-PMI':  # minus log P(choice | 'Answer:')
        return (matrix(arc, model, 'cloze', col) -
                matrix(arc, model, 'uncond', col))
    return matrix(arc, model, {'letter-0': 'letter0',
                               'letter-5': 'letter5'}[conv], col)

def predict(s):  # first index of the largest score, as score_choices
    return np.nan_to_num(s, nan=-np.inf).argmax(1)

def margin(s):  # best minus second-best score
    top = -np.sort(-np.nan_to_num(s, nan=-np.inf), axis=1)
    return top[:, 0] - top[:, 1]
```

<!-- Placeholder (stage 2): the grid, re-scored live on a GPU and compared with the registered scores. -->

```{.python .input #scoring-grid}
%%tab pytorch
path = registered('lm-arc-scores-v1')
v1 = None if path is None else pd.read_parquet(path)
if GPU:
    parts = []
    for m in MODELS:
        tic = time.time()
        model, tok = d2l.load_hf(m)
        parts.append(pd.concat([score_long(model, tok, m, fmt)
                                for fmt in FORMATS]))
        del model
        free()
        print(f'{m}: {time.time() - tic:.0f} s, {peak()}')
        for fmt in FORMATS if v1 is not None else ():
            new, old = matrix(parts[-1], m, fmt), matrix(v1, m, fmt)
            bad = np.flatnonzero(predict(new) != predict(old))
            print(f'    {fmt}: {len(ARC) - len(bad)} of {len(ARC)} '
                  f'predictions as in lm-arc-scores-v1' + ''.join(
                      f'; {ARC_IDS[i]} (margin {margin(old)[i]:.3f})'
                      for i in bad))
    scores = pd.concat(parts, ignore_index=True)
else:
    print('no GPU: the grid is read from lm-arc-scores-v1')
```

<!-- Placeholder (stage 2): the same items at another batch size and in fp32 (regenerated only under RERUN), and what the drift does to predictions. -->

```{.python .input #scoring-batch-size}
%%tab pytorch
BATCH_RUNS = [(PAIR[0], torch.bfloat16, 1, 'll_bs1'),  # model, dtype,
              (PAIR[1], torch.bfloat16, 1, 'll_bs1')]  # batch size, column
if FP32_RUNS:
    BATCH_RUNS += [(PAIR[0], torch.float32, 1, 'll_fp32_bs1'),
                   (PAIR[0], torch.float32, 8, 'll_fp32_bs8')]
if RERUN:
    assert ieee_fp32()
    key, extra, tic = ['model', 'fmt', 'item_id', 'choice'], {}, time.time()
    for m, dtype, size, col in BATCH_RUNS:
        model, tok = d2l.load_hf(m, dtype=dtype)
        extra.setdefault(col, []).extend(
            score_long(model, tok, m, fmt, size)[key + ['ll']].rename(
                columns={'ll': col}) for fmt in ('cloze', 'letter0'))
        del model
        free()
    print(f'{len(BATCH_RUNS)} re-scorings in {(time.time() - tic) / 60:.1f} '
          f'min, {peak()}')
    for col, parts in extra.items():
        scores = scores.merge(pd.concat(parts), on=key, how='left')
    save_rerun('lm-arc-scores.parquet',
               lambda path: scores.to_parquet(path, index=False))
arc = pd.read_parquet(artifact('lm-arc-scores-v1', 'lm-arc-scores.parquet'))
rows = []
for name, m, a, b in [('bf16, batch 8 vs 1', PAIR[0], 'll', 'll_bs1'),
                      ('bf16, batch 8 vs 1', PAIR[1], 'll', 'll_bs1'),
                      ('fp32, batch 1 vs 8', PAIR[0], 'll_fp32_bs1',
                       'll_fp32_bs8')]:
    if b not in arc:
        print(f'{name}: cut from the regeneration run')
        continue
    for conv in ('cloze-sum', 'letter-0'):
        sa, sb = (convention_scores(arc, m, conv, c) for c in (a, b))
        change = np.nanmax(np.abs(sa - sb))
        rows.append(dict(setting=name, model=m, convention=conv,
                         changed=(predict(sa) != predict(sb)).sum(),
                         max_change=change,
                         below_change=(margin(sa) < change).sum()))
print(pd.DataFrame(rows).to_string(index=False))
```

<!-- Placeholder (stage 2): the drift of the pair's accuracy difference between batch sizes, per convention. -->

```{.python .input #scoring-drift}
%%tab pytorch
def pair_difference(conv, col):
    acc = [(predict(convention_scores(arc, m, conv, col)) == GOLD).mean()
           for m in PAIR]
    return acc[1] - acc[0]

measured = {conv: abs(pair_difference(conv, 'll') -
                      pair_difference(conv, 'll_bs1'))
            for conv in ('cloze-sum', 'cloze-token', 'cloze-char',
                         'letter-0')}
DRIFT = {c: measured.get(c, max(measured.values())) for c in CONVENTIONS}
print('drift of the pair difference, batch 8 vs 1:',
      {c: round(float(v), 4) for c, v in measured.items()})
print('the other conventions get the largest measured drift')
```

<!-- Placeholder (stage 2): correctness of every checkpoint under every convention, item by item. -->

```{.python .input #scoring-verdicts}
%%tab pytorch
def gen_column(model, arm, col):
    """One column of a generative arm per subset item (nan if not run)."""
    d = arc_gen[(arc_gen.model == model) & (arc_gen.arm == arm)]
    return d.set_index('item_id')[col].reindex(ARC_IDS)

def correct(model, conv):
    """1 or 0 per subset item; nan where the convention did not run."""
    if conv.startswith('gen-'):
        return gen_column(model, conv[4:], 'correct').to_numpy(float)
    return (predict(convention_scores(arc, model, conv)) == GOLD
            ).astype(float)

CORRECT = {(m, c): correct(m, c) for m in MODELS for c in CONVENTIONS}
```

<!-- Placeholder (stage 2): one item that Qwen3-0.6B answers differently under three conventions. -->

```{.python .input #scoring-one-item}
%%tab pytorch
m = PAIR[0]
verdicts = pd.DataFrame({
    'cloze-sum': [chr(65 + i) for i in predict(convention_scores(
        arc, m, 'cloze-sum'))],
    'letter-0': [chr(65 + i) for i in predict(convention_scores(
        arc, m, 'letter-0'))],
    'gen-direct': gen_column(m, 'direct', 'letter').to_numpy()})
committed = verdicts['gen-direct'].notna().to_numpy()  # a parsed reply
n_verdicts = np.where(committed, verdicts.nunique(axis=1), 0)
rule = 3 if (n_verdicts == 3).any() else 2  # three letters, else two
assert (n_verdicts >= rule).any(), 'no item with two different letters'
i = int(np.argmax(n_verdicts >= rule))
_, q, opts, gold = ARC[i]
print(q + ''.join(f'\n{a}. {o}' for a, o in zip('ABCDE', opts)))
print(f'gold {"ABCDE"[gold]}; Qwen3-0.6B: ' + ', '.join(
    f'{k} {v}' for k, v in verdicts.iloc[i].items()))
print(f'first item with {rule} different letters ({ARC_IDS[i]}); items '
      f'with at least two: {(n_verdicts > 1).sum()} of {committed.sum()} '
      f'with a parsed reply')
```

<!-- Placeholder (stage 2): accuracy with item-bootstrap intervals per checkpoint and convention, and the gold answer's bits per byte. -->

```{.python .input #scoring-convention-table}
%%tab pytorch
ACC = {key: boot(c[~np.isnan(c)]) for key, c in CORRECT.items()
       if not np.isnan(c).all()}

def gold_bpb(model):
    """Bits per byte of each item's gold answer in the cloze format."""
    i = np.arange(len(ARC))
    return -matrix(arc, model, 'cloze')[i, GOLD] / (
        matrix(arc, model, 'cloze', 'n_bytes')[i, GOLD] * math.log(2))

BPB_GOLD = {m: gold_bpb(m) for m in MODELS}
table = pd.DataFrame({m.removeprefix('Qwen/'): {
    c: show(ACC[m, c]) for c in CONVENTIONS if (m, c) in ACC}
    for m in MODELS}).reindex(CONVENTIONS).fillna('')
table.loc['gold bpb'] = [show(boot(BPB_GOLD[m])) for m in MODELS]
print(table.to_string())
print("predictions where 'char' and 'byte' differ, five checkpoints:",
      sum((predict(convention_scores(arc, m, 'cloze-char')) != predict(
          convention_scores(arc, m, 'cloze-byte'))).sum() for m in MODELS))
```

<!-- Placeholder (stage 2): the pair's paired differences per convention, the flip verdict, and gold-answer bpb against accuracy. -->

```{.python .input #scoring-pair-diff}
%%tab pytorch
def paired(a, b):  # mean of a - b on the items both cover, with interval
    ok = ~(np.isnan(a) | np.isnan(b))
    return boot(a[ok], b[ok])

PAIR_CONVS = CONVENTIONS[:7]  # the conventions both models ran
PAIR_DIFF = {c: paired(CORRECT[PAIR[1], c], CORRECT[PAIR[0], c])
             for c in PAIR_CONVS}  # Qwen3.5-0.8B minus Qwen3-0.6B
ahead = {c: PAIR[1] if lo > DRIFT[c] else PAIR[0] if hi < -DRIFT[c] else
         'neither' for c, (_, lo, hi) in PAIR_DIFF.items()}
print(pd.DataFrame({'difference': {c: show(e) for c, e in PAIR_DIFF.items()},
                    'drift': {c: DRIFT[c] for c in PAIR_CONVS},
                    'ahead': ahead}).to_string())
print('ranking flips:', len(set(ahead.values()) - {'neither'}) == 2)
for name, (d, lo, hi) in [('cloze-char accuracy', PAIR_DIFF['cloze-char']),
                          ('gold bpb', boot(BPB_GOLD[PAIR[1]],
                                            BPB_GOLD[PAIR[0]]))]:
    print(f'{name}: {show((d, lo, hi))}, |difference| over half-width '
          f'{abs(d) / ((hi - lo) / 2):.2f}')
```

<!-- Placeholder (stage 2): the pre-declared contrast, letter-0 minus cloze-sum per model and family, and reasoning against direct answers. -->

```{.python .input #scoring-contrasts}
%%tab pytorch
gain = lambda m: CORRECT[m, 'letter-0'] - CORRECT[m, 'cloze-sum']
print('pair, (letter-0 - cloze-sum) difference of differences:',
      show(boot(gain(PAIR[1]), gain(PAIR[0]))))
for m in MODELS:
    print(f'{m}: letter-0 - cloze-sum {show(boot(gain(m)))}')
for base, post in zip(BASES, PAIR):
    print(f'{post} - {base}, (letter-0 - cloze-sum): '
          f'{show(boot(gain(post), gain(base)))}')
for m in ARMS['reasoning'][0]:  # on the first n subset items
    n = gen_column(m, 'reasoning', 'n_tokens').notna().sum()
    a, b = (gen_column(m, arm, 'letter').iloc[:n] for arm in ARMS)
    tokens = [gen_column(m, arm, 'n_tokens').iloc[:n].mean()
              for arm in ARMS]
    d = paired(CORRECT[m, 'gen-reasoning'], CORRECT[m, 'gen-direct'])
    print(f'{m}: reasoning - direct {show(d)}; letters differ on '
          f'{(a.fillna("") != b.fillna("")).sum()} of {n}; mean tokens '
          f'{tokens[1]:.0f} (reasoning) and {tokens[0]:.0f} (direct)')
```

<!-- Placeholder (stage 2): how often sum and char disagree, by the spread of option lengths. -->

```{.python .input #scoring-length-spread}
%%tab pytorch
lengths = matrix(arc, MODELS[0], 'cloze', 'n_chars')  # same for all models
spread = np.nanmax(lengths, 1) / np.nanmin(lengths, 1)
tercile = pd.qcut(pd.Series(spread).rank(method='first'), 3,
                  labels=False).to_numpy()  # ties broken by item order
flips = np.stack([predict(convention_scores(arc, m, 'cloze-sum')) !=
                  predict(convention_scores(arc, m, 'cloze-char'))
                  for m in MODELS]).astype(float)  # (checkpoints, items)
pooled, C = flips.mean(0), boot_counts(len(ARC))
ci = lambda point, reps: show((point, *np.percentile(reps, [2.5, 97.5])))
rates = []  # per resample; items are drawn with all checkpoints together
for t in range(3):
    inside = tercile == t
    rates.append(C[:, inside] @ pooled[inside] / C[:, inside].sum(1))
    print(f'tercile {t + 1}, spread {spread[inside].min():.2f} to '
          f'{spread[inside].max():.2f}: '
          f'{ci(pooled[inside].mean(), rates[t])}')
print('top - bottom:', ci(pooled[tercile == 2].mean() -
                          pooled[tercile == 0].mean(), rates[2] - rates[0]))
print(pd.DataFrame({f'tercile {t + 1}': flips[:, tercile == t].mean(1)
                    for t in range(3)},
                   index=[m.removeprefix('Qwen/') for m in MODELS]).round(3))
```

<!-- Placeholder (stage 2): which letters the letter conventions predict, against the gold letters. -->

```{.python .input #scoring-letter-bias}
%%tab pytorch
counts = {'gold': collections.Counter('ABCDE'[g] for g in GOLD)}
for m in MODELS:
    for conv in ('letter-0', 'letter-5'):
        counts[f'{m.removeprefix("Qwen/")} {conv}'] = collections.Counter(
            'ABCDE'[i] for i in predict(convention_scores(arc, m, conv)))
print(pd.DataFrame(counts).T.reindex(columns=list('ABCDE')).fillna(0)
      .astype(int))
```

<!-- Placeholder (stage 2): the plot below: accuracy per checkpoint and convention, and the pair's paired difference. -->

```{.python .input #scoring-flip-figure}
%%tab pytorch
chance = np.mean([1 / len(it[2]) for it in ARC])
_, axes = d2l.plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={
    'width_ratios': [8, 7], 'wspace': 0.3})
d2l.plot_intervals(CONVENTIONS, {m.removeprefix('Qwen/'): tuple(np.array(
    [ACC.get((m, c), (np.nan,) * 3) for c in CONVENTIONS]).T)
    for m in MODELS}, 'convention', 'accuracy', reference=chance,
    axes=axes[0])
d2l.plot_intervals(PAIR_CONVS, {'Qwen3.5-0.8B - Qwen3-0.6B': tuple(np.array(
    [PAIR_DIFF[c] for c in PAIR_CONVS]).T)}, 'convention',
    'paired difference', reference=0, axes=axes[1])
axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=3,  # above
               frameon=False);  # the panel, clear of every interval
```
