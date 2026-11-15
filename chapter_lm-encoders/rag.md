# Retrieval-Augmented Generation
:label:`sec_lm_rag`

<!-- OPEN (author, 2026-11-14): judge calibration vs the new generator.
The judge (Qwen3.8-27B, thinking on) was validated on lm-rag-judge-labels-v1, whose
answers come from Qwen3-4B, the earlier generator. The generator is now Qwen3.5-9B.
A stronger generator makes subtler mistakes, so re-check the judge on about 100 dev
answers from the new generator, labelled by the same two frontier models (Claude
Opus 5.5, Gemini 4 Argon). Cost: about $2 of OpenRouter calls. Deferred by the author
until the Fable pass, which may change more. Ask Alex before making the call.
Until then the text must say that the calibration labels were made on the earlier
generator's answers. Details: scratch ch19/judge.md and AUTHOR_DECISIONS.md. -->

In :numref:`sec_lm_retrieval` we built a retriever that finds, for nearly
every question about this book, a passage that answers it. A reader will
want more than a ranked list: hand the passages to a language model and read
an answer. So let us wire the retriever to a generator, Qwen3.5-9B, and ask
it what happens to PyTorch modules kept in a plain Python list instead of
`nn.Sequential`. The passage that answers comes back first, and its last
sentence reads "Modules in a plain list stay unregistered, without any
warning." The model writes four fluent sentences about registration, cites
the second and third passages, and never says that nothing warns you. Was
the right passage ranked too low? Was it in the context and ignored? Did a
neighboring passage say something else?

Systems of this kind are called *retrieval-augmented generation* (RAG),
after a 2020 paper that trained a retriever's query encoder together with a
generator :cite:`Lewis.Perez.Piktus.ea.2020`. Today the name covers any
pipeline that retrieves passages and places them in a prompt. Such a
pipeline is a chain of stages: a query is formed, passages are retrieved,
reordered, packed into a context and read by the generator. When the answer
is wrong, any stage may be at fault, and one accuracy number cannot say
which, although each stage needs a different repair. The idea of this
section is that a failure is located by measuring each stage on its own,
with its own interval.

We first build what such measurements need, questions about the book with
known answers and known passages that answer them. Then we measure a
baseline stage by stage, change one stage at a time, sort the failures by
the stage that caused them and test a fix chosen in advance. A short
derivation shows how the original RAG model makes retrieval part of a
probabilistic model whose retriever can learn from the generator. The
section ends on the question that longer context windows keep raising: is
retrieval needed at all when whole chapters fit into the prompt?

```{.python .input #rag-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #rag-extras}
%%tab pytorch
import collections
import hashlib
import json
import os
import pandas as pd
import random
import re
import shutil
import textwrap
import time
import transformers
import warnings
import zipfile
```

The generator, Qwen3.5-9B, runs in bf16 with thinking off. It is a hybrid
of the kind described in :numref:`subsec_lm_hf_config`, in which only one
layer in four keeps keys and values and the others carry a state of fixed
size. The retriever brings the embedder and reranker of
:numref:`sec_lm_retrieval`, and a larger Qwen model, introduced below,
grades the answers.

```{.python .input #rag-revisions}
%%tab pytorch
# Pinned in this notebook, not in the library: adding it to the saved
# HF_REVISIONS would change the fingerprint of every chapter-18 notebook
d2l.HF_REVISIONS['Qwen/Qwen3.5-9B'] = (
    'c202236235762e1c871ad0ccb60c8ee5ba337b9a')
```

Generating and judging every answer takes GPU hours, so a normal run
replays the recorded answers and verdicts and generates a few again as a
check; `D2L_RERUN` regenerates them. The next cell fixes the baseline
(`BASE`) and the configurations that each change one stage of it
(`CONFIGS`). It also fixes, for each kind of failure that
:numref:`subsec_lm_rag_failures` sorts answers into, a repair (`FIXES`),
written down before a single failure was counted.

```{.python .input #rag-config}
%%tab pytorch
GEN, RERANKER = 'Qwen/Qwen3.5-9B', 'Qwen/Qwen3-Reranker-0.6B'
EMB, JUDGE = d2l.DEFAULT_EMBEDDER, d2l.RAG_JUDGE_LOCAL
GPU = torch.cuda.is_available()
SMALL = not GPU  # the CPU path: two live answers, BM25-only coverage
T0 = time.time()  # a judging run stops JUDGE_MINUTES after this
ARTIFACTS = {  # name: (registered key, the file a regeneration writes)
    'rag-gen-base': ('lm-rag-gen-base-v1', 'lm-rag-gen-base.jsonl'),
    'rag-gen-stages': ('lm-rag-gen-stages-v2', 'lm-rag-gen-stages.jsonl'),
    'rag-gen-fix': ('lm-rag-gen-fix-v1', 'lm-rag-gen-fix.jsonl'),
    'rag-judge': ('lm-rag-judge-v4', 'lm-rag-judge.jsonl'),
    'rag-judge-cache': ('lm-rag-judge-cache-v4', 'lm-rag-judge-cache.zip'),
    'rag-cards': ('lm-rag-cards-v1', 'lm-rag-cards.jsonl')}
RERUN = d2l.lm_rerun(ARTIFACTS)
if 'rag-judge-cache' in RERUN:  # the judge's cache is written with it
    RERUN.add('rag-judge')
assert GPU or not RERUN - {'rag-cards'}, 'answers and judgments need GPUs'
ORDER = ('run order, each result registered before the next: rag-gen-base; '
         'rag-judge (until every answer is judged); rag-cards; '
         'rag-gen-stages; rag-judge; rag-gen-fix; rag-judge')
assert not ('rag-judge' in RERUN and RERUN & {'rag-gen-stages',
                                              'rag-gen-fix'}), ORDER
assert not ('rag-gen-fix' in RERUN and RERUN & {'rag-gen-base',
                                                'rag-gen-stages'}), ORDER
K, BUDGET, DEPTH = 5, 3000, 50  # passages; context tokens; ranking kept
MAX_NEW, BATCH = 300, 8  # new tokens per answer; prompts per batch
QUICK = False  # True: 80 stratified dev and test items (development only)
assert not (QUICK and RERUN), 'QUICK results are never saved'
N_LIVE = 8 if GPU else 2  # base answers generated again in a normal run
JUDGE_MINUTES, CHUNK, CONCURRENCY = 50, 64, 8  # judging run; saves; flight
RETRY_TOKENS = 6000  # the one re-judging of an answer left without verdict
N_REPEAT, N_CARDS = 60, 50  # base answers judged twice; failure cards
CARD_GIB = 21.5  # estimates; measured peaks add 1-2 (23.6 GiB usable)
BASE = dict(chunking='sections', context='retrieved', k=K, budget=BUDGET,
            order='rank', rewrite=None, rerank_k=None, prompt='fixed',
            verify=False, abstain_below=None, max_new_tokens=MAX_NEW,
            batch=BATCH, shuffle=None)
CONFIGS = {  # name: what changes against the baseline
    'closed': dict(BASE, context='none'),  # no passages: closed book
    'base': BASE,
    'oracle': dict(BASE, context='gold'),  # the gold chunks only
    'base-repeat': dict(BASE, shuffle=1),  # the same prompts, other batches
    'fixed': dict(BASE, chunking='fixed', k=None, budget='base'),
    'fixed-over': dict(BASE, chunking='fixed', k=None, budget='base',
                       over=255),  # up to a window past the base's tokens
    'rewrite-rerank': dict(BASE, rewrite='generator', rerank_k=50),
    'reverse': dict(BASE, order='reverse'),
    'budget-1500': dict(BASE, k=None, budget=1500),
    'budget-6000': dict(BASE, k=None, budget=6000, batch=4)}
STAGES = ['fixed', 'rewrite-rerank', 'reverse', 'budget-1500', 'budget-6000']
CONTROLS = ['fixed-over']  # checks a stage's result; not a candidate
FIXES = {  # failure class: its fix, written down before any class is counted
    'not_retrieved': dict(BASE, rewrite='fuse'),  # the query's ranking too
    'ranked_low': dict(BASE, rerank_k=100, k=8),
    'ignored': dict(BASE, prompt='quote', order='edges', max_new_tokens=500),
    'unsupported': dict(BASE, verify=True),
    'false_answer': dict(BASE, abstain_below='train'),  # tuned on train
    'false_abstain': dict(BASE, prompt='rule_after')}
CONFIGS.update({f'fix-{c}': v for c, v in FIXES.items()})
REV = {GEN: d2l.HF_REVISIONS[GEN],
       **{m: d2l.HF_ENCODER_REVISIONS[m] for m in (EMB, RERANKER)}}
QA_KEY = 'lm-book-qa-v2'

def settings(name):
    """What a configuration's answers are made with (JSON); reading the
    records checks it."""
    return json.dumps(dict(
        CONFIGS[name], generator=f'{GEN}@{REV[GEN]}', corpus=d2l.R3_VERSION,
        embedder=f'{EMB}@{REV[EMB]}', reranker=f'{RERANKER}@{REV[RERANKER]}',
        bm25=d2l.BookRetriever.BM25_PARAMS, qa=QA_KEY), sort_keys=True)

def run_path(name):
    """The file of name that this run regenerates, else the registered
    one, else None (not registered yet)."""
    key, file = ARTIFACTS[name]
    if name in RERUN:
        return os.path.join('..', 'data', 'lm', 'rerun', file)
    return d2l.lm_registered(key)

print(pd.DataFrame({'model': [GEN, EMB, RERANKER, JUDGE['model']],
                    'revision': [REV[GEN][:12], REV[EMB][:12],
                                 REV[RERANKER][:12], JUDGE['revision'][:12]]},
                   index=['generator', 'embedder', 'reranker', 'judge']))
print(f'judge: thinking {"on" if JUDGE["thinking"] else "off"}, at most '
      f'{JUDGE["max_new_tokens"]:,} new tokens, served by vLLM '
      f'{JUDGE["vllm"]}')
print(f'GPU: {GPU}; regenerating: {", ".join(sorted(RERUN)) or "none"}')
```

## A Question Set over the Book
:label:`subsec_lm_book_qa`

To measure a RAG system we need questions whose answers we know, and we need
to know where in the corpus each answer is written. Public benchmarks give
the first but not the second for our corpus, and any of them may have been
in the training data of every model we use. So we build a set over the
book: we know its content, every answerable question has a *gold span*, the
stretch of book text that answers it, and we decide how it is split. The
book downloads the set together with the recorded model responses it was
built from, so its construction can be replayed without any API call.

```{.python .input #rag-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-book-qa-v1'] = (  # corpus r3.1; 5,865 train, 240 dev, 240 test
    d2l.LM_DATA_URL + 'lm-book-qa-v1.zip',
    '3b8dba5ceae43c88274e52ce6ae7525884ea4168')
d2l.DATA_HUB['lm-book-qa-api-cache-v1'] = (  # its drafts, checks, verdicts
    d2l.LM_DATA_URL + 'lm-book-qa-api-cache-v1.zip',
    '962322fd53b3573daeaa50a92d3f53f25a25572d')
d2l.DATA_HUB['lm-book-qa-v2'] = (  # v1, closed-book flag by the 27B judge
    d2l.LM_DATA_URL + 'lm-book-qa-v2.zip',
    '65a524857b50dfdffa4e75f217b5af66256bd025')
d2l.DATA_HUB['lm-rag-judge-labels-v1'] = (  # 200 pairs, frontier labels
    d2l.LM_DATA_URL + 'lm-rag-judge-labels-v1.zip',
    '2f10e6f4314644e7ebdf7ac7a8c1b3e7182f98d8')
d2l.DATA_HUB['lm-rag-gen-base-v1'] = (  # closed, base, oracle, base-repeat
    d2l.LM_DATA_URL + 'lm-rag-gen-base-v1.jsonl',
    '78065bff472577631de5d7cac287323e77427e7f')
d2l.DATA_HUB['lm-rag-gen-stages-v1'] = (  # the five stage changes, dev
    d2l.LM_DATA_URL + 'lm-rag-gen-stages-v1.jsonl',
    '7c08b175e12fe3138742f46e14720df1ab45e277')
d2l.DATA_HUB['lm-rag-gen-stages-v2'] = (  # v1 and the control fixed-over
    d2l.LM_DATA_URL + 'lm-rag-gen-stages-v2.jsonl',
    'bfd5ba324efcc683b06ded6d96c04b56ca806cb0')
d2l.DATA_HUB['lm-rag-gen-fix-v1'] = (  # dev: fix; test: base, fix, stage
    d2l.LM_DATA_URL + 'lm-rag-gen-fix-v1.jsonl',
    'd94ed6b1f942ea5ca0df0292c5e7cc81bd834d50')
d2l.DATA_HUB['lm-rag-cards-v1'] = (  # 31 base failures, labelled by a model
    d2l.LM_DATA_URL + 'lm-rag-cards-v1.jsonl',
    'dd66f91cc44070f7e2256c63d5e5caf912c4054e')
d2l.DATA_HUB['lm-rag-judge-v1'] = (  # verdicts on the base and stage answers
    d2l.LM_DATA_URL + 'lm-rag-judge-v1.jsonl',
    '8e128dbe149cf3f22b3d0979368c679c5d43c371')
d2l.DATA_HUB['lm-rag-judge-cache-v1'] = (  # the judge's recorded replies
    d2l.LM_DATA_URL + 'lm-rag-judge-cache-v1.zip',
    '12402e3f6d7675bcd3f2667743607677129abd4e')
d2l.DATA_HUB['lm-rag-judge-v2'] = (  # v1 and the verdicts on the fix run
    d2l.LM_DATA_URL + 'lm-rag-judge-v2.jsonl',
    '8e8ad8b9f63e000005e40f060c4e926e66c3312f')
d2l.DATA_HUB['lm-rag-judge-cache-v2'] = (  # every recorded reply
    d2l.LM_DATA_URL + 'lm-rag-judge-cache-v2.zip',
    '9b68fb8b52733e7909000165121512892f4b022c')
d2l.DATA_HUB['lm-rag-judge-v3'] = (  # v2, missing verdicts judged again
    d2l.LM_DATA_URL + 'lm-rag-judge-v3.jsonl',  # with 6,000 new tokens
    '28d257ee9b9545e7035917e6612d978faf8d259d')
d2l.DATA_HUB['lm-rag-judge-cache-v3'] = (  # v2 and those longer replies
    d2l.LM_DATA_URL + 'lm-rag-judge-cache-v3.zip',
    '8711571c3c94646d17215c522aec15d7a3cd269c')
d2l.DATA_HUB['lm-rag-judge-v4'] = (  # v3 and the verdicts on fixed-over
    d2l.LM_DATA_URL + 'lm-rag-judge-v4.jsonl',
    '0169be597e978728e6559cd191c91cb8e30281c5')
d2l.DATA_HUB['lm-rag-judge-cache-v4'] = (  # v3 and those replies
    d2l.LM_DATA_URL + 'lm-rag-judge-cache-v4.zip',
    'cf57712b285710ee87c0a7c56aab0600d13dac7a')
```

Each item holds

- a question and a reference answer;
- its *key facts*, the short statements a correct answer must contain;
- gold spans, as character ranges in the cleaned documents of
  :numref:`sec_lm_retrieval`;
- *alternative spans*, other passages of the book that answer it too;
- flags such as `identifier` or `general_knowledge` (answerable without the
  book).

Which chunks are relevant depends on how the corpus is cut, so items store
spans, and `gold_chunk_ids` computes the chunks for any chunking.

```{.python .input #rag-qa-def}
%%tab pytorch
#@save
def book_qa(split='dev', answerable=None, types=None, n=None, seed=0,
            root=os.path.join('..', 'data', 'lm')):
    """The book's question set (artifact lm-book-qa-v2), one split: a
    BookQA (a list of item dicts), optionally only answerable (or only
    unanswerable) items, only some question types, and a stratified
    sample of n items."""
    items = d2l._book_qa_read(split, root)
    if answerable is not None:
        items = [q for q in items if q['answerable'] == answerable]
    if types is not None:
        items = [q for q in items if q['type'] in types]
    qa = d2l.BookQA(items, split)
    return qa if n is None else qa.subset(n, seed)

def gold_chunk_ids(item, corpus, min_overlap=0.5, include_alt=True):
    """Ids of the chunks of corpus relevant to a question: those that
    overlap one of its gold spans (and, with include_alt, the passages
    judged to answer it as well) by at least min_overlap of the span's or
    of the chunk's characters."""
    spans = item['gold_spans'] + (item.get('alt_spans', []) if include_alt
                                  else [])
    return {i for s in spans for i in corpus.span_to_chunk_ids(
        s['doc_id'], s['char_start'], s['char_end'], min_overlap)}
```

The set was written and checked by models, in four steps:

1. A drafter, DeepSeek-V4.1-Flash, wrote two candidate questions per
   *anchor chunk*, a passage drawn from the book, each with a reference, key
   facts and verbatim evidence. It comes from outside the Qwen family of our
   generator and judge.
2. Rules removed a candidate whose evidence is not a substring of the book,
   whose wording assumes a context the reader lacks ("in the passage
   above"), that copies the anchor's words (word overlap alone would then
   find the answer and make retrieval look easy), or that nearly repeats
   another.
3. A checker, Qwen3.8-27B, tested that each question stands alone and that
   its reference is supported, complete and unique. A search of neighboring
   passages collected other passages that answer it.
4. A frontier model, Claude Opus 5.5, verified every `dev` and `test` item
   in a fresh context.

The items were checked by models, not by people. Model checks are fast,
cheap and recorded, so anyone can replay them. The verifier's 620 calls
cost about \$7, and the whole set, drafts and checks included, about \$15;
a person checking 480 items against their passages would need many hours.
What the checks can miss is what the models share. A reference that
drafter, checker and verifier misread in the same way passes every check,
and the verifier may have read this public book and accept a question it can
answer from memory. The checker is also the judge we use below, so a wrong
reference that it passes is later used to grade answers. The protection
against shared blind spots is agreement with models of other families, and
the last exercise compares hand labels with the verifier's.

```{.python .input #rag-qa-stats}
%%tab pytorch
QA = {s: d2l.book_qa(s) for s in ('train', 'dev', 'test')}  # test: once
DEV, TEST = (QA[s].subset(80) if QUICK else QA[s] for s in ('dev', 'test'))
ITEMS = {q['qid']: q for qa in QA.values() for q in qa}
with zipfile.ZipFile(d2l.download_keyed(QA_KEY)) as z:
    MANIFEST = json.loads(z.read('manifest.json'))
VERIFIER = MANIFEST['snapshots'][MANIFEST['models']['verifier']['requested']]
assert all(q['verified_by'] == VERIFIER for s in ('dev', 'test')
           for q in QA[s]) and all(q['verified_by'] is None
                                   for q in QA['train'])
print(pd.DataFrame({s: qa.stats()['items'] for s, qa in QA.items()})
      .to_string())
served = lambda role: sorted({f'{q[role]["served"]} ({q[role]["provider"]})'
                              for qa in QA.values() for q in qa})
print(f'drafter: {served("drafter")}\nchecker: {served("checker")}\n'
      f'every dev and test item verified by {VERIFIER}')
print(pd.DataFrame({s: {
    'answerable': sum(q['answerable'] for q in qa),
    'book-specific': sum('general_knowledge' not in q['flags'] for q in qa),
    'closed-book correct (Qwen3-4B)': sum(bool(q['closed_book_correct'])
                                          for q in qa),
    'with alternative spans': sum(bool(q['alt_spans']) for q in qa),
    'identifier': sum('identifier' in q['flags'] for q in qa)}
    for s, qa in QA.items()}))
STAGE_ROWS = {'s3_reject_evidence': 'evidence not verbatim',
              's3_reject_context_phrase': 'context phrase',
              's3_reject_lexical': 'lexical copy', 's3_reject_near_duplicate':
              'near duplicate', 's4_reject': 'checker (Qwen3.8-27B)',
              's5_reject_answers_better': 'a neighbour answers better',
              's5_reject_contradicts': 'a neighbour contradicts',
              's5_reject_answered': 'unanswerable, but answered',
              's5_cross_split_alt': 'alternative in another split',
              's8_reject': 'verifier (frontier)'}
stats = MANIFEST['stats']
print(pd.DataFrame({v: stats.get(k, {}) for k, v in STAGE_ROWS.items()}).T
      .fillna(0).astype(int).loc[:, list(MANIFEST['quota'])].to_string())
print(f'alternative spans: {sum(stats["s5_alt_spans"].values())} in '
      f'{sum(stats["s5_alt_items"].values())} candidates')
print('frontier verifier against the checker, per check (dev and test pool):')
print(pd.DataFrame(MANIFEST['verifier_vs_checker']).T[['n', 'agree', 'kappa']]
      .astype(float).round(2))
spend = MANIFEST['spend_usd']  # recorded by the build, per call
print('API spend in US$ per model:\n' + pd.Series(spend['per_model']).round(2)
      .to_string() + f'\n  total {sum(spend["per_model"].values()):.2f}; '
      f'the verifier\'s check of dev and test: '
      f'{spend["per_stage"]["s8-verify"]["calls"]} calls, '
      f'{spend["per_stage"]["s8-verify"]["usd"]:.2f}')
```

In `dev` and `test` the quotas are 120 questions answered by one passage,
40 that ask for an explanation, and 20 each of numeric, code, multi-passage
(`multihop`) and unanswerable questions; `train` passed only the rules and
the checker. Copying the anchor's words was by far the most common reason
for rejection. The closed-book flag comes from Qwen3-4B, an earlier
generator, so below we measure our own generator without passages instead.

On whether an item passes, verifier and checker agree on 84% of the pool.
That sounds like agreement, but both pass most items, so they agree by
default. *Cohen's κ* corrects for this. It compares the observed agreement
$p_o$ with the agreement $p_e$ that two raters with the same pass rates
would reach by chance:

$$\kappa = \frac{p_o - p_e}{1 - p_e}.$$

It is 1 for perfect agreement and 0 for agreement at chance level. Here
κ is only 0.12. Solving for $p_e$ gives 0.82, the chance agreement of two
raters who each pass about nine items in ten ($0.9^2 + 0.1^2 = 0.82$), so
84% is barely above it. The few items each model rejects are mostly
different ones, and the set very likely still holds items that a third
check would reject: the verifier alone still rejected 67 items after every
earlier step had passed them. Here is the first `dev` item of each type
with its gold span.

```{.python .input #rag-qa-examples}
%%tab pytorch
CORPUS = d2l.book_corpus()  # structure-aware chunks of at most 512 tokens

def span_text(s, width=240):
    """The text of a span, shortened for printing."""
    text = CORPUS.docs[s['doc_id']]['text'][s['char_start']:s['char_end']]
    text = ' '.join(text.split())
    return text if len(text) <= width else text[:width] + ' ...'

for t in MANIFEST['quota']:
    q = next(q for q in QA['dev'] if q['type'] == t)
    for line in (f'[{t}] {q["qid"]}: {q["question"]}',
                 f'  reference: {q["reference"]}',
                 f'  key facts: {q["key_facts"]}',
                 *(f'  gold span in {s["doc_id"]}: {span_text(s)}'
                   for s in q['gold_spans'])):
        print(textwrap.fill(line, 96, subsequent_indent='    '))
```

The questions name what they ask about ("the book's `training_memory_gib`
helper"), and most need the book: the 1000 W that NVIDIA specifies for an
RTX 5090 system is not common knowledge. The unanswerable question is a
plausible question about attention that the book does not answer.

The split is made over documents, not questions: if two questions drafted
from one passage landed in `train` and `dev`, a system tuned on `train`
would have seen the passage of a `dev` answer. The cell checks on every item
that its anchors and spans lie in documents of its own split and that each
anchor chunk serves at most one answerable evaluation item.

```{.python .input #rag-qa-leakage}
%%tab pytorch
split_of = CORPUS.split_of
anchor_doc = lambda cid: CORPUS[cid]['doc_id']
for s, qa in QA.items():
    for q in qa:
        docs = ({anchor_doc(a) for a in q['anchor_ids']} |
                {x['doc_id'] for x in q['gold_spans'] + q['alt_spans']})
        assert {split_of(d) for d in docs} == {s}, q['qid']
train_docs = {anchor_doc(a) for q in QA['train'] for a in q['anchor_ids']}
eval_docs = {x['doc_id'] for s in ('dev', 'test') for q in QA[s]
             for x in q['gold_spans']}
assert not train_docs & eval_docs  # no dev or test span meets a train anchor
uses = collections.Counter(a for s in ('dev', 'test') for q in QA[s]
                           if q['answerable'] for a in q['anchor_ids'])
assert max(uses.values()) == 1  # one answerable item per anchor chunk
for s in ('dev', 'test'):
    counts = collections.Counter(q['type'] for q in QA[s])
    fills = sum('type_shortfall_fill' in q['flags'] for q in QA[s])
    assert all(counts[t] == n for t, n in MANIFEST['quota'].items()) or fills
print(pd.DataFrame({s: {
    'documents': len({x['doc_id'] for q in qa for x in q['gold_spans']}),
    'anchor chunks': len({a for q in qa for a in q['anchor_ids']}),
    'shortfall fills': sum('type_shortfall_fill' in q['flags'] for q in qa)}
    for s, qa in QA.items()}))
print('every anchor and span lies in a document of its split; quotas met')
```

`gold_chunk_ids` calls a chunk relevant to a question if it overlaps a gold
or alternative span by at least half of the span's characters or half of
the chunk's, the rule of :numref:`sec_lm_retrieval`. It serves the section
chunks of that section (at most 512 tokens, cut at headings) as well as its
fixed windows of 256 tokens, cut without regard to the text. Consecutive
windows overlap, here by 64 tokens, so that a sentence cut at one boundary
appears whole in the next.

```{.python .input #rag-qa-gold}
%%tab pytorch
CHUNKINGS = {'sec512': CORPUS,
             'fix256o0': d2l.book_corpus('fixed', size=256, overlap=0),
             'fix256o64': d2l.book_corpus('fixed', size=256, overlap=64)}
FIXED = CHUNKINGS['fix256o64']
answerable = {s: [q for q in QA[s] if q['answerable']]
              for s in ('dev', 'test')}
print(pd.DataFrame({s: {tag: np.mean([bool(d2l.gold_chunk_ids(q, c))
                                      for q in qs])
                        for tag, c in CHUNKINGS.items()}
                    for s, qs in answerable.items()}).round(3))
q = next(q for q in answerable['dev'] if len(d2l.gold_chunk_ids(
    q, CORPUS, include_alt=False)) != len(d2l.gold_chunk_ids(
        q, FIXED, include_alt=False)))
print(f'\n{q["qid"]} ({q["type"]}): gold span(s) ' + ', '.join(
    f'{s["doc_id"]} [{s["char_start"]}, {s["char_end"]})'
    for s in q['gold_spans']))
for tag in ('sec512', 'fix256o64'):
    c = CHUNKINGS[tag]
    d2l.print_wrapped(f'  {tag}: ' + ', '.join(
        f'{i.split("#")[-1]} [{c[i]["char_start"]}, {c[i]["char_end"]})'
        for i in sorted(d2l.gold_chunk_ids(q, c, include_alt=False))),
        width=96)
```

Every answerable item has a relevant chunk under all three chunkings, as the
rules demanded. The gold span of `dev-0001` lies in two section chunks but
in five overlapping windows, so "a relevant chunk among the first five"
means very different amounts of text under the two chunkings.

## The Pipeline and Its Measurements
:label:`subsec_lm_rag_pipeline`

A pipeline over this set has five stages, and each can be measured without
the stages after it (:numref:`tab_lm_rag_stages` and
:numref:`fig_lm_rag_stages`). Retrieval and reranking
come from :numref:`sec_lm_retrieval`: BM25 and dense rankings fused by
reciprocal ranks, optionally reordered by a cross-encoder reranker.

:The stages of a retrieval-augmented generator, what each decides, and the measurement that sees it.
:label:`tab_lm_rag_stages`

| stage | what it decides | measured by |
|---|---|---|
| query | the text sent to the retriever: the question, or a rewrite | the stages after it |
| first-stage retrieval | a ranking of chunks by BM25 and dense search, fused | Recall@$k$ |
| reranking (optional) | the order of the first candidates | Recall@$k$ |
| context assembly | which chunks enter the prompt, in what order, within a budget | coverage at the budget |
| grounded generation | an answer with numbered citations, or the abstention sentence | accuracy, faithfulness, citations, abstentions |

![The five stages of the retrieval-augmented generator, top to bottom, followed by the judge that grades its answers; beside each, the measurement that sees it on its own and the failure class that blames it. A wrong answer is blamed on the first stage that lost it: no relevant chunk among the first 50 is *not retrieved*; one there but not in the context is *ranked low*, which reranking and assembly share. The judge's verdicts on correctness and faithfulness give accuracy and faithfulness, and decide among the classes of the generation stage.](../img/mdl-lm-encoders-rag-stages.svg)
:label:`fig_lm_rag_stages`

### Context Assembly and the Prompt

Recall@$k$, the share of questions with a relevant chunk among the first
$k$, cannot compare chunkings, since five section chunks hold more text than
five windows. *Coverage at a budget* of $T$ tokens, `cov@T`, fixes the
amount of text instead: take chunks in rank order until the next would pass
$T$ tokens, and ask whether they hold at least half of one gold span.

`assemble_context` builds the real context the same way. It skips a chunk
whose embedding has a cosine above 0.95 with one already taken, since a near
duplicate adds tokens and no information. It then orders the chunks by rank,
in reverse (the best next to the question), or with the best at both ends
(`edges`). The last order is there because language models use the start
and the end of a long context better than its middle
:cite:`Liu.Lin.Hewitt.ea.2024`.

```{.python .input #rag-stage-defs}
%%tab pytorch
#@save
def cov_at_budget(ranked_chunk_ids, item, corpus, budget_tokens):
    """Whether the chunks of corpus taken in rank order until the next one
    would pass budget_tokens tokens cover at least half the characters of
    one of the item's gold spans (or alternative spans)."""
    taken, used = collections.defaultdict(list), 0
    for cid in ranked_chunk_ids:
        c = corpus[cid]
        if used + c['n_tokens'] > budget_tokens:
            break
        used += c['n_tokens']
        taken[c['doc_id']].append((c['char_start'], c['char_end']))
    for s in item['gold_spans'] + item.get('alt_spans', []):
        a, b = s['char_start'], s['char_end']
        covered = np.zeros(b - a, dtype=bool)
        for lo, hi in taken.get(s['doc_id'], ()):
            covered[max(lo, a) - a:max(min(hi, b) - a, 0)] = True
        if b > a and covered.mean() >= 0.5:
            return True
    return False

def assemble_context(hits, budget_tokens=3000, order='rank', dedup_cos=0.95,
                     embed=None, k=None):
    """The passages for the prompt: hits (chunk dicts with n_tokens, best
    first) taken in rank order, skipping a near duplicate of one already
    taken (cosine above dedup_cos between the unit vectors that embed(hits)
    returns; no check without embed), until k are taken or the next would
    pass budget_tokens; then ordered 'rank' (best first), 'reverse' (best
    last, next to the question) or 'edges' (the best at both ends, the
    weakest in the middle). Returns (passages, tokens used)."""
    if order not in ('rank', 'reverse', 'edges'):
        raise ValueError(f'unknown order {order!r}')
    hits = list(hits)
    V = (torch.as_tensor(embed(hits)).float()
         if embed is not None and hits else None)
    keep, used = [], 0
    for i, h in enumerate(hits):
        if (k is not None and len(keep) == k
                or used + h['n_tokens'] > budget_tokens):
            break
        if V is not None and keep and (V[keep] @ V[i]).max() > dedup_cos:
            continue  # a near duplicate of a passage already taken
        keep.append(i)
        used += h['n_tokens']
    out = [hits[i] for i in keep]
    if order == 'reverse':
        out = out[::-1]
    elif order == 'edges':  # ranks 1, 3, 5, ..., then ..., 4, 2
        out = out[0::2] + out[1::2][::-1]
    return out, used
```

The prompt is fixed. It asks for an answer from the numbered passages, each
shown under its heading path, a citation after each statement, and, if the
passages lack the answer, exactly the sentence "I cannot answer this from
the provided passages." Without passages it becomes the closed-book prompt.
Stage changes alter the context or the query, never this prompt; only the
fixes of :numref:`subsec_lm_rag_failures` change it.

```{.python .input #rag-prompt-defs}
%%tab pytorch
#@save
ABSTAIN = 'I cannot answer this from the provided passages.'

#@save
def rag_prompt(question, hits, abstain=ABSTAIN):
    """Chat messages asking for an answer from numbered passages (hits: Hit
    records or dicts with heading_path and text), each statement citing
    them as [n], or the abstention string; hits=None gives the closed-book
    prompt (no passages, no citations)."""
    if hits is None:
        system = ('You answer questions about the textbook Dive into Deep '
                  'Learning. If you do not know the answer, reply exactly: '
                  + abstain)
        user = f'Question: {question}\nAnswer in at most four sentences.'
    else:
        system = ('You answer questions about the textbook Dive into Deep '
                  'Learning using only the numbered passages. After each '
                  'statement, cite the passage numbers that support it, for '
                  'example [2]. If the passages do not contain the answer, '
                  'reply exactly: ' + abstain)
        get = lambda h, k: h[k] if isinstance(h, dict) else getattr(h, k)
        passages = '\n\n'.join(
            f'[{n}] {" > ".join(get(h, "heading_path"))}\n{get(h, "text")}'
            for n, h in enumerate(hits, start=1))
        user = (f'Passages:\n{passages}\n\nQuestion: {question}\n'
                f'Answer in at most four sentences.')
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': user}]

def parse_citations(answer, n_chunks):
    """The passage numbers an answer cites ([1], [1][3], [1, 3]), in order
    of first mention, without numbers outside 1..n_chunks."""
    out = []
    for group in re.findall(r'\[(\d+(?:\s*,\s*\d+)*)\]', answer):
        for n in map(int, group.split(',')):
            if 1 <= n <= n_chunks and n not in out:
                out.append(n)
    return out
```

`RAGPipeline` runs in three phases so that the generator and the
retriever's models do not compete for memory. The generator writes
rewritten queries if asked. The retriever searches, reranks and assembles
while the generator waits. Then the retriever's models leave the GPU, and
the generator answers greedily in left-padded batches of eight. At the
prefill
Transformers keeps only the last position's logits (`logits_to_keep=1`).
Two more query options, `hyde` and `decompose`, are left unmeasured here;
:numref:`sec_lm_encoders_fr` says what they do.

```{.python .input #rag-pipeline-def}
%%tab pytorch
#@save
RAGResult = collections.namedtuple('RAGResult', [
    'qid', 'question', 'query_used', 'ranked_ids', 'context_ids', 'answer',
    'cited_ids', 'abstained', 'n_prompt_tokens', 'n_new_tokens',
    'latency_s'])

#@save
class RAGPipeline:
    """Retrieval-augmented generation over the book corpus. The question,
    or a query the generator writes (rewrite='generator'), goes to the
    retriever; rewrite='fuse' adds the query's ranking to the question's
    in RRF, hyde embeds a passage the generator writes on the dense side,
    decompose fuses the rankings of up to two sub-questions. The reranker
    may reorder the first rerank_k chunks; assemble_context packs at most
    k of them (k=None: as many as fit) into budget_tokens; the generator
    answers with the fixed grounded prompt (or a variant of a
    pre-registered fix), greedily, in batches. verify drops sentences
    whose citations the generator finds unsupportive; abstain_below
    abstains when the reranker's best score for the context is lower."""
    def __init__(self, retriever, model, tokenizer, k=5, budget_tokens=3000,
                 order='rank', rewrite=None, hyde=False, decompose=False,
                 rerank_k=None, max_new_tokens=300, *, depth=50,
                 dedup_cos=0.95, prompt='fixed', verify=False,
                 abstain_below=None):
        if rewrite not in (None, 'generator', 'fuse'):
            raise ValueError(f'unknown rewrite {rewrite!r}')
        if (rerank_k or abstain_below is not None) and not retriever.reranker:
            raise ValueError('reranking needs a retriever with a reranker')
        self.retriever, self.model, self.tokenizer = (retriever, model,
                                                      tokenizer)
        self.k, self.budget_tokens, self.order = k, budget_tokens, order
        self.rewrite, self.hyde, self.decompose = rewrite, hyde, decompose
        self.rerank_k, self.max_new_tokens = rerank_k, max_new_tokens
        self.depth, self.dedup_cos, self.prompt = depth, dedup_cos, prompt
        self.verify, self.abstain_below = verify, abstain_below
        self._dense = self._pos = None

    def prepare(self, questions, budgets=None):
        """Phases 1 and 2. Per question (a string or an item dict): the
        query used, the first-stage ranking (depth chunk ids), the context
        (chunk dicts in prompt order) and its tokens, the reranker's best
        score (None without one) and whether to abstain unasked."""
        R, t0 = self.retriever, time.time()
        qs = [self._question(q) for q in questions]
        query = qs  # phase 1: the generator writes queries
        if self.rewrite:
            query = self._write([d2l._rag_rewrite_prompt(q) for q in qs], 20)
        dense = (self._write([d2l._rag_hyde_prompt(q) for q in qs], 120)
                 if self.hyde else qs)
        extra = [[r] if self.rewrite == 'fuse' else [] for r in query]
        if self.decompose:
            subs = self._write([d2l._rag_decompose_prompt(q) for q in qs], 60)
            extra = [e + [l.strip() for l in s.splitlines() if l.strip()][:2]
                     for e, s in zip(extra, subs)]
        main = query if self.rewrite == 'generator' else qs
        ids = lambda H: [[h.chunk_id for h in hs] for hs in H]
        pool = max(100, self.depth, self.rerank_k or 0)  # phase 2
        if not (self.hyde or any(extra)):  # the retriever's own fusion
            ranked = ids(R.search_batch(main, k=pool))
        else:  # BM25 of the question, dense of `dense`, more rankings: RRF
            bm = ids(R.search_batch(main, k=pool, mode='bm25'))
            de = (ids(R.search_batch(dense, k=pool, mode='dense'))
                  if R.dense else [[] for _ in qs])
            flat = [e for es in extra for e in es]
            more = iter(ids(R.search_batch(flat, k=pool)) if flat else [])
            ranked = [[i for i, _ in d2l.rrf_fuse(
                [b, d] + [next(more) for _ in es], k=R.rrf_k)][:pool]
                for b, d, es in zip(bm, de, extra)]
        cand, top = ranked, [None] * len(qs)
        if self.rerank_k:
            rr = self._rerank(qs, [r[:self.rerank_k] for r in ranked])
            cand = [[c for c, _ in r] for r in rr]
            top = [r[0][1] if r else None for r in rr]
        budgets = budgets or [self.budget_tokens] * len(qs)
        embed = self._embed if self.dedup_cos and R.dense else None
        out = []
        for j in range(len(qs)):
            ctx, used = d2l.assemble_context(
                [R.corpus[c] for c in cand[j]], budgets[j], self.order,
                self.dedup_cos, embed, self.k)
            out.append(dict(query_used=query[j], context=ctx,
                            ranked_ids=ranked[j][:self.depth],
                            n_context_tokens=used, top_score=top[j],
                            abstain=False))
        if self.abstain_below is not None:
            need = [j for j, o in enumerate(out)
                    if o['top_score'] is None and o['context']]
            rr = self._rerank([qs[j] for j in need], [
                [c['id'] for c in out[j]['context']] for j in need])
            for j, r in zip(need, rr):
                out[j]['top_score'] = r[0][1]
            for o in out:
                o['abstain'] = (o['top_score'] is not None and
                                o['top_score'] < self.abstain_below)
        for o in out:
            o['seconds'] = (time.time() - t0) / max(len(qs), 1)
        return out

    def answer(self, questions, batch_size=8, contexts=None):
        """Phase 3 after prepare (or with the contexts given; a context of
        None is the closed-book prompt): the retriever's models leave the
        GPU, then batched greedy generation. One RAGResult per question."""
        prep = self.prepare(questions) if contexts is None else contexts
        self.retriever.release()
        go = [j for j, p in enumerate(prep) if not p.get('abstain')]
        gen = dict(zip(go, self._generate(
            [d2l._rag_fix_prompt(self._question(questions[j]),
                                 prep[j]['context'], self.prompt)
             for j in go], self.max_new_tokens, batch_size) if go else []))
        answers = {j: (d2l._rag_quoted_answer(g[0]) if self.prompt == 'quote'
                       else g[0]) for j, g in gen.items()}
        if self.verify:
            answers = self._verified(answers, prep, batch_size)
        out = []
        for j, q in enumerate(questions):
            ctx = prep[j]['context'] or []
            text = answers.get(j, d2l.ABSTAIN)
            _, n_prompt, n_new, secs = gen.get(j, ('', 0, 0, 0.0))
            out.append(d2l.RAGResult(
                q.get('qid') if isinstance(q, dict) else None,
                self._question(q), prep[j].get('query_used'),
                list(prep[j].get('ranked_ids') or []),
                [c['id'] for c in ctx], text,
                d2l.parse_citations(text, len(ctx)), d2l._rag_abstained(text),
                n_prompt, n_new, round(prep[j].get('seconds', 0) + secs, 4)))
        return out

    @staticmethod
    def _question(q):
        return q['question'] if isinstance(q, dict) else q

    def _generate(self, messages, max_new_tokens, batch_size):
        """(text, prompt tokens, new tokens, seconds) per chat: greedy,
        thinking off, left padding; Transformers' generate keeps only the
        last position's logits at the prefill (logits_to_keep=1)."""
        tok, model = self.tokenizer, self.model
        texts = [tok.apply_chat_template(m, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
                 for m in messages]
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        out = []
        for b in range(0, len(texts), batch_size):
            enc = tok(texts[b:b + batch_size], return_tensors='pt',
                      padding=True, padding_side='left',
                      add_special_tokens=False).to(model.device)
            t = time.time()
            with torch.no_grad():
                ids = model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=tok.pad_token_id)
            secs = (time.time() - t) / len(enc['input_ids'])
            new = ids[:, enc['input_ids'].shape[1]:].tolist()
            for row, n_prompt in zip(new, enc['attention_mask'].sum(1)):
                n = next((i + 1 for i, x in enumerate(row) if x in eos),
                         len(row))
                out.append((tok.decode(row[:n], skip_special_tokens=True)
                            .strip(), int(n_prompt), n, secs))
        return out

    def _write(self, messages, max_new_tokens):
        """The generator's short replies (queries, passages)."""
        return [t for t, *_ in self._generate(messages, max_new_tokens, 8)]

    def _rerank(self, queries, id_lists):
        """Per query, (chunk id, P(yes)) pairs of its chunks, best first."""
        R = self.retriever
        if self._dense is None:
            self._dense = dict(zip(R.corpus.ids, R.corpus.texts('dense')))
        model, tok = R._reranker()
        out = []
        for q, ids in zip(queries, id_lists):
            p = (d2l.rerank_yes_no(model, tok, q, [self._dense[i] for i in
                                                   ids],
                                   instruction=d2l.BOOK_TASK)
                 if ids else np.zeros(0))
            out.append([(ids[i], float(p[i]))
                        for i in np.argsort(-p, kind='stable')])
        return out

    def _embed(self, hits):
        """Unit vectors of chunk dicts, from the retriever's chunk vectors."""
        R = self.retriever
        if self._pos is None:
            self._pos = {c: i for i, c in enumerate(R.corpus.ids)}
        return R.chunk_vectors()[[self._pos[h['id']] for h in hits]].float(
            ).cpu()

    def _verified(self, answers, prep, batch_size):
        """answers without the sentences whose cited passages the generator
        says do not support them (P(yes) below P(no)); a sentence without
        a citation goes too, and an answer left empty abstains."""
        jobs, sents = [], {}
        for j, a in answers.items():
            ctx = prep[j]['context'] or []
            sents[j] = re.split(r'(?<=[.!?])\s+', a.strip())
            for i, s in enumerate(sents[j]):
                cited = d2l.parse_citations(s, len(ctx))
                if cited and not d2l._rag_abstained(a):
                    jobs.append((j, i, d2l._rag_verify_prompt(
                        s, [ctx[c - 1] for c in cited])))
        keep = set()
        if jobs:
            lp = d2l.label_logprobs(self.model, self.tokenizer,
                                    [m for *_, m in jobs], ['yes', 'no'],
                                    chat=True, enable_thinking=False,
                                    batch_size=batch_size)
            keep = {(j, i) for (j, i, _), l in zip(jobs, lp) if l[0] > l[1]}
        return {j: a if d2l._rag_abstained(a) else ' '.join(
            s for i, s in enumerate(sents[j]) if (j, i) in keep) or d2l.ABSTAIN
                for j, a in answers.items()}
```

### A Judge for Free-Text Answers

Answers in free text need a grader. `rag_judge` shows a second model,
Qwen3.8-27B with thinking on, the question, the reference, the key facts,
the passages and which of them the answer cites, and asks for three
verdicts. An answer is *correct* if it states every key fact and nothing
that contradicts the reference. It is *faithful* if every factual statement
in it is supported by the passages in its context, whatever the reference
says, and *supported* if the passages it cites suffice on their own. An
answer from memory can be correct and unfaithful; one that faithfully
repeats an incomplete passage, faithful and wrong. An answer that draws on
passage [1] but cites only [2] is faithful and unsupported: the support is
in the context, but the citation points elsewhere.

Answers to unanswerable questions are decided by string match, without the
judge. There the abstention sentence is correct, and any other reply,
however it declines, counts as an answer and is incorrect. The abstention
sentence on an answerable question is incorrect too. Two scores need no
judge either. *Citation precision* is the share of cited passages that are
relevant, and *citation recall* the share of relevant passages in the
context that the answer cites.

```{.python .input #rag-judge-defs}
%%tab pytorch
#@save
def rag_judge(judge, tokenizer, item, result, context_texts, batch_size=8,
              thinking=None, max_new_tokens=None):
    """The judge's verdicts on an answer: 'correct' (every key fact stated,
    nothing contradicting the reference), 'faithful' (every statement
    supported by the passages given) and 'supported' (the cited passages
    alone support it). result is a RAGResult or a dict with answer and
    cited_ids (1-based passage numbers). An abstention (the abstention
    sentence in the answer, d2l._rag_abstained) is decided without the
    judge: it is correct exactly for an unanswerable item.
    Without passages only 'correct' is judged. item, result and
    context_texts may also be lists (a list of verdicts comes back, judged
    in batches). judge is a d2l.ChatAPI (tokenizer None; batch_size
    requests in flight; its client sets the thinking switch, records and
    replays) or a Transformers model with its tokenizer. With thinking
    (default: on for a d2l.ChatAPI, off for a model) the judge reasons
    first and its verdict is read after the reasoning; a reply cut off at
    max_new_tokens (default 3,000 with thinking, 64 without) counts as
    malformed. A verdict missing from the judge's JSON after one retry is
    None. A request that fails in transport twice (the server is down, an
    error status) is not a verdict: rag_judge raises, and a recording
    client keeps the replies that did arrive."""
    single = isinstance(item, dict)
    items, results, contexts = (([item], [result], [context_texts])
                                if single else (item, result, context_texts))
    api = hasattr(judge, 'achat')
    thinking = api if thinking is None else thinking
    max_new_tokens = max_new_tokens or (3000 if thinking else 64)
    verdicts, jobs = [], []
    for i, (q, r, ctx) in enumerate(zip(items, results, contexts)):
        r = r._asdict() if hasattr(r, '_asdict') else r
        answer, ctx = r['answer'].strip(), list(ctx or [])
        abstained = d2l._rag_abstained(answer)
        v = {'correct': None, 'faithful': None, 'supported': None}
        if abstained or not q['answerable']:
            v['correct'] = abstained and not q['answerable']
        if abstained:
            v['faithful'] = True  # an abstention states nothing
        verdicts.append(v)
        keys = [] if abstained else [
            k for k in ('correct', 'faithful', 'supported')
            if v[k] is None and (k == 'correct' or ctx)]
        if keys:
            cited = r.get('cited_ids') or []
            jobs.append((i, keys, d2l._rag_judge_messages(
                q, answer, ctx, cited)))
    failed = []
    for attempt in range(2):  # one retry, with the malformed reply shown
        if not jobs:
            break
        replies = d2l._rag_judge_replies(
            judge, tokenizer, [m for _, _, m in jobs], thinking,
            max_new_tokens, batch_size)
        retry = []
        for (i, keys, messages), reply in zip(jobs, replies):
            if isinstance(reply, BaseException):  # no reply: ask again
                if attempt == 0:
                    retry.append((i, keys, messages))
                else:
                    failed.append(reply)
                continue
            parsed = (None if reply is None else
                      d2l._rag_parse_json(reply, keys))
            if parsed is None and attempt == 0:
                retry.append((i, keys, messages + [
                    {'role': 'assistant', 'content': reply or ''},
                    {'role': 'user', 'content': d2l.RAG_JUDGE_V1['retry']}]))
            for k in keys:
                verdicts[i][k] = None if parsed is None else parsed[k]
        jobs = retry
    if failed:
        raise RuntimeError(f'{len(failed)} judge requests failed twice, '
                           f'first: {failed[0]!r}') from failed[0]
    return verdicts[0] if single else verdicts

def citation_metrics(results, items, corpus):
    """Per item, from gold ids: precision (the share of the cited passages
    that are relevant), recall (the share of the relevant passages in the
    context that are cited) and n_cited; NaN where a share has nothing to
    divide by. results are RAGResults or mappings with context_ids and
    cited_ids (1-based passage numbers)."""
    rows = []
    for r, q in zip(results, items):
        r = r._asdict() if hasattr(r, '_asdict') else r
        rel, ctx = d2l.gold_chunk_ids(q, corpus), list(r['context_ids'])
        cited = [ctx[n - 1] for n in r['cited_ids'] if 1 <= n <= len(ctx)]
        found = [c for c in ctx if c in rel]
        rows.append(dict(
            qid=q['qid'], n_cited=len(cited),
            precision=np.mean([c in rel for c in cited]) if cited else np.nan,
            recall=np.mean([c in cited for c in found]) if found else np.nan))
    return pd.DataFrame(rows)

def classify_failure(result, item, judge_out, corpus, depth=50):
    """The stage at which an answer failed: 'correct'; an unanswerable item
    answered: 'false_answer'; an answerable one: 'not_retrieved' (no
    relevant chunk in the first-stage top depth), 'ranked_low' (one there,
    none in the context), 'false_abstain' (one in the context, yet the
    answer abstains), 'ignored' (faithful to the context, not correct) or
    'unsupported' (not faithful). 'unjudged' when the judge gave no
    verdict that the class depends on: such an answer has no class."""
    r = result._asdict() if hasattr(result, '_asdict') else result
    true = lambda v: isinstance(v, (bool, np.bool_)) and bool(v)
    given = lambda v: isinstance(v, (bool, np.bool_))
    if not given(judge_out['correct']):
        return 'unjudged'
    if true(judge_out['correct']):
        return 'correct'
    if not item['answerable']:
        return 'false_answer'
    rel = d2l.gold_chunk_ids(item, corpus)
    if not rel & set(r['context_ids']):
        return ('ranked_low' if rel & set(list(r['ranked_ids'])[:depth])
                else 'not_retrieved')
    if d2l._rag_abstained(r['answer']):
        return 'false_abstain'
    if not given(judge_out.get('faithful')):
        return 'unjudged'
    return 'ignored' if true(judge_out['faithful']) else 'unsupported'
```

### Baseline Answers

Both models need whole cards. Before loading anything we check what fits,
with `d2l.config_memory` of :numref:`subsec_lm_hf_memory`: generation at
eight prompts of 3,300 tokens and at four of 6,300 for the largest budget
below, the retrieval phase with the generator idle on the card, and the
judge.

```{.python .input #rag-memory}
%%tab pytorch
CFG = {m: transformers.AutoConfig.from_pretrained(m, revision=r)
       for m, r in [(GEN, REV[GEN]), (EMB, REV[EMB]),
                    (RERANKER, REV[RERANKER]),
                    (JUDGE['model'], JUDGE['revision'])]}
mem = lambda m, n, b, **kw: {k: v / 2**30 for k, v in d2l.config_memory(
    CFG[m], n, b, **kw).items()}
B6 = CONFIGS['budget-6000']['batch']
g8, g4 = mem(GEN, 3300, BATCH), mem(GEN, 6300, B6)
e, r = mem(EMB, 512, 16), mem(RERANKER, 1024, 16)
j = mem(JUDGE['model'], 3500, 16)  # 16 sequences in flight on the server
table = pd.DataFrame({  # GiB
    f'generation, 3,300 x {BATCH}': (g8['weights'], g8['kv_cache'] +
                                     g8['recurrent_state'],
                                     g8['activations']),
    f'generation, 6,300 x {B6}': (g4['weights'], g4['kv_cache'] +
                                  g4['recurrent_state'], g4['activations']),
    'retrieval phase': (g8['weights'] + e['weights'] + r['weights'], 0.0,
                        max(e['activations'], r['activations'])),
    'judge, FP8 weights': (j['params'], j['kv_cache'] + j['recurrent_state'],
                           0.0)}, index=['weights', 'KV and state',
                                         'activations']).T
table['total'] = table.sum(axis=1)
table['limit'] = ['one 24 GB GPU'] * 3 + ['two GPUs, 2 x 24 GB']
print(table.to_string(float_format='{:.1f}'.format))
print('GiB; the judge stores one byte per weight (FP8), split over two '
      'cards,\nand vLLM reserves 90% of each card for its weights, cache and '
      'activations')
assert table.total.iloc[:2].max() < CARD_GIB  # generation fits one card
KEEP_GEN = table.total.iloc[2] < CARD_GIB  # the generator idles on the card
print('retrieval phase: the generator stays on the card' if KEEP_GEN else
      'retrieval phase: the generator leaves the card and comes back')
```

The generator's weights take 16.7 GiB, so it gets a whole 24 GB card. Its
cache is small, since only 8 of its 32 layers store keys and values: eight
prompts of 3,300 tokens need 1.2 GiB for cache and state together. The
estimate leaves out the allocator's slack, and measured peaks reach
21.7 GiB, so the batch stays at eight. The judge's FP8 weights alone take
25 GiB, and it gets two cards.

The baseline uses section chunks, hybrid retrieval, the first five chunks in
rank order within 3,000 tokens, and the fixed prompt. We answer the 240
`dev` questions with it and with three other arms. The *closed-book* arm
gives no passages, a floor. The *oracle* arm gives only the chunks of the
gold spans, as a perfect retriever would. And `base-repeat` answers the same
questions with the same settings in other batches. A normal run also
generates the first recorded batch again and compares the pipeline's
batched generation with the plain decoding loop `d2l.generate` of
:numref:`sec_lm_d2l_checkpoint`; a difference would mean a padding bug.

```{.python .input #rag-baseline}
%%tab pytorch
RET = {'sections': d2l.BookRetriever(CORPUS, reranker=RERANKER),
       'fixed': d2l.BookRetriever(FIXED, reranker=RERANKER)}
CHUNKS = {c['id']: c for R in RET.values() for c in R.corpus.chunks}
_LOADED = {}

def generator():
    """The generator and its tokenizer, loaded on first use, on the first
    GPU (never split over two visible cards)."""
    if not _LOADED:
        _LOADED['gen'] = d2l.load_hf(
            GEN, REV[GEN], dtype=torch.bfloat16 if GPU else torch.float32,
            device_map={'': 0} if GPU else 'auto')
    return _LOADED['gen']

def free_generator():
    _LOADED.clear()
    d2l.lm_free()

def run_config(name, items, budgets=None, threshold=None):
    """Answer records of configuration name on items, in item order;
    budgets (qid: tokens) sets each question's budget for a configuration
    whose budget is 'base'."""
    c, items = CONFIGS[name], list(items)
    assert (c['budget'] == 'base') == (budgets is not None), name
    order = list(range(len(items)))
    if c['shuffle'] is not None:  # other batches for the same prompts
        random.Random(c['shuffle']).shuffle(order)
    items = [items[i] for i in order]
    if not (KEEP_GEN or c['rewrite']):  # retrieval without the generator
        free_generator()
    assert KEEP_GEN or not c['rewrite'], 'the rewriter needs the card too'
    pipe = d2l.RAGPipeline(  # the generator joins when it is needed
        RET[c['chunking']], *(generator() if c['rewrite'] else (None, None)),
        k=c['k'], order=c['order'],
        budget_tokens=BUDGET if c['budget'] == 'base' else c['budget'],
        rewrite=c['rewrite'], rerank_k=c['rerank_k'], depth=DEPTH,
        max_new_tokens=c['max_new_tokens'], prompt=c['prompt'],
        verify=c['verify'], abstain_below=threshold)
    if c['context'] == 'none':
        ctx = [dict(query_used=None, ranked_ids=[], context=None)
               for _ in items]
    else:
        ctx = pipe.prepare(items, None if budgets is None else
                           [budgets[q['qid']] for q in items])
    if c['context'] == 'gold':  # unanswerable items keep the retrieved
        for p, q in zip(ctx, items):
            if q['answerable']:
                p['context'] = d2l.assemble_context(
                    gold_context(q), BUDGET, k=c['k'])[0]
    pipe.model, pipe.tokenizer = generator()
    res = pipe.answer(items, batch_size=c['batch'], contexts=ctx)
    recs = [dict(config=name, split=q['split'], settings=settings(name),
                 **{k: v for k, v in r._asdict().items() if k != 'question'})
            for q, r in zip(items, res)]
    return [r for _, r in sorted(zip(order, recs), key=lambda t: t[0])]

def gold_context(q):
    """The chunks of the gold spans (not the alternatives), in span
    order."""
    ids = [i for s in q['gold_spans'] for i in CORPUS.span_to_chunk_ids(
        s['doc_id'], s['char_start'], s['char_end'])]
    return [CORPUS[i] for i in dict.fromkeys(ids)]

def save_gens(name, recs):
    d2l.lm_save_table(ARTIFACTS[name][1], pd.DataFrame(recs),
                      on=['config', 'qid'], key=ARTIFACTS[name][0])

def answer_key(r):
    """What a verdict depends on: the answer, its context and citations
    (a regenerated answer needs a new verdict)."""
    return hashlib.sha1(json.dumps([r['answer'], list(r['context_ids']), [
        int(n) for n in r['cited_ids']]]).encode()).hexdigest()[:16]

def load_gens():
    """Every answer made so far, one row per (configuration, question),
    with the answer_sha its verdict must carry; each configuration's
    settings are checked against this notebook's."""
    df = d2l._rag_records([run_path(n) for n in ('rag-gen-base',
                                                 'rag-gen-stages',
                                                 'rag-gen-fix')])
    for name, s in (df.groupby('config')['settings'] if len(df) else []):
        assert (s == settings(name)).all(), f'{name}: other settings'
    if len(df):  # by the current rule, not the one of the run that wrote it
        df['abstained'] = [d2l._rag_abstained(a) for a in df.answer]
    return df.assign(answer_sha=[answer_key(r)
                                 for r in df.to_dict('records')])

def run_configs(names, items, **kw):
    """run_config for each name, with its minutes and peak memory."""
    recs = []
    for name in names:
        t = time.time()
        recs += run_config(name, items, **kw)
        print(f'{name}: {(time.time() - t) / 60:.1f} min, {d2l.lm_peak()}')
    return recs

if 'rag-gen-base' in RERUN:
    save_gens('rag-gen-base', run_configs(
        ['closed', 'base', 'oracle', 'base-repeat'], DEV))
GENS = load_gens()
print(GENS.assign(context_tokens=[sum(CHUNKS[i]['n_tokens'] for i in c)
                                  for c in GENS.context_ids])
      .groupby(['split', 'config'], sort=False).agg(
          answers=('qid', 'size'), context_tokens=('context_tokens', 'mean'),
          prompt_tokens=('n_prompt_tokens', 'mean'),
          new_tokens=('n_new_tokens', 'mean')).round(0).astype(int))

LIVE = d2l.BookQA(list(DEV)[:N_LIVE], 'dev')  # the registered run's first
live = run_config('base', LIVE)               # batch, generated again
print(f'live generation: {d2l.lm_peak()}')
ref = GENS[GENS.config == 'base'].set_index('qid')
tok = generator()[1]  # no global name for the model: freeing it frees it

def prefix_share(a, b):
    """The share of tokens before the first difference of two texts."""
    x, y = (tok(s, add_special_tokens=False)['input_ids'] for s in (a, b))
    n = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v),
             min(len(x), len(y)))
    return n / max(len(x), len(y), 1)

same = [prefix_share(r['answer'], ref.answer[r['qid']]) for r in live]
ctx = sum(r['context_ids'] == list(ref.context_ids[r['qid']]) for r in live)
print(f'{N_LIVE} base answers generated again: {ctx} with the registered '
      f'context, {sum(s == 1 for s in same)} identical answers\n  tokens '
      f'agreeing before the first difference: {np.mean(same):.3f}')
oracle = d2l.generate(*generator(), [d2l.rag_prompt(
    q['question'], [CHUNKS[i] for i in r['context_ids']])
    for q, r in zip(LIVE, live)], chat=True, enable_thinking=False,
    max_new_tokens=MAX_NEW, batch_size=BATCH)
same = [prefix_share(r['answer'], g.text.strip())
        for r, g in zip(live, oracle)]
agree = sum(s == 1 for s in same) >= N_LIVE - 1 and np.mean(same) >= 0.95
print(f'the same prompts through d2l.generate: {sum(s == 1 for s in same)} '
      f'of {N_LIVE} identical\n  tokens agreeing before the first '
      f'difference: {np.mean(same):.3f}: ' + (
          'the two loops agree' if agree else
          'a padding or logits_to_keep bug'))
assert agree, 'batched generation differs from the plain decoding loop'
free_generator()  # the card goes to the retriever and the judge
```

Five section chunks hold about 1,700 tokens and never reach the budget,
since five chunks of at most 512 tokens hold at most 2,560. The limit of
five chunks, not the budget, bounds the baseline's context. (Budgets count
tokens of the corpus's tokenizer, which differ from the generator's prompt
tokens.) Answers run to about 110 tokens. The live batch reproduces the
recorded answers, and the two decoding loops agree token for token.

### Checking the Judge

A regeneration judges with a vLLM server that splits the judge's FP8 weights
over two 24 GB GPUs. vLLM pins its own PyTorch and lives in an environment
of its own (`uv venv`, then `uv pip install vllm==0.33.0`), whose `vllm`
executable `D2L_VLLM` names. Without two such GPUs, the same open weights
can be called on OpenRouter for about \$3.3 per 1,000 judgments; local and
API verdicts agreed on 194 of 200 answers in a check with thinking off. A
reply cut off at 3,000 new tokens, or malformed twice, leaves an answer
without a verdict, as happened to about 3% of them. Each was judged once
more with 6,000 new tokens. An answer still without a verdict is
*unjudged*: it is left out of accuracy and of the failure counts rather
than blamed on the generator. A normal run replays the verdicts and judges
twelve answers again from the recorded replies, with no GPU and no key.

```{.python .input #rag-judge}
%%tab pytorch
LABELS = d2l.api_unpack('lm-rag-judge-labels-v1')
REC_DIR = os.path.join('..', 'data', 'lm-api', 'record',
                       ARTIFACTS['rag-judge-cache'][0])
VERDICTS = ['correct', 'faithful', 'supported']
ANSWERS = {(r['config'], r['qid']): r for r in GENS.to_dict('records')}
read = lambda f: [json.loads(l) for l in open(os.path.join(LABELS, f))]
CAL = [p for p in read('pairs.jsonl') if p['set'] == 'calibration']

def passage(cid):
    c = CHUNKS[cid]
    return ' > '.join(c['heading_path']) + '\n' + c['text']

def judge_client(backend, cache, mode='replay', url='http://127.0.0.1/v1',
                 timeout=120.0):
    """The judge through the local vLLM server or OpenRouter (same weights,
    BF16), recording to or replaying from cache; timeout: seconds a request
    may take (not part of its hash)."""
    if backend == 'local':
        client = d2l.rag_judge_client(url, cache=cache, mode=mode)
    else:
        with open(os.path.join(LABELS, 'listing.json')) as f:
            listing = json.load(f)
        client = d2l.openrouter_api(
            'qwen/qwen3.8-27b', 'deepinfra/bf16', listing=listing,
            cache=cache, mode=mode, cost_cap_usd=15.0,
            reasoning={'enabled': True})
    client.timeout = timeout
    return client

def judge_records(client, recs, concurrency, backend, tokens=None):
    """Judgment records of answer records (dicts); the judge may write
    tokens new tokens (default JUDGE['max_new_tokens'])."""
    tokens = tokens or JUDGE['max_new_tokens']
    v = d2l.rag_judge(client, None, [ITEMS[r['qid']] for r in recs], recs,
                      [[passage(i) for i in r['context_ids']] for r in recs],
                      batch_size=concurrency, max_new_tokens=tokens)
    return [dict(config=r['config'], qid=r['qid'], concurrency=concurrency,
                 backend=backend, answer_sha=answer_key(r),
                 max_new_tokens=tokens, **x) for r, x in zip(recs, v)]

def no_verdict(v, rec):
    """Whether the judge left out a verdict it was asked for on answer
    record rec (its reply cut off or malformed, twice)."""
    if d2l._rag_abstained(rec['answer']):
        return False  # decided by rule
    keys = VERDICTS if len(rec['context_ids']) else ['correct']
    return any(v[k] is None for k in keys)

def calibration_verdicts(cache, tokens=None, pairs=None, mode='replay',
                         **kw):
    """The judge's verdicts on calibration pairs (default: all), replayed
    from cache or recorded to it (the local judge; kw: judge_client's)."""
    pairs = CAL if pairs is None else pairs
    return d2l.rag_judge(judge_client('local', cache, mode, **kw), None,
                         [ITEMS[p['qid']] for p in pairs], pairs,
                         [[passage(i) for i in p['context_ids']]
                          for p in pairs], max_new_tokens=tokens)

def load_judgments(path):
    """Judgment records, verdicts as True, False or None."""
    df = d2l._rag_records([path], on=['config', 'qid', 'concurrency'])
    for k in VERDICTS:
        df[k] = [None if v is None or v != v else bool(v)
                 for v in df.get(k, [])]
    for k in ('backend', 'answer_sha'):
        df[k] = df.get(k, [None] * len(df))
    df['max_new_tokens'] = [  # rows from before the re-judging: the default
        JUDGE['max_new_tokens'] if v is None or v != v else int(v)
        for v in df.get('max_new_tokens', [None] * len(df))]
    return df

CAL_V0 = calibration_verdicts(os.path.join(  # as recorded at selection
    LABELS, 'api', 'judge-local-qwen38-27b-on.jsonl'))
CAL_V = [dict(x) for x in CAL_V0]  # with the pairs judged again
JUD_FILE = run_path('rag-judge')
if 'rag-judge' in RERUN:
    free_generator()
    for R in RET.values():
        R.release()
    big = sum(torch.cuda.get_device_properties(i).total_memory > 22 * 2**30
              for i in range(torch.cuda.device_count()))
    vllm = os.environ.get('D2L_VLLM') or shutil.which('vllm')
    backend = ('local' if vllm and big >= 2 else 'openrouter'
               if os.environ.get('OPENROUTER_API_KEY') else None)
    assert backend, ('judging needs vLLM (D2L_VLLM) and two 24 GB GPUs, '
                     'or OPENROUTER_API_KEY for the same model on OpenRouter')
    print(f'judge backend: {backend}')
    old = load_judgments(JUD_FILE if os.path.exists(JUD_FILE) else
                         d2l.lm_registered(ARTIFACTS['rag-judge'][0]))
    key = dict(zip(zip(GENS.config, GENS.qid), GENS.answer_sha))
    rule = set(zip(GENS.config[GENS.abstained], GENS.qid[GENS.abstained]))
    rows = [r for r in old.to_dict('records')  # verdicts on these answers
            if key.get((r['config'], r['qid'])) == r.get('answer_sha')]
    print(f'{len(old) - len(rows)} verdicts on answers no longer present')
    rows = [r for r in rows  # abstentions: decided again, by rule (no call)
            if (r['config'], r['qid']) not in rule]
    done = {(r['config'], r['qid']) for r in rows
            if r['concurrency'] == CONCURRENCY}
    todo = [r for r in GENS.to_dict('records')
            if (r['config'], r['qid']) not in done]
    os.makedirs(REC_DIR, exist_ok=True)
    server = d2l.rag_judge_server(2) if backend == 'local' else None
    try:
        url = server.start() if server else None
        client = judge_client(backend, os.path.join(REC_DIR, 'judge.jsonl'),
                              'record', url)
        last = 0.0  # a chunk starts only if it should end within budget
        # a request that fails twice raises: the chunks saved so far stay,
        # the recorded replies too, and the next run resumes
        while todo and time.time() - T0 + last < 60 * JUDGE_MINUTES:
            t = time.time()
            rows += judge_records(client, todo[:CHUNK], CONCURRENCY, backend)
            todo, last = todo[CHUNK:], time.time() - t
            d2l.write_jsonl(JUD_FILE, rows)  # a stopped run resumes here
        # once every answer has a judgment: each answer left without a
        # verdict is judged once more, with RETRY_TOKENS new tokens
        redo = [ANSWERS[r['config'], r['qid']] for r in rows
                if r['concurrency'] == CONCURRENCY and r['max_new_tokens'] <
                RETRY_TOKENS and no_verdict(r, ANSWERS[r['config'], r['qid']])]
        retry = judge_client(backend, os.path.join(  # the longer replies
            REC_DIR, 'judge-retry.jsonl'), 'record', url, timeout=1200.0)
        while (not todo and redo and
               time.time() - T0 + last < 60 * JUDGE_MINUTES):
            t = time.time()
            new = {(r['config'], r['qid']): r for r in judge_records(
                retry, redo[:CHUNK], CONCURRENCY, backend, RETRY_TOKENS)}
            rows = [new.get((r['config'], r['qid']), r)
                    if r['concurrency'] == CONCURRENCY else r for r in rows]
            redo, last = redo[CHUNK:], time.time() - t
            d2l.write_jsonl(JUD_FILE, rows)
        if backend == 'local':  # the calibration pairs left without verdict
            calibration_verdicts(os.path.join(  # recorded once (local judge)
                REC_DIR, 'calibration-retry.jsonl'), RETRY_TOKENS, [
                    p for p, x in zip(CAL, CAL_V0) if no_verdict(x, p)],
                'record', url=url, timeout=1200.0)
        base = sorted(r['qid'] for r in rows if r['config'] == 'base' and
                      r['concurrency'] == CONCURRENCY and
                      r['qid'].startswith('dev'))
        if (not any(r['concurrency'] != CONCURRENCY for r in rows)
                and len(base) >= N_REPEAT
                and time.time() - T0 + last < 60 * JUDGE_MINUTES):
            pick = set(random.Random('repeat|0').sample(base, N_REPEAT))
            rows += judge_records(
                judge_client(backend, os.path.join(REC_DIR,
                                                   'judge-rep16.jsonl'),
                             'record', url),
                [r for r in GENS.to_dict('records') if r['config'] == 'base'
                 and r['qid'] in pick and not r['abstained']],  # by rule
                2 * CONCURRENCY, backend)
    finally:
        if server:
            server.stop()
    left = len(todo) + len(redo)
    print(f'{len(todo)} answers still to judge, {len(redo)} to judge again: '
          'run D2L_RERUN=rag-judge again' if left else
          'every answer is judged')
    d2l.lm_save_table(ARTIFACTS['rag-judge'][1], pd.DataFrame(rows),
                      on=['config', 'qid', 'concurrency'],
                      key=ARTIFACTS['rag-judge'][0])
    members = {f: open(os.path.join(REC_DIR, f), 'rb').read()
               for f in sorted(os.listdir(REC_DIR))}
    members['manifest.json'] = json.dumps(dict(
        judge=JUDGE, prompt='RAG_JUDGE_V1', backends=sorted(
            {r['backend'] for r in rows}), concurrency=CONCURRENCY,
        repeat_concurrency=2 * CONCURRENCY, retry_tokens=RETRY_TOKENS,
        n=len(rows)), indent=1).encode()
    d2l.lm_save_rerun(ARTIFACTS['rag-judge-cache'][1],
                      lambda p: d2l.lm_write_zip(p, members))
    print(f'rag-judge: {(time.time() - T0) / 60:.1f} min since the start')
JUD = load_judgments(JUD_FILE)
judged = JUD[JUD.concurrency == CONCURRENCY].merge(  # on these answers
    GENS[['config', 'qid', 'answer_sha']], on=['config', 'qid',
                                               'answer_sha'])
cache = (REC_DIR if 'rag-judge' in RERUN else
         d2l.lm_registered(ARTIFACTS['rag-judge-cache'][0]) and
         d2l.api_unpack(ARTIFACTS['rag-judge-cache'][0]))
b = judged[judged.config == 'base'].sort_values('qid')
for tokens, file, n in ((JUDGE['max_new_tokens'], 'judge.jsonl', 8),
                        (RETRY_TOKENS, 'judge-retry.jsonl', 4)):
    first = b[b.max_new_tokens == tokens].head(n)  # replayed from the cache
    if not (cache and len(first)):
        continue
    recs = [ANSWERS[k] for k in zip(first.config, first.qid)]
    again = pd.DataFrame(judge_records(
        judge_client(first.backend.iloc[0], os.path.join(cache, file)),
        recs, CONCURRENCY, first.backend.iloc[0], tokens))
    assert (again[VERDICTS].astype(str).values ==
            first[VERDICTS].astype(str).values).all()
    print(f'{len(first)} base judgments with {tokens:,} new tokens replayed '
          'from the judge cache: identical')
retry = cache and os.path.join(cache, 'calibration-retry.jsonl')
if retry and os.path.exists(retry):  # the calibration pairs judged again
    redo = [i for i, (p, x) in enumerate(zip(CAL, CAL_V0))
            if no_verdict(x, p)]
    for i, x in zip(redo, calibration_verdicts(retry, RETRY_TOKENS,
                                               [CAL[i] for i in redo])):
        CAL_V[i] = x
print(pd.DataFrame({
    'answers': GENS.groupby('config').size(),
    'judged': judged.groupby('config').size(),
    'judged again': judged.groupby('config').max_new_tokens.apply(
        lambda t: (t == RETRY_TOKENS).sum()),
    'unjudged': judged.assign(none=judged.correct.isna()).groupby(
        'config').none.sum()}).fillna(0).astype(int).to_string())
print(f'judged again: with {RETRY_TOKENS:,} new tokens, once\nunjudged: '
      'still no verdict on correctness (left out of accuracy)')
```

The replayed verdicts match the recorded ones. Every baseline answer has a
verdict, and seven answers of other arms stay unjudged.

Why this judge? It was chosen by measurement. Two frontier models from
other families, Claude Opus 5.5 and Gemini 4 Argon, labeled 200 answers of
an earlier generator, Qwen3-4B, and nine Qwen candidates of three sizes
judged them, with thinking on and off. With thinking off every candidate was
lenient, passing many answers that both labelers call wrong. With thinking
on, a judge works through the key facts before it decides, which is what
catches a missing fact, but the smaller candidates reasoned until their
replies were cut off. Half of the 200 answers served this choice. The other
100, the *calibration set*, test the chosen judge before we read any
accuracy: against each labeler, and against their *consensus*, the answers
on which both agree. The labels come from models, not from people, and they
were made on answers of the earlier generator; ours may make subtler
mistakes, which this check cannot see.

```{.python .input #rag-judge-validation}
%%tab pytorch
cal, v = CAL, CAL_V  # v: with the pairs judged again (rag-judge cell)
lab = {r['qid']: r for r in read('dev_judge_labels.jsonl')}
recorded = {r['qid']: r for r in read(
    os.path.join('candidates', 'qwen38-27b-local-on.jsonl'))}
graded = [not d2l._rag_abstained(p['answer']) for p in cal]
assert all(x[k] == recorded[p['qid']][k] for p, x, g in zip(
    cal, CAL_V0, graded) if g for k in VERDICTS)  # the replies, replayed
again = sum(no_verdict(x, p) for p, x in zip(cal, CAL_V0))
print(f'{len(cal) - sum(graded)} of {len(cal)} calibration answers abstain '
      f'(decided by rule); the judge grades the others\n  {again} without a '
      f'verdict judged again with {RETRY_TOKENS:,} new tokens')

def kappa(a, b):
    """Cohen's kappa of two 0/1 arrays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return ((a == b).mean() - pe) / (1 - pe) if pe < 1 else 1.0

def agreement(a, b, n_boot=2000, seed=0):
    """Agreement and kappa with 95% intervals, pairs resampled."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boot = [(a[i] == b[i]).mean() for i in rng.integers(0, len(a), (
        n_boot, len(a)))]
    rng = np.random.default_rng(seed)
    kb = [kappa(a[i], b[i]) for i in rng.integers(0, len(a), (
        n_boot, len(a)))]
    q = lambda x: np.quantile(x, [0.025, 0.975])
    return ((a == b).mean(), *q(boot)), (kappa(a, b), *q(kb))

rows = []
for k in ('correct', 'faithful'):
    for who, field in (('frontier', f'{k}_frontier'),
                       ('second', f'{k}_second'), ('consensus', k)):
        keep = [(x[k], lab[p['qid']][field]) for p, x in zip(cal, v)
                if None not in (x[k], lab[p['qid']][field])]  # unjudged out
        (acc, alo, ahi), (kap, klo, khi) = agreement(*zip(*keep))
        rows.append({'verdict': k, 'against': who, 'pairs': len(keep),
                     'agreement': f'{acc:.2f} [{alo:.2f}, {ahi:.2f}]',
                     'kappa': f'{kap:.2f} [{klo:.2f}, {khi:.2f}]'})
print(pd.DataFrame(rows).to_string(index=False))
print(f'unjudged (left out): {sum(x["correct"] is None for x in v)} of '
      f'{len(cal)} calibration pairs')
for k in ('correct', 'faithful'):
    pairs = [(lab[p['qid']][f'{k}_frontier'], lab[p['qid']][f'{k}_second'])
             for p in cal]
    print(f'the two labellers on {k}: kappa '
          f'{kappa(*zip(*[t for t in pairs if None not in t])):.2f}')
old = next(c for c in json.load(open(os.path.join(
    LABELS, 'candidates', 'table.json'))) if c['name'] == 'qwen35-4b-off')
oc = old['calibration']['correct']
print(f'the old judge (Qwen3.5-4B, thinking off) on correct: agreement '
      f'{oc["acc"]:.2f}, kappa {oc["kappa"]:.2f} [{oc["k_lo"]:.2f}, '
      f'{oc["k_hi"]:.2f}];\n  it passes '
      f'{old["calibration"]["false_correct"][0]} of the '
      f'{old["calibration"]["false_correct"][1]} answers the labellers call '
      'wrong')
for k in ('correct', 'faithful'):  # the claims that rest on the judge
    (acc, _, _), (_, klo, _) = agreement(*zip(*[
        (x[k], lab[p['qid']][k]) for p, x in zip(cal, v)
        if None not in (x[k], lab[p['qid']][k])]))
    print(f'C0 on {k} against the consensus: agreement {acc:.2f}, kappa '
          f'interval from {klo:.2f}\n  ' + ('passes (>= 0.8, above 0.4)'
          if acc >= 0.8 and klo > 0.4 else 'fails: claims on it carry the '
          'agreement'))
print('the calibration answers come from the earlier generator, Qwen3-4B')
if len(JUD) and 2 * CONCURRENCY in JUD.concurrency.values:
    J = JUD[JUD.config == 'base']
    n16 = (J.concurrency == 2 * CONCURRENCY).sum()
    both = J[J.max_new_tokens == JUDGE['max_new_tokens']].groupby(
        'qid').filter(lambda g: g.concurrency.nunique() == 2)  # same budget
    changed = both.groupby('qid').apply(lambda g: (g[VERDICTS].astype(
        str).nunique() > 1).any(), include_groups=False)
    print(f'judged again at concurrency {2 * CONCURRENCY} (first run '
          f'{CONCURRENCY}): {changed.sum()} of {len(changed)} answers '
          f'changed a verdict\n  (of {N_REPEAT} drawn, '
          f'{N_REPEAT - n16} abstentions are decided by rule and '
          f'{n16 - len(changed)} were judged again with more tokens)')
```

The judge agrees with the consensus on 0.95 of the `correct` verdicts,
κ 0.85 [0.70, 0.97], and on 0.96 of the `faithful` ones, κ 0.87
[0.71, 0.97]. These numbers flatter, because the consensus leaves out the
hard answers on which the labelers disagree. Against each labeler alone
the judge reaches κ 0.83 and 0.71 on `correct`, and the labelers reach 0.81
with each other: the judge is about as reliable as a third labeler. The
judge it replaced, a 4B model with thinking off, passed 14 of the 21 answers
that both labelers call wrong.

### The Baseline Report

Each 95% interval of the report comes from a bootstrap that resamples
documents, since questions drafted from one document share its difficulty
(:numref:`sec_lm_eval_suites`). Differences between arms are paired: both
arms answer the same questions, and the interval is computed for the
per-question differences, which removes the questions' difficulty from the
comparison (:numref:`subsec_mdl-paired-comparisons`). Make a prediction
first:
`base-repeat` sends the same 240 prompts to the same model with greedy
decoding, and only the neighbors in each batch of eight differ. How many
answers will change?

```{.python .input #rag-baseline-report}
%%tab pytorch
CLUSTER = {q['qid']: (q['gold_spans'] or [{'doc_id': CORPUS[
    q['anchor_ids'][0]]['doc_id']}])[0]['doc_id'] for q in ITEMS.values()}

def signflip_p(d, cluster, n=10_000):
    """Two-sided sign-flip p of the mean of per-document mean
    differences."""
    m = np.array([d[cluster == c].mean() for c in np.unique(cluster)])
    flips = np.random.default_rng(0).choice([-1.0, 1.0], (n, len(m)))
    return np.mean(np.abs(flips @ m) >= abs(m.sum()) - 1e-12)

def ci(a, b=None, cluster=None):
    """The bootstrap of the mean of per-question scores a (or of a - b,
    paired), NaNs dropped, resampling documents when there are 30 or
    more, else questions: (result or None, by documents, a, b,
    cluster)."""
    a = np.asarray(a, float)
    b = None if b is None else np.asarray(b, float)
    keep = ~np.isnan(a) & (True if b is None else ~np.isnan(b))
    a, cluster = a[keep], np.asarray(cluster)[keep]
    b = None if b is None else b[keep]
    by_doc = len(set(cluster)) >= 30
    if len(a) < 2:
        return None, by_doc, a, b, cluster
    with warnings.catch_warnings():  # few items: boot's * says so
        warnings.simplefilter('ignore')
        r = (d2l.bootstrap_ci(a, cluster=cluster if by_doc else None)
             if b is None else d2l.paired_bootstrap(
                 a, b, cluster=cluster if by_doc else None))
    return r, by_doc, a, b, cluster

def boot(a, b=None, cluster=None):
    """Mean of per-question scores a (or of a - b, paired), NaNs dropped,
    with a 95% interval that resamples documents, marked * below 100
    questions; below 30 documents questions are resampled instead, marked
    + (and a paired difference gets a document-level sign-flip p)."""
    r, by_doc, a, b, cluster = ci(a, b, cluster)
    if r is None:
        return 'n/a'
    f = '.3f' if b is None else '+.3f'
    out = f'{r.mean:{f}} [{r.ci[0]:{f}}, {r.ci[1]:{f}}]'
    if by_doc:
        return out + ('' if len(a) >= 100 else '*')
    if b is None:
        return out + '+'
    p = signflip_p(a - b, cluster)
    return out + f'+ (sign-flip p {p:.3f})'

def scores(config, split='dev'):
    """One row per judged answer of a configuration on a split, with
    per-question scores (NaN where a score does not apply, and accuracy NaN
    where the judge gave no verdict: an unjudged answer is left out)."""
    g = GENS[(GENS.config == config) & (GENS.split == split)]
    j = JUD[JUD.concurrency == CONCURRENCY][['config', 'qid', 'answer_sha']
                                            + VERDICTS]
    df = g.merge(j, on=['config', 'qid', 'answer_sha']).sort_values(
        'qid', ignore_index=True)
    corpus = RET[CONFIGS[config]['chunking']].corpus
    items = [ITEMS[q] for q in df.qid]
    ans = np.array([q['answerable'] for q in items])
    rel = [d2l.gold_chunk_ids(q, corpus) for q in items]
    cm = d2l.citation_metrics(df.to_dict('records'), items, corpus)
    has_ctx = df.context_ids.map(len).to_numpy() > 0
    answered = has_ctx & ~df.abstained.to_numpy(bool)
    flag = lambda col: np.array([np.nan if v is None else float(v)
                                 for v in df[col]])
    return pd.DataFrame({
        'qid': df.qid, 'cluster': [CLUSTER[q] for q in df.qid],
        'answerable': ans, 'type': [q['type'] for q in items],
        'book_specific': ['general_knowledge' not in q['flags']
                          for q in items],
        'accuracy': flag('correct'),  # NaN: unjudged, left out
        'faithful': np.where(answered, flag('faithful'), np.nan),
        'supported': np.where(answered, flag('supported'), np.nan),
        'cite_precision': np.where(ans, cm.precision, np.nan),
        'cite_recall': np.where(ans, cm.recall, np.nan),
        'abstained': df.abstained.astype(float),
        'hit': np.where(ans, [float(bool(r & set(c))) for r, c in zip(
            rel, df.context_ids)], np.nan),
        'recall@5': np.where(ans & has_ctx, [float(bool(r & set(k[:K])))
                                             for r, k in zip(
                                                 rel, df.ranked_ids)], np.nan),
        'cov@3000': np.where(ans & has_ctx, [float(d2l.cov_at_budget(
            k, q, corpus, BUDGET)) for k, q in zip(df.ranked_ids, items)],
            np.nan),
        'answer': df.answer, 'n_prompt_tokens': df.n_prompt_tokens,
        'n_new_tokens': df.n_new_tokens, 'latency_s': df.latency_s,
        'record': df.to_dict('records'),
        'verdicts': df[VERDICTS].to_dict('records')})

def paired(a, b, col, mask=None):
    """Paired difference a - b of a score on the questions both have."""
    m = a.merge(b, on='qid', suffixes=('_a', '_b'))
    if mask is not None:
        m = m[mask(m)]
    return boot(m[f'{col}_a'], m[f'{col}_b'], m.cluster_a)

def ready(*configs, split='dev'):
    """Whether every answer of each configuration on split carries a
    verdict on that very answer (answer_sha); says which do not. Partial
    or stale judgments are never reported: between judging runs a
    configuration is left out, and a normal run stops on one."""
    k = JUD[JUD.concurrency == CONCURRENCY]  # not judged: GENS may be new
    j = set(zip(k.config, k.qid, k.answer_sha))
    missing = {}
    for c in configs:
        g = GENS[(GENS.config == c) & (GENS.split == split)]
        n = sum(k not in j for k in zip(g.config, g.qid, g.answer_sha))
        if n or not len(g):
            missing[c] = (f'{n} of {len(g)} answers lack a verdict' if len(g)
                          else 'no answers')
    if missing:
        print(f'not ready on {split}: {missing}')
        assert RERUN or all(m == 'no answers' for m in missing.values()), (
            'registered answers without verdicts: register the judge run')
    return not missing

S = {c: scores(c) for c in ('closed', 'base', 'oracle', 'base-repeat')
     if ready(c)}
METRICS = ['recall@5', 'cov@3000', 'accuracy', 'faithful', 'cite_precision',
           'cite_recall', 'supported']
if len(S) == 4:
    rows = {c: {m: boot(S[c][m], cluster=S[c].cluster) for m in METRICS}
            for c in ('closed', 'base', 'oracle')}
    for c in rows:
        s = S[c]
        rows[c]['abstain, unanswerable'] = boot(
            s.abstained[~s.answerable], cluster=s.cluster[~s.answerable])
        rows[c]['abstain, answerable'] = boot(
            s.abstained[s.answerable], cluster=s.cluster[s.answerable])
    print(pd.DataFrame(rows).to_string())
    print('unjudged, left out of accuracy: ' + ', '.join(
        f'{c} {S[c].accuracy.isna().sum()}' for c in S))
    B = S['base']
    ans = lambda m: m.answerable_a
    spec = lambda m: m.answerable_a & m.book_specific_a
    print(pd.DataFrame([
        ('base - closed', 'all', paired(B, S['closed'], 'accuracy')),
        ('oracle - base', 'all', paired(S['oracle'], B, 'accuracy')),
        ('oracle - base', 'answerable', paired(S['oracle'], B, 'accuracy',
                                               ans)),
        ('base - closed', 'book-specific, answerable',
         paired(B, S['closed'], 'accuracy', spec)),
        ('base-repeat - base', 'all (other batches only)',
         paired(S['base-repeat'], B, 'accuracy'))],
        columns=['difference', 'questions', 'accuracy [95% CI]'])
          .to_string(index=False))
    C = S['closed'][S['closed'].answerable & S['closed'].book_specific]
    Bs = B[B.answerable & B.book_specific]
    print(f'book-specific answerable questions ({len(C)}): closed book '
          f'{boot(C.accuracy, cluster=C.cluster)},\n  '
          f'base {boot(Bs.accuracy, cluster=Bs.cluster)}')
    A = B[B.answerable].merge(S['closed'][['qid', 'accuracy']], on='qid',
                              suffixes=('', '_closed'))
    for name, m in (('a relevant chunk in the context', A.hit == 1),
                    ('no relevant chunk in the context', A.hit == 0),
                    ('closed-book arm correct', A.accuracy_closed == 1),
                    ('closed-book arm wrong', A.accuracy_closed == 0)):
        print(f'base accuracy, {name} ({m.sum()}): '
              f'{boot(A.accuracy[m], cluster=A.cluster[m])}')
if {'base', 'base-repeat'} <= set(GENS.config):  # answers, not verdicts
    b, r = (GENS[(GENS.config == c) & (GENS.split == 'dev')].set_index('qid')
            for c in ('base', 'base-repeat'))
    r = r.loc[b.index]
    same_ctx = b.context_ids.map(tuple) == r.context_ids.map(tuple)
    differ = b.answer != r.answer
    print(f'base and base-repeat (same questions, other batches): '
          f'{(~same_ctx).sum()} of {len(b)} contexts differ\n  (query '
          f'embeddings batched differently); with the same context '
          f'{differ[same_ctx].sum()} of {same_ctx.sum()} answers differ')
print('* fewer than 100 questions (documents resampled)\n+ fewer than 30 '
      'documents: questions resampled, an interval that is likely too '
      'narrow')
```

A relevant chunk is among the first five for 0.97 of the answerable
questions, so retrieval is nearly solved here. The generator answers 0.89
[0.85, 0.93] of all `dev` questions correctly, and 0.30 without passages, a
gain of +0.59 [+0.53, +0.66]. The closed-book score is not zero, nor should
it be: the book is public, and some questions ask for general knowledge. On
the 145 answerable questions that the checker marked as specific to the
book, closed-book accuracy falls to 0.12 [0.06, 0.18], against 0.90 with
retrieval.

The oracle arm answers 0.92 [0.89, 0.95], and its gain over the baseline,
+0.03 [$-0.02$, +0.08], is not distinguishable from zero. Why so little? Call
a question a *hit* when its context holds a relevant chunk. Let $h$ be the
share of hits among the answerable questions, and $a_1$ and $a_0$ the
accuracy on hits and on misses. Then

$$\textrm{accuracy} = h\, a_1 + (1 - h)\, a_0.$$
:eqlabel:`eq_lm_rag_hit`

Accuracy is a mixture of how often retrieval succeeds and how well the
generator does in each case. Here 213 of 220 contexts hold a relevant chunk,
$h = 213/220$, with $a_1 = 0.91$, and $a_0 = 0.29$ on the seven misses. If
the accuracy on hits stayed at $a_1$, a perfect retriever would add
$(1 - h)(a_1 - a_0) \approx 0.032 \times 0.63 \approx 0.02$. The oracle
measured +0.04 [$-0.01$, +0.09] on the answerable questions, within noise of
that estimate. It may well add a little more, since it changes the hits
too: it passes only the gold chunks, about 480 context tokens against the
baseline's 1,700, and so removes the distracting passages as well. Either
way the generator limits accuracy. Of the 24 errors on answerable questions,
19 were made with a relevant passage in the context.

Citation precision is 0.72 and recall 0.85, and the judge finds the cited
passages sufficient for 0.82 of the answers. Precision from gold spans is
strict, since a citation of a supporting passage that is not gold counts
against it. The baseline abstains on 18 of the 20 unanswerable questions and
on 0.02 of the answerable ones.

And the batches? Not one of the 240 contexts changed, yet 108 of the 240
answers did: 45%. Accuracy did not move: the difference is $-0.004$, with
an interval from $-0.03$ to +0.02. Nothing is broken. In bf16 a prompt's
logits move by 0.2 to 0.6 when the padding of its batch changes, a few
rounding steps of 0.125 at logits near 25. That flips the greedy choice
wherever the two likeliest tokens are nearly tied, and in two of eight
prompts we checked they were tied exactly. In fp32, on a 2B model of the
same family, padding moves the logits by at most $10^{-4}$ and never
changes the most probable token.

Greedy decoding is therefore deterministic only for a fixed batch and
precision, the near-ties of :numref:`subsec_lm_hf_kernels` once more. An
evaluation must batch the configurations it compares alike, or count
re-batching as noise (:numref:`sec_lm_scoring`). Here re-batching moves
accuracy by less than 0.03, and a stage change has to do better than that.
The judge is batched by its server too, so the same effect reaches the
verdicts: judged again with twice as many requests in flight, 5 of 54
answers changed a verdict. Every arm was judged at the same concurrency, so
this noise widens the paired intervals rather than favoring one arm.

Now the failure of the opening, picked by a fixed rule: the first `dev`
answer, by question id, whose context held a relevant chunk, that the judge
called incorrect, and whose failure class (:numref:`subsec_lm_rag_failures`)
a second grader confirmed on reading the case. The second grader is a
frontier model that labels the failure cards of that subsection, so the rule
makes it unlikely, though not impossible, that the example is a judge
error.

```{.python .input #rag-opening-item}
%%tab pytorch
# The rule: the first base dev answer (in qid order) with a relevant chunk
# in its context that the judge called incorrect, and whose failure card
# carries the same class from the second labeller as from the rule: a
# failure of the pipeline that two graders agree on, not a judge error
path = run_path('rag-cards')
LABEL = {c['qid']: c['labeller']['label'] for c in (
    d2l.read_jsonl(path).to_dict('records') if path else [])
    if c.get('labeller')}
if 'base' in S and LABEL:
    B = S['base']
    cls = [d2l.classify_failure(r, ITEMS[q], v, CORPUS, DEPTH)
           for q, r, v in zip(B.qid, B.record, B.verdicts)]
    pick = [(q, c) for q, c, h in zip(B.qid, cls, B.hit) if h == 1 and c in (
        'ignored', 'unsupported', 'false_abstain') and LABEL.get(q) == c]
    d2l.print_wrapped(f'{len(pick)} base answers qualify: ' + ', '.join(
        f'{q} ({c})' for q, c in pick), width=96)
    row = B.set_index('qid').loc[pick[0][0]]
    q, rec = ITEMS[pick[0][0]], row.record
    rel = d2l.gold_chunk_ids(q, CORPUS)
    rank = next(i for i, c in enumerate(rec['ranked_ids'], 1) if c in rel)
    wrap = lambda t: textwrap.fill(t, 96, subsequent_indent='    ')
    print('\n' + wrap(f'{q["qid"]} ({q["type"]}): {q["question"]}'))
    print(wrap(f'reference: {q["reference"]}'))
    print(wrap(f'key facts: {q["key_facts"]}'))
    print(f'first relevant chunk at rank {rank}; context:')
    for n, c in enumerate(rec['context_ids'], 1):
        print(f'  [{n}] {c.split(":")[1]}' + (' (relevant)' if c in rel
                                               else ''))
    n, c = next((n, c) for n, c in enumerate(rec['context_ids'], 1)
                if c in rel)
    last = re.split(r'(?<=[.!?])\s+', CHUNKS[c]['text'].strip())[-1]
    print(wrap(f'last sentence of [{n}]: {last}'))
    print(wrap(f'answer: {rec["answer"]}'))
    print(f'verdicts: {row.verdicts}\nclass: {pick[0][1]}, by the rule and '
          'by the second labeller')
```

The answering passage was ranked first, and its last sentence, printed above
and quoted at the start of this section, holds the missing key fact. The answer cites passages
two and three and leaves out that no warning is given. The passage was
neither missing nor contradicted. It was ignored.

## Changing One Stage at a Time
:label:`subsec_lm_rag_stages`

The baseline leaves 26 questions wrong. Could another chunking, another
query or another context have saved some of them? We now change one stage
at a time and compare with the baseline on the same questions. Chunking
comes first, and it can be measured before any answer is generated: section
chunks follow the book's headings and paragraphs, fixed windows of 256
tokens overlapping by 64 do not. Which covers more at equal budgets?

```{.python .input #rag-coverage}
%%tab pytorch
ANS = [q for q in DEV if q['answerable']]
mode = 'hybrid' if GPU else 'bm25'  # the CPU path: no dense vectors of
COV = {}                            # the 9,447 fixed windows
for name, tag in (('sections', 'sec512'), ('fixed', 'fix256o64')):
    R = RET[name]
    hits = R.search_batch([q['question'] for q in ANS], k=DEPTH, mode=mode)
    R.release()
    for T in (1000, 2000, 3000):
        COV[tag, T] = np.array([d2l.cov_at_budget(
            [h.chunk_id for h in hs], q, R.corpus, T)
            for hs, q in zip(hits, ANS)], float)
cl = np.array([CLUSTER[q['qid']] for q in ANS])
print(f'{len(ANS)} answerable dev questions, {mode} retrieval')
print(pd.DataFrame({T: {'sections': boot(COV['sec512', T], cluster=cl),
                        'fixed 256/64': boot(COV['fix256o64', T], cluster=cl),
                        'fixed - sections': boot(COV['fix256o64', T],
                                                 COV['sec512', T], cl)}
                    for T in (1000, 2000, 3000)}).T.rename_axis('cov@T')
      .to_string())
```

At 1,000 tokens section chunks cover 0.91 of the questions and windows
0.84, a difference of $-0.07$ [$-0.11$, $-0.02$]. A section holds a span
whole where a window may cut it, and overlapping windows repeat text. By
3,000 tokens both find room, and the difference shrinks to $-0.01$
[$-0.04$, +0.02].

Coverage says that the gold text is in the context, not that the generator
can use it. The next cell answers with five changes:

- fixed windows, each question getting at most the tokens of its baseline
  context (packing stops before the window that would pass them, so the
  windows get slightly fewer);
- the question rewritten by the generator as a search query, with the
  reranker, which reads the original question, reordering the fused top 50;
  the two go together because the reranker decides which of the rewrite's
  candidates reach the context, and so guards against a rewrite that drifts
  from the question;
- the baseline context in reverse order;
- budgets of 1,500 and of 6,000 tokens, filled in rank order without the
  limit of five chunks.

A sixth arm is a control rather than a candidate. Packed as above, fixed
windows get fewer context tokens than the baseline, so a loss could come
from the missing tokens rather than from the cut. The control, `fixed-over`,
packs windows until they pass the baseline's tokens by less than one window
and so gets slightly more context than the baseline. Which of the changes,
would you guess, moves accuracy with a paired interval that excludes zero?

```{.python .input #rag-stages}
%%tab pytorch
def context_tokens(recs):
    """Tokens of the context of each answer record: the budget that
    'fixed' gets for the same question."""
    return {r['qid']: sum(CHUNKS[i]['n_tokens'] for i in r['context_ids'])
            for r in recs}

if 'rag-gen-stages' in RERUN:
    names = os.environ.get('D2L_RAG_STAGES', ','.join(STAGES + CONTROLS))
    names = names.split(',')
    assert set(names) <= set(STAGES + CONTROLS), names
    path = run_path('rag-gen-stages')  # the other configurations are kept
    if not os.path.exists(path):  # from an earlier run, else registered
        path = d2l.lm_registered(ARTIFACTS['rag-gen-stages'][0])
    recs = [r for r in d2l._rag_records([path]).to_dict('records')
            if r['config'] not in names]
    tokens = context_tokens(GENS[(GENS.config == 'base') &
                                 (GENS.split == 'dev')].to_dict('records'))
    for name in names:
        c = CONFIGS[name]
        recs += run_configs([name], DEV, budgets={
            q: t + c.get('over', 0) for q, t in tokens.items()}
                            if c['budget'] == 'base' else None)
    save_gens('rag-gen-stages', recs)
    GENS = load_gens()
rows, BEST = [], None
for c in STAGES + CONTROLS:
    if not (ready('base', c) and 'base' in S):
        continue
    s = scores(c)
    d = s.merge(S['base'], on='qid', suffixes=('', '_b'))
    r = ci(d.accuracy, d.accuracy_b, d.cluster)[0]
    tokens = context_tokens(GENS[(GENS.config == c) & (GENS.split == 'dev')]
                            .to_dict('records'))
    rows.append({'configuration': c,
                 'accuracy': boot(s.accuracy, cluster=s.cluster),
                 'faithful': boot(s.faithful, cluster=s.cluster),
                 'cite precision': boot(s.cite_precision, cluster=s.cluster),
                 'accuracy - base': boot(d.accuracy, d.accuracy_b, d.cluster),
                 'context tokens': f'{np.mean(list(tokens.values())):,.0f}',
                 's per answer': f'{s.latency_s.mean():.2f}',
                 'delta': r.mean, 'excludes 0': r.ci[0] > 0 or r.ci[1] < 0})
if rows:
    T = pd.DataFrame(rows)
    for cols in (['accuracy', 'faithful', 'cite precision'],  # two tables:
                 ['accuracy - base', 'context tokens', 's per answer']):
        print(T[['configuration'] + cols].to_string(index=False))  # width
    print(f'intervals excluding 0: {list(T.configuration[T["excludes 0"]])}')
if sum(r['configuration'] in STAGES for r in rows) == len(STAGES):
    stages = T[T.configuration.isin(STAGES)]  # the rule of the fix step
    BEST = stages.configuration[stages.delta.idxmax()]
    print(f'the largest dev gain: {BEST}')
```

None raised accuracy with an interval that excludes zero, at 240 questions
in 32 documents. The largest gains, from reversing the order and from the
6,000-token budget, were both +0.01, with intervals of [$-0.02$, +0.04] and
[$-0.03$, +0.05].

Two changes lowered accuracy. Fixed windows lost 0.08 [$-0.12$, $-0.04$],
although at 2,000 tokens their coverage trails by only 0.03
[$-0.06$, +0.01]. They had 1,578 context tokens against the baseline's
1,697, but the shortfall is not why they lost: the control, with 1,833
tokens, still lost 0.05 [$-0.09$, $-0.02$]. Similar coverage did not give
similar answers, and chunks that follow the document's structure gave
better ones. The 1,500-token
budget, about 1,300 context tokens against the baseline's 1,700,
cost 0.03 [$-0.06$, $-0.01$]. The 6,000-token budget, about 5,800 context
tokens, gained nothing measurable and doubled the time per answer, from
about 1.3 s to 2.6 s, partly because its longer prompts run in batches of
four instead of eight. None of this should surprise us after
:eqref:`eq_lm_rag_hit`. Retrieval already brings a relevant chunk for 213
of 220 questions, so changes on its side had about 0.02 to gain if the
accuracy on hits stayed put, less than 240 questions can detect.

Is a gain this small real? A rule fixed in advance carries the stage change
with the largest `dev` gain to `test`, read once, to find out. Three changes
lie within 0.004 of one another on `dev`, so which of them leads depends on
a handful of verdicts. `test` was read with the rewrite and reranker, which
led when it was read; on the final verdicts reversing the order leads, as
the cell prints, and `test` is not read again.

## Failures by Stage
:label:`subsec_lm_rag_failures`

If no single stage change helps on average, perhaps the 26 wrong answers of
the baseline fail in different ways, each needing its own repair. To find
out, `classify_failure` asks of each wrong answer the questions below, in
pipeline order, and blames the first stage that lost the answer (right of
:numref:`fig_lm_rag_stages`):

1. An unanswerable question was answered: a *false answer*.
2. No relevant chunk is among the first-stage top 50: *not retrieved*.
3. A relevant chunk is in the top 50 but not in the context: *ranked low*.
4. A relevant chunk is in the context and the answer abstains: a *false
   abstention*.
5. A relevant chunk is in the context and the answer, faithful to the
   passages, is incorrect: *ignored*. The generator used the passages, but
   not the one that answers.
6. A relevant chunk is in the context and the answer is not faithful:
   *unsupported*.

Each class has its own repair (:numref:`tab_lm_rag_fixes`), written into the
configuration cell before any class was counted.

:The failure classes, the stage each blames, and the fix written down for it before the failures were counted.
:label:`tab_lm_rag_fixes`

| class | stage | fix written down in advance |
|---|---|---|
| not retrieved | first-stage retrieval | add the rewritten question as a third ranking to the fusion |
| ranked low | reranking, assembly | rerank the top 100 and pass eight chunks |
| ignored | generation | quote the supporting sentences, then answer; best passages at both ends |
| unsupported | generation | drop each sentence that its cited passages do not support, as the generator judges |
| false answer | generation | abstain below a reranker score tuned on `train` |
| false abstention | generation | state the abstention rule after the passages |

Why write the fixes down first? A fix chosen after reading the failures is
fitted to them, and its gain on the same questions is optimistic, like any
choice made on test data. With the remedies fixed in advance, the data
decide only which class is largest, and the notebook takes that decision by
rule. Experimental science calls this *pre-registration*: the analysis is
written down before the data are seen. The cell counts the classes with
intervals and prints five *failure cards*, records of a failure with its
question, answer, ranking and verdicts. Claude Opus 5.5 also read each card
and labeled its class, and the cell compares these labels, again a
model's, with the rule's.

```{.python .input #rag-failures}
%%tab pytorch
CLASSES = ['not_retrieved', 'ranked_low', 'ignored', 'unsupported',
           'false_answer', 'false_abstain']

def classes(config, split='dev'):
    """The failure class (or 'correct') of each judged answer."""
    s = scores(config, split)
    corpus = RET[CONFIGS[config]['chunking']].corpus
    return s.assign(cls=[d2l.classify_failure(r, ITEMS[q], v, corpus, DEPTH)
                         for q, r, v in zip(s.qid, s.record, s.verdicts)])

def class_shares(cls):
    """Count, share and 95% interval of each class among cls (items
    resampled: a multinomial bootstrap)."""
    out = {}
    for c in CLASSES:
        x = (cls == c).to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = d2l.bootstrap_ci(x) if len(x) > 1 else None
        out[c] = (int(x.sum()), x.mean() if len(x) else np.nan,
                  *(r.ci if r else (np.nan, np.nan)))
    return pd.DataFrame(out, index=['count', 'share', 'lo', 'hi']).T

LARGEST = FIX = None
if ready('base'):
    BC = classes('base')
    fails = BC[~BC.cls.isin(['correct', 'unjudged'])]
    SH = class_shares(fails.cls)
    n_un = (BC.cls == 'unjudged').sum()
    print(f'{len(fails)} of {len(BC) - n_un} judged base answers on dev are '
          f'not correct ({n_un} unjudged, left out)')
    print(SH.assign(share=lambda d: d.apply(
        lambda r: f'{r.share:.2f} [{r.lo:.2f}, {r.hi:.2f}]', axis=1))
          [['count', 'share']].to_string())
    LARGEST = max(CLASSES, key=lambda c: SH.loc[c, 'count'])  # ties: order
    nxt = max((c for c in CLASSES if c != LARGEST),
              key=lambda c: SH.loc[c, 'count'])
    most = SH.loc[LARGEST, 'lo'] > SH.loc[nxt, 'hi']
    FIX = f'fix-{LARGEST}'
    print(f'the largest class: {LARGEST} ({"most failures: its interval "
          "lies above the next" if most else "the intervals overlap"}, '
          f'next {nxt})\nits fix, written down before counting: '
          f'{ {k: v for k, v in CONFIGS[FIX].items() if BASE[k] != v} }')
    rng = random.Random(0)
    sample = sorted(rng.sample(list(fails.qid), min(N_CARDS, len(fails))))
    F_ = fails.set_index('qid')
    cards = [dict(
        qid=q, type=ITEMS[q]['type'], question=ITEMS[q]['question'],
        reference=ITEMS[q]['reference'], key_facts=ITEMS[q]['key_facts'],
        answer=F_.loc[q, 'answer'], ranked_ids=F_.loc[q, 'record'][
            'ranked_ids'][:10], context_ids=F_.loc[q, 'record']['context_ids'],
        relevant=sorted(d2l.gold_chunk_ids(ITEMS[q], CORPUS)),
        cited_ids=F_.loc[q, 'record']['cited_ids'],
        verdicts=F_.loc[q, 'verdicts'], rule=F_.loc[q, 'cls'], labeller=None)
        for q in sample]
    for card in cards[:5]:
        hit = bool(set(card['relevant']) & set(card['context_ids']))
        print('\n' + textwrap.fill(
            f'[{card["rule"]}] {card["qid"]} ({card["type"]}): '
            f'{card["question"]}', 96, subsequent_indent='    '))
        print(textwrap.fill(f'answer: {card["answer"][:300]}', 96,
                            initial_indent='  ', subsequent_indent='    '))
        print(f'  relevant chunk in the context: {hit}\n  verdicts: '
              f'{card["verdicts"]}')
    if 'rag-cards' in RERUN:  # the agent's labels are added before upload
        d2l.lm_save_table(ARTIFACTS['rag-cards'][1], pd.DataFrame(cards),
                          on=['qid'], key=ARTIFACTS['rag-cards'][0])
    path = run_path('rag-cards')
    labelled = [c for c in (d2l.read_jsonl(path).to_dict('records')
                            if path else []) if c.get('labeller')]
    if labelled:
        rule = dict(zip(BC.qid, BC.cls))  # by the current verdicts
        agent = pd.Series({c['qid']: c['labeller']['label']
                           for c in labelled})
        now = agent.index.map(rule)
        print(f'\nthe agent ({labelled[0]["labeller"]["model"]}) against the '
              f'rule on {len(agent)} cards: agreement '
              f'{np.mean(agent.values == now.values):.2f}')
        print(pd.crosstab(pd.Series(now, name='rule'),
                          pd.Series(agent.values, name='agent')))
    else:
        print('\nthe cards carry no labeller labels yet')
```

Of the 26 failures, 12 are ignored, 6 unsupported, 5 ranked low, 2 false
answers and 1 a false abstention; none failed at first-stage retrieval.
Ignored is the largest class, 0.46 [0.27, 0.65] of the failures, but its
interval overlaps that of unsupported, 0.23 [0.08, 0.38], so 26 failures
cannot tell which of the two is larger.

The 31 cards cover all 26 failures and five answers that the judge counts
as correct after its re-judging; the labeler calls those five correct
too, the first row of the table. On the 26 failures, labeler and rule
agree on 13. Of the 13 disagreements, 9 are answers that the labeler calls
correct and our grading does not. In seven the judge insists on a key fact
the labeler lets pass. In two, unanswerable questions were declined in
words other than the abstention sentence, which the string rule counts as
answers. Only four disagreements concern the stage. Rule and labeler disagree about whether an answer
is wrong far more often than about where it went wrong.

The rule picks the ignored class and its fix: the generator quotes the
sentences that support its answer, with their passage numbers, and then
answers, with the best passages at both ends and room for 500 new tokens.
The cell applies it to `dev` and then reads `test` once.

```{.python .input #rag-fix}
%%tab pytorch
def tune_threshold():
    """The reranker score below which the false-answer fix abstains: the
    cut that best separates train unanswerable items from as many train
    answerable ones (balanced accuracy), by the best reranker score of
    each base context."""
    un = [q for q in QA['train'] if not q['answerable']]
    an = d2l.BookQA([q for q in QA['train'] if q['answerable']],
                    'train').subset(len(un), seed=0)
    pipe = d2l.RAGPipeline(RET['sections'], None, None, k=K,  # retrieval
                           budget_tokens=BUDGET, depth=DEPTH,
                           abstain_below=0.0)
    s = np.array([p['top_score'] for p in pipe.prepare(un + list(an))])
    y = np.r_[np.ones(len(un)), np.zeros(len(an))]
    cuts = np.unique(s)
    bal = [((s[y == 1] < t).mean() + (s[y == 0] >= t).mean()) / 2
           for t in cuts]
    print(f'abstain below {cuts[np.argmax(bal)]:.3f} (train balanced '
          f'accuracy {max(bal):.2f})')
    return float(cuts[np.argmax(bal)])

if 'rag-gen-fix' in RERUN:
    assert FIX, 'the fix needs the judged base run (rag-judge)'
    assert BEST, 'test is read once, with the best stage: judge every stage'
    t = time.time()
    thr = tune_threshold() if LARGEST == 'false_answer' else None
    recs = run_config(FIX, DEV, threshold=thr)
    base = run_config('base', TEST)  # the only reads of test in the chapter
    recs += base
    for name in [FIX] + ([BEST] if CONFIGS[BEST] != CONFIGS[FIX] else []):
        recs += run_config(name, TEST, budgets=context_tokens(base) if
                           CONFIGS[name]['budget'] == 'base' else None,
                           threshold=thr if name == FIX else None)
    rule = json.dumps(dict(largest=LARGEST, fix=FIX, best=BEST,
                           threshold=thr))  # what test was read with
    save_gens('rag-gen-fix', [dict(r, rule=rule) for r in recs])
    GENS = load_gens()
    print(f'rag-gen-fix regenerated in {(time.time() - t) / 60:.1f} min, '
          f'{d2l.lm_peak()}')
rules = set(GENS.rule.dropna()) if 'rule' in GENS else set()
if rules:
    RULE = json.loads(rules.pop())
    print(f'test was read with the fix {RULE["fix"]} and the stage '
          f'{RULE["best"]}' + ('' if RULE['threshold'] is None else
                              f', abstaining below a reranker score of '
                              f'{RULE["threshold"]:.3f} (tuned on train)'))
    assert not rules and (RERUN or RULE['fix'] == FIX), (
        'the rule now picks another fix')
    if BEST != RULE['best']:  # stage deltas within noise: a tie can move
        print(f'with the verdicts as they stand now the largest dev gain is '
              f'{BEST};\n  test was read once, with {RULE["best"]}, and is '
              'not read again')
    BEST = RULE['best']  # the stage that test was read with

if FIX and ready('base', FIX):
    rows = []
    for split in ('dev', 'test'):
        for c in [FIX] + ([BEST] if BEST and BEST != FIX else []):
            if not ready('base', c, split=split):
                continue
            a, b = scores(c, split), scores('base', split)
            rows.append({'split': split, 'configuration': c,
                         'accuracy - base': paired(a, b, 'accuracy'),
                         'cite precision - base': paired(
                             a, b, 'cite_precision')})
    print(pd.DataFrame(rows).to_string(index=False))
    after = classes(FIX)
    print(pd.DataFrame({'base': BC.cls.value_counts(),
                        FIX: after.cls.value_counts()})
          .reindex(CLASSES + ['correct', 'unjudged']).fillna(0).astype(int)
          .to_string())
```

The fix failed. It lowered accuracy by 0.08 [$-0.12$, $-0.05$] on `dev` and
by 0.08 [$-0.12$, $-0.03$] on `test`. On `dev` it got 20 fewer answers
right. Ignored answers, the class it aimed at, did not become rarer but rose
from 12 to 19. False answers rose from 2 to 15: on the unanswerable
questions the abstentions fell from 18 of 20 to 5. In most of the other 15
the model said in its own words that the passages do not give the answer,
instead of writing the abstention sentence, and the rest answered anyway.

Is that a failure of the model or of our grading? The prompt asks for the
abstention sentence word for word, and any system that acts on it, by
showing a fallback for instance, breaks when the sentence is missing. The
quoting format broke that agreement between prompt and grader, so a fix's
prompt and the scoring rule have to be tested together. Even without the
false answers, though, the fix did not shrink the class it aimed at. A
change aimed at one class created another.

The `dev` result could have been optimistic, since the fix was chosen from
`dev` failures; it lost there anyway, and `test` agrees. Nor do we now
search for a better fix: one found by looking at these results would need a
fresh test set to be believed. The stage change, too, does nothing on
`test`: $-0.004$, with an interval from $-0.05$ to +0.04.

The plot shows the same results at a glance. On the left are the shares of
`dev` questions in each failure class before and after the fix. On the
right are the accuracy and citation precision on `dev` of the closed book,
the baseline, the stage change and the fix, with the `test` points as
crosses. The last two each change one thing in the baseline; they are not
steps of a chain.

```{.python .input #rag-failure-plot}
%%tab pytorch
if FIX and ready('closed', 'base', FIX):
    d2l.use_svg_display()
    _, axes = d2l.plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    series = {}
    for name, df in (('base', BC), ('fix', after)):
        sh = class_shares(df.cls[df.cls != 'unjudged'])  # judged only
        series[name] = (sh.share, sh.lo, sh.hi)
    d2l.plot_intervals(CLASSES, series, 'failure class',
                       'share of dev questions', axes=axes[0])
    steps = ['closed', 'base'] + ([BEST] if BEST and BEST != FIX else []
                                  ) + [FIX]
    dev = {c: scores(c) for c in steps}
    test = {c: scores(c, 'test') for c in steps[1:] if ready(c, split='test')}
    waterfall, marks = {}, {'test': ('x', {})}
    for m, label in (('accuracy', 'accuracy'),
                     ('cite_precision', 'citation precision')):
        est = [ci(dev[c][m], cluster=dev[c].cluster)[0] for c in steps]
        waterfall[label] = tuple(  # closed book cites nothing: no point
            [np.nan if r is None else f(r) for r in est]
            for f in (lambda r: r.mean, lambda r: r.ci[0],
                      lambda r: r.ci[1]))
        xs = [i for i, c in enumerate(steps) if c in test]
        marks['test'][1][label] = (xs, [test[steps[i]][m].mean()
                                        for i in xs])
    d2l.plot_intervals(steps, waterfall, 'configuration', 'dev score',
                       axes=axes[1], marks=marks)
```

Beyond these fixes lies a large literature that this section does not
measure, from transforming the query and retrieving again during generation
to retrieval learned with reinforcement learning, compressed contexts and
structured indexes (:numref:`sec_lm_encoders_fr`). Whichever we try, the
procedure of this section decides whether it helps: one change at a time, a
paired interval, and a test set read once.

## Marginalizing over Retrieved Documents
:label:`subsec_lm_rag_marginal`

Our largest failure class lives in the generator: it is handed the right
passage and does not use it. Could the generator teach the retriever which
passages it can use? Our pipeline treats retrieval as preprocessing:
passages are chosen, concatenated and read. The model that gave RAG its
name :cite:`Lewis.Perez.Piktus.ea.2020` treats the retrieved document as a
hidden variable instead, which makes the system a probabilistic model whose
query encoder and generator can be trained together. Let $x$ be
the question, $y = (y_1, \ldots, y_n)$ the answer and $z$ a document. The
retriever gives each document a score $s_z$ and the probability
$p(z \mid x) = \mathrm{softmax}(\mathbf{s})_z$, and the generator reads one
document at a time. *RAG-Sequence* draws one document for the whole answer:

$$p_S(y \mid x) = \sum_z p(z \mid x) \prod_{t=1}^{n} p(y_t \mid x, z, y_{<t}).$$
:eqlabel:`eq_lm_rag_seq`

*RAG-Token* may draw a new document for every token:

$$p_T(y \mid x) = \prod_{t=1}^{n} \sum_z p(z \mid x)\, p(y_t \mid x, z, y_{<t}).$$
:eqlabel:`eq_lm_rag_token`

RAG-Sequence asks which single document explains the answer; RAG-Token lets
each token be explained by whichever document supports it. For one document
the two coincide. The smallest case in which they differ has two documents
and two tokens. Let $p(z \mid x) = (0.6, 0.4)$, let document 1 give the two
tokens the probabilities 0.9 and 0.2, and document 2 give 0.1 and 0.8. Then

$$
\begin{aligned}
p_S &= 0.6 \cdot 0.9 \cdot 0.2 + 0.4 \cdot 0.1 \cdot 0.8 = 0.140,\\
p_T &= (0.6 \cdot 0.9 + 0.4 \cdot 0.1)(0.6 \cdot 0.2 + 0.4 \cdot 0.8) = 0.2552.
\end{aligned}
$$

Each document supports half of the answer, and only RAG-Token combines the
halves: the multi-passage question in its smallest form. Our pipeline
combines passages too, through attention over their concatenation, with one
long prompt instead of one short pass per document.

The sum over the corpus is too large, so it runs over the $k$ documents of
largest $p(z \mid x)$, the retrieved set $\mathcal{K}$. What does that drop? Write
$L_z = p(y \mid x, z)$, $p_S^{(k)}$ for the sum restricted to $\mathcal{K}$, and
$m_k = \sum_{z \in \mathcal{K}} p(z \mid x)$ for the retriever's mass on $\mathcal{K}$. The
dropped terms are nonnegative and each $L_z$ is at most $\max_z L_z$, so

$$0 \le p_S - p_S^{(k)} = \sum_{z \notin \mathcal{K}} p(z \mid x)\, L_z \le (1 - m_k) \max_z L_z.$$
:eqlabel:`eq_lm_rag_topk`

The truncation error is at most the probability mass that the retriever
puts outside the retrieved set, since no $L_z$ exceeds 1. A retriever with
peaked probabilities loses little; one that spreads its mass can lose much,
however good its ranking.

How does the generator teach the retriever? We differentiate the logarithm
of :eqref:`eq_lm_rag_seq` with respect to a score.

**Proposition.** Let $p(z \mid x, y) = p(z \mid x) L_z / p_S(y \mid x)$ be
the posterior over documents once the answer is known. Then

$$\frac{\partial \log p_S(y \mid x)}{\partial s_z} = p(z \mid x, y) - p(z \mid x).$$
:eqlabel:`eq_lm_rag_grad`

**Proof.** Write $\log p_S = \log \sum_{z'} e^{s_{z'}} L_{z'} - \log
\sum_{z'} e^{s_{z'}}$. The derivative of the first term with respect to
$s_z$ is $e^{s_z} L_z / \sum_{z'} e^{s_{z'}} L_{z'}$, the posterior; that of
the second is $e^{s_z} / \sum_{z'} e^{s_{z'}}$, the prior. $\blacksquare$

The retriever raises the score of a document that explains the answer
better than it expected and lowers the others, with no label saying which
document was relevant. If the generator ignores the documents, $L_z$ is the
same for all, the posterior equals the prior, and the retriever learns
nothing. That qualifies the answer to our question: the retriever's signal
is only as good as the generator's use of the passages, and a passage the
generator ignores teaches the retriever nothing about it.

The form is general. Take any marginal likelihood over a latent choice $z$,
and differentiate its logarithm with respect to the parameters of
$p(z \mid x)$. The result is an expectation under the posterior,
$\nabla \log p(y \mid x) = \mathbb{E}_{p(z \mid x, y)}[\nabla \log p(z \mid x)]$,
the same posterior that the E-step of :numref:`sec_mdl-latent-em-elbo`
computes. For a softmax prior it is posterior minus prior. The cell checks
the gradient against autograd, the bound for every $k$, the two
marginalizations for one and three documents, and the case by hand.

```{.python .input #rag-marginal}
%%tab pytorch
g = torch.Generator().manual_seed(0)
s = torch.randn(8, dtype=torch.float64, generator=g, requires_grad=True)
L = torch.rand(8, dtype=torch.float64, generator=g)  # p(y | x, z)
prior = s.softmax(0)  # the retriever's p(z | x)
torch.log((prior * L).sum()).backward()
post = (prior * L / (prior * L).sum()).detach()
err = (s.grad - (post - prior.detach())).abs().max().item()
print(f'gradient of log p(y | x) - (posterior - prior): {err:.1e}')
assert err < 1e-12
p, gaps = (prior * L).sum().item(), []
order = prior.argsort(descending=True)
for k in range(1, 9):  # p_S^(k): the k documents of largest prior
    top = order[:k]
    gap = p - (prior[top] * L[top]).sum().item()
    bound = (1 - prior[top].sum().item()) * L.max().item()
    assert -1e-15 <= gap <= bound + 1e-15
    gaps.append((k, gap, bound))
print(pd.DataFrame(gaps, columns=['k', 'p - p_k', 'bound']).round(4)
      .to_string(index=False))
s2 = s.detach().clone().requires_grad_()
torch.log((s2.softmax(0) * 0.3).sum()).backward()  # a constant likelihood
assert s2.grad.abs().max() < 1e-15

def p_seq_tok(pz, P):
    """RAG-Sequence and RAG-Token probabilities of an answer: pz the
    retriever's probabilities (D,), P the token probabilities (D, T)."""
    return (pz * P.prod(1)).sum().item(), (pz @ P).prod().item()

P3 = torch.rand(3, 3, dtype=torch.float64, generator=g)
pz3 = torch.rand(3, dtype=torch.float64, generator=g).softmax(0)
seq, tok_ = p_seq_tok(pz3, P3)
one = p_seq_tok(torch.ones(1, dtype=torch.float64), P3[:1])
assert abs(seq - tok_) > 1e-6 and abs(one[0] - one[1]) < 1e-15
print(f'three documents, three tokens: p_S {seq:.4f}, p_T {tok_:.4f}; '
      f'one document: both {one[0]:.4f}')
pz = torch.tensor([0.6, 0.4], dtype=torch.float64)  # the case by hand
P = torch.tensor([[0.9, 0.2], [0.1, 0.8]], dtype=torch.float64)
print('two documents, two tokens:\n  p_S = 0.6 * 0.9 * 0.2 + 0.4 * 0.1 * '
      '0.8 = {:.3f}\n  p_T = (0.6 * 0.9 + 0.4 * 0.1)(0.6 * 0.2 + 0.4 * 0.8) '
      '= {:.4f}'.format(*p_seq_tok(pz, P)))
```

The gradient matches posterior minus prior to $6 \times 10^{-17}$, every
truncation error lies below its bound (at $k = 5$, 0.028 against 0.037),
and the case by hand gives 0.140 and 0.2552.

Our pipeline concatenates instead of marginalizing, and every passage adds
prompt tokens that the generator must read. What does that cost? The last
cell reports the tokens and seconds per answer of each configuration and
times retrieval and reranking, one question at a time.

```{.python .input #rag-cost}
%%tab pytorch
print(GENS[GENS.split == 'dev'].groupby('config', sort=False).agg(
    prompt_tokens=('n_prompt_tokens', 'mean'),
    new_tokens=('n_new_tokens', 'mean'),
    s_per_answer=('latency_s', 'mean')).round(2).to_string())
R = RET['sections']
qs = [q['question'] for q in DEV.subset(50 if GPU else 3, seed=1)]
depth = DEPTH if GPU else 5  # the CPU path reranks five chunks of one
R.search_batch(qs[:1], k=DEPTH)  # load the query encoder
t = time.time()
hits = [R.search(q, k=DEPTH) for q in qs]
t_ret = (time.time() - t) / len(qs)
pipe = d2l.RAGPipeline(R, None, None, rerank_k=depth)  # no generation
pipe._rerank(qs[:1], [[h.chunk_id for h in hits[0][:2]]])  # load it
rr = qs if GPU else qs[:1]
t = time.time()
pipe._rerank(rr, [[h.chunk_id for h in hs[:depth]] for hs in hits])
t_rr = (time.time() - t) / len(rr)
R.release()
print(f'one question at a time: retrieval {1000 * t_ret:.0f} ms '
      f'({len(qs)} questions), reranking {depth} chunks {1000 * t_rr:.0f} '
      f'ms ({len(rr)} questions)')
```

First-stage retrieval is cheap, about 30 ms per question. Reranking 50
chunks is not: about half a second, some 40% of the time of a baseline
answer. Generation takes the rest and grows with the prompt, from about
1.1 s per answer at 1,500 prompt tokens and 1.3 s at the baseline's 1,900 to
2.6 s at 6,300 tokens in batches of four.

That leaves the question raised by longer context windows. If the generator
can read 6,000 tokens, why not whole chapters, and no retrieval? On this set
a budget of 6,000 tokens, 3.4 times the baseline's context, gained nothing
measurable at twice the time, and the oracle shows that the generator,
given the right passage, still misses about one question in twelve. The
agents of :numref:`chap_lm_agents` take a third path, retrieving when they
decide that they need to. Whether whole chapters in the context would beat
retrieval on the multi-passage questions, at equal cost, we leave open; it
is an experiment for the exercises.

## Summary

When a retrieval-augmented generator answers wrongly, one accuracy number
cannot say which of its stages (query, retrieval, reranking, context
assembly, generation) failed, and each stage needs a different repair. We
located failures by measuring each stage on its own. That took a question
set over the book with gold spans, drafted and checked by models rather
than people, a judge checked against model labels before any accuracy was
read, coverage at a token budget, paired intervals, and failure classes
whose fixes were written down in advance. On this book retrieval is nearly
solved and the generator limits accuracy. The baseline answers 0.89 of the
`dev` questions against 0.30 without passages, and no single stage change
helped. The fix for the largest class, passages ignored, lowered accuracy
by 0.08 on `dev` and on `test`, mostly by creating false answers. The
original RAG model
marginalizes over retrieved documents instead of concatenating them, and
its retriever learns from the generator through the posterior minus the
prior. Whether whole documents in a long context can replace retrieval
remains open.

## Exercises

1. **The truncation error and recall.** Start from :eqref:`eq_lm_rag_topk`.
    1. Show that the bound is tight: give scores and likelihoods for which
       $p_S - p_S^{(k)} = (1 - m_k) \max_z L_z$.
    1. Suppose exactly one document $z^*$ answers each question, and the
       retriever is calibrated: $p(z \mid x)$ is the probability that $z$ is
       $z^*$. Show that the expectation of $1 - m_k$ over questions equals
       the probability that $z^*$ is not in the top $k$, one minus
       Recall@$k$.
    1. With Recall@5 = 0.97 from this section, what would the bound
       promise for RAG-Sequence over five documents if the retriever were
       calibrated? Why is a softmax over BM25 or dense scores rarely
       calibrated in this sense, and how would you calibrate it on `train`?
   :exlabel:`ex_lm_rag-topk`

1. [code] **The position of the answering passage.** Take the answerable
   `dev` questions whose baseline context holds exactly one relevant chunk.
   Build contexts of five chunks with the relevant chunk at positions 1 to 5
   and the other four in rank order.
    1. Predict how accuracy depends on the position, then measure it, with
       paired intervals against position 1. Judging needs the judge server
       on two 24 GB GPUs. With one GPU, measure instead how often the answer
       cites the relevant passage at each position, which needs no judge.
    1. Does the result show the dip in the middle that
       :citet:`Liu.Lin.Hewitt.ea.2024` found in long contexts? Build longer
       contexts from more chunks of the book until it does, or until 6,000
       tokens, and report the length at which the middle positions fall
       behind, if any.
   :exlabel:`ex_lm_rag-position`

1. [code] **Citation precision against support.** Citation precision from
   gold spans counts every cited passage that is not gold as a mistake.
    1. Collect the baseline answers that cite at least one passage outside
       the gold chunks. Read 20 of them with their passages and count the
       citations that do support their sentence.
    1. Compare citation precision with the judge's `supported` verdict per
       answer. On how many answers do they disagree, and on your 20, which
       of the two is right more often?
   :exlabel:`ex_lm_rag-citations`

1. [extended] **Long context against retrieval.** ●● Put whole chapters into
   the prompt instead of five retrieved chunks.
    1. Budget the memory first. From the configuration of Qwen3.5-9B
       (8 full-attention layers of 32, 4 key-value heads of width 256,
       bf16), compute the bytes of cache per token and the cache at 32,000
       tokens, and check with `d2l.config_memory` whether such a prompt fits
       beside the weights on a 24 GB card. Compare with Qwen3-4B, whose 36
       layers all attend.
    1. For the 20 `multihop` `dev` questions, build contexts of about
       32,000 tokens from the documents of their gold spans, filled up with
       other documents of the book. Compare accuracy and seconds per answer
       with the baseline, with paired intervals.
    1. Which wins, at what cost, and can 20 questions in this many
       documents decide it?
   :exlabel:`ex_lm_rag-long-context`

1. **Two marginalizations.** Use :eqref:`eq_lm_rag_seq`
   and :eqref:`eq_lm_rag_token`.
    1. Show that RAG-Sequence and RAG-Token give the same probability for an
       answer of one token, whatever the number of documents.
    1. Is $p_T \ge p_S$ always? Prove it, or give a counterexample with two
       documents and two tokens.
    1. Which kind of answer does each marginalization favor, and which of
       them can a greedy decoder evaluate one token at a time?
   :exlabel:`ex_lm_rag-marginals`

1. **Where to invest.** A colleague's pipeline puts a relevant chunk in the
   context for 90% of the questions, and its accuracy is 0.7 with one and
   0.2 without.
    1. Use :eqref:`eq_lm_rag_hit` to compute its accuracy, and its accuracy
       with an oracle context, assuming that the accuracy given a hit stays
       at 0.7. Which gains more: a perfect retriever, or raising the
       accuracy given a hit to 0.8?
    1. Why is "the accuracy given a hit stays at 0.7" an assumption? Which
       way does the oracle arm of this section suggest it errs, and why?
    1. Suppose two configurations disagree on 10% of 240 questions. Estimate
       the standard error of the paired difference in accuracy, and the
       smallest gain a 95% interval would separate from zero. Which of the
       two improvements could 240 questions detect?
   :exlabel:`ex_lm_rag-invest`

1. [code] **A smaller generator.** Replace Qwen3.5-9B by Qwen3.5-4B of the
   same family and keep everything else fixed.
    1. Before you run it, predict which failure class grows most and whether
       the gain of the oracle context over the baseline becomes measurable.
    1. Regenerate the `base`, `closed` and `oracle` arms, judge them, and
       report the class shares with their intervals and the paired
       differences with the larger generator. Without two GPUs for the
       judge, compare what needs no judge: abstention rates, false
       abstentions and citation precision.
   :exlabel:`ex_lm_rag-smaller`

1. [code] **Hand labels against the verifier.** Draw 20 `dev` items and 10
   candidates that the frontier verifier rejected (from the recorded
   responses of `lm-book-qa-api-cache-v1`), with seed 0, and hide which is
   which.
    1. For each, read the question, the reference and the gold span, and
       decide yourself whether it stands alone, whether the reference is
       supported and complete, and whether the answer is unique.
    1. Compute your agreement with the verifier and Cohen's κ, with a
       bootstrap interval. What can 30 items show, and what can they not?
    1. Read the items on which you disagree. Are they errors of the model,
       or of yours?
   :exlabel:`ex_lm_rag-hand-labels`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §19.5]{.kicker}

Retrieval-Augmented Generation<br>**a question set over the book · stages and their measurements · a checked judge · one change at a time · failures by stage · marginalizing over documents**
:::
:::

::: {.slide title="The Answering Passage Was First, and the Answer Still Missed Its Fact"}
[The problem]{.kicker}

**Question:** in PyTorch, what happens to modules kept in a plain Python
list instead of `nn.Sequential`?

- Passage [1], ranked first: "... Modules in a plain list stay
  unregistered, without any warning."
- Answer: four sentences on registration, citing [2] and [3]. Nothing
  about the warning.
- Judge: not correct, faithful, supported.

::: {.d2l-note .warn}
The baseline answers 0.89 of 240 questions. Ranked too low? Ignored?
Contradicted? One accuracy number cannot tell.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- build a question set with gold passages and no leakage between splits;
- measure each stage: recall, coverage at a token budget, accuracy,
  faithfulness, citations;
- check a model judge before trusting its accuracy;
- blame each failure on a stage, and test a fix chosen in advance;
- derive RAG-Sequence, RAG-Token and the retriever's gradient.

**Running example:** this book in 5,324 section chunks; 240 `dev`
questions about 32 documents; Qwen3.5-9B answers, Qwen3.8-27B judges.
:::

::: {.slide title="A Retriever Finds Passages; This Deck Measures What Is Done with Them"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| movement | section |
|---|---|
| vectors | Word Embeddings |
| | Pretrained Encoders |
| | Text Embeddings from Contrastive Training |
| finding | Sparse, Dense and Approximate Retrieval |
| using | **Retrieval-Augmented Generation** |
:::

::: {.col .narrow}
- **Uses:** BM25 + dense search fused by reciprocal ranks, a reranker,
  paired bootstrap intervals.
- **Skips:** running joint training of retriever and generator (derived,
  not run); retrieval inside agent loops.
:::
:::
:::

::: {.slide title="A Question Set over Your Own Corpus Comes with Gold Passages"}
[The question set]{.kicker}

| type | single | explain | numeric | code | multihop | unanswerable |
|---|---|---|---|---|---|---|
| `dev`, `test` each | 120 | 40 | 20 | 20 | 20 | 20 |

- Item: question, reference, **key facts**, **gold spans** (character
  ranges), alternative spans, flags.
- Split by **document**: 166 `train`, 32 `dev`, 32 `test`. No `dev`
  passage sources a `train` question.
- A chunk is relevant if it overlaps half a span, or half of itself.

::: {.d2l-note}
Spans, not chunk ids: relevance is recomputed for any chunking.
:::
:::

::: {.slide title="Models Drafted, Checked and Verified Every Question, with Rules Between"}
[How the set was made]{.kicker}

| step | by |
|---|---|
| draft two questions per passage | DeepSeek-V4.1-Flash |
| verbatim evidence, no "the passage above", no copied words, no near duplicates | rules |
| stands alone? supported? complete? unique? | Qwen3.8-27B |
| verify every `dev` and `test` item | Claude Opus 5.5 |

Verifier: 620 calls, about \$7; the whole set about \$15.

::: {.d2l-note .warn}
Blind spots the models share pass every check. The verifier may have read
the book. The checker is also the judge.
:::
:::

::: {.slide title="Check: Verifier and Checker Agree on 84% of Items. Is That Agreement?"}
[Check]{.kicker}

Both pass about 9 items in 10. Cohen's kappa corrects observed agreement
$p_o$ for chance agreement $p_e$ at the raters' own pass rates:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $p_e \approx 0.9^2 + 0.1^2 = 0.82$, so
$\kappa \approx (0.84 - 0.82)/0.18 \approx 0.1$ (measured: 0.12). Barely
above chance: the few items each rejects are mostly different ones.
:::
:::
:::

::: {.slide title="Five Stages, Each with Its Own Measurement"}
[The pipeline]{.kicker}

| stage | decides | measured by |
|---|---|---|
| query | question or rewrite | the stages after it |
| retrieval | BM25 + dense, fused | Recall@$k$ |
| reranking | order of the top candidates | Recall@$k$ |
| assembly | chunks, order, token budget | coverage `cov@T` |
| generation | cited answer or abstention | accuracy, faithfulness, citations |

::: {.d2l-note}
`cov@T`: does the context, filled in rank order up to $T$ tokens, hold half
of a gold span? Equal text, not equal chunk counts.
:::
:::

::: {.slide title="One Fixed Prompt; Every Stage Change Alters Only the Context or the Query"}
[Grounded generation]{.kicker}

> **System:** answer about *Dive into Deep Learning* using only the
> numbered passages; cite after each statement, e.g. [2]; else reply
> exactly: *I cannot answer this from the provided passages.*
>
> **User:** Passages: [1] heading path, text ... Question: ... Answer in at
> most four sentences.

- Qwen3.5-9B: thinking off, greedy, bf16, batches of 8, left padding.
- Closed book: the same, without passages.
- Unanswerable question: the abstention sentence is correct, any other
  reply is a false answer (string match, no judge).
:::

::: {.slide title="The Judge Agrees with Each Frontier Labeler as They Agree with Each Other"}
[A judge, checked first]{.kicker}

Qwen3.8-27B, thinking on, FP8 on two 24 GB GPUs. Verdicts: **correct**
(all key facts), **faithful** (passages support it), **supported** (cited
passages suffice).

| `correct`, 100 calibration answers | κ |
|---|---|
| judge vs frontier labeler | 0.83 |
| judge vs second labeler | 0.71 |
| the two labelers | 0.81 |
| judge vs their consensus | 0.85 [0.70, 0.97] |
| old judge, 4B, thinking off, vs consensus | 0.33 |

::: {.d2l-note .warn}
Labels by models, on answers of an earlier generator (Qwen3-4B).
:::
:::

::: {.slide title="Retrieval Lifts Accuracy from 0.30 to 0.89; an Oracle Adds Nothing Measurable"}
[Floor, baseline, ceiling]{.kicker}

| 240 `dev` questions | closed book | baseline | oracle |
|---|---|---|---|
| accuracy | 0.30 | 0.89 | 0.92 |
| faithful | | 0.92 | 0.89 |
| citation precision | | 0.72 | 1.00 |
| abstains, unanswerable | 10 of 20 | 18 of 20 | 17 of 20 |

- Baseline − closed: +0.59 [+0.53, +0.66]; book-specific: 0.12 → 0.90.
- Oracle − baseline: +0.03 [$-0.02$, +0.08].
:::

::: {.slide title="Check: 213 of 220 Hits. What Can a Perfect Retriever Add?"}
[Check]{.kicker}

$$\textrm{accuracy} = h\, a_1 + (1 - h)\, a_0$$

$h = 213/220$: share of answerable questions whose context holds a relevant
chunk (a *hit*); accuracy $a_1 = 0.91$ on hits, $a_0 = 0.29$ on the seven
misses.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** If $a_1$ stays put, $(1 - h)(a_1 - a_0) \approx 0.032 \times
0.63 \approx 0.02$. The oracle measures +0.04 [$-0.01$, +0.09]: it also
drops distractors. 19 of 24 errors had the answer in the context.
:::
:::
:::

::: {.slide title="Same Prompts in Other Batches: 45% of Answers Change, Accuracy Does Not"}
[Batch drift]{.kicker}

- `base` vs `base-repeat`: 0 of 240 contexts differ; 108 of 240 answers
  differ.
- Accuracy: $-0.004$, interval [$-0.03$, +0.02].
- bf16: padding moves logits by 0.2 to 0.6, a few rounding steps, and
  flips near-ties. fp32 (2B model, same family): at most $10^{-4}$.
- The judge's server batches too: 5 of 54 verdicts changed.

::: {.d2l-note .rule}
Greedy decoding is deterministic only for a fixed batch and precision.
Batch alike, or count re-batching as noise.
:::
:::

::: {.slide title="Section Chunks Cover More Than Fixed Windows at Small Budgets"}
[Chunking at equal tokens]{.kicker}

| `cov@T`, 220 questions | sections | fixed 256/64 | fixed − sections |
|---|---|---|---|
| 1,000 tokens | 0.91 | 0.84 | $-0.07$ [$-0.11$, $-0.02$] |
| 2,000 tokens | 0.97 | 0.94 | $-0.03$ [$-0.06$, +0.01] |
| 3,000 tokens | 0.98 | 0.97 | $-0.01$ [$-0.04$, +0.02] |

- Fixed windows: 256 tokens, overlap 64. Sections: at most 512, cut at
  headings.
- Recall@$k$ would mislead: five sections hold more text than five
  windows.
:::

::: {.slide title="No Stage Change Raised Accuracy; Two Lowered It"}
[One change at a time]{.kicker}

| change, 240 `dev` questions | accuracy − baseline | s per answer |
|---|---|---|
| fixed windows, 1,578 tokens (baseline 1,697) | $-0.08$ [$-0.12$, $-0.04$] | 1.11 |
| rewrite + rerank top 50 | +0.00 [$-0.03$, +0.04] | 1.87 |
| reverse order | +0.01 [$-0.02$, +0.04] | 1.14 |
| budget 1,500, no limit of 5 chunks | $-0.03$ [$-0.06$, $-0.01$] | 1.08 |
| budget 6,000, no limit of 5 chunks | +0.01 [$-0.03$, +0.05] | 2.59 |
| control: fixed windows, 1,833 tokens | $-0.05$ [$-0.09$, $-0.02$] | 1.15 |

::: {.d2l-note}
With 213 of 220 hits, retrieval-side changes had about 0.02 to gain.
:::
:::

::: {.slide title="Six Failure Classes, Checked in Pipeline Order"}
[Failures by stage]{.kicker}

| class | test (first that holds) | of 26 |
|---|---|---|
| false answer | unanswerable, yet answered | 2 |
| not retrieved | no relevant chunk in the top 50 | 0 |
| ranked low | in the top 50, not in the context | 5 |
| false abstention | in the context, answer abstains | 1 |
| ignored | in the context, faithful, incorrect | 12 |
| unsupported | in the context, not faithful | 6 |

Ignored 0.46 [0.27, 0.65] and unsupported 0.23 [0.08, 0.38]: the intervals
overlap.
:::

::: {.slide title="Fixes Are Written Down Before Any Failure Is Counted"}
[Pre-registration]{.kicker}

| class | its fix |
|---|---|
| not retrieved | rewritten query as a third ranking |
| ranked low | rerank the top 100, pass 8 chunks |
| ignored | quote the support, then answer; best at both ends |
| unsupported | drop sentences their citations do not support |
| false answer | abstain below a reranker score tuned on `train` |
| false abstention | abstention rule after the passages |

A model labeler agrees with the rule on 13 of the 26 failures; 9 of the
13 disagreements are about *whether* an answer is wrong.
:::

::: {.slide title="The Fix Chosen in Advance Failed on Dev and on Test"}
[Quote, then answer]{.kicker}

| | accuracy − baseline |
|---|---|
| `dev` | $-0.08$ [$-0.12$, $-0.05$] |
| `test`, read once | $-0.08$ [$-0.12$, $-0.03$] |

- Ignored: 12 → 19. False answers: 2 → 15.
- Abstentions on unanswerable questions: 18 → 5 of 20. Most replies
  declined in their own words, without the abstention sentence.

::: {.d2l-note .rule}
A change aimed at one class can create another. Test a fix's prompt and
the scoring rule together, and do not search for a better fix on the same
data.
:::
:::

::: {.slide title="RAG-Sequence Picks One Document per Answer; RAG-Token One per Token"}
[Two marginalizations]{.kicker}

Question $x$, answer $y_1 \ldots y_n$, document $z$ with retriever
probability $p(z \mid x)$:

$$p_S = \sum_z p(z \mid x) \prod_t p(y_t \mid x, z, y_{<t})$$

$$p_T = \prod_t \sum_z p(z \mid x)\, p(y_t \mid x, z, y_{<t})$$

Two documents, $p(z \mid x) = (0.6, 0.4)$; token probabilities
$(0.9, 0.2)$ and $(0.1, 0.8)$. Which is larger?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $p_S = 0.140$, $p_T = 0.2552$: each document supports half the
answer, and only RAG-Token combines the halves.
:::
:::
:::

::: {.slide title="The Retriever Learns Posterior Minus Prior; Truncation Loses the Mass Outside"}
[Training signal]{.kicker}

With $L_z = p(y \mid x, z)$ and $p(z \mid x) = \mathrm{softmax}(\mathbf{s})_z$:

$$\frac{\partial \log p_S}{\partial s_z} = \underbrace{\frac{p(z \mid x) L_z}{p_S}}_{\text{posterior}} - \underbrace{p(z \mid x)}_{\text{prior}}$$

Keep the top $k$, with retriever mass $m_k$ on them:

$$0 \le p_S - p_S^{(k)} \le (1 - m_k) \max_z L_z$$

::: {.d2l-note}
A generator that ignores documents ($L_z$ constant) teaches the retriever
nothing.
:::
:::

::: {.slide title="First-Stage Retrieval Costs Milliseconds; Reranking and Long Prompts Cost Seconds"}
[Cost]{.kicker}

| per question | prompt tokens | seconds |
|---|---|---|
| retrieval | | 0.03 |
| rerank 50 chunks | | 0.53 |
| closed book | 90 | 0.95 |
| baseline | 1,933 | 1.28 |
| budget 6,000 | 6,336 | 2.59 |
| fix: quote, then answer | 1,971 | 2.12 |

Qwen3.5-9B on one RTX 4090, batches of 8 (4 at 6,000 tokens).
:::

::: {.slide title="Recap"}
[Recap]{.kicker}

- 240 `dev` questions with gold spans, made and checked by models.
- Judge ≈ a third frontier labeler (κ 0.71 to 0.83 vs their 0.81).
- Closed 0.30, baseline 0.89, oracle 0.92: the generator limits accuracy.
- No stage change helped; fixed windows $-0.08$, budget 1,500 $-0.03$.
- Re-batching changed 45% of answers in bf16, not accuracy.
- Largest class ignored; its fix lost 0.08 on `dev` and on `test`.
- $p_S$, $p_T$; gradient = posterior − prior.
:::

::: {.slide title="Next: Whole Chapters in Context, or Retrieval on Demand"}
[Next]{.kicker}

A 6,000-token budget, 3.4 times the baseline's context, gained nothing at
twice the time. Would whole chapters in a 32,000-token context beat
retrieval on multi-passage questions? Agents take a third path: they
retrieve when they decide they need to.
:::
