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
every question about this book, a passage that answers it. Nobody asks a
question in order to receive a ranked list, however. The obvious next step
is to hand the passages to a language model and read its answer. So let us
wire the retriever to a generator, Qwen3.5-9B, and ask it what happens to
PyTorch modules kept in a plain Python list instead of `nn.Sequential`. The
passage that answers comes back first, and its last sentence reads "Modules
in a plain list stay unregistered, without any warning." The model writes
four fluent sentences about registration, cites the second and third
passages, and leaves out the one fact that the first passage adds: no
warning is raised. So the passage was there, at the top. Did the model skip
it, or did the passages it cited lead it elsewhere? Had the passage been
ranked tenth and left out of the prompt, the answer might read just the
same. How are we to tell these cases apart?

Systems of this kind are called *retrieval-augmented generation* (RAG),
after a 2020 paper that trained a retriever's query encoder together with a
generator :cite:`Lewis.Perez.Piktus.ea.2020`. Today the name covers any
pipeline that retrieves passages and places them in a prompt. Such a
pipeline is a chain of stages: a query is formed, passages are retrieved,
reordered, packed into a context and read by the generator. When the answer
is wrong, any stage may be at fault, and each stage needs a different
repair. One accuracy number cannot say which. The idea of this section and
the next is to locate a failure by measuring each stage on its own.

The pipeline itself is quickly built. Most of the work, and most of this
section, goes into the instruments: questions whose answers and source
passages we know, and a grader we have reason to trust. With both in hand we
measure a baseline between two bounds, a floor without passages and a
ceiling with only the passages that answer. :numref:`sec_lm_rag_fixes` then
changes one stage at a time and sorts the failures by the stage that caused
them.

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
import json
import os
import pandas as pd
import random
import re
import shutil
import textwrap
import time
import transformers
import zipfile
```

The two cells that follow are setup, and a reader can pass over them. They
pin the models: Qwen3.5-9B generates, the embedder and reranker of
:numref:`sec_lm_retrieval` retrieve, and a larger Qwen model, introduced
with the judge below, grades. Generating and judging every answer takes GPU
hours, so the answers and verdicts were recorded once. A normal run replays
them and generates a few again as a live check; the environment variable
`D2L_RERUN` regenerates them.

```{.python .input #rag-revisions}
%%tab pytorch
# Pinned in this notebook, not in the library: adding it to the saved
# HF_REVISIONS would change the fingerprint of every chapter-18 notebook
d2l.HF_REVISIONS['Qwen/Qwen3.5-9B'] = (
    'c202236235762e1c871ad0ccb60c8ee5ba337b9a')
```

```{.python .input #rag-config}
%%tab pytorch
GEN, RERANKER = d2l.RAG_GENERATOR, d2l.RAG_RERANKER
EMB, JUDGE = d2l.DEFAULT_EMBEDDER, d2l.RAG_JUDGE_LOCAL
REV = {GEN: d2l.HF_REVISIONS[GEN],
       **{m: d2l.HF_ENCODER_REVISIONS[m] for m in (EMB, RERANKER)}}
GPU = torch.cuda.is_available()
T0 = time.time()  # a judging run stops JUDGE_MINUTES after this
ARTIFACTS, CONFIGS = d2l.RAG_ARTIFACTS, d2l.RAG_CONFIGS
QA_KEY = 'lm-book-qa-v2'  # the question set
RERUN = d2l._rag_rerun()  # D2L_RERUN, checked against the run order
assert GPU or not RERUN - {'rag-cards'}, 'answers and judgments need GPUs'
MAX_NEW = d2l.RAG_BASE['max_new_tokens']  # new tokens per answer
BATCH, DEPTH = d2l.RAG_BASE['batch'], 50  # prompts per batch; ranking kept
QUICK = False  # True: 80 stratified dev and test items (development only)
assert not (QUICK and RERUN), 'QUICK results are never saved'
N_LIVE = 8 if GPU else 2  # base answers generated again in a normal run
JUDGE_MINUTES, CHUNK, CONCURRENCY = 50, 64, 8  # judging run; saves; flight
RETRY_TOKENS = 6000  # the one re-judging of an answer left without verdict
N_REPEAT = 60  # base answers judged twice
CARD_GIB = 21.5  # estimates; measured peaks add 0.4-1.5 (23.6 GiB usable)
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
recorded model responses the set was built from are published with it
(`lm-book-qa-api-cache-v1`), so its construction can be checked without any
API call. The next cell registers the set and, with it, every recorded
artifact of this section and the next; for now only the question set
matters.

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
- flags such as `identifier` (the question or reference names a function
  or variable of the book's code) or `general_knowledge` (answerable
  without the book).

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
   above"), that copies a run of the anchor's words (word overlap alone
   would then find the answer and make retrieval look easy), or that nearly
   repeats another.
3. A checker, Qwen3.8-27B, tested that each question stands alone and that
   its reference is supported, complete and unique. A search of neighboring
   passages collected other passages that answer it.
4. A frontier model, Claude Opus 5.5, verified every `dev` and `test` item
   in a fresh context.

The items were checked by models, not by people, and the choice is
deliberate. Model checks are fast, cheap and recorded, so anyone can replay
them. The verifier's 620 calls cost about \$7, and the whole set, drafts
and checks included, about \$15; a person checking 480 items against their
passages would need many hours. What the checks can miss is what the models
share. A reference that drafter, checker and verifier misread in the same
way passes every check, and the verifier may have read this public book and
accept a question it can answer from memory. The checker is also the judge
we use below, so a wrong reference that it passes is later used to grade
answers. The protection against shared blind spots is agreement with models
of other families, and Exercise :exref:`ex_lm_rag-hand-labels` compares
hand labels with the verifier's.

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
(`multihop`) and unanswerable questions. `train` passed only the rules and
the checker, and it carries no alternative spans, which are kept only where
answers are scored. The column `closed-book correct` records whether
Qwen3-4B, a smaller model than our generator, answered correctly without
passages, as graded by the judge introduced below; for our own generator we
measure this later.

Copying the anchor's words was the most common reason for rejection, more
common than all the others together. The checker rejected 165 unanswerable
and 133 code candidates but only 146 single-passage ones, although `dev`
and `test` hold six times as many single-passage questions as either: a
question the book does not answer, or one about its code, is the hardest
kind to write well.
Another 42 candidates meant to be unanswerable were answered by a
neighboring passage.

On whether an item passes, verifier and checker agree on 84% of the 536
`dev` and `test` candidates that both of them checked. That sounds like
agreement, but both pass most items, so they agree by default. *Cohen's κ*
corrects for this. It compares the observed agreement $p_o$ with the
agreement $p_e$ that two raters with the same pass rates would reach by
chance:

$$\kappa = \frac{p_o - p_e}{1 - p_e}.$$

It is 1 for perfect agreement and 0 for agreement at chance level. Two
raters who each pass about nine items in ten, independently of each other,
agree on $p_e = 0.9^2 + 0.1^2 = 0.82$ of them. Two coins that each land
heads nine times in ten would do as well. Against that, 84% is barely
better, and κ is $(0.84 - 0.82)/0.18 \approx 0.1$; the measured value is
0.12. The per-check table shows the extreme case. On whether the answer is
unique, the two agree on 96% of the items, and κ is 0.00: both call almost
every answer unique, and the few that one of them doubts are not the ones
the other doubts. So the items each model rejects are mostly different
ones, and the set very likely still holds items that a third check would
reject: the verifier alone rejected 67 items after every earlier step had
passed them. Here or anywhere else, an agreement rate means little until it
is set against the rate that chance alone would give, and we will hold the
judge of our answers to the same test.

Here is the first `dev` item of each type with its gold span.

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
plausible question about attention that the book does not answer. The rule
against copying is narrower than it may sound. It rejects a question that
repeats a run of five words of its passage, or whose vocabulary mostly
coincides with the passage's; the first item reuses the passage's words in
another order and passes. We will meet the consequence when we measure
retrieval.

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

Whether a question has a relevant chunk depends on how the book is cut.
`gold_chunk_ids` calls a chunk relevant if it overlaps a gold or
alternative span by at least half of the span's characters or half of the
chunk's, the rule of :numref:`sec_lm_retrieval`. We need it for two ways of
cutting, since the next section compares them: the section chunks of
:numref:`sec_lm_retrieval` (`sec512`: at most 512 tokens, cut at headings),
and fixed windows of 256 tokens, cut without regard to the text, either
without overlap (`fix256o0`) or overlapping by 64 tokens (`fix256o64`), so
that a sentence cut at one boundary appears whole in the next.

The next cell checks that every answerable item has a relevant chunk under
each chunking, and shows how differently two chunkings cut one gold span.

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
rules demanded. The gold span of `dev-0001`, the `load_array` question of
the chapter's introduction, lies in two section chunks but in five
overlapping windows, so "a relevant chunk among the first five" means very
different amounts of text under the two chunkings.

## The Pipeline and Its Measurements
:label:`subsec_lm_rag_pipeline`

A pipeline over this set has five stages (:numref:`fig_lm_rag_stages`).
The query is the question, or a rewrite of it. First-stage retrieval and
the optional reranking come from :numref:`sec_lm_retrieval`: BM25 and dense
rankings fused by reciprocal ranks, then perhaps reordered by a
cross-encoder. Context assembly decides which chunks enter the prompt, in
what order, within a token budget, and the generator writes an answer with
numbered citations or declines. Every stage but the first can be measured
without the stages after it; the query shows itself only in what retrieval
does with it. That property is what will let us locate a failure.

![The five stages of the retrieval-augmented generator, top to bottom, with the measurement that sees each one on its own. The judge below them is not a stage but the instrument that grades the answers, giving accuracy and faithfulness. The failure classes on the right blame a wrong answer on the first stage that lost it; they are defined and counted in the next section.](../img/mdl-lm-encoders-rag-stages.svg)
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
The stage changes of the next section alter the context or the query and
leave this prompt alone. Of all the details in this subsection, the
abstention sentence matters most for the scores: it alone decides every
unanswerable question.

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

`RAGPipeline` puts the pieces together. It is long because it carries
every option that the next section switches on, and a first reading needs
only `prepare` and `answer`. They run in three phases, so that the
generator never generates while the embedder and reranker share its GPU.
The generator writes a rewritten query if one is asked for. The retriever
searches, reranks and assembles while the generator waits. Then the
retriever's models leave the GPU, and the generator answers greedily, in
left-padded batches of eight. Two query options, `hyde` and `decompose`,
are never switched on in this chapter and can be skipped;
:numref:`sec_lm_encoders_fr` describes both.

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

Answers in free text cannot be graded by string match: "`is_train` is
passed as `shuffle`" and "the data are shuffled exactly when `is_train` is
true" give the same answer in different words. They need a grader that
reads. `rag_judge` shows a second model, Qwen3.8-27B with thinking on, the
question, the reference, the key facts, the passages and which of them the
answer cites, and asks for verdicts. The first is the one accuracy rests
on: an answer is *correct* if it states every key fact and nothing that
contradicts the reference.

Two more verdicts ask where the answer came from, which is what a user who
wants to check it relies on. An answer is *faithful* if every factual
statement in it is supported by the passages in its context, whatever the
reference says, and *supported* if the passages it cites suffice on their
own. The three can disagree in every direction. An answer from memory can
be correct and unfaithful; one that faithfully repeats an incomplete
passage, faithful and wrong. The subtle case is the third: an answer that
draws on passage [1] but cites only [2] is faithful and unsupported, since
the support is in the context but the citation points elsewhere.

Some scores need no judge. Answers to unanswerable questions are decided by
string match: a reply that contains the abstention sentence is correct,
alone or after an explanation, and any reply without it, however politely
it declines, counts as an answer and is incorrect. A reply with the
abstention sentence to an answerable question is incorrect too. *Citation
precision* is the share of cited passages that are relevant, and *citation
recall* the share of relevant passages in the context that the answer
cites.

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
```

## Baseline Answers

The baseline uses section chunks, hybrid retrieval, the first five chunks in
rank order within 3,000 tokens, and the fixed prompt. We answer the 240
`dev` questions with it and with three other configurations. We call each
such run an *arm*, borrowing the word for the groups of a clinical trial.
The *closed-book* arm gives no passages: a floor. The *oracle* arm gives
only the chunks of the gold spans, as a perfect retriever would: a ceiling
for retrieval. (An unanswerable question has no gold span, so there the
oracle keeps the retrieved passages.) And `base-repeat` answers the same
questions with the same settings in other batches, which will tell us how
large a difference noise alone produces.

Generator and judge each need whole GPUs. Before loading anything we
estimate what fits with `d2l.config_memory` of
:numref:`subsec_lm_hf_memory`: generation at eight prompts of 3,300 tokens,
and at four of 6,300 tokens, about the length of the largest context of the
next section; the retrieval phase, with the generator idle on its GPU; and
the judge.

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

The generator's weights take 16.7 GiB, so it gets a whole 24 GB GPU. Its
cache is small. Qwen3.5-9B is a hybrid of the kind described in
:numref:`subsec_lm_hf_config`: only 8 of its 32 layers store keys and
values, and the others carry a state of fixed size, so eight prompts of
3,300 tokens need 1.2 GiB for cache and state together. The estimate of
21.3 GiB leaves out the allocator's slack. The peaks measured in the full
runs, 21.7 GiB for these batches and 22.4 GiB for the four long prompts,
still fit, so the batch stays at eight. The judge needs two GPUs. The
estimate counts one byte per weight and gives 25 GiB, which is optimistic:
the FP8 checkpoint keeps its embedding and output matrices in bf16, so its
weights take a few GiB more.

The next cell loads the recorded answers of every arm. A normal run also
generates the first recorded batch again and compares the pipeline's
batched generation with the plain decoding loop `d2l.generate` of
:numref:`sec_lm_d2l_checkpoint`; a difference would mean a padding bug.

```{.python .input #rag-baseline}
%%tab pytorch
RET = {'sections': d2l.BookRetriever(CORPUS, reranker=RERANKER),
       'fixed': d2l.BookRetriever(FIXED, reranker=RERANKER)}
CHUNKS = {c['id']: c for R in RET.values() for c in R.corpus.chunks}
if 'rag-gen-base' in RERUN:
    d2l._rag_save_gens('rag-gen-base', d2l._rag_run_configs(
        ['closed', 'base', 'oracle', 'base-repeat'], DEV, RET, KEEP_GEN,
        DEPTH))
GENS = d2l._rag_load_gens(RERUN)
print(GENS.assign(context_tokens=[sum(CHUNKS[i]['n_tokens'] for i in c)
                                  for c in GENS.context_ids])
      .groupby(['split', 'config'], sort=False).agg(
          answers=('qid', 'size'), context_tokens=('context_tokens', 'mean'),
          prompt_tokens=('n_prompt_tokens', 'mean'),
          new_tokens=('n_new_tokens', 'mean')).round(0).astype(int))

LIVE = d2l.BookQA(list(DEV)[:N_LIVE], 'dev')  # the registered run's first
live = d2l._rag_run_config('base', LIVE, RET, KEEP_GEN, DEPTH)  # batch, rerun
print(f'live generation: {d2l.lm_peak()}')
ref = GENS[GENS.config == 'base'].set_index('qid')
tok = d2l._rag_generator()[1]  # only the tokenizer: freeing frees the model

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
oracle = d2l.generate(*d2l._rag_generator(), [d2l.rag_prompt(
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
d2l._rag_free_generator()  # the card goes to the retriever and the judge
```

The table lists every recorded run of this section and the next. Only the
first four `dev` rows concern us now: `base`, `base-repeat`, `closed` and
`oracle`. The other rows are the stage changes and the fix of the next
section, and the `test` rows are read once, at its end.

Five section chunks hold about 1,700 tokens and never reach the budget,
since five chunks of at most 512 tokens hold at most 2,560. The limit of
five chunks, not the budget, bounds the baseline's context. (Budgets count
tokens of the corpus's tokenizer, which differ from the generator's prompt
tokens.) Answers run to about 110 tokens. The live batch reproduces the
recorded answers, and the two decoding loops agree token for token.

### Checking the Judge

Every accuracy from here on rests on a model's verdicts. Why should we
believe them? We answer in two steps: first we look at the verdicts
themselves, then we check them against labels made without the judge.

The next cell loads the recorded verdicts on every answer of this section
and the next, and judges twelve of them again from the judge's recorded
replies, which needs neither a GPU nor a key. Thinking has a drawback here:
a judge that reasons at length can run out of tokens before it gives a
verdict. About 3% of the replies were cut off at 3,000 new tokens or came
back malformed twice, and each of those answers was judged once more with
6,000 new tokens. An answer still without a verdict is *unjudged*. It is
left out of accuracy and of the failure counts, rather than blamed on the
generator.

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
                 backend=backend, answer_sha=d2l._rag_answer_key(r),
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

CAL_V0 = calibration_verdicts(os.path.join(  # as recorded at selection
    LABELS, 'api', 'judge-local-qwen38-27b-on.jsonl'))
CAL_V = [dict(x) for x in CAL_V0]  # with the pairs judged again
JUD_FILE = d2l._rag_run_path('rag-judge', RERUN)
if 'rag-judge' in RERUN:
    d2l._rag_free_generator()
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
    old = d2l._rag_load_judgments(JUD_FILE if os.path.exists(JUD_FILE) else
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
JUD = d2l._rag_load_judgments(JUD_FILE)
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

The replayed verdicts match the recorded ones. The table again covers
every arm of this section and the next, `test` answers included, which is
why `base` has 480. Every baseline answer has a verdict; of the other arms
of this section, the oracle leaves one answer unjudged and `base-repeat`
two.

The judge itself was chosen by measurement. Two frontier models from other
families, Claude Opus 5.5 and Gemini 4 Argon, labeled 200 answers of
another, smaller generator, Qwen3-4B, and nine Qwen candidates of three
sizes judged the same answers, with thinking on and off. With thinking off
every candidate was lenient, passing many answers that both labelers call
wrong. With thinking on, a judge works through the key facts before it
decides, which is what catches a missing fact; the smaller candidates,
however, often reasoned until their replies were cut off, and agreed less
with the labelers. Half of the 200 answers served this choice. The other
100, the *calibration set*, test the chosen judge before we read any
accuracy: against each labeler, and against their *consensus*, the answers
on which both agree.

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
    print(f'the two labelers on {k}: kappa '
          f'{kappa(*zip(*[t for t in pairs if None not in t])):.2f}')
old = next(c for c in json.load(open(os.path.join(
    LABELS, 'candidates', 'table.json'))) if c['name'] == 'qwen35-4b-off')
oc = old['calibration']['correct']
print(f'the old judge (Qwen3.5-4B, thinking off) on correct: agreement '
      f'{oc["acc"]:.2f}, kappa {oc["kappa"]:.2f} [{oc["k_lo"]:.2f}, '
      f'{oc["k_hi"]:.2f}];\n  it passes '
      f'{old["calibration"]["false_correct"][0]} of the '
      f'{old["calibration"]["false_correct"][1]} answers the labelers call '
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

We accept a judge whose agreement with the consensus is at least 0.8 and
whose κ interval stays above 0.4, well clear of chance. This one agrees
with the consensus on 95% of the `correct` verdicts and on 96% of the
`faithful` ones, with κ of 0.85 and 0.87, and both κ intervals start at
0.7. These numbers flatter, because the consensus leaves out the hard
answers on which the labelers disagree. The fair comparison is with each
labeler alone. There the judge reaches κ 0.83 and 0.71 on `correct`, and
the labelers reach 0.81 with each other: the judge is about as reliable as
a third labeler. The judge it replaced, a 4B model with thinking off,
passed 14 of the 21 answers that both labelers call wrong, which is the
leniency the selection was meant to remove. One limit remains that no
calibration on these labels can remove. The labels come from models, not
from people, and they were made on the answers of another, smaller
generator; ours may make subtler mistakes, which this check does not see.

A practical note for readers who want to regenerate the verdicts. This
takes vLLM, an inference server that batches many requests on the GPU,
here splitting the judge's FP8 weights over two 24 GB GPUs. vLLM pins its
own PyTorch, so it lives in an environment of its own (`uv venv`, then
`uv pip install vllm==0.33.0`), whose `vllm` executable the variable
`D2L_VLLM` names. Without two such GPUs the same model, in bf16, can be
called on OpenRouter for about \$3.3 per 1,000 judgments. Local and API
verdicts on `correct` agreed on 194 of 200 answers with thinking off, and
about as often with thinking on.

### The Baseline Report

Before the averages, one answer. The chapter began by asking how the book's
old `load_array` helper decides whether to shuffle its data, and the
retriever of :numref:`sec_lm_retrieval` ranks the passage that says so
first. The next cell prints what the generator replies to that question
without passages and with the baseline's five, each with the judge's
verdicts.

```{.python .input #rag-load-array}
%%tab pytorch
# The question of the chapter's introduction, about load_array (dev-0001),
# answered without passages and by the baseline, with the judge's verdicts
item = ITEMS['dev-0001']
wrap = lambda t, first='': textwrap.fill(t, 96, initial_indent=first,
                                         subsequent_indent='    ')
print(wrap(f'question: {item["question"]}'))
print(wrap(f'reference: {item["reference"]}'))
for c in ('closed', 'base'):
    rec = ANSWERS[c, item['qid']]
    jv = judged[(judged.config == c) & (judged.qid == item['qid'])]
    verdicts = ', '.join(f'{k} {jv[k].iloc[0]}' for k in VERDICTS
                         if len(jv) and pd.notna(jv[k].iloc[0]))
    print(f'\n{c}, {len(rec["context_ids"])} passages; judge: '
          f'{verdicts or "no verdict"}')
    print(wrap(rec['answer'], '    '))
```

Without passages the generator invents a plausible mechanism, a `shuffle`
argument that the caller sets, and never mentions `is_train`; the judge
marks the answer wrong. With the five retrieved chunks it says that
`is_train` is passed to the `DataLoader` as `shuffle`, which is the book's
answer. This is the system doing what it was built for, apart from the
citations it was told to give and left out: the judge calls the answer
correct and faithful, but not supported.

In the report each rate carries a 95% interval that resamples documents,
since questions drafted from one document share its difficulty, and each
difference between arms is paired over the same questions
(:numref:`sec_lm_eval_suites`, :numref:`subsec_mdl-paired-comparisons`).

One row of the report is worth predicting before it is read.
`base-repeat` sends the same 240 prompts to the same model with greedy
decoding, and only the neighbors in each batch of eight differ. How many of
its answers will change?

```{.python .input #rag-baseline-report}
%%tab pytorch
CLUSTER = d2l._rag_clusters(ITEMS.values(), CORPUS)  # qid: its document
boot, paired = d2l._rag_boot, d2l._rag_paired  # 'mean [95% interval]'
scores = lambda c, split='dev': d2l._rag_scores(  # per-question scores
    c, split, GENS, JUD, RET, ITEMS, CLUSTER, CONCURRENCY)
ready = lambda *c, split='dev': d2l._rag_ready(  # are all of c's verdicts in?
    c, split, GENS, JUD, RERUN, CONCURRENCY)
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

Not one of the 240 contexts changed, yet 108 of the 240 answers did: 45%.
Accuracy did not move; the difference is $-0.004$, with an interval from
$-0.03$ to +0.02. Nothing is broken. When the padding of a prompt's batch
changes, so does the shape of every matrix product, and the GPU may add the
same bf16 numbers in another order; the logits move by a few rounding steps.
Wherever the two likeliest tokens are nearly tied, that flips the greedy
choice, and from that token on the answer takes another path.
:numref:`subsec_lm_hf_kernels` met the same near-ties when it changed a
batch from one prompt to two.

Greedy decoding is therefore deterministic only for a fixed batch,
precision and kernel. An evaluation must batch the configurations it
compares alike, or count re-batching as noise (:numref:`sec_lm_scoring`).
Here re-batching moves accuracy by less than 0.03, and a change to the
pipeline has to do better than that before we believe it. The judge's
server batches its requests too, so the same effect reaches the verdicts:
judged again with twice as many requests in flight, 5 of 54 answers changed
a verdict. Every arm was judged at the same concurrency, so this noise
widens the paired intervals rather than favoring one arm.

With the size of the noise known, the rest of the report reads quickly. A
relevant chunk is among the first five for 97% of the answerable
questions: retrieval is nearly solved here. That should not surprise us.
The questions were drafted from the book's passages and, like the first
example, often reuse their words, and :citet:`Lee.Chang.Toutanova.2019`
observed that when the people asking already know the answer, BM25 alone
retrieves well. With those chunks the generator answers 0.89 of all `dev`
questions correctly, and without them 0.30, a gain of about 0.6. The
closed-book score is not zero, nor should it be: the book is public, and
some questions ask for general knowledge. On the 145 answerable questions
that the checker marked as specific to the book, closed-book accuracy falls
to 0.12, against 0.90 with retrieval. That pair of numbers is the case for
retrieval. Nor do the passages merely jog the generator's memory: the
baseline answers about nine in ten answerable questions correctly, both
where the closed-book arm was right and where it was wrong.

The answers mostly come from their passages. The judge calls 0.92 of them
faithful and finds the cited passages sufficient for 0.82. Citation
precision is 0.72 and recall 0.85; precision from gold spans is strict,
since a citation of a supporting passage that is not gold counts against
it. The baseline abstains on 18 of the 20 unanswerable questions and on 2%
of the answerable ones.

The oracle arm answers 0.92 of all questions. (Its Recall@5 and coverage in
the report are those of the retriever it bypasses. Its own context holds a
relevant chunk for every answerable question, which is also why its
citation precision is 1.00 by construction.) On the answerable questions
its gain over the baseline is +0.04, with an interval from $-0.01$ to
+0.09, and cannot be told from zero. Why so little? Call a question a *hit*
when its context holds a relevant chunk. Let $h$ be the share of hits among
the answerable questions, and $a_1$ and $a_0$ the accuracy on hits and on
misses. Then the accuracy on the answerable questions is

$$\textrm{accuracy} = h\, a_1 + (1 - h)\, a_0.$$
:eqlabel:`eq_lm_rag_hit`

Accuracy is a mixture of how often retrieval succeeds and how well the
generator does in each case. Here 213 of 220 contexts hold a relevant chunk,
$h = 213/220$, with $a_1 = 0.91$, and $a_0 = 0.29$ on the seven misses. A
perfect retriever can only turn those seven misses into hits. If the
accuracy on hits stayed at $a_1$, it would add
$(1 - h)(a_1 - a_0) \approx 0.032 \times 0.62 \approx 0.02$: there is no
room left for retrieval to show a gain, and the oracle's +0.04 lies within
noise of that estimate. The bold step is the assumption that $a_1$ stays
put. The oracle passes only the gold chunks, about 480 context tokens
against the baseline's 1,700, and so also removes passages that might
distract the generator. Whether that raises the accuracy on hits, 240
questions cannot tell.

Of everything in this section, the decomposition :eqref:`eq_lm_rag_hit` is
the result to remember. Before a single change has been tried, it says
which stage to improve first, and here it points at the generator: of the
24 errors on answerable questions, 19 were made with a relevant passage in
the context.

The wrong answer that opened this section is one of those nineteen: its
passage was in the context, ranked first. Which stage lost each of them,
and would another chunking, query or context have saved some?
:numref:`sec_lm_rag_fixes` changes one stage at a time and sorts the
failures by stage.

## Summary

When a retrieval-augmented generator answers wrongly, one accuracy number
cannot say which of its stages failed, although each stage needs a
different repair. Locating the failure takes instruments: questions over
the corpus with known answers and gold spans, here drafted and checked by
models rather than people; a judge checked against independent labels
before any accuracy is read; and a measurement of each stage on its own,
between a floor without passages and a ceiling with only the right ones. On
this book the baseline answers 0.89 of the `dev` questions against 0.30
without passages. Retrieval is nearly solved, the decomposition
$h\,a_1 + (1-h)\,a_0$ leaves a perfect retriever almost nothing to add, and
most wrong answers had a relevant passage in the context: the generator
limits accuracy. Greedy decoding in bf16 also depends on the batch, which
changed 45% of the answers and not accuracy.

## Exercises

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
       behind, if any. With one GPU, again count citations of the relevant
       passage instead of judging.
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

1. **Which stage to improve first.** A colleague's pipeline puts a
   relevant chunk in the context for 90% of the questions, and its
   accuracy is 0.7 with one and 0.2 without.
    1. Use :eqref:`eq_lm_rag_hit` to compute its accuracy, and its accuracy
       with an oracle context, assuming that the accuracy given a hit stays
       at 0.7. Which gains more: a perfect retriever, or raising the
       accuracy given a hit to 0.8?
    1. Why is "the accuracy given a hit stays at 0.7" an assumption? Name a
       way in which a perfect retriever would also change the accuracy
       given a hit, and say why the oracle arm of this section cannot tell
       whether it does.
    1. Suppose two configurations disagree on 10% of 240 questions. Estimate
       the standard error of the paired difference in accuracy, and the
       smallest gain a 95% interval would separate from zero. Which of the
       two improvements could 240 questions detect?
   :exlabel:`ex_lm_rag-invest`

1. **Agreement without information.** On whether an item's answer is
   unique, the verifier and the checker agree on 96% of 536 items, with
   κ = 0.00.
    1. Write down a table of counts over the 536 items (both pass, only the
       verifier passes, only the checker passes, both reject) with
       agreement 0.96 and κ = 0. What must one of the two raters do?
    1. Show that κ = 0 exactly when the share of items that both reject
       equals the product of the two raters' rejection rates.
    1. A judge passes 95% of the answers, and a labeler 90%. What is the
       largest κ the two can reach, whatever the answers? What does this
       mean for comparing κ across sets of answers that a generator gets
       right at different rates?
   :exlabel:`ex_lm_rag-kappa`

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

Retrieval-Augmented Generation<br>**a question set over the book · stages and their measurements · a checked judge · floor, baseline and ceiling**
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
- Judge: wrong; faithful; supported.

::: {.d2l-note .warn}
The baseline answers 0.89 of 240 questions. Passage skipped? Misled by
its neighbors? For other questions, ranked too low? One accuracy number
cannot tell.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- build a question set with gold passages and no leakage between splits;
- measure each stage: recall, coverage at a token budget, accuracy,
  faithfulness, citations;
- check a model judge before trusting its accuracy;
- bound what a better retriever can add, between a floor without passages
  and an oracle ceiling.

**Running example:** this book in 5,324 section chunks; 240 `dev`
questions about 32 documents, among them the chapter's `load_array`
question; Qwen3.5-9B answers, Qwen3.8-27B judges.
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
| | Failures and Fixes in RAG |
:::

::: {.col .narrow}
- **Uses:** BM25 + dense search fused by reciprocal ranks, a reranker,
  paired bootstrap intervals.
- **Skips:** changing stages, sorting failures and testing fixes (next
  section); retrieval inside agent loops.
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
- Split by **document**: 168 `train`, 32 `dev`, 32 `test`, so a system
  tuned on `train` never saw the passage of a `dev` answer.
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

::: {.fragment .fade-up}
Extreme case, "is the answer unique?": 96% agreement, κ = 0.00.
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
- Unanswerable question: a reply containing the abstention sentence is
  correct; any other reply is a false answer (string match, no judge).
:::

::: {.slide title="Correct, Faithful and Supported Are Three Different Verdicts"}
[What the judge decides]{.kicker}

| the answer | correct | faithful | supported |
|---|---|---|---|
| right, from memory, not from the passages | yes | no | no |
| repeats an incomplete passage, citing it | no | yes | yes |
| draws on [1], cites only [2] | either | yes | no |

- **correct:** every key fact, nothing contradicting the reference.
- **faithful:** every statement backed by the passages in the context.
- **supported:** the cited passages suffice on their own.

::: {.d2l-note}
Accuracy rests on *correct*; the other two say whether a reader can check
the answer against its sources.
:::
:::

::: {.slide title="The Judge Agrees with the Labelers Almost as Well as They Agree with Each Other"}
[A judge, checked first]{.kicker}

Qwen3.8-27B, thinking on, FP8 on two 24 GB GPUs; 100 calibration answers
labeled by two frontier models.

| `correct` | κ |
|---|---|
| judge vs Claude Opus 5.5 | 0.83 |
| judge vs Gemini 4 Argon | 0.71 |
| the two labelers | 0.81 |
| judge vs their consensus | 0.85 [0.70, 0.97] |
| old judge, 4B, thinking off, vs consensus | 0.33 |

::: {.fragment .fade-up}
The consensus flatters: it drops the hard answers the labelers disagree on.
Compare with each labeler alone.
:::

::: {.d2l-note .warn}
Labels by models, on answers of another, smaller generator (Qwen3-4B).
:::
:::

::: {.slide title="Without Passages the Generator Invents; with Them It Finds `is_train`"}
[One answer]{.kicker}

**Question:** how does the book's legacy `load_array` decide whether its
`DataLoader` shuffles? **Reference:** it passes `shuffle=is_train`.

| context | answer, abridged | judge |
|---|---|---|
| none | "relies on the `shuffle` argument passed to the `DataLoader` constructor" | wrong |
| five chunks, the right one first | "`is_train` ... is passed directly to the `DataLoader` constructor as the `shuffle` parameter" | correct, faithful, not supported |

::: {.d2l-note}
Retrieval turns a plausible invention into the book's answer. The
citations the prompt asked for are missing.
:::
:::

::: {.slide title="Check: The Same 240 Prompts in Other Batches. How Many Answers Change?"}
[Check]{.kicker}

`base-repeat`: same model, same contexts, greedy decoding; only the
neighbors in each batch of 8 differ.

::: {.fragment .fade-up}
- 0 of 240 contexts differ; 108 of 240 answers differ.
- Accuracy: $-0.004$, interval [$-0.03$, +0.02].
- bf16: another padding changes the order of the sums; the logits move by
  a few rounding steps and flip near-ties.
- The judge's server batches too: 5 of 54 answers changed a verdict.
:::

::: {.d2l-note .rule}
Greedy decoding is deterministic only for a fixed batch, precision and
kernel. Batch alike, or count re-batching as noise.
:::
:::

::: {.slide title="Retrieval Lifts Accuracy from 0.30 to 0.89; an Oracle Adds Nothing Measurable"}
[Floor, baseline, ceiling]{.kicker}

| 240 `dev` questions | closed book | baseline | oracle |
|---|---|---|---|
| accuracy | 0.30 | 0.89 | 0.92 |
| citation precision | | 0.72 | 1.00 (by construction) |
| abstains, unanswerable | 10 of 20 | 18 of 20 | 17 of 20 |

- Baseline − closed: +0.59 [+0.53, +0.66]; book-specific: 0.12 → 0.90.
- Oracle − baseline, all questions: +0.03 [$-0.02$, +0.08]. Retrieval
  has no room left to show a gain.
:::

::: {.slide title="Check: 213 of 220 Hits. What Can a Perfect Retriever Add?"}
[Check]{.kicker}

Accuracy on answerable questions:
$$\textrm{accuracy} = h\, a_1 + (1 - h)\, a_0$$

$h = 213/220$: share of answerable questions whose context holds a relevant
chunk (a *hit*); accuracy $a_1 = 0.91$ on hits, $a_0 = 0.29$ on the seven
misses.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** If $a_1$ stays put, $(1 - h)(a_1 - a_0) \approx 0.032 \times
0.62 \approx 0.02$. The oracle measures +0.04 [$-0.01$, +0.09], within
noise of 0.02. 19 of 24 errors had a relevant passage in the context.
:::
:::
:::

::: {.slide title="Recap"}
[Recap]{.kicker}

- 240 `dev` questions with gold spans, made and checked by models.
- Five stages, each with its own measurement; `cov@T` compares chunkings.
- `load_array`: a `shuffle` argument invented without passages, `is_train`
  found with them.
- Judge vs each labeler: κ 0.71 and 0.83; the labelers vs each other:
  0.81.
- Re-batching changed 45% of answers in bf16, not accuracy.
- Closed 0.30, baseline 0.89, oracle 0.92: the generator limits accuracy.
:::

::: {.slide title="Next: Which Stage Lost Each Wrong Answer?"}
[Next]{.kicker}

The baseline leaves 26 `dev` answers wrong, 19 of the 24 answerable ones
with a relevant passage in the context. The next section changes one stage
at a time, sorts the failures by stage and tests a fix chosen in advance.
:::
