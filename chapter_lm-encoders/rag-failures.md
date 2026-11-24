# Failures and Fixes in Retrieval-Augmented Generation
:label:`sec_lm_rag_fixes`

Suppose you have built the pipeline of :numref:`sec_lm_rag` and are looking
at the 21 `dev` questions it gets wrong. The natural impulse is to start
fixing. Most people who build such systems have a favorite repair: smaller
chunks, a rewritten query, more passages in the prompt, a sterner
instruction to the generator. Each is plausible, and each has a paper that
reports a gain. Yet this baseline already puts a relevant chunk into the
context for 213 of the 220 answerable questions, and the wrong answer that
opened :numref:`sec_lm_rag` had its passage ranked first. Which of the
favorites can help a system like that, and how would we know?

In this section the measurements choose. We change one stage at a time and
compare with the baseline on the same questions. We sort the wrong answers
by the first stage that lost them. And we test the repair of the largest
class, written down before any failure was counted, while 240 `test`
questions are read only once. None of this is new: medicine has run its clinical
trials this way for decades, with a control measured on the same patients
and an analysis fixed before the data come in. Here the procedure will
matter more than any of the repairs. A short derivation then asks whether
the answers alone could teach the retriever which passages the generator
can use. The section ends on the question that longer context windows keep
raising: is retrieval needed at all when whole chapters fit into the
prompt?

The first cells load what :numref:`sec_lm_rag` built: its configurations,
the corpus with its two retrievers, the recorded answers with the judge's
verdicts, and the library's helpers for paired intervals. They hold no new
idea, and a first reading can skip to :numref:`subsec_lm_rag_stages`. As
there, a normal run replays the recorded answers and verdicts, and
`D2L_RERUN` regenerates them.

```{.python .input #rag-failures-failures-and-fixes-in-retrieval-augmented-generation-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #rag-failures-failures-and-fixes-in-retrieval-augmented-generation-2}
%%tab pytorch
import json
import os
import pandas as pd
import random
import re
import textwrap
import time
import transformers
import warnings
```

```{.python .input #rag-failures-failures-and-fixes-in-retrieval-augmented-generation-3}
%%tab pytorch
# The settings of the previous section: reading its records checks them
d2l.HF_REVISIONS['Qwen/Qwen3.5-9B'] = (  # pinned in the notebook, as there
    'c202236235762e1c871ad0ccb60c8ee5ba337b9a')
GEN, RERANKER = d2l.RAG_GENERATOR, d2l.RAG_RERANKER
EMB = d2l.DEFAULT_EMBEDDER
REV = {GEN: d2l.HF_REVISIONS[GEN],
       **{m: d2l.HF_ENCODER_REVISIONS[m] for m in (EMB, RERANKER)}}
GPU = torch.cuda.is_available()
ARTIFACTS, CONFIGS = d2l.RAG_ARTIFACTS, d2l.RAG_CONFIGS
BASE, STAGES, CONTROLS = d2l.RAG_BASE, d2l.RAG_STAGES, d2l.RAG_CONTROLS
RERUN = d2l._rag_rerun()  # here: rag-cards, -gen-stages, -gen-fix
assert GPU or not RERUN - {'rag-cards'}, 'answers need a GPU'
K, BUDGET, DEPTH = BASE['k'], BASE['budget'], 50  # passages; tokens; ranking
BATCH = BASE['batch']  # prompts per batch
QUICK = False  # True: 80 stratified dev and test items (development only)
assert not (QUICK and RERUN), 'QUICK results are never saved'
CONCURRENCY, N_CARDS = 8, 50  # requests in flight when judged; cards
CARD_GIB = 21.5  # estimates; measured peaks add 0.4-1.5 (23.6 GiB usable)
```

```{.python .input #rag-failures-failures-and-fixes-in-retrieval-augmented-generation-4}
%%tab pytorch
QA = {s: d2l.book_qa(s) for s in ('train', 'dev', 'test')}  # test: once
DEV, TEST = (QA[s].subset(80) if QUICK else QA[s] for s in ('dev', 'test'))
ITEMS = {q['qid']: q for qa in QA.values() for q in qa}
CORPUS = d2l.book_corpus()  # structure-aware chunks of at most 512 tokens
FIXED = d2l.book_corpus('fixed', size=256, overlap=64)
RET = {'sections': d2l.BookRetriever(CORPUS, reranker=RERANKER),
       'fixed': d2l.BookRetriever(FIXED, reranker=RERANKER)}
CHUNKS = {c['id']: c for R in RET.values() for c in R.corpus.chunks}
CLUSTER = d2l._rag_clusters(ITEMS.values(), CORPUS)  # qid: its document
mem = lambda m, n, b: d2l.config_memory(transformers.AutoConfig
                                        .from_pretrained(m, revision=REV[m]),
                                        n, b)
g, e, r = mem(GEN, 3300, BATCH), mem(EMB, 512, 16), mem(RERANKER, 1024, 16)
KEEP_GEN = (g['weights'] + e['weights'] + r['weights'] + max(
    e['activations'], r['activations'])) / 2**30 < CARD_GIB
print(f'{len(DEV)} dev and {len(TEST)} test questions; {len(CORPUS):,} '
      f'section chunks and {len(FIXED):,} windows')
```

```{.python .input #rag-failures-failures-and-fixes-in-retrieval-augmented-generation-6}
%%tab pytorch
GENS = d2l._rag_load_gens(RERUN)  # the answers and their verdicts
JUD = d2l._rag_load_judgments(d2l._rag_run_path('rag-judge', RERUN))
boot, ci, paired = d2l._rag_boot, d2l._rag_ci, d2l._rag_paired
scores = lambda c, split='dev': d2l._rag_scores(  # per-question scores
    c, split, GENS, JUD, RET, ITEMS, CLUSTER, CONCURRENCY)
ready = lambda *c, split='dev': d2l._rag_ready(  # are all of c's verdicts in?
    c, split, GENS, JUD, RERUN, CONCURRENCY)
S = {'base': scores('base')} if ready('base') else {}
print(f'{len(GENS):,} recorded answers in {GENS.config.nunique()} '
      f'configurations, {len(JUD):,} judgments')
if S:
    print(f'baseline accuracy on dev: '
          f'{boot(S["base"].accuracy, cluster=S["base"].cluster)}')
```

## Changing One Stage at a Time
:label:`subsec_lm_rag_stages`

Before we generate a single answer, we already know roughly how much a
change on the retrieval side can gain. Accuracy on the answerable
questions is a mixture of the accuracy on hits, the questions whose
context holds a relevant chunk, and the accuracy on misses
(:numref:`sec_lm_rag`). The baseline has 213 hits among 220 answerable
questions, so a perfect retriever could add about 0.02, well inside the
paired intervals of about $\pm 0.03$ that 240 questions give. Retrieval has
almost no room left on this book, because the questions were drafted from
its passages and reuse their words (:numref:`sec_lm_text_embeddings`); on
a harder collection the balance can reverse. Changes to the order or the
length of the context act on how the generator reads, and the mixture does
not bound them.

Chunking comes first, because it can be measured before any answer is
generated. Section chunks follow the book's headings and paragraphs. Fixed
windows of 256 tokens, overlapping by 64, are cut without regard to the
text. Which of the two covers more? Recall@$k$ cannot compare them fairly,
since five section chunks hold more text than five windows. Coverage at a budget of $T$
tokens, the `cov@T` of :numref:`subsec_lm_rag_pipeline`, fixes the amount of
text instead: it is the share of answerable questions whose first chunks,
taken in rank order up to $T$ tokens, hold at least half of a gold span.

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

Section chunks cover more when the budget is tight: 0.91 of the questions
at 1,000 tokens against 0.86 for windows, a gap of 0.05 whose interval
excludes zero. A section holds a span whole where a window may cut it, and
overlapping windows spend part of the budget on text they repeat. By 3,000
tokens both find room, and the gap shrinks to 0.01, well within noise.

Coverage says that the gold text is in the context, not that the generator
uses it. For that we must generate. Each configuration that answers the
questions is an arm of the experiment, as in :numref:`sec_lm_rag`. The
baseline fuses the BM25 and dense rankings by reciprocal ranks and passes
the first five section chunks, best first, with no reranker. The next cell
answers the 240 `dev` questions with five arms that change it:

- fixed windows instead of section chunks, each question getting at most
  the tokens of its baseline context;
- the question rewritten by the generator as a search query, with the
  reranker reordering the fused top 50;
- the baseline context in reverse order, the best chunk next to the
  question;
- a budget of 1,500 tokens, filled in rank order without the limit of five
  chunks;
- the same with a budget of 6,000 tokens.

The second arm bends the rule of one change at a time, and on purpose. A
rewritten query can drift from the question, so it is tested together with
a guard: the reranker reads the original question and reorders what the
rewrite found. The arm therefore adds two stages to the baseline, and its
result belongs to the pair.

A sixth arm is a control rather than a candidate. Packing stops before the
window that would pass the baseline's tokens, so fixed windows get slightly
fewer context tokens than the baseline, and a loss could come from the
missing tokens rather than from the cut. The control, `fixed-over`, packs
windows until they pass the baseline's tokens by less than one window, and
so gets slightly more context than the baseline.

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
    path = d2l._rag_run_path('rag-gen-stages', RERUN)  # the others are kept
    if not os.path.exists(path):  # from an earlier run, else registered
        path = d2l.lm_registered(ARTIFACTS['rag-gen-stages'][0])
    recs = [r for r in d2l._rag_records([path]).to_dict('records')
            if r['config'] not in names]
    tokens = context_tokens(GENS[(GENS.config == 'base') &
                                 (GENS.split == 'dev')].to_dict('records'))
    for name in names:
        c = CONFIGS[name]
        budgets = ({q: t + c.get('over', 0) for q, t in tokens.items()}
                   if c['budget'] == 'base' else None)
        recs += d2l._rag_run_configs([name], DEV, RET, KEEP_GEN, DEPTH,
                                     budgets=budgets)
    d2l._rag_save_gens('rag-gen-stages', recs)
    GENS = d2l._rag_load_gens(RERUN)
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

No change raised accuracy measurably. The only gain, from the 6,000-token
budget, is about +0.01, with an interval from $-0.03$ to +0.05; reversing
the order and the rewrite with reranking come out about level with the
baseline. On the retrieval side this is what the
mixture predicted. On the generator's side it is a finding: the generator
did not read measurably better for a different order or for three times the
text.

The two losses teach more. Fixed windows lost 0.09. They had a little less
context than the baseline, about 1,600 tokens against 1,700, but the
control rules the token count out: with about 1,800 tokens, windows still
lost 0.05. Coverage accounts for less than half of the loss. At
budgets near the windows' own, their coverage trails by 0.02 to 0.05. A
question that drops out of coverage falls at worst from the accuracy on
hits to the accuracy on misses, about 0.6 lower, so such a gap could cost
at most about 0.01 to 0.03. The rest is the cut itself, and coverage is
blind to it: it
counts half a gold span as found, and a window can hold that half and cut
off the key fact in the other (Exercise :exref:`ex_lm_rag-whole-span`).
Chunks that follow the document's structure keep a paragraph whole, and
they gave better answers.

Less context costs accuracy where coverage falls. The 1,500-token budget,
about 1,300 context tokens, lost 0.06, and coverage drops from 0.97 at
2,000 tokens to 0.91 at 1,000. More context, by contrast, did nothing for
accuracy: the 6,000-token budget, about 5,800 context tokens, gained
nothing measurable and doubled the time per answer.

A rule fixed in advance carries the stage change with the largest `dev`
gain to `test`, read once, to see whether a gain this small survives fresh
questions. The cell's last line names the 6,000-token budget, ahead of the
others by two verdicts in 240. But `test` has been read already. It was
read once, with the verdicts of the judge's earlier 8-bit build, when the
same rule named reversing the order. A test set read again after every change of the
verdicts would become development data, so we do not read it again: the
reversed order is the stage change that `test` was read with, at the end of
the next subsection.

## Failures by Stage
:label:`subsec_lm_rag_failures`

No change helped on average. Perhaps the 21 wrong answers fail in different
ways, each needing its own repair, and an average over them hides that. To
find out, `classify_failure` asks six questions of each wrong answer, in
pipeline order, and blames the first stage that lost the answer:

1. An unanswerable question was answered: a *false answer*.
2. No relevant chunk is among the first-stage top 50: *not retrieved*.
3. A relevant chunk is in the top 50 but not in the context: *ranked low*.
4. A relevant chunk is in the context and the answer abstains: a *false
   abstention*.
5. A relevant chunk is in the context and the answer, faithful to the
   passages, is incorrect: *ignored*. The answer stays within the passages
   and still misses a key fact that one of them holds.
6. A relevant chunk is in the context and the answer is not faithful:
   *unsupported*.

The order matters more than the names: the first question that holds
decides, so every wrong answer gets exactly one class. The names, in turn,
describe what the answer did, not why; *ignored* says that the fact was
there and did not make it into the answer, which is all the rule can see.
:numref:`fig_lm_rag_stages` places each class beside its stage.

```{.python .input #rag-failures-failures-by-stage-1}
%%tab pytorch
#@save
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

Each class has its own repair (:numref:`tab_lm_rag_fixes`), written into
`d2l.RAG_FIXES` with the configurations of :numref:`sec_lm_rag` before any
class was counted.

:The failure classes, the stage each blames, and the fix written down for it before the failures were counted.
:label:`tab_lm_rag_fixes`

| class | stage | fix written down in advance |
|---|---|---|
| not retrieved | first-stage retrieval | fuse a third ranking, for the rewritten question, with the BM25 and dense rankings |
| ranked low | reranking, assembly | rerank the top 100 and pass eight chunks |
| ignored | generation | quote the supporting sentences, then answer; best passages at both ends; room for 500 new tokens |
| unsupported | generation | drop each sentence that its cited passages do not support, as the generator judges |
| false answer | generation | abstain below a reranker score tuned on `train` |
| false abstention | generation | state the abstention rule after the passages |

Why write the fixes down first? A fix chosen after reading the failures is
fitted to them, and its gain on the same questions is optimistic, like any
choice made on test data. With the remedies fixed in advance, the data
decide only which class is largest, and the notebook takes that decision by
rule. Experimental science calls this *pre-registration*: the analysis is
written down before the data are seen. It is the hard discipline of this
section, because it forbids the one thing we most want to do after reading
a failure, which is to invent a fix for it.

The next cell counts the classes with intervals and prints five *failure
cards*, records of a failure with its question, answer, ranking and
verdicts. The judge's three verdicts are those of :numref:`sec_lm_rag`: an
answer is correct if it states every key fact, faithful if the passages
support every statement in it, and supported if the passages it cites
suffice on their own. Claude Opus 5.5 also read each card and labeled its
class, and the cell compares these labels, again a model's, with the
rule's.

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

def cites_relevant(row):
    """Whether an answer cites a passage that holds a gold span."""
    ctx, gold = row.record['context_ids'], d2l.gold_chunk_ids(ITEMS[row.qid],
                                                              CORPUS)
    return any(ctx[int(n) - 1] in gold for n in row.record['cited_ids']
               if 0 < int(n) <= len(ctx))  # citations count from 1

LARGEST = FIX = None
if ready('base'):
    BC = classes('base')
    fails = BC[~BC.cls.isin(['correct', 'unjudged'])]
    SH = class_shares(fails.cls)
    SH['cites a relevant passage'] = [
        sum(cites_relevant(r) for r in fails[fails.cls == c].itertuples())
        for c in CLASSES]
    n_un = (BC.cls == 'unjudged').sum()
    print(f'{len(fails)} of {len(BC) - n_un} judged base answers on dev are '
          f'not correct ({n_un} unjudged, left out)')
    print(SH.assign(count=SH['count'].astype(int), share=lambda d: d.apply(
        lambda r: f'{r.share:.2f} [{r.lo:.2f}, {r.hi:.2f}]', axis=1))
          [['count', 'share', 'cites a relevant passage']].to_string())
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
    path = d2l._rag_run_path('rag-cards', RERUN)
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
        f = now != 'correct'  # the cards of failures
        a, r = agent.values[f], now.values[f]
        print(f'on the {f.sum()} failures: the agent agrees with the rule on '
              f'{(a == r).sum()},\n  calls {(a == "correct").sum()} correct '
              f'and blames another stage on '
              f'{((a != r) & (a != "correct")).sum()}')
    else:
        print('\nthe cards carry no labels of a second labeler yet')
```

None of the 21 failures happened at first-stage retrieval, which is what
the hit rate led us to expect. Ignored is the largest class, 8 failures or
0.38 of them, but its interval, [0.19, 0.57], overlaps that of
unsupported, 7 failures and [0.14, 0.52]. Twenty-one failures are a small
sample to divide six ways, and they cannot say which of the two classes is
larger.

The last column shows what the two largest classes have in common. Five
of the 8 ignored answers cite a passage that holds the answer, and so do
all 7 unsupported ones. In these answers the generator found the right
passage and even pointed to it. Then it left out a key fact that the
passages state, or added a claim that they do not support.

How far can we trust the classes? Each of the 21 failures has a card, and
the labeler agrees with the rule on 12 of them, 0.57. It calls 4 correct
and blames another stage on 5. The four are answers that the labeler
accepts and our grading does not. In three the judge insists on a key fact
that the labeler finds stated, if clumsily. In one, an unanswerable question was declined in words other than
the abstention sentence of :numref:`sec_lm_rag`, "I cannot answer this
from the provided passages", which is the only reply our grading counts as
declining. Of the five other attributions, three trace an unsupported
answer to a relevant chunk that was ranked below the context: the answer
went beyond its passages because the passage it needed was missing.

Rule and labeler disagree about whether an answer is wrong about as often
as about where it went wrong. The two largest classes fare differently,
which matters for what comes next. The class the rule is about to pick,
ignored, is the one the labeler confirms most often: 6 of the 8 answers.
Of the 7 unsupported answers it confirms only 2, and blames 3 on ranking.

Let us return to the answer that opened :numref:`sec_lm_rag`. It was picked
by a fixed rule: the first `dev` answer, by question id, whose context held
a relevant chunk, that the judge called incorrect, and whose card the
labeler gave the same class as the rule. With two graders required to
agree, the example is unlikely, though not certain, to be a judge error.

```{.python .input #rag-opening-item}
%%tab pytorch
# The rule: the first base dev answer (in qid order) with a relevant chunk
# in its context that the judge called incorrect, and whose failure card
# carries the same class from the second labeler as from the rule: a
# failure of the pipeline that two graders agree on, not a judge error
path = d2l._rag_run_path('rag-cards', RERUN)
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
          'by the second labeler')
```

The answering passage was ranked first, and its last sentence, printed
above and quoted at the start of :numref:`sec_lm_rag`, holds the missing key
fact. The answer cites passages two and three and leaves out that no warning
is given. The passage was neither missing nor contradicted. It was ignored.
And the missing fact is the one a user needs most: nothing tells you that
part of your model is not training.

The rule takes the class with the most failures, ignored, although 21
failures cannot rank it above unsupported; a rule fixed in advance decides
anyway, which is what it is for. Its fix asks the generator to quote before
it answers: to copy the sentences of the passages that answer the
question, each with its passage number, and only then to write the answer.
A fact that has been copied out should be harder to leave out. The fix also
places the best passages at both ends of the context, the positions a
model reads most reliably, and allows 500 new tokens instead of 300, since
the quotes come first.

This is not the fix that `test` was read with. With the judge's earlier
8-bit build the unsupported class was the larger, by 11 failures to 10, so
the rule picked the fix for unsupported answers, and `test` was read once,
with that fix and with the reversed order. That fix checks the answer after
it is written. The answer is split into sentences, and the generator is
asked of each one whether the passages it cites support every fact in it,
a yes or no read off the two labels' probabilities, as the reranker reads
its score. A sentence the check rejects is dropped, and so is a sentence
that cites nothing, since nothing can support it; an answer with no
sentence left abstains. With the 4-bit judge the two classes changed
places, by one failure, which is within the noise of 21 failures. We do not
read `test` again for the new choice. The cell runs the fix for ignored
answers on `dev` only, and reports both fixes on `dev` and, where they were
read, the earlier fix and the reversed order on `test`.

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

def read_rule(recs):
    """The rule that test was read with (a dict), from the records of
    rag-gen-fix: the only records that carry one. None before test."""
    rules = {r for r in recs if isinstance(r, str)}
    assert len(rules) <= 1, 'test is read once'
    return json.loads(rules.pop()) if rules else None

if 'rag-gen-fix' in RERUN:
    assert FIX, 'the fix needs the judged base run (rag-judge)'
    assert BEST, 'test is read once, with the best stage: judge every stage'
    t = time.time()
    thr = tune_threshold() if LARGEST == 'false_answer' else None
    run = lambda name, items, **kw: d2l._rag_run_config(
        name, items, RET, KEEP_GEN, DEPTH, **kw)
    path = d2l.lm_registered(ARTIFACTS['rag-gen-fix'][0])
    old = d2l._rag_records([path]).to_dict('records') if path else []
    if read_rule([r.get('rule') for r in old]) is None:  # test, once
        recs = run(FIX, DEV, threshold=thr)
        base = run('base', TEST)  # the only reads of test in the chapter
        recs += base
        for name in [FIX] + ([BEST] if CONFIGS[BEST] != CONFIGS[FIX]
                             else []):
            recs += run(name, TEST, budgets=context_tokens(base) if
                        CONFIGS[name]['budget'] == 'base' else None,
                        threshold=thr if name == FIX else None)
        rule = json.dumps(dict(largest=LARGEST, fix=FIX, best=BEST,
                               threshold=thr))  # what test was read with
        recs = [dict(r, rule=rule) for r in recs]
    else:  # test was read with an earlier rule: the new fix on dev only
        old = [r for r in old if r['config'] != FIX or r['split'] != 'dev']
        recs = old + [dict(r, rule=None) for r in run(FIX, DEV,
                                                      threshold=thr)]
    d2l._rag_save_gens('rag-gen-fix', recs)
    GENS = d2l._rag_load_gens(RERUN)
    print(f'rag-gen-fix regenerated in {(time.time() - t) / 60:.1f} min, '
          f'{d2l.lm_peak()}')
RULE = read_rule(GENS.rule) if 'rule' in GENS else None
READ = []  # the configurations read on test, besides the baseline
if RULE:
    READ = [RULE['fix']] + ([RULE['best']] if RULE['best'] != RULE['fix']
                            else [])
    print(f'test was read once, with the fix {RULE["fix"]} and the stage '
          f'{RULE["best"]}' + ('' if RULE['threshold'] is None else
                              f', abstaining below a reranker score of '
                              f'{RULE["threshold"]:.3f} (tuned on train)'))
    if FIX != RULE['fix']:  # the verdicts changed after test was read
        print(f'with the verdicts as they stand now the rule picks {FIX}; '
              'it is run on dev only,\n  and test is not read again')
    if BEST != RULE['best']:  # stage deltas within noise: a tie can move
        print(f'with the verdicts as they stand now the largest dev gain is '
              f'{BEST};\n  test was read once, with {RULE["best"]}, and is '
              'not read again')
    BEST = RULE['best']  # the stage that test was read with

FIXES = [FIX] + [c for c in READ if c != FIX and CONFIGS[c] != CONFIGS[
    BEST]]  # the fix of the rule now, then the one test was read with
if FIX and ready('base', FIX):
    rows = []
    for split, arms in (('dev', FIXES + [BEST] * bool(BEST)),
                        ('test', READ)):
        for c in arms:
            if not ready('base', c, split=split):
                continue
            a, b = scores(c, split), scores('base', split)
            rows.append({'split': split, 'configuration': c,
                         'accuracy - base': paired(a, b, 'accuracy'),
                         'cite precision - base': paired(
                             a, b, 'cite_precision')})
    print(pd.DataFrame(rows).to_string(index=False))
    arms = ['base'] + [BEST] * bool(BEST) + FIXES
    print('accuracy:\n' + pd.DataFrame({split: {c: boot(
        scores(c, split).accuracy, cluster=scores(c, split).cluster)
        for c in arms if (split == 'dev' or c in ['base'] + READ)
        and ready(c, split=split)} for split in ('dev', 'test')}
                                       ).to_string())
    b = scores('base').set_index('qid')
    AFTER = {}
    for fx in FIXES:
        if not ready(fx):
            continue
        f, AFTER[fx] = scores(fx).set_index('qid'), classes(fx)
        lost = b.index[b.answerable & (b.accuracy == 1)
                       & (f.accuracy[b.index] == 0)]
        print(f'\n{fx}: {len(lost)} answerable dev questions right in base, '
              'wrong with the fix\n(by class: ' + ', '.join(
                  f'{c} {n}' for c, n in AFTER[fx].set_index('qid').cls[
                      lost].value_counts().items()) + '); the first:')
        if not len(lost):
            continue
        q = min(lost)  # the first answerable dev question the fix lost
        d2l.print_wrapped(f'{q}: {ITEMS[q]["question"]}', width=96)
        for name, sc in (('base', b), (fx, f)):
            print(textwrap.fill(f'{name}: {sc.answer[q][:300].rstrip()} ...',
                                96, initial_indent='  ',
                                subsequent_indent='    '))
    print(pd.DataFrame({'base': BC.cls.value_counts(), **{
        fx: d.cls.value_counts() for fx, d in AFTER.items()}})
          .reindex(CLASSES + ['correct', 'unjudged']).fillna(0).astype(int)
          .to_string())
```

Neither fix worked. The fix for ignored answers lowered `dev` accuracy
from 0.91 to 0.83, by 0.08 with an interval from $-0.14$ to $-0.04$, and
the class it aimed at grew instead of shrinking, from 8 ignored answers to
17. Copying sentences out did not make the generator state the fact it
needed. The larger damage came from an unexpected side: the baseline
declines 19 of the 20 unanswerable questions, the fix only 7, so false
answers rose from 1 to 13. Asked for quotes, the generator found sentences
to quote in passages near the question and answered from them, or
explained in its own words that the passages fall short, without the
abstention sentence that our grading counts as declining. The fix lost 16
answerable questions that the baseline got right, 11 of them to the
ignored class itself.

The fix for unsupported answers did worse. It lowered accuracy from 0.91
to 0.77 on `dev` and from 0.86 to 0.78 on `test`, by 0.15 and 0.09, and
both paired intervals exclude zero. Yet it did what it was built to do:
citation precision rose, by 0.02 on `dev` and 0.04 on `test`, with
intervals above zero. The answers it kept cited their passages better, and
fewer of them were right. It lost 36 answerable `dev` questions that the
baseline got right, and the class it aimed at hardly moved, from 7
unsupported answers to 6.

Where did the 36 go? Fourteen became false abstentions: nothing of the
answer was left, and an empty answer abstains. Twenty-one became ignored:
what was left was faithful to the passages and lacked a key fact. The check
is only half of the fix. The other half drops every sentence that cites
nothing, unread, since no passage can support it. That half assumes a
generator that cites what it states, and ours does not always do so. Told
to cite its passages, it may state the answer in one sentence and cite the
passage in the next.

The first question it lost, which the cell prints, is the `load_array`
question of the chapter's introduction. The baseline's answer said in
three sentences, citing none of them, that `is_train` is passed to the
`DataLoader` as `shuffle`; a fourth sentence cited the passage for all of
them. The fix kept the fourth sentence alone: "This behavior is defined in
the function signature and implementation within the provided code [1]."
It is supported, and it answers nothing. Exercise
:exref:`ex_lm_rag-tolerant` asks how many of the 36 were lost to uncited
sentences alone.

Both fixes were chosen from `dev` failures, so `dev` could have flattered
them; they lost there anyway, and for the one read on `test`, `test`
agrees. Nor do we now search for a better fix, such as one that keeps
uncited sentences: one found by looking at these results would need a
fresh test set to be believed. Of the results in this section, this is the
one to remember. Two plausible repairs, each written down before any
failure was counted and applied by rule, made the system worse, and
nothing in the failures they aimed at warned us. Whether a fix works
depends on its mechanism and on the habits of the model it acts on, and
only a run tests the two together.

The stage change fared no better than noise. On `test` the reversed order
changed accuracy by +0.004, with an interval from $-0.03$ to +0.04, and on
`dev`, with the 4-bit judge's verdicts, by $-0.008$. The earlier verdicts
ranked it first among the stage changes and the new ones rank the
6,000-token budget first; `test` finds nothing either way. A rule that
picks the largest of near-equal numbers picks noise. We do not read `test`
again for any other change: a test set read again after every change made
on `dev` becomes development data.

The plot shows these results at a glance. On the left, both fixes move
failures into other classes: the quoting fix into ignored and false
answers, the sentence check into ignored and false abstentions. On the
right are the accuracy and citation precision on `dev` of the closed book
(the arm without passages), the baseline, the reversed order and the two
fixes. The crosses are `test`, read once, for the baseline, the reversed
order and the fix for unsupported answers: 0.86, 0.87 and 0.78. The last
two lie inside their `dev` intervals, and the baseline's lies just below
its own, which runs from 0.87 to 0.95. Accuracies of two independent sets
of 240 questions differ by that much from noise alone; the differences
from the baseline, which pair each question with itself, agree across the
two splits. Every arm is compared with the baseline, not with the others.

```{.python .input #rag-failure-plot}
%%tab pytorch
if FIX and ready('closed', 'base', FIX):
    d2l.use_svg_display()
    steps = ['closed', 'base'] + [BEST] * bool(BEST) + FIXES
    dev = {c: scores(c) for c in steps}
    test = {c: scores(c, 'test') for c in steps[1:]  # read once, if at all
            if c in ['base'] + READ and ready(c, split='test')}
    series, waterfall, marks = {}, {}, {'test': ('x', {})}
    for name, df in [('base', BC)] + list(AFTER.items()):
        sh = class_shares(df.cls[df.cls != 'unjudged'])  # judged only
        series[name] = (sh.share, sh.lo, sh.hi)
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
    with d2l.plt.rc_context({'font.size': 14}):  # legible on a slide
        _, axes = d2l.plt.subplots(1, 2, figsize=(10, 4.4),
                                   layout='constrained')
        d2l.plot_intervals([c.replace('_', ' ') for c in CLASSES], series,
                           'failure class', 'share of dev questions',
                           axes=axes[0])
        axes[0].set_ylim(top=1.7 * axes[0].get_ylim()[1])  # room for
        axes[0].legend(loc='upper right')  # the legend above the bars
        d2l.plot_intervals(steps, waterfall, 'configuration', 'dev score',
                           axes=axes[1], marks=marks)
```

Beyond these fixes lies a large literature that this section does not
measure, from transforming the query and retrieving again during generation
to retrieval learned with reinforcement learning, compressed contexts and
structured indexes (:numref:`sec_lm_encoders_fr`). Each of these methods
can be judged with the procedure of this section.

## Marginalizing over Retrieved Documents
:label:`subsec_lm_rag_marginal`

A dense retriever is trained on pairs of a question and a relevant passage,
with a contrastive loss (:numref:`sec_lm_text_embeddings`). Nobody tells it
whether the generator can use the passages it finds. Could the answers
alone teach it that? Our pipeline cannot even ask, because it treats
retrieval as preprocessing: passages are chosen, concatenated and read. The
model that gave RAG its name :cite:`Lewis.Perez.Piktus.ea.2020` treats the
retrieved document as a hidden variable instead, which makes the system a
probabilistic model whose query encoder and generator can be trained
together. This is the only proof in our two sections on RAG. It is short,
and its last line answers the question just asked.

Let $x$ be the question, $y = (y_1, \ldots, y_n)$ the answer and $z$ a
document. The retriever gives each document a score $s_z$ and the
probability $p(z \mid x) = \mathrm{softmax}(\mathbf{s})_z$, and the
generator reads one document at a time. *RAG-Sequence* draws one document
for the whole answer:

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
largest $p(z \mid x)$, the retrieved set $\mathcal{K}$. What does that drop?
Write $L_z = p(y \mid x, z)$, $p_S^{(k)}$ for the sum restricted to
$\mathcal{K}$, and $m_k = \sum_{z \in \mathcal{K}} p(z \mid x)$ for the
retriever's mass on $\mathcal{K}$. (Implementations usually renormalize
$p(z \mid x)$ over $\mathcal{K}$; Exercise :exref:`ex_lm_rag-topk` shows
that the bound below survives.) The dropped terms are nonnegative and each
$L_z$ is at most $\max_z L_z$, so

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

Let us read the result. Since the posterior is $p(z \mid x)\, L_z / p_S$,
the gradient is $p(z \mid x)(L_z / p_S - 1)$: a document's score rises
when it explains the answer better than the documents do on average,
weighted by the retriever's probabilities, and falls when it explains it
worse. No label says which document was relevant. That is a great deal to
learn from answers alone. But it is
learned through the generator's eyes. If the generator ignores every
document, $L_z$ is the same for all, the posterior equals the prior, and
the retriever learns nothing. If it ignores one relevant passage while
others explain the answer well, the ignored passage falls below the
average, and its score falls. So the answer to our question is a
qualified yes: the retriever learns which passages the generator uses,
and those are not always the relevant ones.

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

The gradient matches posterior minus prior to rounding error, below
$10^{-16}$; every truncation error lies at or below its bound (at $k = 5$,
0.028 against 0.037); and the case by hand gives 0.140 and 0.2552.

## Cost and Long Context

Our pipeline concatenates instead of marginalizing, and every passage adds
prompt tokens that the generator must read. What does each stage cost in
time? The last cell reports the tokens and seconds per answer of each
configuration, then times retrieval and reranking one question at a time.

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
chunks is not: about half a second, nearly half the time the generator
spends on a baseline answer. It is easy to forget that the reranker, a
model of 0.6 billion parameters, reads 50 pairs of question and chunk for
every question. The baseline does without it, and the stage experiment gave
no reason to add it.

The prompt is the context plus the instructions and the question, about
240 tokens more. Generation time barely moves between 1,500 and 2,100
prompt tokens, about 1.1 to 1.3 s per answer. At 6,300 tokens it doubles, to
2.6 s, partly because such long prompts run in batches of four instead of
eight. The rewrite with reranking takes about 1.9 s per answer, since the
generator first writes the query and the reranker then reads 50 chunks.

That leaves the question that longer context windows keep raising. If the
generator can read 6,000 tokens, why not whole chapters, and no retrieval at
all? On this set three times the baseline's context gained nothing
measurable, and the oracle of :numref:`sec_lm_rag` shows that the
generator, given only the right passages, still gets about one question in
fourteen wrong. More text does not cure a generator that ignores what it
already has. The agents of :numref:`chap_lm_agents` take a third path
and retrieve only when they decide that they need to. Whether whole chapters, at equal cost, would beat
retrieval on the questions that need several passages remains open.
Exercise :exref:`ex_lm_rag-long-context` sets up the experiment.

## Summary

One accuracy number does not say what to fix in a retrieval-augmented
generator, and a favorite fix is a guess; this section let measurements
choose. Changing one stage at a time, with paired intervals on the same
questions, raised accuracy measurably nowhere: retrieval had no room, as the
share of hits predicted, and the generator did not read better for another
order or for more text. Fixed windows and a smaller budget lowered it.
Sorting the wrong answers by the first stage that lost them split them
mostly between answers that go beyond their passages and answers that miss
a fact their own context holds, and 21 failures could not say which class
is larger. Both fixes written down in advance lowered accuracy. Quoting
before answering, for the ignored class, lost 0.08 on `dev` and broke the
abstentions on unanswerable questions. Checking each sentence against its
citations, for the unsupported class and the one fix read on `test`, lost
0.15 on `dev` and 0.09 on `test`, by removing sentences that carried the
answer, among them uncited ones that the check never read: a change aimed
at one class can create others. The original RAG model marginalizes over
retrieved documents, and its retriever learns the posterior minus the
prior: the passages the generator uses, which are not always the ones that
are relevant. Whether whole chapters in a long context can replace
retrieval remains open.

## Exercises

1. **The truncation error and recall.** Start from :eqref:`eq_lm_rag_topk`.
    1. Show that the bound is tight: give scores and likelihoods for which
       $p_S - p_S^{(k)} = (1 - m_k) \max_z L_z$.
    1. Suppose exactly one document $z^*$ answers each question, and the
       retriever is calibrated: $p(z \mid x)$ is the probability that $z$ is
       $z^*$. Show that the expectation of $1 - m_k$ over questions equals
       the probability that $z^*$ is not in the top $k$, one minus
       Recall@$k$.
    1. The baseline's five chunks hold a relevant one for 213 of the 220
       answerable questions, so Recall@5 $\approx 0.97$. What would the
       bound promise for RAG-Sequence over five documents if the retriever
       were calibrated? Why is a softmax over BM25 or dense scores rarely
       calibrated in this sense, and how would you calibrate it on `train`?
    1. Implementations usually renormalize the retriever's probabilities
       over $\mathcal{K}$ and use $p_S^{(k)} / m_k$. Show that the error
       $p_S - p_S^{(k)} / m_k$ can take either sign, and that its absolute
       value obeys the same bound.
   :exlabel:`ex_lm_rag-topk`

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

1. [code] **What the check removed.** The fix of the unsupported class
   drops every sentence that cites nothing and every cited sentence that
   its check rejects. Use the recorded answers in `GENS`; no generation and
   no judge are needed.
    1. For each of the 36 answerable `dev` questions that the fix lost,
       compare the baseline's answer with the fix's, sentence by sentence.
       How many lost only sentences that cited nothing?
    1. Which question types have the most uncited sentences in the
       baseline's answers? Why might a generator leave the sentence that
       states the answer uncited and cite the one that explains it?
    1. Write down, before running anything, a variant of the fix that
       keeps uncited sentences, and predict its effect. Why may its result
       not be read on `test`, and what would it take to believe it?
   :exlabel:`ex_lm_rag-tolerant`

1. [code] **Splitting the fix.** The fix written down for the ignored
   class, which the rule picks with the 4-bit judge, changes three settings
   at once:
   `RAG_FIXES['ignored']`.
    1. Define three arms that each change one of them: the quoting prompt
       alone (`dict(d2l.RAG_BASE, prompt='quote')`), the order with the best
       passages at both ends alone, and room for 500 new tokens alone.
       Before you run them, predict which carries the loss.
    1. Generate the three arms on `dev`, judge them with the local judge, and
       report each paired difference with the baseline and each arm's
       failure classes. Without a 24 GB GPU for the judge, compare the
       abstentions on the unanswerable questions, which need no judge.
    1. Compare with the bundle `fix-ignored`, recorded in `GENS`. Do the
       three effects add up to the bundle's? What does the answer say about writing a fix down
       in advance as a bundle, and why may these results not be read on
       `test`?
   :exlabel:`ex_lm_rag-split-fix`

1. **Failure counts from two graders.** The labeler called 4 of the 21
   failures correct.
    1. From the table of rule against labeler printed in
       :numref:`subsec_lm_rag_failures`, recount the classes with those four
       answers taken as correct. Does the largest class change?
    1. With the bootstrap of the failure cell, or with the normal
       approximation to a binomial share, compute an interval for the share
       of the largest class among the remaining failures. Can the classes
       now be ranked?
    1. What would it take to decide which grader is right on the four, and
       why would asking a third model not settle it?
   :exlabel:`ex_lm_rag-two-graders`

1. [code] **Coverage of the whole span.** Coverage counts a question as
   covered when its chunks hold half of a gold span.
    1. Write a variant of `d2l.cov_at_budget` that requires a whole gold
       span, and recompute the coverage of section chunks and fixed windows
       at 1,000, 2,000 and 3,000 tokens. (Without a GPU, use BM25 rankings
       for both.)
    1. Does the gap between windows and sections grow, and does it now come
       closer to the windows' loss in accuracy?
    1. Gold spans can cross a section boundary too. How often do they, and
       what would you change in the chunker?
   :exlabel:`ex_lm_rag-whole-span`

1. [code] **A smaller generator.** Replace Qwen3.5-9B by Qwen3.5-4B of the
   same family and keep everything else fixed.
    1. Before you run it, predict which failure class grows most and whether
       the gain of the oracle context over the baseline becomes measurable.
    1. Regenerate the `base`, `closed` and `oracle` arms, judge them, and
       report the class shares with their intervals and the paired
       differences with the larger generator. Without a 24 GB GPU for the
       judge, compare what needs no judge: abstention rates, false
       abstentions and citation precision.
   :exlabel:`ex_lm_rag-smaller`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §19.6]{.kicker}

Failures and Fixes in Retrieval-Augmented Generation<br>**one change at a time · failures by stage · a fix chosen in advance · marginalizing over documents · cost**
:::
:::

::: {.slide title="21 Answers Are Wrong, Though Retrieval Finds the Passage for 213 of 220"}
[The problem]{.kicker}

- Baseline: 0.91 of 240 `dev` questions about this book.
- 15 of its 20 errors on answerable questions had the passage in context.
- Favorite repairs: smaller chunks, query rewrites, more context.
- `dev` (240 questions) chooses; `test` (240 more) is read once.

::: {.d2l-note .warn}
Which repair can help such a system, and how would we know?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- predict a perfect retriever's gain from the share of hits;
- change one stage at a time, read a paired interval;
- blame each wrong answer on the first stage that lost it;
- test a fix written down in advance, once;
- derive RAG-Sequence, RAG-Token and the retriever's gradient.

**Running example:** 240 `dev` questions; Qwen3.5-9B answers,
Qwen3.8-27B judges.
:::

::: {.slide title="Locating Failures Comes Before Fixing Them"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| movement | section |
|---|---|
| vectors | Word Embeddings |
| | Pretrained Encoders |
| | Text Embeddings |
| finding | Sparse, Dense and Approximate Retrieval |
| using what is found | Retrieval-Augmented Generation |
| | **Failures and Fixes in RAG** |
:::

::: {.col .narrow}
- **Uses:** the question set, judge, baseline and paired intervals of the
  previous section.
- **Skips:** training retriever and generator together (derived, not run);
  retrieval inside agent loops.
:::
:::
:::

::: {.slide title="With 213 of 220 Hits, a Perfect Retriever Could Add About 0.02"}
[A prediction before any run]{.kicker}

$$\text{accuracy} = h\, a_1 + (1 - h)\, a_0$$

- *Hit*: the context holds a relevant chunk. $h$: share of hits, 213/220.
- $a_1 = 0.93$ on hits, $a_0 = 0.29$ on misses.
- Perfect retriever, $a_1$ unchanged:
  $(1 - h)(a_1 - a_0) \approx 0.032 \times 0.64 \approx 0.02$.

::: {.d2l-note}
Inside the $\pm 0.03$ that 240 questions resolve: retrieval has no room
here.
:::
:::

::: {.slide title="Section Chunks Cover More Than Fixed Windows at Small Budgets"}
[Chunking at equal tokens]{.kicker}

| budget $T$, 220 questions | sections | fixed 256/64 | fixed − sections |
|---|---|---|---|
| 1,000 tokens | 0.91 | 0.86 | $-0.05$ [$-0.09$, $-0.01$] |
| 2,000 tokens | 0.97 | 0.95 | $-0.02$ [$-0.05$, +0.01] |
| 3,000 tokens | 0.99 | 0.97 | $-0.01$ [$-0.04$, +0.01] |

`cov@T`: the first chunks, up to $T$ tokens, hold half a gold span.

::: {.d2l-note}
Compare chunkings at equal tokens, not equal $k$: five sections hold more
text than five windows.
:::
:::

::: {.slide title="No Stage Change Raised Accuracy; Two Lowered It"}
[One change at a time]{.kicker}

| change, 240 `dev` questions | context tokens | accuracy − baseline | interval excludes 0 |
|---|---|---|---|
| baseline: 5 section chunks, no reranker | 1,700 | | |
| fixed windows | 1,600 | $-0.09$ | yes |
| rewrite + rerank top 50 (two changes) | 1,700 | $-0.01$ | no |
| reverse order | 1,700 | $-0.01$ | no |
| budget 1,500, no 5-chunk limit | 1,300 | $-0.06$ | yes |
| budget 6,000, no 5-chunk limit | 5,800 | +0.01 | no |

Baseline ranking: BM25 and dense, fused by reciprocal ranks.

::: {.d2l-note}
As predicted for retrieval; and three times the text did not help the
generator either.
:::
:::

::: {.slide title="The Cut, Not the Token Count, Cost Fixed Windows Their Accuracy"}
[A control]{.kicker}

| arm | context tokens | accuracy − baseline |
|---|---|---|
| fixed windows, up to the baseline's tokens | 1,600 | $-0.09$ |
| control: windows just past the baseline's tokens | 1,800 | $-0.05$ |

- Coverage near these budgets trails by 0.02 to 0.05: worth at most
  0.01 to 0.03 of accuracy.
- Coverage counts half a span; a window can cut off the key fact.

::: {.d2l-note}
Chunks that follow the text's structure give better answers.
:::
:::

::: {.slide title="Each Wrong Answer Is Blamed on the First Stage That Lost It"}
[Failures by stage]{.kicker}

| class | condition, checked in this order | of 21 | answer cites a relevant chunk |
|---|---|---|---|
| false answer | unanswerable, yet answered | 1 | 0 |
| not retrieved | no relevant chunk in the top 50 | 0 | 0 |
| ranked low | in the top 50, not in the context | 5 | 0 |
| false abstention | in the context, answer abstains | 0 | 0 |
| ignored | in the context, faithful, misses a key fact | 8 | 5 |
| unsupported | in the context, not faithful | 7 | 7 |

::: {.d2l-note}
Ignored 0.38 [0.19, 0.57], unsupported 0.33 [0.14, 0.52]: 21 failures
cannot rank the two.
:::
:::

::: {.slide title="Fixes Are Written Down Before Any Failure Is Counted"}
[Pre-registration]{.kicker}

| class | its fix |
|---|---|
| not retrieved | rewritten query as a third ranking |
| ranked low | rerank the top 100, pass 8 chunks |
| ignored | quote, then answer; best passages at both ends; 500 new tokens |
| unsupported | drop sentences their citations do not support |
| false answer | abstain below a reranker score tuned on `train` |
| false abstention | abstention rule after the passages |

::: {.d2l-note}
A rule picks the largest class and carries the largest stage gain to
`test`, read once: with the earlier 8-bit judge, unsupported and reverse
order.
:::
:::

::: {.slide title="Check: The Largest Stage Gain on Dev Is Two Verdicts. What Will Test Show?"}
[Check]{.kicker}

Reverse order led on `dev` by two verdicts of 240 under the 8-bit judge's
verdicts. The rule carried it to `test`. Under the 4-bit judge's it is
$-0.008$ and the 6,000-token budget leads instead.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** +0.004, interval $-0.03$ to +0.04: noise again. Picking the
largest of near-equal numbers picks noise.
:::
:::
:::

::: {.slide title="A Second Labeler Confirms Most Ignored Answers and Few Unsupported Ones"}
[A second labeler]{.kicker}

Claude Opus 5.5 labeled the cards of the 21 failures.

| the labeler | failures |
|---|---|
| agrees with the rule | 12 |
| calls the answer correct | 4 |
| blames another stage | 5 |

Of the 4: in 3 the judge insists on a key fact the labeler finds stated; 1
unanswerable question was declined in other words. The labeler confirms 6
of 8 ignored answers, but only 2 of 7 unsupported ones.
:::

::: {.slide title="The Answering Passage Was Ranked First, and the Answer Ignored It"}
[One failure]{.kicker}

**Q.** What happens to PyTorch modules kept in a plain Python list?

- Passage [1] ends: "Modules in a plain list stay unregistered, without any
  warning."
- The answer cites [2] and [3] and never says that nothing warns.

::: {.d2l-note}
Neither missing nor contradicted. Ignored, say the rule and the labeler;
and the missing fact is the one a user needs.
:::
:::

::: {.slide title="Both Fixes Chosen in Advance Lowered Accuracy"}
[Quote first; check each cited sentence]{.kicker}

| accuracy − baseline | quote, then answer (ignored) | sentence check (unsupported) |
|---|---|---|
| `dev` | $-0.08$ [$-0.14$, $-0.04$] | $-0.15$ [$-0.19$, $-0.10$] |
| `test`, read once | not read | $-0.09$ [$-0.14$, $-0.04$] |

- Baseline: 0.91 on `dev`, 0.86 on `test`.
- The sentence check raised citation precision (+0.02 `dev`, +0.04
  `test`): it did what it was built for.
- `test` was read once, with the earlier judge's choice; the rule's new
  choice runs on `dev` only.
:::

::: {.slide title="Each Fix Filled Classes Other Than the One It Aimed At"}
[Where the failures went]{.kicker}

| `dev` answers | baseline | quote first | sentence check |
|---|---|---|---|
| correct | 219 | 199 | 184 |
| ranked low | 5 | 6 | 5 |
| ignored | 8 | 17 | 31 |
| unsupported | 7 | 5 | 6 |
| false answer | 1 | 13 | 0 |
| false abstention | 0 | 0 | 14 |

- Quote first: the aimed-at class doubled, and 13 of 20 unanswerable
  questions were answered.
- Sentence check: uncited sentences dropped unread; empty answers abstain.

::: {.d2l-note .rule}
Test a fix's mechanism with the habits of the model it acts on. Never
search for a better fix on the same data.
:::
:::

::: {.slide title="On Test, Every Difference from the Baseline Agrees with Dev"}
[The results at a glance]{.kicker}

@!rag-failure-plot

Left: share of `dev` answers per class. Right: `dev` scores with 95%
intervals; crosses are `test`, read once. Closed: no passages.
:::

::: {.slide title="The Original RAG Treats the Retrieved Document as a Hidden Variable"}
[From pipeline to model]{.kicker}

- Our pipeline: retrieve, concatenate, read. The retriever is never told
  whether the generator could use a passage.
- RAG: the generator reads one document $z$ at a time; the answer's
  probability sums over documents, weighted by the retriever's
  $p(z \mid x)$.

::: {.d2l-note}
Then the answers alone give the retriever a gradient.
:::
:::

::: {.slide title="Check: Which Is Larger, RAG-Sequence or RAG-Token?"}
[Two marginalizations]{.kicker}

Question $x$, answer $y_1 \ldots y_n$, document $z$ with retriever
probability $p(z \mid x)$:

$$p_S = \sum_z p(z \mid x) \prod_t p(y_t \mid x, z, y_{<t})$$

$$p_T = \prod_t \sum_z p(z \mid x)\, p(y_t \mid x, z, y_{<t})$$

Two documents, $p(z \mid x) = (0.6, 0.4)$; token probabilities
$(0.9, 0.2)$ and $(0.1, 0.8)$.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $p_S = 0.140$, $p_T = 0.2552$: each document supports half the
answer, and only RAG-Token combines the halves.
:::
:::
:::

::: {.slide title="Keeping the Top k Documents Loses at Most the Mass Outside Them"}
[Truncation]{.kicker}

$L_z = p(y \mid x, z)$; $\mathcal{K}$: the $k$ documents of largest
$p(z \mid x)$; $m_k$: their total probability.

$$0 \le p_S - p_S^{(k)} \le (1 - m_k) \max_z L_z$$

- Eight random documents, $k = 5$: lost 0.028, bound 0.037.

::: {.d2l-note}
A peaked retriever loses little; one that spreads its mass can lose much,
however good its ranking.
:::
:::

::: {.slide title="The Retriever Learns What the Generator Uses, Not What Is Relevant"}
[Training signal]{.kicker}

With $p(z \mid x) = \mathrm{softmax}(\mathbf{s})_z$:

$$\frac{\partial \log p_S}{\partial s_z} = \underbrace{\frac{p(z \mid x) L_z}{p_S}}_{\text{posterior}} - \underbrace{p(z \mid x)}_{\text{prior}}$$

- Score rises if $L_z > p_S$, the average; falls otherwise. No relevance
  label needed. Checked against autograd to within $10^{-16}$.

::: {.d2l-note}
Generator ignores every document: no signal. Ignores one relevant passage
that others outdo: that passage is pushed down.
:::
:::

::: {.slide title="First-Stage Retrieval Costs Milliseconds; Reranking and Long Prompts Cost Seconds"}
[Cost]{.kicker}

| per question | prompt tokens | seconds |
|---|---|---|
| retrieval | | 0.03 |
| rerank 50 chunks (not in the baseline) | | 0.5 |
| closed book | 90 | 0.8 |
| baseline | 1,900 | 1.2 |
| budget 6,000 | 6,300 | 2.6 |
| fix: check each sentence (answer only) | 1,900 | 1.2 |

Qwen3.5-9B on one RTX 4090, batches of 8 (4 at 6,000 tokens); retrieval
and reranking one question at a time.
:::

::: {.slide title="Check: With 70% Hits, What Could a Perfect Retriever Add?"}
[Check]{.kicker}

Another collection: a relevant chunk reaches the context for 70% of the
questions; accuracy 0.9 on hits, 0.2 on misses.

$$\text{accuracy} = h\, a_1 + (1 - h)\, a_0$$

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $0.3 \times 0.7 = 0.21$, ten times our 0.02. There, fix
retrieval first.
:::
:::
:::

::: {.slide title="Recap"}
[Recap]{.kicker}

- 213 of 220 hits: about 0.02 to gain; no stage change helped measurably.
- Fixed windows $-0.09$, and $-0.05$ even with more tokens than the
  baseline: the cut costs accuracy. Budget 1,500: $-0.06$.
- Largest class: ignored, 8 of 21, beside unsupported, 7: too few to rank.
- Both fixes chosen in advance lost: quoting $-0.08$ on `dev`; the
  sentence check $-0.15$ on `dev` and $-0.09$ on `test`.
- Gradient = posterior − prior: the retriever learns what the generator
  uses.
:::

::: {.slide title="Next: Whole Chapters in the Context, or Retrieval on Demand"}
[Next]{.kicker}

Three times the context gained nothing measurable at twice the time.
Would whole chapters in a 32,000-token context beat retrieval on
questions that need several passages? Agents take a third path: they
retrieve when they decide they need to.
:::
