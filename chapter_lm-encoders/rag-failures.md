# Failures and Fixes in Retrieval-Augmented Generation
:label:`sec_lm_rag_fixes`

Suppose you have built the pipeline of :numref:`sec_lm_rag` and are looking
at the 26 `dev` questions it gets wrong. The natural impulse is to start
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
class, written down before any failure was counted, on the 240 `test`
questions, read once. None of this is new: medicine has run its clinical
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
at 1,000 tokens against 0.84 for windows, a gap of 0.07 whose interval
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

No change raised accuracy measurably. The largest gains, from reversing the
order and from the 6,000-token budget, are about +0.01, with intervals that
reach down to $-0.02$ and $-0.03$. On the retrieval side this is what the
mixture predicted. On the generator's side it is a finding: the generator
did not read measurably better for a different order or for three times the
text.

The two losses teach more. Fixed windows lost 0.08. They had a little less
context than the baseline, about 1,600 tokens against 1,700, but the
control rules the token count out: with about 1,800 tokens, windows still
lost 0.05. Coverage accounts for at most about half of the loss. At
budgets near the windows' own, their coverage trails by 0.03 to 0.07. A
question that drops out of coverage falls at worst from the accuracy on
hits to the accuracy on misses, about 0.6 lower, so such a gap could cost
at most about 0.02 to 0.04. The rest is the cut itself, and coverage is
blind to it: it
counts half a gold span as found, and a window can hold that half and cut
off the key fact in the other (Exercise :exref:`ex_lm_rag-whole-span`).
Chunks that follow the document's structure keep a paragraph whole, and
they gave better answers.

Less context costs accuracy where coverage falls. The 1,500-token budget,
about 1,300 context tokens, lost 0.03, and coverage drops from 0.97 at
2,000 tokens to 0.91 at 1,000. More context, by contrast, did nothing for
accuracy: the 6,000-token budget, about 5,800 context tokens, gained
nothing measurable and doubled the time per answer.

A rule fixed in advance carries the stage change with the largest `dev`
gain to `test`, read once, to see whether a gain this small survives fresh
questions. The cell's last line names reversing the order, which ties with
the 6,000-token budget, one verdict ahead of the rewrite with reranking; a
tie goes to the change listed first. Yet `test` was read with the rewrite.
When the rule ran, a few answers still had no verdict, because the
judge's reply had been cut off or garbled, and those answers counted as
wrong. :numref:`sec_lm_rag` judged
each of them once more, with twice the room, and that moved the rewrite
with reranking from a three-way tie at the top to one verdict behind. We
read `test` with the fix and with the rewrite, at the end of the next
subsection.

## Failures by Stage
:label:`subsec_lm_rag_failures`

No change helped on average. Perhaps the 26 wrong answers fail in different
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

None of the 26 failures happened at first-stage retrieval, which is what
the hit rate led us to expect. Ignored is the largest class, 12 failures or
0.46 of them, but its interval, [0.27, 0.65], overlaps that of unsupported,
6 failures and [0.08, 0.38]. Twenty-six failures are a small sample to
divide six ways, and they cannot say which of the two classes is larger.

The last column shows what the two largest classes have in common. Ten of
the 12 ignored answers cite a passage that holds the answer, and so do all
six unsupported ones. In these answers the generator found the right
passage and even pointed to it. Then it left out a key fact that the
passages state, or added a claim that they do not support.

How far can we trust the classes? The cards were drawn from the failures
of the first verdicts, so there are 31 of them: the 26 failures above, and
five answers that had no verdict then and that the judge, asked again,
called correct. The labeler calls those five correct too. The printed
agreement, 0.58, counts all 31 cards. The cell's last two lines count the
26 failures alone: the labeler agrees with the rule on 13, half of them,
calls 9 correct and blames another stage on 4. Those nine are answers
that the labeler accepts and our grading does not. In seven the
judge insists on a key fact that the labeler lets pass. In two,
unanswerable questions were declined in words other than the abstention
sentence of :numref:`sec_lm_rag`, "I cannot answer this from the provided
passages", which is the only reply our grading counts as declining.

Rule and labeler disagree about whether an answer is wrong more than twice
as often as about where it went wrong. Once two graders agree that an
answer is wrong, the stage is the easier question. One caveat matters for
what comes next. The class the rule is about to pick is also the most
disputed one: the labeler calls 4 of the 12 ignored answers correct.

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

The rule takes the class with the most failures, ignored, although 26
failures cannot rank it above unsupported; a rule fixed in advance decides
anyway, which is what it is for. Its fix asks the generator to quote the
sentences that support its answer, with their passage numbers, and only
then to answer. Quoting first should make the generator look at every
passage before it writes. The fix is one idea in three settings: the
quotes need room, so it allows 500 new tokens instead of 300, and it puts
the best passages at both ends of the context, which language models use
better than its middle :cite:`Liu.Lin.Hewitt.ea.2024`. Its result
therefore belongs to the bundle, not to quoting alone; Exercise
:exref:`ex_lm_rag-split-fix` takes the bundle apart. The cell applies the
fix to `dev` and then reads `test` once, with the fix and with the rewrite
and reranking.

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
    run = lambda name, items, **kw: d2l._rag_run_config(
        name, items, RET, KEEP_GEN, DEPTH, **kw)
    recs = run(FIX, DEV, threshold=thr)
    base = run('base', TEST)  # the only reads of test in the chapter
    recs += base
    for name in [FIX] + ([BEST] if CONFIGS[BEST] != CONFIGS[FIX] else []):
        recs += run(name, TEST, budgets=context_tokens(base) if
                    CONFIGS[name]['budget'] == 'base' else None,
                    threshold=thr if name == FIX else None)
    rule = json.dumps(dict(largest=LARGEST, fix=FIX, best=BEST,
                           threshold=thr))  # what test was read with
    d2l._rag_save_gens('rag-gen-fix', [dict(r, rule=rule) for r in recs])
    GENS = d2l._rag_load_gens(RERUN)
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
    arms = ['base'] + ([BEST] if BEST and BEST != FIX else []) + [FIX]
    print('accuracy:\n' + pd.DataFrame({split: {c: boot(
        scores(c, split).accuracy, cluster=scores(c, split).cluster)
        for c in arms if ready(c, split=split)} for split in ('dev', 'test')}
                                       ).to_string())
    b, f = (scores(c).set_index('qid') for c in ('base', FIX))
    lost = b.index[b.answerable & (b.accuracy == 1) & (f.accuracy[b.index] == 0)]
    q = min(lost)  # the first answerable dev question the fix lost
    after = classes(FIX)
    print(f'\n{len(lost)} answerable dev questions right in base, wrong '
          'with the fix\n(by class: ' + ', '.join(
              f'{c} {n}' for c, n in
              after.set_index('qid').cls[lost].value_counts().items())
          + '); the first:')
    d2l.print_wrapped(f'{q}: {ITEMS[q]["question"]}', width=96)
    for name, s in (('base', b), (FIX, f)):
        print(textwrap.fill(f'{name}: {s.answer[q][:300].rstrip()} ...', 96,
                            initial_indent='  ', subsequent_indent='    '))
    print(pd.DataFrame({'base': BC.cls.value_counts(),
                        FIX: after.cls.value_counts()})
          .reindex(CLASSES + ['correct', 'unjudged']).fillna(0).astype(int)
          .to_string())
```

The fix failed. It lowered accuracy from 0.89 to 0.81 on `dev` and from
0.87 to 0.79 on `test`, by 0.08 on both, and both paired intervals exclude
zero. On `dev` it got 20 fewer answers right,
and most of the loss came from the unanswerable questions: false answers
rose from 2 to 15, as the abstentions fell from 18 of 20 to 5. In about
half of the other 15 the model said in its own words that the passages do
not give the answer, instead of writing the abstention sentence, and the
rest answered anyway.

Is that a failure of the model or of our grading? The prompt asks for the
abstention sentence word for word, and any system that acts on it, by
showing a fallback for instance, breaks when the sentence is missing. The
quoting format broke that agreement between prompt and grader. A change
aimed at one class created another, and a fix's prompt and its scoring rule
have to be tested together.

Even without the false answers, the fix did not shrink the class it aimed
at: ignored answers rose from 12 to 19. The replies were not cut off; they
average about 220 new tokens, well below the limit of 500. The fix lost
13 answerable questions that the baseline got right, and 9 of them fall
into the class it aimed at: the fix's reply was faithful to the passages
and still left out a key fact. Read side by side, most of these pairs of
replies make the same points, the fix's more loosely, with a key fact of
the baseline's answer left out or blurred. Quoting first did not make the
answer that follows the quotes more complete.

The first of the 13, which the cell prints, fails in another way. The
question asks where the one-step sampler of the discrete diffusion
experiment puts its mass. Both replies
open with nearly the same sentence and state both key facts: where the
column of one mode crosses the row of another, a place where the data have
none. The baseline then explains the error from the passages of that
experiment. The fix's reply explains it with a sentence about another
sampler, the one-step flow of :numref:`sec_diffusion-gaussian-paths`,
which sends every start to the data mean. Both replies read the same five
passages, only in another order. The judge calls the fix's reply
unfaithful, and incorrect too: a sampler that puts every sample on one
point cannot also put mass at the crossings. Reordering the passages
changed which of them the reply drew on, and the quotes did not keep it to
the experiment in the question.

The `dev` result could have been optimistic, since the fix was chosen from
`dev` failures; it lost there anyway, and `test` agrees. Nor do we now
search for a better fix: one found by looking at these results would need a
fresh test set to be believed. A plausible repair, chosen by a careful
procedure before any failure was counted, made the system worse, and no
reading of the failures beforehand predicted it.

The stage change fared no better. On `test` the rewrite with reranking
changed accuracy by $-0.004$, with an interval from $-0.05$ to +0.04. A rule
that picks the largest of three near-equal numbers picks noise, and `test`
says so. We do not read `test` again for the change that leads now: a test
set read again after every change made on `dev` becomes development data.

The plot shows these results at a glance. On the left, the fix moves
failures into the ignored and false-answer classes. On the right are the
accuracy and citation precision on `dev` of the closed book (the arm
without passages), the baseline, the rewrite with reranking and the fix.
The crosses are `test`, read once. Accuracy there is 0.87 for the
baseline, 0.86 for the rewrite with reranking and 0.79 for the fix, and
every cross lies inside its `dev` interval: `test` agrees with `dev` for
every arm it was read with. Both candidates are compared with the
baseline, not with each other.

```{.python .input #rag-failure-plot}
%%tab pytorch
if FIX and ready('closed', 'base', FIX):
    d2l.use_svg_display()
    steps = ['closed', 'base'] + ([BEST] if BEST and BEST != FIX else []
                                  ) + [FIX]
    dev = {c: scores(c) for c in steps}
    test = {c: scores(c, 'test') for c in steps[1:] if ready(c, split='test')}
    series, waterfall, marks = {}, {}, {'test': ('x', {})}
    for name, df in (('base', BC), ('fix', after)):
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
chunks is not: about half a second, some 40% of the time the generator
spends on a baseline answer. It is easy to forget that the reranker, a
model of 0.6 billion parameters, reads 50 pairs of question and chunk for
every question. The baseline does without it, and the stage experiment gave
no reason to add it.

The prompt is the context plus the instructions and the question, about
240 tokens more. Generation time barely moves between 1,500 and 1,900
prompt tokens, about 1.1 to 1.3 s per answer; prompts of the baseline's own
length, batched differently, already span that range. At 6,300 tokens it
doubles, to 2.6 s, partly because such long prompts run in batches of four
instead of eight. Quoting before answering doubles the new tokens, to about 220,
and raises the time to about 2.1 s.

That leaves the question that longer context windows keep raising. If the
generator can read 6,000 tokens, why not whole chapters, and no retrieval at
all? On this set three times the baseline's context gained nothing
measurable, and the oracle of :numref:`sec_lm_rag` shows that the
generator, given only the right passages, still misses about one question
in twelve. More text does not cure a generator that ignores what it
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
Sorting the wrong answers by the first stage that lost them made the
largest class the answers that miss a fact their own context holds. The
fix for that class, written down in advance, lowered accuracy by 0.08 on
`dev` and on `test`, mostly by creating false answers: a change aimed at
one class can create another. The original RAG model marginalizes over
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

1. [code] **The fix under a tolerant grader.** Most of the fix's loss came
   from unanswerable questions, where any reply other than the abstention
   sentence counts as an answer. Use the recorded answers in `GENS`; no
   generation and no judge are needed.
    1. Read the replies of `fix-ignored` to the 20 unanswerable `dev`
       questions and write a tolerant rule that also counts a reply that
       declines in other words as an abstention. Check the rule against
       your reading of every reply.
    1. Re-score `base` and `fix-ignored` on these 20 questions with the
       tolerant rule. Does the fix still lose there?
    1. Apply the rule to the baseline's answers to the answerable questions.
       How many replies that the judge graded would it now call
       abstentions, and what does that say about loosening a grader after
       seeing a fix's replies?
    1. Read the fix's ignored answers to questions that the baseline got
       right. Do they decline in other words, or answer and miss a key
       fact? Which of the fix's two failures does the tolerant rule
       explain, and which does it leave?
   :exlabel:`ex_lm_rag-tolerant`

1. [code] **Splitting the fix.** The fix of the ignored class changed three
   settings at once.
    1. Define three arms that each change one of them: the quoting prompt
       alone (`dict(d2l.RAG_BASE, prompt='quote')`), the order with the best
       passages at both ends alone, and room for 500 new tokens alone.
       Before you run them, predict which carries the loss.
    1. Generate the three arms on `dev`, judge them with the local judge, and
       report each paired difference with the baseline and each arm's
       failure classes. Without two GPUs for the judge, compare the
       abstentions on the unanswerable questions, which need no judge.
    1. Do the three effects add up to the bundle's? What does the answer say
       about writing a fix down in advance as a bundle, and why may these
       results not be read on `test`?
   :exlabel:`ex_lm_rag-split-fix`

1. **Failure counts from two graders.** The labeler called 9 of the 26
   failures correct.
    1. From the table of rule against labeler printed in
       :numref:`subsec_lm_rag_failures`, recount the classes with those nine
       answers taken as correct. Does the largest class change?
    1. With the bootstrap of the failure cell, or with the normal
       approximation to a binomial share, compute an interval for the share
       of the largest class among the remaining failures. Can the classes
       now be ranked?
    1. What would it take to decide which grader is right on the nine, and
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
       differences with the larger generator. Without two GPUs for the
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

::: {.slide title="26 Answers Are Wrong, Though Retrieval Finds the Passage for 213 of 220"}
[The problem]{.kicker}

- Baseline: 0.89 of 240 `dev` questions about this book.
- 19 of its 24 errors on answerable questions had the passage in context.
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
- $a_1 = 0.91$ on hits, $a_0 = 0.29$ on misses.
- Perfect retriever, $a_1$ unchanged:
  $(1 - h)(a_1 - a_0) \approx 0.032 \times 0.63 \approx 0.02$.

::: {.d2l-note}
Inside the $\pm 0.03$ that 240 questions resolve: retrieval has no room
here.
:::
:::

::: {.slide title="Section Chunks Cover More Than Fixed Windows at Small Budgets"}
[Chunking at equal tokens]{.kicker}

| budget $T$, 220 questions | sections | fixed 256/64 | fixed − sections |
|---|---|---|---|
| 1,000 tokens | 0.91 | 0.84 | $-0.07$ [$-0.11$, $-0.02$] |
| 2,000 tokens | 0.97 | 0.94 | $-0.03$ [$-0.06$, +0.01] |
| 3,000 tokens | 0.98 | 0.97 | $-0.01$ [$-0.04$, +0.02] |

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
| fixed windows | 1,600 | $-0.08$ | yes |
| rewrite + rerank top 50 (two changes) | 1,700 | +0.00 | no |
| reverse order | 1,700 | +0.01 | no |
| budget 1,500, no 5-chunk limit | 1,300 | $-0.03$ | yes |
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
| fixed windows, up to the baseline's tokens | 1,600 | $-0.08$ |
| control: windows just past the baseline's tokens | 1,800 | $-0.05$ |

- Coverage near these budgets trails by 0.03 to 0.07: worth at most
  0.02 to 0.04 of accuracy.
- Coverage counts half a span; a window can cut off the key fact.

::: {.d2l-note}
Chunks that follow the text's structure give better answers.
:::
:::

::: {.slide title="Each Wrong Answer Is Blamed on the First Stage That Lost It"}
[Failures by stage]{.kicker}

| class | condition, checked in this order | of 26 | answer cites a relevant chunk |
|---|---|---|---|
| false answer | unanswerable, yet answered | 2 | 0 |
| not retrieved | no relevant chunk in the top 50 | 0 | 0 |
| ranked low | in the top 50, not in the context | 5 | 0 |
| false abstention | in the context, answer abstains | 1 | 0 |
| ignored | in the context, faithful, misses a key fact | 12 | 10 |
| unsupported | in the context, not faithful | 6 | 6 |

::: {.d2l-note}
Ignored 0.46 [0.27, 0.65], unsupported 0.23 [0.08, 0.38]: 26 failures
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
A rule picks the largest class, ignored, and carries the largest stage
gain to `test`, read once.
:::
:::

::: {.slide title="Check: Three Changes Lie Within One Verdict on Dev. What Will Test Show?"}
[Check]{.kicker}

Reverse order, budget 6,000, rewrite + rerank: +0.00 to +0.01 each on
`dev`, within one verdict of 240. The rule carried rewrite + rerank to
`test`.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $-0.004$, interval $-0.05$ to +0.04. Picking the largest of
near-equal numbers picks noise.
:::
:::
:::

::: {.slide title="Two Graders Disagree More Often About Whether an Answer Is Wrong Than About Where"}
[A second labeler]{.kicker}

Claude Opus 5.5 labeled the cards of the 26 failures.

| the labeler | failures |
|---|---|
| agrees with the rule | 13 |
| calls the answer correct | 9 |
| blames another stage | 4 |

Of the 9: in 7 the judge insists on a key fact; 2 unanswerable questions
were declined in other words. Most disputed class: ignored, 4 of 12.
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

::: {.slide title="The Fix Chosen in Advance Failed on Dev and on Test"}
[Quote, then answer]{.kicker}

| accuracy | baseline | fix | fix − baseline |
|---|---|---|---|
| `dev` | 0.89 | 0.81 | $-0.08$ [$-0.12$, $-0.05$] |
| `test`, read once | 0.87 | 0.79 | $-0.08$ [$-0.12$, $-0.03$] |

- 20 fewer correct answers on `dev`; 13 answerable ones that the baseline
  got right are lost, 9 of them now ignored.
- Chosen from `dev` failures, so `dev` could have flattered it. It lost
  there anyway, and `test` agrees.
- One idea in three settings: the loss belongs to the bundle.
:::

::: {.slide title="A Change Aimed at One Class Created Another"}
[Where the failures went]{.kicker}

::: {.cols .vc}
::: {.col}
| `dev` answers | baseline | fix |
|---|---|---|
| correct | 214 | 194 |
| ranked low | 5 | 4 |
| ignored | 12 | 19 |
| unsupported | 6 | 7 |
| false answer | 2 | 15 |
| false abstention | 1 | 1 |
:::

::: {.col}
Abstentions on the 20 unanswerable questions: 18 → 5. About half of the
other 15 declined in the model's own words, not with "I cannot answer
this from the provided passages."
:::
:::

::: {.d2l-note .rule}
Test a prompt with its scoring rule. Never search for a better fix on the
same data.
:::
:::

::: {.slide title="The Fix Fills Two Failure Classes, and Test Agrees with Dev"}
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
| closed book | 90 | 0.95 |
| baseline | 1,900 | 1.3 |
| budget 6,000 | 6,300 | 2.6 |
| fix: quote, then answer | 2,000 | 2.1 |

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
- Fixed windows $-0.08$, and $-0.05$ even with more tokens than the
  baseline: the cut costs accuracy. Budget 1,500: $-0.03$.
- Largest class: ignored, 12 of 26; graders differ more on whether than
  on where.
- The fix chosen in advance lost 0.08 on `dev` and on `test`.
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
