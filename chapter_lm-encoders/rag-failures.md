# Failures and Fixes in Retrieval-Augmented Generation
:label:`sec_lm_rag_fixes`

A reader who has built the pipeline of :numref:`sec_lm_rag` holds a
baseline that answers 0.89 of the `dev` questions, with a retriever that
puts a relevant passage into the context for 213 of the 220 answerable
ones. Twenty-six answers are still wrong, among them the one that opened
that section: the passage that answers was ranked first, and the answer left
out its key fact. What should change? Another chunking, a rewritten query,
more context, or the generator, which no change to retrieval can reach?

The idea of this section is to let measurements, not intuition, pick the
change. We change one stage at a time and compare with the baseline on the
same questions, sort the wrong answers by the first stage that lost them,
and test the repair of the largest class, written down before any failure
was counted, on `test`, read once. A short derivation then shows how the
original RAG model makes retrieval part of a probabilistic model whose
retriever can learn from the generator. The section ends on the question
that longer context windows keep raising: is retrieval needed at all when
whole chapters fit into the prompt?

The first cells load what :numref:`sec_lm_rag` built: its configurations,
the corpus with its two retrievers, and the recorded answers and verdicts.
The helpers that score answers with paired intervals come from the library,
as there.

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
CARD_GIB = 21.5  # estimates; measured peaks add 1-2 (23.6 GiB usable)
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

We change one stage at a time and compare with the baseline on the same
questions. Chunking comes first, and it can be measured before any answer
is generated: section chunks follow the book's headings and paragraphs,
fixed windows of 256 tokens overlapping by 64 do not. Which covers more at
equal budgets?

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
a handful of verdicts. `test` was read once, with the rewrite and reranker,
the stage the rule picked when it ran. Since then, re-judging a few answers
has put reversing the order in front, as the cell prints. We do not read
`test` again: reading it again after a change made on `dev` is how a test
set leaks into the results.

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

Now back to the answer that opened :numref:`sec_lm_rag`. It was picked by a
fixed rule: the first `dev` answer, by question id, whose context held a
relevant chunk, that the judge called incorrect, and whose card the labeler
gave the same class as the rule. With two graders required to agree, the
example is unlikely, though not certain, to be a judge error.

```{.python .input #rag-opening-item}
%%tab pytorch
# The rule: the first base dev answer (in qid order) with a relevant chunk
# in its context that the judge called incorrect, and whose failure card
# carries the same class from the second labeller as from the rule: a
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
          'by the second labeller')
```

The answering passage was ranked first, and its last sentence, printed
above and quoted at the start of :numref:`sec_lm_rag`, holds the missing key
fact. The answer cites passages two and three and leaves out that no warning
is given. The passage was neither missing nor contradicted. It was ignored.

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
measurable at twice the time, and the oracle of :numref:`sec_lm_rag` shows
that the generator, given the right passage, still misses about one
question in twelve. The agents of :numref:`chap_lm_agents` take a third
path, retrieving when they decide that they need to. Whether whole chapters in the context would beat
retrieval on the multi-passage questions, at equal cost, we leave open; it
is an experiment for the exercises.

## Summary

Changing one stage of the baseline at a time, with paired intervals on the
same questions, found no change that raised accuracy and two that lowered
it: fixed windows instead of section chunks, even with more context tokens,
and a smaller budget. Sorting the 26 wrong answers by the first stage that
lost them made passages ignored the largest class. Its fix, written down
before any failure was counted, lowered accuracy by 0.08 on `dev` and on
`test`, mostly by creating false answers: a change aimed at one class can
create another. The original RAG model marginalizes over retrieved documents
instead of concatenating them, and its retriever learns from the generator
through the posterior minus the prior. Whether whole documents in a long
context can replace retrieval remains open.

## Exercises

1. **The truncation error and recall.** Start from :eqref:`eq_lm_rag_topk`.
    1. Show that the bound is tight: give scores and likelihoods for which
       $p_S - p_S^{(k)} = (1 - m_k) \max_z L_z$.
    1. Suppose exactly one document $z^*$ answers each question, and the
       retriever is calibrated: $p(z \mid x)$ is the probability that $z$ is
       $z^*$. Show that the expectation of $1 - m_k$ over questions equals
       the probability that $z^*$ is not in the top $k$, one minus
       Recall@$k$.
    1. With Recall@5 = 0.97 from :numref:`sec_lm_rag`, what would the bound
       promise for RAG-Sequence over five documents if the retriever were
       calibrated? Why is a softmax over BM25 or dense scores rarely
       calibrated in this sense, and how would you calibrate it on `train`?
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

Failures and Fixes in Retrieval-Augmented Generation<br>**one change at a time · failures by stage · a fix chosen in advance · marginalizing over documents**
:::
:::

::: {.slide title="26 Wrong Answers: Which Stage Lost Each One?"}
[The problem]{.kicker}

- The baseline answers 0.89 of 240 `dev` questions; a relevant passage
  reaches the context for 213 of 220 answerable ones.
- Still 26 answers are wrong. One cites [2] and [3] and misses the fact
  that passage [1], ranked first, states.
- Another chunking? A rewritten query? More context? Or the generator?

::: {.d2l-note .warn}
Each stage needs a different repair, and one accuracy number does not say
which.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- change one stage at a time and read the paired interval;
- compare chunkings at equal tokens, with a control for the token count;
- blame each failure on a stage, and test a fix chosen in advance;
- derive RAG-Sequence, RAG-Token and the retriever's gradient.

**Running example:** the baseline of the previous section on 240 `dev`
questions; Qwen3.5-9B answers, Qwen3.8-27B judges; `test` read once.
:::

::: {.slide title="Measured Stages Become Changed Stages"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| movement | section |
|---|---|
| vectors | Word Embeddings |
| | Pretrained Encoders |
| | Text Embeddings from Contrastive Training |
| finding | Sparse, Dense and Approximate Retrieval |
| using | Retrieval-Augmented Generation |
| | **Failures and Fixes in RAG** |
:::

::: {.col .narrow}
- **Uses:** the question set, judge, baseline and paired bootstrap of the
  previous section.
- **Skips:** running joint training of retriever and generator (derived,
  not run); retrieval inside agent loops.
:::
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

- One change at a time, paired on the same 240 questions.
- No stage change helped; fixed windows $-0.08$, budget 1,500 $-0.03$.
- Largest class ignored; its fix lost 0.08 on `dev` and on `test`.
- A change aimed at one class can create another.
- $p_S$, $p_T$; gradient = posterior − prior.
:::

::: {.slide title="Next: Whole Chapters in Context, or Retrieval on Demand"}
[Next]{.kicker}

A 6,000-token budget, 3.4 times the baseline's context, gained nothing at
twice the time. Would whole chapters in a 32,000-token context beat
retrieval on multi-passage questions? Agents take a third path: they
retrieve when they decide they need to.
:::
