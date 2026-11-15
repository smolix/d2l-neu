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

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "2.5 Retrieval-Augmented Generation · T1").
Blueprint: docs/llm/blueprints/ch02_encoders.md section 2.5.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A RAG system is a chain of stages, and a failure is located by measuring each stage.
**Opening:** The reader wires the 2.4 retriever to Qwen3-4B and asks it a question about this book. The gold passage is retrieved, and the answer is still wrong. Was the passage ranked too low, ignored, or contradicted by a neighbouring chunk? The section builds a question set over the book, measures each stage with its own interval, and closes on the question of whether a long enough context makes retrieval unnecessary.
**Teaches:**
- The stages: chunking (fixed windows against structure-aware chunks, compared by span coverage at a token budget), query rewriting, reranking, context assembly (order, budget, near-duplicate removal), grounded generation with numbered citations and an abstention string.
- Per-stage metrics with paired intervals clustered by document: retrieval recall, coverage at a budget, answer accuracy, faithfulness, citation precision and recall, abstention rates; a reference-based judge (`d2l.rag_judge`) validated against human labels on a calibration set.
- A failure taxonomy (not retrieved, ranked low, ignored, unsupported, false answer, false abstention) applied to 50 cases, and the discipline of fixing the largest class on `dev` and confirming on `test` once.
- RAG-Sequence and RAG-Token as two marginalizations over retrieved documents, and what the top-k approximation and the retriever gradient mean for training (stated for the mechanism; training for RAG is the Further Reading notebook).
**Derivation:** RAG-Sequence marginalizes the whole answer's likelihood over the top-k documents weighted by the retriever's probabilities; RAG-Token marginalizes per generated token, so different tokens may draw on different documents; the top-k approximation drops the mass outside the retrieved set, and the gradient reaches the retriever through the document probabilities. Smallest numeric case: two documents, a two-token answer, probabilities by hand under both marginalizations.
**Notebook:**
1. Load the committed question set (`d2l.book_qa`: 240 human-verified `dev` and 240 `test` items with gold spans, key facts and unanswerable items; the `train` items pass automatic filters only) and the corpus; run the baseline pipeline (section chunks, dense retrieval, top 5, rank order, the fixed grounded prompt) on `dev` and report every metric with its clustered interval; also the closed-book and oracle-context arms as the floor and ceiling.
2. Change one stage at a time (fixed against section chunks at equal token budget; query rewriting plus the reranker; context order and budget) and report which change moves answer accuracy outside its paired interval.
3. Label 50 failures of the baseline by stage, fix the largest class, remeasure on `dev`, then evaluate once on `test`.
4. Claim that survives noise: one stage carries most of the gain (its paired interval excludes zero on `dev` and the sign holds on `test`); the oracle-context arm bounds what retrieval can add; closed-book accuracy is above zero because the book is public and below the baseline on book-specific items. The judge's flip rate between batch sizes on 60 items and its agreement with the human calibration labels are reported before any claim.
5. Reused pieces: `d2l.book_corpus`, `d2l.BookRetriever`, `d2l.rerank_yes_no`, `d2l.rrf_fuse`, `d2l.embed_texts`, `d2l.retrieval_metrics`, the chat template with `enable_thinking=False`, `d2l.paired_bootstrap(cluster=...)`, `d2l.load_hf`, `d2l.config_memory` as the memory pre-flight; new `d2l.book_qa`, `d2l.gold_chunk_ids`, `d2l.cov_at_budget`, `d2l.assemble_context`, `d2l.rag_prompt`, `d2l.RAGPipeline`, `d2l.rag_judge`, `d2l.citation_metrics`, `d2l.classify_failure`.
**Figures:** `fig_lm_rag_stages` (the chain from question to cited answer with the metric measured at each stage); `fig_lm_rag_failures` (plot: failure-class proportions before and after the fix, with the stage waterfall of accuracy and citation precision).
**Restate / forward:** `subsec_mdl-paired-comparisons` (a paired interval per stage removes question difficulty from the comparison); `sec_mdl-latent-em-elbo` (a marginal likelihood over a latent choice, here the document). Forward: 10.4 (retrieval inside an agent loop).
**Exercises:** 6 to 8. Load-bearing: derive the error of the top-k approximation as the retriever mass outside the retrieved set and bound it with the retriever's calibration (derive); measure the position effect by moving the gold passage from first to last in a fixed context of five chunks (measure); check that citation precision computed from gold ids penalizes a valid citation of a supporting non-gold chunk and compare with the judge's support verdict (check); open: place whole chapters in a 32K context at an equal latency budget and compare accuracy with retrieval on multi-passage questions (the long-context arm).
**Writer notes:**
- The notebook has three parts around one question (which stage change moves accuracy outside its paired interval), where the blueprint has six; REALM, RAFT, Self-RAG, late chunking and contextual headers are in the Further Reading notebook; the long-context comparison and the position sweep are exercises; the generator is Qwen3-4B (dense, full attention; the section opening names it) so the judge, Qwen3.5-4B in bf16, is a different model, and the 32K arm is not in the notebook because Qwen3-4B caches about 147 KB per token. The question-set construction stays as in blueprint ch02 2.5 with these name changes: the closed-book probe uses the generator (Qwen3-4B); the authoring-time answer check uses a third model (Qwen3.8-27B through 1.7's client or a T2 pair) so neither the drafter nor the judge filters the set.
- The question set (`d2l-book-qa-v1`): item schema `qid, split, type, question, reference, key_facts, answerable, gold_spans, corpus_version, drafter, human_verified, checks, closed_book_correct, flags`; types `single, explain, numeric, code, multihop, unanswerable` with the quotas of the blueprint; documents, not questions, are split, so no `dev` or `test` passage sources a `train` question; the drafter is DeepSeek-V4.1-Flash-1017 (outside the Qwen3 family, MIT), called once at authoring through 1.7's client under a cost cap (about 4,500 calls; see "Paid API calls") and recorded with its snapshot id and provider; the drafted questions ship with the book; lexical copying is filtered (question-anchor Jaccard at most 0.5, no shared 5-gram outside identifiers); a 100-pair human calibration set ships as `dev_judge_labels.jsonl`; 10% double annotation with Cohen's kappa reported. Human verification of 480 items is 8 to 16 annotator-hours (approved; blueprint README Decision 13).
- Exports (`01_interfaces.md`): `d2l.book_qa(split='dev', answerable=None, types=None, n=None, seed=0) -> d2l.BookQA` with splits `train/dev/test` and no `gold_ids` field; `d2l.rag_judge(...) -> {'correct', 'faithful', 'supported'}` (there is no `book_qa_score`); `d2l.RAGPipeline(retriever, model, tokenizer, k=5, budget_tokens=3000, order='rank', rewrite=None, hyde=False, decompose=False, rerank_k=None, max_new_tokens=300)` returning `RAGResult` records; `d2l.ABSTAIN` is the fixed abstention string; the grounded prompt is fixed and Parts 2 and 3 vary only the context and the query.
- Reference and tolerance: `d2l.generate` is the oracle in one test against Transformers batched generation (the pipeline uses the latter for speed); greedy decoding is deterministic up to batching, so one run is made twice and the output change rate is reported (expected under 2%); judge flip rate on 60 items between batch sizes 1 and 8.
- Memory within a 15 GiB slot: Qwen3-4B bf16 weights about 8 GB plus KV at a 3,000-token budget (about 0.45 GB) and prefill activations; free it before loading the judge. `d2l.config_memory` is the pre-flight check; `logits_to_keep=1` at prefill.
- Pitfalls to state: tuning on `dev` and reporting `dev`; recall across chunk sizes without equalizing tokens; a judge from the generator's family without a human-agreement check (here the families differ, and the check is still reported); correct is not faithful; "not retrieved" and "ranked low" need different fixes; closed-book accuracy is not zero for a public book; needle-in-a-haystack results do not carry over to multi-passage questions.
- Committed artifacts: `d2l-book-qa-v1`, `d2l-r3-ctx-headers-v1` (only if the contextual-header exercise keeps it), `d2l-rag-cache-v1` (generations and judgments per configuration, replayed unless `REGEN=True`), the 50 failure cards and labels; `QUICK=True` evaluates an 80-question stratified subset.
- Pin the drafter model's identity and version at authoring and record it in the item schema (`drafter`); decide at authoring whether `test` should be hidden behind hashes (it is public book text, so grading relies on the private split).
**Sources:**
- Papers: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (RAG-Sequence, RAG-Token and the retriever gradient); [Lost in the Middle](https://arxiv.org/abs/2307.03172) (position effects in long contexts; the exercise); [RAGAS](https://arxiv.org/abs/2309.15217) (reference-free faithfulness and relevance metrics, compared against in an exercise); [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627) (citation precision and recall by entailment: the template for the citation metrics); [Seven Failure Points When Engineering a Retrieval Augmented Generation System](https://arxiv.org/abs/2401.05856) (a failure taxonomy from practice); [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) (hypothetical-document query expansion, the query-side stage).
- Code and documentation: [RAGAS](https://github.com/vibrantlabsai/ragas) (the metric implementations); Transformers [chat templating](https://huggingface.co/docs/transformers/chat_templating) and [generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) (batched grounded generation); [Pyserini](https://github.com/castorini/pyserini) (retrieval baselines).
- Courses, tutorials and blogs: Hugging Face cookbook, [Advanced RAG](https://huggingface.co/learn/cookbook/advanced_rag); Jason Liu, [Systematically Improving Your RAG](https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/); Hamel Husain, [Stop Saying RAG Is Dead](https://hamel.dev/notes/llm/rag/not_dead.html); Eugene Yan, [Patterns for Building LLM-based Systems](https://eugeneyan.com/writing/llm-patterns/); [CMU 11-711](https://cmu-l3.github.io/anlp-spring2026) (the retrieval and RAG lecture).
-->

<!-- Placeholder (prose, opening): start from the reader. The 19.4 retriever
is wired to Qwen3.5-9B and asked a question about this book; the gold passage
is retrieved and the answer is still wrong (the item printed by the
opening-item cell below, after the baseline is measured). Was the passage
ranked too low, ignored, or contradicted by a neighbouring chunk? The section
builds a question set over the book, measures every stage with its own
interval, and closes on whether a long enough context makes retrieval
unnecessary. -->

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
import time
import transformers
import warnings
import zipfile
```

<!-- Placeholder (prose): the models of the section: the generator
Qwen3.5-9B (thinking off, bf16; a hybrid of Gated DeltaNet and full-attention
layers, so only one layer in four grows a KV cache; it takes a whole 24 GB
card), the 19.4 embedder and reranker, the judge Qwen3.8-27B with thinking
on. The generator is pinned like the models of chapter 18. -->

```{.python .input #rag-revisions}
%%tab pytorch
# Pinned in this notebook, not in the library: adding it to the saved
# HF_REVISIONS would change the fingerprint of every chapter-18 notebook
d2l.HF_REVISIONS['Qwen/Qwen3.5-9B'] = (
    'c202236235762e1c871ad0ccb60c8ee5ba337b9a')
```

<!-- Placeholder (prose): the regeneration switches (D2L_RERUN), their
order (each result registered before the next regeneration reads it) and
the registered results the normal run replays; the fixed settings of the
baseline and the configurations changed one stage at a time, with the
pre-registered fixes written down before any failure is counted. -->

```{.python .input #rag-config}
%%tab pytorch
GEN, RERANKER = 'Qwen/Qwen3.5-9B', 'Qwen/Qwen3-Reranker-0.6B'
EMB, JUDGE = d2l.DEFAULT_EMBEDDER, d2l.RAG_JUDGE_LOCAL
GPU = torch.cuda.is_available()
SMALL = not GPU  # the CPU path: two live answers, BM25-only coverage
T0 = time.time()  # a judging run stops JUDGE_MINUTES after this
ARTIFACTS = {  # name: (registered key, the file a regeneration writes)
    'rag-gen-base': ('lm-rag-gen-base-v1', 'lm-rag-gen-base.jsonl'),
    'rag-gen-stages': ('lm-rag-gen-stages-v1', 'lm-rag-gen-stages.jsonl'),
    'rag-gen-fix': ('lm-rag-gen-fix-v1', 'lm-rag-gen-fix.jsonl'),
    'rag-judge': ('lm-rag-judge-v2', 'lm-rag-judge.jsonl'),
    'rag-judge-cache': ('lm-rag-judge-cache-v2', 'lm-rag-judge-cache.zip'),
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
    'rewrite-rerank': dict(BASE, rewrite='generator', rerank_k=50),
    'reverse': dict(BASE, order='reverse'),
    'budget-1500': dict(BASE, k=None, budget=1500),
    'budget-6000': dict(BASE, k=None, budget=6000, batch=4)}
STAGES = ['fixed', 'rewrite-rerank', 'reverse', 'budget-1500', 'budget-6000']
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

<!-- Placeholder (prose, ## A Question Set over the Book): why a question
set over the book: the content is known, every answer has a gold passage,
and the documents, not the questions, are split, so no dev or test passage
sources a train question. The registry below: the set (version 2), the API
responses it was built from (tools/build_book_qa.py --mode replay rebuilds
it), and the labels the judge is checked against. -->

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
```

<!-- Placeholder (stage 2): the six run artifacts of ARTIFACTS
(lm-rag-gen-base-v1, lm-rag-gen-stages-v1, lm-rag-gen-fix-v1,
lm-rag-judge-v1, lm-rag-judge-cache-v1, lm-rag-cards-v1) join the registry
above, in one more #@save block, as each is uploaded (spec D21: each
registered before the next regeneration). -->

<!-- Placeholder (prose): the item schema (question, reference, key facts,
gold spans into the cleaned documents, alternative spans, flags); the loader
returns one split, and relevance is computed for any chunking from the
spans, so there is no stored list of gold chunk ids. -->

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

<!-- Placeholder (prose): how the set was made. DeepSeek-V4.1-Flash-1017
(outside the Qwen family) drafted two candidates per anchor chunk; automatic
filters (verbatim evidence, context phrases, lexical copying, near
duplicates) and a third model, Qwen3.8-27B, checked them; a frontier model
(Claude Opus 5.5) verified every dev and test item in a fresh context. The
items were checked by models, not by people: what that costs, what it can
miss (blind spots the models share; a frontier model may have read the
book). The tables: items per split and type, the models that served, the
rejections per filter, and how often verifier and checker agree. -->

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
print(f'drafter: {served("drafter")}; checker: {served("checker")}; '
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
```

<!-- Placeholder (prose): one dev item of each type with the text of its
gold span: what "self-contained" and "specific to this book" mean in
practice. -->

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
    print(f'[{t}] {q["qid"]}: {q["question"]}\n  reference: '
          f'{q["reference"]}\n  key facts: {q["key_facts"]}')
    for s in q['gold_spans']:
        print(f'  gold span in {s["doc_id"]}: {span_text(s)}')
```

<!-- Placeholder (prose): splits by document, and what has to hold for that
to mean no leakage; the checks run on every item. -->

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

<!-- Placeholder (prose): relevance at chunk level for any chunking: a chunk
is relevant if it overlaps a gold (or alternative) span by half the span's or
half the chunk's characters; the share of answerable items with a relevant
chunk under each shipped chunking, and one item whose relevant chunks differ
between chunkings. -->

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
    print(f'  {tag}: ' + ', '.join(
        f'{i.split("#")[-1]} [{c[i]["char_start"]}, {c[i]["char_end"]})'
        for i in sorted(d2l.gold_chunk_ids(q, c, include_alt=False))))
```

<!-- Placeholder (prose, ## The Pipeline and Its Measurements): the stages
(fig_lm_rag_stages, from the generator): query (rewrite), first-stage
retrieval, rerank, context assembly (order, budget, near-duplicate removal),
grounded generation with numbered citations and the abstention string; the
judge at the end. Under each stage the metric that sees it. -->

<!-- Placeholder (figure): ![The stages of a retrieval-augmented generator
and the metric that measures each.](../img/mdl-lm-encoders-rag-stages.svg)
:label:`fig_lm_rag_stages` (generator tools/gen_mdl_lm_encoders_figures.py,
written in the figures step). -->

<!-- Placeholder (prose): the two measurements of the context before any
answer: coverage at a token budget (does the context hold half of a gold
span?) and the assembly itself: rank order, a token budget, near duplicates
removed by the cosine of their chunk vectors, three orders. -->

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

<!-- Placeholder (prose): the fixed grounded prompt: numbered passages with
their heading paths, a citation after each statement, and one abstention
string; without passages it is the closed-book prompt. Every stage below
changes the context or the query, never this prompt (the pre-registered
fixes are the only exceptions, and they say so). -->

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

<!-- Placeholder (prose): the pipeline in three memory phases (the generator
writes queries; retrieval, reranking and assembly run while the generator
idles; the retriever's models leave the GPU before batched generation);
greedy decoding with left padding, the logits of the last position only at
the prefill; the query-side options (rewrite, HyDE, decomposition) and the
switches the pre-registered fixes use. -->

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

<!-- Placeholder (prose): the judge (rag_judge): one prompt per answer with
the question, the reference, the key facts, the passages and which were
cited; three verdicts (correct, faithful, supported); an abstention decided
by string match. Its model, Qwen3.8-27B with thinking on, and how it was
chosen: nine Qwen candidates measured against two frontier models' labels;
the small judge's leniency and why thinking matters. Then the two metrics
computed from gold ids (citation precision and recall) and the failure
classes with their order of checks. -->

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
    'unsupported' (not faithful). A missing verdict counts as false."""
    r = result._asdict() if hasattr(result, '_asdict') else result
    true = lambda v: isinstance(v, (bool, np.bool_)) and bool(v)
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
    return 'ignored' if true(judge_out['faithful']) else 'unsupported'
```

<!-- Placeholder (prose): the memory pre-flight before any model is loaded.
Qwen3.5-9B's bf16 weights alone take 16.7 GiB, so the generator gets a whole
24 GB card, not a 15 GiB slot; its KV cache is small, because only the eight
full-attention layers keep one, and the 24 Gated DeltaNet layers keep a fixed
state. The rows: generation at eight 3,300-token prompts, and four of 6,300
for budget-6000 (twelve and eight ran out of memory on an RTX 4090 when this
section was written: measured peaks 21.7 and 22.4 GiB at eight and four,
about 1-2 GiB above the estimate, which leaves out the allocator's slack);
the retrieval phase with the generator idle on the card (measured 19.2
GiB); and the judge, too large for one card, which gets two. When the
retrieval phase does not fit, the generator leaves the card while the
retriever works and comes back to answer. -->

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
      'cards, and vLLM reserves 90% of each card for its weights, cache and '
      'activations')
assert table.total.iloc[:2].max() < CARD_GIB  # generation fits one card
KEEP_GEN = table.total.iloc[2] < CARD_GIB  # the generator idles on the card
print('retrieval phase: the generator stays on the card' if KEEP_GEN else
      'retrieval phase: the generator leaves the card and comes back')
```

<!-- Placeholder (prose): the baseline (sections, hybrid retrieval, top 5,
rank order, a 3,000-token budget, the fixed prompt) on the 240 dev questions,
with the closed-book arm (no passages) as the floor, the oracle arm (the gold
chunks only) as the ceiling, and the baseline once more in other batches.
Token budgets count the corpus's tokens (the Qwen3 tokenizer of the embedder
and of 19.4's chunks); the generator, Qwen3.5-9B, has a tokenizer of its own,
so the printed prompt tokens are its count, not the budget's. Answers are
regenerated only with D2L_RERUN; the normal run replays them, generates the
registered run's first batch again, then checks the pipeline's batched
generation against the plain decoding loop of chapter 18 on the same prompts
(they agree token for token; in float32 a left-padded batch changes the
last-position logits of Qwen3.5-2B, the same architecture, by about 1e-4, so
the padding is masked in the linear-attention layers too). In bf16 the logits move by about one
rounding step (0.25 at a scale of 25) with the batch's padding, which flips
near-ties: the same prompt batched with other prompts can give another greedy
answer. -->

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
      f'context, {sum(s == 1 for s in same)} identical answers; tokens '
      f'agreeing before the first difference: {np.mean(same):.3f}')
oracle = d2l.generate(*generator(), [d2l.rag_prompt(
    q['question'], [CHUNKS[i] for i in r['context_ids']])
    for q, r in zip(LIVE, live)], chat=True, enable_thinking=False,
    max_new_tokens=MAX_NEW, batch_size=BATCH)
same = [prefix_share(r['answer'], g.text.strip())
        for r, g in zip(live, oracle)]
agree = sum(s == 1 for s in same) >= N_LIVE - 1 and np.mean(same) >= 0.95
print(f'the same prompts through d2l.generate: {sum(s == 1 for s in same)} '
      f'of {N_LIVE} identical; tokens agreeing before the first difference:'
      f' {np.mean(same):.3f}: ' + ('the two loops agree' if agree else
                                  'a padding or logits_to_keep bug'))
assert agree, 'batched generation differs from the plain decoding loop'
free_generator()  # the card goes to the retriever and the judge
```

<!-- Placeholder (prose): judging. A regeneration starts the judge server
(vLLM, two 24 GB GPUs; or, without them and with a key, the same open
weights on OpenRouter) after the generator is freed, judges every answer
that lacks a verdict in 50-minute runs, and judges 60 baseline answers a
second time at another concurrency. The normal run replays the verdicts
and re-judges eight from the recorded cache: no server, no GPU, no key. -->

```{.python .input #rag-judge}
%%tab pytorch
LABELS = d2l.api_unpack('lm-rag-judge-labels-v1')
REC_DIR = os.path.join('..', 'data', 'lm-api', 'record',
                       ARTIFACTS['rag-judge-cache'][0])
VERDICTS = ['correct', 'faithful', 'supported']

def passage(cid):
    c = CHUNKS[cid]
    return ' > '.join(c['heading_path']) + '\n' + c['text']

def judge_client(backend, cache, mode='replay', url='http://127.0.0.1/v1'):
    """The judge through the local vLLM server or OpenRouter (same weights,
    BF16), recording to or replaying from cache."""
    if backend == 'local':
        return d2l.rag_judge_client(url, cache=cache, mode=mode)
    with open(os.path.join(LABELS, 'listing.json')) as f:
        listing = json.load(f)
    return d2l.openrouter_api('qwen/qwen3.8-27b', 'deepinfra/bf16',
                              listing=listing, cache=cache, mode=mode,
                              cost_cap_usd=15.0, reasoning={'enabled': True})

def judge_records(client, recs, concurrency, backend):
    """Judgment records of answer records (dicts)."""
    v = d2l.rag_judge(client, None, [ITEMS[r['qid']] for r in recs], recs,
                      [[passage(i) for i in r['context_ids']] for r in recs],
                      batch_size=concurrency)
    return [dict(config=r['config'], qid=r['qid'], concurrency=concurrency,
                 backend=backend, answer_sha=answer_key(r), **x)
            for r, x in zip(recs, v)]

def load_judgments(path):
    """Judgment records, verdicts as True, False or None."""
    df = d2l._rag_records([path], on=['config', 'qid', 'concurrency'])
    for k in VERDICTS:
        df[k] = [None if v is None or v != v else bool(v)
                 for v in df.get(k, [])]
    for k in ('backend', 'answer_sha'):
        df[k] = df.get(k, [None] * len(df))
    return df

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
                 and r['qid'] in pick], 2 * CONCURRENCY, backend)
    finally:
        if server:
            server.stop()
    print(f'{len(todo)} answers still to judge: run D2L_RERUN=rag-judge '
          'again' if todo else 'every answer is judged')
    d2l.lm_save_table(ARTIFACTS['rag-judge'][1], pd.DataFrame(rows),
                      on=['config', 'qid', 'concurrency'],
                      key=ARTIFACTS['rag-judge'][0])
    members = {f: open(os.path.join(REC_DIR, f), 'rb').read()
               for f in sorted(os.listdir(REC_DIR))}
    members['manifest.json'] = json.dumps(dict(
        judge=JUDGE, prompt='RAG_JUDGE_V1', backends=sorted(
            {r['backend'] for r in rows}), concurrency=CONCURRENCY,
        repeat_concurrency=2 * CONCURRENCY, n=len(rows)), indent=1).encode()
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
if len(judged) and cache:  # eight base judgments replayed from the cache
    first = judged[judged.config == 'base'].sort_values('qid').head(8)
    recs = GENS.set_index(['config', 'qid']).loc[list(zip(
        first.config, first.qid))].reset_index().to_dict('records')
    again = pd.DataFrame(judge_records(
        judge_client(first.backend.iloc[0], os.path.join(cache,
                                                         'judge.jsonl')),
        recs, CONCURRENCY, first.backend.iloc[0]))
    assert (again[VERDICTS].astype(str).values ==
            first[VERDICTS].astype(str).values).all()
    print('8 base judgments replayed from the judge cache: identical')
print(pd.DataFrame({
    'answers': GENS.groupby('config').size(),
    'judged': judged.groupby('config').size(),
    'no verdict': judged.assign(none=judged.correct.isna()).groupby(
        'config').none.sum()}).fillna(0).astype(int).T.to_string())
```

<!-- Placeholder (prose): the judge is checked before any accuracy is read:
its verdicts on the 100 calibration pairs (replayed from the recorded cache,
identical to the recorded ones) against each frontier labeller and against
their consensus, the labellers' agreement with each other, the old judge's
row (the leniency the change fixed), and how often a verdict changes when the
same answers are judged at another concurrency. These labels come from
models, not from people, and they were made on the answers of the earlier
generator, Qwen3-4B (re-labelling answers of Qwen3.5-9B is a paid call the
author has not approved yet). -->

```{.python .input #rag-judge-validation}
%%tab pytorch
read = lambda f: [json.loads(l) for l in open(os.path.join(LABELS, f))]
cal = [p for p in read('pairs.jsonl') if p['set'] == 'calibration']
lab = {r['qid']: r for r in read('dev_judge_labels.jsonl')}
recorded = {r['qid']: r for r in read(
    os.path.join('candidates', 'qwen38-27b-local-on.jsonl'))}
v = d2l.rag_judge(d2l.rag_judge_client(cache=os.path.join(
    LABELS, 'api', 'judge-local-qwen38-27b-on.jsonl')), None,
    [ITEMS[p['qid']] for p in cal], cal,
    [[passage(i) for i in p['context_ids']] for p in cal])
graded = [not d2l._rag_abstained(p['answer']) for p in cal]
assert all(x[k] == recorded[p['qid']][k] for p, x, g in zip(cal, v, graded)
           if g for k in VERDICTS)  # the judge's replies, replayed
print(f'{len(cal) - sum(graded)} of {len(cal)} calibration answers abstain '
      '(decided by rule); the judge grades the others')

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
        keep = [(x[k] is True, lab[p['qid']][field])
                for p, x in zip(cal, v) if lab[p['qid']][field] is not None]
        (acc, alo, ahi), (kap, klo, khi) = agreement(*zip(*keep))
        rows.append({'verdict': k, 'against': who, 'pairs': len(keep),
                     'agreement': f'{acc:.2f} [{alo:.2f}, {ahi:.2f}]',
                     'kappa': f'{kap:.2f} [{klo:.2f}, {khi:.2f}]'})
print(pd.DataFrame(rows).to_string(index=False))
print(f'no verdict (counted as false): {sum(x["correct"] is None for x in v)}'
      f' of {len(cal)} calibration pairs')
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
      f'{oc["k_hi"]:.2f}]; it passes {old["calibration"]["false_correct"][0]}'
      f' of the {old["calibration"]["false_correct"][1]} answers the '
      'labellers call wrong')
for k in ('correct', 'faithful'):  # the claims that rest on the judge
    (acc, _, _), (_, klo, _) = agreement(*zip(*[
        (x[k] is True, lab[p['qid']][k]) for p, x in zip(cal, v)
        if lab[p['qid']][k] is not None]))
    print(f'C0 on {k} against the consensus: agreement {acc:.2f}, kappa '
          f'interval from {klo:.2f}: ' + ('passes (>= 0.8, above 0.4)'
          if acc >= 0.8 and klo > 0.4 else 'fails: claims on it carry the '
          'agreement'))
print('the calibration answers come from the earlier generator, Qwen3-4B')
if len(JUD) and 2 * CONCURRENCY in JUD.concurrency.values:
    both = JUD[JUD.config == 'base'].groupby('qid').filter(
        lambda g: g.concurrency.nunique() == 2)
    changed = both.groupby('qid').apply(lambda g: (g[VERDICTS].astype(
        str).nunique() > 1).any(), include_groups=False)
    print(f'judged again at concurrency {2 * CONCURRENCY} (first run '
          f'{CONCURRENCY}): {changed.sum()} of {len(changed)} answers '
          'changed a verdict')
```

<!-- Placeholder (prose): the baseline's measurements, each with an interval
that resamples documents (questions from one document are not independent):
retrieval recall and coverage, then accuracy, faithfulness, citation
precision and recall, abstention on unanswerable and on answerable items;
the floor (closed book) and the ceiling (oracle context) as paired
differences; closed-book accuracy on book-specific items; accuracy with and
without a relevant chunk in the context, and split by this section's own
closed-book arm (the question set's closed_book_correct flag came from
Qwen3-4B; name that model if the flag is mentioned at all); and the batching
floor: base-repeat answers the same questions in other batches, so its
accuracy difference to the baseline is what re-batching alone does, and how
many answers change (counted where the context is unchanged, since the query
embeddings are batched differently too). A stage change has to move
accuracy by more than that. -->

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
    per-question scores (NaN where a score does not apply)."""
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
        'accuracy': [float(v is not None and v == v and bool(v))
                     for v in df.correct],
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
          f'{boot(C.accuracy, cluster=C.cluster)}, '
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
          f'{(~same_ctx).sum()} of {len(b)} contexts differ (query '
          f'embeddings batched differently); with the same context '
          f'{differ[same_ctx].sum()} of {same_ctx.sum()} answers differ')
print('* fewer than 100 questions (documents resampled); + fewer than 30 '
      'documents: questions resampled, an interval that is likely too '
      'narrow')
```

<!-- Placeholder (prose): the failure the opening asked about: a question
whose relevant chunk was in the context and whose answer was still judged
wrong. -->

```{.python .input #rag-opening-item}
%%tab pytorch
if 'base' in S:
    B = S['base']  # judged wrong, not merely without a verdict
    wrong = B.verdicts.map(lambda v: v['correct'] is not None
                           and not bool(v['correct']))
    row = B[(B.hit == 1) & wrong].iloc[0]
    q, rec = ITEMS[row.qid], row.record
    rel = d2l.gold_chunk_ids(q, CORPUS)
    rank = next(i for i, c in enumerate(rec['ranked_ids'], 1) if c in rel)
    print(f'{q["qid"]} ({q["type"]}): {q["question"]}\nreference: '
          f'{q["reference"]}\nfirst relevant chunk at rank {rank}; context: '
          + ', '.join(f'[{n}] {c.split(":")[1]}' + (' (relevant)' if c in rel
                                                     else '')
                      for n, c in enumerate(rec['context_ids'], 1)))
    print(f'answer: {rec["answer"]}\nverdicts: {row.verdicts}')
```

<!-- Placeholder (prose, ## Changing One Stage at a Time): chunking first,
measured before any answer: coverage at equal token budgets for
structure-aware sections and fixed 256-token windows with a 64-token
overlap. Recall at k is not compared across chunk sizes: five large chunks
hold more text than five small ones. -->

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

<!-- Placeholder (prose): one stage at a time, each a paired difference
against the baseline on the same questions: fixed windows given at most the
tokens the baseline's context used (packing stops before the window that
would pass them, so up to one window less; the printed prompt tokens show
it); the generator's rewritten query with the
reranker on the fused top 50; the context in reverse order; a smaller and a
larger budget filled in rank order. Which changes move accuracy outside
their paired intervals? -->

```{.python .input #rag-stages}
%%tab pytorch
def context_tokens(recs):
    """Tokens of the context of each answer record: the budget that
    'fixed' gets for the same question."""
    return {r['qid']: sum(CHUNKS[i]['n_tokens'] for i in r['context_ids'])
            for r in recs}

if 'rag-gen-stages' in RERUN:
    names = os.environ.get('D2L_RAG_STAGES', ','.join(STAGES)).split(',')
    assert set(names) <= set(STAGES), names
    path = run_path('rag-gen-stages')  # the other configurations are kept
    if not os.path.exists(path):  # from an earlier run, else registered
        path = d2l.lm_registered(ARTIFACTS['rag-gen-stages'][0])
    recs = [r for r in d2l._rag_records([path]).to_dict('records')
            if r['config'] not in names]
    tokens = context_tokens(GENS[(GENS.config == 'base') &
                                 (GENS.split == 'dev')].to_dict('records'))
    for name in names:
        recs += run_configs([name], DEV, budgets=tokens if CONFIGS[name][
            'budget'] == 'base' else None)
    save_gens('rag-gen-stages', recs)
    GENS = load_gens()
rows, BEST = [], None
for c in STAGES:
    if not (ready('base', c) and 'base' in S):
        continue
    s = scores(c)
    d = s.merge(S['base'], on='qid', suffixes=('', '_b'))
    r = ci(d.accuracy, d.accuracy_b, d.cluster)[0]
    rows.append({'configuration': c,
                 'accuracy': boot(s.accuracy, cluster=s.cluster),
                 'faithful': boot(s.faithful, cluster=s.cluster),
                 'cite precision': boot(s.cite_precision, cluster=s.cluster),
                 'accuracy - base': boot(d.accuracy, d.accuracy_b, d.cluster),
                 'prompt tokens': f'{s.n_prompt_tokens.mean():,.0f}',
                 's per answer': f'{s.latency_s.mean():.2f}',
                 'delta': r.mean, 'excludes 0': r.ci[0] > 0 or r.ci[1] < 0})
if rows:
    T = pd.DataFrame(rows)
    print(T.drop(columns=['delta', 'excludes 0']).to_string(index=False))
    print(f'intervals excluding 0: {list(T.configuration[T["excludes 0"]])}')
if len(rows) == len(STAGES):  # the rule of the fix step, on every stage
    BEST = T.configuration[T.delta.idxmax()]
    print(f'the largest dev gain: {BEST}')
```

<!-- Placeholder (prose, ## Failures by Stage): the taxonomy and the order
of its checks (not retrieved, ranked low, false abstention, ignored,
unsupported; false answer for unanswerable items); the class shares of the
baseline's failures on dev with intervals; 50 failure cards drawn uniformly,
five printed, and the coding agent's labels on them (a model's labels, D3)
against the rule; the largest class and the fix written down for it before
any count was seen. -->

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
    fails = BC[BC.cls != 'correct']
    SH = class_shares(fails.cls)
    print(f'{len(fails)} of {len(BC)} base answers on dev are not correct')
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
          f'next {nxt}); its fix, written down before counting: '
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
        print(f'\n[{card["rule"]}] {card["qid"]} ({card["type"]}): '
              f'{card["question"]}\n  answer: {card["answer"][:300]}\n  '
              f'relevant chunk in the context: {hit}; verdicts: '
              f'{card["verdicts"]}')
    if 'rag-cards' in RERUN:  # the agent's labels are added before upload
        d2l.lm_save_table(ARTIFACTS['rag-cards'][1], pd.DataFrame(cards),
                          on=['qid'], key=ARTIFACTS['rag-cards'][0])
    path = run_path('rag-cards')
    labelled = [c for c in (d2l.read_jsonl(path).to_dict('records')
                            if path else []) if c.get('labeller')]
    if labelled:
        rule = dict(zip(fails.qid, fails.cls))
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

<!-- Placeholder (prose): the fix of the largest class, applied by rule (no
choice by hand) on dev, then a single read of test: the baseline, the fix
and the stage change with the largest dev gain. The dev gain is optimistic,
because the fix was chosen from dev failures; test is the confirmation. The
cell prints the rule test was read with (fix, stage, threshold). Close of the
## with one paragraph, "beyond these fixes": the families of RAG improvement
this section does not measure (query transformation, iterative and adaptive
retrieval, retrieval learned with reinforcement learning, context
compression, structured indexes), pointing to sec_lm_encoders_fr. -->

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
    assert not rules and (RERUN or (RULE['fix'], RULE['best']) == (
        FIX, BEST)), 'the rule now picks another fix or stage'

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
    print(pd.DataFrame({'base': BC.cls.value_counts(), FIX: after.cls
                        .value_counts()}).reindex(CLASSES + ['correct'])
          .fillna(0).astype(int).T.to_string())
```

<!-- Placeholder (prose): the plot below: on the left the share of dev
questions in each failure class before and after the fix; on the right
accuracy and citation precision on dev of four configurations (closed book,
baseline, the best stage, the fix) with the test points of the baseline, the
best stage and the fix. Not a cumulative chain: the best stage and the fix
each change one thing in the baseline, and the fix does not include the best
stage's change; the caption says so. -->

```{.python .input #rag-failure-plot}
%%tab pytorch
if FIX and ready('closed', 'base', FIX):
    d2l.use_svg_display()
    _, axes = d2l.plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    series = {}
    for name, df in (('base', BC), ('fix', after)):
        sh = class_shares(df.cls)
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

<!-- Placeholder (prose, ## Marginalizing over Retrieved Documents): the
derivation. RAG-Sequence marginalizes the whole answer's likelihood over
the retrieved documents (eq_lm_rag_seq), RAG-Token marginalizes per token
(eq_lm_rag_token); the top-k approximation drops the retriever mass outside
the retrieved set (eq_lm_rag_topk); the gradient reaches the retriever's
scores as posterior minus prior (eq_lm_rag_grad), a marginal likelihood over
a latent choice as in sec_mdl-latent-em-elbo. The cell checks each identity
in float64 and computes the smallest case by hand: two documents, a
two-token answer. -->

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
print('two documents, two tokens: p_S = 0.6 * 0.9 * 0.2 + 0.4 * 0.1 * 0.8 '
      '= {:.3f}; p_T = (0.6 * 0.9 + 0.4 * 0.1)(0.6 * 0.2 + 0.4 * 0.8) = '
      '{:.4f}'.format(*p_seq_tok(pz, P)))
```

<!-- Placeholder (prose): what each configuration costs: prompt and
generated tokens and seconds per answer from the records, and the latency
of retrieval and of reranking 50 chunks, measured now. -->

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
