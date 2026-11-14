# Retrieval-Augmented Generation
:label:`sec_lm_rag`

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

<!-- Placeholder (stage 2, infra step): the cells below are the section's
shared library, written before the section; the prose around them comes with
the section. Cell ids follow the section specification. -->

```{.python .input #rag-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

<!-- Placeholder (stage 2): the registry of the section's artifacts: the book's question set and the API responses it was built from (tools/build_book_qa.py --mode replay rebuilds the set from them). -->

```{.python .input #rag-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-book-qa-v1'] = (  # corpus r3.1; 5,865 train, 240 dev, 240 test
    d2l.LM_DATA_URL + 'lm-book-qa-v1.zip',
    '3b8dba5ceae43c88274e52ce6ae7525884ea4168')
d2l.DATA_HUB['lm-book-qa-api-cache-v1'] = (  # its drafts, checks, verdicts
    d2l.LM_DATA_URL + 'lm-book-qa-api-cache-v1.zip',
    '962322fd53b3573daeaa50a92d3f53f25a25572d')
```

<!-- Placeholder (stage 2): the book's question set by split, and the chunks that count as relevant for an item under any chunking. -->

```{.python .input #rag-qa-def}
%%tab pytorch
#@save
def book_qa(split='dev', answerable=None, types=None, n=None, seed=0,
            root=os.path.join('..', 'data', 'lm')):
    """The book's question set (artifact lm-book-qa-v1), one split: a
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
    spans = item['gold_spans'] + (item['alt_spans'] if include_alt else [])
    return {i for s in spans for i in corpus.span_to_chunk_ids(
        s['doc_id'], s['char_start'], s['char_end'], min_overlap)}
```

<!-- Placeholder (stage 2): the fixed grounded prompt with numbered passages and the abstention string; without passages it is the closed-book prompt. -->

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

<!-- Placeholder (stage 2): the reference-based judge (Qwen3.5-4B, prompt RAG_JUDGE_V1): correct, faithful, supported. -->

```{.python .input #rag-judge-defs}
%%tab pytorch
#@save
def rag_judge(judge, tokenizer, item, result, context_texts, batch_size=8):
    """The judge's verdicts on an answer: 'correct' (every key fact stated,
    nothing contradicting the reference), 'faithful' (every statement
    supported by the passages given) and 'supported' (the cited passages
    alone support it). result is a RAGResult or a dict with answer and
    cited_ids (1-based passage numbers). An abstention is decided by string
    match, not by the judge: it is correct exactly for an unanswerable item.
    Without passages only 'correct' is judged. item, result and
    context_texts may also be lists (a list of verdicts comes back, judged
    in batches). A verdict missing from the judge's JSON after one retry is
    None."""
    single = isinstance(item, dict)
    items, results, contexts = (([item], [result], [context_texts])
                                if single else (item, result, context_texts))
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
    for attempt in range(2):  # one retry, with the malformed reply shown
        if not jobs:
            break
        gens = d2l.generate(judge, tokenizer, [m for _, _, m in jobs],
                            max_new_tokens=64, chat=True,
                            enable_thinking=False, batch_size=batch_size)
        retry = []
        for (i, keys, messages), g in zip(jobs, gens):
            parsed = d2l._rag_parse_json(g.text, keys)
            if parsed is None and attempt == 0:
                retry.append((i, keys, messages + [
                    {'role': 'assistant', 'content': g.text},
                    {'role': 'user', 'content': d2l.RAG_JUDGE_V1['retry']}]))
            for k in keys:
                verdicts[i][k] = None if parsed is None else parsed[k]
        jobs = retry
    return verdicts[0] if single else verdicts
```
