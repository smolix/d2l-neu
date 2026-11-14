# Sparse, Dense and Approximate Retrieval
:label:`sec_lm_retrieval`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "2.4 Sparse, Dense and Approximate Retrieval · T1").
Blueprint: docs/llm/blueprints/ch02_encoders.md section 2.4.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** An index trades recall for latency and memory, and BM25 is the floor a dense index must beat.
**Opening:** One query scanned against a million passage vectors takes longer than a reader will wait, and the reader's dense vectors miss a question whose answer contains an exact identifier that a word index would have found at once. The section builds the word index, the dense index and the approximate indexes over the book, measures what each trades, and closes on the question of what the model should read once the passages are found.
**Teaches:**
- Inverted indexes and BM25 (term-frequency saturation with k1, length normalization with b, the Lucene idf) as the baseline (`d2l.BM25Index`); the book corpus as the shared artifact, with its cleaning rules, two chunkings and document splits.
- Approximate nearest-neighbour indexes: inverted file lists from k-means, product quantization with asymmetric distance (`d2l.ProductQuantizer`), and navigable small-world graphs (through Faiss); recall against latency and memory as the one trade.
- Hybrid retrieval by reciprocal-rank fusion (the formula, cited by title) and by tuned score weights; reranking the fused list with Qwen3-Reranker-0.6B; Recall@k, MRR and nDCG with their definitions.
- Learned sparse retrieval and late interaction are named with one sentence each and left to the Further Reading notebook.
**Derivation:** Asymmetric distance computation in product quantization: a vector of dimension d split into M sub-vectors, each quantized to one of 2^b centroids, so a code costs M b bits; the distance from a query to a code is a sum of M table look-ups computed once per query; the memory per vector and the expected squared error as the sum of sub-quantizer distortions. Smallest numeric case: d = 4, M = 2, b = 1, with the two codebooks and one query distance computed by hand.
**Notebook:**
1. Build the book corpus with the default section-aware chunking and print its counts; tokenize for BM25 with identifier splitting (`d2l.bm25_tokenize`) and check `d2l.BM25Index` against an oracle library on 200 train queries (rank-1 agreement and relative score tolerance 1e-4).
2. Retrieve the answerable `dev` questions with BM25, with Qwen3-Embedding-0.6B (flat cosine search on the GPU), with reciprocal-rank fusion and with weighted fusion tuned on `train`; rerank the fused top 20 and top 100.
3. On a stress corpus of the book's chunks plus about 80,000 off-topic passages, build flat, IVF-Flat, IVF-PQ (M = 64, 8 bits) and HNSW indexes; sweep probe counts and search widths; record recall against exact top-10, batch-1 latency p50 and p95, bytes per vector and build time.
4. Plot recall against latency with marker size as memory; plot the fused and reranked stack with intervals.
5. Claim that survives noise: BM25 is within a stated band of the dense index on questions whose answers contain exact identifiers and below it elsewhere; fusion at least matches the better single retriever; HNSW reaches flat recall at a small fraction of the flat scan's latency on the stress corpus, and IVF-PQ reaches the lowest memory with a recall loss that depends on M. If at 84,000 vectors a flat scan is already fastest, the notebook says so and shows the crossover on a clearly labelled synthetic extension to a million vectors.
6. Reused pieces: `d2l.embed_texts`, `d2l.retrieval_metrics`, `d2l.paired_bootstrap`, `d2l.book_qa('train' | 'dev')`, `sec_perf_model`'s roofline arithmetic for the flat scan; new `d2l.book_corpus`, `d2l.BookRetriever`, `d2l.rrf_fuse`, `d2l.kmeans`, `d2l.IVFIndex`, `d2l.ProductQuantizer`, `d2l.rerank_yes_no`.
**Figures:** `fig_lm_ivf_pq` (a vector split into sub-vectors, the codebooks, the code, and the look-up table for one query); `fig_lm_index_tradeoff` (plot: recall against batch-1 latency for each index family along its knob, marker size proportional to memory, with the flat scan as one point).
**Restate / forward:** `sec_mdl-svd-low-rank` (a product quantizer is a block code; the analogy to low rank ends where the text says so). Forward: 2.5 and Chapter 10 consume this index.
**Exercises:** 5 to 7. Load-bearing: derive the memory per vector of product quantization against float16 and the number of table look-ups per query (derive); measure HNSW recall against search width and build time against the graph degree (measure); check the reciprocal-rank-fusion formula on two rankings of three items by hand and against `d2l.rrf_fuse` (check); open: add a learned sparse or a late-interaction row from the Further Reading notebook and place it on the trade-off plot.
**Writer notes:**
- The corpus artifact (`lm-book-r3-v1`): built by `tools/build_r3.py` from a pinned book commit (deterministic: sorted paths, no timestamps, a pinned tokenizer revision; CI rebuilds and compares per-file hashes); includes every `chapter_*` directory except the two retiring NLP chapters, the Language Models part (self-reference and leakage), the attic, preface, installation, notation and references; cleaning keeps PyTorch tabs only, cuts slide decks, rewrites `numref` into section titles, keeps math verbatim in the `display` view and strips it in the `sparse` view, drops exercises; two shipped chunkings (`sections`, structure-aware up to 512 Qwen3 tokens; `fixed`, 256-token windows with overlap 0 or 64); chunk ids `"{tag}:{doc_id}#{k:03d}"` stable within a corpus version; document splits `train/dev/test` by hash within each chapter (`manifest['doc_split']`), used by 2.5, 4.4 and Chapter 10. The full specification is blueprint ch02 2.4; keep it. Name it "the book corpus" in prose, never a code name.
- Exports (`01_interfaces.md`): `d2l.book_corpus(chunking='sections') -> d2l.BookCorpus` with chunk fields `id, doc_id, text, heading_path, char_start, char_end, n_tokens`; `d2l.BookRetriever.search(query, k, mode=...) -> list[d2l.Hit]` with `Hit(chunk_id, score, rank, text, heading_path, scores)`; `d2l.BM25Index(docs_tokens, k1=1.2, b=0.75, idf='lucene')`; `d2l.rrf_fuse(rankings, k=60, weights=None)`.
- Tuning discipline: BM25 parameters and fusion weights are tuned on `train` questions only (the notebook passes only `book_qa('train')` to the tuners); comparing Recall@k across chunk sizes is unfair (larger chunks win trivially), which 2.5 fixes with coverage at a token budget.
- Reference and tolerance: `bm25s` (Lucene variant, k1 = 1.2, b = 0.75) as the BM25 oracle; `pytrec_eval` for the metrics; Faiss CPU wheels for IVF-PQ and HNSW (if `faiss-cpu` fails to install, the from-scratch IVF and PQ carry the section and the HNSW row is dropped with a sentence); `nvidia-smi`-level machine state printed beside latencies.
- Runtime: the reranker over the top 100 for 240 queries is 24,000 forward passes of a 0.6B model; cut to the top 50 if it exceeds about 3 minutes. The stress embeddings (about 84,000 × 1,024 in float16, about 0.17 GB) are committed as `d2l-r3-nq-stress-emb`; `STRESS_N=20_000` and no HNSW build is the reduced path.
- Pitfalls to state: ANN recall against exact search is not task recall; fusing raw scores without normalization (ranks avoid it); reranking the top 10 and calling it a Recall@100 gain; HNSW memory is the vectors, not the graph, and deletes are expensive; "dense beats BM25" is a claim about a corpus, a query set and a model.
- SPLADE and ColBERT rows are in the Further Reading notebook (named in one sentence here); the chapter's judge is Qwen3.5-4B (no effect on 2.4).
- Committed artifacts: the corpus archive, the stress embeddings, the trained IVF-PQ codebooks, the BM25 index, the results JSON.
**Sources:**
- Papers: Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009; cited by title: term saturation and length normalization); [Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57) (the source of the derivation); [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://arxiv.org/abs/1603.09320) (HNSW); [The Faiss Library](https://arxiv.org/abs/2401.08281) (index families and their trade-offs); Cormack, Clarke and Büttcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (SIGIR 2009; cited by title: the fusion formula); [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) (the dense baseline and in-batch training); [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) (the cross-encoder reranker recipe); [BEIR](https://arxiv.org/abs/2104.08663) (BM25 as a strong zero-shot baseline across domains).
- Code and documentation: [Faiss](https://github.com/facebookresearch/faiss) and its [wiki](https://github.com/facebookresearch/faiss/wiki) (index factory strings, parameter guidance); [bm25s](https://github.com/xhluca/bm25s) (the BM25 oracle); [Pyserini](https://github.com/castorini/pyserini) (Lucene-backed sparse baselines); [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) (the yes/no scoring prompt).
- Courses, tutorials and blogs: [BEIR repository](https://github.com/beir-cellar/beir) (datasets and evaluation scripts); [Sentence Transformers documentation](https://sbert.net/) (semantic search and reranking examples); [CMU 11-868](https://llmsystem.github.io/llmsystem2026spring/) (the RAG and HNSW lecture).
-->

<!-- Placeholder (stage 2, cells step): every code cell of the section is
written; the prose comes later. Each placeholder names the `##` heading its
text goes under and what that text says. -->

<!-- Placeholder (prose, opening): start from the reader: one query scanned against a million passage vectors takes longer than a reader will wait, and a dense vector can miss a question whose answer holds an exact identifier that a word index finds at once. The finished retriever of this section (BM25 and the embedder of :numref:`sec_lm_text_embeddings`) answers the book's dev questions both ways first. -->

```{.python .input #retrieval-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #retrieval-extras}
%%tab pytorch
import bm25s
import collections
import faiss
import hashlib
import json
import os
import pandas as pd
import platform
import pytrec_eval
import re
import time
import transformers
import warnings
```

<!-- Placeholder (prose, opening): the models (the chapter's embedder and reranker, pinned), the flags of the CPU path (fewer training questions embedded, eight dev questions reranked live, the rest read from the registered results) and of regeneration (D2L_RERUN names ret-stress-emb, ret-rerank, ret-ann, or ret-crossover), and the machine the timings come from: latencies depend on it and on what else runs there. -->

```{.python .input #retrieval-config}
%%tab pytorch
EMB, RERANKER = d2l.DEFAULT_EMBEDDER, 'Qwen/Qwen3-Reranker-0.6B'
GPU = torch.cuda.is_available()
SMALL = not GPU  # the CPU path: fewer questions embedded and reranked live
DTYPE, DEVICE = (torch.bfloat16, 'cuda') if GPU else (torch.float32, 'cpu')
ARTIFACTS = {  # name: (registered key, the file a regeneration writes)
    'ret-stress-emb': ('lm-enc-r3-stress-emb-v1', 'lm-enc-r3-stress-emb.npz'),
    'ret-rerank': ('lm-enc-ret-rerank-v1', 'lm-enc-ret-rerank.json'),
    'ret-ann': ('lm-enc-ann-v1', 'lm-enc-ann.json'),
    'ret-crossover': ('lm-enc-ret-crossover-v1', 'lm-enc-ret-crossover.json')}
RERUN = d2l.lm_rerun(ARTIFACTS)
assert GPU or not RERUN & {'ret-stress-emb', 'ret-rerank'}, \
    'encoding and reranking for a regeneration need a GPU'
K_RRF, DEPTH = 60, 100  # fusion constant; candidates kept per retriever
RR_DEPTHS, RR_MAX_LEN, RR_BATCH = (20, 100), 1024, 32  # reranked prefixes
RR_LIVE = None if GPU else 8  # dev questions reranked live (None: all)
N_TUNE_DENSE = None if GPU else 1_000  # train questions embedded for fusion
BM25_GRID = dict(k1=(0.9, 1.2, 1.5, 2.0), b=(0.4, 0.75, 0.9),
                 split_identifiers=(True, False))
LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # dense weight
STRESS_N = None  # all; 20_000 is the reduced path for small machines
N_NQ_QUERIES, N_WARM, BUILD_THREADS = 1_000, 100, 8
NLIST = 1024 if STRESS_N is None else 128  # IVF cells: about sqrt(N)
NPROBES = (1, 2, 4, 8, 16, 32, 64)
PQ_MS, PQ_SEEDS = (64, 32), (0, 1, 2)  # sub-vectors; k-means seeds
HNSW_M, EF_CONSTRUCTION, EF_SEARCHES = 32, 200, (16, 32, 64, 128, 256)
QUERY = d2l._EMBEDDERS[EMB]  # pooling, padding side, query format
NQ_TASK = ('Given a web search query, retrieve relevant passages that '
           'answer the query')  # the model card's task for web questions
REV = {m: d2l.HF_ENCODER_REVISIONS[m] for m in (EMB, RERANKER)}
faiss.omp_set_num_threads(BUILD_THREADS)  # builds; timed searches use one
CONFIG = {  # what each result is made with; reading it checks this
    'ret-stress-emb': dict(model=EMB, revision=REV[EMB],
                           corpus=d2l.R3_VERSION, nq_skip=20_000,
                           n_queries=N_NQ_QUERIES, task=NQ_TASK, max_len=512),
    'ret-rerank': dict(model=RERANKER, revision=REV[RERANKER], embedder=EMB,
                       embedder_revision=REV[EMB], corpus=d2l.R3_VERSION,
                       bm25=d2l.BookRetriever.BM25_PARAMS, k_rrf=K_RRF,
                       depths=RR_DEPTHS, max_len=RR_MAX_LEN, batch=RR_BATCH,
                       instruction=d2l.BOOK_TASK, view='dense'),
    'ret-ann': dict(stress=ARTIFACTS['ret-stress-emb'][0], n=STRESS_N,
                    nlist=NLIST, nprobes=NPROBES, pq_ms=PQ_MS, seeds=PQ_SEEDS,
                    hnsw=(HNSW_M, EF_CONSTRUCTION), ef=EF_SEARCHES,
                    queries=N_NQ_QUERIES, threads=BUILD_THREADS),
    'ret-crossover': dict(stress=ARTIFACTS['ret-stress-emb'][0], n=10**6,
                          noise='median nearest-neighbour distance',
                          hnsw=(HNSW_M, EF_CONSTRUCTION),
                          threads=BUILD_THREADS)}
# The chunk vectors (ret-stress-emb, ret-rerank) and the dev questions
# (ret-rerank) these results depend on are added as sha1s once loaded.

def result_path(name):
    """The file this run regenerates for name, else the registered one."""
    key, file = ARTIFACTS[name]
    return d2l.lm_artifact(key, file if name in RERUN else None)

def read_result(name, strict=True):
    """The results of name, checked against this notebook's settings; with
    strict=False, None when they were made with other settings."""
    with open(result_path(name)) as f:
        result = json.load(f)
    if result['config'] == json.loads(json.dumps(CONFIG[name])):
        return result
    assert not strict, f'{name} was made with other settings: D2L_RERUN={name}'
    return None

def write_result(name, **fields):
    """Write a regenerated result with its settings and machine."""
    def write(path):
        with open(path, 'w') as f:
            json.dump(dict(config=CONFIG[name], machine=machine(), **fields),
                      f)
    d2l.lm_save_rerun(ARTIFACTS[name][1], write)

def machine():
    """CPU, cores, load, threads and GPU: the timings depend on them."""
    try:
        with open('/proc/cpuinfo') as f:
            cpu = next(l.split(':', 1)[1].strip() for l in f
                       if l.startswith('model name'))
    except (OSError, StopIteration):
        cpu = platform.processor() or platform.machine()
    gpu = 'none'
    if GPU:
        free, total = torch.cuda.mem_get_info()
        gpu = (f'{torch.cuda.get_device_name()}, {(total - free) / 2**30:.1f}'
               f' of {total / 2**30:.0f} GiB in use')
    return dict(cpu=cpu, cores=os.cpu_count(), load=os.getloadavg()[0],
                torch_threads=torch.get_num_threads(),
                faiss_threads=faiss.omp_get_max_threads(), gpu=gpu)

print(pd.DataFrame({'model': [EMB, RERANKER], 'revision': [
    REV[m][:12] for m in (EMB, RERANKER)]}, index=['embedder', 'reranker']))
print(f'GPU: {GPU}; regenerating: {", ".join(sorted(RERUN)) or "none"}')
print(machine())
```

<!-- Placeholder (stage 2): the book corpus, a snapshot of the book's sources at a named commit (built by `tools/build_r3.py`); the chunk vectors of the chapter's embedder and the four results of this section's regeneration runs are added to this cell when they are registered. -->

```{.python .input #retrieval-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-book-r3-v1'] = (  # book commit 098bf463, corpus r3.1
    d2l.LM_DATA_URL + 'lm-book-r3-v1.zip',
    '0b7f11e044a4838139ac02dffb6e2d7eb32b48b8')
```

<!-- Placeholder (prose, opening): the retriever this section ends with, used before it is built: BM25 alone and the embedder alone on the 220 answerable dev questions; Recall@10 of each with document-clustered intervals, the share of questions exactly one of them finds (their errors differ), and one question of each kind (the first in the question set's order): an identifier question BM25 finds and the embedder misses, and a question without an identifier the other way round. -->

```{.python .input #retrieval-opening}
%%tab pytorch
RET = d2l.BookRetriever()  # BM25 and the embedder over the book corpus
FULL = RET.corpus
DEV = d2l.book_qa('dev', answerable=True)
QUESTIONS = [q['question'] for q in DEV]
GOLD = [d2l.gold_chunk_ids(q, FULL) for q in DEV]
CLUSTER = np.array([q['gold_spans'][0]['doc_id'] for q in DEV])
IDENT = np.array(['identifier' in q['flags'] for q in DEV])

def signflip_p(d, cluster, n=10_000):
    """Two-sided sign-flip p-value of the mean of the per-document mean
    differences: each document's mean keeps or flips its sign at random."""
    m = np.array([d[cluster == c].mean() for c in np.unique(cluster)])
    flips = np.random.default_rng(0).choice([-1.0, 1.0], (n, len(m)))
    return np.mean(np.abs(flips @ m) >= abs(m.sum()) - 1e-12)

def boot(a, b=None, cluster=CLUSTER):
    """Mean of per-question scores a (or of a - b, paired) with its 95%
    interval, documents resampled. Marked *: fewer than 100 questions or
    30 documents; with fewer than 30 documents, questions are resampled (an
    interval that ignores the documents' correlation: likely too narrow),
    and a paired difference gets a document-level sign-flip p."""
    by_doc = len(set(cluster)) >= 30
    with warnings.catch_warnings():  # few items: the * and the note say so
        warnings.simplefilter('ignore')
        r = (d2l.bootstrap_ci(a, cluster=cluster if by_doc else None)
             if b is None else d2l.paired_bootstrap(
                 a, b, cluster=cluster if by_doc else None))
    f = '.3f' if b is None else '+.3f'
    out = f'{r.mean:{f}} [{r.ci[0]:{f}}, {r.ci[1]:{f}}]'
    if by_doc and len(a) >= 100:
        return out
    if by_doc or b is None:
        return out + '*'
    p = signflip_p(np.asarray(a, float) - np.asarray(b, float), cluster)
    return out + f'* (sign-flip p {p:.3f}, {len(set(cluster))} documents)'

HITS = {m: RET.search_batch(QUESTIONS, k=10, mode=m)
        for m in ('bm25', 'dense')}
FOUND = {m: d2l.retrieval_metrics([[h.chunk_id for h in hits] for hits in H],
                                  GOLD, ks=(10,))['recall@10']
         for m, H in HITS.items()}
RET.release()  # the query encoder leaves the GPU; the chunk vectors stay
print(f'{len(DEV)} dev questions; Recall@10: BM25 {boot(FOUND["bm25"])}, '
      f'dense {boot(FOUND["dense"])}')
print(f'found by exactly one: '
      f'{boot((FOUND["bm25"] != FOUND["dense"]).astype(float))}')
for kind, win, mask in (('identifier question', 'bm25', IDENT),
                        ('question without an identifier', 'dense', ~IDENT)):
    lose = 'dense' if win == 'bm25' else 'bm25'
    i = next((i for i in np.flatnonzero(mask)
              if FOUND[win][i] > FOUND[lose][i]), None)  # the first one
    if i is None:
        print(f'\nno {kind} that {win} finds and {lose} misses')
        continue
    print(f'\n{kind} ({DEV[i]["qid"]}): {QUESTIONS[i]}')
    for m in ('bm25', 'dense'):
        rank = next((h.rank for h in HITS[m][i] if h.chunk_id in GOLD[i]),
                    None)
        print(f'  {m}: first relevant chunk at rank {rank or "> 10"}')
```

<!-- Placeholder (prose, ## The Book as a Corpus): what the corpus includes (every chapter of the book at a named commit) and what it leaves out and why (the Language Models part: self-reference and leakage, since its notebooks hold questions, answers and results about this corpus; the Attic; front matter); the cleaning rules as a list (PyTorch tabs only, slide decks and exercises cut, references rewritten as titles, math kept in the text and removed for the word index); documents split into train, dev and test by a hash within each chapter. The loader below downloads the archive once and checks its version. -->

```{.python .input #retrieval-book-corpus-def}
%%tab pytorch
#@save
def book_corpus(chunking='sections', *, size=256, overlap=0, source='hub',
                root=os.path.join('..', 'data', 'lm')):
    """The book corpus (artifact lm-book-r3-v1) under one chunking:
    'sections' (structure-aware, up to 512 tokens) or 'fixed' (windows of
    size tokens sharing overlap tokens: 256 and 0 or 64 are shipped). With
    source='local' it is rebuilt in memory from the git checkout named by
    the environment variable D2L_BOOK_ROOT."""
    tag = 'sec512' if chunking == 'sections' else f'fix{size}o{overlap}'
    if tag not in ('sec512', 'fix256o0', 'fix256o64'):
        raise ValueError(f'no shipped chunking {chunking!r} with size '
                         f'{size} and overlap {overlap}')
    if source == 'local':
        import transformers
        rev = d2l.HF_REVISIONS[d2l.R3_TOKENIZER]
        tok = transformers.AutoTokenizer.from_pretrained(d2l.R3_TOKENIZER,
                                                         revision=rev)
        manifest, docs, chunks = d2l._r3_local(os.environ['D2L_BOOK_ROOT'],
                                                tok, rev)
        if 'lm-book-r3-v1' in d2l.DATA_HUB:  # what changed since the release
            shipped = d2l._r3_read_zip(d2l.download_keyed(
                'lm-book-r3-v1', root), ())[0]['files']
            drift = sorted(set(shipped.items()) ^ set(
                manifest['files'].items()))
            if drift:
                print(f'warning: {len({d for d, _ in drift})} files differ '
                      f'from the registered corpus')
        chunks = chunks[tag]
    else:
        path = d2l.download_keyed('lm-book-r3-v1', root)
        manifest, docs, chunks = d2l._r3_read_zip(path, (tag,))
        chunks = chunks[tag]
    if manifest['corpus_version'] != d2l.R3_VERSION:
        raise ValueError(
            f'{manifest["corpus_version"]} is not corpus version '
            f'{d2l.R3_VERSION}: delete {root}/lm-book-r3-v1 and download again')
    return d2l.BookCorpus(manifest, docs, chunks, tag)
```

<!-- Placeholder (prose, ## The Book as a Corpus): the manifest: the commit, documents per split, chunks per chunking, the token lengths of the section chunks, references whose target lies outside the corpus, the excluded directories; one chunk with its id ("{tag}:{doc_id}#{k}", stable within a corpus version), heading path and character span. Every fixed window is an exact slice of its document's text. The views: 'display' for a generator, 'dense' (long code cells cut to 30 lines) for an embedder, 'sparse' (math removed) for a word index. -->

```{.python .input #retrieval-corpus}
%%tab pytorch
FIXED = d2l.book_corpus('fixed', size=256, overlap=64)
m = FULL.manifest
print(f'book commit {m["book_commit"][:8]}, corpus {FULL.version}: '
      f'{m["counts"]["docs"]} documents, ' + ', '.join(
          f'{m["counts"]["docs_" + s]} {s}' for s in ('train', 'dev', 'test')))
print('chunks: ' + ', '.join(f'{m["counts"]["chunks_" + t]:,} {t}'
                             for t in m['chunking']))
print('sec512 tokens per chunk, quantiles:', m['token_quantiles']['sec512'])
print(f'{m["unresolved_refs"]["count"]} references to labels outside the '
      f'corpus')
print(pd.DataFrame(m['excluded']).groupby('reason').agg(
    directories=('directory', 'size'), files=('files', 'sum')))
c = FULL['sec512:chapter_transformers/kv-cache#001']
print(f'\n{c["id"]}\n{" > ".join(c["heading_path"])}\ncharacters '
      f'{c["char_start"]}-{c["char_end"]}, {c["n_tokens"]} tokens, '
      f'{c["kind"]}\n{c["text"][:300]} ...')
assert all(c['text'] == FIXED.docs[c['doc_id']]['text'][
    c['char_start']:c['char_end']] for c in FIXED.chunks)
```

<!-- Placeholder (prose, ## The Book as a Corpus): the two chunkings on one short document (a chapter introduction): sections pack whole paragraphs, lists, equations and code cells under their heading up to 512 tokens; fixed windows of 256 tokens with 64 shared cut through sentences. The library's section chunker reproduces the stored chunks of that document. -->

```{.python .input #retrieval-chunking-demo}
%%tab pytorch
tok = transformers.AutoTokenizer.from_pretrained(
    d2l.R3_TOKENIZER, revision=d2l.HF_REVISIONS[d2l.R3_TOKENIZER])
doc = FULL.docs['chapter_generative-adversarial-networks/index']
for name, chunks in (
        ('sections', d2l.chunk_sections(doc, tok)),
        ('fixed, 256 tokens, 64 shared', d2l.chunk_fixed(
            doc, tok, size=256, overlap=64, tag='fix256o64'))):
    print(f'{name}: {len(chunks)} chunks of {doc["n_tokens"]} tokens')
    for c in chunks:
        print(f'  {c["id"]:58s} chars {c["char_start"]:4d}-'
              f'{c["char_end"]:4d}  {c["n_tokens"]:3d} tokens  '
              f'{c["text"][:28]!r}')
assert [(c['char_start'], c['char_end']) for c in d2l.chunk_sections(
    doc, tok)] == [(c['char_start'], c['char_end']) for c in FULL.chunks
                   if c['doc_id'] == doc['doc_id']]
```

<!-- Placeholder (prose, ## The Book as a Corpus): the question set in one sentence (:numref:`sec_lm_rag`), and the metrics with their definitions (static table tab_lm_ret_metrics): Recall@k is 1 when some relevant chunk is among the first k (the answer was retrieved; trec_eval calls it success@k), MRR@10 the reciprocal rank of the first relevant chunk, nDCG@10 the discounted gain against the ideal order. A worked example of three queries with five ranked chunks each, computed by hand in the text and checked against pytrec_eval. -->

```{.python .input #retrieval-metrics-demo}
%%tab pytorch
ranked = [list('abcde'), list('fghij'), list('klmno')]
gold = [{'c'}, {'f', 'i'}, {'z'}]  # the third query's chunk is missed
ours = d2l.retrieval_metrics(ranked, gold, ks=(1, 5))
qrel = {f'q{i}': {d: 1 for d in g} for i, g in enumerate(gold)}
run = {f'q{i}': {d: float(5 - j) for j, d in enumerate(r)}
       for i, r in enumerate(ranked)}
ref = pytrec_eval.RelevanceEvaluator(
    qrel, {'success.1,5', 'recip_rank', 'ndcg_cut.10'}).evaluate(run)
table = pd.DataFrame({
    'Recall@1': ours['recall@1'], 'Recall@5': ours['recall@5'],
    'MRR@10': ours['mrr@10'], 'nDCG@10': ours['ndcg@10'],
    'trec_eval': [', '.join(f'{ref[q][k]:.3f}' for k in (
        'success_1', 'success_5', 'recip_rank', 'ndcg_cut_10'))
                  for q in qrel]}, index=list(qrel))
print(table.round(3))
for i, q in enumerate(qrel):
    assert np.allclose([ours[k][i] for k in ('recall@1', 'recall@5', 'mrr@10',
                                             'ndcg@10')], [ref[q][k] for k in (
        'success_1', 'success_5', 'recip_rank', 'ndcg_cut_10')])
```

<!-- Placeholder (prose, ## Word Indexes and BM25): an inverted index maps each term to the chunks that hold it; as a matrix it is a sparse term-by-chunk matrix of weights, and scoring a query is one sparse product. Index terms for technical text: words and numbers lower-cased; an identifier such as d2l.book_corpus or BookRetriever is kept whole and also split at dots, underscores and camel case, so a question that names a part still matches. -->

```{.python .input #retrieval-bm25-tokenize}
%%tab pytorch
#@save
def bm25_tokenize(text, split_identifiers=True):
    """Lower-cased words and numbers of text; an identifier such as
    d2l.book_corpus or BookRetriever also gives its parts (d2l, book,
    corpus; book, retriever) when split_identifiers is set."""
    out = []
    for tok in re.findall(r'\w+(?:\.\w+)*', text):
        out.append(tok.lower())
        if not split_identifiers or re.fullmatch(r'[\d.]+', tok):
            continue
        parts = []
        for p in re.split(r'[._]+', tok):
            if re.search(r'[a-z][A-Z]|[A-Z][A-Z][a-z]', p):  # camelCase
                parts += re.findall(r'[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+|\d+', p)
            elif p:
                parts.append(p)
        if len(parts) > 1:
            out += [p.lower() for p in parts]
    return out
```

<!-- Placeholder (prose, ## Word Indexes and BM25): BM25 (eq_lm_bm25, Robertson and Zaragoza, cited by title): term frequency saturates through k1, document length is normalized through b against the mean length, and Lucene's idf log(1 + (N - df + 0.5)/(df + 0.5)) weighs rare terms up; stored as the weight matrix W[t, d], so a query's scores are the sum of its terms' rows. -->

```{.python .input #retrieval-bm25-def}
%%tab pytorch
#@save
class BM25Index:
    """BM25 over tokenized documents: W[t, d] = idf(t) tf / (tf + k1 (1 - b
    + b |d| / avgdl)) with Lucene's idf log(1 + (N - df + 0.5) / (df +
    0.5)), stored as a sparse matrix; a query's scores are the sum of the
    rows of its tokens."""
    def __init__(self, docs_tokens, k1=1.2, b=0.75, idf='lucene'):
        from scipy import sparse
        if idf != 'lucene':
            raise ValueError("only idf='lucene' is implemented")
        self.k1, self.b, self.vocab = k1, b, {}
        rows, cols = [], []
        for d, toks in enumerate(docs_tokens):
            for t in toks:
                rows.append(self.vocab.setdefault(t, len(self.vocab)))
                cols.append(d)
        N, V = len(docs_tokens), len(self.vocab)
        tf = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(V, N))  # duplicates are summed
        tf.sum_duplicates()
        self.doc_len = np.array([len(t) for t in docs_tokens], dtype=float)
        df = np.diff(tf.indptr)
        self.idf = np.log1p((N - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * self.doc_len / self.doc_len.mean())
        tf.data = (np.repeat(self.idf, df) * tf.data
                   / (tf.data + norm[tf.indices]))
        self.W = tf  # (vocabulary, documents)

    def scores(self, query_tokens):
        """BM25 score of every document (a repeated token counts twice)."""
        ids = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not ids:
            return np.zeros(self.W.shape[1])
        return np.asarray(self.W[ids].sum(axis=0)).ravel()

    def search(self, query_tokens, k):
        """Indices and scores of the k best documents, best first."""
        s = self.scores(query_tokens)
        ids = np.argsort(-s, kind='stable')[:k]
        return ids, s[ids]
```

<!-- Placeholder (prose, ## Word Indexes and BM25): checked against bm25s, a fast BM25 library, given the same tokens and its Lucene variant at k1 = 1.2 and b = 0.75: the first-ranked chunk agrees for the 200 training questions and the scores agree to 1e-4. -->

```{.python .input #retrieval-bm25-oracle}
%%tab pytorch
TOKENS = {s: [d2l.bm25_tokenize(t, s) for t in FULL.texts('sparse')]
          for s in (True, False)}  # the chunks, with and without splitting
TRAIN = d2l.book_qa('train', answerable=True)
index = d2l.BM25Index(TOKENS[True], k1=1.2, b=0.75)
oracle = bm25s.BM25(method='lucene', k1=1.2, b=0.75)
oracle.index(bm25s.tokenization.Tokenized(
    ids=[[index.vocab[t] for t in toks] for toks in TOKENS[True]],
    vocab=index.vocab), show_progress=False)
agree, worst = [], 0.0
for q in TRAIN.subset(200, seed=0):
    toks = d2l.bm25_tokenize(q['question'])
    ids = [index.vocab[t] for t in toks if t in index.vocab]
    if not ids:  # no word of the question occurs in the book
        continue
    s, r = index.scores(toks), oracle.get_scores(ids)
    agree.append(np.argmax(s) == np.argmax(r))
    worst = max(worst, np.max(np.abs(s - r) / np.maximum(np.abs(r), 1e-6)))
print(f'rank-1 agreement {np.mean(agree):.3f} on {len(agree)} questions; '
      f'largest relative score difference {worst:.1e}')
assert np.mean(agree) >= 0.99 and worst <= 1e-4
```

<!-- Placeholder (prose, ## Word Indexes and BM25): k1 and b (and whether identifiers are split) are chosen on the training questions only, never on dev: the grid, the rule (Recall@10, then nDCG@10, then nearest to the usual 1.2 and 0.75, then splitting), and the stored choice of the retriever, which the assertion holds to it. Scoring all training questions at once is one sparse product of the queries' term counts with W. -->

```{.python .input #retrieval-bm25-tune}
%%tab pytorch
def bm25_scores(index, token_lists):
    """BM25 scores (queries, documents): the queries' term counts times W."""
    from scipy import sparse
    pairs = [(i, index.vocab[t]) for i, toks in enumerate(token_lists)
             for t in toks if t in index.vocab]
    rows, cols = zip(*pairs) if pairs else ((), ())
    counts = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(len(token_lists), len(index.vocab)))
    return (counts @ index.W).toarray()

def top_ids(S, k, corpus=FULL):
    """Chunk ids of the k best scores of each row of S, best first."""
    order = np.argsort(-S, axis=1, kind='stable')[:, :k]
    return [[corpus.ids[i] for i in row] for row in order]

TRAIN_GOLD = [d2l.gold_chunk_ids(q, FULL) for q in TRAIN]
TRAIN_TOKENS = {s: [d2l.bm25_tokenize(q['question'], s) for q in TRAIN]
                for s in (True, False)}
rows = []
for s in BM25_GRID['split_identifiers']:
    for k1 in BM25_GRID['k1']:
        for b in BM25_GRID['b']:
            index = d2l.BM25Index(TOKENS[s], k1=k1, b=b)
            ranked = [r for j in range(0, len(TRAIN), 1024) for r in top_ids(
                bm25_scores(index, TRAIN_TOKENS[s][j:j + 1024]), 10)]
            m = d2l.retrieval_metrics(ranked, TRAIN_GOLD, ks=(10,))
            rows.append(dict(split_identifiers=s, k1=k1, b=b,
                             recall=m['recall@10'].mean(),
                             ndcg=m['ndcg@10'].mean(),
                             dist=math.hypot(k1 - 1.2, b - 0.75)))
GRID = pd.DataFrame(rows)
best = GRID.sort_values(['recall', 'ndcg', 'dist', 'split_identifiers'],
                        ascending=[False, False, True, False]).iloc[0]
BM25_BEST = dict(k1=float(best.k1), b=float(best.b),
                 split_identifiers=bool(best.split_identifiers))
print(f'Recall@10 of {len(TRAIN):,} training questions:')
print(GRID.pivot_table('recall', ['split_identifiers', 'k1'], 'b').round(3))
print('chosen:', BM25_BEST, f'(nDCG@10 {best.ndcg:.3f})')
assert BM25_BEST == d2l.BookRetriever.BM25_PARAMS, \
    'the stored BM25 parameters of BookRetriever differ from this choice'
```

<!-- Placeholder (prose, ## Dense Retrieval, Fusion and Reranking): the dev questions through BM25 with the chosen and with the usual parameters (the difference is measured below, C8), timed. -->

```{.python .input #retrieval-bm25-dev}
%%tab pytorch
RANKED, MS = {}, {}  # top-100 chunk ids per dev question; ms per question
for name, p in (('BM25 (k1 1.2, b 0.75)', dict(
        k1=1.2, b=0.75, split_identifiers=BM25_BEST['split_identifiers'])),
                ('BM25', BM25_BEST)):
    index = d2l.BM25Index(TOKENS[p['split_identifiers']], k1=p['k1'],
                          b=p['b'])
    t0 = time.perf_counter()
    S_BM25 = bm25_scores(index, [d2l.bm25_tokenize(
        q, p['split_identifiers']) for q in QUESTIONS])
    RANKED[name] = top_ids(S_BM25, DEPTH)
    MS[name] = (time.perf_counter() - t0) / len(DEV) * 1e3
BM25 = index  # the chosen parameters; S_BM25 holds their dev scores
assert np.allclose(S_BM25, [RET.bm25.scores(d2l.bm25_tokenize(
    q, BM25_BEST['split_identifiers'])) for q in QUESTIONS])
print(f'{MS["BM25"]:.2f} ms per question, including tokenization')
```

<!-- Placeholder (prose, ## Dense Retrieval, Fusion and Reranking): flat dense search is a matrix product: the question's vector against all N chunk vectors of m coordinates, 2Nm FLOPs and Nm elements read; at batch 1 the arithmetic intensity is one FLOP per two bytes (float16), so a scan of a matrix larger than the cache is memory-bound (roofline of :numref:`sec_perf_model`), and a batch of B questions reuses each vector B times. The chunk vectors are the registered ones; 64 chunks are encoded again here and must agree with them. -->

```{.python .input #retrieval-dense}
%%tab pytorch
X = RET.chunk_vectors()  # (N, m) unit vectors, float16, on DEVICE
X_SHA1 = hashlib.sha1(X.cpu().numpy().tobytes()).hexdigest()
for name in ('ret-stress-emb', 'ret-rerank'):  # results made from them
    CONFIG[name]['chunk_vectors'] = X_SHA1
CONFIG['ret-rerank']['dev_questions'] = hashlib.sha1(json.dumps(
    [[q['qid'], q['question']] for q in DEV]).encode()).hexdigest()
TEXTS = FULL.texts('dense')  # what the embedder (and the reranker) reads
encoder = d2l.load_encoder(EMB, padding_side=QUERY['padding_side'],
                           dtype=DTYPE)

def embed_queries(encoder, questions, task=d2l.BOOK_TASK):
    """Unit vectors of questions with the model card's instruction."""
    return d2l.embed_texts(*encoder, [QUERY['query_format'].format(
        task=task, query=q) for q in questions], pooling=QUERY['pooling'],
                           max_len=512)

check = np.random.default_rng(0).choice(len(FULL), 64 if GPU else 16,
                                        replace=False)
again = d2l.embed_texts(*encoder, [TEXTS[i] for i in check],
                        pooling=QUERY['pooling'], max_len=1024, batch_size=16)
cos = (again * X[torch.as_tensor(check)].float().cpu()).sum(1)
print(f'chunk vectors sha1 {X_SHA1[:12]}; {len(check)} chunks encoded '
      f'again: min cosine {cos.min():.4f}')
assert cos.min() >= (0.999 if GPU else 0.99)

def dense_scores(Q):
    """Cosines (questions, chunks) as the retriever computes them."""
    return (Q.to(X.device, X.dtype) @ X.T).float().cpu().numpy()

t0 = time.perf_counter()
Q_DEV = embed_queries(encoder, QUESTIONS)
S_DENSE = dense_scores(Q_DEV)
RANKED['dense'] = [[FULL.ids[i] for i in row] for row in torch.from_numpy(
    S_DENSE).topk(DEPTH, dim=1).indices.tolist()]  # as the retriever ranks
MS['dense'] = (time.perf_counter() - t0) / len(DEV) * 1e3
assert all(r[:10] == [h.chunk_id for h in hits]
           for r, hits in zip(RANKED['dense'], HITS['dense']))
TUNE = TRAIN if N_TUNE_DENSE is None else TRAIN.subset(N_TUNE_DENSE, seed=0)
Q_TUNE = embed_queries(encoder, [q['question'] for q in TUNE])
del encoder
d2l.lm_free()
for name in ('BM25', 'dense'):
    r10 = d2l.retrieval_metrics(RANKED[name], GOLD, ks=(10,))['recall@10']
    print(f'{name}: dev Recall@10 {boot(r10)}, {MS[name]:.2f} ms per question')
```

<!-- Placeholder (prose, ## Dense Retrieval, Fusion and Reranking): the scan itself, timed at batch 1 and with all dev questions in one batch, against the roofline's estimate (on the GPU of the measurement; on the CPU path only the times): well under a millisecond per question either way (C7). The book's matrix (11 MB) fits in the GPU's 72 MB L2 cache, so at batch 1 the time is mostly the cost of launching the kernels and the measurement cannot show the memory bound; a synthetic matrix of 2^17 random unit vectors (256 MiB, GPU only) is read from memory on every scan and shows the time approaching bytes over bandwidth at batch 1 and the reuse of a batch. -->

```{.python .input #retrieval-dense-timing}
%%tab pytorch
def scan_seconds(Q, X, reps=50):
    """Median seconds of the scores Q X^T and their top 10."""
    times = []
    for r in range(reps + 5):  # the first five warm up
        if GPU:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        (Q @ X.T).topk(10, dim=1)
        if GPU:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times[5:])

m = X.shape[1]
P_PEAK, BANDWIDTH = 165.2e12, 1.008e12  # RTX 4090: bf16 FLOP/s, bytes/s
MATRICES = {'book': X}  # 11 MB: it stays in a GPU's L2 cache (72 MB here)
if GPU:  # 2^17 random unit vectors, 256 MiB: read from memory every time
    g = torch.Generator(device='cuda').manual_seed(0)
    R = torch.randn(2**17, m, generator=g, device='cuda')
    MATRICES['synthetic'] = (R / R.norm(dim=1, keepdim=True)).to(X.dtype)
    del R
for name, Xm in MATRICES.items():
    for B in (1, len(DEV)):
        t = scan_seconds(Q_DEV[:B].to(Xm.device, Xm.dtype), Xm)
        flops, nbytes = 2 * len(Xm) * m * B, 2 * (len(Xm) + B) * m  # 16 bit
        line = (f'{name:9s} N {len(Xm):7,}, batch {B:3d}: {t * 1e3:.3f} ms, '
                f'{t / B * 1e3:.4f} ms per question; {flops / 1e9:5.2f} '
                f'GFLOP, {nbytes / 2**20:5.1f} MiB; achieved '
                f'{nbytes / t / 1e9:4.0f} GB/s, {flops / t / 1e12:5.2f} '
                f'TFLOP/s')
        if GPU and '4090' in torch.cuda.get_device_name():
            roof = max(flops / P_PEAK, nbytes / BANDWIDTH)
            line += f'; roofline {roof * 1e3:.4f} ms'
        print(line)
del MATRICES
```

<!-- Placeholder (prose, ## Dense Retrieval, Fusion and Reranking): reciprocal-rank fusion (eq_lm_rrf, Cormack, Clarke and Büttcher, cited by title): each chunk scores the sum over the rankings of 1/(k + rank), with k = 60; it needs no score scales, only ranks. -->

```{.python .input #retrieval-rrf-def}
%%tab pytorch
#@save
def rrf_fuse(rankings, k=60, weights=None):
    """Reciprocal-rank fusion: each id scores sum_r w_r / (k + rank_r) over
    the rankings (lists of ids, best first) that contain it. Returns
    [(id, score)], best first, ties in order of first appearance."""
    weights = [1.0] * len(rankings) if weights is None else weights
    score, first = {}, {}
    for w, ranking in zip(weights, rankings):
        for r, i in enumerate(ranking, start=1):
            score[i] = score.get(i, 0.0) + w / (k + r)
            first.setdefault(i, len(first))
    return sorted(score.items(), key=lambda x: (-x[1], first[x[0]]))
```

<!-- Placeholder (prose, ## Dense Retrieval, Fusion and Reranking): two ways to fuse: RRF of the two top-100 lists, and a weighted sum of the two scores after z-normalizing each over the chunks (raw BM25 scores and cosines live on different scales: summing them unnormalized is a pitfall), with the weight lambda of the dense score chosen on training questions. The two retrievers fail on different questions: the share that exactly one finds, and the recall of an oracle that takes whichever finds it. -->

```{.python .input #retrieval-fusion}
%%tab pytorch
def zscore(S):
    """Each row of S shifted and scaled to mean 0 and variance 1."""
    sd = S.std(1, keepdims=True)
    return np.where(sd > 0, (S - S.mean(1, keepdims=True)) / np.where(
        sd > 0, sd, 1), 0)

def best_lambda(S_bm25, S_dense, gold):
    """The dense weight of the z-scored sum with the best Recall@10 (then
    nDCG@10, then the weight nearest 0.5) on the questions given."""
    Zb, Zd, rows = zscore(S_bm25), zscore(S_dense), []
    for lam in LAMBDAS:
        m = d2l.retrieval_metrics(top_ids(lam * Zd + (1 - lam) * Zb, 10),
                                  gold, ks=(10,))
        rows.append((-m['recall@10'].mean(), -m['ndcg@10'].mean(),
                     abs(lam - 0.5), lam))
    return min(rows)[-1]

LAMBDA = best_lambda(bm25_scores(BM25, [d2l.bm25_tokenize(
    q['question'], BM25_BEST['split_identifiers']) for q in TUNE]),
                     dense_scores(Q_TUNE),
                     [d2l.gold_chunk_ids(q, FULL) for q in TUNE])
t0 = time.perf_counter()
RANKED['RRF'] = [[i for i, _ in d2l.rrf_fuse([b, d], k=K_RRF)][:DEPTH]
                 for b, d in zip(RANKED['BM25'], RANKED['dense'])]
MS['RRF'] = (MS['BM25'] + MS['dense']
             + (time.perf_counter() - t0) / len(DEV) * 1e3)
t0 = time.perf_counter()
RANKED['weighted'] = top_ids(LAMBDA * zscore(S_DENSE)
                             + (1 - LAMBDA) * zscore(S_BM25), DEPTH)
MS['weighted'] = (MS['BM25'] + MS['dense']
                  + (time.perf_counter() - t0) / len(DEV) * 1e3)
M = {n: d2l.retrieval_metrics(r, GOLD) for n, r in RANKED.items()}
print(f'dense weight lambda = {LAMBDA} (chosen on {len(TUNE):,} training '
      f'questions)')
print(pd.DataFrame({n: {'Recall@10': boot(M[n]['recall@10']),
                        'nDCG@10': boot(M[n]['ndcg@10']),
                        'Recall@100': f'{M[n]["recall@100"].mean():.3f}'}
                    for n in ('BM25', 'dense', 'RRF', 'weighted')}).T)
b10, d10 = M['BM25']['recall@10'], M['dense']['recall@10']
print(f'found by exactly one of BM25 and dense: '
      f'{boot((b10 != d10).astype(float))}; by either: '
      f'{boot(np.maximum(b10, d10))}')
```

<!-- Placeholder (prose, ## Dense Retrieval, Fusion and Reranking): bi-encoders encode question and passage apart (vectors computed once, compared by a dot product); a cross-encoder reads them together and scores the pair, more accurately and at a forward pass per pair. Qwen3-Reranker-0.6B is a decoder asked a yes/no question about each pair; its score is the probability of "yes" against "no" as the next token, the label scoring of :numref:`sec_lm_prompting`. -->

```{.python .input #retrieval-rerank-def}
%%tab pytorch
#@save
def rerank_yes_no(model, tokenizer, query, docs, instruction=None,
                  batch_size=16, max_len=1024):
    """P(yes) of a yes/no reranker (Qwen3-Reranker) that each document
    meets the query: the model card's prompt with the document cut to fit
    max_len tokens, scored by the normalized probability of 'yes' against
    'no' as the next token."""
    instruction = instruction or ('Given a web search query, retrieve '
                                  'relevant passages that answer the query')
    head = ('<|im_start|>system\nJudge whether the Document meets the '
            'requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            f'<|im_start|>user\n<Instruct>: {instruction}\n'
            f'<Query>: {query}\n<Document>: ')
    tail = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    enc = lambda s: tokenizer(s, add_special_tokens=False)['input_ids']
    room = max_len - len(enc(head)) - len(enc(tail))
    prompts = [head + tokenizer.decode(enc(d)[:room]) + tail for d in docs]
    lp = d2l.label_logprobs(model, tokenizer, prompts, ['yes', 'no'],
                            sep='', batch_size=batch_size)
    return torch.exp(lp[:, 0]).numpy()
```

<!-- Placeholder (prose, ## Dense Retrieval, Fusion and Reranking): the reranker reads each question with the instruction the embedder gets (BOOK_TASK) and each chunk in the dense view (heading line first), the inputs fixed before any dev number was seen; it reorders the first 20 of the fused list, live, and the first 100 with the registered scores (D2L_RERUN=ret-rerank recomputes them: about 22,000 pairs); both rows reorder the live fused lists, the registered scores are stored per chunk, and the live scores are checked against them. On the CPU path both rows use the registered scores. Milliseconds per question for each depth: the reranker's cost grows with the depth, its gain does not (a reranker cannot lift Recall@k beyond the depth it reads, pitfall). -->

```{.python .input #retrieval-rerank}
%%tab pytorch
reranker = d2l.load_hf(RERANKER, revision=REV[RERANKER], dtype=DTYPE)
TEXT_OF = dict(zip(FULL.ids, TEXTS))  # the dense view: headings, then text
D0, D1 = RR_DEPTHS

def rerank(rows, depth):
    """P(yes) of the first depth fused chunks of the dev questions rows, as
    {question id: {chunk id: P(yes)}} in fused order, and the seconds per
    question. The first D0 chunks are one call, as the live run scores
    them, so the two runs share their batches."""
    P, t0 = {}, time.perf_counter()
    for i in rows:
        ids = RANKED['RRF'][i][:depth]
        p = np.concatenate([d2l.rerank_yes_no(
            *reranker, QUESTIONS[i], [TEXT_OF[c] for c in part],
            instruction=d2l.BOOK_TASK, batch_size=RR_BATCH,
            max_len=RR_MAX_LEN) for part in (ids[:D0], ids[D0:]) if part])
        P[DEV[i]['qid']] = dict(zip(ids, p.tolist()))
    return P, (time.perf_counter() - t0) / len(rows)

def reorder(fused, p_yes, depth):
    """The fused list with its first depth ids sorted by P(yes) (by chunk
    id; ties and ids without a score keep the fused order)."""
    head = fused[:depth]
    return (sorted([c for c in head if c in p_yes], key=lambda c: -p_yes[c])
            + [c for c in head if c not in p_yes] + fused[depth:])

if 'ret-rerank' in RERUN:
    P, sec = rerank(range(len(DEV)), D1)
    write_result('ret-rerank', p_yes=P, seconds_per_question=sec)
REG = read_result('ret-rerank')
live = range(len(DEV)) if RR_LIVE is None else range(RR_LIVE)
P0, sec0 = rerank(live, D0)  # the shorter depth, live
same = np.mean([list(REG['p_yes'][q['qid']]) == RANKED['RRF'][i][:D1]
                for i, q in enumerate(DEV)])
cover = np.mean([c in REG['p_yes'][q['qid']] for i, q in enumerate(DEV)
                 for c in RANKED['RRF'][i][:D1]])
diff = np.array([abs(p - REG['p_yes'][qid][c]) for qid, ps in P0.items()
                 for c, p in ps.items() if c in REG['p_yes'][qid]])
print(f'live fused top {D1} as registered for {same:.3f} of the questions;'
      f' registered P(yes) for {cover:.4f} of their chunks')
print(f'live against registered P(yes), {len(diff):,} pairs: 99th '
      f'percentile of |difference| {np.quantile(diff, 0.99):.4f}, largest '
      f'{diff.max():.4f}')
assert cover >= (0.99 if GPU else 0.9), 'registered for other fused lists'
assert not GPU or np.quantile(diff, 0.99) <= 0.02, 'P(yes) disagree'
for depth in RR_DEPTHS:  # every row reorders the live fused lists
    scores = P0 if depth == D0 and RR_LIVE is None else REG['p_yes']
    RANKED[f'RRF + rerank@{depth}'] = [
        reorder(RANKED['RRF'][i], scores[q['qid']], depth)
        for i, q in enumerate(DEV)]
    MS[f'RRF + rerank@{depth}'] = MS['RRF'] + 1e3 * (
        sec0 if depth == D0 else REG['seconds_per_question'])
print(f'reranking {D0} chunks: {sec0 * 1e3:.0f} ms per question on '
      f'{DEVICE} (live); {D1} chunks: '
      f'{REG["seconds_per_question"] * 1e3:.0f} ms per question '
      f'(registered: {REG["machine"]["gpu"].split(",")[0]}); scores of '
      f'rerank@{D0}: {"live" if RR_LIVE is None else "registered"}, of '
      f'rerank@{D1}: registered')
del reranker
d2l.lm_free()
```

<!-- Placeholder (prose, ## Dense Retrieval, Fusion and Reranking): the whole stack on the dev questions with document-clustered intervals and time per question; the paired differences that carry the claims: tuned against usual BM25 (C8), dense against BM25 overall and split by whether the question names an identifier (C2; an interval marked * comes from fewer than 100 questions or 30 documents; the identifier questions come from fewer than 30 documents, so their interval resamples questions and is likely too narrow, and a difference there counts as excluding 0 only if the document-level sign-flip test agrees), fusion against each single retriever (C3), reranking at 20 and at 100 (C4). "Dense beats BM25" is a claim about a corpus, a question set and a model. -->

```{.python .input #retrieval-stack}
%%tab pytorch
R0, R1 = (f'RRF + rerank@{n}' for n in RR_DEPTHS)
ORDER = ['BM25 (k1 1.2, b 0.75)', 'BM25', 'dense', 'RRF', 'weighted', R0, R1]
M = {n: d2l.retrieval_metrics(RANKED[n], GOLD) for n in ORDER}
KS = ('recall@1', 'recall@5', 'recall@10', 'mrr@10', 'ndcg@10')
print(pd.DataFrame({n: {k: boot(M[n][k]) for k in KS} | {
    'recall@100': f'{M[n]["recall@100"].mean():.3f}',
    'ms': f'{MS[n]:.1f}'} for n in ORDER}).T.to_string())
better = max(('BM25', 'dense'), key=lambda n: M[n]['recall@10'].mean())
DELTAS = [  # (row, against, metric, questions)
    ('BM25', 'BM25 (k1 1.2, b 0.75)', 'recall@10', 'all'),
    *[('dense', 'BM25', k, sub) for k in ('recall@10', 'ndcg@10')
      for sub in ('all', 'identifier', 'other')],
    *[('RRF', n, k, 'all') for n in ('BM25', 'dense')
      for k in ('recall@10', 'ndcg@10')],
    ('weighted', 'RRF', 'recall@10', 'all'),
    ('weighted', 'RRF', 'ndcg@10', 'all'),
    (R0, 'RRF', 'ndcg@10', 'all'), (R0, 'RRF', 'recall@10', 'all'),
    (R1, R0, 'ndcg@10', 'all'), (R1, R0, 'recall@10', 'all')]
rows = []
for a, b, k, sub in DELTAS:
    keep = {'all': IDENT | ~IDENT, 'identifier': IDENT, 'other': ~IDENT}[sub]
    rows.append({'difference': f'{a} - {b}', 'metric': k,
                 'questions': f'{sub} ({keep.sum()})', 'mean [95% CI]': boot(
                     M[a][k][keep], M[b][k][keep], CLUSTER[keep])})
print(pd.DataFrame(rows).to_string(index=False))
print(f'the better single retriever on dev is {better}')
print('* fewer than 100 questions or 30 documents: the interval is '
      'unreliable; with fewer than 30 documents it resamples questions,\n'
      '  ignores their correlation within a document and is likely too '
      "narrow; the sign-flip p over the documents' mean differences decides")
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 2, figsize=(8.5, 3.6), layout='constrained')
for ax, k in zip(axes, ('recall@10', 'ndcg@10')):
    cis = [d2l.bootstrap_ci(M[n][k], cluster=CLUSTER) for n in ORDER]
    d2l.plot_intervals(ORDER, {k: ([r.mean for r in cis], [
        r.ci[0] for r in cis], [r.ci[1] for r in cis])}, 'retriever',
                       f'dev {k}', axes=ax)
```

<!-- Placeholder (prose, ## Approximate Nearest Neighbours): an exact scan of a million vectors costs N m per query; an index looks at a fraction. Inverted files: k-means cuts the space into nlist cells around centroids; a query scans the vectors of its nprobe nearest cells, at a cost of nlist m + nprobe (N / nlist) m, and misses a neighbour whose cell it does not probe. k-means here: k-means++ starts, Lloyd's iterations, a cell that empties keeps its centroid. -->

```{.python .input #retrieval-kmeans-ivf-def}
%%tab pytorch
#@save
def kmeans(X, n_clusters, n_iter=20, seed=0):
    """Lloyd's k-means of the rows of X from a seeded k-means++ start:
    (centroids (n_clusters, d), assign (N,)) as tensors on X's device,
    assign naming each row's nearest centroid. A cluster that loses all
    its rows keeps its centroid."""
    X = torch.as_tensor(X, dtype=torch.float32)
    x2, gen = (X ** 2).sum(1), torch.Generator().manual_seed(seed)

    def nearest(C):  # ||x||^2 is the same for every centroid: left out
        c2 = (C ** 2).sum(1)
        return torch.cat([(c2 - 2 * Xb @ C.T).argmin(1)
                          for Xb in X.split(8192)])

    def pick(d2):  # a row drawn with probability proportional to d2
        w = d2.double().cpu().clamp(min=0)
        if w.sum() == 0:
            return torch.randint(len(X), (1,), generator=gen).item()
        return torch.multinomial(w, 1, generator=gen).item()

    C = [X[pick(torch.ones(len(X)))]]
    d2 = x2 - 2 * X @ C[0] + C[0] @ C[0]
    for _ in range(1, n_clusters):  # k-means++: far rows are likelier
        C.append(X[pick(d2)])
        d2 = torch.minimum(d2, x2 - 2 * X @ C[-1] + C[-1] @ C[-1])
    C = torch.stack(C)
    for _ in range(n_iter):
        assign = nearest(C)
        counts = torch.bincount(assign, minlength=n_clusters)
        sums = torch.zeros_like(C).index_add_(0, assign, X)
        keep = counts > 0
        C[keep] = sums[keep] / counts[keep, None]
    return C, nearest(C)

#@save
class IVFIndex:
    """Inverted file index: the k-means cells of the training vectors; a
    query scans the vectors of its nprobe nearest cells exactly, by
    squared Euclidean distance (for unit vectors, the cosine order)."""
    def __init__(self, nlist, nprobe=1):
        self.nlist, self.nprobe = nlist, nprobe

    def train(self, X, n_iter=20, seed=0):
        self.centroids = d2l.kmeans(X, self.nlist, n_iter, seed)[0]

    def _cells(self, X, n):
        """The n nearest cells of each row of X."""
        d = (self.centroids ** 2).sum(1) - 2 * X @ self.centroids.T
        return d.topk(n, dim=1, largest=False).indices

    def add(self, X):
        """Store the rows of X (ids 0, 1, ...) grouped by cell."""
        X = torch.as_tensor(X, dtype=torch.float32)
        cell = self._cells(X, 1)[:, 0]
        self.ids = torch.argsort(cell, stable=True)
        self.X = X[self.ids]
        sizes = torch.bincount(cell, minlength=self.nlist)
        self.start = torch.cat([torch.zeros(1, dtype=torch.long),
                                sizes.cumsum(0)])  # cell c: start[c:c + 2]

    def search(self, Q, k):
        """(distances, ids) of the k nearest stored rows per query, nearest
        first, among the probed cells; id -1 where they hold fewer."""
        Q = torch.as_tensor(Q, dtype=torch.float32)
        D = torch.full((len(Q), k), torch.inf)
        I = torch.full((len(Q), k), -1, dtype=torch.long)
        for j, cells in enumerate(self._cells(Q, self.nprobe).tolist()):
            rows = torch.cat([torch.arange(self.start[c], self.start[c + 1])
                              for c in cells])
            top = ((self.X[rows] - Q[j]) ** 2).sum(1).topk(
                min(k, len(rows)), largest=False)
            n = len(top.values)
            D[j, :n], I[j, :n] = top.values, self.ids[rows[top.indices]]
        return D, I
```

<!-- Placeholder (prose, ## Approximate Nearest Neighbours): product quantization (Jégou, Douze and Schmid): split a vector of d coordinates into M sub-vectors and replace each by the index of its nearest centroid in a codebook of 2^b; a code holds M b bits. Derivation: asymmetric distance ‖y − x̂‖² = Σ_j ‖y^(j) − c_{j,k_j}‖² (eq_lm_adc), M look-ups in a table of M 2^b entries built once per query with 2^b d multiply-adds; memory M b / 8 bytes per vector plus 2^b d floats of codebooks; the error |d(y, x) − d(y, x̂)| ≤ ‖x − x̂‖ by the triangle inequality, and the expected squared error is the sum of the sub-quantizers' distortions. At d = 1,024, M = 64, b = 8: 64 bytes per vector against 4,096 in float32. Figure fig_lm_ivf_pq. A product quantizer is a block code; the analogy with low rank (:numref:`sec_mdl-svd-low-rank`) ends here: it approximates each vector by a sum of block centroids, not by a projection. -->

```{.python .input #retrieval-pq-def}
%%tab pytorch
#@save
class ProductQuantizer:
    """Product quantization: a vector of d coordinates is cut into m
    sub-vectors, each replaced by the index of the nearest of 2^nbits
    centroids of its own k-means codebook (one byte per sub-vector here).
    A query's squared distances to all codes are sums of m entries of a
    table computed once per query: the asymmetric distance."""
    def __init__(self, m, nbits=8):
        if not 1 <= nbits <= 8:
            raise ValueError('codes are stored as one byte per sub-vector')
        self.m, self.nbits = m, nbits

    def _split(self, X):  # (..., d) -> (..., m, d / m)
        X = torch.as_tensor(X, dtype=torch.float32)
        if X.shape[-1] % self.m:
            raise ValueError(f'{X.shape[-1]} coordinates in {self.m} parts')
        return X.reshape(*X.shape[:-1], self.m, X.shape[-1] // self.m)

    def fit(self, X, n_iter=20, seed=0):
        S = self._split(X)
        self.codebooks = torch.stack([d2l.kmeans(  # (m, 2^nbits, d / m)
            S[:, j], 2 ** self.nbits, n_iter, seed)[0] for j in range(self.m)])
        return self

    def encode(self, X):
        """Codes (N, m), uint8: the nearest centroid of each sub-vector."""
        S, c2 = self._split(X), (self.codebooks ** 2).sum(-1)
        return torch.cat([(c2 - 2 * torch.einsum(
            'nmd,mkd->nmk', block, self.codebooks)).argmin(-1)
                          for block in S.split(1024)]).to(torch.uint8)

    def decode(self, codes):
        """The reconstructions x-hat (N, d) of codes."""
        codes = torch.as_tensor(codes).long()
        return self.codebooks[torch.arange(self.m), codes].flatten(1)

    def adc_table(self, q):
        """(m, 2^nbits) squared distances of q's sub-vectors to the
        centroids."""
        return ((self._split(q)[:, None, :] - self.codebooks) ** 2).sum(-1)

    def adc_search(self, q, codes, k):
        """(distances, ids) of the k codes nearest to the query q, or to
        each row of q, by m table look-ups per code."""
        q = torch.as_tensor(q, dtype=torch.float32)
        codes = torch.as_tensor(codes).long()  # a no-op if already int64
        if q.ndim == 2:
            out = [self.adc_search(row, codes, k) for row in q]
            return tuple(torch.stack(t) for t in zip(*out))
        d = self.adc_table(q)[torch.arange(self.m), codes].sum(1)
        top = d.topk(min(k, len(d)), largest=False)
        return top.values, top.indices
```

<!-- Placeholder (prose, ## Approximate Nearest Neighbours): the smallest case by hand, d = 4, M = 2, b = 1 (two codebooks of two centroids each): the code of x, its reconstruction, the query's table and the distance as two look-ups, against the exact distance and the error bound. Then the identities on 20,000 points of a mixture of 16 Gaussians in 64 dimensions (M = 8, b = 8): table distances equal the distances to the reconstructions, the bound holds on 10^5 pairs, the cross term y^T e averages to zero (the pairs are drawn with replacement, so its standard error is slightly optimistic), Faiss, given our codebooks, assigns our codes, and the mean squared error is the sum of the eight blocks' distortions. -->

```{.python .input #retrieval-pq-check}
%%tab pytorch
pq = d2l.ProductQuantizer(2, nbits=1)  # d = 4, M = 2, b = 1, set by hand
pq.codebooks = torch.tensor([[[1., 0.], [0., 1.]], [[1., 1.], [-1., 0.]]])
x, y = torch.tensor([[0.9, 0.2, -0.8, 0.1]]), torch.tensor([1., 1., 0., 0.])
code = pq.encode(x)
x_hat, table = pq.decode(code), pq.adc_table(y)
adc_hand = table[torch.arange(2), code[0].long()].sum()  # two look-ups
print(f'code {code[0].tolist()}, x_hat {x_hat[0].tolist()}')
print(pd.DataFrame(table.numpy(), index=['j = 1', 'j = 2'],
                   columns=['k = 0', 'k = 1']))
print(f'ADC {adc_hand:.2f}; |y - x_hat|^2 {((y - x_hat) ** 2).sum():.2f}; '
      f'|y - x|^2 {((y - x) ** 2).sum():.2f}; |d - d_A| '
      f'{abs((y - x).norm() - (y - x_hat).norm()):.3f} <= |x - x_hat| '
      f'{(x - x_hat).norm():.3f}')
rng = np.random.default_rng(0)  # 16 Gaussians in R^64
mu = 2 * rng.standard_normal((16, 64))
draw = lambda n: torch.tensor(mu[rng.integers(16, size=n)]
                              + rng.standard_normal((n, 64)))
Xg, Yg = draw(20_000).float(), draw(1_000).float()
pq = d2l.ProductQuantizer(8, nbits=8).fit(Xg)
codes = pq.encode(Xg)
Xh = pq.decode(codes)
adc = torch.stack([pq.adc_table(y)[torch.arange(8), codes.long()].sum(1)
                   for y in Yg[:20]]).double()  # (20 queries, all codes)
exact = ((Yg[:20, None].double() - Xh.double()) ** 2).sum(-1)
print(f'(1) table against reconstruction distances: largest relative '
      f'difference {((adc - exact).abs() / exact).max():.1e}')
qi, xi = rng.integers(1_000, size=10**5), rng.integers(20_000, size=10**5)
Y, X0, X1 = Yg.double()[qi], Xg.double()[xi], Xh.double()[xi]
gap = ((Y - X0).norm(dim=1) - (Y - X1).norm(dim=1)).abs()
print(f'(2) |d - d_A| <= |x - x_hat| on {len(qi):,} pairs: '
      f'{(gap <= (X0 - X1).norm(dim=1) + 1e-9).float().mean():.4f}')
cross = (Y * (X0 - X1)).sum(1)
print(f'(3) mean of y^T e {cross.mean():.4f}, standard error '
      f'{cross.std() / len(cross) ** 0.5:.4f}')
fpq = faiss.IndexPQ(64, 8, 8)
faiss.copy_array_to_vector(pq.codebooks.numpy().ravel(), fpq.pq.centroids)
fpq.is_trained = True
same = (fpq.sa_encode(Xg.numpy()) == codes.numpy()).all(1).mean()
print(f'(4) Faiss codes with our codebooks equal ours for {same:.4f} of '
      f'the vectors')
err2 = ((Xg - Xh) ** 2).double()  # (N, 64): squared error per coordinate
blocks = err2.reshape(-1, 8, 8).sum(-1).mean(0)  # each block's distortion
print(f'(5) mean |e|^2 {err2.sum(1).mean():.3f} = sum of the 8 blocks\' '
      f'distortions {blocks.sum():.3f} ('
      + ', '.join(f'{b:.2f}' for b in blocks) + ')')
assert abs(adc_hand - 2.0) < 1e-6
assert ((adc - exact).abs() / exact).max() < 1e-5
assert bool((gap <= (X0 - X1).norm(dim=1) + 1e-9).all())
assert abs(cross.mean()) < 3 * cross.std() / len(cross) ** 0.5
assert same >= 0.999
assert abs(err2.sum(1).mean() - blocks.sum()) < 1e-9 * blocks.sum()
```

<!-- Placeholder (prose, ## Approximate Nearest Neighbours): the stress corpus: the book's chunk vectors plus about 60,000 off-topic passages from Natural Questions (rows not used for training in :numref:`sec_lm_text_embeddings`) and 1,000 of their questions; it tests whether an index finds the exact neighbours, not how hard the task is. D2L_RERUN=ret-stress-emb encodes the passages again (about 5 minutes on a GPU). -->

```{.python .input #retrieval-stress}
%%tab pytorch
if 'ret-stress-emb' in RERUN:
    nq = pd.read_parquet(d2l.download_keyed('nq-pairs'))
    rows = np.random.default_rng(0).permutation(len(nq))[20_000:]
    passages = list(dict.fromkeys(nq.answer.iloc[rows]))  # distinct, in order
    at = {p: j for j, p in enumerate(passages)}
    pick = rows[np.random.default_rng(0).choice(len(rows), N_NQ_QUERIES,
                                                replace=False)]
    encoder = d2l.load_encoder(EMB, padding_side=QUERY['padding_side'],
                               dtype=DTYPE)
    P = d2l.embed_texts(*encoder, passages, pooling=QUERY['pooling'],
                        max_len=512)
    Q = embed_queries(encoder, nq['query'].iloc[pick].tolist(), NQ_TASK)
    del encoder
    d2l.lm_free()
    d2l.lm_save_rerun(ARTIFACTS['ret-stress-emb'][1], lambda path: np.savez(
        path, X=np.concatenate([X.cpu().numpy(), P.half().numpy()]),
        ids=np.array(FULL.ids + [f'nq:{j}' for j in range(len(passages))]),
        Q=Q.half().numpy(), q_gold=np.array([len(FULL) + at[p] for p in
                                             nq.answer.iloc[pick]]),
        config=json.dumps(CONFIG['ret-stress-emb'])))
with np.load(result_path('ret-stress-emb')) as Z:
    assert json.loads(str(Z['config'])) == json.loads(json.dumps(
        CONFIG['ret-stress-emb'])), 'D2L_RERUN=ret-stress-emb'
    XS, STRESS_IDS = Z['X'][:STRESS_N].astype(np.float32), Z['ids'][:STRESS_N]
    QS = Z['Q'].astype(np.float32)
    book_diff = np.abs(Z['X'][:len(FULL)].astype(np.float32)
                       - X.float().cpu().numpy()).max()
assert book_diff == 0, 'the stress file holds other book chunk vectors'
QD = Q_DEV.numpy()
print(f'{len(XS):,} vectors of {XS.shape[1]} coordinates '
      f'({XS.nbytes / 2**20:,.0f} MiB in float32), '
      f'{np.mean(np.char.startswith(STRESS_IDS, "sec512:")):.3f} of them '
      f'book chunks; {len(QS):,} timed questions; book vectors differ from '
      f'the chunk vectors by at most {book_diff:.1e}')
```

<!-- Placeholder (prose, ## Approximate Nearest Neighbours): HNSW (Malkov and Yashunin) through Faiss: a hierarchy of proximity graphs, each vector linked to about M neighbours per layer (efConstruction candidates kept while building); a search descends the layers and keeps efSearch candidates at the bottom. Memory is the vectors plus the links, and the vectors dominate; deletes are expensive. The measurement: for every index family along its knob, Recall@10 against the exact top 10 (which is not task recall: the second column is the dev questions' Recall@10), the 50th and 95th percentile latency of 1,000 one-question searches on one thread after 100 warm-up searches, bytes per vector of the codes and of the whole index, and build seconds (8 threads); IVF-PQ with M = 64 is rebuilt with two more k-means seeds. The rows marked "ours" (IVFIndex and ProductQuantizer) show recall, not speed: their latency is Python's loop and indexing overhead. Always measured live; D2L_RERUN=ret-ann stores the table, and a normal build prints the live table against the registered one (seeded builds give the same recall; HNSW's threaded build differs a little; latency depends on the machine and its load). The reduced path (STRESS_N = 20,000) uses 128 cells, builds no HNSW graph and makes no comparison with the registered table. -->

```{.python .input #retrieval-ann}
%%tab pytorch
d = XS.shape[1]
exact = faiss.IndexFlatL2(d)
exact.add(XS)
TRUE = exact.search(QS, 10)[1]  # the exact ten nearest of each question
ROWS = []

def threads(n):
    faiss.omp_set_num_threads(n)
    torch.set_num_threads(n)

def build(key, data, seed=0):
    """A Faiss index from its factory string, trained and filled with data
    (k-means seeded), and its build seconds."""
    threads(BUILD_THREADS)
    t0 = time.perf_counter()
    index = faiss.index_factory(data.shape[1], key)
    if key.startswith('IVF'):
        faiss.extract_index_ivf(index).cp.seed = seed
        if 'PQ' in key:
            index.pq.cp.seed = seed
    if key.startswith('HNSW'):
        index.hnsw.efConstruction = EF_CONSTRUCTION
    index.train(data)
    index.add(data)
    return index, time.perf_counter() - t0

def latency(search, Q):
    """p50 and p95 (ms) of one-question searches on one thread, after
    warm-up searches with the dev questions."""
    threads(1)
    for q in QD[:N_WARM]:
        search(q[None])
    t = []
    for q in Q:
        t0 = time.perf_counter()
        search(q[None])
        t.append(time.perf_counter() - t0)
    threads(BUILD_THREADS)
    return np.median(t) * 1e3, np.percentile(t, 95) * 1e3

def overlap(found, true):
    """Recall@10 against exact search: the share of the true ten found."""
    return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, true)])

def record(family, knob, search, code_bytes, index_bytes, build_s, seed=0):
    """Recall against exact search, dev Recall@10, and one-thread latency."""
    threads(BUILD_THREADS)
    task = d2l.retrieval_metrics([[STRESS_IDS[i] for i in r if i >= 0]
                                  for r in search(QD)[1]], GOLD, ks=(10,))
    p50, p95 = latency(search, QS)
    ROWS.append(dict(family=family, knob=knob, seed=seed,
                     recall=overlap(search(QS)[1], TRUE),
                     task_recall=task['recall@10'].mean(), p50=p50, p95=p95,
                     code_bytes=code_bytes, index_bytes=index_bytes,
                     build_s=build_s))

def size(index):
    return len(faiss.serialize_index(index)) / index.ntotal

index, t = build('Flat', XS)
record('flat', '', lambda Q: index.search(Q, 10), 4 * d, size(index), t)
for fam, key, code in [('IVF-Flat', f'IVF{NLIST},Flat', 4 * d)] + [
        (f'IVF-PQ{M_}', f'IVF{NLIST},PQ{M_}', M_) for M_ in PQ_MS]:
    for seed in (PQ_SEEDS if fam == f'IVF-PQ{PQ_MS[0]}' else PQ_SEEDS[:1]):
        index, t = build(key, XS, seed)
        for p in (NPROBES if seed == PQ_SEEDS[0] else (16,)):
            faiss.extract_index_ivf(index).nprobe = p
            record(fam, f'nprobe {p}', lambda Q: index.search(Q, 10), code,
                   size(index), t, seed)
if STRESS_N is None:  # the reduced path builds no graph
    index, t = build(f'HNSW{HNSW_M}', XS)
    for ef in EF_SEARCHES:
        index.hnsw.efSearch = ef
        record(f'HNSW{HNSW_M}', f'efSearch {ef}',
               lambda Q: index.search(Q, 10), 4 * d, size(index), t)
threads(BUILD_THREADS)
t0 = time.perf_counter()
ivf = d2l.IVFIndex(NLIST, nprobe=16)
ivf.train(XS)
ivf.add(XS)
record('IVF-Flat (ours)', 'nprobe 16', lambda Q: [
    a.numpy() for a in ivf.search(Q, 10)], 4 * d, (ivf.X.numel() * 4 + len(
        ivf.ids) * 8 + ivf.centroids.numel() * 4) / len(XS),
       time.perf_counter() - t0)
t0 = time.perf_counter()
pq = d2l.ProductQuantizer(PQ_MS[0]).fit(XS)
codes = pq.encode(XS)  # one byte per sub-vector
idx = codes.long()  # the table look-ups take int64: converted once
record(f'PQ{PQ_MS[0]} (ours)', 'all codes', lambda Q: [
    a.numpy() for a in pq.adc_search(Q, idx, 10)], PQ_MS[0], (
        codes.numel() + pq.codebooks.numel() * 4) / len(XS),
       time.perf_counter() - t0)
if GPU:  # every question at once: a batched point, not a batch-1 latency
    XG = torch.from_numpy(XS).cuda()
    QG = torch.from_numpy(QS).cuda()
    secs = scan_seconds(QG, XG, reps=20)
    found = (QG @ XG.T).topk(10, dim=1).indices.cpu().numpy()
    ROWS.append(dict(family='flat, GPU, batched', knob=f'{len(QS)} at once',
                     seed=0, recall=overlap(found, TRUE),
                     p50=secs / len(QS) * 1e3, code_bytes=4 * d))
    del XG, QG
TABLE = pd.DataFrame(ROWS)
print(machine())
print(TABLE.round(4).to_string(index=False))
seeds = TABLE[(TABLE.family == f'IVF-PQ{PQ_MS[0]}') & (TABLE.knob ==
                                                        'nprobe 16')]
print(f'IVF-PQ{PQ_MS[0]}, nprobe 16, k-means seeds {list(seeds.seed)}: '
      f'recall {seeds.recall.min():.3f} to {seeds.recall.max():.3f}')
if 'ret-ann' in RERUN:
    write_result('ret-ann', rows=json.loads(TABLE.to_json(orient='records')))
reg = read_result('ret-ann', strict=False)
if reg is None:  # e.g. the reduced path, STRESS_N set
    print(f'the registered table was made with other settings than these '
          f'(N = {len(XS):,}, {NLIST} cells): no comparison')
else:
    reg = pd.DataFrame(reg['rows'])
    both = TABLE.merge(reg, on=['family', 'knob', 'seed'],
                       suffixes=('', '_reg'))
    both['recall_diff'] = (both.recall - both.recall_reg).abs()
    both['p50_ratio'] = both.p50 / both.p50_reg
    print(both.groupby('family', sort=False).agg(
        largest_recall_difference=('recall_diff', 'max'),
        median_p50_live_over_registered=('p50_ratio', 'median')).round(3))
```

<!-- Placeholder (prose, ## Approximate Nearest Neighbours): the trade-off: recall against the exact top 10 and latency (log scale) for each family along its knob, marker area proportional to bytes per vector, the flat scan as one point; the right panel the dev questions' Recall@10 against latency, where the approximate indexes lose less than the left panel suggests, because most of their misses are off-topic neighbours. -->

```{.python .input #retrieval-tradeoff}
%%tab pytorch
PLOT = TABLE[(TABLE.seed == 0) & ~TABLE.family.str.contains('batched')]
families = list(PLOT.family.unique())
_, axes = d2l.plt.subplots(1, 2, figsize=(8.5, 3.6), layout='constrained')
for ax, col, label in zip(axes, ('recall', 'task_recall'), (
        'Recall@10 against exact search', 'dev Recall@10 (task)')):
    lines = [PLOT[PLOT.family == f] for f in families]
    d2l.plot([t.p50.tolist() for t in lines], [t[col].tolist() for t in lines],
             'p50 latency (ms), one question', label, xscale='log',
             legend=families if col == 'task_recall' else None, axes=ax,
             fmts=[f'C{i}-' for i in range(len(lines))],
             legend_loc='lower right')
    for i, t in enumerate(lines):
        ax.scatter(t.p50, t[col], s=10 + 150 * t.code_bytes / (4 * d),
                   color=f'C{i}', alpha=0.6)
```

<!-- Placeholder (prose, ## Approximate Nearest Neighbours): which index wins at this size: if HNSW reaches recall 0.95 at a lower latency than the flat scan, the ratio (one significant figure, C6); if the flat scan were already the fastest at about 65,000 vectors, a synthetic extension to a million vectors (copies of the stress vectors, each moved by noise as large as the median nearest-neighbour distance, so the copies are not near-duplicates; clearly labelled synthetic; D2L_RERUN=ret-crossover, about 20 minutes and 9 GB of memory) would show where the graph takes over. -->

```{.python .input #retrieval-crossover}
%%tab pytorch
cpu = TABLE[~TABLE.family.str.contains('batched') & (TABLE.seed == 0)]
good = cpu[cpu.recall >= 0.95].sort_values('p50')
flat = cpu[cpu.family == 'flat'].iloc[0]
hnsw = good[good.family == f'HNSW{HNSW_M}'].head(1)
for _, r in hnsw.iterrows():  # the fastest HNSW search with recall >= 0.95
    print(f'{r.family}, {r.knob}: recall {r.recall:.3f} at p50 {r.p50:.3f} ms;'
          f' flat scan {flat.p50:.2f} ms, {flat.p50 / r.p50:.2g} times longer')
if hnsw.empty:
    print(f'no HNSW{HNSW_M} search reaches recall 0.95' if STRESS_N is None
          else 'no HNSW graph on the reduced path')
if good.iloc[0].family != 'flat':
    print(f'fastest configuration with recall >= 0.95 at N = {len(XS):,}: '
          f'{good.iloc[0].family}, {good.iloc[0].knob}')
elif 'ret-crossover' in RERUN:  # synthetic: a million noisy copies
    rng, n = np.random.default_rng(0), CONFIG['ret-crossover']['n']
    nn = exact.search(XS[:2_000], 2)[0][:, 1]  # squared, nearest other row
    noise = float(np.sqrt(np.median(nn)))  # copies as far apart as neighbours
    big = XS[np.arange(n) % len(XS)] + noise / d ** 0.5 * rng.standard_normal(
        (n, d), dtype=np.float32)
    big /= np.linalg.norm(big, axis=1, keepdims=True)
    flat_big, t_flat = build('Flat', big)
    true_big, rows = flat_big.search(QS, 10)[1], []
    hnsw_big, t_hnsw = build(f'HNSW{HNSW_M}', big)
    for name, index, knob, t in [('flat', flat_big, '', t_flat)] + [
            (f'HNSW{HNSW_M}', hnsw_big, ef, t_hnsw) for ef in EF_SEARCHES]:
        if knob:
            index.hnsw.efSearch = knob
        search = lambda Q: index.search(Q, 10)
        p50, p95 = latency(search, QS)
        rows.append(dict(family=name, knob=knob and f'efSearch {knob}',
                         recall=overlap(search(QS)[1], true_big), p50=p50,
                         p95=p95, build_s=t))
    write_result('ret-crossover', n=n, noise=noise, rows=rows)
    print(f'synthetic: {n:,} copies of the stress vectors, noise of norm '
          f'{noise:.3f} (the median nearest-neighbour distance)')
    print(pd.DataFrame(rows).round(4).to_string(index=False))
elif ARTIFACTS['ret-crossover'][0] in d2l.DATA_HUB:
    print('synthetic, 10^6 noisy copies of the stress vectors:')
    print(pd.DataFrame(read_result('ret-crossover')['rows']).round(4))
else:
    print(f'the flat scan is the fastest at N = {len(XS):,}; the synthetic '
          f'crossover is not registered')
```

<!-- Placeholder (prose, ## The Book Retriever): the retriever that the next section and later chapters call. Its defaults: BM25 over the sparse view with the parameters chosen above, the chapter's embedder over the registered chunk vectors (one flat scan), reciprocal-rank fusion with k = 60, no reranker unless one is given and rerank_k is set; index='hnsw' or 'ivfpq' builds a Faiss index on first use. The embedder that it and the later chapters use is a constant. -->

```{.python .input #retrieval-default-embedder}
%%tab pytorch
#@save
# The embedder that the retriever, this chapter and later chapters use
DEFAULT_EMBEDDER = 'Qwen/Qwen3-Embedding-0.6B'
```

<!-- Placeholder (prose, ## The Book Retriever): the Hit record (chunk id, score, rank, text, heading path, and each stage's score) and the class. -->
```{.python .input #retrieval-retriever-def}
%%tab pytorch
#@save
Hit = collections.namedtuple('Hit',
                             'chunk_id score rank text heading_path scores')

#@save
class BookRetriever:
    """Hybrid retrieval over the book corpus: BM25 over the 'sparse' view
    and a dense embedder over the 'dense' view, fused by reciprocal rank,
    optionally reranked by a yes/no reranker (which reads the 'dense' view
    with the embedder's instruction, BOOK_TASK). dense=None gives a
    BM25-only retriever; index 'flat' scans all chunk vectors, 'hnsw' and
    'ivfpq' build Faiss indexes on first use."""
    # BM25 parameters chosen on the book's training questions
    BM25_PARAMS = dict(k1=1.2, b=0.9, split_identifiers=True)

    def __init__(self, corpus=None, dense=DEFAULT_EMBEDDER, index='flat',
                 reranker=None, fusion='rrf', rrf_k=60, device=None,
                 cache=True, bm25_params=None):
        if fusion != 'rrf' or index not in ('flat', 'hnsw', 'ivfpq'):
            raise ValueError(f'fusion {fusion!r} or index {index!r}')
        self.corpus = d2l.book_corpus() if corpus is None else corpus
        self.bm25_params = dict(self.BM25_PARAMS if bm25_params is None
                                else bm25_params)
        p = self.bm25_params
        self.bm25 = d2l.BM25Index(
            [d2l.bm25_tokenize(t, p['split_identifiers'])
             for t in self.corpus.texts('sparse')], k1=p['k1'], b=p['b'])
        self.dense, self.index, self.reranker = dense, index, reranker
        self.rrf_k, self.cache = rrf_k, cache
        self.device = device or ('cuda' if torch.cuda.is_available()
                                 else 'cpu')
        self._X = self._ann = self._enc = self._rr = self._texts = None

    def _encoder(self):
        if self._enc is None:
            dtype = torch.bfloat16 if self.device != 'cpu' else torch.float32
            self._enc = d2l.load_encoder(
                self.dense, padding_side=d2l._EMBEDDERS[self.dense][
                    'padding_side'], dtype=dtype, device=self.device)
        return self._enc

    def chunk_vectors(self):
        """Unit vectors of the chunks (N, m), float16: the registered set
        for this embedder if there is one, else the local cache, else
        encoded now (and cached)."""
        if self._X is None:
            spec, c = d2l._EMBEDDERS[self.dense], self.corpus
            rev = d2l.HF_ENCODER_REVISIONS[self.dense]
            key = d2l._emb_cache_key(c.version, c.tag, 'dense', self.dense,
                                     rev, 1024, spec['pooling'], '')
            reg = d2l._R3_EMB_KEYS.get(self.dense)
            X = (np.load(d2l.download_keyed(reg)) if c.tag == 'sec512'
                 and reg in d2l.DATA_HUB else
                 d2l._emb_cache_load(key) if self.cache else None)
            if X is None:
                model, tok = self._encoder()
                X = d2l.embed_texts(model, tok, c.texts('dense'),
                                    pooling=spec['pooling'], max_len=1024,
                                    batch_size=16).half().numpy()
                if self.cache:
                    d2l._emb_cache_save(key, X)
            assert X.shape[0] == len(c), 'chunk vectors do not match corpus'
            self._X = torch.tensor(X, device=self.device)
        return self._X

    def _dense_search(self, queries, depth):
        spec = d2l._EMBEDDERS[self.dense]
        model, tok = self._encoder()
        Q = d2l.embed_texts(model, tok, [spec['query_format'].format(
            task=d2l.BOOK_TASK, query=q) for q in queries],
            pooling=spec['pooling'], max_len=512)
        X = self.chunk_vectors()
        if self.index == 'flat':
            S = (Q.to(X.device, X.dtype) @ X.T).float().cpu()
            s, i = S.topk(min(depth, len(X)), dim=1)
            return i.numpy(), s.numpy(), S.numpy()
        if self._ann is None:
            import faiss
            Xf = np.ascontiguousarray(X.float().cpu().numpy())
            m = Xf.shape[1]
            if self.index == 'hnsw':
                self._ann = faiss.IndexHNSWFlat(m, 32,
                                                faiss.METRIC_INNER_PRODUCT)
                self._ann.hnsw.efConstruction, self._ann.hnsw.efSearch = (
                    200, 128)
            else:
                nlist = max(1, len(Xf) // 64)
                self._ann = faiss.index_factory(
                    m, f'IVF{nlist},PQ64', faiss.METRIC_INNER_PRODUCT)
                if len(Xf) < 256 * 39:  # below what Faiss asks per codebook
                    print(f'IVF-PQ: codebooks of 256 centroids trained on '
                          f'{len(Xf):,} vectors')
                    self._ann.pq.cp.min_points_per_centroid = 1
                self._ann.train(Xf)
                self._ann.nprobe = 16
            self._ann.add(Xf)
        s, i = self._ann.search(np.ascontiguousarray(Q.numpy()), depth)
        keep = i >= 0  # Faiss pads with id -1 when it finds fewer
        return [r[k] for r, k in zip(i, keep)], [
            r[k] for r, k in zip(s, keep)], None

    def search_batch(self, queries, k=10, mode='hybrid', rerank_k=None):
        """search for each of several queries."""
        if mode not in ('bm25', 'dense', 'hybrid'):
            raise ValueError(f'mode {mode!r}')
        if self.dense is None:
            if mode == 'dense':
                raise ValueError('a BM25-only retriever has no dense mode')
            mode = 'bm25'
        if rerank_k and not self.reranker:
            raise ValueError('rerank_k needs a retriever with a reranker')
        depth = max(100, k, rerank_k or 0)
        p, ids = self.bm25_params, self.corpus.ids
        bm25 = [self.bm25.scores(d2l.bm25_tokenize(q, p['split_identifiers']))
                for q in queries]
        if mode != 'bm25':
            di, ds, dense_all = self._dense_search(queries, depth)
        out = []
        for j, q in enumerate(queries):
            stages = {'bm25': bm25[j]}
            bm_rank = np.argsort(-bm25[j], kind='stable')[:depth]
            if mode == 'bm25':
                ranked = [(int(i), float(bm25[j][i])) for i in bm_rank]
            else:
                dense_s = (dense_all[j] if dense_all is not None else
                           dict(zip(di[j].tolist(), ds[j].tolist())))
                stages['dense'] = dense_s
                if mode == 'dense':
                    ranked = list(zip(di[j].tolist(), ds[j].tolist()))
                else:
                    ranked = d2l.rrf_fuse([bm_rank.tolist(), di[j].tolist()],
                                          k=self.rrf_k)[:depth]
                    stages['rrf'] = dict(ranked)
            if rerank_k:
                if self._texts is None:
                    self._texts = self.corpus.texts('dense')
                top = [i for i, _ in ranked[:rerank_k]]
                pr = d2l.rerank_yes_no(*self._reranker(), q,
                                       [self._texts[i] for i in top],
                                       instruction=d2l.BOOK_TASK)
                stages['rerank'] = dict(zip(top, pr.tolist()))
                order = np.argsort(-pr, kind='stable')
                ranked = [(top[o], float(pr[o])) for o in order]
            hits = []
            for r, (i, s) in enumerate(ranked[:k], start=1):
                c = self.corpus.chunks[i]
                hits.append(d2l.Hit(c['id'], s, r, c['text'],
                                    c['heading_path'], {
                    name: float(v[i]) if not isinstance(v, dict)
                    else v.get(i) for name, v in stages.items()}))
            out.append(hits)
        return out

    def search(self, query, k=10, mode='hybrid', rerank_k=None):
        """The k best chunks for query as a list of Hit, best first; with
        rerank_k, the reranker reorders the first rerank_k of the fused
        ranking."""
        return self.search_batch([query], k, mode, rerank_k)[0]

    def _reranker(self):
        if self._rr is None:
            self._rr = d2l.load_hf(
                self.reranker,
                revision=d2l.HF_ENCODER_REVISIONS[self.reranker],
                device_map={'': self.device})
        return self._rr

    def release(self):
        """Free the GPU models (query encoder, reranker); a later search
        loads them again."""
        self._enc = self._rr = None
        d2l.lm_free()
```

<!-- Placeholder (prose, ## The Book Retriever): the retriever with its defaults, for the question of the next section's example: the five hits, their headings and each stage's score. The next section reads these passages to answer; :numref:`chap_lm_agents` and later chapters call the same retriever. -->

```{.python .input #retrieval-retriever}
%%tab pytorch
retriever = BookRetriever()
for h in retriever.search('What does the key-value cache store per token?',
                          k=5):
    print(f'{h.rank}. {h.chunk_id}\n   {" > ".join(h.heading_path)}\n   '
          + ', '.join(f'{k} {v:.3f}' for k, v in h.scores.items()
                      if v is not None))
retriever.release()
```

<!-- Placeholder (prose): the pitfalls (recall against exact search is not task recall; raw scores fused without normalization; reranking the top 10 and calling it a Recall@100 gain; HNSW memory is the vectors; Recall@k compared across chunk sizes, which :numref:`sec_lm_rag` replaces with coverage at a token budget; tuning on dev); learned sparse retrieval and late interaction in one sentence each (:numref:`sec_lm_encoders_fr`); which passages the model should read once they are found, hand-off to :numref:`sec_lm_rag`; Summary; Exercises. -->
