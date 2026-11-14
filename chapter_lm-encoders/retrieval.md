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

<!-- Placeholder (stage 2, infra step): the cells below are the section's
shared library, written before the section; the prose around them comes with
the section. Cell ids follow the section specification. -->

```{.python .input #retrieval-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

<!-- Placeholder (stage 2): the book corpus, a snapshot of the book's sources at a named commit (built by `tools/build_r3.py`). -->

```{.python .input #retrieval-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-book-r3-v1'] = (  # book commit 098bf463, corpus r3.1
    d2l.LM_DATA_URL + 'lm-book-r3-v1.zip',
    '0b7f11e044a4838139ac02dffb6e2d7eb32b48b8')
```

<!-- Placeholder (stage 2): the corpus loader; `BookCorpus` itself, the cleaning and the chunkers are library code explained in a sentence each. -->

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

<!-- Placeholder (stage 2): words and numbers as index terms; identifiers split at underscores, dots and camel case, the whole identifier kept as well. -->

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

<!-- Placeholder (stage 2): BM25 as a sparse term-document matrix of precomputed weights; a query is one sparse product. -->

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

<!-- Placeholder (stage 2): reciprocal-rank fusion. -->

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

<!-- Placeholder (stage 2): the reranker's yes/no protocol from its model card, scored with the label scoring of the prompting section. -->

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

<!-- Placeholder (stage 2): the default embedder, chosen by the rule fixed before the comparison (Recall@10 on train questions). -->

```{.python .input #retrieval-default-embedder}
%%tab pytorch
#@save
# Qwen3-Embedding-0.6B until the selection on train questions: the 2026
# candidate (harrier-oss-v1-0.6b) replaces it only if its Recall@10 is
# higher with a document-clustered paired interval that excludes zero
DEFAULT_EMBEDDER = 'Qwen/Qwen3-Embedding-0.6B'
```

<!-- Placeholder (stage 2): the retriever that the next section and later chapters call. -->

```{.python .input #retrieval-retriever-def}
%%tab pytorch
#@save
Hit = collections.namedtuple('Hit',
                             'chunk_id score rank text heading_path scores')

#@save
class BookRetriever:
    """Hybrid retrieval over the book corpus: BM25 over the 'sparse' view
    and a dense embedder over the 'dense' view, fused by reciprocal rank,
    optionally reranked by a yes/no reranker. dense=None gives a BM25-only
    retriever; index 'flat' scans all chunk vectors, 'hnsw' and 'ivfpq'
    build Faiss indexes on first use."""
    # BM25 parameters tuned on train questions (19.4, cell bm25-tune)
    BM25_PARAMS = dict(k1=1.2, b=0.75, split_identifiers=True)

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
        self._X = self._ann = self._enc = self._rr = None

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
                self._ann.train(Xf)
                self._ann.nprobe = 16
            self._ann.add(Xf)
        s, i = self._ann.search(np.ascontiguousarray(Q.numpy()), depth)
        return i, s, None

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
                top = [i for i, _ in ranked[:rerank_k]]
                pr = d2l.rerank_yes_no(*self._reranker(), q,
                                       [self.corpus.chunks[i]['text']
                                        for i in top])
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
