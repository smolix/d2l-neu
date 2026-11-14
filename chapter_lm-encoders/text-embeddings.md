# Text Embeddings from Contrastive Training
:label:`sec_lm_text_embeddings`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "2.3 Text Embeddings from Contrastive Training · T1").
Blueprint: docs/llm/blueprints/ch02_encoders.md section 2.3.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Pooled encoder states retrieve poorly until contrastive training spreads them on the sphere.
**Opening:** The reader mean-pools the token vectors of 2.2's raw encoder for every chunk of this book and retrieves with cosine similarity. The hits are near-random, and the mean cosine between unrelated chunks is close to one. The section explains the geometry, repairs it with a contrastive objective the reader already knows, and closes on the question of scale: what the released embedders trained on far more pairs add.
**Teaches:**
- Anisotropy and pooling (CLS, mean, last token), normalization, and why frequency dominates the raw geometry (`d2l.pool_hidden`, `d2l.anisotropy`).
- In-batch negatives with temperature (`d2l.info_nce_loss`), the batch size as the number of negatives, mined hard negatives and the false-negative risk audited with the reranker; weak pairs then supervised pairs as the two-stage recipe.
- Decoder-derived embedders (Qwen3-Embedding with last-token pooling, left padding and a query instruction), asymmetric query and document prefixes; Matryoshka dimensions, int8 and binary quantization; MTEB and why a leaderboard average does not predict a domain.
**Derivation:** The large-batch limit of InfoNCE as an alignment term (positive pairs close) plus a uniformity term (the log of the mean Gaussian kernel over all pairs) on the unit sphere, which is why contrastive training removes anisotropy. Smallest numeric case: four points on a circle, where the uniformity term is minimized by the square and the alignment term by coincidence, and the temperature sets the trade. The log-B information ceiling is restated from `sec_mdl-infonce`, not re-proved.
**Notebook:**
1. Encode the book corpus (`d2l.book_corpus('sections')`) and the answerable `dev` questions with raw ModernBERT under CLS, mean and centered mean pooling; report Recall@10, nDCG@10 and anisotropy against the BM25 line from 2.4.
2. Train the encoder contrastively on 20,000 generic question-answer pairs (none from the book) with in-batch negatives at batch 64, then with one mined hard negative per query from ranks 11 to 50 of the model's own ranking; three seeds for the in-batch rung to calibrate training noise.
3. Audit the mined negatives with Qwen3-Reranker-0.6B and report the false-negative rate.
4. Encode with Qwen3-Embedding-0.6B with and without the query instruction; sweep Matryoshka dimensions (768 to 64) for a Matryoshka-trained checkpoint and for the plain one; binary quantization with and without float rescoring.
5. Claim that survives noise: contrastive training lifts Recall@10 by tens of points over raw pooling (paired bootstrap over queries excludes zero and the gain exceeds the seed range); the hard-negative gain may lie within seed noise, and the audit explains why; the released embedder exceeds the notebook's checkpoint, trained on far fewer pairs by design.
6. Reused pieces: `d2l.load_hf`, `d2l.paired_bootstrap`, `d2l.book_corpus`, `d2l.BookRetriever(mode='bm25')`, `d2l.book_qa('dev')`; new `d2l.embed_texts`, `d2l.train_embedder`, `d2l.retrieval_metrics`, `d2l.align_uniform`, `d2l.truncate_embeddings`, `d2l.binarize`, `d2l.hamming_scores`.
**Figures:** `fig_lm_anisotropy` (raw pooled vectors crowded in a cone against contrastively trained vectors spread on the sphere, two panels of a 2-D projection); `fig_lm_matryoshka` (plot: Recall@10 against dimension for the Matryoshka and plain checkpoints, with binary quantization as a second series).
**Restate / forward:** `sec_mdl-infonce` (InfoNCE with B negatives bounds mutual information by log B; the bound is the reason batch size matters). Forward: 2.4 (the index that serves these vectors). Re-check the Hub for a 2026 embedder before authoring.
**Exercises:** 5 to 7. Load-bearing: derive the uniformity term from the InfoNCE denominator as the batch grows (derive); measure Recall@10 against temperature 0.02 to 0.2 at fixed batch and explain the shape by the gradient weighting of hard negatives (measure); check that last-token pooling under right padding reads a pad token and fix it by indexing with the mask (check); open: adapt the embedder on `book_qa('train')` pairs and report transfer to `dev` with the held-out discipline of 2.5.
**Writer notes:**
- Exports (`01_interfaces.md`): `d2l.embed_texts(model, tokenizer, texts, pooling='mean', batch_size=64, max_len=512, normalize=True, prefix='', padding_side=None, device=None)` returning a float32 CPU tensor, batches sorted by length and the original order recovered; `d2l.info_nce_loss(q, d, tau=0.05, hard_neg=None, symmetric=False)`; `d2l.train_embedder(...)` with a no-duplicate-queries sampler; `d2l.retrieval_metrics(ranked_ids, gold_ids, ks=(1, 5, 10, 100))` returning per-query arrays so the bootstrap can resample queries (2.4 gives the metrics their full treatment).
- 2.3 depends on 2.2 only through the raw `answerdotai/ModernBERT-base` checkpoint (no fine-tuned 2.2 model), so the LLM course path that skips 2.1 and 2.2 has no gap; 2.3 opens with a half-page recap of what the raw encoder outputs. It uses `d2l.book_corpus` and `d2l.book_qa('dev')` as shipped artifacts and forward-references their construction with `numref`; if the 2.5 artifact is late, a 100-question hand-made stand-in from 10 sections is the fallback.
- Models and data: `Qwen/Qwen3-Embedding-0.6B` (last-token pooling, `padding_side='left'`, instruction `Instruct: {task}\nQuery:{query}` on the query side only, dimensions 32 to 1,024); `Qwen/Qwen3-Reranker-0.6B` through its yes/no logit protocol; training pairs `sentence-transformers/natural-questions` (confirm CC BY-SA 3.0 at authoring; never redistributed), fallback `sentence-transformers/all-nli`. No training pair comes from the book, so the ladder measures transfer; domain adaptation is the open exercise.
- Memory: 150M parameters in fp32 with Adam are 2.4 GB; batch 128 with gradient checkpointing stays under 12 GiB; if the batch-size sweep's largest batch does not fit a 15 GiB slot, use a cached-gradient loss (GradCache) for that point and state it is equivalent, or sweep up to 128.
- Pitfalls to state: pooling and prefix mismatch with a checkpoint's training convention; dot product on unnormalized vectors ranks by norm; duplicate queries in a batch are false negatives; hard negatives from the top ranks are often relevant; truncating a model not trained for Matryoshka loses much more; Hamming-only search discards magnitude; long chunks truncate silently at `max_len` (log the count); a released embedder may have seen this book online, so its numbers on the book corpus are optimistic by an unknown amount.
- Do not re-teach: the InfoNCE bound and its proof (`sec_mdl-infonce`), cross-entropy, the encoder (2.2).
- Committed artifacts: `d2l-modernbert-embed` and `d2l-modernbert-embed-mrl` (bf16, about 0.3 GB each), the sweep results JSON; `LOAD_CKPT=True` skips training.
- Check the Hub at authoring for a 2026 successor to Qwen3-Embedding and Qwen3-Reranker; if one exists in budget with a permissive licence, swap it in and keep the Qwen3 pair as the second row.
**Sources:**
- Papers: [SimCSE](https://arxiv.org/abs/2104.08821) (contrastive sentence embeddings with in-batch negatives); [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242) (the source of the derivation); [Sentence-BERT](https://arxiv.org/abs/1908.10084) (the pooled-encoder baseline); [On the Sentence Embeddings from Pre-trained Language Models](https://arxiv.org/abs/2011.05864) (anisotropy and frequency); [Qwen3 Embedding](https://arxiv.org/abs/2506.05176) (the decoder-derived embedder and reranker, their training stages); [E5](https://arxiv.org/abs/2212.03533) (weak pairs then supervised pairs; the prefix convention); [LLM2Vec](https://arxiv.org/abs/2404.05961) (decoder-to-encoder adaptation beyond changing the mask); [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) (nested dimensions); [MTEB](https://arxiv.org/abs/2210.07316) (the benchmark and its pitfalls).
- Code and documentation: [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (pooling, padding side and instruction format on the card); [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) (the yes/no scoring prompt); [sentence-transformers](https://github.com/huggingface/sentence-transformers) (losses, cached-gradient loss, quantization helpers); [mteb](https://github.com/embeddings-benchmark/mteb).
- Courses, tutorials and blogs: [Sentence Transformers training overview](https://sbert.net/docs/sentence_transformer/training_overview.html); Hugging Face, [Embedding quantization](https://huggingface.co/blog/embedding-quantization) (int8 and binary with rescoring); [CS224N](https://web.stanford.edu/class/cs224n/) (the retrieval lecture).
-->

<!-- Placeholder (stage 2, cells step): every code cell of the section is
written; the prose comes later. Each placeholder names the `##` heading its
text goes under and what that text says. -->

<!-- Placeholder (prose, opening): start from the reader: a half-page recap of what the raw encoder of sec_lm_bert outputs (one vector per token, no vector per text); mean-pool the token vectors of every chunk of this book and retrieve by cosine for a question about the book; the book corpus (:numref:`sec_lm_retrieval`) and the book's question set (:numref:`sec_lm_rag`) in one sentence each. -->

```{.python .input #text-embeddings-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #text-embeddings-extras}
%%tab pytorch
import collections
import json
import pandas as pd
import time
import zipfile
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss)
```

<!-- Placeholder (prose, opening): the models (the raw encoder of the previous section, the released embedder and the reranker of this chapter), the flags of the CPU path (the chunks of the dev documents only) and of regeneration (D2L_RERUN=emb-train retrains every model and writes the results file and both checkpoints; a normal build reads them and checks them live), and the training settings: 20,000 question-passage pairs, batch 64, temperature 0.05, three learning rates checked on held-out pairs, three seeds. -->

```{.python .input #text-embeddings-config}
%%tab pytorch
ENC = 'answerdotai/ModernBERT-base'  # the raw encoder of the last section
EMB_QWEN, RERANKER = 'Qwen/Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Reranker-0.6B'
GPU = torch.cuda.is_available()
SMALL = not GPU  # the CPU path: the chunks of the dev documents only
DTYPE = torch.bfloat16 if GPU else torch.float32
ARTIFACTS = {  # name: (registered key, the file a regeneration writes)
    'emb-train': ('lm-enc-emb-results-v1', 'lm-enc-emb-results.json'),
    'emb-ckpt': ('lm-enc-modernbert-embed-v1', 'lm-enc-modernbert-embed.zip'),
    'emb-ckpt-mrl': ('lm-enc-modernbert-embed-mrl-v1',
                     'lm-enc-modernbert-embed-mrl.zip')}
RERUN = d2l.lm_rerun(ARTIFACTS)  # D2L_RERUN: one run writes all three
RERUN = set(ARTIFACTS) if RERUN else RERUN
assert GPU or not RERUN, 'regenerating the results needs a GPU'
N_PAIRS, N_HELD, B, TAU = 20_000, 300, 64, 0.05
LRS, SEEDS = (1e-5, 3e-5, 1e-4), (0, 1, 2)
N_CAND, LR_SEED = 5_000, 3  # held-out candidates; the lr check's own seed
MRL_DIMS = (768, 512, 256, 128, 64)  # the Matryoshka prefixes
HARD_RANKS, AUDIT_N = (11, 50), 500  # mined negatives; audited per bucket
BUCKETS = ((1, 5), (6, 10), (11, 30), (31, 50))
MAX_LEN = {'chunk': 1024, 'query': 128, 'nq_query': 48, 'nq_passage': 160}
PREFIX = {'query': 'query: ', 'passage': 'passage: '}  # E5's convention
QWEN = d2l._EMBEDDERS[EMB_QWEN]  # pooling, padding side, query format
INSTRUCT = QWEN['query_format']  # from the model card
CONFIG = dict(  # what the results are made with; reading them checks it
    model=ENC, revision=d2l.HF_ENCODER_REVISIONS[ENC], corpus=d2l.R3_VERSION,
    pairs=N_PAIRS, held=N_HELD, cands=N_CAND, batch=B, tau=TAU, lrs=LRS,
    lr_seed=LR_SEED, seeds=SEEDS,
    mrl_dims=MRL_DIMS, hard_ranks=HARD_RANKS, audit_n=AUDIT_N,
    buckets=BUCKETS, max_len=MAX_LEN, prefix=PREFIX, pooling='mean',
    reranker=RERANKER, reranker_revision=d2l.HF_ENCODER_REVISIONS[RERANKER])

def result_path(name):
    """The file this run regenerates for name, else the registered one."""
    key, file = ARTIFACTS[name]
    return d2l.lm_artifact(key, file if name in RERUN else None)

def read_result(name):
    """The results of name, checked against this notebook's settings."""
    with open(result_path(name)) as f:
        result = json.load(f)
    assert result['config'] == json.loads(json.dumps(CONFIG)), \
        f'{name} was made with other settings: D2L_RERUN={name}'
    return result

print(pd.DataFrame({'model': [ENC, EMB_QWEN, RERANKER], 'revision': [
    d2l.HF_ENCODER_REVISIONS[m][:12] for m in (ENC, EMB_QWEN, RERANKER)]},
                   index=['encoder', 'embedder', 'reranker']))
print(f'query instruction: {INSTRUCT!r}')
print(f'GPU: {GPU}; regenerating: {", ".join(sorted(RERUN)) or "none"}')
```

<!-- Placeholder (stage 2): the released embedder and the reranker, pinned like the encoder of the previous section. -->

```{.python .input #text-embeddings-revisions}
%%tab pytorch
#@save
d2l.HF_ENCODER_REVISIONS['Qwen/Qwen3-Embedding-0.6B'] = (
    '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
d2l.HF_ENCODER_REVISIONS['Qwen/Qwen3-Reranker-0.6B'] = (
    'e61197ed45024b0ed8a2d74b80b4d909f1255473')
```

<!-- Placeholder (stage 2): question-passage pairs from Natural Questions, pinned to a commit and downloaded at run time (never redistributed). The three results of the regeneration run (the results file and the two checkpoints) are added to this cell when they are registered. -->

```{.python .input #text-embeddings-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['nq-pairs'] = (
    'https://huggingface.co/datasets/sentence-transformers/'
    'natural-questions/resolve/f9e894e1081e206e577b4eaa9ee6de2b06ae6f17/'
    'pair/train-00000-of-00001.parquet',
    '545c1495f10231652b59e67f7ab02e6d21252fd0')
```

<!-- Placeholder (prose, ## Pooled Encoder States as Text Vectors): the previous section kept one vector per token and trained a head per task; here one vector per text serves every task. Three ways to pool: the first token ([CLS]), the mean over the tokens the mask keeps, the last token (decoders); the mask decides, so left and right padding both work. -->

```{.python .input #text-embeddings-pool-def}
%%tab pytorch
#@save
def pool_hidden(hidden, attention_mask, mode):
    """One vector per sequence from hidden states (B, T, m): the first
    ('cls'), the mean ('mean') or the last ('last') of the positions the
    mask keeps; left and right padding both work."""
    mask = attention_mask.to(hidden.dtype)
    if mode == 'mean':
        return (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    if mode == 'cls':
        return hidden[rows, attention_mask.argmax(1)]
    if mode == 'last':
        T = attention_mask.shape[1]
        return hidden[rows, T - 1 - attention_mask.flip(1).argmax(1)]
    raise ValueError(f'unknown pooling {mode!r}')
```

<!-- Placeholder (prose, ## Pooled Encoder States as Text Vectors): encoding many texts: batches of similar length (sorted by length, the input order restored), the vectors scaled to unit length so that a dot product is a cosine; texts longer than max_len are cut, and the function says how many (a silent cut is a pitfall). -->

```{.python .input #text-embeddings-embed-def}
%%tab pytorch
#@save
def embed_texts(model, tokenizer, texts, pooling='mean', batch_size=64,
                max_len=512, normalize=True, prefix='', padding_side=None,
                device=None):
    """Vectors (N, m), float32 on the CPU, of prefix + text for each text:
    pooled last hidden states of an encoder, unit length if normalize.
    Batches hold texts of similar length; the output is in input order."""
    device = device or next(model.parameters()).device
    side = padding_side or tokenizer.padding_side
    pad = tokenizer.pad_token_id
    ids = tokenizer([prefix + t for t in texts], truncation=True,
                    max_length=max_len)['input_ids']
    full = tokenizer([prefix + t for t in texts])['input_ids']
    n_cut = sum(len(a) < len(b) for a, b in zip(ids, full))
    if n_cut:
        print(f'{n_cut} of {len(texts)} texts truncated to {max_len} tokens')
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
    out = torch.empty(len(ids), model.config.hidden_size)
    for j in range(0, len(order), batch_size):
        rows = [ids[i] for i in order[j:j + batch_size]]
        T = max(len(r) for r in rows)
        fill = lambda r, v: ([v] * (T - len(r)) + r if side == 'left'
                             else r + [v] * (T - len(r)))
        X = torch.tensor([fill(r, pad) for r in rows], device=device)
        mask = torch.tensor([fill([1] * len(r), 0) for r in rows],
                            device=device)
        with torch.inference_mode():
            h = model(input_ids=X, attention_mask=mask).last_hidden_state
            z = d2l.pool_hidden(h.float(), mask, pooling)
            if normalize:
                z = F.normalize(z, dim=-1)
        out[order[j:j + batch_size]] = z.cpu()
    return out
```

<!-- Placeholder (prose, opening): how a ranking is scored: Recall@k is 1 when some relevant chunk is among the first k (what the next sections need: the answer was retrieved), MRR@10 the reciprocal rank of the first relevant chunk, nDCG@10 the discounted gain against the ideal order; :numref:`sec_lm_retrieval` treats the metrics in full. -->

```{.python .input #text-embeddings-metrics-def}
%%tab pytorch
#@save
def retrieval_metrics(ranked_ids, gold_ids, ks=(1, 5, 10, 100)):
    """Per-query metrics of ranked lists against gold sets (binary
    relevance): recall@k is 1 if a relevant id is among the first k (the
    question is answerable from what was retrieved), mrr@10 the reciprocal
    rank of the first relevant id within 10, ndcg@10 the discounted gain
    over the ideal ordering. NaN for a query with an empty gold set."""
    out = {f'recall@{k}': [] for k in ks} | {'mrr@10': [], 'ndcg@10': []}
    for ranked, gold in zip(ranked_ids, gold_ids):
        gold = set(gold)
        if not gold:
            for v in out.values():
                v.append(np.nan)
            continue
        rel = np.array([i in gold for i in ranked], dtype=float)
        for k in ks:
            out[f'recall@{k}'].append(float(rel[:k].any()))
        hits = np.flatnonzero(rel[:10])
        out['mrr@10'].append(1 / (hits[0] + 1) if len(hits) else 0.0)
        disc = 1 / np.log2(np.arange(2, 12))
        idcg = disc[:min(len(gold), 10)].sum()
        out['ndcg@10'].append((rel[:10] * disc[:len(rel[:10])]).sum() / idcg)
    return {k: np.array(v) for k, v in out.items()}
```

<!-- Placeholder (prose, opening): the book corpus in sections, the answerable dev questions of the book's question set, the relevant chunks of each question, and the floor: BM25 counts words and has no training at all. Intervals resample documents, since questions about one document are not independent. -->

```{.python .input #text-embeddings-book}
%%tab pytorch
FULL = d2l.book_corpus('sections')
CORPUS = d2l.BookCorpus(FULL.manifest, FULL.docs.values(), [
    c for c in FULL.chunks if FULL.split_of(c['doc_id']) == 'dev'],
                        FULL.tag) if SMALL else FULL
CHUNKS = CORPUS.texts('dense')
ITEMS = d2l.book_qa('dev', answerable=True)
QUESTIONS = [q['question'] for q in ITEMS]
GOLD = [d2l.gold_chunk_ids(q, CORPUS) for q in ITEMS]
CLUSTER = [q['gold_spans'][0]['doc_id'] for q in ITEMS]  # one per document
RANKED = {'BM25': [[h.chunk_id for h in hits] for hits in d2l.BookRetriever(
    CORPUS, dense=None).search_batch(QUESTIONS, k=10)]}
METRICS = {'BM25': d2l.retrieval_metrics(RANKED['BM25'], GOLD, ks=(1, 5, 10))}

def ci(x):
    """Mean of per-question scores with its document-clustered interval."""
    b = d2l.bootstrap_ci(x, cluster=CLUSTER)
    return f'{b.mean:.3f} [{b.ci[0]:.3f}, {b.ci[1]:.3f}]'

print(f'{len(CORPUS):,} chunks of {len({c["doc_id"] for c in CORPUS.chunks})}'
      f' documents; {len(ITEMS)} questions from {len(set(CLUSTER))} '
      f'documents, {np.mean([len(g) for g in GOLD]):.2f} relevant chunks each')
print(f'BM25: Recall@10 {ci(METRICS["BM25"]["recall@10"])}, '
      f'nDCG@10 {ci(METRICS["BM25"]["ndcg@10"])}')
if SMALL:
    print('CPU path: the chunks of the dev documents only, an easier problem; '
          'the numbers in the text come from the GPU run over all chunks')
```

<!-- Placeholder (prose, opening): the raw encoder of the previous section, one forward over every chunk and question per pooling: the first token, the mean, and the mean with the corpus mean vector subtracted; the hits are poor. -->

```{.python .input #text-embeddings-opening}
%%tab pytorch
encoder, enc_tok = d2l.load_encoder(ENC, dtype=DTYPE)

def top10(S, keep=None):
    """Ids of the ten best chunks per row of the scores S (queries, chunks),
    among the chunks keep marks if given."""
    S = S if keep is None else S.masked_fill(~keep, -torch.inf)
    return [[CORPUS.ids[i] for i in row]
            for row in S.topk(10).indices.tolist()]

RAW = {}  # pooling: unit vectors of the questions and of the chunks
for pooling in ('cls', 'mean'):
    RAW[pooling] = (
        d2l.embed_texts(encoder, enc_tok, QUESTIONS, pooling,
                        max_len=MAX_LEN['query']),
        d2l.embed_texts(encoder, enc_tok, CHUNKS, pooling, batch_size=16,
                        max_len=MAX_LEN['chunk']))
mu = RAW['mean'][1].mean(0)  # the mean chunk vector, removed from both sides
RAW['centered'] = tuple(F.normalize(Z - mu, dim=-1) for Z in RAW['mean'])
ROWS = dict(zip(('raw CLS', 'raw mean', 'raw centered mean'), RAW.values()))
for name, (Q, X) in ROWS.items():
    RANKED[name] = top10(Q @ X.T)
    METRICS[name] = d2l.retrieval_metrics(RANKED[name], GOLD, ks=(1, 5, 10))
print(f'Recall@10: raw mean {ci(METRICS["raw mean"]["recall@10"])}, '
      f'BM25 {ci(METRICS["BM25"]["recall@10"])}')
for i in range(3):  # the first three questions, top 1 against gold
    print(f'\n{QUESTIONS[i]}\n  top 1: '
          f'{CORPUS.header(CORPUS[RANKED["raw mean"][i][0]])}\n  gold:  '
          f'{CORPUS.header(CORPUS[min(GOLD[i])])}')
del encoder
d2l.lm_free()
```

<!-- Placeholder (prose, ## Pooled Encoder States as Text Vectors): anisotropy: the mean cosine between unrelated texts. For unit vectors z_1..z_n it equals (||sum z_i||^2 - n) / (n (n - 1)), so a long mean vector means a high mean cosine (one line); Wang and Isola's alignment and uniformity, the two quantities the derivation below works with. -->

```{.python .input #text-embeddings-geometry-def}
%%tab pytorch
#@save
def anisotropy(Z, n_pairs=200_000, seed=0):
    """Mean cosine similarity between distinct rows of Z, over n_pairs
    random pairs, or over all pairs if n_pairs is None or covers them."""
    Z = F.normalize(torch.as_tensor(Z, dtype=torch.float64), dim=-1)
    n = len(Z)
    if n_pairs is None or n_pairs >= n * (n - 1) // 2:
        G = Z @ Z.T
        return float((G.sum() - G.diagonal().sum()) / (n * (n - 1)))
    rng = np.random.default_rng(seed)
    i, j = rng.integers(n, size=n_pairs), rng.integers(n - 1, size=n_pairs)
    j = j + (j >= i)  # a partner other than i, uniformly
    return float((Z[i] * Z[j]).sum(-1).mean())

#@save
def align_uniform(Zx, Zy, t=2.0, Zu=None):
    """Wang and Isola's alignment, the mean squared distance between the
    unit vectors of positive pairs (rows of Zx and Zy), and uniformity,
    the log of the mean of exp(-t ||z - z'||^2) over the distinct pairs
    of the rows of Zu, or of all rows of Zx and Zy if Zu is None."""
    Zx, Zy = (F.normalize(torch.as_tensor(Z, dtype=torch.float64), dim=-1)
              for Z in (Zx, Zy))
    Zu = (torch.cat([Zx, Zy]) if Zu is None else F.normalize(
        torch.as_tensor(Zu, dtype=torch.float64), dim=-1))
    sq = torch.pdist(Zu) ** 2
    uniformity = torch.logsumexp(-t * sq, 0) - math.log(len(sq))
    return float(((Zx - Zy) ** 2).sum(-1).mean()), float(uniformity)
```

<!-- Placeholder (prose, ## Pooled Encoder States as Text Vectors): the three poolings on 2,000 random chunks: all three crowd into a narrow cone; centering removes the mean direction and helps retrieval somewhat, but the raw geometry is dominated by token frequency (Li et al. 2020, Ethayarajh 2019), not by meaning. The identity is checked on the same chunks. -->

```{.python .input #text-embeddings-anisotropy}
%%tab pytorch
SAMPLE = np.random.default_rng(0).choice(len(CORPUS), min(2000, len(CORPUS)),
                                         replace=False)
print(pd.DataFrame([dict(pooling=name, anisotropy=d2l.anisotropy(X[SAMPLE]),
                         recall_10=METRICS[name]['recall@10'].mean(),
                         ndcg_10=METRICS[name]['ndcg@10'].mean())
                    for name, (Q, X) in ROWS.items()]).round(3).to_string(
                        index=False))
Z = F.normalize(ROWS['raw mean'][1][SAMPLE].double(), dim=-1)  # in float64
n = len(Z)
exact = d2l.anisotropy(Z, n_pairs=None)  # every pair
identity = ((Z.sum(0) ** 2).sum() - n) / (n * (n - 1))
print(f'mean cosine over all {n * (n - 1) // 2:,} pairs {exact:.6f}, from '
      f'the sum of the vectors {identity:.6f}; squared length of the mean '
      f'vector {(Z.mean(0) ** 2).sum():.6f}')
assert abs(exact - identity) < 1e-9
```

<!-- Placeholder (prose, ## Contrastive Training): InfoNCE in the section's notation (eq_lm_infonce): for a batch of B question-passage pairs, the scores s_ij = q_i . d_j / tau and the cross-entropy of the right passage among all B; the other B - 1 passages of the batch are the negatives, so the batch size is the number of negatives; the log B ceiling of sec_mdl-infonce restated (not re-proved); the temperature. -->

```{.python .input #text-embeddings-infonce-def}
%%tab pytorch
#@save
def info_nce_loss(q, d, tau=0.05, hard_neg=None, symmetric=False):
    """InfoNCE with in-batch negatives for unit vectors q, d (B, m), row i
    of d the positive passage of query i: the cross-entropy of the scores
    q_i . d_j / tau over all d_j, and over all hard_neg_j too if given.
    symmetric averages in the passage-to-query direction (each d_i
    against all q_j). Returns the loss and the share of queries whose
    positive scores highest."""
    S = q @ (d if hard_neg is None else torch.cat([d, hard_neg])).T / tau
    labels = torch.arange(len(q), device=q.device)
    loss = F.cross_entropy(S, labels)
    if symmetric:
        loss = (loss + F.cross_entropy(S[:, :len(q)].T, labels)) / 2
    return loss, (S.argmax(1) == labels).float().mean()
```

<!-- Placeholder (prose, ## Contrastive Training): the gradient with respect to a score is (p_ij - delta_ij) / (tau B) (one display): the positive is pulled with weight 1 - p_ii, which equals the total push on the negatives, and each negative is pushed in proportion to its softmax weight. The cell checks the identity with autograd in float64, and prints how strongly a temperature of 0.05 favours the negatives that score highest. -->

```{.python .input #text-embeddings-infonce-grad}
%%tab pytorch
gen = torch.Generator().manual_seed(0)
q, d = (F.normalize(torch.randn(32, 16, generator=gen, dtype=torch.float64),
                    dim=-1).requires_grad_() for _ in range(2))
loss, _ = d2l.info_nce_loss(q, d, TAU)
loss.backward()
P = torch.softmax(q @ d.T / TAU, dim=1).detach()
G = (P - torch.eye(len(q), dtype=P.dtype)) / (TAU * len(q))  # dL/ds_ij
err = max((q.grad - G @ d.detach()).abs().max().item(),
          (d.grad - G.T @ q.detach()).abs().max().item())
pull_push = (G.sum(1)).abs().max().item()  # 1 - p_ii = sum_{j != i} p_ij
print(f'largest gradient error {err:.1e}; largest |pull - push| per query '
      f'{pull_push:.1e}')
assert err < 1e-10 and pull_push < 1e-12
print(pd.DataFrame({'score gap': [0.1, 0.3], f'weight ratio at tau={TAU}':
                    np.exp(np.array([0.1, 0.3]) / TAU)}).round(1))
```

<!-- Placeholder (prose, ### Alignment and Uniformity): the derivation: as the batch grows, InfoNCE minus log B tends to an alignment term (-E[q . d+] / tau) plus a uniformity term (the log of the mean of exp(q . d / tau) over the data) (eq_lm_align_uniform); the smallest case: four unit vectors on a circle, pairs (1, 2) and (3, 4) positive; the square is the most uniform, all-coincident the best aligned, two coincident pairs on opposite sides both; with the four points as the whole data, the in-batch loss of the four pairs minus log 4 is the limit exactly. Uniformity rewards spreading the vectors, which removes the shared direction behind anisotropy. -->

```{.python .input #text-embeddings-uniformity}
%%tab pytorch
rows = []
for name, deg in {'square': (0, 90, 180, 270),
                  'two coincident pairs': (0, 0, 180, 180),
                  'all coincident': (0, 0, 0, 0)}.items():
    rad = np.radians(deg)
    Z = torch.tensor(np.stack([np.cos(rad), np.sin(rad)], 1))
    align, unif = d2l.align_uniform(Z[[0, 2]], Z[[1, 3]])
    row = dict(configuration=name, alignment=align, uniformity=unif)
    for tau in (0.05, 0.5):  # every point a query; its partner the positive
        loss, _ = d2l.info_nce_loss(Z, Z[[1, 0, 3, 2]], tau)
        row[f'InfoNCE - log 4, tau={tau}'] = loss.item() - math.log(4)
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))
```

<!-- Placeholder (prose, ### Training on Question-Answer Pairs): the training loop: questions get the prefix "query: ", passages "passage: " (E5's convention); mean pooling; AdamW with warmup and decay; the batch sampler never puts a question or a passage text twice in one batch, since a duplicate would be scored as a negative of its own question (a false negative); with hard negatives, one more passage per question joins the candidates; with Matryoshka dimensions, the loss is averaged over the leading 768, 512, ..., 64 coordinates. -->

```{.python .input #text-embeddings-train-def}
%%tab pytorch
#@save
def no_duplicate_batches(queries, passages, batch_size, seed=0,
                         negatives=None):
    """Batches of indices into the pairs (queries[i], passages[i]), in a
    seeded random order, none holding a query text twice or a passage text
    twice (with negatives[i], an extra passage per pair, counted among the
    passages): a duplicate would be a negative of its own query. A pair
    that would repeat a text waits for a later batch."""
    pending = collections.deque(
        np.random.default_rng(seed).permutation(len(queries)).tolist())
    batches = []
    while pending:
        batch, seen, waiting = [], set(), []
        while pending and len(batch) < batch_size:
            i = pending.popleft()
            texts = {('q', queries[i]), ('p', passages[i])}
            if negatives is not None:
                texts.add(('p', negatives[i]))
            if texts & seen:
                waiting.append(i)
            else:
                batch.append(i)
                seen |= texts
        pending.extendleft(reversed(waiting))
        batches.append(batch)
    return batches

#@save
def train_embedder(model, tokenizer, pairs, *, pooling='mean', tau=0.05,
                   batch_size=64, lr=3e-5, steps=None, epochs=1,
                   hard_negs=None, mrl_dims=None, q_prefix='query: ',
                   d_prefix='passage: ', max_len_q=48, max_len_d=160,
                   seed=0):
    """Train an encoder in place on (query, passage) pairs: InfoNCE with
    in-batch negatives at temperature tau, plus hard_negs[i] as one more
    candidate per pair if given; with mrl_dims, the mean of the losses on
    the leading mrl_dims coordinates (Matryoshka). Batches from
    no_duplicate_batches; AdamW (weight decay 0.01 on matrices), linear
    warmup over the first 10% of the steps then linear decay, gradients
    clipped to norm 1, bf16 autocast on a GPU. Returns the loss per step,
    the steps, the training seconds and the peak GPU memory in GiB."""
    device = next(model.parameters()).device
    cuda = device.type == 'cuda'
    qs, ds = [q for q, _ in pairs], [d for _, d in pairs]
    plan = [b for e in range(epochs) for b in d2l.no_duplicate_batches(
        qs, ds, batch_size, seed + e, hard_negs)]
    steps = steps or len(plan)

    def embed(texts, prefix, max_len):  # pooled, not yet unit length
        ids = tokenizer([prefix + t for t in texts], truncation=True,
                        max_length=max_len)['input_ids']
        T = max(map(len, ids))
        X = torch.tensor([r + [tokenizer.pad_token_id] * (T - len(r))
                          for r in ids], device=device)
        M = torch.tensor([[1] * len(r) + [0] * (T - len(r)) for r in ids],
                         device=device)
        with torch.autocast(device.type, torch.bfloat16, enabled=cuda):
            h = model(input_ids=X, attention_mask=M).last_hidden_state
        return d2l.pool_hidden(h.float(), M, pooling)

    def loss_fn(b):
        q, d = embed([qs[i] for i in b], q_prefix, max_len_q), embed(
            [ds[i] for i in b], d_prefix, max_len_d)
        h = (None if hard_negs is None else
             embed([hard_negs[i] for i in b], d_prefix, max_len_d))
        cut = lambda Z, k: None if Z is None else F.normalize(Z[:, :k], dim=-1)
        dims = mrl_dims or (q.shape[1],)
        return sum(d2l.info_nce_loss(cut(q, k), cut(d, k), tau, cut(h, k))[0]
                   for k in dims) / len(dims)

    model.train()
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': [p for p in params if p.ndim >= 2], 'weight_decay': 0.01},
        {'params': [p for p in params if p.ndim < 2], 'weight_decay': 0.0}],
        lr=lr)
    n_warm = max(1, round(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / n_warm if s < n_warm else
        max(0.0, (steps - s) / max(1, steps - n_warm))))
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0, losses = time.perf_counter(), []
    for s in range(steps):
        loss = loss_fn(plan[s % len(plan)])
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.detach())
    if cuda:
        torch.cuda.synchronize(device)
    model.eval()
    return dict(loss=torch.stack(losses).tolist(), steps=steps,
                seconds=time.perf_counter() - t0,
                peak_gib=(torch.cuda.max_memory_allocated(device) / 2**30
                          if cuda else float('nan')))
```

<!-- Placeholder (prose, ### Training on Question-Answer Pairs): the training pairs: Natural Questions (search-engine questions with a Wikipedia passage that answers them); none comes from the book, so the book measures transfer. 20,000 pairs from one seed-0 permutation (the next section reuses the rule). For the learning rate, the next 300 pairs whose question and passage the training pairs do not contain (a held-out pair with a training passage would favour the rate that fits the training passages best), each question ranked among 5,000 unseen passages so that the score does not saturate. Some passages answer several questions: these pairs must not share a batch. -->

```{.python .input #text-embeddings-nq}
%%tab pytorch
nq = pd.read_parquet(d2l.download_keyed('nq-pairs'))
perm = np.random.default_rng(0).permutation(len(nq))
TRAIN_IDX = perm[:N_PAIRS]
PAIRS = list(nq.iloc[TRAIN_IDX][['query', 'answer']].itertuples(
    index=False, name=None))
qtext, ptext = nq['query'].to_numpy(), nq['answer'].to_numpy()
seen_q, seen_p = {q for q, _ in PAIRS}, {d for _, d in PAIRS}
unseen = [i for i in perm[N_PAIRS:]  # rows sharing no text with training
          if qtext[i] not in seen_q and ptext[i] not in seen_p]
HELD = [(qtext[i], ptext[i]) for i in unseen[:N_HELD]]
CANDS = list(dict.fromkeys([d for _, d in HELD] + [  # distinct passages
    ptext[i] for i in unseen[N_HELD:]]))[:N_CAND]
shared = collections.Counter(d for _, d in PAIRS)
print(f'{len(nq):,} pairs, {nq.answer.nunique():,} distinct passages; '
      f'training on {len(PAIRS):,} pairs with {len(shared):,} distinct '
      f'passages ({sum(n for n in shared.values() if n > 1):,} pairs share '
      f'theirs); {len(HELD)} unseen pairs held out for the learning rate, '
      f'ranked among {len(CANDS):,} unseen passages')
```

<!-- Placeholder (prose, ### Training on Question-Answer Pairs): helpers for the runs: the raw encoder in fp32 for training, encoding with the prefixes, the top 10 of a trained encoder over the book (stored in the results, also over the dev documents alone for the CPU path), the held-out score that picks the learning rate (never the book), mining, and the checkpoint as a zip of bf16 weights. -->

```{.python .input #text-embeddings-run-defs}
%%tab pytorch
DEV = torch.tensor([CORPUS.split_of(c['doc_id']) == 'dev'
                    for c in CORPUS.chunks])
FULL_INDEX = {c: i for i, c in enumerate(FULL.ids)}

def fresh():
    """The raw encoder in fp32, to train (with bf16 autocast)."""
    return d2l.load_encoder(ENC, dtype=torch.float32)[0]

def encode(model, texts, kind, prefix=''):
    return d2l.embed_texts(model, enc_tok, texts, 'mean', max_len=MAX_LEN[
        kind], prefix=prefix, batch_size=16 if kind == 'chunk' else 64)

def book_rankings(model):
    """The top 10 per dev question over the corpus and over the dev
    documents only, as indices into the full corpus."""
    S = encode(model, QUESTIONS, 'query', PREFIX['query']) @ encode(
        model, CHUNKS, 'chunk', PREFIX['passage']).T
    return {name: [[FULL_INDEX[c] for c in row] for row in top10(S, keep)]
            for name, keep in (('top10', None), ('top10_dev', DEV))}

def stored(run):
    """A stored run's top 10 over this build's corpus, as chunk ids."""
    return [[FULL.ids[i] for i in row]
            for row in run['top10_dev' if SMALL else 'top10']]

def nq_rr(model):
    """Reciprocal rank (0 beyond 10) of each held-out question's passage
    among the unseen candidate passages."""
    S = encode(model, [q for q, _ in HELD], 'nq_query', PREFIX['query']) @ \
        encode(model, CANDS, 'nq_passage', PREFIX['passage']).T
    gold = [[CANDS.index(d)] for _, d in HELD]
    return d2l.retrieval_metrics(S.topk(10).indices.tolist(), gold, ks=(
        10,))['mrr@10']

def mine(model, depth=50):
    """The distinct training passages and, per training question, the
    depth passages the model ranks highest, its own text left out."""
    texts = sorted({d for _, d in PAIRS})
    where = {t: i for i, t in enumerate(texts)}
    own = torch.tensor([where[d] for _, d in PAIRS])
    Q = encode(model, [q for q, _ in PAIRS], 'nq_query', PREFIX['query'])
    D = encode(model, texts, 'nq_passage', PREFIX['passage'])
    ranks = []
    for i in range(0, len(Q), 1024):
        S = Q[i:i + 1024] @ D.T
        S[torch.arange(len(S)), own[i:i + 1024]] = -torch.inf
        ranks.append(S.topk(depth).indices)
    return texts, torch.cat(ranks).numpy()

def write_json(obj):
    def write(path):
        with open(path, 'w') as f:
            json.dump(obj, f)
    return write

BASE = dict(model=ENC, revision=d2l.HF_ENCODER_REVISIONS[ENC],
            pooling='mean', prefix=PREFIX)  # how a checkpoint is used

def save_ckpt(model):
    """A writer for the encoder's weights as a zip of bf16 safetensors."""
    members = {'model.safetensors': save_safetensors({
        k: v.to(torch.bfloat16).contiguous().cpu()
        for k, v in model.state_dict().items()}),
               'base.json': json.dumps(BASE).encode()}
    return lambda path: d2l.lm_write_zip(path, members)

def load_ckpt(path):
    """The raw encoder with the trained weights of a checkpoint zip."""
    model, _ = d2l.load_encoder(ENC, dtype=DTYPE)
    with zipfile.ZipFile(path) as z:
        assert json.loads(z.read('base.json')) == BASE, f'{path}: other base'
        model.load_state_dict(load_safetensors(z.read('model.safetensors')))
    return model.eval()
```

<!-- Placeholder (prose, ### Training on Question-Answer Pairs): the regeneration (D2L_RERUN=emb-train, about 12 minutes on one card): (a) the learning rate: three rates, each trained with a seed of its own that no reported model uses (so the reported runs are not the winners of a selection), scored on the 300 unseen NQ pairs, never on the book; (b) the chosen rate with seeds 0, 1 and 2, for the training noise; the seed-0 model is saved. A normal build reads the stored results and loads the saved checkpoint. -->

```{.python .input #text-embeddings-train}
%%tab pytorch
if 'emb-train' in RERUN:
    TIC = time.perf_counter()
    OUT = dict(config=CONFIG, qids=[q['qid'] for q in ITEMS], lr_check={},
               runs={})

    def train(lr, seed=0, **kw):
        """A fresh encoder trained on PAIRS, cast to DTYPE (bf16 on a GPU:
        evaluated as the saved weights are), and its training record."""
        model = fresh()
        r = d2l.train_embedder(
            model, enc_tok, PAIRS, tau=TAU, batch_size=B, lr=lr, seed=seed,
            q_prefix=PREFIX['query'], d_prefix=PREFIX['passage'],
            max_len_q=MAX_LEN['nq_query'], max_len_d=MAX_LEN['nq_passage'],
            **kw)
        return model.to(DTYPE), dict(
            lr=lr, seed=seed, seconds=r['seconds'], peak_gib=r['peak_gib'],
            steps=r['steps'], loss=[round(v, 4) for v in r['loss']])

    for lr in LRS:  # (a) the learning rate, on held-out NQ pairs
        model, _ = train(lr, LR_SEED)  # a seed no reported model uses
        OUT['lr_check'][str(lr)] = nq_rr(model).round(4).tolist()
        del model
        d2l.lm_free()
    OUT['lr'] = max(LRS, key=lambda lr: np.mean(OUT['lr_check'][str(lr)]))
    for seed in SEEDS:  # (b) the chosen rate, three seeds: training noise
        model, info = train(OUT['lr'], seed)
        OUT['runs'][f'contrastive seed {seed}'] = info | book_rankings(model)
        if seed == SEEDS[0]:  # kept for mining, and saved
            R2 = model
            d2l.lm_save_rerun(ARTIFACTS['emb-ckpt'][1], save_ckpt(R2))
        del model
        d2l.lm_free()
```

<!-- Placeholder (prose, ### Hard Negatives and False Negatives): (c) mining: the seed-0 model ranks the distinct training passages for every training question (its own passage left out); one hard negative per question is drawn uniformly from ranks 11 to 50. The audit: for 500 questions, one mined passage from each rank bucket (1-5, 6-10, 11-30, 31-50) is scored by Qwen3-Reranker-0.6B; P(yes) > 0.5 marks it as probably relevant, i.e. a false negative if trained against. -->

```{.python .input #text-embeddings-mine}
%%tab pytorch
if 'emb-train' in RERUN:
    texts, ranks = mine(R2)  # with the seed-0 model
    R2 = None
    d2l.lm_free()
    rng = np.random.default_rng(0)
    lo, hi = HARD_RANKS
    HARD = [texts[ranks[i, rng.integers(lo - 1, hi)]]
            for i in range(len(PAIRS))]
    reranker = d2l.load_hf(RERANKER, d2l.HF_ENCODER_REVISIONS[RERANKER],
                           dtype=DTYPE)
    OUT['audit'] = [  # P(yes) of one mined passage per bucket
        d2l.rerank_yes_no(*reranker, PAIRS[i][0], [
            texts[ranks[i, rng.integers(a - 1, b)]] for a, b in BUCKETS])
        .round(4).tolist()
        for i in rng.choice(len(PAIRS), AUDIT_N, replace=False)]
    del reranker
    d2l.lm_free()
```

<!-- Placeholder (prose, ### Hard Negatives and False Negatives): (d) the same recipe with the mined negative as one more candidate per question (seed 0); (e) the Matryoshka loss over 768, 512, 256, 128 and 64 coordinates (seed 0), saved. Then the results file: per run its training seconds, peak memory, losses and top 10 per dev question. -->

```{.python .input #text-embeddings-train-variants}
%%tab pytorch
if 'emb-train' in RERUN:
    for name, kw in (('contrastive + hard negatives', dict(hard_negs=HARD)),
                     ('Matryoshka', dict(mrl_dims=MRL_DIMS))):  # (d), (e)
        model, info = train(OUT['lr'], **kw)
        OUT['runs'][name] = info | book_rankings(model)
        if name == 'Matryoshka':
            d2l.lm_save_rerun(ARTIFACTS['emb-ckpt-mrl'][1], save_ckpt(model))
        del model
        d2l.lm_free()
    OUT['minutes'] = (time.perf_counter() - TIC) / 60
    d2l.lm_save_rerun(ARTIFACTS['emb-train'][1], write_json(OUT))
    print(f'regenerated in {OUT["minutes"]:.1f} min; {d2l.lm_peak()}')
```

<!-- Placeholder (prose, ### Hard Negatives and False Negatives): the stored results: the learning-rate check (each rate's MRR@10 with its interval, and the chosen rate minus each other, so the reader sees whether the choice was within noise), a training summary (one epoch is about 310 steps; the hard-negative run encodes a third text per pair), and the audit: the share of probably relevant mined passages per rank bucket, with intervals over the 500 questions. -->

```{.python .input #text-embeddings-train-read}
%%tab pytorch
EMB = read_result('emb-train')
assert EMB['qids'] == [q['qid'] for q in ITEMS], 'other dev questions'
best = np.array(EMB['lr_check'][str(EMB['lr'])])
lr_rows = []
for lr, rr in EMB['lr_check'].items():  # intervals over the 300 questions
    rr = np.array(rr)
    b, r = d2l.bootstrap_ci(rr), d2l.paired_bootstrap(best, rr)
    lr_rows.append({'lr': lr, 'MRR@10': b.mean, 'low': b.ci[0],
                    'high': b.ci[1], 'chosen minus this': r.mean,
                    'delta low': r.ci[0], 'delta high': r.ci[1]})
print(pd.DataFrame(lr_rows).round(3).to_string(index=False))
print('chosen lr:', EMB['lr'])
print(pd.DataFrame({name: dict(
    seconds=r['seconds'], peak_gib=r['peak_gib'], steps=r['steps'],
    loss_first_10=np.mean(r['loss'][:10]), loss_last_10=np.mean(r['loss'][
        -10:])) for name, r in EMB['runs'].items()}).T.round(2))
relevant = np.array(EMB['audit']) > 0.5  # (questions, buckets)
rates = [d2l.bootstrap_ci(col.astype(float)) for col in relevant.T]
print(pd.DataFrame({'mined ranks': [f'{a}-{b}' for a, b in BUCKETS],
                    'probably relevant': [b.mean for b in rates],
                    'low': [b.ci[0] for b in rates],
                    'high': [b.ci[1] for b in rates]}).round(3).to_string(
                        index=False))
print('falls from bucket to bucket:', bool(np.all(np.diff(
    relevant.mean(0)) <= 0)))
```

<!-- Placeholder (prose, ### Training on Question-Answer Pairs): the ladder on the book's dev questions: BM25, raw pooling, the contrastive models (seed 0 and the Matryoshka model live from their checkpoints; seeds 1 and 2 and the hard-negative model from the stored top 10), and the released embedder with and without its query instruction (last-token pooling, left padding; its chunk vectors are the retriever's). Anisotropy on the same 2,000 chunks; both live models are checked against their stored rankings (a swapped or wrongly registered checkpoint would fail). -->

```{.python .input #text-embeddings-ladder}
%%tab pytorch
for name, key in (('contrastive seed 0', 'emb-ckpt'),
                  ('Matryoshka', 'emb-ckpt-mrl')):
    model = load_ckpt(result_path(key))
    ROWS[name] = (encode(model, QUESTIONS, 'query', PREFIX['query']),
                  encode(model, CHUNKS, 'chunk', PREFIX['passage']))
    del model
    d2l.lm_free()
emb, emb_tok = d2l.load_encoder(EMB_QWEN, dtype=DTYPE,
                                padding_side=QWEN['padding_side'])
X5 = d2l.BookRetriever(FULL, dense=EMB_QWEN).chunk_vectors().float().cpu()[
    [FULL_INDEX[c] for c in CORPUS.ids]]
for name, fmt in (('Qwen3-Embedding', '{query}'),
                  ('Qwen3-Embedding + instruction', INSTRUCT)):
    ROWS[name] = (d2l.embed_texts(emb, emb_tok, [fmt.format(
        task=d2l.BOOK_TASK, query=q) for q in QUESTIONS], QWEN['pooling'],
                                  max_len=MAX_LEN['query']), X5)
RANKED |= {name: top10(Q @ X.T) for name, (Q, X) in ROWS.items()}
RANKED |= {name: stored(EMB['runs'][name]) for name in (
    'contrastive seed 1', 'contrastive seed 2',
    'contrastive + hard negatives')}
METRICS |= {name: d2l.retrieval_metrics(r, GOLD, ks=(1, 5, 10))
            for name, r in RANKED.items()}
print(pd.DataFrame({name: {k: m[k].mean() for k in m if k != 'ndcg@10'} | {
    'ndcg@10 [CI]': ci(m['ndcg@10']), 'anisotropy': d2l.anisotropy(
        ROWS[name][1][SAMPLE]) if name in ROWS else np.nan}
                    for name, m in METRICS.items()}).T.round(3).to_string())
for name in ('contrastive seed 0', 'Matryoshka'):  # live against stored
    reg = stored(EMB['runs'][name])
    same = np.mean([a == b for a, b in zip(RANKED[name], reg)])
    agree = np.mean(METRICS[name]['recall@10'] == d2l.retrieval_metrics(
        reg, GOLD, ks=(10,))['recall@10'])
    print(f'live {name} model against its stored ranking: same top 10 for '
          f'{same:.3f} of the questions, same Recall@10 for {agree:.3f}')
    assert SMALL or agree >= 0.98, f'{name}: checkpoint and results differ'
```

<!-- Placeholder (prose, ### Hard Negatives and False Negatives and ## Embedders Derived from Language Models): paired differences on the same questions, intervals resampling documents: raw mean pooling against BM25 (C1), contrastive training against raw pooling (C3, against the best raw pooling too), the mined negative (C4), the released embedder against the notebook's model (C5), and the instruction (C6); the seed range of the contrastive recipe is the noise floor each gain is read against. -->

```{.python .input #text-embeddings-ladder-compare}
%%tab pytorch
def delta(a, b, metric='recall@10'):
    """Paired difference b - a of a per-question metric, with its interval."""
    r = d2l.paired_bootstrap(METRICS[b][metric], METRICS[a][metric],
                             cluster=CLUSTER)
    return {'from': a, 'to': b, 'delta': r.mean, 'low': r.ci[0],
            'high': r.ci[1]}

seed_r10 = [METRICS[f'contrastive seed {s}']['recall@10'].mean()
            for s in SEEDS]
seed_range = max(seed_r10) - min(seed_r10)
best_raw = max(('raw CLS', 'raw mean', 'raw centered mean'),
               key=lambda n: METRICS[n]['recall@10'].mean())
D = pd.DataFrame([delta(a, b) for a, b in dict.fromkeys((  # no repeats
    ('raw mean', 'BM25'), ('raw mean', 'contrastive seed 0'),
    ('raw centered mean', 'contrastive seed 0'),
    (best_raw, 'contrastive seed 0'),
    ('contrastive seed 0', 'contrastive + hard negatives'),
    ('contrastive seed 0', 'Qwen3-Embedding + instruction'),
    ('Qwen3-Embedding', 'Qwen3-Embedding + instruction')))])
D['CI excludes 0'] = (D.low > 0) | (D.high < 0)
D['beyond seed range'] = D.delta.abs() > seed_range
print(D.round(3).to_string(index=False))
print(f'Recall@10 of the three seeds: {np.round(seed_r10, 3)}, range '
      f'{seed_range:.3f}; best raw pooling: {best_raw}')
gain = D[(D['from'] == best_raw) & (D['to'] == 'contrastive seed 0')].iloc[0]
assert SMALL or gain.low > 0, 'no gain from contrastive training: a bug'
```

<!-- Placeholder (prose, ### Alignment and Uniformity): alignment on the dev questions paired with each of their relevant chunks, uniformity on the 2,000 sampled chunks (the chunks' own spread, the quantity anisotropy is about; pooling questions and chunks would count the gap between the two prefixes as spread), before and after contrastive training and for the released embedder; the plot below projects 2,000 unit chunk vectors on the top two right singular vectors of each model's (uncentered) matrix, so the shared direction stays visible: raw mean pooling crowds around a long mean vector (the star), the contrastive model spreads around the origin. -->

```{.python .input #text-embeddings-alignment}
%%tab pytorch
CIDX = {c: j for j, c in enumerate(CORPUS.ids)}
qi, cj = map(list, zip(*[(i, CIDX[c]) for i, g in enumerate(GOLD)
                         for c in sorted(g)]))
print(f'alignment over {len(qi)} (question, relevant chunk) pairs, '
      f'uniformity over {len(SAMPLE):,} chunks')
print(pd.DataFrame({name: dict(zip(('alignment', 'uniformity'), (
    d2l.align_uniform(Q[qi], X[cj], Zu=X[SAMPLE])))) for name, (Q, X) in (
        (n, ROWS[n]) for n in ('raw mean', 'contrastive seed 0',
                               'Qwen3-Embedding + instruction'))}).T.round(3))
_, axes = d2l.plt.subplots(1, 2, figsize=(7, 3.6))
circle = np.linspace(0, 2 * np.pi, 200)
for ax, name in zip(axes, ('raw mean', 'contrastive seed 0')):
    Z = ROWS[name][1][SAMPLE].double()
    V = torch.linalg.svd(Z, full_matrices=False).Vh[:2]  # uncentered
    V = V * torch.sign(V @ Z.mean(0))[:, None]  # the mean on the + side
    P, m = (Z @ V.T).numpy(), (Z.mean(0) @ V.T).numpy()
    d2l.plot([P[:, 0], np.cos(circle), [m[0]]],
             [P[:, 1], np.sin(circle), [m[1]]], xlabel=name,
             legend=(['chunks', 'unit circle', 'mean vector']
                     if name == 'raw mean' else None),  # one legend
             fmts=('C0.', 'k-', 'C3*'), axes=ax)
    ax.set_aspect('equal')
```

<!-- Placeholder (prose, ## Smaller Vectors): three ways to store fewer bytes per vector: keep the leading coordinates (and rescale to unit length), keep 8 bits per coordinate, or keep one bit, the sign; binary vectors are compared by counting equal bits (the Hamming distance), with a popcount table. -->

```{.python .input #text-embeddings-compress-def}
%%tab pytorch
#@save
def truncate_embeddings(Z, dim):
    """The first dim coordinates of each row of Z, rescaled to unit length."""
    return F.normalize(Z[:, :dim], dim=-1)

#@save
def binarize(Z):
    """The signs of Z (N, m) as bits, eight per byte: uint8 (N, m / 8)."""
    return np.packbits(np.asarray(Z) > 0, axis=1)

#@save
def hamming_scores(Qbits, Dbits):
    """The number of equal bits of each packed query and each packed
    document (m minus their Hamming distance): int32 (nq, nd)."""
    pop = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
        1, dtype=np.uint8)  # set bits per byte value
    out = np.empty((len(Qbits), len(Dbits)), np.int32)
    for i in range(0, len(Qbits), 16):
        out[i:i + 16] = 8 * Qbits.shape[1] - pop[
            Qbits[i:i + 16, None] ^ Dbits[None]].sum(-1, dtype=np.int32)
    return out
```

<!-- Placeholder (prose, ## Smaller Vectors): Recall@10 against bytes per vector: truncation of the plain contrastive model, of the Matryoshka model and of the released embedder (trained for truncation down to 32 coordinates); int8 with one symmetric scale per coordinate taken from the corpus; binary vectors by Hamming distance alone (integer scores tie often; ties are put in random order, seeded, rather than in corpus order), and with the top 100 rescored by the float query against the document's signs (the documents stay one bit per coordinate). The memory table counts bytes per vector and per 10^8 vectors. -->

```{.python .input #text-embeddings-compress}
%%tab pytorch
def recall10(S):
    return d2l.retrieval_metrics(top10(S), GOLD, ks=(10,))['recall@10']

MODELS = {'contrastive': 'contrastive seed 0', 'Matryoshka': 'Matryoshka',
          'Qwen3-Embedding': 'Qwen3-Embedding + instruction'}
R10, rows = {}, []
gen = torch.Generator().manual_seed(0)  # breaks ties of equal bit counts
for label, name in MODELS.items():
    Q, X = ROWS[name]
    m = X.shape[1]
    for dim in (m, 512, 256, 128, 64):
        R10[label, dim] = recall10(d2l.truncate_embeddings(Q, dim) @
                                   d2l.truncate_embeddings(X, dim).T)
        rows.append((label, f'float32, {dim}', 4 * dim, R10[label, dim]))
    scale = X.abs().max(0).values / 127  # int8: one scale per coordinate
    rows.append((label, f'int8, {m}', m, recall10(
        Q @ (torch.round(X / scale).to(torch.int8).float() * scale).T)))
    Xb = d2l.binarize(X)
    H = torch.from_numpy(d2l.hamming_scores(d2l.binarize(Q), Xb)).float()
    H += 0.5 * torch.rand(H.shape, generator=gen)  # ties in random order
    rows.append((label, f'binary, {m}', m // 8, recall10(H)))
    top = H.topk(100).indices  # rescored: float query, document signs
    signs = torch.from_numpy(np.unpackbits(Xb, axis=1)).float() * 2 - 1
    S = torch.full_like(H, -torch.inf).scatter_(1, top, torch.einsum(
        'qm,qkm->qk', Q, signs[top]))
    R10[label, 'binary + rescoring'] = recall10(S)
    rows.append((label, f'binary + rescoring, {m}', m // 8,
                 R10[label, 'binary + rescoring']))
MEM = pd.DataFrame(rows, columns=['model', 'format', 'bytes', 'recall@10'])
MEM['recall@10'] = MEM['recall@10'].map(np.mean)
MEM['GB per 10^8 vectors'] = MEM['bytes'] * 1e8 / 1e9
print(MEM.round(3).to_string(index=False))
lines = [MEM[(MEM.model == l) & MEM.format.str.startswith('float')]
         for l in MODELS]
points = [MEM[(MEM.model == l) & MEM.format.str.startswith('binary +')]
          for l in MODELS]
d2l.plot([t.bytes.tolist() for t in lines + points],
         [t['recall@10'].tolist() for t in lines + points],
         'bytes per vector', 'Recall@10', xscale='log', figsize=(5, 3.5),
         legend=list(MODELS) + [f'{l}, binary, rescored' for l in MODELS],
         fmts=('C0-o', 'C1-s', 'C2-^', 'C0*', 'C1*', 'C2*'))
```

<!-- Placeholder (prose, ## Smaller Vectors): paired differences at 128 coordinates (the Matryoshka model against the truncated plain one, C7: two single-seed models, so the gap is read against the seed range of the full-width models as well as its interval; "keeps more" only if it clears both) and of binary vectors with rescoring against float vectors at full width, at 1/32 of the bytes. -->

```{.python .input #text-embeddings-compress-compare}
%%tab pytorch
pairs = [('Matryoshka, 128 - contrastive, 128', R10['Matryoshka', 128],
          R10['contrastive', 128])]
pairs += [(f'{l}: binary + rescoring - float', R10[l, 'binary + rescoring'],
           R10[l, ROWS[n][1].shape[1]]) for l, n in MODELS.items()]
C = pd.DataFrame([dict(comparison=c, delta=r.mean, low=r.ci[0], high=r.ci[1])
                  for c, r in ((c, d2l.paired_bootstrap(a, b, cluster=CLUSTER))
                               for c, a, b in pairs)])
C['CI excludes 0'] = (C.low > 0) | (C.high < 0)
C['beyond seed range'] = C.delta.abs() > seed_range
print(C.round(3).to_string(index=False))
print(f'seed range of Recall@10 at full width (a proxy for the training '
      f'noise at 128 coordinates): {seed_range:.3f}')
```

<!-- Placeholder (prose, ## Embedders Derived from Language Models): in practice: sentence-transformers loads the released embedder with its pooling, padding side and normalization from the model's own configuration; the notebook's instruction format and last-token pooling are checked against that configuration (a mismatch with a checkpoint's training convention is a silent pitfall); its vectors agree with embed_texts, and its MultipleNegativesRankingLoss at scale 1/tau is the InfoNCE loss above. Then LLM2Vec, NV-Embed and E5-Mistral one clause each; MTEB and why a leaderboard average does not predict a domain; contamination. -->

```{.python .input #text-embeddings-production}
%%tab pytorch
st = SentenceTransformer(
    EMB_QWEN, revision=d2l.HF_ENCODER_REVISIONS[EMB_QWEN],
    device='cuda' if GPU else 'cpu', model_kwargs={'dtype': DTYPE},
    processor_kwargs={'padding_side': QWEN['padding_side']})
card_task = ('Given a web search query, retrieve relevant passages that '
             'answer the query')  # the model's own query prompt
assert st.prompts['query'] == INSTRUCT.format(task=card_task, query='')
assert st[1].pooling_mode == 'lasttoken' and QWEN['pooling'] == 'last'
prompt = INSTRUCT.format(task=d2l.BOOK_TASK, query='')
ours = torch.cat([d2l.embed_texts(emb, emb_tok, CHUNKS[:16], QWEN['pooling'],
                                  max_len=MAX_LEN['chunk']),
                  d2l.embed_texts(emb, emb_tok, [prompt + q for q in
                                                 QUESTIONS[:4]],
                                  QWEN['pooling'])])
theirs = torch.cat([st.encode(texts, prompt=p, normalize_embeddings=True,
                              convert_to_tensor=True).float().cpu()
                    for texts, p in ((CHUNKS[:16], ''),
                                     (QUESTIONS[:4], prompt))])
cos = (ours * theirs).sum(-1).min().item()
q16, d16 = ROWS['Qwen3-Embedding + instruction'][0][:16], X5[[
    CIDX[min(g)] for g in GOLD[:16]]]
gap = abs(MultipleNegativesRankingLoss(st, scale=1 / TAU)
          .compute_loss_from_embeddings([q16, d16], None).item()
          - d2l.info_nce_loss(q16, d16, TAU)[0].item())
print(f'smallest cosine, sentence-transformers against embed_texts: '
      f'{cos:.4f}; |MNRL - InfoNCE| = {gap:.1e}')
assert cos >= (0.99 if GPU else 0.999) and gap < 1e-5
del st, emb
d2l.lm_free()
```

<!-- Placeholder (prose): the pitfalls (pooling and prefix mismatch, unnormalized dot products, duplicate queries, top-rank hard negatives, truncating a non-Matryoshka model, Hamming-only search, silent truncation at max_len, contamination); what released embedders trained on far more pairs add; hand-off to :numref:`sec_lm_retrieval` (the index that serves the vectors); Summary; Exercises. -->
