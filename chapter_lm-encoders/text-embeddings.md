# Text Embeddings from Contrastive Training
:label:`sec_lm_text_embeddings`

In :numref:`sec_lm_bert` we fine-tuned an encoder one task at a time: a
small head on top of its token vectors, trained on thousands of labeled
examples, decided whether one sentence entails another. Search asks for
something else. Suppose a reader of this book wants to know how the book's
old `load_array` helper decides whether to shuffle its data. Somewhere among
the thousands of passages of the book is the paragraph that says so, and no
label set describes that question, or the next one. What we want is one
vector per text, computed once for every passage and once for every
question, such that the passage that answers a question lies closest to it.

ModernBERT, the encoder of :numref:`sec_lm_bert`, returns one vector of 768
numbers for every token. It was pretrained to fill in hidden tokens, and
nothing in that objective asks for a vector of a whole text. The obvious way
to get one anyway is to average the token vectors. Make a guess before
reading on: if we average the vectors of every passage of the book and of a
question about it, how often will the ten nearest passages include one that
answers the question?

The answer will be discouraging, and the repair has a history.
:citet:`Reimers.Gurevych.2019` found that averaged BERT vectors judged the similarity
of two sentences worse than averaged GloVe word vectors
(:numref:`sec_lm_word_embeddings`), and that training the encoder on pairs
of related sentences fixed this. Their successors settled on a contrastive
loss: show the encoder a question, its answer and a batch of other
passages, and ask it to pick out the answer. We first find out why the
averaged vectors fail, then work out what arrangement of vectors the
contrastive loss asks for, and finally train an embedder on 20,000 pairs
that have nothing to do with this book and set it against a released
embedder trained on far more.

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

Besides the raw encoder, two models appear, each fixed to one revision:
Qwen3-Embedding-0.6B, a released embedder, and Qwen3-Reranker-0.6B, which
reads a question and a passage together. Training this section's models
takes about 13 minutes on one RTX 4090; a normal build downloads the
checkpoints and their training records instead. Without a GPU the notebook
searches only the passages of the documents the questions ask about, an
easier problem; the text quotes the GPU run. This cell and the two after it
are setup and can be skimmed: names such as the Matryoshka prefixes and the
audit buckets are explained where they are used.

```{.python .input #text-embeddings-config}
%%tab pytorch
ENC = 'answerdotai/ModernBERT-base'  # the raw encoder of the last section
EMB_QWEN, RERANKER = 'Qwen/Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Reranker-0.6B'
GPU = torch.cuda.is_available()
SMALL = not GPU  # the CPU path: the chunks of the dev documents only
DTYPE = torch.bfloat16 if GPU else torch.float32
ARTIFACTS = {  # name: (stored key, the file a regeneration writes)
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
CARD_TASK = ('Given a web search query, retrieve relevant passages that '
             'answer the query')  # the model card's default task
CONFIG = dict(  # what the results are made with; reading them checks it
    model=ENC, revision=d2l.HF_ENCODER_REVISIONS[ENC], corpus=d2l.R3_VERSION,
    pairs=N_PAIRS, held=N_HELD, cands=N_CAND, batch=B, tau=TAU, lrs=LRS,
    lr_seed=LR_SEED, seeds=SEEDS,
    mrl_dims=MRL_DIMS, hard_ranks=HARD_RANKS, audit_n=AUDIT_N,
    buckets=BUCKETS, max_len=MAX_LEN, prefix=PREFIX, pooling='mean',
    reranker=RERANKER, reranker_revision=d2l.HF_ENCODER_REVISIONS[RERANKER])

def result_path(name):
    """The file this run regenerates for name, else the stored one."""
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

```{.python .input #text-embeddings-revisions}
%%tab pytorch
#@save
d2l.HF_ENCODER_REVISIONS['Qwen/Qwen3-Embedding-0.6B'] = (
    '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
d2l.HF_ENCODER_REVISIONS['Qwen/Qwen3-Reranker-0.6B'] = (
    'e61197ed45024b0ed8a2d74b80b4d909f1255473')
```

```{.python .input #text-embeddings-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['nq-pairs'] = (
    'https://huggingface.co/datasets/sentence-transformers/'
    'natural-questions/resolve/f9e894e1081e206e577b4eaa9ee6de2b06ae6f17/'
    'pair/train-00000-of-00001.parquet',
    '545c1495f10231652b59e67f7ab02e6d21252fd0')
d2l.DATA_HUB['lm-enc-emb-results-v1'] = (  # D2L_RERUN=emb-train, RTX 4090
    d2l.LM_DATA_URL + 'lm-enc-emb-results-v1.json',
    'd0bd19be14746bdb7216a190a20a8abaca4145d5')
d2l.DATA_HUB['lm-enc-modernbert-embed-v1'] = (  # contrastive, seed 0, bf16
    d2l.LM_DATA_URL + 'lm-enc-modernbert-embed-v1.zip',
    'f0de3203bde6008307a9782b45733085b02f89e2')
d2l.DATA_HUB['lm-enc-modernbert-embed-mrl-v1'] = (  # Matryoshka, bf16
    d2l.LM_DATA_URL + 'lm-enc-modernbert-embed-mrl-v1.zip',
    '17b1215b6d2e2ba5d8e7ef78621afe0894718095')
```

## Pooled Encoder States as Text Vectors
:label:`subsec_lm_emb_pooling`

Before any training we need a rule that turns the token vectors of a text
into one vector, and a way to measure how well such vectors retrieve.

### Pooling and Normalization

Three ways of turning token vectors into one are common. *CLS pooling* takes
the vector at the first position, where BERT-style tokenizers place a special
token `[CLS]`; BERT trained that position with a sentence-pair task.
*Mean pooling* averages the vectors of all real tokens. *Last-token
pooling* takes the vector of the final token, the natural choice for a
decoder, whose causal attention lets only the last position see the whole
text. `pool_hidden` finds the real tokens through the attention mask, so it
works whether a batch is padded on the right or on the left.

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

Every pooled vector is then scaled to unit length, so that the dot product
of two vectors is the cosine of their angle. Without this, a passage whose
vector happens to be long ranks high for every question. `embed_texts`
encodes texts in batches of similar length and returns the vectors in input
order. A text longer than `max_len` tokens is cut, and the function prints
how many were cut: a passage whose answer lies past the cut can no longer
be found, and nothing later would notice.

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

### Raw Vectors on the Book

Our collection is this book itself, without the Language Models part, cut
along its section boundaries into passages of at most 512 tokens of the
Qwen3 tokenizer. :numref:`sec_lm_retrieval` describes the cutting, and it
builds the word-counting ranker, BM25, that serves us as a floor.
The questions come from the book's question set, which is split by document
into train, dev and test like any benchmark and also holds questions the
book cannot answer. We use the answerable dev questions. Language models
drafted and checked them; no person did (:numref:`sec_lm_rag` says how),
and each was drafted from one stretch of text, its *gold span*. A passage
counts as *relevant* to a question if at least half of the gold span lies
in it, or at least half of the passage lies in the gold span; another span
judged to answer the question counts the same way. Questions about one
document are not independent, so our intervals resample documents, as in
:numref:`sec_lm_eval_suites`.

Our main measure is *Recall@k*. A question scores 1 if at least one
relevant passage is among the first $k$ retrieved and 0 otherwise, so the
average over questions is the share of questions whose answer reaches a
reader of the top $k$. Two further measures look at the order. *MRR@10*
scores a question by the reciprocal of the rank of its first relevant
passage, and 0 if none is among the first ten. *nDCG@10* lets a relevant
passage at rank $r$ count $1/\log_2(r + 1)$. It divides the sum over the
first ten ranks by its value for the best possible order, so it is 1 when
the relevant passages come first. Take a question whose only relevant
passage is ranked third: it scores Recall@1 $= 0$, Recall@10 $= 1$,
MRR@10 $= 1/3$ and nDCG@10 $= 1/\log_2 4 = 0.5$. Recall@10 asks whether the
answer arrives at all, the other two whether it arrives early. That matters
as soon as whoever reads the results stops after the first few.

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

The next cell loads the corpus and the questions and ranks the passages
with BM25, which counts the question's words in each passage, weights rare
words up and involves no training.

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

The book has 5,324 passages, and the 220 questions ask about 32 of its
documents. BM25 finds a relevant passage among its first ten for 96% of the
questions. That floor is high, and the reason lies in our questions more
than in BM25. A language model drafted each question while reading the
passage that answers it, and it borrowed the passage's rare words:
`load_array`, "unified memory". BM25 rewards exactly such words. A user who
has never seen the passage tends to ask in other words and would give BM25
less to hold on to. This sets the scale for the rest of the chapter: on
these questions any method has little room above counting words, and a
technique that gains nothing here may simply have had nowhere to go.

Now the raw encoder, with CLS and mean pooling. Each question is matched
with the ten passages of highest cosine. For the first three questions we
print the heading of the best passage and of the relevant one; the second
is our `load_array` question.

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
top1 = lambda name: len({r[0] for r in RANKED[name]})  # distinct passages
print(f'distinct passages at rank 1 for the {len(QUESTIONS)} questions: '
      f'raw mean {top1("raw mean")}, BM25 {top1("BM25")}')
for i in range(3):  # the first three questions, top 1 against gold
    d2l.print_wrapped(f'\n{QUESTIONS[i]}\n  top 1: '
          f'{CORPUS.header(CORPUS[RANKED["raw mean"][i][0]])}\n  gold:  '
          f'{CORPUS.header(CORPUS[min(GOLD[i])])}', width=96)
del encoder
d2l.lm_free()
```

So the answer to the opening guess is 6%: mean pooling puts a relevant
passage among the first ten for 6% of the questions, against BM25's 96%.
For scale, ten passages drawn at random would contain a relevant one for
about $10 \times 1.35 / 5{,}324 \approx 0.25\%$ of the questions. The
average is about 25 times better than chance and still of little use. Our
`load_array` question lands on the book's page of utility functions, one
heading above the passage on training utilities that answers it. The
question on saturating attention lands on the introduction of the attention
chapter, not on the subsection about score variance and the $1/\sqrt{d}$
factor. The question about unified-memory machines, which the chapter on
tools answers, lands on the opening of a section on attention. All three
top passages open a page or a section, and the three are not exceptions.
BM25 sends the 220 questions to 210 different first passages; the averaged
vectors send them to only 73, so a few broad passages collect the first
rank of many questions that have little in common. At best the vectors know
roughly where in the book a text lives; at worst, as for the unified-memory
question, not even that. They know little about what it says. No passage
reached the cap of 1,024 ModernBERT tokens at which we encode passages (the
model accepts 8,192), so truncation is not the cause.

### Anisotropy

A first clue is how similar the raw vectors are to each other. Call the mean
cosine between the vectors of distinct texts the *anisotropy* of a set of
vectors: near 0 when their directions show no preference, 1 when all point
the same way. For unit vectors $\mathbf{z}_1, \ldots, \mathbf{z}_n$, expand
the squared length of their sum:

$$
\Big\|\sum_{i=1}^n \mathbf{z}_i\Big\|^2 = \sum_{i=1}^n \|\mathbf{z}_i\|^2 + \sum_{i \neq j} \mathbf{z}_i^\top \mathbf{z}_j = n + n(n-1)\,\bar c,
$$

where $\bar c$ is the mean cosine over the $n(n-1)$ ordered pairs. Solving
for $\bar c$, with $\bar{\mathbf{z}}$ the mean vector,

$$
\bar c = \frac{n \|\bar{\mathbf{z}}\|^2 - 1}{n - 1} \approx \|\bar{\mathbf{z}}\|^2.
$$
:eqlabel:`eq_lm_anisotropy`

Thus the mean cosine is the squared length of the mean vector, up to a term
of order $1/n$. Vectors that point in many directions cancel in their mean;
vectors that share a direction have a mean almost as long as each of them.
If the vectors do share a direction, there is an obvious repair: subtract
the mean passage vector from questions and passages alike before
normalizing. The opening cell already computed this *centered* variant, so
that we can measure it alongside the others. The next cell also defines
`align_uniform`, for two quantities we meet in
:numref:`subsec_lm_emb_align_uniform`.

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

We measure anisotropy on 2,000 random passages for CLS pooling, mean
pooling and centered mean pooling, and check :eqref:`eq_lm_anisotropy` in
float64.

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
      f'the sum of the vectors {identity:.6f}\n  squared length of the mean '
      f'vector {(Z.mean(0) ** 2).sum():.6f}')
assert abs(exact - identity) < 1e-9
```

The table settles CLS pooling first: not one of the 220 questions finds a
relevant passage, no better than chance. The masked-token loss
scores only the positions it hid, and the corruption rule of
:numref:`sec_lm_bert` never hides a special token such as `[CLS]`.
ModernBERT has no sentence-pair task either. Nothing in its pretraining
ever asked the final vector at the first position to describe the text.

Mean pooling fails for a reason that is less obvious. Raw passage vectors
share one direction: the mean cosine between two unrelated passages is 0.95
with CLS pooling and 0.94 with mean pooling (the formula reproduces it to
six digits). The mean vector therefore has length about
$\sqrt{0.94} \approx 0.97$, and every passage vector lies in a narrow cone
around it. :citet:`Ethayarajh.2019` found a similar cone in the token
vectors of the upper layers of BERT and GPT-2, and
:citet:`Li.Zhou.He.ea.2020` traced its shape to word frequency: in the raw
space, how common the words of a text are moves its vector more than what
the text means. A cone this narrow is a tempting explanation of the
failure. Centering puts it to the test.

Centering removes the cone: anisotropy drops to 0.001, and Recall@10 stays
at 0.05. So the shared direction is not what hurts retrieval, however
striking it looks. What remains after centering is still an average of token
states trained to predict hidden tokens, and the opening showed what such an
average knows: roughly where in the book a text lives. Nothing in
pretraining asked a question to lie near the passage that answers it. That
relation has to be taught, with pairs of texts that belong together. The
shared direction will return, as something the training has to shrink.

## Contrastive Training
:label:`subsec_lm_emb_contrastive`

Suppose we have pairs of texts that belong together, a question and the
passage that answers it. A question's vector should then be closer to its
own passage than to any other. Take a batch of $B$ pairs, encode the
questions as unit vectors $\mathbf{q}_1, \ldots, \mathbf{q}_B$ and the
passages as unit vectors $\mathbf{d}_1, \ldots, \mathbf{d}_B$, with
$\mathbf{d}_i$ the passage of question $i$, and score every question against
every passage by the cosine $c_{ij} = \mathbf{q}_i^\top \mathbf{d}_j$. A
softmax over the $B$ passages is asked to point at the right one:

$$
\ell = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(c_{ii}/\tau)}{\sum_{j=1}^{B} \exp(c_{ij}/\tau)}.
$$
:eqlabel:`eq_lm_infonce`

There is nothing new in this loss: it is a cross-entropy whose classes are
the passages of the batch. The other $B - 1$ passages are the *negatives* of
question $i$, and they cost nothing: their vectors were computed anyway, for
their own questions. The *temperature* $\tau$ divides the cosines; at
$\tau = 0.05$ a cosine difference of 0.1 becomes a logit difference of 2.

The loss is InfoNCE :cite:`Oord.Li.Vinyals.2018`. The mathematics appendix
proves that it bounds the mutual information $I$ between question and
passage from below, $I \geq \log B - \ell$, a bound that never exceeds
$\log B$ (:numref:`sec_mdl-infonce`). With 64 candidates the loss can
certify at most $\log 64 \approx 4.16$ nats, however good the model; a
larger batch raises the ceiling.

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

Which negatives does the loss work on? Write $p_{ij}$ for the softmax
weight of passage $j$ for question $i$, and $\delta_{ij}$ for 1 if $i = j$
and 0 otherwise. Differentiating :eqref:`eq_lm_infonce` with respect to one
cosine gives

$$
\frac{\partial \ell}{\partial c_{ij}} = \frac{p_{ij} - \delta_{ij}}{\tau B}, \qquad p_{ij} = \frac{\exp(c_{ij}/\tau)}{\sum_{k=1}^{B} \exp(c_{ik}/\tau)},
$$
:eqlabel:`eq_lm_infonce_grad`

the softmax-minus-target of every cross-entropy, scaled by $1/(\tau B)$. A
step raises $c_{ii}$ with weight $1 - p_{ii}$ and lowers each $c_{ij}$ with
weight $p_{ij}$; the weights of a row sum to one, so the push on the
negatives adds up to the pull on the positive. The push is far from even:
two negatives whose cosines differ by $\Delta$ receive weights in the ratio
$e^{\Delta/\tau}$. The cell checks the gradient against automatic
differentiation and prints that ratio.

```{.python .input #text-embeddings-infonce-grad}
%%tab pytorch
gen = torch.Generator().manual_seed(0)
q, d = (F.normalize(torch.randn(32, 16, generator=gen, dtype=torch.float64),
                    dim=-1).requires_grad_() for _ in range(2))
loss, _ = d2l.info_nce_loss(q, d, TAU)
loss.backward()
P = torch.softmax(q @ d.T / TAU, dim=1).detach()
G = (P - torch.eye(len(q), dtype=P.dtype)) / (TAU * len(q))  # dL/dc_ij
err = max((q.grad - G @ d.detach()).abs().max().item(),
          (d.grad - G.T @ q.detach()).abs().max().item())
pull_push = (G.sum(1)).abs().max().item()  # 1 - p_ii = sum_{j != i} p_ij
print(f'largest gradient error {err:.1e}; largest |pull - push| per query '
      f'{pull_push:.1e}')
assert err < 1e-10 and pull_push < 1e-12
print(pd.DataFrame({'score gap': [0.1, 0.3], f'weight ratio at tau={TAU}':
                    np.exp(np.array([0.1, 0.3]) / TAU)}).round(1))
```

Automatic differentiation agrees with the formula to $10^{-16}$. At
$\tau = 0.05$, a negative 0.1 closer to the question than another receives
7.4 times its push, and one 0.3 closer receives 403 times. A low temperature
puts nearly all of its gradient on the few negatives that look most like
the answer, the *hard negatives*. That helps when they are true negatives
and hurts when they are not. The same weights say when a batch stops
teaching: once the right passage takes almost all of the softmax,
$1 - p_{ii}$ and every $p_{ij}$ are near zero, and random negatives
contribute almost nothing.

### Alignment and Uniformity
:label:`subsec_lm_emb_align_uniform`

What arrangement of vectors does the loss want? The gradient answers this
for one batch at a time, which is too local to picture. :citet:`Wang.Isola.2020`
answered it for the whole data by letting the batch grow. This is the
hardest step of the section, and everything we later measure about the
geometry of trained embedders refers back to it. Subtract $\log B$ from the
loss of one question and move the $B$ inside the logarithm:

$$
\ell_i - \log B = -\frac{c_{ii}}{\tau} + \log \frac{1}{B} \sum_{j=1}^{B} e^{c_{ij}/\tau}.
$$

As $B$ grows, with the passages drawn from the data, the average tends to
$\mathbb{E}_{\mathbf{d}}\, e^{\mathbf{q}_i^\top \mathbf{d}/\tau}$; the
positive is one term of $B$, and its share vanishes. For unit vectors,
$\mathbf{q}^\top \mathbf{d} = 1 - \tfrac{1}{2}\|\mathbf{q} - \mathbf{d}\|^2$.
Substituting this in both terms, the constants $-1/\tau$ and $+1/\tau$
cancel, and averaging over questions gives

$$
\lim_{B \to \infty} \big(\ell - \log B\big) = \frac{1}{2\tau}\, \mathbb{E}\big[\|\mathbf{q} - \mathbf{d}^+\|^2\big] + \mathbb{E}_{\mathbf{q}} \Big[\log \mathbb{E}_{\mathbf{d}}\, e^{-\|\mathbf{q} - \mathbf{d}\|^2/(2\tau)}\Big],
$$
:eqlabel:`eq_lm_align_uniform`

where $\mathbf{d}^+$ is the passage of question $\mathbf{q}$. The first term,
the *alignment*, is the mean squared distance within positive pairs. The
second, the *uniformity*, is the logarithm of a Gaussian kernel averaged
over all passages, small when the passages keep their distance from every
question. In the large-batch limit the loss has only these two terms: one
pulls pairs together, the other pushes all vectors apart.

The loss's uniformity term compares questions with passages. To compare
embedders by their passages alone, Wang and Isola measure the same kernel
on two independent passages $\mathbf{z}$ and $\mathbf{z}'$,

$$
\mathcal{U}_t = \log \mathbb{E}\, e^{-t\|\mathbf{z} - \mathbf{z}'\|^2},
$$

which also falls as the vectors spread out. The loss's own kernel has
$t = 1/(2\tau) = 10$; we use their $t = 2$, so that our numbers compare
with theirs. `align_uniform` computes the alignment and $\mathcal{U}_2$,
over distinct pairs of passages.

Let us start with the smallest case that shows the tension between the
two terms. Place four unit vectors on a circle, points 1 and 2 forming one
positive pair and points 3 and 4 the other. Each point acts as a question
whose passage is its partner, so the batch has four rows, and each question
also meets itself among the passages, as the passage of its partner. A
square, with the points at 0°, 90°, 180° and 270°, spreads them as far as
possible but puts each point a quarter turn from its partner. Collapsing
all four onto one point aligns them perfectly. A third configuration puts
each pair on one spot and the two spots on opposite sides. Which does
InfoNCE prefer? The four points are the whole data, so this batch computes
the right side of :eqref:`eq_lm_align_uniform` with no approximation.

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
        row[f'tau {tau}'] = loss.item() - math.log(4)
    rows.append(row)
print('InfoNCE - log 4 in the last two columns, at two temperatures')
print(pd.DataFrame(rows).round(3).to_string(index=False))
```

The square has the best uniformity, $\mathcal{U}_2 = -4.4$, and the worst
alignment, 2.0, and InfoNCE punishes it hard. Each question finds itself
among the negatives at cosine 1 and its partner at cosine 0. At
$\tau = 0.05$ that is a logit gap of 20, and a loss 18.6 nats above
$\log 4$. The collapse gives perfect alignment and the loss of a blind
guess, $\log 4$. Two coincident pairs on opposite sides win at both
temperatures. Each question then finds two passages at its own spot, its
partner and itself, so the softmax can at best split evenly between them: a
loss of about $\log 2$, which is $\log 2$ below $\log 4$. The temperature
sets how many nats a bad arrangement costs: at $\tau = 0.5$ the square
costs 0.87 nats instead of 18.6.

The collapse is the extreme of anisotropy, and the loss rejects it. More
generally, a shared direction caps how far uniformity can fall, and the
proof takes one line.

**Proposition.** *For unit vectors drawn from one distribution, with
anisotropy $a$ (the mean cosine between two independent draws), the
uniformity $\mathcal{U}_t$ is at least $-2t(1 - a)$ for every $t > 0$.*

**Proof.** The logarithm is concave, so $\log \mathbb{E}\, e^{X} \geq
\mathbb{E}\, X$ by Jensen's inequality. Take $X = -t\|\mathbf{z} -
\mathbf{z}'\|^2 = -t(2 - 2\mathbf{z}^\top \mathbf{z}')$, whose expectation
is $-2t(1 - a)$. $\blacksquare$

The uniformity $\mathcal{U}_t$ is never above 0, its worst value, and while
$a$ is near 1 it cannot go far below 0 either. The same argument applies to
the term of :eqref:`eq_lm_align_uniform`, with the mean cosine between
questions and passages in place of $a$. Training that lowers the term
must lower that mean cosine, and since alignment holds each question near
its passage, this means shortening the mean vector: contrastive training
reduces anisotropy because it cannot do its job otherwise. The converse fails, as centering showed; that is what
the alignment term is for.

### Training on Question-Answer Pairs
:label:`subsec_lm_emb_train`

The book's questions are what we test on, so we train on pairs from
elsewhere and measure transfer. `train_embedder` minimizes
:eqref:`eq_lm_infonce` over batches of question-passage pairs with
mean-pooled vectors. Questions get the prefix `query: ` and passages
`passage: `, the convention of the E5 embedders :cite:`Wang.Yang.Huang.ea.2022`, which
lets one encoder treat the two sides differently. AdamW with warmup and
linear decay, clipping at norm 1 and bf16 complete the recipe. None of this
is particular to embedders, and the details can be read off the code.

Duplicates are the exception, and they need care. If two pairs in a batch
share a passage, that passage is the positive of one question and a negative
of the other, and the loss pushes the second question away from its own
answer. Such a passage is a *false negative*. `no_duplicate_batches`
prevents them within the training pairs: a pair that would repeat a question
or a passage already in the batch waits for a later batch.

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

The pairs come from Natural Questions :cite:`Kwiatkowski.Palomaki.Redfield.ea.2019`: questions
that people typed into a search engine, each with a Wikipedia passage that
answers it. We train on 20,000 pairs chosen by one seeded permutation. To
choose the learning rate we hold out 300 later pairs whose question and
passage occur nowhere in training, so that the rate that best memorizes
training passages gains nothing. For each held-out question we rank 5,000
unseen passages and score MRR@10, which here is the reciprocal of the rank
of the question's own passage.

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
print(f'{len(nq):,} pairs, {nq.answer.nunique():,} distinct passages\n'
      f'training on {len(PAIRS):,} pairs with {len(shared):,} distinct '
      f'passages ({sum(n for n in shared.values() if n > 1):,} pairs share '
      f'theirs)\n{len(HELD)} unseen pairs held out for the learning rate, '
      f'ranked among {len(CANDS):,} unseen passages')
```

Of the 20,000 training pairs, 3,236 share their passage with another pair,
the duplicates the sampler keeps apart. The next cell holds helpers for
encoding, ranking, scoring, mining and checkpoints.

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

The training run first tries three learning rates, all with seed 3, which
no reported model uses, so that the reported models are not winners of a
selection, and scores them on the held-out pairs. It then trains the chosen
rate with seeds 0, 1 and 2, which shows how much Recall@10 moves from
training noise alone, and saves the seed-0 model.

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

### Hard Negatives and False Negatives
:label:`subsec_lm_emb_hardneg`

A random passage from Wikipedia is rarely about the subject of a given
question, and by :eqref:`eq_lm_infonce_grad` such easy negatives receive
almost no gradient. The hard negatives, passages that the current model
ranks high for a question although they are not its passage, are the ones
the gradient works on. Why not supply them on purpose? Finding them by
searching with the model itself is called *mining*. We mine
with the seed-0 model: for each training question it ranks the distinct
training passages, its own left out, and one hard negative is drawn
uniformly from ranks 11 to 50.

Why not from the top? A passage ranked first is often a correct answer that
happens not to be the labeled one, and training against it pushes the
question away from an answer. We measure how often. For 500 training
questions we take one mined passage from each of four rank buckets and ask
Qwen3-Reranker-0.6B whether the passage answers the question. The reranker
is a cross-encoder in the sense of :numref:`sec_lm_bert`: it reads
question and passage together and answers yes or no. A probability of yes
above 0.5 marks a passage as probably relevant. These labels come from a
model, not from people.

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

Two variants follow at the chosen rate, seed 0. One adds the mined negative
as one more candidate per question, so that every question sees 128
passages. The other, the *Matryoshka* model, trains for vectors that can be
shortened, with a loss defined in :numref:`subsec_lm_emb_compress`, and is
saved.

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

The next cell reads what the training run recorded: first the
learning-rate check and a summary of each run, then the audit, with
intervals over the held-out and the audited questions.

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

The learning rate $10^{-4}$, the largest we tried, won with MRR@10 of 0.87
on the held-out pairs. $3 \times 10^{-5}$ came within noise of it and
$10^{-5}$ did not, so the optimum is flat at the top of the grid. One epoch,
313 steps, takes about a minute on one RTX 4090. The loss falls from about
3.2 nats over the first ten steps to 0.05 over the last ten. Is the model
memorizing? Memorization cannot explain it: each pair is seen once, and the
held-out pairs, whose texts never occur in training, score 0.87. The model
simply picks the right passage among 64 almost every time, so by the
gradient weights above, random negatives had little left to teach by the
end of the epoch.

The audit shows why the top ranks are left alone. Among mined passages at
ranks 1 to 5, the reranker calls 30% probably relevant: by its judgment
nearly a third of the "negatives" there are answers. The share falls with
every bucket, to 10%, 2% and under 1%, so mining from ranks 11 to 50 keeps
false negatives, by the reranker's labels, to one or two percent.

So far no model has seen a question about the book. Whether the trained
encoder retrieves anything there, and whether the mined negative helps, is
the subject of the next part, where both meet the book's questions beside a
released embedder.

## Trained and Released Embedders on the Book
:label:`subsec_lm_emb_llm`

Our encoder has seen 20,000 pairs about Wikipedia, once. Before it meets
the book's questions, let us look at the competition it will face there.

### Released Embedders

Released embedders are trained on far more, in stages. E5
:cite:`Wang.Yang.Huang.ea.2022` first trains on weak
pairs gathered from the web, such as titles with their articles, then on
smaller labeled sets with mined hard negatives. Qwen3-Embedding
:cite:`Zhang.Li.Long.ea.2025` had a larger Qwen3 model write its weak pairs, then
trained on labeled and filtered synthetic pairs with hard negatives, and
finally merged several of its checkpoints. RocketQA :cite:`Qu.Ding.Liu.ea.2021`
introduced the step our audit imitates: a cross-encoder filters the mined
negatives, so that false negatives are dropped rather than pushed away.

Qwen3-Embedding-0.6B starts from a pretrained decoder of the Qwen3 family
and keeps its causal attention. Three conventions follow, and a model used
against any of them loads and runs without complaint. The wrong pooling or
padding reads the state of the wrong token. A missing instruction changes
what the query vector encodes, in a direction that has to be measured.

- **Last-token pooling.** Under causal attention only the last position has
  seen the whole text.
- **Left padding.** Short texts are padded on the left, so the last position
  of every row holds a real token.
- **An instruction on the query side.** A question enters as
  `Instruct: {task}\nQuery:{query}`. Passages are encoded without it, so one
  passage index serves many tasks. The task is free text. The model card's
  default is a task about web search queries; for the book we wrote
  `Given a question about the book, retrieve passages that answer it`.

Last-token pooling is not the only way out of causal attention. Some
recipes remove the causal mask instead and fine-tune the decoder with
attention in both directions, so that every position sees the whole text
:cite:`BehnamGhader.Adlakha.Mosbach.ea.2024,Lee.Roy.Xu.ea.2025`;
:numref:`sec_lm_encoders_fr` says what each recipe changes.

### Comparison on the Book

Now all models answer the same 220 questions. How far do 20,000 pairs about
Wikipedia carry toward a book about deep learning? The cell also checks
that the two saved checkpoints reproduce the rankings recorded at training
time.

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
for name, fmt, task in (
        ('Qwen3-Embedding', '{query}', None),
        ('Qwen3-Embedding + instruction', INSTRUCT, d2l.BOOK_TASK),
        ('Qwen3-Embedding + card task', INSTRUCT, CARD_TASK)):
    ROWS[name] = (d2l.embed_texts(emb, emb_tok, [fmt.format(
        task=task, query=q) for q in QUESTIONS], QWEN['pooling'],
                                  max_len=MAX_LEN['query']), X5)
RANKED |= {name: top10(Q @ X.T) for name, (Q, X) in ROWS.items()}
RANKED |= {name: stored(EMB['runs'][name]) for name in (
    'contrastive seed 1', 'contrastive seed 2',
    'contrastive + hard negatives')}
METRICS |= {name: d2l.retrieval_metrics(r, GOLD, ks=(1, 5, 10))
            for name, r in RANKED.items()}
aniso = {name: f'{d2l.anisotropy(ROWS[name][1][SAMPLE]):.3f}'
         for name in ROWS}  # three decimals: the page's width
print(pd.DataFrame({name: {k: f'{m[k].mean():.3f}' for k in m
                          if k != 'ndcg@10'} | {
    'ndcg@10 [CI]': ci(m['ndcg@10']), 'anisotropy': aniso.get(name, 'n/a')}
                    for name, m in METRICS.items()}).T.to_string())
print('distinct passages at rank 1: ' + ', '.join(f'{n} {top1(n)}' for n in (
    'BM25', 'raw mean', 'contrastive seed 0', 'Qwen3-Embedding')))
for name in ('contrastive seed 0', 'Matryoshka'):  # live against stored
    reg = stored(EMB['runs'][name])
    same = np.mean([a == b for a, b in zip(RANKED[name], reg)])
    agree = np.mean(METRICS[name]['recall@10'] == d2l.retrieval_metrics(
        reg, GOLD, ks=(10,))['recall@10'])
    print(f'live {name} model against its stored ranking:\n  same top 10 '
          f'for {same:.3f} of the questions, same Recall@10 for {agree:.3f}')
    assert SMALL or agree >= 0.98, f'{name}: checkpoint and results differ'
```

Most of the way. Contrastive training turns the encoder into a retriever:
Recall@10 rises from 0.06 to 0.76 for seed 0, the three seeds span 0.73 to
0.78, and anisotropy falls from 0.94 to 0.52, as the proposition says it
must. The first ranks spread out as well: the 220 questions now land on 188
different first passages, against 73 before training. The model still
trails BM25, and by more in the order of its results than in what it finds: its nDCG@10 is about 0.52, BM25's 0.82, with
intervals far apart. The saved checkpoints reproduce the recorded
Recall@10 for at least 99.5% of the questions, although bf16 rounding lets
near-tied passages swap places.

Qwen3-Embedding finds a relevant passage among its first ten for 98% of the
questions or more, in each of the three ways we pose the query. At the first
rank it is no better than BM25: Recall@1 is 0.76 for BM25, against about
0.70 for Qwen3-Embedding without an instruction or with the model card's
task, and 0.67 with ours.

The next cell computes paired differences on the same questions, with
intervals that resample documents, and marks whether each exceeds the range
of the three seeds. A second table compares Qwen3-Embedding with BM25, and
the three forms of the query with each other, on one measure of finding and
two of order.

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
print(D.round(3).rename(columns={'CI excludes 0': 'excl. 0',
                                  'beyond seed range': 'beyond seeds'})
      .to_string(index=False))
print(f'Recall@10 of the three seeds: {np.round(seed_r10, 3)}, range '
      f'{seed_range:.3f}; best raw pooling: {best_raw}')
fmt = lambda d: f'{d["delta"]:+.3f} [{d["low"]:+.3f}, {d["high"]:+.3f}]'
print('\npaired differences, to - from, on three measures:')
print(pd.DataFrame({f'{t} - {f}'.replace('Qwen3-Embedding', 'Qwen3')
                    .replace('instruction', 'instr.'): {
    m: fmt(delta(f, t, m)) for m in ('recall@1', 'mrr@10', 'recall@10')}
    for f, t in (('BM25', 'Qwen3-Embedding'),
                 ('BM25', 'Qwen3-Embedding + instruction'),
                 ('Qwen3-Embedding', 'Qwen3-Embedding + instruction'),
                 ('Qwen3-Embedding', 'Qwen3-Embedding + card task'))}).T
      .to_string())
gain = D[(D['from'] == best_raw) & (D['to'] == 'contrastive seed 0')].iloc[0]
assert SMALL or gain.low > 0, 'no gain from contrastive training: a bug'
```

Contrastive training on 20,000 pairs that never mention the book lifts
Recall@10 by 70 points over raw mean pooling, more than fifteen times the
seed range of 4.5 points. This is the result of the section. Nothing in a
pretrained encoder says which passage a question should be near, and a
minute of contrastive training on someone else's questions puts it there.

The released embedder adds about 20 points more, and that gap bundles
several causes. Qwen3-Embedding has a larger backbone, pretrained on far
more text. It saw far more training pairs, many of them written by a
language model. And it may have read this book, which has been on the web
for years. The run cannot separate these causes, and the last one makes the
0.98 optimistic by an amount we cannot measure. ModernBERT's pretraining
text :cite:`Warner.Chaffin.Clavie.ea.2024` may have included the book too,
but the raw encoder's 0.06 shows that such exposure alone retrieves
nothing; the 0.76 measures what generic pairs transfer.

The mined hard negative raised every measure in its one run: Recall@10 by
about 3 points, with an interval that reaches zero, and the other measures
by 3 to 5. That is the direction the gradient weights predict, but one run
against a seed spread of 4.5 points does not establish it. The comparison
has a second weakness. The variant gives each question 128 candidates
instead of 64, so even a clear gain could come from more negatives rather
than harder ones. Settling the question takes several seeds of the variant
and a control with one random extra negative per question; we have not run
them.

Finally, the instruction we wrote for the book made Qwen3-Embedding worse.
Every measure of order came out lower with it: MRR@10 by about 0.03, with an
interval that excludes zero, Recall@1 by 3 points and nDCG@10 by 2.
Recall@10 moved by $-0.014$, an interval that reaches zero. This runs
against the model card, which reports gains of 1% to 5% from instructions
and recommends writing one for the task at hand.

Is the fault in instructions as such, or in ours? The format is the card's, as the library
check below confirms, and the last row of the second table answers the
question. The card's own task, a sentence about web search queries in the
same format, is level with no instruction on all three measures, every
interval containing zero. So the damage comes from our wording, not from
the instruction format. The card's task brings no gain here either; on
questions that copy their passages' rare words there may be little left for
an instruction to add. The wording of an instruction is a parameter like
any other, and it has to be chosen on questions other than those it is
evaluated on; Exercise :exref:`ex_lm_emb-instruction` does this with the
training questions. The rest of the chapter keeps our wording, at the cost
just measured.

The second table also settles the comparison with BM25. Without an
instruction, Qwen3-Embedding finds a relevant passage within ten for about
4 points more of the questions than BM25, an interval from 1 to 6 points,
and the two cannot be told apart at the first rank or by MRR@10. With our
instruction, BM25 puts a relevant passage first more often: by 9 points of
Recall@1 and by 0.065 in MRR@10, both intervals excluding zero. On questions
that repeat the rare words of their passages, counting those words remains
a strong way to pick the top passage.

### Geometry after Training

Does training change the geometry as :eqref:`eq_lm_align_uniform` predicts?
We measure alignment on all pairs of a dev question and one of its relevant
passages, and $\mathcal{U}_2$ on the 2,000 sampled passages. The plot
projects the passage vectors onto the two directions along which they
extend furthest, found by an SVD without subtracting the mean, so that a
shared direction stays visible as a long mean vector (the star).

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

Training moved pairs apart: the alignment, the mean squared distance within
pairs, grows from 0.19 for raw mean pooling to 0.60. That seems to
contradict the pull of the alignment term, until we compare it with how far
apart passages lie in general. For unit vectors
$\|\mathbf{z} - \mathbf{z}'\|^2 = 2 - 2\,\mathbf{z}^\top \mathbf{z}'$, so
two random passages lie at mean squared distance $2 - 2a$, with $a$ the
anisotropy. The table sets the two distances side by side
(:numref:`tab_lm_emb_distances`).

:Mean squared distance between a question and its relevant passage (the alignment) and between two random passages ($2 - 2a$), and their ratio.
:label:`tab_lm_emb_distances`

| model | question to its passage | two random passages | ratio |
|---|---|---|---|
| raw mean pooling | 0.19 | $2 - 2 \times 0.940 = 0.12$ | 0.6 |
| contrastive, seed 0 | 0.60 | $2 - 2 \times 0.522 \approx 0.96$ | 1.6 |
| Qwen3-Embedding + instruction | 0.61 | $2 - 2 \times 0.283 \approx 1.43$ | 2.4 |

Before training, a question lies farther from its own passage than two
random passages lie from each other. Training moved everything apart, and
unrelated passages much more than pairs: after training, two random passages
lie 1.6 times as far apart as a question and its passage, and 2.4 times for
Qwen3-Embedding. Retrieval depends on this ratio. The values of
$\mathcal{U}_2$, $-0.23$, $-1.82$ and $-2.78$, lie just above the bounds
$-4(1 - a)$ of the proposition, $-0.24$, $-1.91$ and $-2.87$: in all three
spaces the shared direction limits the spread. In the plot the raw vectors
form a thin arc around a mean vector of length 0.97, the trained ones a much
wider cap around a mean of length about $\sqrt{0.52} \approx 0.72$. Training
has shortened the mean vector without removing it.

## Smaller Vectors
:label:`subsec_lm_emb_compress`

So far each passage vector of Qwen3-Embedding has been 1,024 numbers. In
float32 that is 4 KB per passage (our stored copy uses float16, half that),
and for the book about 22 MB, which no machine notices. An index of $10^8$ web passages would need about 400 GB,
and an exact search reads all of it for every question. How much of the
0.98 survives with fewer bytes? We can keep fewer coordinates, or fewer
bits per coordinate.

### Truncation and Quantization

*Truncation* keeps the first $k$ coordinates and rescales them to unit
length. In a model trained with the plain loss the first coordinates are no
more important than the others. *Matryoshka representation learning*
:cite:`Kusupati.Bhatt.Rege.ea.2022` makes them so by averaging the loss over prefixes,

$$
\ell_{\textrm{MRL}} = \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \ell^{(k)},
$$

with $\ell^{(k)}$ the loss :eqref:`eq_lm_infonce` on the first $k$
coordinates, renormalized, and $\mathcal{K}$ a set of widths. Each prefix
must then be an embedding on its own, as each doll of a Russian nesting set
is a complete doll. Our Matryoshka model was trained with
$\mathcal{K} = \{768, 512, 256, 128, 64\}$. Qwen3-Embedding supports such
prefixes: its model card allows any width from 32 to 1,024 coordinates.

*Quantization* keeps all coordinates at lower precision. The simplest form,
int8, stores each coordinate in one byte, with one scale per coordinate
taken from the largest value in the corpus, a quarter of the float32 bytes.

Binary vectors go further and keep only the sign of each coordinate, one
bit. Two of them are compared by their *Hamming distance*, the number of
differing bits: one XOR and one population count per machine word. Why
should signs preserve similarity? For a random direction $\mathbf{r}$, the
signs of $\mathbf{r}^\top \mathbf{a}$ and $\mathbf{r}^\top \mathbf{b}$
differ with probability $\theta/\pi$, where $\theta$ is the angle between
$\mathbf{a}$ and $\mathbf{b}$ :cite:`Charikar.2002`. An embedder's coordinates
are not random directions, so the formula is not exact, but the sign
agreement still falls as the angle grows.

Hamming distance discards magnitude: a coordinate where the query is nearly
0 counts as much as its largest one. *Rescoring* repairs this on the query
side. It takes the top 100 passages by Hamming distance and reorders them by
the dot product of the float query with the passages' signs, so the
passages stay at one bit per coordinate.

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

The cell measures Recall@10 against bytes per vector for the plain
contrastive model, the Matryoshka model and Qwen3-Embedding. Binary scores
tie often; ties are put in seeded random order rather than in corpus order,
which would favor passages early in the book.

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
    for dim in (m, 512, 256, 128, 64, 32, 16):
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
labels = list(MODELS) + [f'{l}, binary, rescored' for l in MODELS]
d2l.plot([t.bytes.tolist() for t in lines + points],
         [t['recall@10'].tolist() for t in lines + points],
         'bytes per vector', 'Recall@10', xscale='log', figsize=(5, 3.5),
         legend=labels, fmts=('C0-o', 'C1-s', 'C2-^', 'C0*', 'C1*', 'C2*'))
d2l.plt.gca().legend(labels, loc='upper left', bbox_to_anchor=(1, 1));
```

For Qwen3-Embedding, int8 costs nothing we can see: Recall@10 stays at 0.98
at a quarter of the bytes. Binary vectors keep about 0.96, with or without
rescoring, at 1/32 of the bytes. Truncation costs more: 0.90 at 128 of
1,024 coordinates, 0.84 at 64, and below that recall collapses, to 0.62 at
32 coordinates, the smallest width the model card allows, and 0.24 at 16,
below it. Compare two formats of
the same size. The signs of all 1,024 coordinates fill 128 bytes, and so do
32 coordinates in float32; the first keeps 0.96, the second 0.62. At equal
bytes, many coarse coordinates rank far better than a few precise ones.
That is the fact to remember from this subsection; the formats themselves
are details to look up when needed.

Our two models lose more to truncation. From full width to 128 coordinates
the plain model falls from 0.76 to 0.69 and the Matryoshka model from 0.74
to 0.66; at 32 coordinates both are near 0.4, at 16 near 0.15. Which of the
differences between the two are real? Both are single seeds, so a
difference is read against the seed range of the full-width models as well
as against its interval.

```{.python .input #text-embeddings-compress-compare}
%%tab pytorch
pairs = [(f'Matryoshka - contrastive, {k}', R10['Matryoshka', k],
           R10['contrastive', k]) for k in (128, 64, 32, 16)]
pairs += [(f'{l}: binary + rescoring - float', R10[l, 'binary + rescoring'],
           R10[l, ROWS[n][1].shape[1]]) for l, n in MODELS.items()]
C = pd.DataFrame([dict(comparison=c, delta=r.mean, low=r.ci[0], high=r.ci[1])
                  for c, r in ((c, d2l.paired_bootstrap(a, b, cluster=CLUSTER))
                               for c, a, b in pairs)])
C['CI excludes 0'] = (C.low > 0) | (C.high < 0)
C['beyond seed range'] = [  # a tie is not; one model's two formats: no
    '' if 'binary' in c else d > seed_range + 1e-9  # training noise
    for c, d in zip(C.comparison, C.delta.abs())]
print(C.round(3).to_string(index=False))
print(f'seed range of Recall@10 at full width (a proxy for the training '
      f'noise\n  at smaller widths): {seed_range:.3f}')
```

Binary vectors with rescoring cost Qwen3-Embedding and the Matryoshka model
about 2 points, with intervals that reach zero, and the plain model about 4
points, an interval that excludes zero. These are two formats of one set of
trained weights on the same questions, so training noise plays no part, and
the paired interval alone decides. The seed range matters only when two
separately trained models are compared, as next. Even at 128 bytes per passage, Qwen3-Embedding stays far ahead of
our model at full precision.

The Matryoshka model, however, is not ahead at any width. The differences
run from $-0.03$ at 128 coordinates to $+0.02$ at 16, every interval
contains zero, and every difference lies within the seed range. This should
make us suspicious. :citet:`Kusupati.Bhatt.Rege.ea.2022` found the advantage
to grow as the prefix shrinks, and when an accepted method fails to show
its effect, the code is the first suspect. The test is to ask both models
the kind of question they were trained on: the 300 held-out pairs of
Natural Questions, each question searched among the 5,000 passages that no
training batch contained, at the same widths.

```{.python .input #text-embeddings-compress-nq}
%%tab pytorch
# In domain: Recall@10 of the 300 held-out NQ questions among the 5,000
# unseen NQ passages, both checkpoints truncated to the same widths
NQ_GOLD = [[CANDS.index(d)] for _, d in HELD]
NQ_R10 = {}
for label, key in (('contrastive', 'emb-ckpt'),
                   ('Matryoshka', 'emb-ckpt-mrl')):
    model = load_ckpt(result_path(key))
    Q = encode(model, [q for q, _ in HELD], 'nq_query', PREFIX['query'])
    X = encode(model, CANDS, 'nq_passage', PREFIX['passage'])
    del model
    d2l.lm_free()
    for k in (Q.shape[1], 128, 64, 32, 16):
        S = d2l.truncate_embeddings(Q, k) @ d2l.truncate_embeddings(X, k).T
        NQ_R10[label, k] = d2l.retrieval_metrics(
            S.topk(10).indices.tolist(), NQ_GOLD, ks=(10,))['recall@10']
rows = []
for k in (Q.shape[1], 128, 64, 32, 16):  # questions resampled: one each
    r = d2l.paired_bootstrap(NQ_R10['Matryoshka', k],
                             NQ_R10['contrastive', k])
    rows.append({'width': k, 'contrastive': NQ_R10['contrastive', k].mean(),
                 'Matryoshka': NQ_R10['Matryoshka', k].mean(),
                 'delta': r.mean, 'low': r.ci[0], 'high': r.ci[1]})
NQ_C = pd.DataFrame(rows)
NQ_C['CI excludes 0'] = (NQ_C.low > 0) | (NQ_C.high < 0)
print(f'NQ, {len(HELD)} held-out questions among {len(CANDS):,} passages: '
      'Recall@10')
print(NQ_C.round(3).to_string(index=False))
```

There the loss does what its paper reports. Down to 64 coordinates the two
models are level. At 32 coordinates the Matryoshka model keeps a Recall@10
of 0.83 against 0.76, and at 16 it keeps 0.62 against 0.51: gains of about
8 and 11 points, with intervals that exclude zero, below even the shortest
prefix it was trained on. The code is not at fault. On the book the same
comparison gave $-0.01$ and $+0.02$, with intervals that exclude gains of
that size. The loss works on the distribution it was trained on, and on
this book its advantage did not carry over.

Why not? The two settings differ in several ways, and the easiest to test
is length. Training cut every Natural Questions passage to 160 tokens,
while the book's passages, cut at 512 tokens of the Qwen3 tokenizer, are
encoded whole. Perhaps a short prefix never learned to summarize a long
text. The next cell cuts the book's passages to their first 160 tokens and
compares the two checkpoints again.

```{.python .input #text-embeddings-compress-160}
%%tab pytorch
R160 = {}  # the book's questions; its passages cut to 160 tokens
for label, key in (('contrastive', 'emb-ckpt'),
                   ('Matryoshka', 'emb-ckpt-mrl')):
    model = load_ckpt(result_path(key))
    Q = encode(model, QUESTIONS, 'query', PREFIX['query'])
    X = encode(model, CHUNKS, 'nq_passage', PREFIX['passage'])
    del model
    d2l.lm_free()
    for k in (Q.shape[1], 128, 64, 32, 16):
        R160[label, k] = recall10(d2l.truncate_embeddings(Q, k) @
                                  d2l.truncate_embeddings(X, k).T)
rows = []
for k in (Q.shape[1], 128, 64, 32, 16):
    r = d2l.paired_bootstrap(R160['Matryoshka', k], R160['contrastive', k],
                             cluster=CLUSTER)
    rows.append({'width': k, 'contrastive': R160['contrastive', k].mean(),
                 'Matryoshka': R160['Matryoshka', k].mean(), 'delta': r.mean,
                 'low': r.ci[0], 'high': r.ci[1]})
C160 = pd.DataFrame(rows)
C160['CI excludes 0'] = (C160.low > 0) | (C160.high < 0)
print(f'the book, passages cut to {MAX_LEN["nq_passage"]} tokens: Recall@10')
print(C160.round(3).to_string(index=False))
```

The cut reaches 4,824 of the 5,324 passages and costs both models recall
at full width, since some answers lie past the first 160 tokens. The gap we
were looking for does not appear. At 32 and 16 coordinates the Matryoshka
model is ahead by about 2 points, with intervals that contain zero and stop
short of the 8 and 11 points it gained on Natural Questions. Length is ruled
out. What remains is the material itself: questions about deep learning
that share rare words with their passages, against the general-knowledge
questions of Natural Questions. Which part of that difference matters we do
not know; Exercise :exref:`ex_lm_emb-matryoshka` proposes the next test.

In practice, then: down to 64 of its 768 coordinates, a plain contrastive
model truncates as well as a Matryoshka one in both settings. Below that, Matryoshka training is what keeps recall
on data like its training data, and whether it helps on other data has to
be measured on one's own questions.

## Using a Released Embedder
:label:`subsec_lm_emb_practice`

### A Library Check

In practice a released embedder is loaded through a library such as
sentence-transformers, which reads its pooling, normalization and prompts
from files in its repository; the padding side we pass ourselves, as the
model card asks. Does it do what we
derived? The cell checks the repository's query prompt and pooling, encodes
16 passages and 4 questions both ways, and compares the library's
contrastive loss, `MultipleNegativesRankingLoss` at scale $1/\tau$, with
`info_nce_loss`.

```{.python .input #text-embeddings-production}
%%tab pytorch
st = SentenceTransformer(
    EMB_QWEN, revision=d2l.HF_ENCODER_REVISIONS[EMB_QWEN],
    device='cuda' if GPU else 'cpu', model_kwargs={'dtype': DTYPE},
    processor_kwargs={'padding_side': QWEN['padding_side']})
assert st.prompts['query'] == INSTRUCT.format(task=CARD_TASK, query='')
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

The two encodings agree to a cosine of at least 0.998, the difference bf16
arithmetic leaves, and the library's loss equals :eqref:`eq_lm_infonce` to
$4 \times 10^{-6}$. The library reads the conventions from the repository,
and the method underneath is the one derived above. What the library cannot
check is whether the task we wrote into the instruction suits the model;
that, as we saw, has to be measured.

### Choosing an Embedder

Released embedders are ranked on MTEB, the Massive Text Embedding Benchmark
:cite:`Muennighoff.Tazi.Magne.ea.2023`, which averages scores over many tasks and
languages. An average over tasks says little about one domain: a model can
rise by clustering news well, and the training data of some models overlaps
the benchmark's. The reliable way to choose is the one this section
followed: a few hundred questions about your own documents, a word-counting
floor, and paired differences with intervals that resample documents. If
you can, have the questions written by people who have not just read the
answer; otherwise, as here, the floor comes out high. On the book this
procedure does not crown a single winner. The released embedder finds
slightly more answers, and BM25 puts the answer first at least as often. The
case for the embedder rests on questions phrased in other words than their
passages, which this set hardly contains. So we keep both: a released
embedder, in int8 or as binary vectors with rescoring when memory is short,
with BM25 beside it. Training on pairs from the domain is the next step
only when such a set shows a gap, and then the training questions must come
from other documents than the evaluation questions.

Contrastive training makes vectors that a scan can compare. It does not make
the scan cheap, and a question that turns on an exact identifier can still
be missed by every vector of this section. :numref:`sec_lm_retrieval` builds
the index that serves these vectors and the word index that catches what
they miss.

## Summary

Search needs one vector per text, and averaging the token vectors of a
pretrained encoder does not supply a useful one. On the book's questions
the average found a relevant passage among the first ten for 6% of them,
where counting words found one for 96%. The averaged vectors share one
direction, but removing it does not help, because pretraining never asked a
question to lie near its answer. Contrastive training asks exactly that, by
making each question pick its passage out of a batch; in the large-batch
limit the loss pulls pairs together and pushes all vectors apart, which
forces the shared direction to shrink. A minute of such training on generic
pairs lifted Recall@10 by 70 points, and a released embedder went further
and kept nearly all of its recall at one bit per coordinate. Mined
negatives must come from below the top ranks, where a reranker judged
nearly a third of them to be answers. Every other choice, from the wording
of an instruction to the width of a vector, had to be measured on our own
questions: the instruction we wrote lowered the first ranks, while the
model card's own wording cost nothing, and Matryoshka training kept its
advantage on its own data but not on the book.

## Exercises

1. **The certified information.** A model trained with batches of 256
   pairs reports an InfoNCE loss of 0.1 nats.
    1. How much mutual information between question and passage does this
       loss certify, and what is the most any loss could certify at this
       batch size?
    1. Our training ended at about 0.05 nats with batches of 64. What do the
       two certificates say about which model is better, and why can a loss
       near 0 not rank models trained with different batch sizes?
   :exlabel:`ex_lm_emb-certified`

1. **Uniformity from the denominator.** Derive :eqref:`eq_lm_align_uniform`
   with care.
    1. For a finite batch whose negatives are drawn independently from the
       passages, show that the expectation over batches of
       $\log \frac{1}{B - 1}\sum_{j \neq i} e^{c_{ij}/\tau}$ is at most
       $\log \mathbb{E}_{\mathbf{d}}\, e^{\mathbf{q}_i^\top \mathbf{d}/\tau}$.
       Which way does the positive's own term push the finite-batch value?
       Does a small batch overstate or understate the uniformity term?
    1. Show that the uniformity of :eqref:`eq_lm_align_uniform`, with the
       logarithm inside the expectation over questions, is at most the
       version with the logarithm outside, at the same kernel width, when
       both expectations run over independent draws (a point may be drawn
       twice).
    1. Evaluate both versions at $t = 1/(2\tau)$ for the three four-point
       configurations of :numref:`subsec_lm_emb_align_uniform`, and check
       the two InfoNCE columns of the table. Explain why `align_uniform`, which
       averages over distinct pairs only, gives a value no higher than
       either for these configurations, in which every point sees the same
       distances. Construct a set of points for which it is higher than the
       version with the logarithm inside.
   :exlabel:`ex_lm_emb-uniformity`

1. [code] **Temperature.** ● Train the contrastive model at
   $\tau \in \{0.02, 0.05, 0.1, 0.2\}$ with $B = 64$ and the chosen learning
   rate, with seeds 0, 1 and 2 each, and report Recall@10 on the dev
   questions with paired, document-clustered intervals against
   $\tau = 0.05$.
    1. Describe the shape of Recall@10 against $\tau$, and explain it
       with the weight ratio $e^{\Delta/\tau}$ of the negatives
       (:eqref:`eq_lm_infonce_grad`). What does a very low temperature do to
       the gradient of a false negative?
    1. Which differences exceed the seed range?
   :exlabel:`ex_lm_emb-temperature`

1. [code] **Last-token pooling and padding.** Encode three questions of
   different lengths with Qwen3-Embedding, padded on the right.
    1. Take the hidden state at the last position of each row, as code that
       ignores the mask would, and show that for the two shorter questions
       it is the state of a padding token. Compare with
       `d2l.pool_hidden(..., 'last')`.
    1. Encode all dev questions this way and report the change in
       Recall@10 against the correct pooling, with its interval.
    1. Qwen3's padding token is the same `<|endoftext|>` that the tokenizer
       appends to every text. Why does the state at a padding position still
       differ from the state at the real end of the text, and why might the
       damage be smaller than you expected?
   :exlabel:`ex_lm_emb-padding`

1. [code] **Choosing an instruction.** ● The instruction we wrote lowered
   Qwen3-Embedding's order measures on the dev questions, and the model
   card's default task left them level with no instruction.
    1. Take 200 answerable questions from `d2l.book_qa('train')` and search
       the whole book for each. Compare MRR@10 without an instruction, with
       our task, with the model card's default task (the string
       `CARD_TASK` of the setup cell) and with two wordings of your own,
       using paired, document-clustered intervals.
    1. Choose the best wording on these training questions alone, then
       evaluate it once on the dev questions. Does its advantage survive?
       Why would a wording chosen on the dev questions overstate it?
   :exlabel:`ex_lm_emb-instruction`

1. **The symmetric loss.** The symmetric variant of
   :eqref:`eq_lm_infonce` averages the loss of questions choosing among
   passages and of passages choosing among questions.
    1. Derive $\partial \ell / \partial c_{ij}$ for the symmetric loss.
    1. Let passage $j$ answer question $i$ as well as $\mathbf{d}_i$ does,
       so that $c_{ij} = c_{ii}$. With what weight is $c_{ij}$ pushed down,
       in each direction, and how does that weight change as $\tau$ falls?
   :exlabel:`ex_lm_emb-symmetric`

1. [code] **Why Matryoshka training did not carry over.** On held-out
   Natural Questions pairs the Matryoshka model kept about 8 and 11 points
   more Recall@10 than the truncated plain model at 32 and 16 coordinates;
   on the book it kept none, even with the book's passages cut to the 160
   tokens of training.
    1. The questions differ: the book's share rare words with their
       passages. Propose a measurement on the stored checkpoints that tests
       whether this explains the missing gain, and run it. The rewritten
       questions of Exercise :exref:`ex_lm_emb-other-words` are one
       starting point.
    1. Cutting the passages to 160 tokens cost both models recall at full
       width. Find the dev questions that lose their relevant passage to
       the cut, and check whether the gold span of each lies past the first
       160 tokens of that passage.
    1. Add 32 and 16 to the training prefixes $\mathcal{K}$ and train again.
       What changes on Natural Questions at those widths, and on the book?
   :exlabel:`ex_lm_emb-matryoshka`

1. [extended] **Adapting to the book.** ●● Fine-tune the seed-0 contrastive
   model on pairs of a question from `d2l.book_qa('train')` and one of its
   relevant passages, with hard negatives mined only from passages of
   `train` documents.
    1. Report the gain in Recall@10 on the dev questions with a paired,
       document-clustered interval, against the seed range of three
       fine-tuning runs.
    1. Explain what would leak into the dev score if the negatives were
       mined from all passages of the book, and measure whether it does.
    1. Does the adapted model close the gap to Qwen3-Embedding?
   :exlabel:`ex_lm_emb-adapt`

1. [code] **Questions in other words.** Our questions borrow the rare words
   of their passages. Rewrite 40 dev questions so that they ask the same
   thing in other words, replacing every identifier and technical term that
   also occurs in the relevant passage by a description, by hand or with
   the local generator of :numref:`sec_lm_rag`.
    1. Compare BM25 and Qwen3-Embedding on the original and the rewritten
       questions by Recall@10 and Recall@1, with paired, document-clustered
       intervals. How much of BM25's 0.96 survives, and does its standing
       at the first rank?
    1. Which kinds of question lose most, and which method suffers more?
   :exlabel:`ex_lm_emb-other-words`

1. [code] **Checking the reranker's labels.** The audit's labels come from
   Qwen3-Reranker-0.6B, not from people. Mine again with the stored seed-0
   checkpoint, draw 30 training questions, and take one passage from ranks
   1 to 5 for each.
    1. Label each passage by hand, relevant or not, before looking at the
       reranker's probability of yes. Report your agreement with its label
       (probability above 0.5), with an interval.
    1. Which kinds of passage does the reranker call answers that you would
       not, or the reverse? How would your labels change the choice of
       ranks 11 to 50?
   :exlabel:`ex_lm_emb-audit-labels`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §19.3]{.kicker}

Text Embeddings from Contrastive Training<br>**pooling · anisotropy · InfoNCE · alignment and uniformity · hard negatives · decoder embedders · smaller vectors**
:::
:::

::: {.slide title="Averaged Encoder Vectors Find the Answer for 6% of Book Questions"}
[The problem]{.kicker}

Search the book's 5,324 passages with 220 questions. Is a relevant passage
among the ten nearest?

| ranking | training for retrieval | Recall@10 |
|---|---|---|
| BM25: count the question's words | none | 0.96 |
| ModernBERT, mean of its token vectors | none | 0.06 |

"How does the book's `load_array` helper decide whether to shuffle?"
Averaged vectors land on the page of utility functions, one heading above
the answer. The 220 questions share 73 different first passages; under
BM25, 210.

::: {.d2l-note .warn}
An encoder gives one vector per token. Nothing trained it to give one
vector per text.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- pool token vectors into one unit vector per text;
- measure how strongly a set of vectors shares one direction;
- derive the gradient of the contrastive loss and its limit for large
  batches;
- train an embedder on question-passage pairs and audit mined negatives;
- choose a released embedder and a storage format by measurement.

**Running example:** this book as 5,324 passages; 220 questions about 32
of its documents, drafted from the passages by a language model; BM25
(word counting) as the floor; the `load_array` question.
:::

::: {.slide title="One Vector per Text Is What Search Needs"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| movement | section |
|---|---|
| vectors | Word Embeddings |
| | Pretrained Encoders |
| | **Text Embeddings from Contrastive Training** |
| finding | Sparse, Dense and Approximate Retrieval |
| using | Retrieval-Augmented Generation |
| | Failures and Fixes in RAG |
:::

::: {.col .narrow}
- **Uses:** ModernBERT (one vector per token), cross-entropy, mutual
  information.
- **Skips:** indexes for large corpora, encoder pretraining.
:::
:::
:::

::: {.slide title="Three Ways to Pool, and Always Normalize"}
[Pooling]{.kicker}

| pooling | reads | natural for |
|---|---|---|
| CLS | the first position | encoders with a sentence-level task |
| mean | the average over real tokens | encoders |
| last token | the final real token | decoders: only it saw the whole text |

- The attention mask marks padding, so left and right padding both work.
- Scale to unit length: the dot product becomes the cosine. Unnormalized,
  long vectors rank high for every query.
- Raw ModernBERT with CLS pooling: Recall@10 = 0. Its masked-token loss
  never scores the first position.
:::

::: {.slide title="Raw Vectors Share One Direction: Mean Cosine 0.94"}
[Anisotropy]{.kicker}

Anisotropy: the mean cosine $\bar c$ of distinct pairs. For unit vectors
$\mathbf{z}_1, \ldots, \mathbf{z}_n$ with mean $\bar{\mathbf{z}}$,
$\|\sum_i \mathbf{z}_i\|^2 = n + n(n-1)\,\bar c$, so

$$\bar c = \frac{n\|\bar{\mathbf{z}}\|^2 - 1}{n - 1} \approx \|\bar{\mathbf{z}}\|^2.$$

| pooling, 2,000 book passages | anisotropy | Recall@10 |
|---|---|---|
| CLS | 0.95 | 0.00 |
| mean | 0.94 | 0.06 |
| mean, centered | 0.00 | 0.05 |

::: {.d2l-note}
Centering, subtracting the mean vector, removes the shared direction and
leaves Recall@10 at 0.05: the shared direction is not the cause.
:::
:::

::: {.slide title="Check: A Mean Cosine of 0.94. How Far Is a Passage from the Mean Direction?"}
[Check]{.kicker}

2,000 unit vectors of raw passages, mean cosine 0.94 between distinct
pairs. What is the angle between a typical vector and the mean direction?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The mean of $\mathbf{z}_i^\top \bar{\mathbf{z}}/\|\bar{\mathbf{z}}\|$
is $\|\bar{\mathbf{z}}\| \approx \sqrt{0.94} \approx 0.97$, the cosine of
about 14 degrees: a typical angle. The passages fill a narrow cone.
:::
:::
:::

::: {.slide title="The Other Passages of the Batch Are the Negatives"}
[InfoNCE]{.kicker}

$B$ pairs, unit vectors $\mathbf{q}_i$ (question) and $\mathbf{d}_i$ (its
passage), cosines $c_{ij} = \mathbf{q}_i^\top \mathbf{d}_j$, temperature
$\tau$:

$$\ell = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{c_{ii}/\tau}}{\sum_{j=1}^{B} e^{c_{ij}/\tau}}$$

- A $B$-way cross-entropy: $B - 1$ negatives per question, already encoded.
- Mutual information $I$ of question and passage: $I \geq \log B - \ell$,
  a bound never above $\log B$. At $B = 64$, 4.16 nats.

::: {.d2l-note}
The number of candidates, $B$, caps what the loss can certify: $\log B$.
:::
:::

::: {.slide title="A Negative 0.3 Closer Gets 403 Times the Push"}
[Gradient]{.kicker}

$$\frac{\partial \ell}{\partial c_{ij}} = \frac{p_{ij} - \delta_{ij}}{\tau B}, \qquad p_{ij} = \mathrm{softmax}_j(c_{ij}/\tau), \qquad \delta_{ij} = [i = j]$$

- Pull on the positive: $1 - p_{ii}$. Total push on the negatives: the same.
- Two negatives $\Delta$ apart in cosine: weight ratio $e^{\Delta/\tau}$.
- $p_{ii} \to 1$: every weight vanishes, and easy batches stop teaching.

@!text-embeddings-infonce-grad

::: {.d2l-note .warn}
A low temperature puts its gradient on the hardest negatives, true or
false.
:::
:::

::: {.slide title="As the Batch Grows, InfoNCE Becomes Alignment Plus Uniformity"}
[Derivation]{.kicker}

I. $\ell_i - \log B = -c_{ii}/\tau + \log \frac{1}{B}\sum_j e^{c_{ij}/\tau}$

II. $B \to \infty$: the average becomes $\mathbb{E}_{\mathbf{d}}\, e^{\mathbf{q}_i^\top \mathbf{d}/\tau}$

III. Unit vectors: $\mathbf{q}^\top \mathbf{d} = 1 - \tfrac12 \|\mathbf{q} - \mathbf{d}\|^2$; the $1/\tau$ cancel:

$$\ell - \log B \to \underbrace{\frac{1}{2\tau}\mathbb{E}\|\mathbf{q} - \mathbf{d}^+\|^2}_{\text{alignment}} + \underbrace{\mathbb{E}_{\mathbf{q}} \log \mathbb{E}_{\mathbf{d}}\, e^{-\|\mathbf{q} - \mathbf{d}\|^2/(2\tau)}}_{\text{uniformity}}$$

::: {.d2l-note}
Only two terms remain: one pulls pairs together, one pushes all vectors
apart.
:::

*Source: Wang and Isola (2020).*
:::

::: {.slide title="Four Points: Two Coincident Pairs on Opposite Sides Win"}
[The smallest case]{.kicker}

Pairs (1, 2) and (3, 4) on the unit circle; each point is a question whose
passage is its partner, so it also meets itself among the passages.

| arrangement | alignment | $\mathcal{U}_2$ | InfoNCE $- \log 4$, $\tau = 0.05$ | $\tau = 0.5$ |
|---|---|---|---|---|
| square: 0°, 90°, 180°, 270° | 2.0 | $-4.4$ | 18.6 | 0.87 |
| pairs coincide: 0°, 0°, 180°, 180° | 0.0 | $-1.1$ | $-0.69$ | $-0.68$ |
| all at 0° | 0.0 | 0.0 | 0.0 | 0.0 |

::: {.d2l-note}
Square: itself at cosine 1, partner at 0, a logit gap of 20. Coincident
pairs: partner and itself tie, a loss of $\log 2$.
:::
:::

::: {.slide title="Uniformity Cannot Improve While the Vectors Share a Direction"}
[Proposition]{.kicker}

Measured on passages alone (Wang and Isola), for independent passages
$\mathbf{z}, \mathbf{z}'$: $\mathcal{U}_t = \log \mathbb{E}\, e^{-t\|\mathbf{z} - \mathbf{z}'\|^2}$.
The loss's own term has $t = 1/(2\tau)$; we measure at $t = 2$.

Jensen, $\log \mathbb{E} e^X \geq \mathbb{E} X$, with anisotropy $a$:

$$\mathcal{U}_t \;\geq\; -2t\,(1 - a)$$

Raw mean pooling: $a = 0.94$, so $\mathcal{U}_2 \geq -0.24$; measured
$-0.23$.

::: {.d2l-note}
To spread the vectors, training must shorten the mean vector.
:::
:::

::: {.slide title="20,000 Wikipedia Pairs, One Epoch, One Minute"}
[Training]{.kicker}

- Natural Questions: search questions with Wikipedia passages; the book
  measures transfer.
- `query: ` / `passage: ` prefixes, mean pooling, $B = 64$, $\tau = 0.05$.
- Learning rate $10^{-4}$, chosen on 300 held-out pairs.
- No batch holds a text twice: it would be a negative of its own question.
- 313 steps, about 60 s on one RTX 4090; seeds 0, 1, 2.

Loss: 3.2 nats (first ten steps) to 0.05 (last ten).
:::

::: {.slide title="Check: What Does a Final Loss of 0.05 Nats at B = 64 Certify?"}
[Check]{.kicker}

The bound: $I \geq \log B - \ell$.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\log 64 - 0.05 \approx 4.11$ nats, of at most 4.16. The model
picks its passage among 64 nearly every time, so $p_{ii} \approx 1$ and the
gradient weights vanish. Harder negatives or larger batches give it more to
learn from; whether that helps must be measured.
:::
:::
:::

::: {.slide title="A Third of the Top-5 Mined Negatives Answer the Question"}
[Hard negatives]{.kicker}

Hard negatives: passages the model ranks high that are not the labeled
answer. Mined by the seed-0 model; 500 questions, one passage per bucket,
judged by Qwen3-Reranker-0.6B, which reads question and passage together
(a model, not people):

| mined ranks | probably relevant |
|---|---|
| 1 to 5 | 30% |
| 6 to 10 | 10% |
| 11 to 30 | 2% |
| 31 to 50 | under 1% |

Adding one negative from ranks 11 to 50: +0.03 Recall@10 in one run,
interval reaching 0, and 128 candidates instead of 64. Not yet shown to
help.

::: {.d2l-note .rule}
Mine below the top ranks, audit with a model that reads question and
passage together, and compare gains with the seed range.
:::
:::

::: {.slide title="A Decoder Embedder Reads Its Vector at the Last Token"}
[Qwen3-Embedding-0.6B]{.kicker}

- A Qwen3 decoder with causal attention, trained on far more pairs: weak
  pairs written by a larger Qwen3 model, then labeled pairs with hard
  negatives, then a merge of checkpoints.
- **Last-token pooling**, **left padding**: every row ends in a real token.
- **Query instruction**: `Instruct: {task}\nQuery:{query}`; passages bare.
  The task is free text.

::: {.d2l-note .warn}
A wrong convention raises no error. Pooling and padding errors read the
wrong token; the instruction's effect must be measured.
:::
:::

::: {.slide title="Check: Right Padding, Last Position. What Does the Vector Read?"}
[Check]{.kicker}

A batch of three questions, padded on the right; the code takes the
hidden state at position $T - 1$ of every row.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** For every question shorter than the longest, the state of a
padding token. Index the last real token through the attention mask, or
pad on the left.
:::
:::
:::

::: {.slide title="Contrastive Training Adds 70 Points; a Released Embedder About 20 More"}
[Results on the book]{.kicker}

| model | Recall@1 | Recall@10 | nDCG@10 | anisotropy |
|---|---|---|---|---|
| BM25 | 0.76 | 0.96 | 0.82 | |
| raw mean pooling | 0.04 | 0.06 | 0.04 | 0.94 |
| contrastive, 3 seeds | 0.36 to 0.41 | 0.73 to 0.78 | 0.50 to 0.53 | 0.52 (seed 0) |
| contrastive + 1 hard negative | 0.42 | 0.80 | 0.56 | |
| Qwen3-Embedding + instruction | 0.67 | 0.98 | 0.79 | 0.28 |

Paired: raw $\to$ contrastive +0.70 Recall@10, 15 times the seed spread;
contrastive $\to$ Qwen3-Embedding +0.22.

::: {.d2l-note .warn}
The questions borrow their passages' rare words: no embedder here beats
BM25 at rank 1. Qwen3-Embedding may also have read this book on the web.
:::
:::

::: {.slide title="Our Instruction Lowers the First Ranks; the Card's Costs Nothing"}
[Instructions]{.kicker}

Qwen3-Embedding, 220 book questions. Change against no instruction, paired,
intervals over documents. Our task: questions about the book; the card's:
web search queries.

| task in the instruction | Recall@1 | MRR@10 | Recall@10 |
|---|---|---|---|
| our task | $-0.032$ [$-0.068$, 0.000] | $-0.026$ [$-0.048$, $-0.004$] | $-0.014$ [$-0.028$, 0.000] |
| the card's task | +0.005 [$-0.025$, +0.034] | +0.007 [$-0.012$, +0.027] | $-0.005$ [$-0.015$, 0.000] |

The card reports +1 to 5% from instructions. Here neither wording gains.

::: {.d2l-note .rule}
An instruction's wording is a parameter: choose it on training questions,
never on the questions you report.
:::
:::

::: {.slide title="Training Spreads Random Passages Apart Faster Than Pairs"}
[Geometry]{.kicker}

For unit vectors with anisotropy $a$, two random passages lie at mean
squared distance $2 - 2a$. Alignment: the mean squared distance from a
question to its passage.

| model | question to its passage | two random passages | ratio |
|---|---|---|---|
| raw mean pooling | 0.19 | 0.12 | 0.6 |
| contrastive, seed 0 | 0.60 | 0.96 | 1.6 |
| Qwen3-Embedding | 0.61 | 1.43 | 2.4 |

::: {.d2l-note}
Training moved pairs apart too, but unrelated passages much more.
Retrieval depends on the ratio.
:::
:::

::: {.slide title="Int8 Loses Nothing Measurable; 1,024 Signs Beat 32 Floats"}
[Smaller vectors]{.kicker}

| Qwen3-Embedding stored as | bytes | Recall@10 | GB per $10^8$ |
|---|---|---|---|
| float32, 1,024 coordinates | 4,096 | 0.98 | 410 |
| int8, 1,024 | 1,024 | 0.98 | 102 |
| float32, first 128 | 512 | 0.90 | 51 |
| float32, first 32 | 128 | 0.62 | 12.8 |
| binary + rescoring, 1,024 | 128 | 0.96 | 12.8 |

::: {.d2l-note}
At equal bytes, many coarse coordinates rank far better than a few precise
ones.
:::
:::

::: {.slide title="Signs Estimate Angles; Rescoring Restores the Query's Magnitudes"}
[Binary vectors]{.kicker}

- Random direction $\mathbf{r}$: $P(\mathrm{sign}\,\mathbf{r}^\top\mathbf{a} \neq \mathrm{sign}\,\mathbf{r}^\top\mathbf{b}) = \theta/\pi$.
- Hamming distance: XOR and population count. It ignores magnitude.
- Rescoring: top 100 by Hamming, reordered by float query $\cdot$ passage
  signs. Passages stay at one bit.

| model | float | binary + rescoring |
|---|---|---|
| contrastive, plain | 0.76 | 0.72 |
| contrastive, Matryoshka | 0.74 | 0.72 |
| Qwen3-Embedding | 0.98 | 0.96 |

::: {.d2l-note}
A loss of about 2 to 4 points at 1/32 of the bytes.
:::
:::

::: {.slide title="Matryoshka Training Helps Where It Was Trained, Not on the Book"}
[Truncation]{.kicker}

$\ell_{\textrm{MRL}}$: the mean of InfoNCE on the first $k$ coordinates,
renormalized, over $k \in \{768, 512, 256, 128, 64\}$.

| first $k$ of 768 | NQ, plain | NQ, Matryoshka | book, plain | book, Matryoshka |
|---|---|---|---|---|
| 64 | 0.91 | 0.92 | 0.60 | 0.58 |
| 32 | 0.76 | 0.83 | 0.41 | 0.40 |
| 16 | 0.51 | 0.62 | 0.14 | 0.16 |

::: {.d2l-note}
On held-out Natural Questions: +8 and +11 points at 32 and 16, intervals
excluding 0. On the book: within noise, even with its passages cut to the
160 tokens of training. Length is not the cause.
:::
:::

::: {.slide title="A Leaderboard Average Does Not Choose Your Embedder"}
[Choosing]{.kicker}

- MTEB averages many tasks and languages; one domain is a small part.
- Build your own: a few hundred questions about your documents, ideally
  written by people who have not just read the answer, and a BM25 floor.
- Compare with paired differences, intervals over documents, and the seed
  range.
- On the book the embedder finds more within ten (+0.04 Recall@10 without
  an instruction, interval excludes 0). At rank 1 the two are level, and
  BM25 leads once our instruction is added. Keep both, the embedder in int8
  or binary + rescoring.

::: {.d2l-note .rule}
A library loads a model's pooling and prompts. Check them against the card.
:::
:::

::: {.slide title="Check: A Hundred Million Passages as 1,024-Bit Vectors. How Many GB?"}
[Check]{.kicker}

Qwen3-Embedding, binary: one bit per coordinate, $10^8$ passages.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** 1,024 bits are 128 bytes, and $128 \times 10^8$ bytes are
12.8 GB, against 410 GB in float32. With rescoring, Recall@10 stays at
0.96 of 0.98.
:::
:::
:::

::: {.slide title="Recap"}
[Recap]{.kicker}

- Raw mean pooling: Recall@10 0.06; BM25 0.96. Mean cosine 0.94 =
  squared length of the mean vector; removing it does not help.
- InfoNCE: $B - 1$ free negatives, a $\log B$ ceiling, gradient weights
  $e^{\Delta/\tau}$; large-batch limit = alignment + uniformity.
- 20,000 pairs, one minute: +0.70 Recall@10. A reranker calls 30% of the
  top-5 mined passages answers: mine below them.
- Qwen3-Embedding: 0.98, possibly having read the book; 0.96 at 128 bytes.
  Our instruction wording cost MRR@10 0.03; the card's cost nothing.

**Next:** a flat scan is one matrix product for 5,324 passages. Millions of
passages, and questions about exact identifiers, need an index and a word
index beside it.
:::
