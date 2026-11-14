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

<!-- Placeholder (stage 2, infra step): the cells below are the section's
shared library, written before the section; the prose around them comes with
the section. Cell ids follow the section specification. -->

```{.python .input #text-embeddings-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

<!-- Placeholder (stage 2): the two released embedders and the reranker, pinned. -->

```{.python .input #text-embeddings-revisions}
%%tab pytorch
#@save
d2l.HF_ENCODER_REVISIONS['Qwen/Qwen3-Embedding-0.6B'] = (
    '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
d2l.HF_ENCODER_REVISIONS['Qwen/Qwen3-Reranker-0.6B'] = (
    'e61197ed45024b0ed8a2d74b80b4d909f1255473')
d2l.HF_ENCODER_REVISIONS['microsoft/harrier-oss-v1-0.6b'] = (
    'f9b9dc8d367d443f2479d27aa5d8d2850c0774ee')
```

<!-- Placeholder (stage 2): question-passage pairs from Natural Questions, pinned to a commit and downloaded at run time. -->

```{.python .input #text-embeddings-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['nq-pairs'] = (
    'https://huggingface.co/datasets/sentence-transformers/'
    'natural-questions/resolve/f9e894e1081e206e577b4eaa9ee6de2b06ae6f17/'
    'pair/train-00000-of-00001.parquet',
    '545c1495f10231652b59e67f7ab02e6d21252fd0')
```

<!-- Placeholder (stage 2): three ways to pool token states into one vector. -->

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

<!-- Placeholder (stage 2): batched encoding of many texts, longest first. -->

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

<!-- Placeholder (stage 2): Recall@k, MRR@10 and nDCG@10 per query. -->

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
