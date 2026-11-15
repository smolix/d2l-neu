# Word Embeddings
:label:`sec_lm_word_embeddings`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "2.1 Word Embeddings · T1").
Blueprint: docs/llm/blueprints/ch02_encoders.md section 2.1.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** A word vector is a factorization of co-occurrence statistics.
**Opening:** The reader has seen the token-embedding table of `sec_gpt` trained end to end and never asked what its rows mean. Two words that share contexts ("physician" and "doctor") sit near each other in that table. The section builds the simplest model whose rows are explained by counts, verifies the explanation numerically, and closes on the one thing a static vector cannot do: separate two meanings of one word.
**Teaches:**
- Skip-gram with negative sampling: the loss over center and context scores with K noise words, the 0.75-power noise distribution, subsampling of frequent words; the labels `subsec_negative-sampling` and `sec_word2vec_data` are defined here because `sec_mdl-mi-estimation` cites them.
- CBOW (the context predicts the center word) and fastText's character n-grams (a word's vector is the sum of its n-gram vectors, so an unseen word still gets one; 3.2's quality classifier is a fastText model) in one sentence each.
- The optimum as shifted pointwise mutual information, stated and checked numerically (the derivation is the exercise in `sec_mdl-mi-estimation`); truncated SVD of the positive PMI matrix as the closed-form counterpart; GloVe as weighted least squares on log co-occurrences, in one sentence with its one difference.
- Analogies as a cosine query with the inputs excluded, and their caveats; bias as a description of the corpus, and why projection hides rather than removes it; tied input and output tables.
- Static vectors against the input table of a language model: one point per word type, and what a contextual encoder adds (2.2).
- Closing paragraph `subsec_lm_categorical_embeddings`: an embedding table for any categorical variable, the target of a pointer `sec_pandas` may add.
**Derivation:** none (the shifted-PMI result is stated; its proof is `ex_mdl-mutual-information-negative-sampling-pmi`). The numerical check: on a 4-word toy corpus, compute the count matrix, the PMI shifted by log K, and the SGNS scores after convergence, and show they agree.
**Notebook:**
1. Build a word-level slice of WikiText-103 from the download `d2l.WikiText103` uses (whitespace tokens, minimum count 10), with `d2l.Vocab`; count co-occurrences in a window of 5 as a sparse matrix.
2. Train SGNS with 5 negatives for 5 epochs (3 seeds), compute SVD of positive PMI at dimension 100, and train GloVe with the same window; evaluate Spearman correlation with human similarity judgments on the covered pairs of two similarity sets (coverage printed).
3. Verify the shifted-PMI relation on the toy corpus of the derivation and, approximately, on the slice (scatter of SGNS scores against shifted PMI for frequent pairs).
4. Inspect nearest neighbours of a few words in Qwen3-0.6B's token-embedding table (loaded through `d2l.load_qwen3`) and in the SGNS table; show one word with two meanings sharing one vector.
5. Claim that survives noise: all three methods correlate with human judgments alike (within the pair bootstrap and the seed range), and the choice of noise exponent and subsampling moves the correlation by as much as the choice of method. No ordering is asserted.
6. Reused pieces: `d2l.Vocab`, `d2l.WikiText103`'s download, `d2l.load_qwen3`; new `d2l.sgns_loss`, `d2l.ppmi`, `d2l.cooccurrence_counts`, `d2l.spearman_eval`, `d2l.nearest_neighbors`.
**Figures:** `fig_lm_sgns_pmi` (the window, the center and context tables, the noise draws, and the PMI matrix that the optimum factorizes).
**Restate / forward:** `subsec_mdl-pmi` (PMI measures how far a pair's co-occurrence exceeds independence); `sec_mdl-svd-low-rank` (the truncated SVD is the best low-rank approximation in Frobenius norm, `subsec_mdl-eckart-young`). Forward: 2.2 (one vector per token in context).
**Exercises:** 5 to 7. Load-bearing: derive the effect of subsampling with threshold t on the expected count of a word of frequency f and on the PMI of pairs involving it (derive); measure analogy accuracy against dimension 25 to 300 with and without excluding the query words (measure); check numerically that the SGNS optimum with free scores on the toy corpus equals PMI minus log K (check); open: fit a 32-dimensional embedding table for a categorical column of a tabular dataset and compare with one-hot features under the same linear model.
**Writer notes:**
- The corpus is a word-level WikiText-103 slice, not Penn Treebank; the exercise owner is `sec_mdl-mi-estimation`, not `sec_mdl-mutual-information`; the closing paragraph `subsec_lm_categorical_embeddings` belongs to this section. `d2l.WikiText103` (defined in `chapter_optimization/adam.md`) yields byte-level data; 2.1 reads the same downloaded parquet as words. Cap the slice (a few million words) so SGNS trains in minutes on CPU; print the vocabulary size and the coverage of the similarity pairs, which is far better than PTB's (the blueprint measured 217 of 353 WordSim pairs covered under PTB's 6,719-type vocabulary).
- Legacy names: `d2l.DATA_HUB['ptb']` moves to `lstm.md`; the retiring `read_ptb`, `load_data_ptb`, `get_negatives`, `batchify` helpers are not kept unless an exercise needs them; `d2l.TokenEmbedding` with the `glove.6b.50d` and `glove.6b.100d` keys stays for the analogy and bias cells; state dropped names in the migration note. `d2l.TokenEmbedding` is defined in the retiring `similarity-analogy.md`: define it in this section under the same name and signature so it survives the deletion. Hierarchical softmax (the retiring `approx-training.md`) is not carried over. Mine `docs/exercise-catalog/17-nlp-pretraining-a.md` for exercises.
- The two labels: define `subsec_negative-sampling` on the `##` subsection that states the SGNS loss with K noise words and center and context scores, and `sec_word2vec_data` on the subsection that states the count^0.75 noise distribution, so `mdl-mi-estimation.md:1230-1247` resolves unchanged.
- Pitfalls to state: evaluating the center table, the context table or their sum changes the numbers (state the choice); padded context positions must be masked out of the loss; subsampling per occurrence, not per type; SGNS "is" PMI factorization only at the optimum with free scores; pairs outside the vocabulary silently shrink the test set.
- Data: `almogtavor/WordSim353` and `Yuti/SimLex-999` (both CC BY 4.0) downloaded at run time; pretrained GloVe-6B through the book's mirror.
- Do not re-teach: PMI (`subsec_mdl-pmi`), the SVD (`sec_mdl-svd-low-rank`), the shifted-PMI proof (`sec_mdl-mi-estimation`).
- Committed artifacts: the similarity result table as JSON for the freshness check; nothing else (T0-capable).
- Optional: the preliminaries chapter may add a one-line pointer in `sec_pandas` to `subsec_lm_categorical_embeddings` (nothing breaks either way).
**Sources:**
- Papers: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (skip-gram and CBOW); [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (negative sampling, subsampling, the 3/4 power); Levy and Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (NeurIPS 2014; cited by key `Levy.Goldberg.2014`: the shifted-PMI optimum); [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) (weighted least squares on log counts); [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016/) (hyperparameters matter as much as the method: the notebook's thesis); [Man is to Computer Programmer as Woman is to Homemaker?](https://arxiv.org/abs/1607.06520) (bias in embeddings); [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737) (the closing paragraph).
- Code and documentation: the [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext) (the `wikitext-103-raw-v1` parquet files that `d2l.WikiText103` downloads and 2.1 reads as words); the Qwen3-0.6B card for the tied table ([Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)).
- Courses, tutorials and blogs: [CS224N](https://web.stanford.edu/class/cs224n/) (word-vector lectures and assignment A1); Jay Alammar, [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/).
-->

<!-- Placeholder (stage 2, cells step): the code cells of the section in the
order of the section specification, with one-line placeholders where the
prose goes. Headings and their labels come with the prose; the legacy labels
`subsec_negative-sampling` and `sec_word2vec_data` are defined here only after
the retiring NLP chapters (which still define them) are deleted. -->

<!-- Placeholder (stage 2), opening: the token table of sec_gpt was trained end to end and nobody asked what its rows mean; one-hot vectors make every pair of words orthogonal; which statistic of a corpus does a word vector encode? -->

```{.python .input #word-embeddings-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #word-embeddings-imports-extras}
%%tab pytorch
import collections
import hashlib
import json
import os
import pandas as pd
import scipy.sparse.linalg
import time
import transformers
from safetensors import safe_open
```

<!-- Placeholder (stage 2): the switches of the build (GPU, regeneration of the comparison grid) and the hyperparameters every method shares. -->

```{.python .input #word-embeddings-config}
%%tab pytorch
GPU = torch.cuda.is_available()
# A job given a GPU (CUDA_VISIBLE_DEVICES set) must not fall back to the CPU
assert GPU or not os.environ.get('CUDA_VISIBLE_DEVICES'), (
    'a GPU is assigned, but PyTorch sees none')
device = d2l.try_gpu()
ARTIFACTS = {  # measured result: (registered key, file a regeneration writes)
    'we-grid': ('lm-enc-we-grid-v1', 'lm-enc-we-grid.json')}
RERUN = d2l.lm_rerun(ARTIFACTS)  # D2L_RERUN=we-grid regenerates the grid
assert GPU or not RERUN, 'regenerating the grid needs a GPU'
N_WORDS, MIN_COUNT, WINDOW = 3_000_000, 10, 5  # the corpus slice, the window
DIM, K, T_SUB, ALPHA = 100, 5, 1e-4, 0.75  # width, noise words, subsampling
EPOCHS, BATCH, LR = 5, 1024, 2e-3  # SGNS: centers per minibatch, Adam
SEEDS = (0, 1, 2)
TIC = time.time()
print(f'device {device}, RERUN={",".join(sorted(RERUN)) or False}')
```

<!-- Placeholder (stage 2), ## Co-occurrence Counts: the distributional hypothesis; the corpus is a word-level slice of WikiText-103, read from the file that d2l.WikiText103 downloads; two sets of word pairs with human similarity ratings. -->

```{.python .input #word-embeddings-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['wikitext-103-raw-train-0'] = (
    'https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
    'b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/'
    'train-00000-of-00002.parquet',
    '3cbc792dfc0eeedf8d7333acdc15a9f536032d9a')
d2l.DATA_HUB['wordsim353'] = (  # Finkelstein et al.'s combined ratings
    'https://raw.githubusercontent.com/piskvorky/gensim/'
    '37f90ec121eb7cd401448a947e80953e0c53ccdc/gensim/test/test_data/'
    'wordsim353.tsv', '5207d1bb6dd863994fa4032f48f149bf28a5468e')
d2l.DATA_HUB['simlex999'] = (
    'https://huggingface.co/datasets/Yuti/SimLex-999/resolve/'
    'e87197824261f272ad1e9c02e32db3144bbeb0ca/data/'
    'train-00000-of-00001.parquet',
    'f3b074e2e08f83aa469f2d05c5b789b5ad2d3c49')
d2l.DATA_HUB['lm-enc-we-grid-v1'] = (  # D2L_RERUN=we-grid, RTX 4090
    d2l.LM_DATA_URL + 'lm-enc-we-grid-v1.json',
    '8c6bc0c598c8493fa407b3662f7a26472522169e')
```

<!-- Placeholder (stage 2): pretrained GloVe vectors from the book's mirror, for the analogy and bias probes. Not yet `#@save`: the retiring word-similarity section registers the same two keys, and a second registration would change the library shard its notebook depends on; the marker is added when that section is deleted (the values are identical). -->

```{.python .input #word-embeddings-glove-hub}
%%tab pytorch
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')
```

<!-- Placeholder (stage 2): the slice: headings dropped, lowercase, alphabetic whitespace tokens (one apostrophe allowed, as in "'s"), a break at each line end; words below the minimum count leave the stream. -->

```{.python .input #word-embeddings-slice}
%%tab pytorch
def check_sha1(path, key):  # d2l.download checks only a cached file
    with open(path, 'rb') as f:
        got = hashlib.sha1(f.read()).hexdigest()
    assert got == d2l.DATA_HUB[key][1], f'{path}: sha1 {got}'
    return path

path = check_sha1(d2l.download('wikitext-103-raw-train-0', os.path.join(
    '..', 'data', 'wikitext-103')), 'wikitext-103-raw-train-0')
lines, n = [], 0
for line in pd.read_parquet(path)['text']:
    line = line.strip().lower()
    if not line or line.startswith('= '):  # an empty line or a heading
        continue
    words = [w for w in line.split() if w.replace("'", '', 1).isalpha()]
    lines.append(words[:N_WORDS - n])
    n += len(lines[-1])
    if n == N_WORDS:
        break
vocab = d2l.Vocab([w for words in lines for w in words], min_freq=MIN_COUNT)
V = len(vocab)
ids = np.concatenate([np.array(vocab[words] + [-1]) for words in lines])
oov = np.mean(ids[ids >= 0] == vocab.unk)
ids = ids[ids != vocab.unk]  # word ids, -1 at each line end
counts = np.bincount(ids[ids >= 0], minlength=V)
print(f'{n:,} tokens in {len(lines):,} lines, '
      f'{len(collections.Counter(w for l in lines for w in l)):,} types\n'
      f'vocabulary {V:,} (count >= {MIN_COUNT}), {oov:.1%} of tokens '
      'outside it')
```

<!-- Placeholder (stage 2): a window of WINDOW words on each side; the counts form a sparse symmetric matrix; pairs never cross a line break. PMI restated in one sentence (subsec_mdl-pmi). -->

```{.python .input #word-embeddings-cooccurrence}
%%tab pytorch
#@save
def cooccurrence_counts(corpus, vocab_size, window=5, weighting='uniform'):
    """Word-context counts within `window` positions as a symmetric
    scipy.sparse.csr_matrix (float64). `corpus` is a 1-D array of word ids
    with -1 at sentence breaks, which no pair crosses; `weighting` is
    'uniform' (each pair counts 1) or 'harmonic' (1/distance)."""
    import scipy.sparse
    if weighting not in ('uniform', 'harmonic'):
        raise ValueError(f'unknown weighting {weighting!r}')
    corpus = np.asarray(corpus)
    sentence = np.cumsum(corpus < 0)  # the sentence of each position
    X = scipy.sparse.csr_matrix((vocab_size, vocab_size))
    for d in range(1, window + 1):  # all pairs at distance d
        a, b = corpus[:-d], corpus[d:]
        ok = (a >= 0) & (b >= 0) & (sentence[:-d] == sentence[d:])
        a, b = a[ok], b[ok]
        w = 1.0 if weighting == 'uniform' else 1.0 / d
        X = X + scipy.sparse.csr_matrix(  # duplicate entries are summed
            (np.full(2 * len(a), w), (np.r_[a, b], np.r_[b, a])),
            shape=(vocab_size, vocab_size))
    return X
```

```{.python .input #word-embeddings-counts}
%%tab pytorch
tic = time.time()
X = d2l.cooccurrence_counts(ids, V, WINDOW)
print(f'{X.nnz:,} nonzero counts of {V * V:,} ({X.nnz / V**2:.2%}), '
      f'{X.sum():,.0f} (word, context) pairs, {time.time() - tic:.1f} s')
# X[w, o]: how often o occurs in the window of w; X is symmetric, so its
# upper triangle lists each pair of words once
upper = scipy.sparse.triu(X).tocoo()
for i in np.argsort(-upper.data)[:5]:
    a, b = (vocab.idx_to_token[j] for j in (upper.row[i], upper.col[i]))
    print(f'    {a:>5} {b:<5} {upper.data[i]:9,.0f}')
```

<!-- Placeholder (stage 2), evaluating word vectors: Spearman correlation between cosine similarity and human ratings, on the pairs whose two words are in the vocabulary; a pair bootstrap gives the interval. -->

```{.python .input #word-embeddings-spearman}
%%tab pytorch
#@save
SpearmanResult = collections.namedtuple(
    'SpearmanResult', 'rho ci n_covered n_total cosines')

#@save
def spearman_eval(embeddings, vocab, pairs, n_boot=2000, seed=0):
    """Spearman correlation between the cosine similarities of embeddings
    and the human scores of `pairs` (a DataFrame with columns word1, word2,
    score), on the pairs whose two words `vocab` (a d2l.Vocab or a
    TokenEmbedding) holds; `ci` is the 95% percentile interval of n_boot
    resamples of those pairs (nan for n_boot=0), `cosines` their cosines."""
    from scipy.stats import rankdata
    idx = vocab.token_to_idx
    covered = [w1 in idx and w2 in idx
               for w1, w2 in zip(pairs.word1, pairs.word2)]
    p = pairs[covered]
    Z = F.normalize(torch.as_tensor(embeddings).float(), dim=1)
    i = torch.tensor([idx[w] for w in p.word1], dtype=torch.long)
    j = torch.tensor([idx[w] for w in p.word2], dtype=torch.long)
    cos = (Z[i] * Z[j]).sum(1).double().numpy()
    human = p.score.to_numpy(np.float64)

    def rho(a, b):  # Pearson correlation of the ranks, along the last axis
        a, b = rankdata(a, axis=-1), rankdata(b, axis=-1)
        a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
        return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))

    ci = (math.nan, math.nan)
    if n_boot:
        b = np.random.default_rng(seed).integers(0, len(p), (n_boot, len(p)))
        ci = tuple(np.quantile(rho(cos[b], human[b]), [0.025, 0.975]))
    return SpearmanResult(float(rho(cos, human)), tuple(map(float, ci)),
                          len(p), len(pairs), cos)
```

<!-- Placeholder (stage 2): WordSim353 (Finkelstein et al., the combined mean ratings; relatedness and similarity mixed; the pair money/cash occurs twice, as in the original) and SimLex-999 (similarity only); how many pairs the slice's vocabulary covers. -->

```{.python .input #word-embeddings-similarity-data}
%%tab pytorch
SOURCES = {'WordSim353': 'wordsim353', 'SimLex-999': 'simlex999'}
SETS = {
    'WordSim353': pd.read_csv(d2l.download_keyed('wordsim353'), sep='\t',
                              comment='#', header=None),
    'SimLex-999': pd.read_parquet(d2l.download_keyed('simlex999'))}
covered = {}
for name, df in SETS.items():
    df = df.iloc[:, :3].set_axis(['word1', 'word2', 'score'], axis=1)
    SETS[name] = df = df.assign(word1=df.word1.str.lower(),
                                word2=df.word2.str.lower())
    covered[name] = sum(w1 in vocab.token_to_idx and w2 in vocab.token_to_idx
                        for w1, w2 in zip(df.word1, df.word2))
    print(f'{name}: {covered[name]} of {len(df)} pairs covered')
```

<!-- Placeholder (stage 2): the floor: random vectors of the same width correlate with the ratings only by chance. -->

```{.python .input #word-embeddings-floor}
%%tab pytorch
floor = {name: [d2l.spearman_eval(
    torch.randn(V, DIM, generator=torch.Generator().manual_seed(s)), vocab,
    p, n_boot=0).rho for s in range(20)] for name, p in SETS.items()}
for name, r in floor.items():
    print(f'{name}: random vectors, rho mean {np.mean(r):+.3f}, '
          f'range [{min(r):+.3f}, {max(r):+.3f}] over 20 seeds')
```

<!-- Placeholder (stage 2), ## Skip-Gram with Negative Sampling: the loss eq_lm_sgns_loss with center vector v_w, context vector u_o, score u_o^T v_w and K noise words; why the softmax over the vocabulary is avoided; CBOW and fastText in one sentence each; tied tables in one sentence. Padded slots are masked out of the loss. -->

```{.python .input #word-embeddings-sgns-loss}
%%tab pytorch
#@save
def sgns_loss(v, u, centers, contexts_negatives, labels, masks):
    """Skip-gram with negative sampling: the binary cross-entropy of the
    scores u_o^T v_w of each center (centers, shape (B,)) with its context
    and noise words (contexts_negatives, (B, L)), labels 1 for a context
    word and 0 for a noise word, averaged over the slots where masks is 1.
    v and u are the center and the context tables (nn.Embedding)."""
    scores = torch.einsum('bd,bld->bl', v(centers), u(contexts_negatives))
    loss = F.binary_cross_entropy_with_logits(
        scores, labels.float(), weight=masks.float(), reduction='sum')
    return loss / masks.sum()
```

<!-- Placeholder (stage 2), ### Subsampling and the Noise Distribution: discard probability max(1 - sqrt(t/f(w)), 0) per occurrence; noise P_alpha(w) proportional to n(w)^alpha; a window size drawn per center; the batch layout, shown once. -->

```{.python .input #word-embeddings-sgns-data}
%%tab pytorch
def keep_mask(ids, counts, t, rng):
    """Which positions survive subsampling: a word of frequency f is kept
    with probability min(1, sqrt(t / f)), independently per occurrence;
    breaks (-1) are always kept."""
    f = counts / counts.sum()
    keep = np.minimum(1.0, np.sqrt(t / np.maximum(f, 1e-30)))
    return (ids < 0) | (rng.random(len(ids)) < keep[np.maximum(ids, 0)])

def noise_table(counts, alpha):
    """The noise distribution, proportional to count^alpha over the words
    that occur."""
    w = np.where(counts > 0, counts.astype(np.float64) ** alpha, 0.0)
    return torch.tensor(w / w.sum(), dtype=torch.float32)

def pairs(ids, window, rng):
    """Each center with the ids of its context words as a row of 2 window
    slots (-1 where a slot is empty): a window size b in 1..window is drawn
    per center, and a context word lies at most b positions away in the
    same sentence. Centers without a context word are dropped."""
    n, pos = len(ids), np.arange(len(ids))
    sentence, b = np.cumsum(ids < 0), rng.integers(1, window + 1, n)
    ctx = np.full((n, 2 * window), -1)
    for d in range(1, window + 1):
        for col, j in ((window - d, pos - d), (window + d - 1, pos + d)):
            inside = (j >= 0) & (j < n)
            j = np.clip(j, 0, n - 1)
            ok = inside & (d <= b) & (sentence[j] == sentence) & (ids[j] >= 0)
            ctx[ok, col] = ids[j[ok]]
    keep = (ids >= 0) & (ctx >= 0).any(1)
    return ids[keep], ctx[keep]

def sgns_batches(centers, ctx, noise, K, batch_size, gen):
    """Shuffled minibatches (centers, contexts_negatives, labels, masks):
    the 2 window context slots, then K noise words per slot, drawn from
    noise; a noise word counts only if its context slot does."""
    dev = noise.device
    c, ctx = torch.from_numpy(centers).to(dev), torch.from_numpy(ctx).to(dev)
    S = ctx.shape[1]
    labels = torch.cat([torch.ones(S), torch.zeros(S * K)]).to(dev)
    perm = torch.randperm(len(c), generator=gen, device=dev)
    for i in range(0, len(perm), batch_size):
        idx = perm[i:i + batch_size]
        x = ctx[idx]
        neg = torch.multinomial(noise, len(idx) * S * K, replacement=True,
                                generator=gen).view(len(idx), S * K)
        m = (x >= 0).float()
        yield (c[idx], torch.cat([x.clamp(min=0), neg], 1),
               labels.expand(len(idx), -1),
               torch.cat([m, m.repeat_interleave(K, 1)], 1))

rng, gen = np.random.default_rng(0), torch.Generator(device).manual_seed(0)
centers, ctx = pairs(ids[keep_mask(ids, counts, T_SUB, rng)], WINDOW, rng)
print(f'one epoch: {len(centers):,} of {np.sum(ids >= 0):,} positions are '
      f'centers ({np.mean(ctx >= 0) * 2 * WINDOW:.1f} context words each)')
batch = next(sgns_batches(centers, ctx, noise_table(counts, ALPHA).to(device),
                          K, BATCH, gen))
for part, x in zip(('centers', 'contexts_negatives', 'labels', 'masks'),
                   batch):
    print(f'{part:>18}: {tuple(x.shape)}')
words = [vocab.idx_to_token[x] if m else '<pad>' for x, m in zip(
    batch[1][0].tolist(), batch[3][0].tolist())]
j = batch[3][0, :2 * WINDOW].argmax().item()  # the first filled slot
print(f'first row: center {vocab.idx_to_token[int(batch[0][0])]!r}')
print('    context slots, offsets -5..-1 and +1..+5:\n       ',
      ' '.join(words[:2 * WINDOW]))
print(f'    noise words of slot {j} ({words[j]!r}):',
      ' '.join(words[2 * WINDOW + j * K:2 * WINDOW + (j + 1) * K]))
print(f'    labels: {int(batch[2][0].sum())} ones, then zeros; masks keep '
      f'{int(batch[3][0].sum())} of {batch[3].shape[1]} slots')
```

<!-- Placeholder (stage 2), ### Training: two tables, Adam, a fresh subsample, fresh windows and fresh noise words in every epoch. The section evaluates the center table V; V + U is the other common choice. -->

```{.python .input #word-embeddings-sgns-train}
%%tab pytorch
def train_sgns(ids, counts, alpha=ALPHA, subsample=True, seed=0,
               epochs=EPOCHS, verbose=True):
    """Train SGNS on the id stream; returns the center and the context
    tables V and U as CPU tensors."""
    torch.manual_seed(seed)
    rng, gen = np.random.default_rng(seed), torch.Generator(device)
    gen.manual_seed(seed)
    v, u = nn.Embedding(V, DIM), nn.Embedding(V, DIM)
    nn.init.uniform_(v.weight, -0.5 / DIM, 0.5 / DIM)  # as word2vec does
    nn.init.zeros_(u.weight)
    v, u = v.to(device), u.to(device)
    opt = torch.optim.Adam([v.weight, u.weight], lr=LR)
    noise = noise_table(counts, alpha).to(device)
    for epoch in range(epochs):
        tic, total, n = time.time(), 0.0, 0
        kept = ids[keep_mask(ids, counts, T_SUB, rng)] if subsample else ids
        centers, ctx = pairs(kept, WINDOW, rng)
        for batch in sgns_batches(centers, ctx, noise, K, BATCH, gen):
            loss = d2l.sgns_loss(v, u, *batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item(), n + 1
        if verbose:
            print(f'epoch {epoch + 1}: loss {total / n:.4f}, '
                  f'{time.time() - tic:.1f} s')
    return v.weight.detach().cpu(), u.weight.detach().cpu()

def report(E, name):
    """Spearman correlation with both rating sets, with intervals."""
    out = [f'{s} {r.rho:.3f} [{r.ci[0]:.3f}, {r.ci[1]:.3f}]' for s, r in
           ((s, d2l.spearman_eval(E, vocab, p)) for s, p in SETS.items())]
    print(f'{name:>10}: ' + '; '.join(out))

tic = time.time()
sgns_v, sgns_u = train_sgns(ids, counts)
print(f'{time.time() - tic:.0f} s')
report(sgns_v, 'SGNS V')
report(sgns_v + sgns_u, 'SGNS V+U')
```

<!-- Placeholder (stage 2), ## What the Optimum Factorizes: with free scores the expected loss is minimized at s*(w, o) = pmi(w, o) - log K + log(p(o)/P(o)) (proof: ex_mdl-mutual-information-negative-sampling-pmi). A check on a toy corpus of four words, window 1. -->

```{.python .input #word-embeddings-toy-pmi}
%%tab pytorch
TOY = ('ice ice steam gas steam ice solid ice steam gas ice solid steam '
       'solid steam gas gas steam gas steam steam ice gas steam solid gas '
       'steam solid solid ice solid ice ice solid gas steam gas steam gas '
       'solid').split()
toy_words = ['ice', 'steam', 'solid', 'gas']
Xt = d2l.cooccurrence_counts([toy_words.index(w) for w in TOY], 4,
                             window=1).toarray()
n_w, n_o, total = Xt.sum(1), Xt.sum(0), Xt.sum()
pmi = np.log(Xt * total / np.outer(n_w, n_o))

def free_optimum(X, K, P):
    """Scores s(w, o), one free number per pair, minimizing the expected
    SGNS loss: X[w, o] positives, K X[w].sum() P(o) noise draws."""
    X, P = torch.tensor(X), torch.tensor(P)
    s = torch.zeros_like(X, requires_grad=True)
    opt = torch.optim.LBFGS([s], max_iter=500, tolerance_grad=1e-12,
                            tolerance_change=1e-16,
                            line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = (X * F.softplus(-s) + K * X.sum(1, keepdim=True) * P
                * F.softplus(s)).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return s.detach().numpy()

show4 = lambda A: pd.DataFrame(A, toy_words, toy_words).round(3)
print('counts (window 1):', show4(Xt).astype(int), sep='\n')
for alpha in (1.0, 0.75):
    P = n_o ** alpha / (n_o ** alpha).sum()
    s = free_optimum(Xt, K, P)
    shifted = pmi - np.log(K)
    err = [np.abs(s - shifted).max(),
           np.abs(s - shifted - np.log(n_o / total / P)).max()]
    print(f'alpha = {alpha}: max |s* - (pmi - log K)| = {err[0]:.1e}, '
          f'max |s* - (pmi - log K + log(p/P))| = {err[1]:.1e}')
    assert err[1] < 1e-6  # the stated optimum, for either noise
print('s* for alpha = 0.75:', show4(s), sep='\n')
```

<!-- Placeholder (stage 2): the closed-form counterpart: truncated SVD of the positive PMI matrix (subsec_mdl-eckart-young), with the context distribution smoothed by the same 0.75 power. -->

```{.python .input #word-embeddings-ppmi}
%%tab pytorch
#@save
def ppmi(counts, alpha=1.0, shift=0.0):
    """Positive (shifted) PMI of a count matrix as a scipy.sparse.csr_matrix:
    max(log(p(w, o) / (p(w) p_alpha(o))) - shift, 0), where p_alpha(o) is
    proportional to n(o)^alpha over the contexts that occur."""
    import scipy.sparse
    C = scipy.sparse.coo_matrix(counts)
    total = C.sum()
    p_w = np.asarray(C.sum(1)).ravel() / total
    n_o = np.asarray(C.sum(0)).ravel()
    p_o = np.where(n_o > 0, n_o ** alpha, 0.0)
    p_o = p_o / p_o.sum()
    pmi = (np.log(C.data / total) - np.log(p_w[C.row])
           - np.log(p_o[C.col]) - shift)
    keep = pmi > 0
    return scipy.sparse.csr_matrix(
        (pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)
```

```{.python .input #word-embeddings-svd}
%%tab pytorch
def svd_embed(M, k, seed=0):
    """Word vectors U_k Sigma_k^(1/2) from the top-k singular triplets."""
    U, S, _ = scipy.sparse.linalg.svds(M, k=k, rng=np.random.default_rng(seed))
    order = np.argsort(-S)
    return torch.tensor(U[:, order] * np.sqrt(S[order]), dtype=torch.float32)

tic = time.time()
M = d2l.ppmi(X, alpha=ALPHA)
svd_vecs = svd_embed(M, DIM)
print(f'PPMI: {M.nnz:,} positive entries; rank-{DIM} SVD in '
      f'{time.time() - tic:.0f} s')
# Eckart-Young on the block of the 2,000 most frequent words: the residual
# of the truncated SVD equals the root sum of the discarded sigma_i^2
B = M[1:2001, 1:2001].toarray()  # row 0 is <unk>
U, S, Vt = scipy.sparse.linalg.svds(B, k=DIM, rng=np.random.default_rng(0))
tail = np.sqrt(np.sum(np.linalg.svd(B, compute_uv=False)[DIM:] ** 2))
rel = abs(np.linalg.norm(B - (U * S) @ Vt) - tail) / tail
print(f'||B - B_k||_F = {tail:.4f} (relative difference to the bound '
      f'{rel:.1e})')
assert rel < 1e-8
report(svd_vecs, 'SVD-PPMI')
```

<!-- Placeholder (stage 2): GloVe is weighted least squares on the log counts, w^T w~ + b + b~ ~ log X, over the nonzero counts only (its one difference: the zeros are never seen); AdaGrad as in the paper, 50 passes for fewer than 300 dimensions; present it as the paper's recipe, not tuned (there is no development split for word similarity, and the recipe was fixed before the grid). -->

```{.python .input #word-embeddings-glove}
%%tab pytorch
GLOVE = dict(epochs=50, lr=0.05, batch_size=65536, x_max=100)  # the paper's

def train_glove(X, epochs, lr, batch_size, x_max, seed=0, verbose=True):
    """GloVe on the nonzero entries of X; returns W + W~ on the CPU."""
    torch.manual_seed(seed)
    gen = torch.Generator(device).manual_seed(seed)
    C = X.tocoo()
    rows, cols = (torch.tensor(a, dtype=torch.long, device=device)
                  for a in (C.row, C.col))
    log_x = torch.tensor(np.log(C.data), dtype=torch.float32, device=device)
    h = torch.tensor(np.minimum((C.data / x_max) ** 0.75, 1),
                     dtype=torch.float32, device=device)
    W, Wt = (((torch.rand(V, DIM, device=device) - 0.5) / DIM)
             .requires_grad_() for _ in range(2))
    b, bt = (torch.zeros(V, device=device, requires_grad=True)
             for _ in range(2))
    # The paper's AdaGrad: squared gradients summed from 1, so a small
    # gradient makes a small step
    opt = torch.optim.Adagrad([W, Wt, b, bt], lr=lr,
                              initial_accumulator_value=1.0)
    for epoch in range(epochs):
        total = 0.0
        perm = torch.randperm(len(log_x), generator=gen, device=device)
        for i in range(0, len(perm), batch_size):
            j = perm[i:i + batch_size]
            r, c = rows[j], cols[j]
            err = (W[r] * Wt[c]).sum(1) + b[r] + bt[c] - log_x[j]
            loss = 0.5 * (h[j] * err ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}: loss {total / len(log_x):.4f}')
    return (W + Wt).detach().cpu()

tic = time.time()
glove_vecs = train_glove(X, **GLOVE)
print(f'{time.time() - tic:.0f} s')
report(glove_vecs, 'GloVe')
```

<!-- Placeholder (stage 2): the trained SGNS scores against the shifted PMI of the 2,000 most frequent pairs, computed from the counts SGNS is shown: windows over the subsampled stream (which reach past deleted words) with a window size drawn per center (the average of the fixed-window counts); at 100 dimensions the relation is approximate. -->

```{.python .input #word-embeddings-scatter}
%%tab pytorch
# The counts SGNS is shown: windows over the subsampled stream (they reach
# past the deleted words), and a window size drawn from 1..WINDOW, which
# averages the fixed-window counts; three subsampling draws
rng = np.random.default_rng(0)
kept = [ids[keep_mask(ids, counts, T_SUB, rng)] for _ in range(3)]
X_seen = (sum(d2l.cooccurrence_counts(k, V, b) for k in kept
              for b in range(1, WINDOW + 1)) / (len(kept) * WINDOW)).tocoo()
top = np.argsort(-X_seen.data)[:2000]  # the most frequent (w, o) pairs
w, o = X_seen.row[top], X_seen.col[top]
n_word = np.asarray(X_seen.sum(1)).ravel()
n_ctx, N = np.asarray(X_seen.sum(0)).ravel(), X_seen.sum()
P = counts ** ALPHA / (counts ** ALPHA).sum()  # the noise of training
pmi_top = np.log(X_seen.data[top] * N / (n_word[w] * n_ctx[o]))
target = pmi_top - np.log(K) + np.log(n_ctx[o] / N / P[o])
score = (sgns_v[w] * sgns_u[o]).sum(1).numpy()
print(f'Pearson r = {np.corrcoef(target, score)[0, 1]:.2f} over {len(top)} '
      f'pairs; mean score - target {np.mean(score - target):+.2f}')
lim = [min(target.min(), score.min()), max(target.max(), score.max())]
d2l.plot([target, lim], [score, lim],
         r'$\mathrm{pmi}(w, o) - \log K + \log(p(o) / P(o))$',
         r'SGNS score $\mathbf{u}_o^\top \mathbf{v}_w$', fmts=('.', 'k--'),
         figsize=(3.5, 3.5))
```

<!-- Placeholder (stage 2), ## Evaluating Word Vectors: three methods, the noise exponent and subsampling, three seeds each; the grid is regenerated with D2L_RERUN=we-grid and read from the registered file otherwise, after checking that it was made with the current settings; the agreement line matches only on the device that made the grid (other devices draw other random numbers). -->

```{.python .input #word-embeddings-grid}
%%tab pytorch
def grid_row(method, E, seconds, **settings):
    """One run of the grid: its settings, rho and interval per rating set,
    and the cosines of the covered pairs (for paired comparisons)."""
    row = dict(method=method, seconds=round(seconds, 1), **settings)
    for name, p in SETS.items():
        r = d2l.spearman_eval(E, vocab, p)
        row[name] = dict(rho=r.rho, ci=r.ci, n=r.n_covered,
                         cosines=np.round(r.cosines, 6).tolist())
    return row

def write_json(path, records):
    with open(path, 'w') as f:
        json.dump(records, f)

ALPHAS = (0.0, 0.75, 1.0)  # SGNS noise exponents; SVD-PPMI takes 1 and ALPHA
CONFIG = json.loads(json.dumps(dict(  # everything the grid's numbers rest on
    corpus=hashlib.sha1(ids.astype(np.int64).tobytes()).hexdigest(), V=V,
    window=WINDOW, dim=DIM, K=K, t_sub=T_SUB, alpha=ALPHA, alphas=ALPHAS,
    epochs=EPOCHS, batch=BATCH, lr=LR, seeds=SEEDS, glove=GLOVE,
    sets={name: dict(sha1=d2l.DATA_HUB[SOURCES[name]][1], n=covered[name])
          for name in SETS})))

if 'we-grid' in RERUN:
    rows = []
    for seed in SEEDS:
        for alpha in ALPHAS:
            for sub in (True, False):
                tic = time.time()
                E, _ = train_sgns(ids, counts, alpha, sub, seed, verbose=False)
                rows.append(grid_row('SGNS', E, time.time() - tic,
                                     alpha=alpha, subsample=sub, seed=seed))
        tic = time.time()
        E = train_glove(X, **GLOVE, seed=seed, verbose=False)
        rows.append(grid_row('GloVe', E, time.time() - tic, alpha=None,
                             subsample=False, seed=seed))
    for alpha in (1.0, ALPHA):  # deterministic: one run each
        tic = time.time()
        E = svd_embed(d2l.ppmi(X, alpha=alpha), DIM)
        rows.append(grid_row('SVD-PPMI', E, time.time() - tic, alpha=alpha,
                             subsample=False, seed=0))
    d2l.lm_save_rerun(ARTIFACTS['we-grid'][1], lambda path: write_json(
        path, dict(config=CONFIG, runs=rows)))
key, file = ARTIFACTS['we-grid']
with open(d2l.lm_artifact(key, file if 'we-grid' in RERUN else None)) as f:
    grid = json.load(f)
assert grid['config'] == CONFIG, 'the grid was made with other settings'
grid = grid['runs']

def setting(r):
    return r['method'] + ('' if r['method'] == 'GloVe' else
                          f", alpha={r['alpha']:g}") + (
        ', subsampled' if r['subsample'] else '')

def runs(method, **settings):  # the grid's runs of one setting, by seed
    return sorted((r for r in grid if r['method'] == method and all(
        r[k] == v for k, v in settings.items())), key=lambda r: r['seed'])

summary = {}
for r in grid:
    if r['seed'] == 0:
        same = runs(r['method'], alpha=r['alpha'], subsample=r['subsample'])
        summary[setting(r)] = {}
        for name in SETS:
            rho = [s[name]['rho'] for s in same]
            summary[setting(r)][name] = (
                f"{r[name]['rho']:.3f} [{r[name]['ci'][0]:.2f}, "
                f"{r[name]['ci'][1]:.2f}]"
                + (f'; seeds [{min(rho):.3f}, {max(rho):.3f}]'
                   if len(same) > 1 else ''))
for name in SETS:  # one table per benchmark: the page's width
    print(pd.DataFrame(summary).T[[name]].to_string())
default = runs('SGNS', alpha=ALPHA, subsample=True)[0]
live = {name: d2l.spearman_eval(sgns_v, vocab, p, n_boot=0).rho
        for name, p in SETS.items()}
# The same on the device that made the grid; another device draws other
# random numbers, so the two then differ by a few hundredths
print('this run vs the grid (SGNS defaults, seed 0): ' + '; '.join(
    f"{n} {live[n]:.3f} vs {default[n]['rho']:.3f}" for n in SETS))
```

<!-- Placeholder (stage 2): left, each method at its default settings: seed 0 with its pair-bootstrap interval, the other seeds as crosses; right, rho against the noise exponent with and without subsampling, the band spanning the seeds. -->

```{.python .input #word-embeddings-grid-plot}
%%tab pytorch
methods = {'SGNS': runs('SGNS', alpha=ALPHA, subsample=True),
           'SVD-PPMI': runs('SVD-PPMI', alpha=ALPHA), 'GloVe': runs('GloVe')}
series, seeds = {}, {}
for k, name in enumerate(SETS):  # seed 0 and its interval; all seeds
    first = [rs[0][name] for rs in methods.values()]
    series[name] = ([d['rho'] for d in first], [d['ci'][0] for d in first],
                    [d['ci'][1] for d in first])
    xs, ys = zip(*[(i + 0.15 * (2 * k - 1), r[name]['rho'])
                   for i, rs in enumerate(methods.values()) for r in rs])
    seeds[name] = (xs, ys)
fig, axes = d2l.plt.subplots(1, 2, figsize=(7.5, 3))
d2l.plot_intervals(list(methods), series, 'method', r'Spearman $\rho$',
                   axes=axes[0], marks={'seeds': ('x', seeds)})
bands = {}
for name in SETS:  # mean over seeds; the band spans them
    for sub in (True, False):
        rho = np.array([[r[name]['rho'] for r in runs(
            'SGNS', alpha=a, subsample=sub)] for a in ALPHAS])
        bands[f"{name}{'' if sub else ', no subsampling'}"] = (
            rho.mean(1), rho.min(1), rho.max(1))
d2l.plot_intervals(ALPHAS, bands, r'noise exponent $\alpha$',
                   r'Spearman $\rho$', kind='band', axes=axes[1])
for ax in axes:
    ax.legend(fontsize='small')
d2l.plt.tight_layout()
```

<!-- Placeholder (stage 2): paired differences of rho: both members of a comparison are evaluated on the same resampled pairs; the seed range says how much a difference moves from run to run. -->

```{.python .input #word-embeddings-grid-compare}
%%tab pytorch
def rank_corr(a, b):  # Spearman along the last axis, as in spearman_eval
    from scipy.stats import rankdata
    a, b = rankdata(a, axis=-1), rankdata(b, axis=-1)
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))

sgns = runs('SGNS', alpha=ALPHA, subsample=True)
COMPARE = {  # name: (kind, runs A, runs B); a single run pairs every seed
    'SGNS - SVD-PPMI': ('method', sgns, runs('SVD-PPMI', alpha=ALPHA)),
    'SGNS - GloVe': ('method', sgns, runs('GloVe')),
    'SVD-PPMI - GloVe': ('method', runs('SVD-PPMI', alpha=ALPHA),
                         runs('GloVe')),
    'alpha 0.75 - 0': ('setting', sgns, runs('SGNS', alpha=0.0,
                                             subsample=True)),
    'alpha 1 - 0.75': ('setting', runs('SGNS', alpha=1.0, subsample=True),
                       sgns),
    'SVD alpha 0.75 - 1': ('setting', runs('SVD-PPMI', alpha=ALPHA),
                           runs('SVD-PPMI', alpha=1.0)),
    'subsampling on - off': ('setting', sgns, runs('SGNS', alpha=ALPHA,
                                                   subsample=False))}
rows = []
for label, (kind, A, B) in COMPARE.items():
    n = max(len(A), len(B))  # 3 seeds, or 1 (SVD) repeated
    pairs_ab = list(zip(A * (n // len(A)), B * (n // len(B))))
    for name, p in SETS.items():
        idx = vocab.token_to_idx
        human = p[[w1 in idx and w2 in idx for w1, w2 in
                   zip(p.word1, p.word2)]].score.to_numpy()
        a, b = (np.array(r[name]['cosines']) for r in (A[0], B[0]))
        assert len(a) == len(b) == len(human)  # the same pairs, in order
        boot = np.random.default_rng(0).integers(0, len(human),
                                                 (2000, len(human)))
        diff = rank_corr(a[boot], human[boot]) - rank_corr(b[boot],
                                                           human[boot])
        by_seed = [x[name]['rho'] - y[name]['rho'] for x, y in pairs_ab]
        rows.append(dict(comparison=label, kind=kind, set=name,
                         delta=rank_corr(a, human) - rank_corr(b, human),
                         ci_lo=np.quantile(diff, 0.025),
                         ci_hi=np.quantile(diff, 0.975),
                         seed_lo=min(by_seed), seed_hi=max(by_seed)))
deltas = pd.DataFrame(rows)
print(deltas.round(3).to_string(index=False))
for kind in ('method', 'setting'):
    d = deltas[deltas.kind == kind]
    big = d.loc[d.delta.abs().idxmax()]
    print(f'largest |delta rho| by {kind}: {abs(big.delta):.2f} '
          f'({big.comparison}, {big.set})')
```

<!-- Placeholder (stage 2): nearest neighbours by cosine, under the three tables trained here. -->

```{.python .input #word-embeddings-neighbors-def}
%%tab pytorch
#@save
def nearest_neighbors(E, vocab, word, k=5, exclude_self=True):
    """The k rows of E nearest to word's row in cosine, as a list of
    (token, cosine). vocab is a d2l.Vocab or a TokenEmbedding (whose
    unknown token at index 0 is never returned), or a Hugging Face
    tokenizer, for which word must be a single token."""
    if hasattr(vocab, 'token_to_idx'):
        i, skip = vocab.token_to_idx[word], [0]
        name = lambda j: vocab.idx_to_token[j]
    else:
        ids = vocab.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'{word!r} is {len(ids)} tokens, not one')
        i, skip, name = ids[0], [], lambda j: vocab.decode([j])
    Z = F.normalize(torch.as_tensor(E)[:len(vocab)].float(), dim=1)
    cos = Z @ Z[i]
    cos[skip + ([i] if exclude_self else [])] = -math.inf
    top = torch.topk(cos, k)
    return [(name(int(j)), float(c)) for c, j in zip(top.values, top.indices)]
```

```{.python .input #word-embeddings-neighbors}
%%tab pytorch
tables = {'SGNS': sgns_v, 'SVD-PPMI': svd_vecs, 'GloVe': glove_vecs}
for w in ('doctor', 'chip', 'stock', 'bank'):  # a line per method
    print(w + ''.join(f'\n  {m:9s} ' + ', '.join(
        t for t, _ in d2l.nearest_neighbors(E, vocab, w))
        for m, E in tables.items()))
```

<!-- Placeholder (stage 2), analogies and bias: pretrained GloVe vectors (6 billion tokens, 400,000 words) through the interface of the earlier edition. -->

```{.python .input #word-embeddings-token-embedding}
%%tab pytorch
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

<!-- Placeholder (stage 2): 3CosAdd, argmax over x of cos(x, b - a + c), with and without the three query words excluded (Linzen 2016). -->

```{.python .input #word-embeddings-analogy}
%%tab pytorch
check_sha1(d2l.download('glove.6b.100d'), 'glove.6b.100d')
glove6b = d2l.TokenEmbedding('glove.6b.100d')
Z6b = F.normalize(glove6b.idx_to_vec.float(), dim=1)

def analogy(a, b, c, exclude=True):
    """3CosAdd: the word x maximizing cos(x, b - a + c), for a : b :: c : x."""
    i = [glove6b.token_to_idx[w] for w in (a, b, c)]
    scores = Z6b @ (Z6b[i[1]] - Z6b[i[0]] + Z6b[i[2]])
    scores[[0] + (i if exclude else [])] = -math.inf  # 0 is <unk>
    return glove6b.idx_to_token[int(scores.argmax())]

QUERIES = [('man', 'woman', 'king'), ('paris', 'france', 'tokyo'),
           ('bad', 'worst', 'big'), ('walk', 'walked', 'go'),
           ('do', 'did', 'see')]
n_input = 0
for a, b, c in QUERIES:
    x, y = analogy(a, b, c), analogy(a, b, c, exclude=False)
    n_input += y in (a, b, c)
    print(f'{a}:{b} :: {c}:?  excluded -> {x:<8} not excluded -> {y}')
print(f'{n_input} of {len(QUERIES)} answers are a query word when the '
      'query words are not excluded')
```

<!-- Placeholder (stage 2): bias describes the corpus; removing the projection on a he - she direction zeroes that coordinate and leaves the neighbourhoods nearly as they were (Gonen and Goldberg 2019). -->

```{.python .input #word-embeddings-bias}
%%tab pytorch
PROFESSIONS = ['nurse', 'receptionist', 'librarian', 'housekeeper',
               'hairdresser', 'nanny', 'secretary', 'homemaker', 'dancer',
               'teacher', 'engineer', 'carpenter', 'mechanic', 'architect',
               'programmer', 'surgeon', 'captain', 'philosopher',
               'scientist', 'manager']
g = F.normalize(Z6b[glove6b.token_to_idx['he']]
                - Z6b[glove6b.token_to_idx['she']], dim=0)
Z_hidden = F.normalize(Z6b - (Z6b @ g)[:, None] * g, dim=1)  # all words
lean = Z6b @ g  # every word's original projection on he - she
rows = []
for w in PROFESSIONS:
    near = [[glove6b.token_to_idx[t] for t, _ in
             d2l.nearest_neighbors(Z, glove6b, w, k=10)]
            for Z in (Z6b, Z_hidden)]
    i = glove6b.token_to_idx[w]
    rows.append(dict(word=w, before=float(lean[i]),
                     after=float(Z_hidden[i] @ g),
                     overlap=len(set(near[0]) & set(near[1])),
                     # the original lean of its neighbours after removal
                     neighbours=float(lean[near[1]].mean())))
bias = pd.DataFrame(rows).sort_values('before')
print(bias.round(3).to_string(index=False))
assert bias.after.abs().max() < 1e-6  # zero by construction
print(f'median overlap of the top 10 neighbours: '
      f'{bias.overlap.median():.0f} of 10\ncorrelation of a profession\'s '
      f'original projection with its neighbours\' after removal: '
      f'{np.corrcoef(bias.before, bias.neighbours)[0, 1]:.2f}')
```

<!-- Placeholder (stage 2), ## Static Vectors and a Language Model's Table: the input table of Qwen3-0.6B, one row per token, shared with the output layer; one vector per word type, so "bank" is one point for the river and the money. -->

```{.python .input #word-embeddings-qwen-table}
%%tab pytorch
QWEN = 'Qwen/Qwen3-0.6B'
model = d2l.load_qwen3(QWEN, device=device)
tok = transformers.AutoTokenizer.from_pretrained(QWEN,
                                                 revision=model.revision)
table = model.token_emb.weight.detach().float()
n_params = sum(p.numel() for p in model.parameters())
_, _, files = d2l._qwen3_files(QWEN, model.revision)
n_file = 0
for path in files:  # what the checkpoint stores, from the file headers
    with safe_open(path, 'pt') as f:
        n_file += sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
print(f'table {tuple(table.shape)}: {table.numel():,} of {n_params:,} '
      f'parameters ({table.numel() / n_params:.1%})\n  the checkpoint file '
      f'holds {n_file:,} numbers')
tied = (model.hf_config['tie_word_embeddings']
        and not hasattr(model, 'lm_head'))
print(f'tied: {tied}; the output layer multiplies by the same table')
assert tied and abs(table.numel() / n_params - 0.261) < 5e-4
words = ['doctor', 'bank', 'river', 'money']
near = lambda E, v, w: ', '.join(repr(t) for t, _ in
                                 d2l.nearest_neighbors(E, v, w))
for w in words:  # a line per table
    print(f'{w}\n  Qwen3-0.6B table   {near(table, tok, " " + w)}\n'
          f'  SGNS (this slice)  {near(sgns_v, vocab, w)}')
index = {'SGNS': (sgns_v, vocab.token_to_idx),
         'GloVe-6B': (Z6b, glove6b.token_to_idx),
         'Qwen3 table': (table, {w: tok.encode(' ' + w)[0] for w in words})}
for o in ('river', 'money'):  # one vector for both senses of "bank"
    print(f'cos(bank, {o}): ' + ', '.join(
        f'{m} {F.cosine_similarity(E[ix["bank"]], E[ix[o]], dim=0):.2f}'
        for m, (E, ix) in index.items()))
del model, table
d2l.lm_free()
print(f'notebook: {time.time() - TIC:.0f} s')
```

<!-- Placeholder (stage 2), ### Embeddings for Categorical Variables: closing paragraph, no cell (entity embeddings, Guo and Berkhahn 2016). Then Summary and Exercises. -->
