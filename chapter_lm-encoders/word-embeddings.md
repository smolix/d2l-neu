# Word Embeddings
:label:`sec_lm_word_embeddings`

The language model of :numref:`sec_gpt` begins with a table that holds one
row of numbers for every token in its vocabulary. Gradient descent filled
that table along with everything else, and we never asked what its rows
mean. Here is a question we could have asked. *Physician* and *doctor* are
different tokens, and nothing in the training said that they are
interchangeable. Do their rows end up close anyway?

Before training there is no reason why they should. Looking up a row is
the same as multiplying the table by a *one-hot* vector, all zeros except
for a one at the word's index. One-hot vectors carry no similarity at all:
any two different words are orthogonal, and *doctor* is as far from
*physician* as from *banana*. Whatever similarity the rows hold, training
put there. In the 1950s the linguists Zellig Harris and J. R. Firth
proposed where it could come from: words that occur in similar contexts
tend to mean similar things, or in Firth's phrase, "you shall know a word
by the company it keeps". Physicians and doctors keep the same company.
Both examine patients and write prescriptions.

The question matters beyond word lists. Later in this chapter, whether
nearby vectors mean related things decides whether a search finds the
passage that answers a question, and word vectors are the smallest case
of that idea. This section asks which statistic of a corpus a learned word
vector encodes, and whether we can check the answer. We train the
skip-gram model of word2vec
:cite:`Mikolov.Chen.Corrado.ea.2013,Mikolov.Sutskever.Chen.ea.2013`, which
made word vectors cheap enough to train on billions of words, show that
at its optimum its scores factorize a matrix of co-occurrence statistics
:cite:`Levy.Goldberg.2014`, and check the claim on a toy corpus and on a
slice of Wikipedia. Two methods that factorize the counts directly, and
two sets of human similarity judgments, tell us how much the choice of
method matters. The section ends with what no such vector can do: give
one word two meanings.

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
The configuration cell sets the hyperparameters that every method
shares, a window of five words on each side and vectors of 100
dimensions, and skip-gram's $K = 5$ noise words, whose role we meet with
the model.

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
## Co-occurrence Counts and Human Ratings
:label:`subsec_lm_cooccurrence`

The idea that a word is known by its company is the *distributional
hypothesis*. To use it we make "company" countable: we slide a window over
a corpus and count how often each word occurs near each other word. To
judge the vectors we will compute from such counts, we also need a
yardstick before any vector exists, and the standard one is people's
judgments of which words are alike.

### A Slice of WikiText-103

Our corpus is the beginning of WikiText-103
:cite:`Merity.Xiong.Bradbury.ea.2016`, a collection of Wikipedia articles
that the site's editors rated good or featured. `d2l.WikiText103` reads the
same file as bytes in :numref:`sec_adam`; here we read it as words. The
registry cell pins the file, the two sets of similarity ratings introduced
below, and the stored results of the 23 training runs compared in
:numref:`subsec_lm_we_eval`.

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
A second registry cell points to pretrained word vectors on the book's
mirror (GloVe, introduced below), which we need only for the analogy and
bias tests at the end of the section.

```{.python .input #word-embeddings-glove-hub}
%%tab pytorch
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')
```
We drop headings, lowercase the text, split it at whitespace and keep the
tokens made of letters, allowing one apostrophe as in *'s*. A line end
becomes a break that no window crosses. Words seen fewer than ten times
leave the stream, as in word2vec, because their rows would be trained on
almost nothing.

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
Three million tokens use 85,598 different words, yet the vocabulary of
words seen ten times or more has only 17,997 entries, and the words left
out make up only 5.4% of the tokens. A frequency cutoff loses most word
types and few tokens. This is Zipf's law
(:numref:`subsec_natural-lang-stat`) at work: the frequency of a word
falls roughly as a power of its rank, so a few words make up most of the
text and most words are rare.

### The Count Matrix

Let $X_{wo}$ be the number of times that word $o$ occurs within $W = 5$
positions of word $w$. We call $w$ the *center* word and $o$ a *context*
word. Every pair is counted from both ends, so $\mathbf{X}$ is symmetric.
`cooccurrence_counts` builds it as a sparse matrix, one sparse addition per
distance $d = 1, \ldots, W$, and leaves out pairs that straddle a break.

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
Before running the next cell, guess which pairs have the largest counts.

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
The matrix has 6.5 million nonzero entries, 2.00% of its 324 million
cells, holding 27 million (word, context) pairs. The largest counts pair
*the* with itself and with *of*, *in*, *and* and *to*. Raw counts mostly
measure how frequent the two words are, not whether they belong together.

Pointwise mutual information corrects for this (:numref:`subsec_mdl-pmi`).
Write $p(w, o) = X_{wo} / \sum_{w', o'} X_{w'o'}$ for the share of all
pairs that are $(w, o)$, and $p(w)$ and $p(o)$ for its row and column sums.
Then

$$\textrm{pmi}(w, o) = \log \frac{p(w, o)}{p(w)\, p(o)}$$

compares how often the pair occurs with how often it would occur if the
two words were placed independently. It is positive for pairs that
attract and negative for pairs that repel. By this measure *the* and *of*,
the second-largest count, are only weakly associated, while *prime* and
*minister*, which meet far less often, attract strongly. For the 98% of
pairs that never occur, the pmi is $-\infty$, which no finite score can
match.

### Human Similarity Ratings

*WordSim353* (Finkelstein et al., 2002) holds 353 word pairs rated by
people, and its
ratings mix relatedness with similarity: *cup* and *coffee* score high.
*SimLex-999* (Hill, Reichart and Korhonen, 2015) holds 999 pairs rated for
similarity alone. *Clothes* and *closet*, which WordSim353 rates as
closely related, score low there, since a closet is not a kind of
clothing. For a table of vectors we compute the cosine of each pair and
the Spearman correlation $\rho$ between the cosines and the ratings. A
rank correlation uses only the order of the two lists, so the scales of
cosines and ratings do not matter. `spearman_eval` also resamples the
pairs with replacement 2,000 times and reports the central 95% of the
resampled correlations, a *pair bootstrap* interval.

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
A pair with a word outside the vocabulary cannot be scored, and dropping
it shrinks the test set without warning, so we count the pairs that
remain.

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
The slice covers 295 of the 353 WordSim353 pairs and 865 of the 999
SimLex-999 pairs. A correlation computed under another vocabulary is
therefore a correlation on another test set.

What do random vectors score? Zero on average, of course. For $n$ pairs
without association a rank correlation has a standard deviation of about
$1/\sqrt{n - 1}$: 0.058 for 295 pairs and 0.034 for 865. Twenty random
tables should spread over about twice these amounts on either side of
zero.

```{.python .input #word-embeddings-floor}
%%tab pytorch
floor = {name: [d2l.spearman_eval(
    torch.randn(V, DIM, generator=torch.Generator().manual_seed(s)), vocab,
    p, n_boot=0).rho for s in range(20)] for name, p in SETS.items()}
for name, r in floor.items():
    print(f'{name}: random vectors, rho mean {np.mean(r):+.3f}, '
          f'range [{min(r):+.3f}, {max(r):+.3f}] over 20 seeds')
```
On WordSim353 they do, from $-0.10$ to $+0.13$. On SimLex-999 they stay
somewhat narrower, from $-0.05$ to $+0.04$. A method has to clear this
floor before its correlation means anything; on WordSim353, a correlation
of 0.1 means nothing.

## Skip-Gram with Negative Sampling
:label:`subsec_negative-sampling`

Skip-gram is the GPT of :numref:`sec_gpt` with everything between the
token table and the output layer removed, and asked to predict the words
around a word instead of the next one. It keeps two tables with one row
per word. The row $\mathbf{v}_w$ of $\mathbf{V}$, the *center vector*, is
used when $w$ sits in the middle of a window, and the row $\mathbf{u}_o$
of $\mathbf{U}$, the *context vector*, when $o$ is one of the neighbors.
The model scores a pair by the inner product $s(w, o) =
\mathbf{u}_o^\top \mathbf{v}_w$. Unlike that GPT, skip-gram does not tie
its two tables. With one table, a word's score with itself would be
$\|\mathbf{v}_w\|^2$, which is never negative, yet most words of our slice
never occur within five positions of themselves.

The obvious predictor is a softmax over the vocabulary, $p(o \mid w) =
\exp(s(w, o)) / \sum_{o'} \exp(s(w, o'))$, whose normalizer needs a score
for every row: 17,997 for every pair in our slice, millions for real
vocabularies. A GPT pays this cost too, but there it is one cost among
many layers. Skip-gram has no layers, so the normalizer would be nearly
all of its work, and word2vec was written to run over billions of words
on ordinary processors. *Negative sampling* replaces the one large
classification by a few binary ones :cite:`Mikolov.Sutskever.Chen.ea.2013`.
The model learns to tell the observed context word $o$ from $K$ noise
words $o_1, \ldots, o_K$ drawn from a distribution $P$. For one observed
pair the loss is

$$\ell(w, o) = -\log \sigma\big(\mathbf{u}_o^\top \mathbf{v}_w\big) - \sum_{k=1}^{K} \log \sigma\big(-\mathbf{u}_{o_k}^\top \mathbf{v}_w\big),$$
:eqlabel:`eq_lm_sgns_loss`

where $\sigma(s) = 1/(1 + e^{-s})$ is the sigmoid. The first term pushes
the score of the real pair up, and the sum pushes the scores of the noise
pairs down. A pair now costs $1 + K = 6$ inner products instead of 17,997.
We call each scored pair of a center word with a context or noise word a
*slot*. `sgns_loss` computes :eqref:`eq_lm_sgns_loss` as a binary
cross-entropy and divides by the number of slots, $1 + K$ per observed
pair, so it reports a loss per slot. At the start, when every score is
zero, a slot costs $\log 2 \approx 0.693$.

Two relatives change one thing each. *CBOW*, the continuous bag of words,
reverses the prediction: the mean of the context vectors predicts the
center word :cite:`Mikolov.Chen.Corrado.ea.2013`. *fastText* makes a word's
vector the sum of the vectors of its character $n$-grams, so a word never
seen in training still gets a vector from its pieces
:cite:`Bojanowski.Grave.Joulin.ea.2017`.

<!-- FIGURE pending (generator step, not in this commit): fig_lm_sgns_pmi,
img/mdl-lm-encoders-sgns-pmi.svg. A window of 5 around a center word in a
short sentence; the center table V and the context table U with the scored
pairs; three noise words drawn from count^0.75; a small PMI matrix with the
shifted entries that the optimum reproduces. Insert it here with a caption
and :label:`fig_lm_sgns_pmi`, and cite it in the paragraph above. -->

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
### The word2vec Recipe
:label:`subsec_lm_word2vec_recipe`

Three choices of the word2vec recipe decide which pairs the loss sees. The
first is *subsampling*. Pairs with *the* dominate the counts, say little
about either word, and would take most of the training time. So each
occurrence of a word that makes up a share $f(w)$ of the text is
discarded with probability

$$\max\left(1 - \sqrt{t / f(w)},\, 0\right), \quad t = 10^{-4}.$$

Words rarer than $t$ are always kept, and a word that makes up 1% of the
text is kept with probability $\sqrt{10^{-4}/10^{-2}} = 0.1$. The decision
is made per occurrence, so a frequent word stays in the vocabulary and
loses most of its occurrences. The windows over the thinned stream then
reach further.

The second is the noise distribution. Drawn by frequency, a rare word
would almost never serve as a noise word, and little would push its
scores down. Instead, $o$ is drawn with probability $P_\alpha(o) \propto
n(o)^\alpha$, where $n(o)$ counts the occurrences of $o$ and $\alpha =
0.75$. With $\alpha = 1$ noise words are drawn by frequency, with $\alpha
= 0$ uniformly. The power in between flattens the frequencies: a word 100
times as frequent as another is drawn only about 32 times as often.

The third is the window. Near neighbors say more about a word than distant
ones, so word2vec draws a width $b$ uniformly from $1, \ldots, W$ for each
center word and uses the context words within $b$ positions. A word at
distance $d$ is then used with probability $(W - d + 1)/W$, so the
distances 1 to 5 carry the weights 1, 0.8, 0.6, 0.4 and 0.2.

Centers have different numbers of context words, so a minibatch gives each
center $2W = 10$ context slots, marks the empty ones in a mask, and draws
$K$ noise words per slot. A masked slot and its noise words must not enter
the loss, or the model learns to score padding. How many centers does one
epoch have after subsampling?

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
About half: 1,411,837 of the 2,838,873 word positions survive as centers,
with 5.7 context words each on average, so subsampling halves the work of
an epoch. A minibatch is a block of 1,024 rows of 60 entries: 10 context
slots, then 5 noise words for each slot. In the printed row the drawn
window was two words wide, so the mask keeps four context words and their
20 noise words, 24 of 60 entries. Its center, *bl*, is what the cleaning
leaves of road names such as "BL I-69" in an article about a Michigan
freeway.

### Training

Two tables of 17,997 rows and 100 columns hold $\mathbf{V}$ and
$\mathbf{U}$, initialized as in word2vec, and Adam updates both. Every
epoch draws a fresh subsample, fresh windows and fresh noise words. We
evaluate the center table $\mathbf{V}$ and, for comparison, $\mathbf{V} +
\mathbf{U}$.

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
The loss per slot, averaged over the epoch, falls from 0.457 in the first
to 0.375 in the fifth, at about three seconds per epoch on one GPU. The
center vectors correlate with the ratings at $\rho = 0.56$ on WordSim353
(interval 0.48 to 0.64) and 0.18 on SimLex-999 (0.11 to 0.25), far above
the random floor on both. SimLex-999 is the harder test because a window
records which words appear together, which is relatedness, while
SimLex-999 asks for similarity. *Hot* and *cold* share their contexts and
are not similar. The sum $\mathbf{V} + \mathbf{U}$ scores a little higher,
0.58 and 0.20. Papers differ in which table they report, so we fix
$\mathbf{V}$ throughout; whether the difference is real is Exercise
:exref:`ex_lm_we-tables`.

## Factorizing Co-occurrence Statistics
:label:`subsec_lm_sgns_pmi`

What did the loss teach the scores? Suppose the vectors had enough
dimensions for every score $s(w, o)$ to be chosen freely. Then the loss,
averaged over the corpus, separates into one logistic regression per pair:
the pair $(w, o)$ is a positive example $X_{wo}$ times and a noise pair
about $K\, n_w\, P(o)$ times, where $n_w = \sum_o X_{wo}$ counts the
observed pairs with center $w$. A single free score $s$ that meets $a$
positives and $b$ negatives incurs the loss $-a \log \sigma(s) - b \log
\sigma(-s)$. Its derivative vanishes where $\sigma(s) = a/(a + b)$, that is
at $s = \log(a/b)$: the best score is the log of the ratio of the two
counts. Here that ratio is $X_{wo} / (K n_w P(o))$, and since $X_{wo}/n_w
= p(w, o)/p(w)$, its logarithm splits into three terms. The optimum is

$$s^*(w, o) = \log \frac{X_{wo}}{K\, n_w\, P(o)} = \textrm{pmi}(w, o) - \log K + \log \frac{p(o)}{P(o)}.$$
:eqlabel:`eq_lm_sgns_optimum`

At its optimum, skip-gram scores a pair by its pointwise mutual
information, shifted down by $\log K$. With noise drawn by frequency the
last term vanishes, and the matrix of scores $\mathbf{V}\mathbf{U}^\top$
factorizes the shifted PMI matrix
:cite:`Levy.Goldberg.2014`. A full proof is Exercise
:exref:`ex_mdl-mutual-information-negative-sampling-pmi` of
:numref:`sec_mdl-mi-estimation`. With $\alpha = 0.75$ the last term is an
offset that depends on the context word alone. A rare word is drawn as
noise more often than its frequency, $P(o) > p(o)$, so the offset lowers
its scores. This counteracts the best-known weakness of PMI, which gives
two rare words that meet once a very large value.

The result holds only at the optimum with free scores. A table of 100
columns cannot represent an arbitrary matrix of 17,997 rows, and for the
98% of pairs with no count the optimum is $-\infty$. We first check the
result where it must hold exactly, on a corpus small enough to have free
scores. Then we compute two factorizations of the counts directly, and
finally measure how close our trained model came.

### A Toy Corpus

The toy corpus has 40 tokens of four words, *ice*, *steam*, *solid* and
*gas*. With a window of one word all 16 pairs occur. The cell minimizes
the expected loss over one free score per pair by L-BFGS in float64, with
noise by frequency and with $\alpha = 0.75$. In the first case each score
should be the pmi of its pair minus $\log 5 \approx 1.609$.

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
Both predictions hold to the precision of the optimizer. With $\alpha = 1$
the scores equal $\textrm{pmi} - \log K$ to $10^{-7}$. With $\alpha = 0.75$
they miss that target by 0.05 and match :eqref:`eq_lm_sgns_optimum`,
offset included, to $4 \times 10^{-8}$; the offset is small because the
four words have similar frequencies. The largest pmi can be read off the
printed counts. *Steam* and *gas* occur together 13 times among 78 pairs,
and their rows sum to 24 and 20, so their pmi is $\log(13 \cdot 78 / (24
\cdot 20)) \approx 0.75$. That is below $\log 5$, and so every optimal
score is negative; the largest, $-0.81$, belongs to this pair.

### Positive PMI and the Truncated SVD

If skip-gram approximates a known matrix, why not compute the matrix and
factorize it directly? The pmi of a pair that never occurs is $-\infty$,
and almost all pairs never occur. The standard remedy is the *positive
PMI*, $\max(\textrm{pmi}(w, o), 0)$, which keeps every missing pair at zero
and the matrix sparse. It also drops the negative values, which say only
that a pair is rarer than chance and which a small corpus estimates
poorly. `ppmi` raises the context distribution to the power
$\alpha$ before computing the pmi, the counterpart of the smoothed noise,
and we use $\alpha = 0.75$ as for skip-gram.

The best factorization of a given rank is known. By the Eckart--Young
theorem (:numref:`subsec_mdl-eckart-young`), the truncated SVD that keeps
the $k$ largest singular values is the best rank-$k$ approximation in the
Frobenius norm, and its error is the square root of the sum of the
squared singular values left out. As word vectors we take the left
singular vectors scaled by the square roots of their singular values,
splitting each singular value evenly between words and contexts, as
$\mathbf{V}$ and $\mathbf{U}$ split the scores.

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
The cell computes the positive PMI of the slice and its rank-100 SVD, and
checks the error formula on the block of the 2,000 most frequent words,
where a full SVD is cheap.

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
Of the 6.5 million nonzero counts, 5.5 million have a positive pmi, and
the SVD takes seconds. The residual matches the formula to a relative
$1.7 \times 10^{-15}$, which is rounding error. The SVD vectors correlate
with the ratings at 0.57 on WordSim353 (interval 0.48 to 0.65) and 0.18 on
SimLex-999 (0.12 to 0.25), close to skip-gram on both.

### GloVe

GloVe :cite:`Pennington.Socher.Manning.2014` fits the logarithm of the
counts directly, by weighted least squares. In our notation it minimizes

$$\sum_{w, o:\, X_{wo} > 0} h(X_{wo}) \big(\mathbf{u}_o^\top \mathbf{v}_w + b_w + c_o - \log X_{wo}\big)^2, \qquad h(x) = \min\big((x / x_{\max})^{0.75},\, 1\big),$$
:eqlabel:`eq_lm_glove`

with biases $b_w$ and $c_o$ and $x_{\max} = 100$. The weight $h$ lets
frequent pairs count more, up to a cap. The biases absorb the logarithms
of the two words' counts, so the inner product again ends up close to a
pmi, up to a constant. GloVe differs from the other two methods in
several ways: it fits log counts with learned biases, it has no
counterpart of the 0.75 power, and, most visibly, its sum runs over the
nonzero counts only. A pair that never occurs never enters its loss,
whereas skip-gram meets such pairs among its noise words and the SVD fits
the zeros of the positive PMI. We use the paper's optimizer
settings (AdaGrad at a learning rate of 0.05, 50 passes, and the sum of
the two tables as word vectors) on the same window counts as the other
methods, with an update per minibatch rather than per count. We do not
tune it: word similarity has no development split, and tuning on the
rated pairs would be tuning on the answers.

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
The loss falls from 0.0113 after 10 passes to 0.0056 after 50. The vectors
correlate with the ratings at 0.29 on WordSim353 (interval 0.18 to 0.40)
and 0.14 on SimLex-999 (0.08 to 0.20), well below the other two methods on
WordSim353.

### Trained Scores and Shifted PMI

The toy corpus confirmed the optimum where every score is free. How close
did our skip-gram model on the slice come to :eqref:`eq_lm_sgns_optimum`?
We compare its scores $\mathbf{u}_o^\top \mathbf{v}_w$ with the right-hand
side for the 2,000 most frequent pairs. The right-hand side must use the
counts that skip-gram was shown, not the raw window counts, and the two
differ in two ways. Subsampling deleted most occurrences of frequent
words, so windows reach past them. The drawn widths weight distance $d$
by $(W - d + 1)/W$, which is the same as averaging the counts of the fixed
windows of width 1 to 5. The cell rebuilds those counts over three
subsampling draws. At the free-score optimum every point would lie on the
dashed identity line.

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
The points follow the line with a Pearson correlation of about 0.9 and a
mean offset of 0.01. The cloud is narrow where the shifted pmi is negative
and fans out among the strongly associated pairs, where 100 dimensions
force a compromise. "Skip-gram is PMI factorization" is a statement about
an optimum; a model trained for five epochs is near it for frequent pairs,
not at it.

## Evaluating Word Vectors
:label:`subsec_lm_we_eval`

We have three tables from one slice: skip-gram with negative sampling
(SGNS), the SVD of positive PMI (SVD-PPMI) and GloVe. Which is best? Levy,
Goldberg and Dagan (2015) argued that settings such as the noise exponent,
which one method's recipe fixed and another's left out, move the results
as much as the choice of method. We test this on our slice.

### Methods and Settings

The grid trains SGNS with noise exponents $\alpha \in \{0, 0.75, 1\}$, with
and without subsampling, three seeds each; GloVe with three seeds; and
SVD-PPMI with context exponents 1 and 0.75, which needs no seed. Each run
stores $\rho$ with its interval and the cosine of every covered pair. The
cell reads the stored runs, checks that they were made with the current
settings, and prints the default SGNS run beside the one trained above;
setting `D2L_RERUN=we-grid` regenerates all 23 runs in about ten minutes
on one GPU.

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
Every method clears the random floor on both sets: the lowest lower
bounds, GloVe's 0.18 on WordSim353 and 0.08 on SimLex-999, lie above the
largest random correlations, 0.13 and 0.04. On WordSim353, SGNS ranges from
0.52 to 0.56 over its six settings, SVD-PPMI scores 0.57 to 0.59, and
GloVe 0.29; on SimLex-999 all runs lie between 0.13 and 0.21. The plot
shows each method at its default settings on the left, with the interval
of seed 0 and the other seeds as crosses, and $\rho$ against the noise
exponent on the right, with bands that span the seeds.

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
bands, styles = {}, []
for k, name in enumerate(SETS):  # mean over seeds; the band spans them
    for sub in (True, False):
        rho = np.array([[r[name]['rho'] for r in runs(
            'SGNS', alpha=a, subsample=sub)] for a in ALPHAS])
        bands[f"{name}{'' if sub else ', no subsampling'}"] = (
            rho.mean(1), rho.min(1), rho.max(1))
        styles.append((f'C{k}', '-' if sub else '--'))
d2l.plot_intervals(ALPHAS, bands, r'noise exponent $\alpha$',
                   r'Spearman $\rho$', kind='band', axes=axes[1])
for line, band, (color, style) in zip(axes[1].lines, axes[1].collections,
                                      styles):  # hue: rating set, as left
    line.set(color=color, linestyle=style)
    band.set_color(color)
for ax in axes:
    ax.legend(fontsize='small')
d2l.plt.tight_layout()
```
The intervals on the left are wide and overlap, but unpaired intervals
overstate the uncertainty of a difference. Both members of a comparison
are scored on the same pairs, and a pair that is hard for one table tends
to be hard for the other. As in :numref:`sec_lm_eval_suites`, we therefore
compare the two on the same items. A rank correlation is not a mean of
per-item scores, so instead of resampling differences, the next cell
resamples the pairs and recomputes both correlations on each resample. The
seed range says how much a difference moves from run to run.

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
SGNS and SVD-PPMI are indistinguishable. Their difference is $-0.01$ on
WordSim353 (interval $-0.08$ to 0.05) and $-0.003$ on SimLex-999 ($-0.04$
to 0.04), as :eqref:`eq_lm_sgns_optimum` predicts for factorizations of
closely related matrices. On this slice both exceed GloVe by about 0.3 on
WordSim353; the difference from SGNS is 0.27 (interval 0.17 to 0.38, seeds
0.27 to 0.30). On SimLex-999 no difference between methods is detectable.

The claim we set out to test therefore holds for the two factorizations
of PMI and fails for GloVe. On this slice the method moved $\rho$ by up to
0.3, the noise exponent and subsampling by up to 0.05, but the 0.3 is
GloVe's gap alone. The noise exponent, in contrast, moves SGNS
detectably: on WordSim353, 0.75 beats both 0 and 1 by 0.01 to 0.04 in
every seed. Subsampling changes $\rho$ by an amount whose sign flips
between seeds, so its benefit here is speed.

Why does GloVe trail? We did not tune it, and its most visible
difference, the missing zeros, is a likely cause. On three million words,
98% of the count matrix is zero, and GloVe is the only method that never
sees the zeros, so nothing in its loss keeps apart two words that never
meet. The two departures from the paper's procedure are not the cause:
much smaller minibatches and counts weighted by the inverse of their
distance did not close the gap. Exercise :exref:`ex_lm_we-glove-corpus`
tests the explanation.

For a corpus of your own, SVD-PPMI is a good first choice: it is
deterministic, takes seconds and matched SGNS on both sets. Its context
exponent made no detectable difference here. SGNS streams over the text
and scales to corpora whose count matrix would not fit in memory; with
SGNS, use the 0.75 power.

### Nearest Neighbors

A correlation summarizes a thousand pairs. The nearest neighbors of a
word, the rows with the largest cosine to its row, show what a table
learned about one word.

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
Does *doctor* land next to *physician*, as the opening hoped?

```{.python .input #word-embeddings-neighbors}
%%tab pytorch
tables = {'SGNS': sgns_v, 'SVD-PPMI': svd_vecs, 'GloVe': glove_vecs}
for w in ('doctor', 'chip', 'stock', 'bank'):  # a line per method
    print(w + ''.join(f'\n  {m:9s} ' + ', '.join(
        t for t, _ in d2l.nearest_neighbors(E, vocab, w))
        for m, E in tables.items()))
```
It does not, although *physician* occurs often enough in the slice to
have a row of its own. On this slice, a *doctor* is mostly the Doctor of
*Doctor Who* (Amy, Torchwood, the TARDIS) or a character in *The X-Files*
(Scully, Mulder, Skinner). *Stock* sits among financing and profits for
SGNS and among film studios for SVD-PPMI, and *bank* among Swiss banks for
all three methods. *Chip* occurs too rarely, and in too many senses, for
any method: its lists look like noise. GloVe's lists are the noisiest
overall, in line with its correlations. Nearest neighbors describe the
corpus, not the language, and the articles of this slice include a good
deal of television.

### Analogies

Word vectors became famous for analogies: *king* $-$ *man* $+$ *woman*
lies close to *queen*. Analogy tests are usually run on vectors trained on
billions of words, so we use the pretrained GloVe-6B vectors, which
GloVe's authors trained on six billion tokens of Wikipedia and newswire.
`d2l.TokenEmbedding` loads them.

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
For an analogy $a : b :: c : x$, the method called 3CosAdd answers with the
word $x$ whose unit vector has the largest cosine with $\mathbf{b} -
\mathbf{a} + \mathbf{c}$, all three normalized. If the offset $\mathbf{b}
- \mathbf{a}$ is small compared with $\mathbf{c}$, the nearest word is $c$
itself, so the standard evaluation excludes the query words from the
answers. We run five analogies with and without that exclusion.

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
With the query words excluded, all five answers are right: *queen*,
*japan*, *biggest*, *went* and *saw*. Without the exclusion, 2 of the 5
answers are a query word: *king* for *man* : *woman* :: *king*, and *see*
for *do* : *did* :: *see*. The famous analogy holds because the method
forbids the answer *king*, as Linzen (2016) pointed out. An analogy
accuracy means little unless it says whether the inputs were excluded.

### Bias

Word vectors record the associations of their text, including its
stereotypes (Bolukbasi et al., 2016). A simple probe is a direction: the
unit vector $\mathbf{g}$ along $\mathbf{he} - \mathbf{she}$, and the
projection of each word's unit vector onto it. A simplified version of
the repair that Bolukbasi et al. proposed removes the projection onto
$\mathbf{g}$ from every word, so that no word leans either way.

Gonen and Goldberg (2019) argued that this hides the bias rather than
removing it, and we can test the argument. After the removal, we find each
profession's ten nearest neighbors and average their *original*
projections. If the removal had erased the association, the new neighbors
of a profession would lean toward *she* or *he* no more than chance
allows. If it only hid it, their lean would still follow the profession's
own original lean.

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
Before the removal, *homemaker* ($-0.37$), *housekeeper* ($-0.32$) and
*nurse* ($-0.28$) lean toward *she*, and *manager* ($+0.28$) and *captain*
($+0.21$) toward *he*. These numbers describe how the corpus of GloVe-6B
uses the words. *Secretary* leans toward *he* ($+0.18$), probably because
in newswire a secretary is often a Secretary of State. After the removal
every projection is zero by
construction, and the neighborhoods barely change: the median profession
keeps 9 of its 10 nearest neighbors, and *homemaker*, the word with the
strongest lean, keeps 5. The new neighbors still lean the way the word
did, with a correlation of about 0.9 over the 20 professions. The
association is spread over many directions, and removing one of them hides
it from one probe.

## Static Vectors and a Language Model's Table
:label:`subsec_lm_static_vs_table`

We return to the table that opened the section. Qwen3-0.6B
:cite:`Yang.Li.Yang.ea.2025`, loaded with `d2l.load_qwen3` as in
:numref:`sec_lm_d2l_checkpoint`, has a token table of 151,936 rows of
width 1,024. Which share of its 596 million parameters does the table
hold? The cell also prints the nearest neighbors of four words in the
table and under SGNS, and the cosines of *bank* with *river* and *money*.

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

The table holds 155,582,464 of the 596,049,920 parameters, 26.1%: a
quarter of the model is a lookup table. It is tied, so the output layer
multiplies by the same matrix, and the checkpoint file, which stores it
for the input and again for the output, holds 751,632,384 numbers, more
than the model has.

Nothing in such a table requires the categories to be words. A lookup is
a one-hot vector times the table, so an embedding table is a linear layer
on one-hot inputs, and any categorical variable with many values can have
one: store numbers, product codes, postal codes or user ids. Guo and
Berkhahn (2016) trained such *entity embeddings* with a network that
predicts from tabular data, and categories that behave alike ended up
with nearby rows, as words with similar company do. For a store, one row
per value is what we want, since a store is the same store wherever it
appears.

The neighbors in Qwen's table are of a different kind. *Doctor* sits next
to *doctors*, *Doctor*, the Chinese word for doctor, and itself without
the leading space; *bank*, *river* and *money* sit next to their other
spellings and capitalizations, and the first two next to their Chinese
translations. The table has two jobs. It hands the token to the layers
above, which work out its meaning from the context. Because it is tied,
it also scores the token as the next one, and *doctor* and *Doctor* are
predicted in the same places, so their rows end up close.

This settles the question we started with, as far as nearest neighbors
can: neither in the three tables we trained nor in the language model's
is *physician* among the five nearest rows of *doctor*. On our slice
*doctor* keeps the company of *Doctor Who*, and in the language model's
table the company of its own spellings.

The cosines of *bank* show what every table of this section shares. Under
SGNS on our slice, *bank* is closer to *river* (0.53) than to *money*
(0.32), although its nearest neighbors are Swiss banks. In GloVe-6B the
order is reversed (0.33 and 0.57), and in Qwen's table neither cosine is
large (0.24 and 0.22). Each table places *bank* at one point, and its
corpus decides where that point lies between the bank of a river and the
bank that keeps our money. A vector computed from the sentence around the
word would not have to choose, and :numref:`sec_lm_bert` computes one.

## Summary

A word vector encodes how often words occur together, compared with how
often they would by chance. At its optimum with free scores, skip-gram
with negative sampling scores a pair by its pointwise mutual information
shifted down by $\log K$, plus an offset set by the noise distribution. A
toy corpus confirmed this to the precision of the optimizer, and on a
slice of Wikipedia the trained scores followed it with a correlation of
about 0.9. The truncated SVD of the positive PMI matrix computes such a
factorization in closed form and matched skip-gram on human similarity
ratings, while GloVe, which never sees the pairs that do not occur,
trailed by about 0.3 on this small corpus. A fair comparison of word
vectors states the covered pairs, the random floor and paired
differences, and the analogy and bias tests need the same care. Every
static table, word2vec's or a language model's, gives *bank* one vector
for both of its senses; vectors computed from context, and search built
on them, come next.

## Exercises

1. **Subsampling and the optimum.** Each occurrence of a word $w$ of
   relative frequency $f(w) > t$ is kept with probability $r_w =
   \sqrt{t/f(w)}$; rarer words have $r_w = 1$.
    1. Show that the expected number of kept occurrences of $w$ in a corpus
       of $N$ tokens is $N\sqrt{t f(w)}$. Subsampling thus replaces a
       frequency by the geometric mean of the frequency and $t$. What does
       it do to the ratio of the counts of two frequent words?
    1. Ignore that windows reach further after deletions, so that a pair
       survives with probability $r_w r_o$, and keep the noise distribution
       $P$ computed from the full counts, as the code does. Show that the
       free-score optimum on the subsampled text is
       $s^*(w, o) + \log r_o - \log \bar r_w$, where $\bar r_w = \sum_{o'}
       r_{o'} X_{wo'} / \sum_{o'} X_{wo'}$. Why does the keep rate of the
       center word cancel?
    1. Which scores does subsampling lower the most, and how does its
       effect compare with that of the noise exponent in
       :eqref:`eq_lm_sgns_optimum`?
    1. A pair at distance 6 in the original text can enter the counts that
       skip-gram sees. Explain how, and say which pairs gain most from it.

1. [code] **The optimum for other numbers of noise words.** Rerun the toy
   check of :numref:`subsec_lm_sgns_pmi` with $K = 1$, 5 and 15.
    1. Before running, predict from the printed counts which optimal
       scores are positive for $K = 1$ with noise drawn by frequency, and
       why. Then check your prediction, and check that the optimum
       follows :eqref:`eq_lm_sgns_optimum` for every $K$.
    1. Delete the pair *steam*–*gas* from the counts, setting both of its
       entries to zero, and run the optimizer again. What happens to its
       score as the iterations proceed, and how do positive PMI and GloVe
       each avoid the problem?
   :exlabel:`ex_lm_we-toy-k`

1. [code] **Analogy accuracy and width.** ● Download the Google analogy set
   `questions-words.txt` (Apache-2.0) and train SGNS on the slice with
   `DIM` $\in \{25, 50, 100, 300\}$, three seeds each.
    1. Report the number of covered questions and the 3CosAdd accuracy with
       and without excluding the query words, with the seed range. At
       which width does the difference between the two protocols shrink?
    1. For the questions that fail only without the exclusion, which query
       word is returned? Relate the answer to the length of $\mathbf{b} -
       \mathbf{a}$ compared with that of $\mathbf{c}$.

1. [code] **Center table, context table, or their sum.** Evaluate
   $\mathbf{U}$, $\mathbf{V}$ and $\mathbf{V} + \mathbf{U}$ of the default
   run on both rating sets, with paired intervals for the differences
   computed as in :numref:`subsec_lm_we_eval`.
    1. Which differences can you detect?
    1. Explain why the choice matters for skip-gram and does not arise for
       the tied table of Qwen3-0.6B.
    1. With $\mathbf{V} + \mathbf{U}$, the cosine of two words contains
       the terms $\mathbf{v}_w^\top \mathbf{u}_{w'}$. Use
       :eqref:`eq_lm_sgns_optimum` to say what these terms measure.
   :exlabel:`ex_lm_we-tables`

1. [code] **CBOW.** Implement CBOW: the mean of the context vectors of a
   window predicts the center word, with $K$ noise words per window drawn
   from the same distribution. Train it with the same budget of epochs and
   the same noise distribution, and compare $\rho$ on both rating sets, with
   paired intervals against SGNS, and the seconds per epoch. Which updates
   does CBOW perform fewer of, and what does that predict for rare words?

1. [code] **GloVe and the size of the corpus.** ● The explanation offered
   for GloVe's gap in :numref:`subsec_lm_we_eval` makes a prediction. For
   the first two parts, set `N_WORDS` to 30 million and skip the cells of
   the grid, whose stored results belong to the three-million-word slice.
    1. Measure the share of zero entries in the count matrix of the most
       frequent 18,000 words.
    1. Train GloVe with the same settings and compute SVD-PPMI on the
       larger slice, and compare their $\rho$ on WordSim353 with a paired
       interval. Does the gap shrink as the share of zeros falls?
    1. Train GloVe once more on the three-million-word slice with ten
       times as many passes. Does a lower loss raise $\rho$? What does
       the answer say about the cause of the gap?
   :exlabel:`ex_lm_we-glove-corpus`

1. [extended] **An embedding for a categorical column.** ●● Take the house
   prices of :numref:`sec_kaggle_house` and one categorical column with
   many values, such as the neighborhood.
    1. Show that a table of width $d$ followed by a linear layer is a
       linear function of the one-hot vector of rank at most $d$. Under a
       linear model, when can it differ from one-hot features at all?
    1. Fit the linear model of :numref:`sec_kaggle_house` with one-hot
       features and with tables of width $d \in \{2, 4, 32\}$, and then
       with a small MLP on top of each. Compare the $k$-fold validation
       errors of that section over three seeds.
    1. Plot the rows of the width-2 table, and say whether neighborhoods
       with similar prices lie close together.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §19.1]{.kicker}

Word Embeddings<br>**co-occurrence counts · skip-gram with negative sampling · shifted PMI · SVD and GloVe · evaluation · a language model's table**
:::
:::

::: {.slide title="One-Hot Vectors Make Every Pair of Words Orthogonal"}
[The problem]{.kicker}

- A vocabulary of 17,997 entries: each word is a vector of length 17,997
  with a single 1.
- cos(*doctor*, *physician*) = cos(*doctor*, *banana*) = 0.
- A language model's token table: one-hot vector times the table = one
  row per token. Nothing in its training says which words are alike.

::: {.d2l-note .warn}
Which statistic of a text does a learned row encode, and can we check it?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- count co-occurrences and turn them into pointwise mutual information;
- train skip-gram with negative sampling (SGNS) and explain its recipe;
- state what SGNS converges to, and check it exactly on a toy corpus;
- compute word vectors in closed form with an SVD;
- compare word vectors fairly: coverage, intervals, paired differences;
- say what a static vector cannot represent.

**Running example:** the first 3 million words of WikiText-103 (Wikipedia),
a vocabulary of the words that occur at least 10 times (17,997 entries),
windows of 5, vectors of 100 dimensions, $K = 5$ noise words.
:::

::: {.slide title="Word Vectors Open the Chapter's First Movement"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| movement | section |
|---|---|
| vectors | **Word Embeddings** |
| | Pretrained Encoders |
| | Text Embeddings from Contrastive Training |
| finding | Sparse, Dense and Approximate Retrieval |
| using what is found | Retrieval-Augmented Generation |
:::

::: {.col .narrow}
- **Uses:** the tied token table of a GPT; the truncated SVD.
- **Skips:** hierarchical softmax, phrase vectors, vectors for many
  languages.
:::
:::
:::

::: {.slide title="Raw Counts Measure Frequency; PMI Measures Association"}
[Co-occurrence counts]{.kicker}

$X_{wo}$: how often word $o$ occurs within 5 positions of word $w$.

- 6.5 million nonzero counts, 2% of the 324 million cells.
- Largest counts: *the*–*the* 141,058, *the*–*of* 116,432, *the*–*in*
  69,654.

$$\textrm{pmi}(w, o) = \log \frac{p(w, o)}{p(w)\, p(o)}$$

$p(w, o)$: share of all pairs; $p(w)$, $p(o)$: its marginals. Positive when
a pair occurs more often than independence predicts: *the*–*of* is only
weakly associated, *prime*–*minister* strongly.

::: {.d2l-note}
98% of the pairs never occur: their pmi is $-\infty$.
:::
:::

::: {.slide title="Human Ratings Judge Word Vectors; Random Vectors Reach 0.13"}
[Human similarity ratings]{.kicker}

| set | pairs rated by people for | covered by the slice |
|---|---|---|
| WordSim353 | relatedness and similarity: *cup*–*coffee* high | 295 of 353 |
| SimLex-999 | similarity alone: *clothes*–*closet* low | 865 of 999 |

- $\rho$: Spearman correlation between the cosines of the pairs and the
  ratings; 95% interval from resampling the pairs.
- Random vectors, 20 seeds: $-0.10$ to $0.13$ and $-0.05$ to $0.04$.

::: {.d2l-note}
A method must clear this floor before its $\rho$ means anything.
:::

[Finkelstein et al., 2002; Hill, Reichart and Korhonen, 2015]{.small}
:::

::: {.slide title="Negative Sampling Turns a 17,997-Way Softmax into Six Binary Decisions"}
[Skip-gram with negative sampling]{.kicker}

Center vector $\mathbf{v}_w$ (table $\mathbf{V}$), context vector
$\mathbf{u}_o$ (table $\mathbf{U}$), score $\mathbf{u}_o^\top
\mathbf{v}_w$; noise words $o_1, \ldots, o_K \sim P$; sigmoid $\sigma(s) =
1/(1 + e^{-s})$:

$$\ell(w, o) = -\log \sigma(\mathbf{u}_o^\top \mathbf{v}_w) - \sum_{k=1}^{K} \log \sigma(-\mathbf{u}_{o_k}^\top \mathbf{v}_w)$$

- A GPT with no layers, predicting the neighbors of a word.
- The softmax over the vocabulary needs 17,997 scores per pair. This loss
  needs $1 + K = 6$.

::: {.d2l-note}
Two tables: with one, a word's score with itself, $\|\mathbf{v}_w\|^2$,
could never be negative, yet most words never occur near themselves.
:::
:::

::: {.slide title="Three Choices Decide Which Pairs the Loss Sees"}
[The word2vec recipe]{.kicker}

| choice | rule | why |
|---|---|---|
| subsampling | drop each occurrence with prob. $\max(1 - \sqrt{t/f(w)}, 0)$, $t = 10^{-4}$ | pairs with *the* teach little, cost most |
| noise | $P(o) \propto n(o)^{0.75}$ | rare words drawn more often than their frequency |
| window | width $b \sim \{1, \ldots, 5\}$ per center | near words count more: distance $d$ weighted $(6 - d)/5$ |

$f(w)$: share of the text that is $w$; $n(o)$: occurrences of $o$.

- A minibatch row: 10 context slots, 5 noise words per slot, 60 entries.
- Empty slots and their noise words are masked out of the loss.
:::

::: {.slide title="Check: How Strong Are Subsampling and the 0.75 Power?"}
[Check]{.kicker}

A word makes up 1% of the text. With what probability is an occurrence
kept? A word is 100 times more frequent than another. How much more often
is it drawn as noise?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Kept with $\sqrt{10^{-4}/10^{-2}} = 0.1$. Drawn
$100^{0.75} \approx 32$ times as often, not 100. On the slice, subsampling
leaves 1.41 of 2.84 million positions as centers.
:::
:::
:::

::: {.slide title="Five Epochs Give Vectors That Correlate 0.56 with Human Ratings"}
[Training]{.kicker}

::: {.cols .vc}
::: {.col}
@!word-embeddings-sgns-train
:::

::: {.col .narrow}
- Loss per slot (one scored pair, real or noise); it starts at $\log 2 =
  0.693$.
- $\rho$ on WordSim353: 0.56 (interval 0.48 to 0.64); on SimLex-999: 0.18.
- Random vectors reach at most 0.13 and 0.04.
- SimLex-999 is harder: windows record relatedness, not similarity.
:::
:::
:::

::: {.slide title="At the Free-Score Optimum, SGNS Scores Are Shifted PMI"}
[The optimum]{.kicker}

Each pair becomes its own logistic regression: $X_{wo}$ positives,
$K\, n_w\, P(o)$ noise draws, with $n_w = \sum_o X_{wo}$. A free score
meeting $a$ positives and $b$ negatives is best at $s = \log(a/b)$:

$$s^*(w, o) = \log \frac{X_{wo}}{K\, n_w\, P(o)} = \textrm{pmi}(w, o) - \log K + \log \frac{p(o)}{P(o)}$$

- Noise by frequency ($P = p$): the score matrix $\mathbf{V}\mathbf{U}^\top$
  factorizes $\textrm{pmi} - \log K$.
- Power 0.75: rare context words get lower scores, against PMI's bias
  toward rare pairs.

::: {.d2l-note .warn}
Only with free scores: 100 columns cannot hold every pmi, and no finite
score reaches $-\infty$.
:::

[Levy and Goldberg, 2014]{.small}
:::

::: {.slide title="A Four-Word Corpus Confirms the Optimum to 10⁻⁷"}
[A toy corpus]{.kicker}

*ice, steam, solid, gas*: 40 tokens, window 1, $K = 5$, one free score per
pair, L-BFGS in float64.

| noise | max error against $\textrm{pmi} - \log K$ | against $s^*$ with the offset |
|---|---|---|
| $\alpha = 1$ | $1.0 \times 10^{-7}$ | $1.0 \times 10^{-7}$ |
| $\alpha = 0.75$ | 0.05 | $4.3 \times 10^{-8}$ |

- Largest pmi: *steam*–*gas*, $\log(13 \cdot 78 / (24 \cdot 20)) \approx
  0.75$, below $\log 5 = 1.61$.
- So every optimal score is negative; the largest, $-0.81$, is this pair's.
:::

::: {.slide title="The SVD of Positive PMI Is the Closed-Form Counterpart"}
[Factorizing the counts]{.kicker}

- Positive PMI: $\max(\textrm{pmi}, 0)$. Missing pairs stay 0, so the
  matrix stays sparse: 5.5 million positive entries.
- Rank-100 truncated SVD, in seconds. Word vectors: left singular vectors
  scaled by $\sqrt{\sigma_i}$.
- Eckart–Young: the best rank-$k$ approximation, with residual
  $\sqrt{\sum_{i > k} \sigma_i^2}$ (checked to a relative $1.7 \times
  10^{-15}$).

| | WordSim353 | SimLex-999 |
|---|---|---|
| SGNS | 0.56 | 0.18 |
| SVD-PPMI | 0.57 | 0.18 |
:::

::: {.slide title="GloVe Fits Log Counts, but Only Where a Count Is Nonzero"}
[GloVe]{.kicker}

$$\sum_{X_{wo} > 0} h(X_{wo}) \big(\mathbf{u}_o^\top \mathbf{v}_w + b_w + c_o - \log X_{wo}\big)^2$$

- $h(x) = \min((x/100)^{0.75}, 1)$: frequent pairs count more, up to a cap.
- Biases $b_w$, $c_o$ absorb the log counts of $w$ and $o$: the inner
  product is again close to a pmi, up to a constant.
- The paper's optimizer settings on our counts, untuned: $\rho = 0.29$ and
  0.14.

::: {.d2l-note}
A pair that never occurs never enters the loss: 98% of the pairs on this
slice.
:::
:::

::: {.slide title="Trained SGNS Scores Follow Shifted PMI, r ≈ 0.9"}
::: {.cols .vc}
::: {.col .narrow}
@!word-embeddings-scatter
:::

::: {.col}
- The 2,000 most frequent pairs; dashed: the free-score optimum.
- Counts as SGNS saw them: the subsampled stream, and drawn window widths,
  which average the fixed windows of width 1 to 5.
- Tight below 0, wider among strongly associated pairs: 100 dimensions
  force a compromise.
:::
:::
:::

::: {.slide title="SGNS and SVD-PPMI Agree; GloVe Trails on WordSim353"}
[Evaluation]{.kicker}

::: {.cols .vc}
::: {.col .fig .big}
@!word-embeddings-grid-plot
:::

::: {.col}
- Left: each method at its defaults, seed 0 with its pair-bootstrap
  interval; crosses: the other seeds.
- Right: SGNS against the noise exponent, bands over 3 seeds.
- Every lower bound clears the random floor (0.13 and 0.04).
:::
:::
:::

::: {.slide title="Check: Is SGNS Better than SVD-PPMI?"}
[Check]{.kicker}

On WordSim353, SGNS scores 0.56 and SVD-PPMI 0.57 on the same 295 pairs.
Unpaired intervals: 0.48 to 0.64 and 0.48 to 0.65. Is either better?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Score both on the same resampled pairs: the difference is
$-0.01$, interval $-0.08$ to 0.05. No difference is detectable. Report
"alike", not an order.
:::
:::
:::

::: {.slide title="The Method Moved ρ by up to 0.3, the Settings by up to 0.05"}
[Paired differences]{.kicker}

| comparison (WordSim353) | $\Delta\rho$ | interval | seeds |
|---|---|---|---|
| SGNS − SVD-PPMI | −0.01 | −0.08 to 0.05 | −0.02 to −0.01 |
| SGNS − GloVe | 0.27 | 0.17 to 0.38 | 0.27 to 0.30 |
| noise 0.75 − 0 | 0.04 | 0.003 to 0.07 | 0.01 to 0.04 |
| noise 1 − 0.75 | −0.04 | −0.06 to −0.01 | −0.04 to −0.02 |
| subsampling on − off | 0.03 | −0.02 to 0.08 | −0.01 to 0.04 |

- SimLex-999: no difference between methods is detectable.
- GloVe never sees the 98% of pairs that do not occur.

::: {.d2l-note .rule}
SVD-PPMI: deterministic, seconds, as good as SGNS here. SGNS: use the 0.75
power, which beat 0 and 1 in every seed on WordSim353.
:::
:::

::: {.slide title="Neighbors Describe the Corpus: On Wikipedia a Doctor Travels in Time"}
[Nearest neighbors]{.kicker}

| word | SGNS | SVD-PPMI |
|---|---|---|
| doctor | vincent, amy, torchwood, tardis, utterson | scully, skinner, mulder, jackman, amy |
| stock | enterprises, purchasing, financing, bankrupt, profits | pictures, shops, paramount, share, goldwyn |
| bank | ubs, winterthur, asset, swiss, banks | ubs, winterthur, merger, suisse, banks |

::: {.d2l-note}
*Physician* has a row but is not among them: the slice's doctors come from
*Doctor Who* and *The X-Files*.
:::
:::

::: {.slide title="3CosAdd Finds Queen Only Because King Is Excluded"}
[Analogies]{.kicker}

$a : b :: c : x$ answered by $\arg\max_x \cos(\mathbf{x}, \mathbf{b} -
\mathbf{a} + \mathbf{c})$, on GloVe-6B, 100 dimensions:

@!word-embeddings-analogy

::: {.d2l-note .rule}
State whether the query words were excluded. Without it, the nearest word
to $\mathbf{b} - \mathbf{a} + \mathbf{c}$ is often $c$ (Linzen, 2016).
:::
:::

::: {.slide title="Removing the He–She Direction Hides the Bias from One Probe"}
[Bias]{.kicker}

Projection of 20 professions on the unit vector along
$\mathbf{he} - \mathbf{she}$ in GloVe-6B, then that projection removed:

| | homemaker | nurse | captain | manager |
|---|---|---|---|---|
| projection before | −0.37 | −0.28 | 0.21 | 0.28 |
| top-10 neighbors kept | 5 | 9 | 8 | 8 |

- After the removal every projection is 0; the median profession keeps 9
  of 10 neighbors.
- The new neighbors' original lean tracks the word's: correlation about 0.9.

[Bolukbasi et al., 2016; Gonen and Goldberg, 2019]{.small}
:::

::: {.slide title="Check: How Much of Qwen3-0.6B Is Its Token Table?"}
[Check]{.kicker}

Qwen3-0.6B: 151,936 tokens of width 1,024, 596,049,920 parameters, output
layer tied to the table. What share is the table? How many numbers does a
file hold that stores the table for input and output separately?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $151{,}936 \times 1{,}024 = 155{,}582{,}464$, or 26.1%. The
file holds $596{,}049{,}920 + 155{,}582{,}464 = 751{,}632{,}384$.
:::
:::
:::

::: {.slide title="A Language Model's Table Groups Spellings, and Bank Stays One Point"}
[Static vectors]{.kicker}

| word | nearest rows in Qwen3-0.6B's table |
|---|---|
| doctor | ' doctors', ' Doctor', '医生', 'Doctor', 'doctor' |
| bank | ' Bank', 'bank', 'Bank', '银行', '_bank' |

| cosine | SGNS (slice) | GloVe-6B | Qwen3 table |
|---|---|---|---|
| *bank*, *river* | 0.53 | 0.33 | 0.24 |
| *bank*, *money* | 0.32 | 0.57 | 0.22 |

::: {.d2l-note}
One row per word: the corpus decides where *bank* sits between its two
senses.
:::
:::

::: {.slide title="A Word Vector Is a Factorization of Co-occurrence Counts"}
[Recap]{.kicker}

- Counts: 6.5 million nonzero, 2% of the matrix. PMI separates association
  from frequency.
- SGNS: $1 + K$ binary decisions per pair. Its free-score optimum is
  $\textrm{pmi} - \log K$ plus a noise offset: exact to $10^{-7}$ on a toy
  corpus, $r \approx 0.9$ on the slice.
- SVD-PPMI matches SGNS (0.57 and 0.56 on WordSim353). GloVe trails by
  about 0.3 there; the settings moved $\rho$ by up to 0.05.
- Analogies need the query words excluded; removing a direction hides a
  bias from one probe.
- A static vector is one point per word, in a word2vec table and in a
  language model's.

**Next:** an encoder that gives each token a vector computed from its
sentence.
:::
