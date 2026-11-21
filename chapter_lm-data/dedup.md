# Deduplication and Decontamination
:label:`sec_lm_dedup`

<!-- Rights rule P1 (for maintainers): no cell prints text of our own crawl
extraction (c0, c1, c2). Displayed text is FineWeb's (ODC-By) from the audit
stream and its same-host documents, or benchmark items under their licences;
everything else about the crawl is counts, hashes and ids. -->

The filters of :numref:`sec_lm_quality_filtering` judge one page at a time.
They can tell a manual from an advertisement for love spells, but they cannot
notice that they have just kept the same manual for the third time. The web is
full of such repeats. Mirror sites copy whole pages, blogs republish a post
with a paragraph added, and shop software prints the same notice about card
payments under thousands of products. A model trained on a corpus with repeats
reads those texts again and again, and what a model reads often it learns to
reproduce word for word. The repeats also use compute that new text would have
used better.

Exact copies are easy to find: hash every document and look for equal hashes.
The repeats that matter are rarely exact. The cell below prints two posts from
one music blog in FineWeb's output, the second of which is the first with one
paragraph added, and their SHA-256 hashes have nothing in common. To catch
such a pair we need a measure of similarity, and comparing every pair of
documents is out of reach: the cell times about eight microseconds per pair,
which is half an hour for the 20,000 documents of our stream and about $10^5$
years for a billion.

The way out goes back to Andrei Broder, who in 1997 needed to find
near-duplicate pages in the index of the AltaVista search engine. Give every
document a short *signature* such that two signatures agree in any one
position with probability equal to the similarity of the documents. Cut the
signatures into pieces and use the pieces as keys of a hash table, and similar
documents meet in a bucket without being compared with the rest. How the
pieces are cut sets a threshold, which we can tune on labelled pairs like that
of any classifier. This section builds the signature, checks it against its
derivation and tunes it, then adds two methods that see what document
signatures cannot: long spans shared by otherwise different documents, and
documents that say the same thing in other words. It closes on the duplicates
that matter most, test items that leaked into training data, and on the last
three stages of the manifest that :numref:`sec_lm_crawl` began.

```{.python .input #dedup-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #dedup-extras}
%%tab pytorch
import collections
import difflib
import gzip
import hashlib
import io
import itertools
import json
import os
import pathlib
import re
import resource
import sys
import textwrap
import time
import unicodedata
import pandas as pd
from datasketch import MinHash, MinHashLSH
from scipy import stats
```

Everything in this notebook runs on the CPU, with four worker processes, in a
few minutes and about 7 GiB of memory. Two inputs were made once on a GPU and
are read from registered files: the paraphrased test items, written by
Qwen3.5-4B, and the embeddings of the test items and of windows of FineWeb
documents, made by Qwen3-Embedding-0.6B. The labelled pairs, the decisions for
our crawl, the manifest and the section's numbers are registered as well, so
that a reader's run can be compared with ours. Setting `D2L_RERUN` to names
from `ARTIFACTS` regenerates those files under new names. `dedup-plants` and
`dedup-emb` need a GPU, and `dedup-plants` runs alone, or with `dedup-pairs`,
because the detectors read the plants before the cell that makes them. The
last cell compares whatever this run computed with the registered results.

```{.python .input #dedup-config}
%%tab pytorch
K = 128            # MinHash rows per signature
TAU = 0.8          # a pair is a near-duplicate when its Jaccard is >= TAU
R_GRID = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16)  # rows; b = K // r
SEEDS = (0, 1, 2)  # hash seeds 0xD2 + s (seed 0 is the specification's)
N_GRAM = 13        # word n-grams of the decontamination index (GPT-3)
N_WORKERS = min(4, os.cpu_count())  # each spawned worker imports d2l
PARAPHRASER = 'Qwen/Qwen3.5-4B'      # writes the paraphrased plants (GPU)
EMBEDDER = 'Qwen/Qwen3-Embedding-0.6B'
GPU = torch.cuda.is_available()
ARTIFACTS = {  # name: (registered key, file a regeneration writes)
    'dedup-plants': ('lm-contam-plant-v1', 'lm-contam-plant-v1.jsonl'),
    'dedup-pairs': ('lm-dedup-pairs-v1', 'lm-dedup-pairs-v1.jsonl.gz'),
    'dedup-emb': ('lm-data-dedup-emb-v1', 'lm-data-dedup-emb-v1.npz'),
    'dedup-results': ('lm-data-dedup-results-v1',
                      'lm-data-dedup-results-v1.json'),
    'dedup-c2': ('lm-c2-docs-v1', 'lm-c2-docs-v1.jsonl.gz'),
    'dedup-manifest': ('lm-corpus-manifest-v1', 'lm-corpus-manifest-v1.json')}
RERUN = d2l.lm_rerun(ARTIFACTS)
assert GPU or not ({'dedup-plants', 'dedup-emb'} & RERUN), (
    'these regenerations need a GPU')
# The plants are written by a run of their own, with the pairs at most (the
# detectors read them before the cell that makes them); register, then the
# rest. A regeneration run compares only with what is registered already.
assert 'dedup-plants' not in RERUN or RERUN <= {'dedup-plants',
                                                'dedup-pairs'}, (
    'regenerate dedup-plants (and dedup-pairs) first, then register them')
PLANTS_PENDING = 'dedup-plants' in RERUN
R, TIMES, MEM = {}, {}, {}  # results (compared at the end); wall clocks
# and memory sizes (depend on the machine and Python: never compared)
ATTRIBUTION = ('Text: FineWeb, HuggingFaceFW/fineweb, CC-MAIN-2025-05 '
               '(ODC-By 1.0)')


def say(*args, sep=' '):
    """print, each line wrapped to the page's 100 columns."""
    print('\n'.join(l if len(l) <= 100 else textwrap.fill(
        l, 100, subsequent_indent='    ', break_long_words=False,
        break_on_hyphens=False) for l in sep.join(map(str, args)).split('\n')))


say(f'K={K}, TAU={TAU}, seeds {[hex(0xD2 + s) for s in SEEDS]}, '
    f'workers {N_WORKERS}, GPU={GPU}, regenerating: '
    f'{", ".join(sorted(RERUN)) or "none"}')
```

None of the registered files holds text of our crawl. The labelled pairs are
stored as document ids and edit parameters, and the cell that reads them
rebuilds both texts of every pair from FineWeb's stream. The planted test set
holds benchmark items under their licences, FineWeb ids and character offsets,
and for every paraphrase the hash of its prompt and the model's revision. The
embeddings are of FineWeb text and of benchmark items only. The decisions for
our crawl and the manifest are ids, reason codes, counts and hashes.

<!-- Registered (run step, 2026-11-21): the six keys of the `#@save` block below. The section also reads `lm-data-audit-stream-v1` (with the host_pair rows), `lm-data-emb-qwen3emb-v1`, `lm-c1-docs-v1` and `lm-data-quality-results-v1` (20.2), `lm-c0-docs-v1` and `lm-data-crawl-results-v1` (20.1), and the reference tokenizer `lm-r2-tokenizer-v1` (20.6); GSM8K and ARC-Challenge through chapter 18's keys, MMLU through chapter 18's loader. -->

```{.python .input #dedup-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-contam-plant-v1'] = (  # 100 GSM8K, ARC-Challenge and
    # MMLU test items each, verbatim, reformatted and paraphrased by
    # Qwen3.5-4B, with FineWeb ids and offsets (20.3); MIT, CC BY-SA 4.0
    d2l.LM_DATA_URL + 'lm-contam-plant-v1.jsonl',
    '54e171c603a7133729715881267efb4e2f35a503')
d2l.DATA_HUB['lm-dedup-pairs-v1'] = (  # 12,000 labelled pairs as ids and
    # edit parameters, no text (20.3)
    d2l.LM_DATA_URL + 'lm-dedup-pairs-v1.jsonl.gz',
    '366ff9d4242d86faffa6796c1af130cb022c8860')
d2l.DATA_HUB['lm-data-dedup-emb-v1'] = (  # Qwen3-Embedding vectors of the
    # benchmark items and of 128-word windows of FineWeb documents (20.3)
    d2l.LM_DATA_URL + 'lm-data-dedup-emb-v1.npz',
    'a1d73933ad4fa3a34b53427802c003962d2c0532')
d2l.DATA_HUB['lm-c2-docs-v1'] = (  # keep or drop, stage and reason for
    # every c0 document, no text (20.3)
    d2l.LM_DATA_URL + 'lm-c2-docs-v1.jsonl.gz',
    '9bb87a9e2572038a42f64fcd921ab8661f26b0d4')
d2l.DATA_HUB['lm-corpus-manifest-v1'] = (  # the nine-stage manifest of
    # lm-corpus-v1 (20.3)
    d2l.LM_DATA_URL + 'lm-corpus-manifest-v1.json',
    '4c2ff314b12af6863aadf776801939e2853637e2')
d2l.DATA_HUB['lm-data-dedup-results-v1'] = (  # 20.3's numbers
    d2l.LM_DATA_URL + 'lm-data-dedup-results-v1.json',
    '90e8d6b236630aad59762b8d13824acf35492f5d')

DEDUP_KEYS = tuple(key for key, _ in ARTIFACTS.values())
UPSTREAM = ('lm-data-audit-stream-v1', 'lm-data-emb-qwen3emb-v1',
            'lm-c1-docs-v1', 'lm-data-quality-results-v1', 'lm-c0-docs-v1',
            'lm-data-crawl-results-v1', 'lm-r2-tokenizer-v1', 'gsm8k-test',
            'arc-challenge-test')
for name, keys in (('this section', DEDUP_KEYS), ('upstream', UPSTREAM)):
    say(f'{name}: registered', ', '.join(
        k for k in keys if k in d2l.DATA_HUB) or 'none', '| pending:',
        ', '.join(k for k in keys if k not in d2l.DATA_HUB) or 'none')
say('MMLU test:', d2l.EVAL_TASKS['mmlu'].dataset,
    d2l.EVAL_TASKS['mmlu'].revision[:8], '(chapter 18 loader)')
```

The opening cell reads the 20,000 documents of the audit stream of
:numref:`sec_lm_quality_filtering` and 6,000 more, four pages from each of
1,500 hosts, so that pages of one site can be compared with each other. It
finds the most similar pair of same-host pages that are not identical, prints
the lines in which they differ, and then times the exact comparison of all
pairs among 1,000 documents.

```{.python .input #dedup-opening}
%%tab pytorch
STREAM = pd.read_parquet(d2l.lm_artifact('lm-data-audit-stream-v1'))
SLICE = STREAM[STREAM.role.isin(['stream', 'host_pair'])].reset_index(
    drop=True)
SLICE['host'] = [d2l._url_host(u) for u in SLICE.url]
STREAM_S = SLICE[SLICE.role == 'stream'].reset_index(drop=True)
TEXT = dict(zip(SLICE.id, SLICE.text))
HOSTS = SLICE[SLICE.role == 'host_pair'].groupby('host').id.apply(list)
HOST_PAIRS = [p for ids in HOSTS for p in itertools.combinations(ids, 2)]
say(f'slice: {len(STREAM_S):,} stream documents and '
    f'{(SLICE.role == "host_pair").sum():,} same-host documents '
    f'({len(HOSTS):,} hosts, {len(HOST_PAIRS):,} same-host pairs)')

host_sets = {i: set(d2l.shingles(TEXT[i]))
             for i in SLICE.id[SLICE.role == 'host_pair']}
J_host = np.array([d2l.jaccard_exact(host_sets[a], host_sets[b])
                   for a, b in HOST_PAIRS])
top = int(np.argmax(np.where(J_host < 1.0, J_host, -1.0)))
a, b = HOST_PAIRS[top]  # the most similar same-host pair that is not equal
same_hash = (hashlib.sha256(TEXT[a].encode()).digest()
             == hashlib.sha256(TEXT[b].encode()).digest())
host_of = dict(zip(SLICE.id, SLICE.host))
small, large = sorted((host_sets[a], host_sets[b]), key=len)
say(f'\nmost similar same-host pair below J = 1: {host_of[a]}, '
    f'J = {J_host[top]:.3f}, containment of the smaller '
    f'{d2l.containment(small, large):.3f}, '
    f'{len(TEXT[a].split()):,} and {len(TEXT[b].split()):,} words, '
    f'SHA-256 equal: {same_hash}')
diff = [line for line in difflib.unified_diff(
    TEXT[a].split('\n'), TEXT[b].split('\n'), lineterm='', n=0)
        if line[:1] in '+-' and line[:3] not in ('---', '+++')]
for line in diff[:6]:  # the lines in which the two pages differ
    say('  ' + (line if len(line) <= 96 else line[:95] + '…'))
say(ATTRIBUTION)

sets = [set(d2l.shingles(t)) for t in STREAM_S.text[:1000]]
t0 = time.perf_counter()
for i in range(len(sets)):
    for j in range(i + 1, len(sets)):
        d2l.jaccard_exact(sets[i], sets[j])
TIMES['all_pairs_1000'] = time.perf_counter() - t0
per_pair = TIMES['all_pairs_1000'] / (1000 * 999 / 2)
say(f'\nexact Jaccard of all pairs of 1,000 documents: measured '
    f'{TIMES["all_pairs_1000"]:.1f} s ({1e6 * per_pair:.1f} µs per pair);'
    ' extrapolated (quadratic):')
for n in (20_000, 10**9):
    sec = n * (n - 1) / 2 * per_pair
    say(f'  {n:>13,} documents: {sec:.3g} s = {sec / 3600:.3g} hours = '
        f'{sec / 3.156e7:.3g} years')
R['opening'] = dict(pair=[a, b], J=J_host[top], n_hosts=len(HOSTS),
                    n_host_pairs=len(HOST_PAIRS), same_sha256=same_hash)
del host_sets, sets
```

The two pages are 193 and 268 words long, and the only difference is a
paragraph that the longer one adds about two EPs on Bandcamp. Their hashes
differ, so an exact method treats them as two documents. The resemblance $J$
of :numref:`subsec_lm_dedup_jaccard` puts them at 0.70, below the threshold of
0.8 that we will use, while the share of the shorter page found in the longer
one is 1.00. Which of these numbers describes "a duplicate" is the first
decision of the section. The timing is the second fact. A quadratic method
that costs half an hour on the 20,000 documents of the stream would take about
$10^5$ years on $10^9$ documents, and no amount of hardware closes a gap of
that size. The method has to avoid comparing pairs, not compare them faster.

## Exact Duplicates
:label:`subsec_lm_dedup_exact`

A document hash finds copies. We hash the NFC text of every document with
SHA-256 and keep the first document of each hash. NFC matters here as it did
in :numref:`sec_lm_crawl`: the same "é" written as one code point or as two
gives different bytes, and different bytes give different hashes. A collision
of two different texts under SHA-256 is not a practical concern, so equal
hashes mean equal texts.

The same idea applied to lines finds shared parts. Hash every line of at least
five words and count the documents that contain it. A line that occurs in many
documents is almost always boilerplate: a cookie notice, a navigation bar, the
text that blog or shop software prints under every post. Such a line is a span
problem, not a document problem. The documents that carry it differ, and
removing them would throw away their content to get rid of one sentence. C4
handled this at the span level, discarding all but one of every three-sentence
span that occurred more than once in the corpus :cite:`raffel2020exploring`.

One pitfall belongs here because it is easy to commit and hard to notice.
Deduplication must come before the split into training and validation data.
After the split, copies of one page land on both sides, and the validation
loss partly measures how well the model remembers its training set.

```{.python .input #dedup-exact}
%%tab pytorch
def nfc_sha256(text):
    return hashlib.sha256(unicodedata.normalize('NFC', text).encode(
        'utf-8')).hexdigest()

def paragraphs(text, min_words=5):
    """The distinct lines of a document with at least min_words words."""
    return {p.strip() for p in text.split('\n')
            if len(p.split()) >= min_words}

C0 = d2l.c0_documents()
with gzip.open(d2l.lm_artifact('lm-c1-docs-v1'), 'rt') as f:
    FLAGS = {r['id']: r for r in map(json.loads, f)}
C1 = [d for d in C0 if not FLAGS[d.id]['codes']]
for name, texts in (('slice', SLICE.text.tolist()),
                    ('c1', [d.text for d in C1])):
    h = collections.Counter(map(nfc_sha256, texts))
    para = collections.Counter(p for t in texts for p in paragraphs(t))
    shared = {p for p, c in para.items() if c > 1}
    docs = sum(bool(paragraphs(t) & shared) for t in texts)
    R.setdefault('exact', {})[name] = dict(
        docs=len(texts), distinct=len(h), copies=len(texts) - len(h),
        paragraphs=len(para), shared_paragraphs=len(shared),
        docs_with_shared=docs)
    say(f'{name}: {len(texts):,} documents, {len(h):,} distinct NFC '
        f'hashes ({len(texts) - len(h):,} exact copies); {len(shared):,} '
        f'of {len(para):,} distinct paragraphs occur in more than one '
        f'document, and {docs:,} documents contain one')

# five paragraphs shared by different slice documents (by count, ties by
# the paragraph's hash), each with the two documents of lowest id
by_para = collections.defaultdict(list)
for i, t in sorted(TEXT.items()):
    for p in paragraphs(t):
        by_para[p].append(i)
common = sorted((p for p, ids in by_para.items() if len(ids) > 1),
                key=lambda p: (-len(by_para[p]), nfc_sha256(p)))[:5]
say()
for p in common:
    ids = by_para[p]
    say(f'{len(ids):>4} documents, e.g. {ids[0][10:18]}… and '
        f'{ids[1][10:18]}…: {p[:100]!r}')
say(ATTRIBUTION)
R['exact']['common'] = [len(by_para[p]) for p in common]
```

FineWeb's output has no exact copies at all: 26,000 documents, 26,000 distinct
hashes. Yet 1,482 lines occur in more than one of its documents, and 1,742
documents, about one in fifteen, contain at least one of them. The most
frequent is a notice about secure card payments in 36 documents, followed by
the notices that shop software prints under products and WordPress's "Your
email address will not be published". Our own crawl after the rules, c1, has
12 exact copies among its 1,032 documents. Nothing has deduplicated it yet,
while FineWeb removed copies within its dump before we drew the stream.

## MinHash and Banding
:label:`subsec_lm_dedup_minhash_banding`

An exact hash answers one question, equal or not. It cannot be a little bit
equal: change one character and the two digests share nothing. Near-duplicates
need a similarity and a way to find similar pairs without the comparison of
all pairs that the opening cell timed. The three subsections below build this.
We define similarity as the overlap of two sets of word sequences, show that a
signature of 128 numbers per document estimates it, and cut the signature into
bands that act as hash keys, so that similar documents meet in a bucket and
dissimilar ones almost never do.

### Similarity as Set Overlap
:label:`subsec_lm_dedup_jaccard`

A document becomes a set by *shingling*: its *shingles* are its runs of $n$
consecutive words. FineWeb uses $n = 5$
:cite:`Penedo.Kydlicek.BenAllal.ea.2024`, and so do we. For two documents with
shingle sets $A$ and $B$, Broder defined two measures of overlap,

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}, \qquad
C(A, B) = \frac{|A \cap B|}{|A|}.
$$

The *resemblance* or Jaccard similarity $J$ is symmetric and equals 1 only for
equal sets. The *containment* $C$ of $A$ in $B$ asks how much of $A$ is found
in $B$. The opening pair is the case in which the two disagree. When $A$
lies inside $B$, the containment is 1 and the resemblance is $|A|/|B|$, so a
page cut to half its length is a perfect match by $C$ and a poor one by $J$.
FineWeb and this section use $J$. A truncated copy therefore counts as a
duplicate only if most of the page survives, which is a choice and should be
recognised as one. Readers who met the intersection over union of boxes in
:numref:`sec_anchor` know $J$ already, with areas of pixels in place of counts
of shingles.

The code follows a fixed specification, so that two implementations can be
compared bit for bit. Text is normalized to NFC and lowercased, and a word is
a match of the regular expression `\w+`. A document with fewer than five
words has no shingles, and later no signature, so it never matches anything.
Shingles are hashed with blake2b, never with Python's built-in `hash`, which
is salted afresh in every process: signatures built on it would change from
run to run, and so would every manifest hash computed from them.

```{.python .input #dedup-shingle-def}
%%tab pytorch
#@save
def shingles(text, n=5):
    """The word n-grams of text as strings (the MinHash specification):
    NFC, lowercase, the regex \\w+ tokens, n consecutive tokens joined by
    one space. Empty if text has fewer than n words."""
    import unicodedata
    import regex
    words = regex.findall(r'\w+', unicodedata.normalize('NFC', text).lower())
    return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

#@save
def jaccard_exact(a, b):
    """The resemblance |A & B| / |A | B| of two shingle collections taken
    as sets; 0 if both are empty."""
    a = a if isinstance(a, (set, frozenset)) else set(a)
    b = b if isinstance(b, (set, frozenset)) else set(b)
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0

#@save
def containment(a, b):
    """The containment |A & B| / |A| of A in B (Broder); 0 if A is empty."""
    a = a if isinstance(a, (set, frozenset)) else set(a)
    b = b if isinstance(b, (set, frozenset)) else set(b)
    return len(a & b) / len(a) if a else 0.0
```

How similar are web pages to each other? The cell shingles all 26,000
documents and computes the exact $J$ of 5,000 random pairs and of 5,000 pairs
of pages from one host.

```{.python .input #dedup-jaccard-hist}
%%tab pytorch
t0 = time.perf_counter()
SH = {i: set(d2l.shingles(t)) for i, t in TEXT.items()}
TIMES['shingle_slice'] = time.perf_counter() - t0
sizes = np.array([len(s) for s in SH.values()])
mem = sum(sys.getsizeof(s) + sum(map(sys.getsizeof, s)) for s in SH.values())
say(f'{len(SH):,} documents: shingles per document median '
    f'{np.median(sizes):,.0f}, 90th percentile '
    f'{np.percentile(sizes, 90):,.0f}; all shingle sets take '
    f'{mem / 2**30:.2f} GiB as Python sets, signatures of {K} rows '
    f'{len(SH) * K * 8 / 2**20:.0f} MiB')
rng = np.random.default_rng(0)
ids = STREAM_S.id.to_numpy()
rand = [tuple(ids[rng.choice(len(ids), 2, replace=False)])
        for _ in range(5000)]
host = [HOST_PAIRS[i] for i in rng.choice(len(HOST_PAIRS), 5000,
                                          replace=False)]
J_rand, J_same = (np.array([d2l.jaccard_exact(SH[x], SH[y]) for x, y in ps])
                  for ps in (rand, host))
edges = np.linspace(0, 1, 51)
counts = [np.histogram(j, edges)[0].astype(float) for j in (J_rand, J_same)]
for c in counts:
    c[c == 0] = np.nan  # empty bins are left out on the log scale
d2l.plot((edges[:-1] + edges[1:]) / 2, counts, 'exact Jaccard of the pair',
         'pairs', legend=['random pairs', 'same-host pairs'], yscale='log',
         xlim=[0, 1], fmts=('o-', 's--'), figsize=(4.5, 3))
for name, j in (('random', J_rand), ('same-host', J_same)):
    say(f'{name:>9} pairs: median J {np.median(j):.3f}, 99th percentile '
        f'{np.percentile(j, 99):.3f}, J >= {TAU}: {np.sum(j >= TAU)}')
MEM['shingle_sets'] = mem
R['jaccard'] = dict(median_shingles=np.median(sizes),
                    random=np.percentile(J_rand, [50, 90, 99]).tolist(),
                    same_host=np.percentile(J_same, [50, 90, 99]).tolist(),
                    random_dup=int(np.sum(J_rand >= TAU)),
                    same_host_dup=int(np.sum(J_same >= TAU)))
del SH  # 1.8 GiB: the signatures below replace the sets
```

A document has a median of 360 shingles, and all the shingle sets together
take 1.76 GiB as Python sets; signatures of 128 numbers take 25 MiB, about
seventy times less. Two random pages share no five-word sequence at all, up to
the 99th percentile. Pages of one site share more, through their templates,
and the figure shows their tail on the log scale, but even their 99th
percentile is 0.32, and no pair in either sample reaches 0.8. This is the
shape the method has to exploit. Almost every pair is far below the threshold,
and the work worth saving is the work of looking at those pairs at all.

### MinHash
:label:`subsec_lm_dedup_minhash`

Let us start with a thought experiment that is too expensive to run. Put all
possible shingles in a random order $\pi$, and let $h_\pi(A)$ be the shingle
of $A$ that comes first in this order, the *minimum* of $A$ under $\pi$. Two
documents agree in this one value with a probability that we can compute.

**Proposition.** *For a uniformly random order $\pi$ and non-empty sets $A$
and $B$, $\Pr[h_\pi(A) = h_\pi(B)] = J(A, B)$.*

**Proof.** Let $x^*$ be the element of $A \cup B$ that comes first under
$\pi$. The order is uniform, so every element of the union is equally likely
to be $x^*$, and the probability that $x^*$ falls in a subset is the size of
the subset over the size of the union, as for any uniform choice in
:numref:`sec_prob`. If $x^*$ lies in $A \cap B$, it is the minimum of both
sets and the two values agree. If $x^*$ lies in $A$ but not in $B$, it is the
minimum of $A$, while the minimum of $B$ comes later, so the values differ,
and the same holds with the roles exchanged. The values therefore agree
exactly when $x^* \in A \cap B$, which has probability $|A \cap B| / |A \cup
B|$. $\blacksquare$

One order gives one coin flip with success probability $J$. A *MinHash
signature* repeats the experiment with $k$ independent orders and stores the
$k$ minima. The share of positions in which two signatures agree estimates
$J$, and since the number of agreeing positions is a binomial count of $k$
trials (:numref:`sec_prob`), the estimate is unbiased with a variance we can
write down:

$$
\hat J = \frac{1}{k}\sum_{i=1}^{k}
\mathbf{1}\bigl[h_{\pi_i}(A) = h_{\pi_i}(B)\bigr],
\qquad
\mathbb{E}[\hat J] = J,
\qquad
\operatorname{Var}[\hat J] = \frac{J(1 - J)}{k}.
$$
:eqlabel:`eq_lm_dedup_minhash`

The remarkable part is what the signature does not depend on. It has $k$
numbers whether the document has 50 shingles or 50,000, and two signatures
estimate the similarity without the documents. The estimate is noisy: at $J =
0.8$ and $k = 128$ the standard deviation is $\sqrt{0.16/128} \approx 0.035$,
so pairs at 0.75 and at 0.85 are not reliably told apart.

A random order of all possible shingles cannot be stored, so implementations
use hash functions instead. Each shingle is hashed to a number $x$ below the
prime $p = 2^{61} - 1$, and position $i$ of the signature is the minimum of
$(a_i x + c_i) \bmod p$ over the document's shingles. For $a_i \neq 0$ this
map is a bijection of $\{0, \ldots, p - 1\}$, so it does order the shingles,
but it is one of about $p^2$ such maps rather than a uniform choice among all
orders, and the proposition holds only approximately. Whether that matters is
a question for measurement. The specification fixes every remaining choice:

1. NFC, lower case, `\w+` words, shingles of five words joined by one space;
2. each shingle hashed by blake2b with an 8-byte digest, read as a
   little-endian integer and reduced modulo $p$;
3. $(a_i, c_i)$ for $i = 1, \ldots, 128$ drawn from
   `numpy.random.default_rng(0xD2)` as integers in $[1, p)$;
4. the product $a_i x$ computed exactly.

The last point hides a trap. Both factors are below $2^{61}$, so their product
needs up to 122 bits, and NumPy's `uint64` arithmetic silently discards
everything above 64. The library splits the multiplication into 32-bit halves.

<!-- Figure pending (write step CODE REQUEST): ![…](../img/mdl-lm-data-minhash.svg) :label:`fig_lm_minhash`, generated by tools/gen_mdl_lm_data_figures.py; the prose does not reference it yet. -->

```{.python .input #dedup-minhash-def}
%%tab pytorch
#@save
MINHASH_PRIME = (1 << 61) - 1  # the Mersenne prime 2^61 - 1

#@save
def minhash_signature(shingle_list, k=128, seed=0xD2):
    """The MinHash signature, a (k,) uint64 array, of a shingle collection:
    row i is the minimum over shingles of (a_i x + c_i) mod p, where x is
    the shingle's blake2b hash (8 bytes, little-endian) mod p, p = 2^61 - 1
    and (a, c) are drawn from numpy.random.default_rng(seed). None for an
    empty collection (fewer words than the shingle size): such a document
    never matches."""
    if not shingle_list:
        return None
    x = np.unique(d2l._shingle_hashes(shingle_list)) % np.uint64(
        d2l.MINHASH_PRIME)
    a, c = d2l._minhash_params(k, seed)
    return d2l._minhash_rows(x, a, c)  # exact mulmod: 32-bit halves

#@save
MINHASH_FIXTURE = (  # five strings whose signatures the library pins
    'The quick brown fox jumps over the lazy dog.',
    'Deduplication keeps one copy of every document in the corpus.',
    'Café crème: une boisson chaude, servie le matin avec du pain.',
    'ONE two Three four FIVE six seven, eight; nine: ten!',
    'Near-duplicate pages differ only by a date or a sidebar.')
MINHASH_FIXTURE_SHA256 = (  # of their k = 128, seed 0xD2 signatures' bytes
    'b9b07bf416ce76f2b5cf1ae3d12fc617bdfa5bc7e464230c6bca1ada3bda0195')
```

We check the estimator on sets whose similarity we choose: for each $J$ from
0.1 to 0.9, 200 pairs with unions of 200 random words. If the hash family
behaves like random orders, the mean of $\hat J$ lies within three standard
errors of $J$ and the variance across pairs is close to $J(1 - J)/k$. The same
pairs go through datasketch's `MinHash`, an independent implementation.
Finally, the signatures of five fixed strings, whose hash the library pins,
are compared position by position with a version of the specification in
Python integers, which never overflow.

```{.python .input #dedup-minhash-check}
%%tab pytorch
rng = np.random.default_rng(0)
rows, n_pairs, U = [], 200, 200  # 200 pairs per J, unions of 200 shingles
for J in np.round(np.arange(0.1, 1.0, 0.1), 1):
    ours, oracle = [], []
    for _ in range(n_pairs):
        words = [f'w{x}' for x in rng.integers(0, 2**62, U)]
        inter = round(J * U)
        a_only = (U - inter) // 2
        A, B = words[:inter + a_only], words[:inter] + words[inter + a_only:]
        assert d2l.jaccard_exact(A, B) == J  # (distinct draws: 2^62 range)
        ours.append(np.mean(d2l.minhash_signature(A, K)
                            == d2l.minhash_signature(B, K)))
        ma, mb = MinHash(num_perm=K), MinHash(num_perm=K)
        ma.update_batch([w.encode() for w in A])
        mb.update_batch([w.encode() for w in B])
        oracle.append(ma.jaccard(mb))
    se = math.sqrt(J * (1 - J) / K / n_pairs)  # of the mean of n_pairs
    rows.append(dict(J=J, bias=np.mean(ours) - J,
                     z=(np.mean(ours) - J) / se,
                     var_ratio=np.var(ours, ddof=1) / (J * (1 - J) / K),
                     z_datasketch=(np.mean(oracle) - J) / se))
D1 = pd.DataFrame(rows)
say(D1.round(4).to_string(index=False))
assert (D1.z.abs() <= 3).all() and (D1.z_datasketch.abs() <= 3).all()
assert D1.var_ratio.between(0.7, 1.3).all()
say(f'95% half-width at J = 0.8, k = {K}: '
    f'{1.96 * math.sqrt(0.8 * 0.2 / K):.4f}')

def reference(shingle_list, k=K, seed=0xD2):
    """The specification in Python integers: no NumPy arithmetic."""
    p = d2l.MINHASH_PRIME
    a, c = np.random.default_rng(seed).integers(1, p, size=(2, k),
                                                dtype=np.uint64)
    xs = {int.from_bytes(hashlib.blake2b(s.encode('utf-8'), digest_size=8)
                         .digest(), 'little') % p for s in shingle_list}
    return [min((int(ai) * x + int(ci)) % p for x in xs)
            for ai, ci in zip(a, c)]

sigs = [d2l.minhash_signature(d2l.shingles(t), K)
        for t in d2l.MINHASH_FIXTURE]
same = [s.tolist() == reference(d2l.shingles(t))
        for s, t in zip(sigs, d2l.MINHASH_FIXTURE)]
say(f'fixture: signatures equal to the Python-integer specification: '
    f'{sum(same)} of {len(same)}')
assert all(same), 'not bit-exact to the specification'
fix = hashlib.sha256(b''.join(s.astype('<u8').tobytes() for s in sigs))
say(f'fixture signatures: SHA-256 {fix.hexdigest()[:16]}…, pinned '
    f'{d2l.MINHASH_FIXTURE_SHA256[:16]}…')
assert fix.hexdigest() == d2l.MINHASH_FIXTURE_SHA256, 'fixture changed'
R['minhash'] = dict(table=D1.to_dict('list'), fixture=fix.hexdigest(),
                    reference_equal=sum(same))
```

In every bin the bias is within two standard errors (the largest $|z|$ is
1.85), and the variance is between 0.88 and 1.16 times its formula. Datasketch
behaves the same way. The fixture signatures are equal to the integer
reference in all five strings, and their hash equals the pinned one, so any
implementation of the specification can test itself against these five. For
sets of this size, then, the $(a x + c) \bmod p$ family is as good as a random
order, and the 95% half-width at $J = 0.8$ is $1.96 \sqrt{0.8 \cdot 0.2 / 128}
= 0.069$, as the formula says.

### Banding and the S-Curve
:label:`subsec_lm_dedup_lsh`

An estimate of $J$ for one pair still leaves us comparing pairs. *Banding*
turns the signature into a search. Cut the $k = br$ positions of every
signature into $b$ bands of $r$ positions, and use each band, hashed, as the
key of a bucket. Two documents become a *candidate pair* if they share a
bucket in at least one band. For a pair of similarity $s$, one position agrees
with probability $s$, and the positions come from different hash functions, so
a whole band agrees with probability $s^r$. All $b$ bands disagree with
probability $(1 - s^r)^b$, and the pair becomes a candidate with probability

$$
P(s) = 1 - (1 - s^r)^b.
$$
:eqlabel:`eq_lm_dedup_scurve`

For small $s$ the curve is close to $b s^r$, tiny, and for $s$ near 1 it is
close to 1; in between it rises steeply, which gives it its name, the
*S-curve*. Its derivative $P'(s) = b r s^{r-1} (1 - s^r)^{b-1}$ is largest
where $(r - 1)(1 - s^r) = (b - 1) r s^r$, that is at $s^* = ((r - 1)/(br -
1))^{1/r}$, close to $(1/b)^{1/r}$. More rows per band move the step to the
right, more bands move it to the left. The smallest case needs four numbers.
With $b = r = 2$, $P(s) = 1 - (1 - s^2)^2$, which is 0.4375 at $s = 0.5$ and
0.964 at $s = 0.9$, so a signature of four values already separates these two
similarities most of the time.

The budget $k$ is fixed by memory and by the cost of hashing, so the design
question is how to split it. A natural answer compares the S-curve with the
ideal step at the threshold $\tau$. The area under the curve left of $\tau$
counts pairs below the threshold that become candidates, the area above the
curve right of $\tau$ counts duplicates that are missed, and two weights say
what each error costs:

$$
E(b, r) = w_{\textrm{fp}} \int_0^{\tau} P(s)\, ds
+ w_{\textrm{fn}} \int_{\tau}^{1} \bigl(1 - P(s)\bigr)\, ds.
$$
:eqlabel:`eq_lm_dedup_br`

`best_br` minimizes $E$ over $r$ with $b = \lfloor k/r \rfloor$, as
datasketch's `MinHashLSH` does internally. The integral weighs every
similarity in $[0, 1]$ equally, an assumption that will matter.

```{.python .input #dedup-lsh-def}
%%tab pytorch
#@save
def scurve(s, b, r):
    """The probability that a pair of Jaccard s becomes a candidate under b
    bands of r rows: 1 - (1 - s^r)^b."""
    return 1 - (1 - np.asarray(s, dtype=float) ** r) ** b

#@save
def _br_error(b, r, tau, w_fp=1.0, w_fn=1.0, n_grid=20_001):
    """The weighted error areas of an S-curve against the step at tau:
    w_fp times the area under P on [0, tau] (pairs below the threshold that
    become candidates) plus w_fn times the area above P on [tau, 1]
    (trapezoid rule on n_grid points)."""
    s = np.linspace(0, 1, n_grid)
    P = d2l.scurve(s, b, r)
    lo, hi = s <= tau, s >= tau
    return (w_fp * np.trapezoid(P[lo], s[lo])
            + w_fn * np.trapezoid(1 - P[hi], s[hi]))

#@save
def best_br(k, tau, w_fp=1.0, w_fn=1.0):
    """The (b, r) with b = k // r that minimizes the weighted error areas
    of its S-curve against the step at tau (the smallest r on a tie)."""
    errors = {(k // r, r): d2l._br_error(k // r, r, tau, w_fp, w_fn)
              for r in range(1, k + 1)}
    return min(errors, key=errors.get)

#@save
def lsh_candidates(sigs, b, r, *, max_bucket=1000, seed=0):
    """Candidate pairs (i, j), i < j, of signatures (a sequence of arrays,
    None for an empty document) that agree in all r rows of at least one of
    b bands; the bucket key of band j is blake2b (8 bytes) of its rows'
    bytes. A bucket with more than max_bucket members (templated pages) is
    cut to max_bucket members drawn with default_rng((seed, j)). Returns
    (set of pairs, report of bucket sizes and the cut buckets)."""
    import itertools
    pairs, sizes, cut = set(), collections.Counter(), []
    rows = [i for i, s in enumerate(sigs) if s is not None]
    for j in range(b):
        buckets = collections.defaultdict(list)
        for i in rows:
            band = sigs[i][j * r:(j + 1) * r].astype('<u8')
            buckets[hashlib.blake2b(band.tobytes(), digest_size=8)
                    .digest()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            sizes[len(members)] += 1
            if max_bucket and len(members) > max_bucket:
                cut.append((j, len(members)))
                members = sorted(np.random.default_rng((seed, j)).choice(
                    members, max_bucket, replace=False).tolist())
            pairs.update(itertools.combinations(members, 2))
    return pairs, dict(n_pairs=len(pairs), bucket_sizes=dict(sizes),
                       largest=max(sizes, default=0), cut=cut)

#@save
def union_find_clusters(n, pairs):
    """Connected components of n items under pairs, by union-find (path
    compression, union by size); each item's label is the smallest index
    of its component, so the result does not depend on the pair order."""
    parent, size = list(range(n)), [1] * n
    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            if size[ri] < size[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            size[ri] += size[rj]
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    low = np.full(n, n, dtype=np.int64)
    np.minimum.at(low, roots, np.arange(n))
    return low[roots]

#@save
def near_dedup(texts, threshold=0.8, ngram=5, num_perm=128, seed=0, *,
               workers=1):
    """The indices of texts to keep after near-duplicate removal: MinHash
    signatures (hash seed 0xD2 + seed), banding at best_br(num_perm,
    threshold), union-find clusters; each cluster keeps its lowest index.
    workers > 1 computes the signatures in separate processes."""
    sigs = d2l._minhash_texts(texts, num_perm, (0xD2 + seed,), ngram,
                              workers)[0xD2 + seed]
    pairs, _ = d2l.lsh_candidates(sigs, *d2l.best_br(num_perm, threshold))
    labels = d2l.union_find_clusters(len(sigs), pairs)
    return [i for i in range(len(sigs)) if labels[i] == i]
```

```{.python .input #dedup-scurve-check}
%%tab pytorch
paper = (0.56, 0.77, 0.92, 0.988)  # FineWeb's matched probabilities
ours = [round(float(d2l.scurve(s, 14, 8)), 2 if p < 0.98 else 3)
        for s, p in zip((0.7, 0.75, 0.8, 0.85), paper)]
say('(14, 8) at s = 0.7, 0.75, 0.8, 0.85:', ours, '| paper:', paper)
assert tuple(ours) == paper
say(f'b = r = 2: P(0.5) = {d2l.scurve(0.5, 2, 2):.4f}, '
    f'P(0.9) = {d2l.scurve(0.9, 2, 2):.4f}')
s = np.linspace(0, 1, 200_001)
for b, r in ((14, 8), (9, 13), (20, 5)):
    slope = np.gradient(d2l.scurve(s, b, r), s)
    steep = ((r - 1) / (b * r - 1)) ** (1 / r)
    say(f'({b}, {r}): steepest at {s[np.argmax(slope)]:.4f}; '
        f'((r - 1)/(br - 1))^(1/r) = {steep:.4f}; '
        f'(1/b)^(1/r) = {(1 / b) ** (1 / r):.4f}')
    assert abs(s[np.argmax(slope)] - steep) < 1e-4

BR = d2l.best_br(K, TAU)
brute = {(b, r): d2l._br_error(b, r, TAU) for r in range(1, K + 1)
         for b in range(1, K // r + 1)}  # every b r <= K
best = min(brute, key=brute.get)
say(f'best_br({K}, {TAU}) = {BR}, error {d2l._br_error(*BR, TAU):.4f}; '
    f'brute force over b r <= {K}: {best}, error {brute[best]:.4f}')
assert BR == (9, 13) and abs(brute[best] - d2l._br_error(*BR, TAU)) < 1e-9
for r in (12, 13, 14):
    say(f'  ({K // r}, {r}): error {d2l._br_error(K // r, r, TAU):.4f}, '
        f'P({TAU}) = {d2l.scurve(TAU, K // r, r):.3f}')
lsh = MinHashLSH(threshold=TAU, num_perm=K)
DS_BR = (lsh.b, lsh.r)
BR5 = d2l.best_br(K, TAU, w_fn=5.0)
say(f'datasketch MinHashLSH(threshold={TAU}, num_perm={K}): (b, r) = '
    f'{DS_BR}; best_br with w_fn = 5: {BR5}')
R['scurve'] = dict(paper=list(ours), best_br=list(BR), brute=list(best),
                   datasketch=list(DS_BR), best_br_w5=list(BR5))
```

FineWeb's setting, 112 positions as 14 bands of 8 rows, gives candidate
probabilities of 0.56, 0.77, 0.92 and 0.988 at similarities 0.7, 0.75, 0.8 and
0.85, the four values its paper reports. Its steepest point is at 0.71,
because FineWeb aimed at pages about 75% similar. The steepest points agree
with the formula to four decimals. Under equal weights the best split of 128
positions for $\tau = 0.8$ is $(9, 13)$, found alike by `best_br`, by a
brute-force search over all $br \leq 128$ and by datasketch. A pair at
the threshold becomes a candidate with probability 0.40 there, and the
steepest point lies at 0.84, so this optimum misses more borderline duplicates
than it admits borderline non-duplicates. Weighting misses five times as
heavily moves the choice to $(12, 10)$.

To tune $(b, r)$ on data we need pairs with known similarity. The stream
cannot provide near-duplicates, because FineWeb deduplicated its dump with
MinHash before we drew from it, so the pairs are made by editing stream
documents. From 2,000 documents of at least 50 words, the generator makes
three copies each, every copy by one random edit: words substituted or deleted
at a rate between 1% and 30%, a block of 20 to 200 words of boilerplate
inserted, the paragraphs shuffled, or the text truncated to 30% to 90% of its
words. Then come 3,000 pairs of different pages from one host, which share
templates, and 3,000 random pairs. The label is the exact $J \geq 0.8$ of the
final texts. This makes the task a test of the sketch: estimate a known $J$
well enough to threshold it. Whether $J \geq 0.8$ is the right definition of a
duplicate is a separate question, which no labelled pair can settle. The pairs
are split into training and test halves by the SHA-256 of the base document,
so that the copies of one document stay together.

```{.python .input #dedup-pairs-def}
%%tab pytorch
#@save
DUP_PAIRS_SPEC = dict(  # the labelled pair set lm-dedup-pairs-v1
    n_base=2000, variants=3, min_words=50,
    ops=('sub', 'del', 'boiler', 'shuffle', 'trunc'),
    rates=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3),  # of the words, sub and del
    boiler_words=(20, 200), trunc=(0.3, 0.9),
    n_neg_host=3000, n_neg_random=3000, ngram=5, tau=0.8)

#@save
def pair_texts(row, texts, vocab=None):
    """The two texts of a labelled pair: texts[a] and texts[b], or texts[a]
    and its perturbed copy by row['op'] and row['params'] (vocab: the
    substitution words, needed for 'sub')."""
    a = texts[row['a']]
    if row['b'] is not None:
        return a, texts[row['b']]
    return a, d2l._dup_perturb(a, row['op'], row['params'], vocab)

#@save
def make_dup_pairs(docs, spec, seed=0):
    """The labelled pairs of spec from docs (a DataFrame with id, url, text
    and role 'stream' or 'host_pair'): spec['variants'] perturbed copies of
    each of spec['n_base'] stream documents (the first of a seeded
    permutation with at least spec['min_words'] words), pairs of different
    documents of one host (neg_host) and random stream pairs (neg_random).
    Each row: pair_id, a, b (None for a copy), op, params (with the copy's
    seed), J, C (containment of the smaller shingle set in the larger),
    label (J >= tau) and split (train or test by the SHA-256 of a)."""
    import itertools
    import regex
    rng = np.random.default_rng(seed)
    texts = dict(zip(docs.id, docs.text))
    stream = docs.id[docs.role == 'stream'].tolist()
    vocab = d2l._dup_vocab(texts[i] for i in stream)
    long_enough = (stream[i] for i in rng.permutation(len(stream))
                   if len(regex.findall(r'\w+', texts[stream[i]]))
                   >= spec['min_words'])
    todo = []
    for a in itertools.islice(long_enough, spec['n_base']):
        for _ in range(spec['variants']):
            op = str(rng.choice(spec['ops']))
            params = dict(seed=int(rng.integers(2**31)))
            if op in ('sub', 'del'):
                params['rate'] = float(rng.choice(spec['rates']))
            elif op == 'boiler':
                lo, hi = spec['boiler_words']
                params['words'] = int(rng.integers(lo, hi + 1))
            elif op == 'trunc':
                params['frac'] = round(float(rng.uniform(*spec['trunc'])), 4)
            todo.append((a, None, op, params))
    hosts = docs[docs.role == 'host_pair'].assign(
        host=[d2l._url_host(u) for u in docs.url[docs.role == 'host_pair']])
    same = [p for ids in hosts.groupby('host').id.apply(list)
            for p in itertools.combinations(ids, 2)]
    for k in sorted(rng.choice(len(same), spec['n_neg_host'],
                               replace=False).tolist()):
        todo.append((*same[k], 'neg_host', {}))
    seen = set()
    while len(seen) < spec['n_neg_random']:
        i, j = sorted(rng.choice(len(stream), 2, replace=False).tolist())
        if (i, j) not in seen:
            seen.add((i, j))
            todo.append((stream[i], stream[j], 'neg_random', {}))
    rows = []
    for n, (a, b, op, params) in enumerate(todo):
        row = dict(pair_id=f'p{n:05d}', a=a, b=b, op=op, params=params)
        sa, sb = (set(d2l.shingles(t, spec['ngram']))
                  for t in d2l.pair_texts(row, texts, vocab))
        small, large = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
        J = d2l.jaccard_exact(sa, sb)
        half = int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 2
        rows.append(dict(row, J=J, C=d2l.containment(small, large),
                         label=bool(J >= spec['tau']),
                         split=('train', 'test')[half]))
    return rows
```

```{.python .input #dedup-pairs}
%%tab pytorch
PAIRS_KEY, PAIRS_FILE = ARTIFACTS['dedup-pairs']
VOCAB = d2l._dup_vocab(STREAM_S.text)
if 'dedup-pairs' in RERUN:
    t0 = time.perf_counter()
    rows = d2l.make_dup_pairs(SLICE, d2l.DUP_PAIRS_SPEC, seed=0)
    say(f'made {len(rows):,} pairs in {time.perf_counter() - t0:.0f} s')
    d2l.lm_save_rerun(PAIRS_FILE, lambda f: d2l.write_docs(f, rows))
with gzip.open(d2l.lm_artifact(
        PAIRS_KEY, PAIRS_FILE if 'dedup-pairs' in RERUN else None),
        'rt') as f:
    PAIR_ROWS = [json.loads(line) for line in f]  # b is None for copies
PAIRS = pd.DataFrame(PAIR_ROWS)
t0 = time.perf_counter()
PAIR_TEXT = [d2l.pair_texts(row, TEXT, VOCAB) for row in PAIR_ROWS]
J_live = np.array([d2l.jaccard_exact(d2l.shingles(x), d2l.shingles(y))
                   for x, y in PAIR_TEXT])
TIMES['pairs_regen'] = time.perf_counter() - t0
gap = np.abs(J_live - PAIRS.J.to_numpy()).max()
say(f'{len(PAIRS):,} pairs regenerated from the stream in '
    f'{TIMES["pairs_regen"]:.0f} s; largest |J - stored J| {gap:.1e}')
assert gap <= 1e-12, 'the regenerated texts differ from the registered ones'
counts = PAIRS.groupby(['op', 'split']).size().unstack()
counts['label rate'] = PAIRS.groupby('op').label.mean().round(3)
counts['median J'] = PAIRS.groupby('op').J.median().round(3)
say(counts.to_string())
R['pairs'] = dict(n=len(PAIRS), by_op=PAIRS.op.value_counts().to_dict(),
                  labels=int(PAIRS.label.sum()))
```

The rebuilt texts reproduce every stored $J$ exactly. The table shows how
unevenly the edits move a page across the threshold. Shuffled paragraphs stay
duplicates most often, because only the shingles that cross a paragraph
boundary change. Substitutions do surprising damage. Each changed word removes
up to five shingles, so a substitution rate $q$ leaves a share of about $x =
(1 - q)^5$ of each document's shingles intact, and then $J = x/(2 - x)$. One
word in fifty gives $J \approx 0.82$, a duplicate; one word in twenty gives
$0.63$, not one. Truncation is the case of containment: a copy cut to a share
$f$ of its words has $J \approx f$, so only copies that keep at least 80% of
the page are labelled duplicates.

The derivation predicts that a pair of exact similarity $J_i$ becomes a
candidate with probability $P(J_i)$, and the pairs let us test this. We test
13 settings, the 12 splits of 128 positions with $r$ from 4 to 16 and
FineWeb's $(14, 8)$ on the first 112 positions, with three hash seeds each. In
every bin of $J$ of width 0.05, the number of candidates is a sum of
independent trials with different probabilities, a *Poisson-binomial* count,
whose exact distribution gives a 95% interval around the expected count
$\sum_i P(J_i)$. The curve at the bin's centre would not do: at $(9, 13)$ it
rises from 0.40 to 0.69 between 0.80 and 0.85. A hash that behaves as derived
leaves about 5% of the (setting, bin, seed) cells outside their intervals, or
fewer; a flawed one leaves many more.

```{.python .input #dedup-scurve-measure}
%%tab pytorch
t0 = time.perf_counter()
key_a = PAIRS.a.tolist()
key_b = [r['b'] if r['b'] is not None else r['pair_id'] + '/b'
         for r in PAIR_ROWS]
doc_text = dict(zip(key_a, (x for x, _ in PAIR_TEXT)))
doc_text.update(zip(key_b, (y for _, y in PAIR_TEXT)))
uniq = list(doc_text)
pos = {k: i for i, k in enumerate(uniq)}
ia, ib = (np.array([pos[k] for k in keys]) for keys in (key_a, key_b))
PSIGS = d2l._minhash_texts([doc_text[k] for k in uniq], K,
                           [0xD2 + s for s in SEEDS], 5, N_WORKERS)
CONFIGS = [(K // r, r) for r in R_GRID] + [(14, 8)]  # (14, 8): rows 0-111
DECIDE = {}
for s in SEEDS:
    Sg, ok = d2l._sig_matrix(PSIGS[0xD2 + s], K)
    both = ok[ia] & ok[ib]
    for b, r in CONFIGS:
        DECIDE[b, r, s] = d2l._band_match(Sg[ia], Sg[ib], b, r) & both
cand, rep = d2l.lsh_candidates(PSIGS[0xD2], 9, 13)  # the bucket pipeline
via = np.array([(min(i, j), max(i, j)) in cand for i, j in zip(ia, ib)])
assert (via == DECIDE[9, 13, 0]).all() and not rep['cut']
TIMES['scurve_measure'] = time.perf_counter() - t0
say(f'{len(uniq):,} distinct pair documents, {len(SEEDS)} seeds, '
    f'{len(CONFIGS)} settings: {TIMES["scurve_measure"]:.0f} s; bucket '
    f'pipeline at (9, 13) = per-pair band test on all {len(PAIRS):,} pairs')

J = PAIRS.J.to_numpy()
BINS = np.minimum((J / 0.05).astype(int), 19)
table = []
for b, r in CONFIGS:
    p = d2l.scurve(J, b, r)
    for g in np.unique(BINS):
        m = BINS == g
        lo, hi = d2l._pb_interval(p[m])
        for s in SEEDS:
            obs = int(DECIDE[b, r, s][m].sum())
            table.append(dict(b=b, r=r, seed=s, bin=g * 0.05, n=int(m.sum()),
                              J_mean=J[m].mean(), expected=p[m].sum(), lo=lo,
                              hi=hi, observed=obs, inside=lo <= obs <= hi))
SC = pd.DataFrame(table)
inf = SC[SC.n >= 30]
per = inf.groupby(['b', 'r']).inside.mean()
n_out, n_cells = int((~inf.inside).sum()), len(inf)
bound = stats.binom.ppf(0.999, n_cells, 0.05)
say(f'{n_cells} (setting, bin, seed) cells with n >= 30: {n_out} outside '
    f'their 95% interval (a correct hash: about 5% or fewer, at most '
    f'{bound:.0f} at the 0.999 level); share inside per (b, r): '
    + ', '.join(f'{b}x{r} {v:.2f}' for (b, r), v in per.items()))
assert n_out <= bound, 'the candidate rates depart from the S-curve'
show = SC[(SC.b == 9) & (SC.r == 13) & (SC.n >= 30)].pivot_table(
    index=['bin', 'n', 'J_mean', 'expected', 'lo', 'hi'], columns='seed',
    values='observed').reset_index()
say('\n(9, 13): observed candidates per seed against the expected count '
    'and its interval')
say(show.round(3).to_string(index=False))

s_grid = np.linspace(0, 1, 501)
theory = [(14, 8), (9, 13), (20, 5), (8, 16), (450, 20)]
d2l.set_figsize((5, 3.5))
ax = d2l.plt.gca()
d2l.plot_intervals(s_grid, {f'{b} x {r}': (d2l.scurve(s_grid, b, r),) * 3
                            for b, r in theory}, 'exact Jaccard of the pair',
                   'P(candidate)', kind='band', axes=ax)
ax.axvline(TAU, linestyle='--', color='black', linewidth=0.8,
           label=f'threshold {TAU}')
pooled = SC[SC.n >= 30].groupby(['b', 'r', 'bin']).agg(
    J_mean=('J_mean', 'first'), n=('n', 'first'), obs=('observed', 'sum'),
    exp=('expected', 'first')).reset_index()
xs = pooled[(pooled.b == 14) & (pooled.r == 8)].J_mean.to_numpy()
series, exp_marks = {}, {}
for b, r in ((14, 8), (9, 13)):
    q = pooled[(pooled.b == b) & (pooled.r == r)]
    ci = np.array([d2l.wilson_ci(o, 3 * n) for o, n in zip(q.obs, q.n)])
    series[f'{b} x {r} measured'] = (q.obs / (3 * q.n), ci[:, 0], ci[:, 1])
    exp_marks[f'{b} x {r} measured'] = (xs, q.exp / q.n)
d2l.plot_intervals(xs, series, 'exact Jaccard of the pair', 'P(candidate)',
                   axes=ax, marks={'expected (bin mean of the curve)': (
                       'x', exp_marks)})
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))  # beside the axes
R['scurve_measure'] = dict(n_cells=n_cells, n_outside=n_out,
                           inside=per.round(4).to_dict(),
                           table=SC.round(6).to_dict('list'))
```

Of the 780 cells with at least 30 pairs, 19 fall outside their interval, 2.4%,
against an upper bound of 59 at the 0.999 level. Every setting keeps at least
93% of its cells inside. In the table for $(9, 13)$ the bin from 0.80 to 0.85
holds 813 pairs and expects 445 candidates, with an interval from 418 to 473,
and the three seeds give 431, 438 and 429; below 0.5 no pair becomes a
candidate at all. The figure shows the same agreement against five theoretical
curves, among them RefinedWeb's 450 bands of 20 rows
:cite:`penedo2023refinedweb`, and the bucket pipeline makes the same
decisions as the pair-by-pair test. The intervals treat the pairs of a bin as
independent, although the three copies of one document share shingles, but
the margin to the bound is wide. The sketch does what the derivation says.

Now we choose $(b, r)$ like a hyperparameter. On the training pairs we count
false positives and false negatives of each setting, averaged over the three
seeds, and take the setting with the fewest errors when a miss weighs
$w_{\textrm{fn}} = 1$ and when it weighs 5. On the test pairs we report
precision, recall and F1, and we compare the chosen setting with its
neighbours on the grid and with the optimum of :eqref:`eq_lm_dedup_br` by a
bootstrap that resamples base documents, since a document and its copies move
together.

```{.python .input #dedup-choose}
%%tab pytorch
y = PAIRS.label.to_numpy()
cluster = PAIRS.a.to_numpy()  # a base document and its copies, together
split = {name: (PAIRS.split == name).to_numpy() for name in ('train', 'test')}

def prf(pred, m):
    tp, fp, fn = (np.sum(v[m]) for v in (pred & y, pred & ~y, ~pred & y))
    return dict(P=tp / max(tp + fp, 1), R=tp / max(tp + fn, 1),
                F1=2 * tp / max(2 * tp + fp + fn, 1), FP=fp, FN=fn)

rows = []
for (b, r) in CONFIGS + ([DS_BR] if DS_BR not in CONFIGS else []):
    if (b, r, 0) not in DECIDE:  # datasketch's setting, if not on the grid
        for s in SEEDS:
            Sg, ok = d2l._sig_matrix(PSIGS[0xD2 + s], K)
            DECIDE[b, r, s] = (d2l._band_match(Sg[ia], Sg[ib], b, r)
                               & ok[ia] & ok[ib])
    for name, m in split.items():
        per_seed = pd.DataFrame([prf(DECIDE[b, r, s], m) for s in SEEDS])
        rows.append(dict(b=b, r=r, split=name, **per_seed.mean().to_dict(),
                         F1_min=per_seed.F1.min(), F1_max=per_seed.F1.max()))
GRID = pd.DataFrame(rows)
train = GRID[(GRID.split == 'train') & GRID.r.isin(R_GRID)
             & (GRID.b * GRID.r <= K) & (GRID.b == K // GRID.r)]
CHOSEN = {}
for w in (1, 5):
    err = train.FP + w * train.FN
    best_row = train.loc[err.idxmin()]
    CHOSEN[w] = (int(best_row.b), int(best_row.r))
say('the grid, means over three seeds (FP, FN: counts on the split):')
say(GRID.round(3).to_string(index=False))
steps = abs(R_GRID.index(CHOSEN[1][1]) - R_GRID.index(BR[1]))
say(f'\nchosen on train: w_fn = 1 -> {CHOSEN[1]}, w_fn = 5 -> '
    f'{CHOSEN[5]}; the D3 optimum {BR} is {steps} grid step(s) from the '
    f'w_fn = 1 choice; FineWeb (14, 8) on 112 rows and datasketch '
    f'{DS_BR} are rows of the table')

def f1_diff_ci(pa, pb, m, n_boot=2000, seed=0):
    """F1(pa) - F1(pb) on the pairs m with a cluster-bootstrap 95% interval
    (base documents resampled; per-pair decisions averaged over seeds)."""
    g = np.unique(cluster[m], return_inverse=True)[1]
    C = np.random.default_rng(seed).multinomial(
        g.max() + 1, np.full(g.max() + 1, 1 / (g.max() + 1)), size=n_boot)
    def f1(pred):
        yy = y[m].astype(float)
        tp, fp, fn = (np.bincount(g, v) for v in (
            pred * yy, pred * (1 - yy), (1 - pred) * yy))
        point = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
        return point, 2 * (C @ tp) / (2 * (C @ tp) + C @ fp + C @ fn)
    (fa, ra), (fb, rb) = f1(pa[m]), f1(pb[m])
    return fa - fb, np.quantile(ra - rb, [0.025, 0.975])

def mean_decision(b, r):
    return np.mean([DECIDE[b, r, s] for s in SEEDS], axis=0)

def compare(br, br2, w):
    """F1(br) - F1(br2) and the weighted errors per pair that br saves
    against br2, on the test pairs, with cluster-bootstrap intervals."""
    pa, pb = mean_decision(*br), mean_decision(*br2)
    d, (lo, hi) = f1_diff_ci(pa, pb, split['test'])
    cost = lambda p: p * ~y + w * (1 - p) * y  # weighted errors per pair
    eb = d2l.paired_bootstrap(cost(pb)[split['test']],
                              cost(pa)[split['test']], n_boot=2000,
                              cluster=cluster[split['test']])
    say(f'test, w_fn = {w}: F1 of {br[0]} x {br[1]} minus {br2[0]} x '
        f'{br2[1]}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]; weighted errors per '
        f'pair saved: {eb.mean:+.4f} [{eb.ci[0]:+.4f}, {eb.ci[1]:+.4f}]')
    return [d, lo, hi, eb.mean, *eb.ci]

R['choose'] = dict(grid=GRID.round(6).to_dict('list'),
                   chosen={str(w): list(v) for w, v in CHOSEN.items()},
                   d3=list(BR), steps=steps, boot={}, expected={})
for w, (b, r) in CHOSEN.items():
    i = R_GRID.index(r)
    for r2 in R_GRID[max(i - 1, 0):i + 2]:
        if r2 != r:
            R['choose']['boot'][f'{w}:{r}-{r2}'] = compare(
                (b, r), (K // r2, r2), w)
R['choose']['boot']['1:chosen-d3'] = compare(CHOSEN[1], BR, 1)

# D3 weighs similarities uniformly on [0, 1]; the expected weighted error
# under the training pairs' own similarities is a sum over those pairs
J_tr, y_tr = PAIRS.J.to_numpy()[split['train']], y[split['train']]
for w in (1, 5):
    exp_err = {}
    for r in R_GRID:
        p = d2l.scurve(J_tr, K // r, r)
        exp_err[K // r, r] = float(np.sum(p * ~y_tr + w * (1 - p) * y_tr))
    arg = min(exp_err, key=exp_err.get)
    say(f'w_fn = {w}: expected weighted error on train under the pairs\' '
        f'J: least at {arg} ({exp_err[arg]:.1f}); at the D3 optimum '
        f'{BR}: {exp_err[BR]:.1f}; measured choice {CHOSEN[w]}')
    assert arg == CHOSEN[w], 'the hashed choice departs from its expectation'
    R['choose']['expected'][str(w)] = dict(argmin=list(arg),
                                           at_argmin=exp_err[arg],
                                           at_d3=exp_err[BR])

tr, te = (GRID[(GRID.split == n) & (GRID.b == K // GRID.r)
               & GRID.r.isin(R_GRID)] for n in ('train', 'test'))
ax = d2l.plot_intervals(
    np.array(R_GRID), {
        'train': (tr.F1, tr.F1_min, tr.F1_max),
        'test': (te.F1, te.F1_min, te.F1_max)},
    'rows per band r (b = 128 // r)', 'F1 against J >= 0.8',
    marks={f'chosen, w_fn = {w}': (m, {'train': (
        [CHOSEN[w][1]], tr.F1[tr.r == CHOSEN[w][1]].to_numpy() + 0.015)})
        for w, m in ((1, 'v'), (5, 'D'))})  # a marker above the point
_ = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))  # beside
```

With $w_{\textrm{fn}} = 1$ the training pairs choose $(11, 11)$, and with
$w_{\textrm{fn}} = 5$ they choose $(14, 9)$: a heavier weight on misses moves
the choice to fewer rows per band, as the derivation suggests. On the test
pairs $(11, 11)$ reaches precision 0.86, recall 0.89 and F1 0.87, ahead of the
equal-weight optimum $(9, 13)$ by $+0.04$ $[+0.03, +0.05]$ in F1. Its lead
over its neighbours on the grid is small: against $(12, 10)$ the F1 difference
is not resolved, although the error count it was chosen for is lower by a
small resolved margin. FineWeb's $(14, 8)$ has recall 0.99 at precision 0.72
on our labels, since a curve built for 0.75 counts many pairs between 0.7 and
0.8 as duplicates.

Why do the pairs disagree with :eqref:`eq_lm_dedup_br`? The hashes are not to
blame; the measurement above showed that. The integral weighs every similarity
between 0 and 1 equally, and the pairs do not: most negatives sit at $J = 0$,
where every setting is right, and the copies crowd around the threshold.
Replace the uniform weight by the pairs' own similarities, so that the
expected weighted error is $\sum_i [P(J_i)(1 - y_i) + w_{\textrm{fn}}(1 -
P(J_i))\, y_i]$, and its minimum lies at $(11, 11)$ and at $(14, 9)$,
the two settings that the hashed decisions chose; the cell asserts it. The
derivation is right about the curve and wrong about the data. A deployment
should tune $(b, r)$ on the similarities of the pairs it will actually decide,
and labelled pairs, even synthetic ones, are the cheapest way to see them.

A threshold that looks right on labelled pairs can still fail at scale. By
linearity of expectation, the expected number of candidates among $N$
documents is $\sum_{i < j} P(s_{ij})$, the number of pairs times the mean of
$P$ over their similarities. Nearly all pairs are dissimilar, so the total is
decided far below the threshold, by $P(0.3)$ and $P(0.5)$, and every candidate
from that bulk costs a lookup and a union like a true one. The count can also
explode through templates. Templated pages share whole bands, so a single
bucket can collect thousands of documents, and a bucket of $m$ documents
yields $m(m - 1)/2$ pairs. `lsh_candidates` therefore cuts any bucket above
`max_bucket` members to a seeded sample and reports every cut.

```{.python .input #dedup-cost}
%%tab pytorch
N = 10**6
for br in (CHOSEN[1], (14, 8)):
    p3, p5 = d2l.scurve(0.3, *br), d2l.scurve(0.5, *br)
    exp = N * (N - 1) / 2 * 1e-3 * p5
    say(f'{br}: P(0.3) = {p3:.2e}, P(0.5) = {p5:.4f}; N = 10^6 with 0.1% '
        f'of pairs at s = 0.5: {exp:.2e} expected candidate pairs')
t0 = time.perf_counter()
SLICE_SIGS = d2l._minhash_texts(SLICE.text, K, (0xD2,), 5,
                                N_WORKERS)[0xD2]
TIMES['slice_signatures'] = time.perf_counter() - t0
n = len(SLICE)
say(f'slice: {n:,} signatures in {TIMES["slice_signatures"]:.0f} s '
    f'({N_WORKERS} workers); candidate pairs, per 1,000 documents, and '
    'scaled by (N/n)^2 to N = 10^6 (a measured count, not a corpus '
    'estimate: FineWeb is already deduplicated within its dump):')
R['cost'] = dict(n=n)
for br in (CHOSEN[1], (14, 8), (32, 4)):
    cand, rep = d2l.lsh_candidates(SLICE_SIGS, *br)
    say(f'  {br}: {len(cand):,} pairs, {len(cand) / (n / 1000):.2f} per '
        f'1,000, {len(cand) * (N / n) ** 2:.2e} scaled; largest bucket '
        f'{rep["largest"]} members, {len(rep["cut"])} cut to max_bucket')
    R['cost'][f'{br[0]}x{br[1]}'] = dict(candidates=len(cand),
                                         largest_bucket=rep['largest'],
                                         cut=len(rep['cut']))
```

At $(11, 11)$ a pair of similarity 0.3 becomes a candidate with probability $2
\times 10^{-5}$, at FineWeb's $(14, 8)$ with about $10^{-3}$, fifty times
more. For a million documents with 0.1% of their pairs at similarity 0.5, the
two settings expect about $3 \times 10^6$ and $3 \times 10^7$ candidates. On
our slice neither setting produces a single candidate, and even the loose
$(32, 4)$ finds only 104 pairs, about four per thousand documents, with no
bucket above six members. This is a measurement of an already deduplicated
corpus, not an estimate for a raw crawl, which is where the giant
buckets live.

## From Pairs to a Deduplicated Corpus
:label:`subsec_lm_dedup_cluster`

Candidate pairs are not yet a corpus. Treat them as edges of a graph; the
*clusters* are its connected components, which `union_find_clusters` computes
in almost linear time. Clusters are transitive: if A resembles B and B
resembles C, all three land in one cluster even when A and C are below the
threshold, as with a chain of revisions of one page. Each cluster keeps one
document, and which one is a choice. We keep the one with the lowest index, so
that the result depends neither on the order of the pairs nor on a random
draw. Keeping the document with the highest quality score or the longest one
is also reasonable, and whichever rule is used belongs in the manifest. The
function `near_dedup` wraps shingling, signatures, banding and clusters in one
call, and :numref:`sec_lm_instruction_data` uses it on instruction data.

Deduplication also needs a scope. FineWeb deduplicates each Common Crawl dump
on its own :cite:`Penedo.Kydlicek.BenAllal.ea.2024`. Its authors tried
deduplicating all dumps together and found that it did not improve the models
trained on the result: in the older dumps, the documents that survived global
deduplication made worse training data than those it removed. Global
deduplication sounds stricter and therefore better, and on their evidence it
is not. Our corpus is one archive range of one dump, so the question returns
only in the last exercise.

```{.python .input #dedup-dedup-run}
%%tab pytorch
cand, rep = d2l.lsh_candidates(SLICE_SIGS, *BR)
LABELS = d2l.union_find_clusters(len(SLICE), cand)
keep = [i for i in range(len(SLICE)) if LABELS[i] == i]
t0 = time.perf_counter()
assert d2l.near_dedup(SLICE.text.tolist(), TAU, workers=N_WORKERS) == keep
TIMES['near_dedup'] = time.perf_counter() - t0
size = collections.Counter(LABELS.tolist())
FLAGGED = np.array([size[c] > 1 for c in LABELS])  # in a cluster of two+
say(f'slice, near_dedup(threshold={TAU}) at {BR}: {len(keep):,} of '
    f'{len(SLICE):,} kept; by role: ' + ', '.join(
        f'{role} {FLAGGED[(SLICE.role == role).to_numpy()].mean():.2%} in '
        'clusters' for role in ('stream', 'host_pair')))
R['dedup_run'] = dict(slice_kept=len(keep), slice_largest_bucket=rep[
    'largest'])

# our own crawl after quality filtering (c1) has not been deduplicated:
# clusters by size, with host counts and Jaccard ranges (no text, rule P1),
# at the setting chosen on the training pairs, which the manifest uses
c1_text = [d.text for d in C1]
c1_sigs = [d2l.minhash_signature(d2l.shingles(t), K) for t in c1_text]
c1_cand, c1_rep = d2l.lsh_candidates(c1_sigs, *CHOSEN[1])
c1_lab = d2l.union_find_clusters(len(C1), c1_cand)
c1_keep = [i for i in range(len(C1)) if c1_lab[i] == i]
c1_keep_d3 = d2l.near_dedup(c1_text, TAU)  # its default (b, r): D3's
c1_size = collections.Counter(c1_lab.tolist())
multi = {c: k for c, k in c1_size.items() if k > 1}
say(f'c1 at {CHOSEN[1]} (chosen on train; the manifest below): '
    f'{len(c1_keep):,} of {len(C1):,} kept; {len(multi)} clusters of two '
    f'or more ({sum(multi.values())} documents); largest bucket '
    f'{c1_rep["largest"]}, cut buckets: {c1_rep["cut"] or "none"}')
say(f'c1 by near_dedup(threshold={TAU}) at its default {BR}: '
    f'{len(c1_keep_d3):,} kept')
largest = sorted(multi, key=lambda c: (-multi[c], C1[c].id))[:5]
c1_sets = {}
for c in largest:
    members = np.flatnonzero(c1_lab == c)
    for i in members:
        c1_sets.setdefault(i, set(d2l.shingles(C1[i].text)))
    Js = [d2l.jaccard_exact(c1_sets[c], c1_sets[i]) for i in members[1:]]
    hosts = {d2l._url_host(C1[i].url) for i in members}
    say(f'  {len(members):>3} documents on {len(hosts)} host(s), J to the '
        f'representative {min(Js):.2f}-{max(Js):.2f}')
R['dedup_run'].update(c1_kept=len(c1_keep), c1_kept_d3=len(c1_keep_d3),
                      c1_clusters=len(multi),
                      c1_in_clusters=sum(multi.values()),
                      c1_largest=[multi[c] for c in largest])
```

The slice keeps all 26,000 documents, as expected after FineWeb's own
deduplication with a looser setting. Our crawl is different. At $(11, 11)$,
the setting chosen on the training pairs and the one the manifest uses, c1
keeps 1,014 of its 1,032 documents. Its 13 clusters hold 31 documents; the
largest is four copies of one page on three hosts, and the next ones are
copies within one site. The default of `near_dedup` is the equal-weight
optimum $(9, 13)$, because a library cannot know the reader's pairs, and it
keeps one document more.

### Substring Duplicates
:label:`subsec_lm_dedup_substring`

MinHash compares whole documents, and some duplication lives in parts of them.
A footer of 60 words shared by two otherwise different pages of a few hundred
words each adds about 56 shared shingles and leaves $J$ near 0.1. No setting
of $(b, r)$ will flag such pairs, and none should, because the pages are
different, yet a model reads a footer on thousands of pages thousands of
times. Lee and colleagues (2022) found in C4, for example, one sentence of 61
words that occurred more than 60,000 times. Their *exact-substring*
deduplication finds every span of at least 50 tokens that occurs more than
once and removes all copies but one.

The tool is a *suffix array*: the starting positions of all suffixes of the
token sequence, sorted lexicographically. Two occurrences of a span start two
suffixes that begin with it, so they sort next to each other, and the repeats
appear as long common prefixes of neighbouring suffixes. The array `lcp`
stores the length of the prefix that each suffix in sorted order shares with
the previous one, and every entry of at least 50 marks a span of 50 tokens
that occurs twice. *Prefix doubling* sorts by the first token, then by the
first two as a pair of ranks, then by the first four, each round one sort of
integer pairs, so $\log n$ rounds of $O(n \log n)$ finish the job. Kasai's
algorithm computes `lcp` in $O(n)$, because the common prefix shrinks by at
most one from the suffix at $i$ to the one at $i + 1$. The production tool of
Lee and colleagues is written in Rust and builds the array over the whole
corpus, with memory that grows with it. Ours runs in NumPy on 2 MiB of the
slice, tokenized with the reference tokenizer of :numref:`sec_lm_tokenizers`,
with a unique separator after every document so that no repeat crosses a
document boundary.

```{.python .input #dedup-sa-def}
%%tab pytorch
#@save
def suffix_array(ids):
    """The suffix array of an integer sequence by prefix doubling: sa[i] is
    the start of the i-th smallest suffix (a suffix that is a prefix of
    another sorts first). Ranks by the first 2^k tokens, sorted with
    np.lexsort, until all ranks differ: O(n log^2 n)."""
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rank = np.unique(ids, return_inverse=True)[1].astype(np.int64)
    k = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        second[:n - k] = rank[k:] if k < n else second[:0]
        sa = np.lexsort((second, rank))
        r1, r2 = rank[sa], second[sa]
        new = np.empty(n, dtype=np.int64)
        new[sa] = np.concatenate([[0], np.cumsum(
            (r1[1:] != r1[:-1]) | (r2[1:] != r2[:-1]))])
        rank = new
        if rank[sa[-1]] == n - 1 or k >= n:
            return sa
        k *= 2

#@save
def lcp_array(ids, sa):
    """lcp[i] = the length of the longest common prefix of the suffixes
    sa[i - 1] and sa[i] (lcp[0] = 0), by Kasai's algorithm in O(n)."""
    s, sa = [int(v) for v in ids], [int(v) for v in sa]
    n = len(s)
    rank = [0] * n
    for i, p in enumerate(sa):
        rank[p] = i
    lcp, h = [0] * n, 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            h = max(h - 1, 0)
        else:
            h = 0
    return np.array(lcp, dtype=np.int64)

#@save
def repeated_spans(ids, sa, lcp, min_len):
    """The maximal repeats of at least min_len tokens, as (start, length,
    count): every interval of the suffix array whose suffixes share a
    prefix of that length (found with a stack over lcp), kept if extending
    it one token to the left would not keep all its occurrences equal.
    start is the first position of an occurrence; longest first."""
    n, out, stack = len(sa), [], [(0, 0)]  # (lcp value, left bound)
    for i in range(1, n + 1):
        cur, lb = (int(lcp[i]) if i < n else 0), i - 1
        while cur < stack[-1][0]:
            length, lb = stack.pop()
            if length >= min_len:
                occ = sa[lb:i]
                left = {int(ids[p - 1]) if p > 0 else None for p in occ}
                if len(left) > 1 or None in left:
                    out.append((int(occ.min()), length, len(occ)))
        if cur > stack[-1][0]:
            stack.append((cur, lb))
    return sorted(out, key=lambda t: (-t[1], -t[2], t[0]))
```

```{.python .input #dedup-substring}
%%tab pytorch
rng = np.random.default_rng(0)
for _ in range(20):  # against the definition on short random strings
    x = rng.integers(0, 3, int(rng.integers(1, 400)))
    sa = d2l.suffix_array(x)
    assert sa.tolist() == sorted(range(len(x)), key=lambda i: x[i:].tolist())
    naive = [0] + [next((h for h in range(len(x)) if h >= len(x) - max(
        sa[i], sa[i - 1]) or x[sa[i] + h] != x[sa[i - 1] + h]))
                   for i in range(1, len(x))]
    assert d2l.lcp_array(x, sa).tolist() == naive

TOK = d2l.r2_tokenizer()  # the reference tokenizer (sec_lm_tokenizers)
order = SLICE.sort_values('id')
nbytes = np.cumsum([len(t.encode('utf-8')) for t in order.text])
sample = order.iloc[:int(np.searchsorted(nbytes, 2 * 2**20)) + 1]
doc_ids = d2l._doc_token_ids(TOK, sample.text.tolist())
seq = np.concatenate([np.append(np.asarray(t, dtype=np.int64), -1 - d)
                      for d, t in enumerate(doc_ids)])  # unique separators
starts = np.cumsum([0] + [len(t) + 1 for t in doc_ids])[:-1]
t0 = time.perf_counter()
sa = d2l.suffix_array(seq)
lcp = d2l.lcp_array(seq, sa)
spans = d2l.repeated_spans(seq, sa, lcp, 50)
TIMES['suffix_array'] = time.perf_counter() - t0
longest = np.maximum(lcp, np.append(lcp[1:], 0))  # per SA rank
cover = np.zeros(len(seq) + 1, dtype=np.int64)
for p, m in zip(sa[longest >= 50], longest[longest >= 50]):
    cover[p] += 1
    cover[p + m] -= 1
covered = np.cumsum(cover[:-1]) > 0
doc_of = np.searchsorted(starts, np.arange(len(seq)), side='right') - 1
n_cov = np.bincount(doc_of[covered], minlength=len(doc_ids))
touched = n_cov >= 50
flag = dict(zip(SLICE.id, FLAGGED))
in_mh = np.array([flag[i] for i in sample.id])
say(f'{len(sample):,} documents, {nbytes[len(sample) - 1] / 2**20:.2f} '
    f'MiB, {len(seq) - len(doc_ids):,} tokens: suffix array and LCP in '
    f'{TIMES["suffix_array"]:.1f} s; {len(spans):,} maximal repeats of 50+ '
    f'tokens covering {covered.mean():.2%} of the tokens')
for start, length, count in spans[:3]:
    d = doc_of[start]
    text = TOK.decode(seq[start:start + length].tolist()).replace('\n', ' ')
    say(f'  {length} tokens x {count} (from {sample.id.iloc[d][10:18]}…): '
        f'{text[:100]!r}')
say(f'documents with 50+ repeated tokens: {touched.sum()}; of these in a '
    f'MinHash cluster: {in_mh[touched].sum()}; MinHash-clustered '
    f'documents in the sample: {in_mh.sum()}')
say(ATTRIBUTION)
R['substring'] = dict(docs=len(sample), tokens=len(seq) - len(doc_ids),
                      spans=len(spans), covered=covered.mean(),
                      longest=[list(t[1:]) for t in spans[:3]],
                      touched=int(touched.sum()),
                      touched_minhash=int(in_mh[touched].sum()))
```

The sample has 627 documents and about 450,000 tokens, and the arrays take a
second or two. It contains three maximal repeats of at least 50 tokens,
covering 0.08% of its tokens: a blog's footer inviting readers to newer posts,
a function signature from a software manual, and the Fourth Amendment to the
Constitution of the United States. None of the four documents involved is in a
MinHash cluster. A sample this small cannot contain the 60,000-fold sentences,
but it shows the kind, and it shows a trap. Exact-substring deduplication
would delete the amendment from one of the two pages that quote it, whatever
that page was about. The same trap returns in decontamination.

### Semantic Duplicates
:label:`subsec_lm_dedup_semantic`

The third relative works in a different space. *SemDeDup* (Abbas and
colleagues, 2023) embeds every document, clusters the vectors with k-means,
and within each cluster removes a document whose cosine similarity to another
member exceeds $1 - \varepsilon$. The clusters do the job that bands do for
MinHash: they confine comparisons to documents that are already close, so a
pass costs $O(n^2/k)$ instead of $O(n^2)$ for $k$ clusters. The one difference
from MinHash is the similarity. Two texts close in embedding space can share
few words, and this is the point of the method. We use the Qwen3-Embedding
vectors that :numref:`sec_lm_quality_filtering` registered for the whole
stream (:numref:`sec_lm_text_embeddings` describes what such vectors encode),
the k-means of :numref:`sec_lm_retrieval` with 64 clusters, and remove a
document when an earlier member of its cluster is closer than the threshold.

```{.python .input #dedup-semantic}
%%tab pytorch
emb = np.load(d2l.lm_artifact('lm-data-emb-qwen3emb-v1'))
assert (emb['stream_ids'] == STREAM_S.id.to_numpy().astype(str)).all()
sub = np.arange(len(STREAM_S))  # the whole stream (vectors registered)
X = emb['stream'][sub].astype(np.float64)  # k-means casts to float32
X /= np.linalg.norm(X, axis=1, keepdims=True)
_, assign = d2l.kmeans(X, 64, seed=0)
assign = assign.numpy()
best_sim, partner = np.full(len(sub), -1.0), np.full(len(sub), -1)
for c in range(64):  # within a cluster: the most similar earlier document
    m = np.flatnonzero(assign == c)
    if len(m) < 2:
        continue
    sim = X[m] @ X[m].T
    sim[np.triu_indices(len(m))] = -np.inf
    best_sim[m[1:]] = sim[1:].max(axis=1)
    partner[m[1:]] = m[sim[1:].argmax(axis=1)]
mh = np.array([flag[i] for i in STREAM_S.id.iloc[sub]])
q = np.quantile(best_sim[best_sim > -1], [0.5, 0.99, 1.0])
say(f'{len(sub):,} stream documents in 64 clusters; cosine to the most '
    f'similar earlier member: median {q[0]:.3f}, 99th percentile '
    f'{q[1]:.3f}, largest {q[2]:.3f}')
rows, before = [], np.zeros(len(sub), dtype=bool)
for eps in (0.05, 0.1, 0.15, 0.2, 0.3):
    gone = best_sim > 1 - eps
    rows.append(dict(eps=eps, removed=int(gone.sum()), fraction=gone.mean(),
                     also_minhash=int((gone & mh).sum())))
    new = np.flatnonzero(gone & ~before)  # removed first at this eps
    before = gone
    for j in new[np.argsort(best_sim[new])][:2]:  # nearest the threshold
        x, z = STREAM_S.id.iloc[sub[j]], STREAM_S.id.iloc[sub[partner[j]]]
        J_xz = d2l.jaccard_exact(d2l.shingles(TEXT[x]), d2l.shingles(TEXT[z]))
        say(f'eps {eps}: cosine {best_sim[j]:.3f}, J {J_xz:.2f}\n    '
            f'{TEXT[x][:90]!r}\n    {TEXT[z][:90]!r}')
SEM = pd.DataFrame(rows)
say(SEM.round(5).to_string(index=False))
say(f'documents in MinHash clusters among them: {mh.sum()}')
say(ATTRIBUTION)
R['semantic'] = dict(SEM.round(6).to_dict('list'), n=len(sub),
                     cos_quantiles=q.round(6).tolist())
```

The median cosine of a document to its most similar earlier neighbour is 0.46,
the 99th percentile 0.70 and the largest 0.94. These numbers belong to this
embedder, and so must $\varepsilon$: a threshold that removes nothing here
could remove a great deal under a model whose cosines between unrelated texts
sit at 0.8, and SemDeDup's values do not transfer to ours. At $\varepsilon =
0.1$ the method removes one document, a genuine near-duplicate that MinHash
missed: two pages from one template about signing files, at cosine 0.94 and $J
= 0.17$, whose wording was varied phrase by phrase. From $\varepsilon = 0.15$
on it removes something else, different pages on one topic: online gambling,
essay-writing services, plumbers' advice on frozen pipes, surfing in Ericeira,
all at $J$ near 0. At $\varepsilon = 0.3$ it removes 181 documents, 0.9% of
the stream. Whether a second article about frozen pipes should go is a
question about how much of each topic a corpus should hold. D4 (Tirumala and
colleagues, 2023) builds on SemDeDup to select documents for diversity, and
:numref:`sec_lm_data_mixtures` takes up the question of proportions with
training runs.

## Decontamination
:label:`subsec_lm_dedup_decontam`

The duplicates that matter most are not pages copied from each other but test
items copied into training data. :numref:`subsec_lm_eval_contamination` looked
at this from the model's side, comparing accuracy on GSM8K problems with
accuracy on variants of them. Here we look from the data's side:
*decontamination* removes training text that overlaps evaluation items before
any model sees it. The authors of GPT-3 :cite:`brown2020language` looked for
overlapping sequences of 13 words, ignoring case and punctuation, between
their training data and their test sets and removed overlapping documents from
training. A bug in the filter left part of the overlap in place, and their
paper measured its effect rather than retrain. We build the same test as an
index: every 13-word sequence of every test item, normalized like the shingles
and hashed, is stored with the item's id, and a document is looked up by its
own 13-word sequences.

The index misses some things by construction. A paraphrase shares no long
sequence with its source, a translation shares none at all, a problem with
changed numbers breaks every 13-gram that contains a number, and an item
shorter than 13 words has no 13-gram to find. To measure what it finds we need
leaks whose location we know, so we plant them. We draw 100 test items from
each of GSM8K, ARC-Challenge and MMLU and insert each in three forms into 900
different documents of the stream: *verbatim* (the question, its lettered
options and an answer line), *reformatted* by rule (lower case, spacing and
punctuation changed, options in reverse order with new letters), and
*paraphrased* by a language model. A further 1,000 documents stay untouched as
controls. The paraphrases themselves are examined after the two detectors.

```{.python .input #dedup-index-def}
%%tab pytorch
#@save
class NgramIndex:
    """Hashed word n-grams of indexed texts, for decontamination: a key is
    blake2b (8 bytes) of n consecutive NFC-lowercased \\w+ words joined by
    one space, so casing, punctuation and spacing do not matter."""
    def __init__(self, n=13):
        self.n, self.keys = n, {}

    def _grams(self, text):
        w = d2l.shingles(text, 1)  # the words, as the shingles see them
        return {hashlib.blake2b(' '.join(w[i:i + self.n]).encode('utf-8'),
                                digest_size=8).digest()
                for i in range(len(w) - self.n + 1)}

    def add(self, text, tag):
        for g in self._grams(text):
            self.keys.setdefault(g, set()).add(tag)
        return self

    def query(self, text):
        """[(tag, number of the text's distinct n-grams in tag's text)],
        most hits first."""
        hits = collections.Counter()
        for g in self._grams(text):
            hits.update(self.keys.get(g, ()))
        return sorted(hits.items(), key=lambda t: (-t[1], str(t[0])))
```

```{.python .input #dedup-decontam-ngram}
%%tab pytorch
t0 = time.perf_counter()
ITEMS = d2l.benchmark_items()
ITEM_POS = {it['item_id']: i for i, it in enumerate(ITEMS)}
INDEX = d2l.NgramIndex(N_GRAM)
for it in ITEMS:
    INDEX.add(it['text'], it['item_id'])
TIMES['index'] = time.perf_counter() - t0
words = lambda t: len(d2l.shingles(t, 1))
short = collections.Counter(it['benchmark'] for it in ITEMS
                            if words(it['text']) < N_GRAM)
qshort = collections.Counter(it['benchmark'] for it in ITEMS
                             if words(it['question']) < N_GRAM)
n_items = collections.Counter(it['benchmark'] for it in ITEMS)
assert dict(n_items) == dict(gsm8k=1319, arc=1172, mmlu=14042), n_items
say(f'{len(ITEMS):,} test items ({dict(n_items)}), {len(INDEX.keys):,} '
    f'distinct {N_GRAM}-grams, built in {TIMES["index"]:.0f} s')
for bench in n_items:
    say(f'  {bench}: {short[bench]} items with fewer than {N_GRAM} words '
        f'(question and answer), {qshort[bench]} with a shorter question')
R['decontam'] = dict(items=dict(n_items), short=dict(short),
                     short_question=dict(qshort))
PLANTS_KEY, PLANTS_FILE = ARTIFACTS['dedup-plants']
if PLANTS_PENDING:
    say('the plants are made in this run (below); the detectors run '
        'once they are registered')
else:
    PLANTS_PATH = d2l.lm_artifact(PLANTS_KEY)
    with open(PLANTS_PATH, encoding='utf-8') as f:
        PLANTS = [json.loads(line) for line in f]
    DOCS = d2l.contam_documents(PLANTS, STREAM_S)
    HITS = {r['doc_id']: INDEX.query(DOCS[r['doc_id']]) for r in PLANTS}
    PL = pd.DataFrame(PLANTS)
    PL['ngram_found'] = [r['item_id'] in dict(HITS[r['doc_id']])
                         for r in PLANTS]
    PL['ngram_any'] = [bool(HITS[r['doc_id']]) for r in PLANTS]
    rows = []
    for (bench, form), g in PL[PL.form != 'control'].groupby(
            ['benchmark', 'form']):
        k = int(g.ngram_found.sum())
        rows.append(dict(benchmark=bench, form=form, n=len(g), found=k,
                         recall=k / len(g), lo=d2l.wilson_ci(k, len(g))[0],
                         hi=d2l.wilson_ci(k, len(g))[1]))
    DECON = pd.DataFrame(rows)
    ctrl = PL[PL.form == 'control']
    fp = int(ctrl.ngram_any.sum())
    say('\n13-gram index: planted items found, by benchmark and form')
    say(DECON.round(3).to_string(index=False))
    say(f'controls flagged: {fp} of {len(ctrl)} (Wilson 95% interval '
        f'{d2l.wilson_ci(fp, len(ctrl))[0]:.4f}-'
        f'{d2l.wilson_ci(fp, len(ctrl))[1]:.4f})')
    for _, r in ctrl[ctrl.ngram_any].iterrows():  # look at the span
        tag = HITS[r.doc_id][0][0]
        grams = set(d2l.shingles(ITEMS[ITEM_POS[tag]]['text'], N_GRAM))
        span = next(g for g in d2l.shingles(DOCS[r.doc_id], N_GRAM)
                    if g in grams)
        say(f'  control {r.doc_id[10:18]}… shares {HITS[r.doc_id][0][1]} '
            f'{N_GRAM}-gram(s) with {tag}, e.g. {span!r}')
    exact_forms = DECON[DECON.form.isin(['verbatim', 'reformat'])]
    assert (exact_forms.found == exact_forms.n).all(), 'a missed plant: a bug'
    R['decontam'].update(ngram=DECON.round(6).to_dict('list'),
                         ngram_fp=fp, n_controls=len(ctrl))
```

The index holds 633,848 distinct 13-grams of 16,533 test items, and it has
blind spots before any plant is placed: 102 ARC-Challenge items and 1,589 MMLU
items are shorter than 13 words even with their answer, and 304 and 4,080 of
their questions are shorter on their own. No leak of these can be found this
way. The plants were drawn from items whose question has at least 13 words, so
that a missed verbatim plant would be a bug.

All 300 verbatim and all 300 reformatted plants are found. The second result
says less than it seems. The index ignores case, spacing and punctuation, and
reversing the options keeps each option's words, so the reformatted form is
found by construction; its recall of 1 shows that the normalization works. The
paraphrases are a different matter, 0, 2 and 6 of 100. A 13-gram detector
finds copies and nothing else.

One control is flagged. The page quotes "we hold these truths to be self
evident that all men are created", and so does MMLU item 3828. A rule that
removes every overlapping document deletes every page that quotes the
Declaration of Independence or a famous poem, because benchmarks quote them
too. Before removing a document, look at the span.

Why do leaked items and duplicates deserve this effort? Language models
reproduce text they have seen often. Carlini and colleagues
:cite:`Carlini.Tramer.Wallace.ea.2021` extracted hundreds of verbatim
sequences from GPT-2's training data, among them names, phone numbers and
code, and later work by the same group found that the share of training data a
model reproduces grows with the model's size, with the number of copies of a
sequence and with the length of the prompt. Lee and colleagues found that
models trained on their deduplicated corpora emit memorized text about ten
times less often. Deduplication is a privacy measure as well as a quality
measure, and :numref:`sec_lm_safety` measures extraction on a model whose
training data is known. An overlap, however, is evidence of leakage and not of
memorization: a model can see an item once and not learn it, and what a leak
did to a score can only be measured on the model, as in
:numref:`subsec_lm_eval_contamination`.

A paraphrase keeps the meaning of an item, and meaning is what embeddings
measure. The second detector cuts every document into windows of 128 words
starting every 64 words, so that a short item is not lost in a long page, and
scores each window by the cosine of its closest test item; a document's score
is its best window's. The threshold is the 99th percentile of the 1,000
controls, a false-positive budget of 1%, and since a threshold fits the
documents it was set on, the false positives are counted on 1,000 other
untouched documents. The fair comparison for the paraphrases is the verbatim
form under the same detector, its ceiling, not a recall of 1. The vectors were
made once on a GPU; the cell checks 16 of them on the CPU.

```{.python .input #dedup-decontam-emb}
%%tab pytorch
EMB_KEY, EMB_FILE = ARTIFACTS['dedup-emb']
LIVE_COS_MIN = 0.999  # CPU fp32 against the registered GPU vectors
SIZE, STRIDE = 128, 64  # words per window, words between window starts
if not PLANTS_PENDING:
    # held-out controls: the threshold is set on the plants file's controls
    # and its false positives are counted on 1,000 other untouched documents
    used = {r['doc_id'] for r in PLANTS}
    HOLDOUT = sorted((i for i in STREAM_S.id if i not in used), key=lambda
                     i: hashlib.sha256(i.encode('utf-8')).hexdigest())[:1000]
    WIN = [(r['doc_id'], s, t) for r in PLANTS
           for s, t in d2l.word_windows(DOCS[r['doc_id']], SIZE, STRIDE)]
    WIN += [(i, s, t) for i in HOLDOUT
            for s, t in d2l.word_windows(TEXT[i], SIZE, STRIDE)]
    with open(PLANTS_PATH, 'rb') as f:
        plants_sha1 = hashlib.sha1(f.read()).hexdigest()
    if 'dedup-emb' in RERUN:
        t0 = time.perf_counter()
        arrays = dict(
            item_ids=np.array([it['item_id'] for it in ITEMS]),
            items=d2l.embed_docs([it['text'] for it in ITEMS], EMBEDDER),
            win_doc=np.array([w[0] for w in WIN]),
            win_start=np.array([w[1] for w in WIN], dtype=np.int64),
            windows=d2l.embed_docs([w[2] for w in WIN], EMBEDDER),
            holdout_ids=np.array(HOLDOUT), plants_sha1=np.array(plants_sha1),
            made_by=np.array(f'{EMBEDDER}@'
                             f'{d2l.HF_ENCODER_REVISIONS[EMBEDDER]} '
                             'max_len=512 pool=last bf16'))
        say(f'embedded {len(ITEMS):,} items and {len(WIN):,} windows in '
            f'{time.perf_counter() - t0:.0f} s, {d2l.lm_peak()}')
        def write(path):
            members = {}
            for k, v in arrays.items():
                buf = io.BytesIO()
                np.save(buf, v, allow_pickle=False)
                members[k + '.npy'] = buf.getvalue()
            d2l.lm_write_zip(path, members)
        d2l.lm_save_rerun(EMB_FILE, write)
    E = dict(np.load(d2l.lm_artifact(
        EMB_KEY, EMB_FILE if 'dedup-emb' in RERUN else None)))
    assert E['item_ids'].tolist() == [it['item_id'] for it in ITEMS]
    assert E['win_doc'].tolist() == [w[0] for w in WIN]
    assert E['win_start'].tolist() == [w[1] for w in WIN]
    assert E['holdout_ids'].tolist() == HOLDOUT
    assert str(E['plants_sha1']) == plants_sha1, 'vectors of other plants'
    # float64 products: the same on any CPU and thread count up to rounding
    Iv, Wv = E['items'].astype(np.float64), E['windows'].astype(np.float64)
    top = np.concatenate([(w @ Iv.T).max(axis=1) for w in np.array_split(
        Wv, max(1, len(Wv) // 2048))])  # each window's best item
    doc_top = pd.Series(top).groupby(E['win_doc']).max()
    win_of = collections.defaultdict(list)
    for j, d in enumerate(E['win_doc'].tolist()):
        win_of[d].append(j)
    PL['emb_top'] = PL.doc_id.map(doc_top)
    planted = PL[PL.form != 'control'].copy()
    planted['emb_own'] = [  # the document's best window against its item
        float((Wv[win_of[r.doc_id]] @ Iv[ITEM_POS[r.item_id]]).max())
        for _, r in planted.iterrows()]
    planted['words'] = [len(t.split()) for t in planted.text]
    first = [len(DOCS[r.doc_id][:int(r.char_start)].split())
             for _, r in planted.iterrows()]  # the plant's first word
    planted['fits'] = [any(s <= w0 and w0 + n <= s + SIZE
                           for s in E['win_start'][win_of[d]])
                       for d, w0, n in zip(planted.doc_id, first,
                                           planted.words)]
    calib = PL[PL.form == 'control'].emb_top
    THR = float(np.quantile(calib, 0.99))
    held = doc_top.loc[HOLDOUT]
    fp = int((held > THR).sum())
    fp_lo, fp_hi = d2l.wilson_ci(fp, len(held))
    planted['found'] = planted.emb_own > THR
    rows = []
    for _, g in itertools.chain(
            planted.groupby(['form', 'fits']),
            ((k, g.assign(fits='all')) for k, g in planted.groupby('form'))):
        k = int(g.found.sum())
        lo, hi = d2l.wilson_ci(k, len(g))
        rows.append(dict(form=g.form.iloc[0], fits=g.fits.iloc[0], n=len(g),
                         found=k, recall=k / len(g), lo=lo, hi=hi,
                         median_words=g.words.median(),
                         ngram=PL.loc[g.index].ngram_found.mean()))
    EMBT = pd.DataFrame(rows)
    say(f'embedding detector: {len(WIN):,} windows of {SIZE} words every '
        f'{STRIDE}; threshold {THR:.4f} (99th percentile of the '
        f'{len(calib):,} controls of the plants file); held-out controls '
        f'above it: {fp} of {len(held):,} (Wilson 95% interval '
        f'{fp_lo:.4f}-{fp_hi:.4f})')
    say('recall by form; fits: some window holds the whole plant')
    say(EMBT.round(3).to_string(index=False))

    texts = [it['text'] for it in ITEMS[:8]] + [w[2] for w in WIN[:8]]
    live = d2l.embed_docs(texts, EMBEDDER, device='cpu').astype(np.float32)
    reg = np.concatenate([E['items'][:8], E['windows'][:8]]).astype(
        np.float32)
    cos = np.sum(live * reg, 1) / np.linalg.norm(live, axis=1) / \
        np.linalg.norm(reg, axis=1)
    say(f'live check, 8 items and 8 windows on the CPU: cosine to the '
        f'registered vectors min {cos.min():.4f}')
    assert cos.min() >= LIVE_COS_MIN
    R['decontam_emb'] = dict(
        table=json.loads(EMBT.to_json(orient='records')), thr=THR,
        n_calibrate=len(calib), n_holdout=len(held), fp=fp,
        fp_ci=[fp_lo, fp_hi], n_windows=len(WIN), size=SIZE, stride=STRIDE)
```

The held-out controls confirm the budget: 7 of 1,000 are above the threshold.
The detector finds 0.42 of the paraphrases, about two in five, with an
interval of $[0.37, 0.48]$, against 0.49 of the verbatim plants and 0.54 of
the reformatted ones. Paraphrasing costs it little. What limits it is
dilution, as the split by length shows. About 20 plants per form are too long
for any window to hold whole, most of them over 100 words; they fill their
windows, and 95% to 100% of them are found. The other 280 or so have a median
under 50 words, share their window with about 80 words of the host page, and
only 38% to 51% are found. Smaller windows would help the short plants and add
false positives, since every window is another chance to resemble some test
item by accident. We keep this detector out of the pipeline: at its budget it
would remove about one innocent document in a hundred to find fewer than half
of the plants, which makes it a tool for reviewing a corpus rather than for
filtering one.

The planted set is an artifact of its own. Its 300 items were drawn with a
fixed seed from items whose question has at least 13 words and whose page form
has at most 180, and each form was inserted as a line of its own at a sentence
start of a stream document. Qwen3.5-4B wrote the paraphrases once, with
thinking off and greedy decoding, and the prompt's hash and the model's
revision are stored in every row. It takes two prompts. A prompt that mentions
answer options at all makes this model invent four options for GSM8K problems,
which have none, so pages with lettered options and pages without get
different instructions. Detection is by n-grams and embeddings, so no model
grades its own paraphrases. GSM8K and MMLU are under the MIT licence and
ARC-Challenge under CC BY-SA 4.0, whose share-alike term is the reason the
planted items live in a file of their own with its licence stated.

```{.python .input #dedup-plants}
%%tab pytorch
# one prompt for pages with lettered options (ARC, MMLU) and one for pages
# without (GSM8K): a prompt that mentions options at all made Qwen3.5-4B
# invent four options for 15-33 of the 100 GSM8K questions
PARAPHRASE_PROMPTS = dict(
    options='Rewrite the following test question so that it asks the same '
    'thing in different words. Change the wording and the sentence '
    'structure of the question and of each answer option. Keep every '
    'number as it is written, keep the options in the same order with the '
    'same letters, and keep the final answer line unchanged. Write only '
    'the rewritten question.\n\n{text}',
    plain='Rewrite the following word problem so that it asks the same '
    'thing in different words. Change the wording and the sentence '
    'structure. Keep every number as it is written, and keep the final '
    'answer line unchanged. Write only the rewritten problem.\n\n{text}')
PROMPT_SHA = hashlib.sha256(json.dumps(
    PARAPHRASE_PROMPTS, sort_keys=True).encode('utf-8')).hexdigest()
opts = lambda t: len(re.findall(r'(?m)^\s*\(?[A-J][).]\s', t))
if PLANTS_PENDING:
    model, tok = d2l.load_hf(PARAPHRASER)

    def paraphrase(texts):
        t0 = time.perf_counter()
        gens = d2l.generate(model, tok, [[{
            'role': 'user', 'content': PARAPHRASE_PROMPTS[
                'options' if opts(t) else 'plain'].format(text=t)}]
            for t in texts], max_new_tokens=256, chat=True,
            enable_thinking=False, batch_size=16)
        assert not any('<think>' in g.text for g in gens)
        cut = sum(g.finish_reason == 'length' for g in gens)
        say(f'{len(gens)} paraphrases in {time.perf_counter() - t0:.0f} s,'
            f' {cut} cut at 256 tokens, {d2l.lm_peak()}')
        assert cut == 0, 'a paraphrase was cut: raise max_new_tokens'
        return [g.text.strip() for g in gens]

    PLANTS = d2l.make_plants(ITEMS, STREAM_S, paraphrase, seed=0,
                             prompt_sha256=PROMPT_SHA,
                             model_revision=d2l.HF_REVISIONS[PARAPHRASER])
    del model
    d2l.lm_free()
    d2l.lm_save_rerun(PLANTS_FILE, lambda f: d2l.write_jsonl(f, PLANTS))
PL0 = pd.DataFrame(PLANTS)
say(PL0.groupby(['form', 'benchmark'], dropna=False).size().unstack(
    fill_value=0).to_string())
para = PL0[PL0.form == 'paraphrase']
stale = ((para.prompt_sha256 != PROMPT_SHA)
         | (para.model_revision != d2l.HF_REVISIONS[PARAPHRASER])).sum()
say(f'paraphrases written with these prompts and {PARAPHRASER}@'
    f'{d2l.HF_REVISIONS[PARAPHRASER][:8]}: {len(para) - stale} of '
    f'{len(para)}')
assert stale == 0, 'the plants were made with another prompt or model'
# what the paraphraser wrote, printed and never filtered on: 5-gram overlap
# with the verbatim page (an echo is near 1), empty outputs, a lost answer
# line, a changed set of numbers (GSM8K's answers depend on them)
verb = PL0[PL0.form == 'verbatim'].set_index('item_id').text
nums = lambda t: sorted(re.findall(r'\d+(?:\.\d+)?', t.replace(',', '')))
rows = []
for bench, g in para.groupby('benchmark'):
    v = verb.loc[g.item_id].tolist()
    J_vp = [d2l.jaccard_exact(d2l.shingles(a), d2l.shingles(b))
            for a, b in zip(v, g.text)]
    rows.append(dict(
        benchmark=bench, n=len(g), J_median=np.median(J_vp),
        J_max=max(J_vp), blank=int((g.text.str.strip() == '').sum()),
        no_answer_line=sum(not re.search(r'(?im)^\s*answer\s*:', t)
                           for t in g.text),
        numbers_changed=sum(nums(a) != nums(b) for a, b in zip(v, g.text)),
        options_added=sum(opts(b) > opts(a) for a, b in zip(v, g.text))))
PCHK = pd.DataFrame(rows)
say('\nparaphrases against their verbatim pages (5-gram Jaccard):')
say(PCHK.round(3).to_string(index=False))
assert PCHK.blank.sum() == 0, 'an empty paraphrase'
R['plants'] = dict(check=json.loads(PCHK.to_json(orient='records')))
first = PL0[PL0.benchmark == 'arc'].item_id.iloc[0]
for _, r in PL0[PL0.item_id == first].iterrows():
    say(f'\n{r.form}:\n' + textwrap.fill(textwrap.shorten(
        r.text, 300, placeholder=' …'), 96, initial_indent='  ',
        subsequent_indent='  '))
say('\nItem: ARC-Challenge test, allenai/ai2_arc (CC BY-SA 4.0)')
```

The three forms of the ARC item show what the paraphraser does: it keeps the
question and the answer and changes almost every phrase, so that "a cat has a
kitten with a broken leg" becomes "a feline offspring is born with a fractured
limb". The check table, printed and never used to filter, says the same for
all 300: the median 5-gram $J$ between a paraphrase and its verbatim page is
between 0.01 and 0.06, and no paraphrase is empty, lost its answer line or
gained options. Two departures remain. In 26 of the 100 GSM8K paraphrases the
set of digits changed, mostly because numbers were written as words ("two
dozen"), which leaves the problem the same and hides it from the 13-gram
index. One MMLU "paraphrase" repeats its item unchanged. Removing such rows
after seeing the detectors' results would tune the test to the detectors, so
they stay.

Planted paraphrases are not a hypothetical worry. Yang and colleagues (2023)
showed that test items rephrased by a language model pass n-gram
decontamination and that a 13B model trained on them reaches scores on the
original benchmarks far above its ability. The perturbed variants of
GSM-Symbolic :cite:`Mirzadeh.Alizadeh.Shahrokhi.ea.2025`, used in
:numref:`subsec_lm_eval_contamination`, attack the problem from the other end,
with problems that a model cannot have seen.

### The Version-1 Manifest
:label:`subsec_lm_dedup_manifest`

The teaching pipeline ends with three stages on c1. `exact_dedup` keeps the
first document of each hash of the NFC text, in c0 order. `minhash_dedup`
clusters the rest at $(11, 11)$, the setting chosen on the training pairs, and
keeps the lowest index of each cluster. `decontam` removes documents with a
13-gram of any test item and charges each to the benchmark of its first hit.
The result is c2. The manifest `lm-corpus-manifest-v1` records all nine stages
from the archive range of :numref:`sec_lm_crawl` on, with parameters, counts
and hashes, and every step is deterministic: no salted hash, fixed hash seeds,
lowest index per cluster, documents in their order. The cell builds c2 twice,
computing the signatures with one process and with four, and compares the
manifest hashes. Token counts are of the reference tokenizer; for the first
stages they are of the c0 text, which is the text after personal data was
removed. `lm-c2-docs-v1` holds, for every c0 document, whether it is kept, the
stage that dropped it, the reason and the document it copied, without text.

```{.python .input #dedup-c2}
%%tab pytorch
QRES = json.load(open(d2l.lm_artifact('lm-data-quality-results-v1')))
C1M = QRES['c1_manifest']
assert d2l.docs_hash(C1) == C1M['stages'][-1]['out_hash']
NTOK = dict(zip((d.id for d in C0), map(len, d2l._doc_token_ids(
    TOK, [d.text for d in C0]))))
tokens = lambda docs: sum(NTOK[d.id] for d in docs)
BENCH = dict(gsm8k=d2l.DATA_HUB['gsm8k-test'][1],
             arc=d2l.DATA_HUB['arc-challenge-test'][1],
             mmlu=d2l.EVAL_TASKS['mmlu'].revision)
B2, R2 = CHOSEN[1]

def build_c2(workers):
    """c1 -> exact_dedup -> minhash_dedup -> decontam: the manifest and one
    row per c0 document (keep, first dropping stage, reason, kept copy)."""
    m = d2l.CorpusManifest('lm-corpus-v1', C1M['sources'], C1M['stages'])
    drop = {d.id: ('heuristics', FLAGS[d.id]['codes'][0], None)
            for d in C0 if FLAGS[d.id]['codes']}
    seen, exact = {}, []
    for d in C1:
        h = nfc_sha256(d.text)
        if h in seen:
            drop[d.id] = ('exact_dedup', 'exact', seen[h])
        else:
            seen[h] = d.id
            exact.append(d)
    m.add_stage('exact_dedup', dict(key='sha256(NFC text)',
                                    keep='first in c0 order'),
                len(C1), len(exact), {'exact': len(C1) - len(exact)},
                d2l.docs_hash(C1), d2l.docs_hash(exact),
                impl='hashlib.sha256', tokens_out=tokens(exact))
    sigs = d2l._minhash_texts([d.text for d in exact], K, (0xD2,), 5,
                              workers)[0xD2]
    pairs, rep = d2l.lsh_candidates(sigs, B2, R2)
    lab = d2l.union_find_clusters(len(exact), pairs)
    near = [d for i, d in enumerate(exact) if lab[i] == i]
    for i, d in enumerate(exact):
        if lab[i] != i:
            drop[d.id] = ('minhash_dedup', 'near_dup', exact[lab[i]].id)
    m.add_stage('minhash_dedup', dict(
        ngram=5, k=K, b=B2, r=R2, seed=0xD2, threshold=TAU, max_bucket=1000,
        keep='lowest index of a cluster', cut_buckets=len(rep['cut'])),
                len(exact), len(near), {'near_dup': len(exact) - len(near)},
                d2l.docs_hash(exact), d2l.docs_hash(near),
                impl='d2l.minhash_signature, lsh_candidates, '
                'union_find_clusters', tokens_out=tokens(near))
    clean, charged = [], collections.Counter()
    for d in near:
        hit = INDEX.query(d.text)
        if hit:
            bench = ITEMS[ITEM_POS[hit[0][0]]]['benchmark']
            charged[bench] += 1
            drop[d.id] = ('decontam', bench, hit[0][0])
        else:
            clean.append(d)
    m.add_stage('decontam', dict(n=N_GRAM, key='blake2b-8 of lowercased '
                                 'word 13-grams', items=dict(n_items),
                                 sources=BENCH, charge='first hit'),
                len(near), len(clean), dict(charged), d2l.docs_hash(near),
                d2l.docs_hash(clean), impl='d2l.NgramIndex',
                tokens_out=tokens(clean))
    rows = [dict(id=d.id, keep=d.id not in drop,
                 stage=drop.get(d.id, (None,) * 3)[0],
                 reason=drop.get(d.id, (None,) * 3)[1],
                 copy_of=drop.get(d.id, (None,) * 3)[2]) for d in C0]
    return m, rows

t0 = time.perf_counter()
MAN, C2_ROWS = build_c2(1)
TIMES['c2_1'] = time.perf_counter() - t0
t0 = time.perf_counter()
man4, rows4 = build_c2(4)
TIMES['c2_4'] = time.perf_counter() - t0
say(f'manifest SHA-256 with 1 worker {MAN.sha256()[:16]}… '
    f'({TIMES["c2_1"]:.0f} s), with 4 workers {man4.sha256()[:16]}… '
    f'({TIMES["c2_4"]:.0f} s)')
assert man4.sha256() == MAN.sha256() and rows4 == C2_ROWS
assert [s['name'] for s in MAN.stages] == [
    'extract', 'lang_id', 'nfc', 'pii', 'heuristics', 'edu_score',
    'exact_dedup', 'minhash_dedup', 'decontam']
if not c1_rep['cut']:  # the same clusters as the c1 report above
    assert MAN.stages[-2]['n_out'] == len(c1_keep)
# c0 holds the text after pii, so the first stages' counts are of that text
tok_after = dict(lang_id=tokens(C0), nfc=tokens(C0), pii=tokens(C0),
                 heuristics=tokens(C1), edu_score=tokens(C1))
def top_drops(drops, n=3):
    items = sorted(drops.items(), key=lambda kv: -kv[1])
    text = ', '.join(f'{k} {v}' for k, v in items[:n]) or '—'
    return text + (f' (+{len(items) - n} more)' if len(items) > n else '')

FUNNEL = pd.DataFrame([dict(
    stage=s['name'], n_in=s['n_in'], n_out=s['n_out'],
    tokens_out=s['tokens_out'] or tok_after.get(s['name']),
    tokens_of='c0 text' if s['name'] in ('lang_id', 'nfc', 'pii') else
    'output', drops=top_drops(s['drops'], 2)) for s in MAN.stages])
FUNNEL['tokens_out'] = FUNNEL.tokens_out.astype('Int64')
FUNNEL.loc[FUNNEL.tokens_out.isna(), 'tokens_of'] = '—'
say(FUNNEL.to_string(index=False))
print('\n' + MAN.to_card())
C2_KEY, C2_FILE = ARTIFACTS['dedup-c2']
MAN_KEY, MAN_FILE = ARTIFACTS['dedup-manifest']
if 'dedup-c2' in RERUN:
    d2l.lm_save_rerun(C2_FILE, lambda f: d2l.write_docs(f, C2_ROWS))
elif RERUN and C2_KEY not in d2l.DATA_HUB:
    say(f'{C2_KEY} is not registered yet: nothing to compare with')
else:
    with gzip.open(d2l.lm_artifact(C2_KEY), 'rt') as f:
        reg = [json.loads(line) for line in f]
    same = sorted(reg, key=lambda r: r['id']) == sorted(
        C2_ROWS, key=lambda r: r['id'])
    say(f'against {C2_KEY}: {"identical" if same else "DIFFERENT"}')
    assert same
if 'dedup-manifest' in RERUN:
    d2l.lm_save_rerun(MAN_FILE, MAN.save)
elif RERUN and MAN_KEY not in d2l.DATA_HUB:
    say(f'{MAN_KEY} is not registered yet: nothing to compare with')
else:
    with open(d2l.lm_artifact(MAN_KEY), 'rb') as f:
        reg_sha = hashlib.sha256(f.read()).hexdigest()
    say(f'against {MAN_KEY}: SHA-256 '
        f'{"identical" if reg_sha == MAN.sha256() else "DIFFERENT"}')
    assert reg_sha == MAN.sha256()
R['c2'] = dict(manifest_sha256=MAN.sha256(),
               funnel=json.loads(FUNNEL.to_json(orient='records')),
               kept=sum(r['keep'] for r in C2_ROWS))
```

The two manifests have the same SHA-256, and so do the registered ones. The
funnel shows how small deduplication's part is in one archive range: of 1,032
documents, 12 exact and 6 near-duplicates go, about 0.6% of the tokens, and
the decontamination stage removes nothing, since our range holds no 13-gram of
these test sets. It may still hold paraphrases, which the 13-gram index cannot
see. On a quarter of one archive file a page rarely meets its copy; across a
whole dump it meets it often, which is why deduplication removes far more at
the scale of FineWeb. Of 9,731 records, 1,014 documents with 918,627 tokens
reach the corpus, about one record in ten and a third of the English text that
extraction produced.

```{.python .input #dedup-results}
%%tab pytorch
def rounded(x):
    """Floats to 4 decimals, numpy values to Python, NaN to None."""
    if isinstance(x, dict):
        return {str(k): rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rounded(v) for v in x]
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        x = x.item()
    if isinstance(x, float):
        return None if x != x else round(x, 4)
    return x

def loose_bad(new, old, path=()):
    """The paths of results that depend on float32 arithmetic (k-means)
    or on vector products where new and old differ by more than the
    tolerance: 2 for a count, 1e-3 for the threshold, 0.01 for a rate."""
    if isinstance(new, dict) and isinstance(old, dict):
        keys = set(new) | set(old)
        return [p for k in sorted(keys, key=str) for p in (loose_bad(
            new.get(k), old.get(k), path + (k,)) if k in new and k in old
            else [path + (k,)])]
    if isinstance(new, list) and isinstance(old, list):
        if len(new) != len(old):
            return [path]
        return [p for i, (u, v) in enumerate(zip(new, old))
                for p in loose_bad(u, v, path + (i,))]
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if num(new) and num(old):
        tol = (2 if isinstance(new, int) and isinstance(old, int) else
               1e-3 if path[-1] == 'thr' else 0.01)
        return [] if abs(new - old) <= tol else [path]
    return [] if new == old else [path]

LOOSE = ('semantic', 'decontam_emb')  # compared with loose_bad
RES_KEY, RES_FILE = ARTIFACTS['dedup-results']
if PLANTS_PENDING:
    say('a dedup-plants run: the results are written by the next run')
elif 'dedup-results' in RERUN:
    blob = json.dumps(rounded(R), sort_keys=True, indent=1).encode('utf-8')
    d2l.lm_save_rerun(RES_FILE, lambda f: pathlib.Path(f).write_bytes(blob))
elif RERUN and RES_KEY not in d2l.DATA_HUB:
    say(f'{RES_KEY} is not registered yet: nothing to compare with')
else:
    reg = json.load(open(d2l.lm_artifact(RES_KEY)))
    live = rounded(R)
    strict = lambda x: {k: v for k, v in x.items() if k not in LOOSE}
    a = d2l.json_agreement(strict(live), strict(reg))
    bad = a['mismatches'] + a['missing']
    loose = loose_bad({k: live.get(k) for k in LOOSE},
                      {k: reg.get(k) for k in LOOSE})
    say(f'against {RES_KEY}: {a["equal"]}/{a["values"]} values equal, '
        f'largest numeric difference {a["max_diff"]:.2g}; '
        f'{", ".join(LOOSE)} within tolerance: {not loose}')
    for k in (bad + loose)[:20]:
        say('  differs or one-sided:', '/'.join(map(str, k)))
    assert not bad and not loose and a['max_diff'] <= 1e-9
# the kernel's own peak; each of the N_WORKERS spawned workers adds its
# own (a fresh Python that imports d2l), not visible in this number
MEM['peak_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
say('wall clock (s): ' + ', '.join(f'{k} {v:.0f}' for k, v in TIMES.items()))
say(f'memory: shingle sets {MEM["shingle_sets"] / 2**30:.2f} GiB; peak '
    f'RSS of this process {MEM["peak_rss"] / 2**30:.1f} GiB, plus up to '
    f'{N_WORKERS} workers')
```

Deduplication decides that each text is seen once. It does not decide which
texts deserve to be seen more than once, or how much of a corpus should be
code, mathematics or prose. Those are questions about proportions, and
:numref:`sec_lm_data_mixtures` answers them with training runs.

## Summary

Repeated text wastes compute and raises memorization, repeated test items
inflate scores, and most repeats on the web are near copies that exact hashes
miss. MinHash gives every document a signature of $k$ numbers whose positions
agree with probability equal to the Jaccard similarity of the two shingle
sets, an unbiased estimate with variance $J(1 - J)/k$, and banding turns
signatures into hash-table lookups that make a pair a candidate with
probability $1 - (1 - s^r)^b$. On 12,000 labelled pairs the measured candidate
rates follow this S-curve for 13 settings and three hash seeds, and the best
split of 128 positions, $(11, 11)$ with test F1 0.87, differs from the
uniform-weight optimum only because the pairs' similarities are not uniform.
Suffix arrays find repeated spans inside different documents, and embeddings
find rewordings and, at loose thresholds, pages that merely share a topic. A
13-gram index finds every verbatim and reformatted test item planted in
FineWeb pages and almost no paraphrase. An embedding detector finds about half
of the copies and two in five paraphrases, limited by the host text around
short items. On our crawl the three stages remove 1.7% of the documents and
leave a manifest whose hash does not depend on the number of workers.

## Exercises

1. **Containment.** A document has 100 five-gram shingles, all of which
   occur in a document of 1,000 shingles.
    1. Compute $J$ and the containment of the smaller in the larger. Will
       MinHash at $\tau = 0.8$ make the pair a candidate?
    1. Show that the containment can be computed from $J$ and the sizes
       $|A|$ and $|B|$. Why is an estimate of $C$ from $\hat J$ noisy when
       $J$ is small, and why would banding not find the pair anyway?
    1. Should a deduplicator remove the shorter document? Give one case in
       which it should and one in which it should not.
   :exlabel:`ex_lm_dedup-containment`

1. [code] **Short documents and salted hashes.**
    1. Show that `minhash_signature(shingles(t))` is `None` for every text
       with fewer than five words, and that `lsh_candidates` never pairs such
       a document. What should a pipeline do with such documents instead?
    1. Replace blake2b by Python's built-in `hash` in a copy of the
       signature function and compute the fixture's signatures in two
       separate processes. Name two places in this section's pipeline where
       the result would break something.
   :exlabel:`ex_lm_dedup-short-salted`

1. **Candidates from the bulk.** A corpus of $N$ documents has a share
   $\phi$ of its pairs at similarity $s$ and the rest at similarity 0.
    1. For $s < \tau$, derive the expected number of false-positive
       candidates under $b$ bands of $r$ rows.
    1. Show that $P(s) \leq b s^r$ by the union bound over bands, and that
       the bound is tight when $b s^r \ll 1$. Use it to evaluate the expected
       count for $N = 10^8$, $\phi = 10^{-4}$, $s = 0.3$ and $(b, r) =
       (14, 8)$.
    1. Keeping $b$ fixed, by what factor does lowering $r$ by one multiply
       this count? Why is that factor large where the count matters most?
   :exlabel:`ex_lm_dedup-bulk`

1. **Limits of the S-curve.**
    1. Write $P(s)$ for $r = 1$ and for $b = 1$, and explain in words what
       each extreme does.
    1. Among the splits of 128 positions with $b$ a power of two, which has
       $(1/b)^{1/r}$ closest to 0.8? Compare its exact steepest point with
       that of $(9, 13)$ and say which approximation is worse, and why.
   :exlabel:`ex_lm_dedup-limits`

1. **Shingle size.**
    1. A copy replaces a share $q$ of the words of a document at random.
       Derive the approximate $J$ of the pair under $n$-word shingles, and
       find the largest $q$ at which the pair is still a duplicate at
       $\tau = 0.8$ for $n = 3$, 5 and 8.
    1. [code] Before measuring, predict how the test F1 of the best $(b, r)$
       changes with $n$. Then rebuild the labelled pairs' labels and
       decisions for $n$ from 3 to 8 and plot F1 against $n$. Is the change
       a property of the sketch or of the labels?
   :exlabel:`ex_lm_dedup-shingle-size`

1. [code] **The signature budget.** For $k \in \{64, 128, 256\}$, predict,
   then measure, the setting chosen on the training pairs, its test F1, and
   the number of candidate pairs it produces on c1. Report each F1 difference
   with a bootstrap over base documents. Is doubling $k$ worth its cost in
   memory and hashing time at $\tau = 0.8$?
   :exlabel:`ex_lm_dedup-budget`

1. [code] **Changed numbers.** Make a fourth form of the 100 planted GSM8K
   problems that needs no model: replace every number by a different number
   of the same magnitude, and keep the wording. Before running, predict the
   recall of the 13-gram index and of the embedding detector on this form.
   Measure both with intervals, and explain the 13-gram result from the
   number of words between the numbers of a typical problem.
   :exlabel:`ex_lm_dedup-numbers`

1. [extended] **Two snapshots.** Extract documents from the same byte range
   of the archive files of two Common Crawl dumps a year apart, with the
   pipeline of :numref:`sec_lm_crawl` and :numref:`sec_lm_quality_filtering`.
   Deduplicate each dump on its own and both together, and compare the
   retained tokens, the near-duplicate rate within and across dumps, and the
   edu scores of the documents that global deduplication removes from the
   older dump. Does your result agree with FineWeb's argument for per-dump
   deduplication?
   :exlabel:`ex_lm_dedup-snapshots`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §20.3]{.kicker}

Deduplication and Decontamination<br>**exact hashes · MinHash · banding and the S-curve · substrings · embeddings · test-set leakage**
:::
:::

::: {.slide title="One Blog Post Contains Another, and Their Hashes Share Nothing"}
[The problem]{.kicker}

Two posts on one music blog in FineWeb's output: the second is the first plus
one paragraph.

| measure | value |
|---|---|
| SHA-256 equal | no |
| Jaccard of 5-word shingles, $J$ | 0.70 |
| share of the shorter found in the longer, $C$ | 1.00 |

Exact comparison of all pairs, about 8 µs per pair:

| documents | time |
|---|---|
| 20,000 | half an hour |
| $10^9$ | about $10^5$ years |

::: {.d2l-note .warn}
The method has to avoid comparing pairs, not compare them faster.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- find exact copies and shared lines with hashes;
- prove that a MinHash position agrees with probability $J$;
- derive the banding S-curve and choose $(b, r)$ on labelled pairs;
- explain what suffix arrays and embeddings catch that MinHash cannot;
- measure what n-gram and embedding decontamination miss, with planted items.

**Running example:** 26,000 FineWeb documents (the audit stream plus 6,000
same-host pages) and c1, our 1,032 crawl documents after the rules.
:::

::: {.slide title="Six Steps From a Crawl Record to a Token on Disk"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| step | section |
|---|---|
| get text | Document Extraction from Web Crawls |
| choose text | Quality Filtering |
| | **Deduplication and Decontamination** |
| mix text | Data Mixtures |
| | Synthetic and Rephrased Data |
| cut text | Tokenizers at Scale |
:::

::: {.col .narrow}
- **Here:** c1 → exact → near → decontam → c2, and the version-1 manifest.
- **Skips:** how often a text *should* be seen.
:::
:::
:::

::: {.slide title="FineWeb Has No Exact Copies, but One in Fifteen Pages Shares a Line"}
[Exact duplicates]{.kicker}

| corpus | documents | exact copies | documents with a shared line |
|---|---|---|---|
| FineWeb slice | 26,000 | 0 | 1,742 |
| c1, our crawl | 1,032 | 12 | 73 |

- Most shared: a card-payment notice (36 pages), shop review notices,
  "Your email address will not be published".
- Shared lines are a **span** problem: C4 kept one copy of every repeated
  three-sentence span.

::: {.d2l-note .warn}
Deduplicate **before** the train/validation split, or validation loss
measures memory.
:::
:::

::: {.slide title="Resemblance and Containment of Shingle Sets"}
[Similarity as set overlap]{.kicker}

Shingles: runs of 5 words (NFC, lower case, `\w+`).

$$J(A,B) = \frac{|A\cap B|}{|A\cup B|}, \qquad C(A,B) = \frac{|A\cap B|}{|A|}$$

- $A \subseteq B$: $C = 1$, $J = |A|/|B|$; a truncation to $f$ has $J \approx f$.
- Substitution rate $q$ keeps about $x = (1-q)^5$ of the shingles:
  $J = x/(2-x)$, so 2% gives 0.82 and 5% gives 0.63.
- Random pages: $J = 0$ up to the 99th percentile; same host: 0.32.

::: {.d2l-note}
Hash shingles with blake2b. Python's `hash` is salted per process.
:::
:::

::: {.slide title="Check: 100 Shingles Inside a Page of 1,000. A Duplicate?"}
[Check]{.kicker}

Every shingle of a short page occurs in a long one. $J$? $C$? Does MinHash
at $\tau = 0.8$ flag the pair?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $J = 100/1{,}000 = 0.1$, $C = 1$. A pair at $J = 0.1$ is almost
never a candidate. Containment needs its own test; under $J$, a short copy is
a different document.
:::
:::
:::

::: {.slide title="One Random Order: Agreement Has Probability J"}
[MinHash]{.kicker}

$h_\pi(A)$: the shingle of $A$ that comes first in a random order $\pi$.

Let $x^*$ be the first element of $A \cup B$, uniform over the union.

- $x^* \in A \cap B$: both minima are $x^*$, they agree.
- $x^* \in A \setminus B$: $A$'s minimum is $x^*$, $B$'s comes later.

$$\Pr[h_\pi(A) = h_\pi(B)] = \frac{|A\cap B|}{|A \cup B|} = J$$

$k$ orders: $\hat J$ = share of agreeing positions, a binomial count,
$\operatorname{Var}\hat J = J(1-J)/k$.
:::

::: {.slide title="128 Numbers per Document, Bit-Exact to a Specification"}
[The signature]{.kicker}

| step | choice |
|---|---|
| shingle hash | blake2b, 8 bytes, little-endian, mod $p = 2^{61}-1$ |
| position $i$ | $\min_x (a_i x + c_i) \bmod p$ |
| $(a_i, c_i)$ | `default_rng(0xD2)`, integers in $[1, p)$ |
| product $a_i x$ | exact: 122 bits; NumPy `uint64` overflows silently |

Checks on 200 pairs per $J$ and on the fixture:

- bias within 2 standard errors in every bin; variance 0.88 to 1.16 × formula;
- 5 of 5 fixture signatures equal to a Python-integer reference.

95% half-width at $J = 0.8$: $1.96\sqrt{0.16/128} = 0.069$.
:::

::: {.slide title="Bands Turn the Signature Into Hash-Table Lookups"}
[Banding]{.kicker}

$k = br$ positions as $b$ bands of $r$; a shared band bucket makes a candidate.

$$P(s) = 1 - (1 - s^r)^b, \qquad s^* = \Bigl(\frac{r-1}{br-1}\Bigr)^{1/r} \approx \Bigl(\frac{1}{b}\Bigr)^{1/r}$$

| $(b, r)$ | $P(0.5)$ | $P(0.8)$ | $P(0.9)$ | steepest |
|---|---|---|---|---|
| $(2, 2)$ | 0.44 | | 0.96 | 0.58 |
| $(14, 8)$, FineWeb | 0.053 | 0.92 | | 0.71 |
| $(9, 13)$ | | 0.40 | | 0.84 |

FineWeb's paper: 0.56, 0.77, 0.92, 0.988 at 0.7 to 0.85. Reproduced.
:::

::: {.slide title="Measured Candidate Rates Follow the Curve"}
[The S-curve, measured]{.kicker}

12,000 labelled pairs: 6,000 edited copies, 6,000 negatives; label $J \geq 0.8$.

13 settings × 3 hash seeds × 0.05-wide bins of $J$; expected count
$\sum_i P(J_i)$, exact Poisson-binomial 95% interval.

| | value |
|---|---|
| cells with $n \geq 30$ | 780 |
| outside their interval | 19 (2.4%) |
| bound at the 0.999 level | 59 |
| $(9,13)$, bin 0.80 to 0.85 | expected 445 [418, 473]; seeds 431, 438, 429 |

::: {.d2l-note}
A flawed hash would leave many cells outside.
:::
:::

::: {.slide title="Choosing (b, r) Like a Hyperparameter"}
[Tuning]{.kicker}

$$E(b,r) = w_{\textrm{fp}}\!\int_0^\tau\! P\,ds + w_{\textrm{fn}}\!\int_\tau^1\!(1-P)\,ds \;\Rightarrow\; (9, 13) \textrm{ at } k = 128,\ \tau = 0.8$$

| test pairs | precision | recall | F1 |
|---|---|---|---|
| $(11, 11)$, chosen on train, $w_{\textrm{fn}} = 1$ | 0.86 | 0.89 | **0.87** |
| $(9, 13)$, minimum of $E$, datasketch | 0.90 | 0.77 | 0.83 |
| $(14, 8)$, FineWeb, 112 positions | 0.72 | 0.99 | 0.83 |

$(11, 11) - (9, 13)$: F1 $+0.04$ [+0.03, +0.05]. $w_{\textrm{fn}} = 5$
chooses $(14, 9)$: fewer rows, as predicted.
:::

::: {.slide title="Check: Why Do the Pairs Choose (11, 11), Not (9, 13)?"}
[Check]{.kicker}

The hashes follow the S-curve. So why does the measured optimum differ from
the minimum of $E$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $E$ weighs every $s \in [0,1]$ equally; the copies crowd near
$\tau$ and most negatives sit at 0. With the pairs' own $J$ in place of the
uniform weight, the expected error is least at $(11, 11)$ and, for
$w_{\textrm{fn}} = 5$, at $(14, 9)$. Right about the curve, wrong about the
data.
:::
:::
:::

::: {.slide title="The Dissimilar Bulk Decides the Cost"}
[At scale]{.kicker}

$$\mathbb{E}[\#\textrm{candidates}] = \sum_{i<j} P(s_{ij})$$

| | $(11, 11)$ | $(14, 8)$ |
|---|---|---|
| $P(0.3)$ | $2 \times 10^{-5}$ | $9 \times 10^{-4}$ |
| $P(0.5)$ | 0.005 | 0.053 |
| $10^6$ documents, 0.1% of pairs at 0.5 | $3 \times 10^6$ | $3 \times 10^7$ |

- Templated pages fill single buckets: $m$ members give $m(m-1)/2$ pairs;
  `max_bucket` caps and reports them.
- Our slice, already deduplicated: 0 candidates at both; 104 at $(32, 4)$.
:::

::: {.slide title="Pairs Become Clusters; Each Cluster Keeps One Page"}
[From pairs to a corpus]{.kicker}

- Union-find over candidate pairs: A~B and B~C join A and C.
- Keep the **lowest index**: independent of pair order and of any draw.
- FineWeb deduplicates each dump on its own: globally deduplicated old
  dumps trained no better.

| c1 at $(11, 11)$ | |
|---|---|
| kept | 1,014 of 1,032 |
| clusters of two or more | 13 (31 documents) |
| largest | one page, 4 copies on 3 hosts |
| `near_dedup` default $(9, 13)$ | 1,015 kept |
:::

::: {.slide title="Suffix Arrays Find Spans That MinHash Cannot See"}
[Substring duplicates]{.kicker}

A 60-word footer on thousands of different pages: $J \approx 0.1$, never a
candidate, read thousands of times.

- Sort all suffixes (prefix doubling, $O(n \log^2 n)$); repeats become long
  common prefixes of neighbours (Kasai, $O(n)$).
- Lee et al.: one 61-word sentence over 60,000 times in C4.

2 MiB of FineWeb, about 450,000 tokens: 3 repeats of 50+ tokens.

| repeat | tokens |
|---|---|
| a blog's "read more recent posts" footer | 74 |
| the Fourth Amendment | 66 |
| a function signature from a manual | 50 |

::: {.d2l-note .warn}
Keeping one copy of every span also deletes quotations from the pages that
need them.
:::
:::

::: {.slide title="Embeddings Find Rewordings, Then Pages on the Same Topic"}
[Semantic duplicates]{.kicker}

SemDeDup: k-means (64 clusters), remove a page whose cosine to an earlier
member exceeds $1 - \varepsilon$. 20,000 stream pages.

| $\varepsilon$ | removed | what goes |
|---|---|---|
| 0.05 | 0 | |
| 0.1 | 1 | a template page, cosine 0.94, $J = 0.17$ |
| 0.15 to 0.2 | 6 to 27 | gambling, essay mills, frozen pipes ($J \approx 0$) |
| 0.3 | 181 (0.9%) | FAQs, estate planning |

Cosine to the nearest earlier member: median 0.46, largest 0.94.

::: {.d2l-note}
$\varepsilon$ belongs to one embedder; at large $\varepsilon$ this is topic
balancing, a mixture decision.
:::
:::

::: {.slide title="A 13-Gram Index Finds Copies and Nothing Else"}
[Decontamination]{.kicker}

16,533 test items, 633,848 hashed 13-grams. 100 items per benchmark planted
in three forms into 900 FineWeb pages; 1,000 controls.

| found by 13-grams | GSM8K | ARC | MMLU |
|---|---|---|---|
| verbatim | 100 | 100 | 100 |
| reformatted (case, punctuation, options reversed) | 100 | 100 | 100 |
| paraphrased by Qwen3.5-4B | 2 | 0 | 6 |

- Reformatted: found **by construction**, since the index ignores
  these very changes.
- One control flagged: it quotes the Declaration of Independence, as MMLU
  item 3828 does. Look at the span.
:::

::: {.slide title="Embeddings Find Paraphrases Nearly as Often as Copies"}
[A second detector]{.kicker}

128-word windows every 64 words; threshold at the controls' 99th percentile;
held-out controls above it: 7 of 1,000.

| recall | all plants | long plants (about 20) | short plants (about 280) |
|---|---|---|---|
| verbatim | 0.49 | 0.95 | 0.46 |
| reformatted | 0.54 | 0.95 | 0.51 |
| paraphrased | **0.42** [0.37, 0.48] | 1.00 | 0.38 |

Short plants have a median under 50 words.
:::

::: {.slide title="Check: Paraphrase Recall 0.42, Verbatim 0.49. What Limits the Detector?"}
[Check]{.kicker}

The detector finds long plants almost always and short plants about half the
time, whatever their form.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Dilution, not paraphrase. A 50-word plant shares its 128-word
window with about 80 words of the host page, and the window's vector is
about the host as much as about the item. Smaller windows help and add false
positives.
:::
:::
:::

::: {.slide title="What Every Detector Here Misses"}
[Limits]{.kicker}

| leak | 13-gram index | embeddings |
|---|---|---|
| paraphrase | misses | about 2 in 5 |
| numbers written as words ("two dozen") | breaks nearby 13-grams | |
| item under 13 words (ARC 102, MMLU 1,589) | cannot match | |
| translation | misses | |
| famous quotation | false positive | |

- Overlap is evidence of leakage, not of memorization.
- Memorization grows with model size, copies and prompt length (Carlini et
  al.); deduplication cuts memorized output about tenfold (Lee et al.).
:::

::: {.slide title="The Version-1 Manifest: Nine Stages, One Hash"}
[In the pipeline]{.kicker}

| stage | out | tokens |
|---|---|---|
| extract, of 9,731 records | 9,286 | |
| lang_id, nfc, pii (c0) | 3,288 | 2,850,469 |
| heuristics, edu_score (c1) | 1,032 | 923,725 |
| exact_dedup | 1,020 | 920,541 |
| minhash_dedup at $(11, 11)$ | 1,014 | 918,627 |
| decontam (c2) | 1,014 | 918,627 |

- Manifest SHA-256 identical with 1 and 4 workers.
- `lm-c2-docs-v1`: keep, stage, reason and kept copy per c0 id, no text.
:::

::: {.slide title="Recap"}
[Recap]{.kicker}

- Exact hashes: no copies in FineWeb's output, 12 in our crawl; shared lines
  are spans.
- MinHash: $\Pr[\textrm{agree}] = J$, variance $J(1-J)/k$; bit-exact checks.
- Banding: $1 - (1-s^r)^b$; measured rates follow it (19 of 780 cells out).
- Tune $(b, r)$ on the pairs' own similarities: $(11, 11)$, F1 0.87.
- Suffix arrays: shared spans; embeddings: rewordings, then topics.
- 13-grams find copies only; embeddings about 2 in 5, limited by dilution.

**Next:** how much of each kind of text a model should see.
:::
