# Deduplication and Decontamination
:label:`sec_lm_dedup`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "3.3 Deduplication and Decontamination · T0").
Blueprint: docs/llm/blueprints/ch03_data.md section 3.3.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Near-duplicates are found by hashing, and the hash threshold is a tunable S-curve.
**Opening:** The same licence text sits in a thousand documents of the 3.2 corpus, and two of them differ only by a date. Comparing every pair is out of reach at a billion documents. The section builds a sketch whose agreement probability is the similarity itself, tunes its threshold on labelled pairs, and closes on the test set: the duplicates that matter most are the benchmark items that leaked into training.
**Teaches:**
- Exact hashing of documents and of substrings (a suffix array over a sample, `d2l.suffix_array`, `d2l.repeated_spans`); Jaccard similarity on word 5-gram shingles and why containment matters when a short document sits inside a long one.
- MinHash signatures with k hash rows (`d2l.minhash_signature`, bit-exact to the A3 specification) and locality-sensitive banding (`d2l.lsh_candidates`): b bands of r rows, candidate pairs from bucket collisions, union-find clusters with a seeded representative; `d2l.near_dedup` as the wrapper 7.3 reuses.
- Decontamination by 13-gram overlap (`d2l.NgramIndex`) and by embedding similarity; what each detector misses (paraphrase, reformatting, short items); what deduplication does to memorization, in one paragraph.
- Semantic deduplication with embeddings as a relative with one difference (clusters in embedding space instead of shingle sets).
**Derivation:** The probability that one MinHash row agrees equals the Jaccard similarity (the minimum over a random permutation of the union falls in the intersection with that probability); with b bands of r rows the candidate probability is 1 - (1 - s^r)^b, an S-curve whose threshold is near (1/b)^(1/r); choosing (b, r) under a signature budget k = b r as a weighted false-positive and false-negative trade (`d2l.best_br`). Smallest numeric case: b = 2, r = 2 gives 1 - (1 - s^2)^2, which at s = 0.5 is 0.4375 and at s = 0.9 is 0.964.
**Notebook:**
1. Implement shingling, the signature and banding from scratch; check the signature bit for bit against the committed fixture and the estimator's unbiasedness on 500 fixture pairs against the binomial standard error.
2. On a committed labelled pair set (6,000 perturbed pairs by substitution, deletion, boilerplate insertion, paragraph shuffle and truncation, plus 6,000 negatives from the same host and at random, regenerated from the rebuilt documents), evaluate the candidate rate per Jaccard bin for a grid of (b, r) at k = 128, with three hash seeds; fit nothing, compare with the curve.
3. Choose (b, r) on the train half under two false-negative weights and report precision, recall and F1 on the test half with a paired bootstrap clustered by base document; compare with FineWeb's (14, 8) setting and with an oracle library's choice.
4. Plant 100 items from each of three benchmarks into 900 documents of the 20,000-document slice in verbatim, reformatted and paraphrased form (paraphrases written once by Qwen3.5-4B at authoring); detect with the 13-gram index and with embedding nearest neighbours; report recall by form and the false-positive rate on 1,000 clean documents.
5. Claim that survives noise: the empirical candidate rate per bin lies within binomial error of the S-curve for every (b, r) (a departure means a flawed hash); all verbatim and reformatted plants are found by the 13-gram index with no false positives, and paraphrased plants are found only by the embedding detector at a stated recall and false-positive rate; the chosen (b, r) is the derived optimum or an adjacent grid point.
6. Reused pieces: `d2l.Doc`, `d2l.CorpusManifest`, `d2l.docs_hash` (3.1), `lm-c1-docs` and `d2l.embed_docs` (3.2), `d2l.paired_bootstrap` (1.5); new `d2l.shingles`, `d2l.minhash_signature`, `d2l.lsh_candidates`, `d2l.union_find_clusters`, `d2l.near_dedup`, `d2l.jaccard_exact`, `d2l.scurve`, `d2l.best_br`, `d2l.suffix_array`, `d2l.lcp_array`, `d2l.repeated_spans`, `d2l.NgramIndex`, `d2l.make_dup_pairs`.
**Figures:** `fig_lm_lsh_scurve` (plot: candidate probability against exact Jaccard for several (b, r), theory as lines and the measured bins with binomial intervals, the threshold marked); `fig_lm_minhash` (a document as a shingle set, one permutation, the minimum landing in the intersection or not, the signature and the bands).
**Restate / forward:** `sec_prob` (a MinHash value is a minimum over a random permutation, and the agreement event is the minimum falling in the intersection). Forward: 8.7 (memorization extraction on a small model).
**Exercises:** 5 to 7. Load-bearing: derive the expected number of false-positive candidate pairs among N documents under banding when a stated fraction of pairs has Jaccard s (derive); measure F1 against shingle size 3 to 8 on the labelled pairs and explain the shape (measure); check that the signature of a document with fewer than five words is empty and never matches (check); open: deduplicate across two crawl snapshots and compare per-snapshot with global deduplication on retained tokens and on near-duplicate rate.
**Writer notes:**
- The MinHash specification is normative and shared with A3: NFC, lowercase, `\w+` tokens, word 5-gram shingles joined by one space, shingle hash from blake2b with an 8-byte digest, prime 2^61 - 1, rows `(a_i x + c_i) mod p` with `a_i, c_i` drawn from `numpy.random.default_rng(seed=0xD2)`, band keys from blake2b over the band's bytes; use Python integers or split 64-bit multiplication, because NumPy `uint64` overflows the modular product; the bit-exact test enforces equality with the Python-integer version.
- Exports with contracts (`01_interfaces.md`): `d2l.near_dedup(texts, threshold=0.8, ngram=5, num_perm=128, seed=0) -> list[int]` (indices to keep; the representative of a cluster is its lowest index; on 300 documents with planted near-duplicates it returns exactly one index per planted cluster, and `list(range(n))` at threshold 1.0 on distinct texts); `d2l.lsh_candidates(sigs, b, r)` subsamples buckets over `max_bucket` members and reports it (templated pages make giant buckets).
- Labelled pairs: no natural near-duplicate labels exist in the stream because FineWeb has already deduplicated per dump; the pair set is synthetic by construction (`d2l.make_dup_pairs` from `(a, op, params)`), regenerated from the rebuilt documents so the artifact holds no crawled text; natural pairs in the corpus are inspected, not scored.
- Plants: GSM8K and MMLU items are MIT, ARC-Challenge items are CC BY-SA 4.0, so the planted file is a separate artifact with its licence stated; the paraphrases are generated once by Qwen3.5-4B (non-thinking) with the prompt and revision stored per row; detection is by n-grams and embeddings, so no model grades its own paraphrases. If the generator revision changes, regenerate and update A3's constants.
- Runtime at T0: shingling 20,000 documents in pure Python may exceed 10 minutes on 8 cores; vectorize the signature over all shingles of a document at once or use 10,000 documents; MiniLM on CPU over 4,000 documents may need 2,000 and `max_len=128`; a 2 MB sample for the suffix array is fine in NumPy.
- Pitfalls to state: Python's built-in `hash` is salted per process and breaks the manifest hash; deduplicating after the train-validation split leaks; n-gram decontamination misses paraphrase, translation and changed numbers, and 13-grams miss short items; removing every document that overlaps a benchmark deletes famous quotations (look at the span); per-dump against global deduplication changes the result (the FineWeb comparison).
- The memorization paragraph follows the blueprint: one paragraph with the "Extracting Training Data" pointer, and the extraction experiment itself is 8.7's.
- Committed artifacts: `lm-dedup-pairs`, `lm-contam-plant`, `lm-c2-docs` (keep and drop lists with reason codes, no text), `lm-corpus-manifest-v1` (stages `extract, lang_id, nfc, pii, heuristics, edu_score, exact_dedup, minhash_dedup, decontam`), the S-curve data as JSON.
- Do not re-teach: binomial counting and independence (`sec_prob`), the IoU of boxes (one sentence of contrast with Jaccard on sets).
**Sources:**
- Papers: Broder, "On the resemblance and containment of documents" (1997; cited by title: MinHash and the resemblance-containment distinction); [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) (near-duplicate and exact-substring deduplication, effects on memorization and perplexity); [SemDeDup](https://arxiv.org/abs/2303.09540) (semantic deduplication with embeddings); [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) (memorization as the reason duplicates matter; the pointer to 8.7); [GSM-Symbolic](https://arxiv.org/abs/2410.05229) (perturbed benchmark items as a contamination control); [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (the contamination appendix and its 13-gram rule).
- Code and documentation: [datasketch](https://github.com/ekzhu/datasketch) (the MinHash and LSH oracle and its (b, r) optimizer); [deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) (the suffix-array deduplicator); [Datatrove](https://github.com/huggingface/datatrove) (the MinHash stages and examples); [dolma](https://github.com/allenai/dolma) (the toolkit behind Dolma's deduplication and decontamination).
- Courses, tutorials and blogs: [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) (the deduplication ablations, per-dump against global); [CS336](https://cs336.stanford.edu/) (the deduplication lecture and assignment A4's MinHash part).
-->

<!-- Placeholder (stage 2, cells step): the cells follow the section
specification's cell table (S/spec/20.3.md part 3); one-line comments mark
where prose goes. Rights rule P1: no cell prints text of our own crawl
extraction (c0, c1, c2); displayed text is FineWeb's (ODC-By) from the audit
stream and its host pairs, or benchmark items under their licences;
everything else about the crawl is counts, hashes and ids. -->

<!-- Prose: the opening (the same licence text in many documents, two pages that differ by a date); comparing every pair does not scale; the section builds a sketch whose agreement probability is the similarity itself, tunes its threshold on labelled pairs and ends with the test set. -->

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

<!-- Prose: what the notebook computes (all on the CPU), what it reads from registered files (the labelled pairs, the planted test items and the embeddings, made once on a GPU), and the D2L_RERUN names of the regenerations. -->

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
print(f'K={K}, TAU={TAU}, seeds {[hex(0xD2 + s) for s in SEEDS]}, '
      f'workers {N_WORKERS}, GPU={GPU}, regenerating: '
      f'{", ".join(sorted(RERUN)) or "none"}')
```

<!-- Prose: the registry; the pairs file holds ids and edit parameters, no text (the texts are regenerated from FineWeb's stream); the plants file holds benchmark items under their licences (ARC-Challenge is CC BY-SA 4.0); the embeddings are of FineWeb text and benchmark items only. -->

<!-- Pending registrations (cells step, 2026-11-16): the six keys below join this cell as a `#@save` block once uploaded (W9). The section also reads `lm-data-audit-stream-v1` (W2 builder; with the host_pair rows), `lm-data-emb-qwen3emb-v1`, `lm-c1-docs-v1` and `lm-data-quality-results-v1` (20.2), `lm-c0-docs-v1` and `lm-data-crawl-results-v1` (20.1), and the reference tokenizer `lm-r2-tokenizer-v1` (20.6, W5); GSM8K and ARC-Challenge through chapter 18's keys, MMLU through chapter 18's loader. -->

```{.python .input #dedup-data-hub}
%%tab pytorch
DEDUP_KEYS = tuple(key for key, _ in ARTIFACTS.values())
UPSTREAM = ('lm-data-audit-stream-v1', 'lm-data-emb-qwen3emb-v1',
            'lm-c1-docs-v1', 'lm-data-quality-results-v1', 'lm-c0-docs-v1',
            'lm-data-crawl-results-v1', 'lm-r2-tokenizer-v1', 'gsm8k-test',
            'arc-challenge-test')
for name, keys in (('this section', DEDUP_KEYS), ('upstream', UPSTREAM)):
    print(f'{name}: registered', ', '.join(
        k for k in keys if k in d2l.DATA_HUB) or 'none', '| pending:',
          ', '.join(k for k in keys if k not in d2l.DATA_HUB) or 'none')
print('MMLU test:', d2l.EVAL_TASKS['mmlu'].dataset,
      d2l.EVAL_TASKS['mmlu'].revision[:8], '(chapter 18 loader)')
```

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
print(f'slice: {len(STREAM_S):,} stream documents and '
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
print(f'\nmost similar same-host pair below J = 1: {host_of[a]}, '
      f'J = {J_host[top]:.3f}, containment of the smaller '
      f'{d2l.containment(small, large):.3f}, '
      f'{len(TEXT[a].split()):,} and {len(TEXT[b].split()):,} words, '
      f'SHA-256 equal: {same_hash}')
diff = [line for line in difflib.unified_diff(
    TEXT[a].split('\n'), TEXT[b].split('\n'), lineterm='', n=0)
        if line[:1] in '+-' and line[:3] not in ('---', '+++')]
for line in diff[:6]:  # the lines in which the two pages differ
    print('  ' + (line if len(line) <= 100 else line[:99] + '…'))
print(ATTRIBUTION)

sets = [set(d2l.shingles(t)) for t in STREAM_S.text[:1000]]
t0 = time.perf_counter()
for i in range(len(sets)):
    for j in range(i + 1, len(sets)):
        d2l.jaccard_exact(sets[i], sets[j])
TIMES['all_pairs_1000'] = time.perf_counter() - t0
per_pair = TIMES['all_pairs_1000'] / (1000 * 999 / 2)
print(f'\nexact Jaccard of all pairs of 1,000 documents: measured '
      f'{TIMES["all_pairs_1000"]:.1f} s ({1e6 * per_pair:.1f} µs per pair);'
      ' extrapolated (quadratic):')
for n in (20_000, 10**9):
    sec = n * (n - 1) / 2 * per_pair
    print(f'  {n:>13,} documents: {sec:.3g} s = {sec / 3600:.3g} hours = '
          f'{sec / 3.156e7:.3g} years')
R['opening'] = dict(pair=[a, b], J=J_host[top], n_hosts=len(HOSTS),
                    n_host_pairs=len(HOST_PAIRS), same_sha256=same_hash)
del host_sets, sets
```

## Exact Duplicates
:label:`subsec_lm_dedup_exact`

<!-- Prose: a document hash of the NFC text finds copies; line and paragraph hashes find shared parts; boilerplate (cookie notices, navigation) is a span problem, not a document problem; C4's rule of keeping one copy of each repeated three-sentence span (cited); pitfall: deduplicate before the train-validation split, never after. -->

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
    print(f'{name}: {len(texts):,} documents, {len(h):,} distinct NFC '
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
print()
for p in common:
    ids = by_para[p]
    print(f'{len(ids):>4} documents, e.g. {ids[0][10:18]}… and '
          f'{ids[1][10:18]}…: {p[:100]!r}')
print(ATTRIBUTION)
R['exact']['common'] = [len(by_para[p]) for p in common]
```

## MinHash and Banding
:label:`subsec_lm_dedup_minhash_banding`

<!-- Prose: one paragraph: comparing every pair is quadratic (the opening's measurement); the rest of this section replaces the comparison by a sketch per document and a lookup. -->

### Similarity as Set Overlap
:label:`subsec_lm_dedup_jaccard`

<!-- Prose: word 5-gram shingles (FineWeb's choice); resemblance J and containment C (Broder); J misses a truncated copy (J = |A|/|B| when A is inside B) where C does not; one sentence contrasting the IoU of boxes in :numref:`sec_anchor`; Python's built-in hash is salted per process, so a sketch built on it changes between runs. -->

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

```{.python .input #dedup-jaccard-hist}
%%tab pytorch
t0 = time.perf_counter()
SH = {i: set(d2l.shingles(t)) for i, t in TEXT.items()}
TIMES['shingle_slice'] = time.perf_counter() - t0
sizes = np.array([len(s) for s in SH.values()])
mem = sum(sys.getsizeof(s) + sum(map(sys.getsizeof, s)) for s in SH.values())
print(f'{len(SH):,} documents: shingles per document median '
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
    print(f'{name:>9} pairs: median J {np.median(j):.3f}, 99th percentile '
          f'{np.percentile(j, 99):.3f}, J >= {TAU}: {np.sum(j >= TAU)}')
MEM['shingle_sets'] = mem
R['jaccard'] = dict(median_shingles=np.median(sizes),
                    random=np.percentile(J_rand, [50, 90, 99]).tolist(),
                    same_host=np.percentile(J_same, [50, 90, 99]).tolist(),
                    random_dup=int(np.sum(J_rand >= TAU)),
                    same_host_dup=int(np.sum(J_same >= TAU)))
del SH  # 1.8 GiB: the signatures below replace the sets
```

### MinHash
:label:`subsec_lm_dedup_minhash`

<!-- Prose: the derivation eq_lm_dedup_minhash (P[agree] = J: under a random permutation the minimum of the union is equally likely to be any of its elements, and the two minima agree exactly when it lies in the intersection; restating :numref:`sec_prob` in one sentence), with fig_lm_minhash; the estimator's variance J(1 - J)/k as a binomial count (by reference to :numref:`sec_prob`); hash functions (a x + c) mod p in place of permutations; the normative signature specification (NFC, lowercase, \w+ tokens, word 5-grams, blake2b-8, p = 2^61 - 1, a and c from default_rng(0xD2), the split multiplication because NumPy's uint64 product overflows). -->

<!-- Figure: ![…](../img/mdl-lm-data-minhash.svg) :label:`fig_lm_minhash` (generated by tools/gen_mdl_lm_data_figures.py; W4). -->

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
MINHASH_FIXTURE = (  # five strings whose signatures assignment A3 pins
    'The quick brown fox jumps over the lazy dog.',
    'Deduplication keeps one copy of every document in the corpus.',
    'Café crème: une boisson chaude, servie le matin avec du pain.',
    'ONE two Three four FIVE six seven, eight; nine: ten!',
    'Near-duplicate pages differ only by a date or a sidebar.')
MINHASH_FIXTURE_SHA256 = (  # of their k = 128, seed 0xD2 signatures' bytes
    'b9b07bf416ce76f2b5cf1ae3d12fc617bdfa5bc7e464230c6bca1ada3bda0195')
```

<!-- Prose: the estimator checked on sets of known Jaccard (bias within three standard errors, variance against J(1 - J)/k), against datasketch's MinHash as an oracle, and bit for bit against a Python-integer version of the specification on the fixture that assignment A3 grades with (and the fixture's pinned hash); the 95% half-width at J = 0.8 and k = 128 is 1.96 sqrt(0.8 x 0.2 / 128) = 0.069. -->

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
print(D1.round(4).to_string(index=False))
assert (D1.z.abs() <= 3).all() and (D1.z_datasketch.abs() <= 3).all()
assert D1.var_ratio.between(0.7, 1.3).all()
print(f'95% half-width at J = 0.8, k = {K}: '
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
print(f'fixture: signatures equal to the Python-integer specification: '
      f'{sum(same)} of {len(same)}')
assert all(same), 'not bit-exact to the specification'
fix = hashlib.sha256(b''.join(s.astype('<u8').tobytes() for s in sigs))
print(f'fixture signatures: SHA-256 {fix.hexdigest()[:16]}…, pinned '
      f'{d2l.MINHASH_FIXTURE_SHA256[:16]}…')
assert fix.hexdigest() == d2l.MINHASH_FIXTURE_SHA256, 'fixture changed'
R['minhash'] = dict(table=D1.to_dict('list'), fixture=fix.hexdigest(),
                    reference_equal=sum(same))
```

### Banding and the S-Curve
:label:`subsec_lm_dedup_lsh`

<!-- Prose: the derivation eq_lm_dedup_scurve (b bands of r rows: a band agrees with probability s^r, no band with (1 - s^r)^b, so P = 1 - (1 - s^r)^b; its steepest point ((r - 1)/(br - 1))^(1/r) is near (1/b)^(1/r)); the smallest case b = r = 2 (0.4375 at s = 0.5, 0.9639 at 0.9); choosing (b, r) under k = br by weighted false-positive and false-negative areas (eq_lm_dedup_br); FineWeb's (14, 8) reproduces the paper's 56, 77, 92 and 98.8%; candidate pairs from bucket collisions; union-find in the next section. -->

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
print('(14, 8) at s = 0.7, 0.75, 0.8, 0.85:', ours, '| paper:', paper)
assert tuple(ours) == paper
print(f'b = r = 2: P(0.5) = {d2l.scurve(0.5, 2, 2):.4f}, '
      f'P(0.9) = {d2l.scurve(0.9, 2, 2):.4f}')
s = np.linspace(0, 1, 200_001)
for b, r in ((14, 8), (9, 13), (20, 5)):
    slope = np.gradient(d2l.scurve(s, b, r), s)
    steep = ((r - 1) / (b * r - 1)) ** (1 / r)
    print(f'({b}, {r}): steepest at {s[np.argmax(slope)]:.4f}; '
          f'((r - 1)/(br - 1))^(1/r) = {steep:.4f}; '
          f'(1/b)^(1/r) = {(1 / b) ** (1 / r):.4f}')
    assert abs(s[np.argmax(slope)] - steep) < 1e-4

BR = d2l.best_br(K, TAU)
brute = {(b, r): d2l._br_error(b, r, TAU) for r in range(1, K + 1)
         for b in range(1, K // r + 1)}  # every b r <= K
best = min(brute, key=brute.get)
print(f'best_br({K}, {TAU}) = {BR}, error {d2l._br_error(*BR, TAU):.4f}; '
      f'brute force over b r <= {K}: {best}, error {brute[best]:.4f}')
assert BR == (9, 13) and abs(brute[best] - d2l._br_error(*BR, TAU)) < 1e-9
for r in (12, 13, 14):
    print(f'  ({K // r}, {r}): error {d2l._br_error(K // r, r, TAU):.4f}, '
          f'P({TAU}) = {d2l.scurve(TAU, K // r, r):.3f}')
lsh = MinHashLSH(threshold=TAU, num_perm=K)
DS_BR = (lsh.b, lsh.r)
BR5 = d2l.best_br(K, TAU, w_fn=5.0)
print(f'datasketch MinHashLSH(threshold={TAU}, num_perm={K}): (b, r) = '
      f'{DS_BR}; best_br with w_fn = 5: {BR5}')
R['scurve'] = dict(paper=list(ours), best_br=list(BR), brute=list(best),
                   datasketch=list(DS_BR), best_br_w5=list(BR5))
```

<!-- Prose: labelled pairs: FineWeb is already deduplicated within its dump, so the stream holds few near-duplicates; the pairs are made by perturbing stream documents (word substitutions and deletions at stated rates, inserted boilerplate, shuffled paragraphs, truncation), plus different pages of one host and random pairs; the file holds ids and parameters, and the texts are regenerated from the stream. -->

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
    print(f'made {len(rows):,} pairs in {time.perf_counter() - t0:.0f} s')
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
print(f'{len(PAIRS):,} pairs regenerated from the stream in '
      f'{TIMES["pairs_regen"]:.0f} s; largest |J - stored J| {gap:.1e}')
assert gap <= 1e-12, 'the regenerated texts differ from the registered ones'
counts = PAIRS.groupby(['op', 'split']).size().unstack()
counts['label rate'] = PAIRS.groupby('op').label.mean().round(3)
counts['median J'] = PAIRS.groupby('op').J.median().round(3)
print(counts.to_string())
R['pairs'] = dict(n=len(PAIRS), by_op=PAIRS.op.value_counts().to_dict(),
                  labels=int(PAIRS.label.sum()))
```

<!-- Prose: the measured S-curve: for every (b, r) of the grid, three hash seeds and every 0.05 bin of exact Jaccard, the share of pairs that became candidates against the expected share, the mean of scurve(J_i) over the bin's pairs (not the curve at the bin centre, which in a steep bin is off: at (9, 13) the curve rises from 0.40 to 0.69 across 0.80-0.85), with a Poisson-binomial 95% interval of the count; the claim is the pooled one: across all (setting, bin, seed) cells the share outside is about 5%, below the 0.999 binomial bound (a flawed hash would exceed it); not "95% of bins for every (b, r)" (one setting sits at 0.93), and the interval treats a bin's pairs as independent although the three copies of one base document share shingles. -->

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
print(f'{len(uniq):,} distinct pair documents, {len(SEEDS)} seeds, '
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
print(f'{n_cells} (setting, bin, seed) cells with n >= 30: {n_out} outside '
      f'their 95% interval (a correct hash: about 5% or fewer, at most '
      f'{bound:.0f} at the 0.999 level); share inside per (b, r): '
      + ', '.join(f'{b}x{r} {v:.2f}' for (b, r), v in per.items()))
assert n_out <= bound, 'the candidate rates depart from the S-curve'
show = SC[(SC.b == 9) & (SC.r == 13) & (SC.n >= 30)].pivot_table(
    index=['bin', 'n', 'J_mean', 'expected', 'lo', 'hi'], columns='seed',
    values='observed').reset_index()
print('\n(9, 13): observed candidates per seed against the expected count '
      'and its interval')
print(show.round(3).to_string(index=False))

s_grid = np.linspace(0, 1, 501)
theory = [(14, 8), (9, 13), (20, 5), (8, 16), (450, 20)]
d2l.set_figsize((5, 3.5))
ax = d2l.plt.gca()
d2l.plot_intervals(s_grid, {f'{b} x {r}': (d2l.scurve(s_grid, b, r),) * 3
                            for b, r in theory}, 'exact Jaccard of the pair',
                   'P(candidate)', kind='band', axes=ax,
                   marks={f'threshold {TAU}': ('--', {
                       '9 x 13': ([TAU, TAU], [0, 1])})})
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

<!-- Prose: choosing (b, r) on the training pairs by the weighted error count (false positives plus w_fn times false negatives, averaged over the three seeds) under w_fn = 1 and 5; precision, recall and F1 on the test pairs; FineWeb's (14, 8) on 112 rows and datasketch's choice as further rows; differences between the chosen r and its neighbours, and between the w_fn = 1 choice and the D3 optimum, with a bootstrap over base documents (a document and its copies move together); why the choice is not D3's (9, 13): D3's areas weigh every similarity in [0, 1] equally, while the pair set has many pairs near tau; the expected weighted error under the pairs' own J distribution has its least value exactly at the hashed choices (asserted), so the hashes behave as derived and the objective, not the sketch, moves the optimum; quote the F1 gap to (9, 13) from its interval, never "within the F1 interval". -->

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
print('the grid, means over three seeds (FP, FN: counts on the split):')
print(GRID.round(3).to_string(index=False))
steps = abs(R_GRID.index(CHOSEN[1][1]) - R_GRID.index(BR[1]))
print(f'\nchosen on train: w_fn = 1 -> {CHOSEN[1]}, w_fn = 5 -> '
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
    print(f'test, w_fn = {w}: F1 of {br[0]} x {br[1]} minus {br2[0]} x '
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
    print(f'w_fn = {w}: expected weighted error on train under the pairs\' '
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
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))  # beside the axes
```

<!-- Prose: the cost of candidates at scale (the expected number among N documents is the number of pairs times the mean of P over their similarities, so the low-similarity bulk matters through P(0.3) and P(0.5)); the slice's observed candidates scaled up, with the caveat that the slice's similarity distribution is not a corpus's; giant buckets of templated pages and max_bucket. -->

```{.python .input #dedup-cost}
%%tab pytorch
N = 10**6
for br in (CHOSEN[1], (14, 8)):
    p3, p5 = d2l.scurve(0.3, *br), d2l.scurve(0.5, *br)
    exp = N * (N - 1) / 2 * 1e-3 * p5
    print(f'{br}: P(0.3) = {p3:.2e}, P(0.5) = {p5:.4f}; N = 10^6 with 0.1% '
          f'of pairs at s = 0.5: {exp:.2e} expected candidate pairs')
t0 = time.perf_counter()
SLICE_SIGS = d2l._minhash_texts(SLICE.text, K, (0xD2,), 5,
                                N_WORKERS)[0xD2]
TIMES['slice_signatures'] = time.perf_counter() - t0
n = len(SLICE)
print(f'slice: {n:,} signatures in {TIMES["slice_signatures"]:.0f} s '
      f'({N_WORKERS} workers); candidate pairs, per 1,000 documents, and '
      'scaled by (N/n)^2 to N = 10^6 (a measured count, not a corpus '
      'estimate: FineWeb is already deduplicated within its dump):')
R['cost'] = dict(n=n)
for br in (CHOSEN[1], (14, 8), (32, 4)):
    cand, rep = d2l.lsh_candidates(SLICE_SIGS, *br)
    print(f'  {br}: {len(cand):,} pairs, {len(cand) / (n / 1000):.2f} per '
          f'1,000, {len(cand) * (N / n) ** 2:.2e} scaled; largest bucket '
          f'{rep["largest"]} members, {len(rep["cut"])} cut to max_bucket')
    R['cost'][f'{br[0]}x{br[1]}'] = dict(candidates=len(cand),
                                         largest_bucket=rep['largest'],
                                         cut=len(rep['cut']))
```

## From Pairs to a Deduplicated Corpus
:label:`subsec_lm_dedup_cluster`

<!-- Prose: candidate pairs to clusters by union-find (A~B and B~C put A and C together even when A and C differ); the lowest index represents its cluster (deterministic; FineWeb keeps a random one); per-dump against global deduplication (FineWeb's finding, cited); near_dedup as the wrapper later chapters reuse; say once that the c1 report and the manifest use the setting chosen on train, while near_dedup's default is the D3 optimum (both counts printed). -->

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
print(f'slice, near_dedup(threshold={TAU}) at {BR}: {len(keep):,} of '
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
print(f'c1 at {CHOSEN[1]} (chosen on train; the manifest below): '
      f'{len(c1_keep):,} of {len(C1):,} kept; {len(multi)} clusters of two '
      f'or more ({sum(multi.values())} documents); largest bucket '
      f'{c1_rep["largest"]}, cut buckets: {c1_rep["cut"] or "none"}')
print(f'c1 by near_dedup(threshold={TAU}) at its default {BR}: '
      f'{len(c1_keep_d3):,} kept')
largest = sorted(multi, key=lambda c: (-multi[c], C1[c].id))[:5]
c1_sets = {}
for c in largest:
    members = np.flatnonzero(c1_lab == c)
    for i in members:
        c1_sets.setdefault(i, set(d2l.shingles(C1[i].text)))
    Js = [d2l.jaccard_exact(c1_sets[c], c1_sets[i]) for i in members[1:]]
    hosts = {d2l._url_host(C1[i].url) for i in members}
    print(f'  {len(members):>3} documents on {len(hosts)} host(s), J to the '
          f'representative {min(Js):.2f}-{max(Js):.2f}')
R['dedup_run'].update(c1_kept=len(c1_keep), c1_kept_d3=len(c1_keep_d3),
                      c1_clusters=len(multi),
                      c1_in_clusters=sum(multi.values()),
                      c1_largest=[multi[c] for c in largest])
```

### Substring Duplicates
:label:`subsec_lm_dedup_substring`

<!-- Prose: what MinHash cannot see: a 60-token licence block shared by thousands of otherwise different documents; Lee et al.'s exact-substring deduplication with a suffix array: sort all suffixes (prefix doubling: rank by the first 2^k tokens, double k), adjacent suffixes share the longest common prefixes (Kasai), and every span of at least 50 tokens that occurs twice is a repeat; the production tool is a Rust pipeline whose memory grows with the corpus; ours is O(n log^2 n) NumPy on a 2 MB sample. -->

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
print(f'{len(sample):,} documents, {nbytes[len(sample) - 1] / 2**20:.2f} '
      f'MiB, {len(seq) - len(doc_ids):,} tokens: suffix array and LCP in '
      f'{TIMES["suffix_array"]:.1f} s; {len(spans):,} maximal repeats of 50+ '
      f'tokens covering {covered.mean():.2%} of the tokens')
for start, length, count in spans[:3]:
    d = doc_of[start]
    text = TOK.decode(seq[start:start + length].tolist()).replace('\n', ' ')
    print(f'  {length} tokens x {count} (from {sample.id.iloc[d][10:18]}…): '
          f'{text[:100]!r}')
print(f'documents with 50+ repeated tokens: {touched.sum()}; of these in a '
      f'MinHash cluster: {in_mh[touched].sum()}; MinHash-clustered '
      f'documents in the sample: {in_mh.sum()}')
print(ATTRIBUTION)
R['substring'] = dict(docs=len(sample), tokens=len(seq) - len(doc_ids),
                      spans=len(spans), covered=covered.mean(),
                      longest=[list(t[1:]) for t in spans[:3]],
                      touched=int(touched.sum()),
                      touched_minhash=int(in_mh[touched].sum()))
```

### Semantic Duplicates
:label:`subsec_lm_dedup_semantic`

<!-- Prose: SemDeDup as a relative with one difference: clusters in embedding space instead of shingle sets (k-means, then within a cluster a document goes when an earlier one has cosine above 1 - eps); the registered Qwen3-Embedding vectors of 20.2; what it removes that MinHash missed, and at large eps different pages on one topic; cost O(n^2 / k) per pass. -->

```{.python .input #dedup-semantic}
%%tab pytorch
emb = np.load(d2l.lm_artifact('lm-data-emb-qwen3emb-v1'))
assert (emb['stream_ids'] == STREAM_S.id.to_numpy().astype(str)).all()
sub = np.sort(np.random.default_rng(0).choice(len(STREAM_S), 4000,
                                              replace=False))
X = emb['stream'][sub].astype(np.float32)
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
rows, before = [], np.zeros(len(sub), dtype=bool)
for eps in (0.01, 0.02, 0.05, 0.1, 0.2):
    gone = best_sim > 1 - eps
    rows.append(dict(eps=eps, removed=int(gone.sum()), fraction=gone.mean(),
                     also_minhash=int((gone & mh).sum())))
    new = np.flatnonzero(gone & ~before)  # removed first at this eps
    before = gone
    for j in new[np.argsort(best_sim[new])][:2]:  # nearest the threshold
        x, z = STREAM_S.id.iloc[sub[j]], STREAM_S.id.iloc[sub[partner[j]]]
        J_xz = d2l.jaccard_exact(d2l.shingles(TEXT[x]), d2l.shingles(TEXT[z]))
        print(f'eps {eps}: cosine {best_sim[j]:.3f}, J {J_xz:.2f}\n    '
              f'{TEXT[x][:100]!r}\n    {TEXT[z][:100]!r}')
SEM = pd.DataFrame(rows)
print(SEM.round(4).to_string(index=False))
print(f'MinHash clusters among the 4,000: {mh.sum()} documents')
print(ATTRIBUTION)
R['semantic'] = SEM.round(6).to_dict('list')
```

## Decontamination
:label:`subsec_lm_dedup_decontam`

<!-- Prose: decontamination removes training text that overlaps evaluation items; GPT-3's 13-gram rule (confirm the wording in its contamination appendix) as an index of hashed word 13-grams (lowercased, punctuation and spacing dropped); what it misses by construction: paraphrases, translations, changed numbers and items shorter than 13 words; the reformatted form (case, spacing, punctuation, reordered options) is found by construction because the index normalizes exactly these and every plant's question has 13+ words, so its recall of 1 shows the normalization works, not robustness; removing every overlapping document also deletes famous quotations, so look at the span; overlap is evidence of leakage, not of memorization (Carlini et al.: extraction grows with duplication; one paragraph, with the pointer to the memorization experiment only if its label resolves, else the papers); GSM-Symbolic's perturbed items as a contamination control. -->

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
print(f'{len(ITEMS):,} test items ({dict(n_items)}), {len(INDEX.keys):,} '
      f'distinct {N_GRAM}-grams, built in {TIMES["index"]:.0f} s')
for bench in n_items:
    print(f'  {bench}: {short[bench]} items with fewer than {N_GRAM} words '
          f'(question and answer), {qshort[bench]} with a shorter question')
R['decontam'] = dict(items=dict(n_items), short=dict(short),
                     short_question=dict(qshort))
PLANTS_KEY, PLANTS_FILE = ARTIFACTS['dedup-plants']
if PLANTS_PENDING:
    print('the plants are made in this run (below); the detectors run '
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
    print('\n13-gram index: planted items found, by benchmark and form')
    print(DECON.round(3).to_string(index=False))
    print(f'controls flagged: {fp} of {len(ctrl)} (Wilson 95% interval '
          f'{d2l.wilson_ci(fp, len(ctrl))[0]:.4f}-'
          f'{d2l.wilson_ci(fp, len(ctrl))[1]:.4f})')
    for _, r in ctrl[ctrl.ngram_any].iterrows():  # look at the span
        tag = HITS[r.doc_id][0][0]
        grams = set(d2l.shingles(ITEMS[ITEM_POS[tag]]['text'], N_GRAM))
        span = next(g for g in d2l.shingles(DOCS[r.doc_id], N_GRAM)
                    if g in grams)
        print(f'  control {r.doc_id[10:18]}… shares {HITS[r.doc_id][0][1]} '
              f'{N_GRAM}-gram(s) with {tag}, e.g. {span!r}')
    exact_forms = DECON[DECON.form.isin(['verbatim', 'reformat'])]
    assert (exact_forms.found == exact_forms.n).all(), 'a missed plant: a bug'
    R['decontam'].update(ngram=DECON.round(6).to_dict('list'),
                         ngram_fp=fp, n_controls=len(ctrl))
```

<!-- Prose: a second detector: every document cut into 128-word windows (stride 64), each window's top cosine to any benchmark item under the registered Qwen3-Embedding vectors; the threshold is the 99th percentile of the 1,000 controls of the plants file (a 1% false-positive budget), and the false positives are counted on 1,000 other untouched documents (held out, with a Wilson interval); recall by form, compared with the verbatim form's embedding recall (the ceiling), never with 1; a window holds a plant whole only if the plant has at most 65 words or happens to align, and even then shares it with 60+ host words, so recall is shown by that flag; vectors made once on a GPU, checked live on 8 windows and 8 items on the CPU. -->

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
        print(f'embedded {len(ITEMS):,} items and {len(WIN):,} windows in '
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
    print(f'embedding detector: {len(WIN):,} windows of {SIZE} words every '
          f'{STRIDE}; threshold {THR:.4f} (99th percentile of the '
          f'{len(calib):,} controls of the plants file); held-out controls '
          f'above it: {fp} of {len(held):,} (Wilson 95% interval '
          f'{fp_lo:.4f}-{fp_hi:.4f})')
    print('recall by form; fits: some window holds the whole plant')
    print(EMBT.round(3).to_string(index=False))

    texts = [it['text'] for it in ITEMS[:8]] + [w[2] for w in WIN[:8]]
    live = d2l.embed_docs(texts, EMBEDDER, device='cpu').astype(np.float32)
    reg = np.concatenate([E['items'][:8], E['windows'][:8]]).astype(
        np.float32)
    cos = np.sum(live * reg, 1) / np.linalg.norm(live, axis=1) / \
        np.linalg.norm(reg, axis=1)
    print(f'live check, 8 items and 8 windows on the CPU: cosine to the '
          f'registered vectors min {cos.min():.4f}')
    assert cos.min() >= LIVE_COS_MIN
    R['decontam_emb'] = dict(
        table=json.loads(EMBT.to_json(orient='records')), thr=THR,
        n_calibrate=len(calib), n_holdout=len(held), fp=fp,
        fp_ci=[fp_lo, fp_hi], n_windows=len(WIN), size=SIZE, stride=STRIDE)
```

<!-- Prose: the planted test set: 100 items from each benchmark (seed 0; questions of at least 13 words), each in three forms: verbatim, reformatted by rule (lower case, spacing, punctuation, the options reversed) and paraphrased once by Qwen3.5-4B (non-thinking, greedy; prompt and revision stored per row), inserted at a sentence start of 900 distinct stream documents; 1,000 stream documents stay untouched as controls; detection is by n-grams and embeddings, so no model grades its own paraphrases; licences: GSM8K and MMLU MIT, ARC-Challenge CC BY-SA 4.0 (the file's card says so). -->

```{.python .input #dedup-plants}
%%tab pytorch
PARAPHRASE_PROMPT_V1 = (
    'Rewrite the following test question so that it asks the same thing in '
    'different words. Change the wording and the sentence structure of the '
    'question and of each answer option. Keep every number, keep the '
    'options in the same order with the same letters, and keep the final '
    'answer line. Write only the rewritten question.\n\n{text}')
PROMPT_SHA = hashlib.sha256(PARAPHRASE_PROMPT_V1.encode('utf-8')).hexdigest()
if PLANTS_PENDING:
    model, tok = d2l.load_hf(PARAPHRASER)

    def paraphrase(texts):
        t0 = time.perf_counter()
        gens = d2l.generate(model, tok, [[{
            'role': 'user', 'content': PARAPHRASE_PROMPT_V1.format(text=t)}]
            for t in texts], max_new_tokens=256, chat=True,
            enable_thinking=False, batch_size=16)
        assert not any('<think>' in g.text for g in gens)
        cut = sum(g.finish_reason == 'length' for g in gens)
        print(f'{len(gens)} paraphrases in {time.perf_counter() - t0:.0f} s,'
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
print(PL0.groupby(['form', 'benchmark'], dropna=False).size().unstack(
    fill_value=0).to_string())
para = PL0[PL0.form == 'paraphrase']
stale = ((para.prompt_sha256 != PROMPT_SHA)
         | (para.model_revision != d2l.HF_REVISIONS[PARAPHRASER])).sum()
print(f'paraphrases written with this prompt and {PARAPHRASER}@'
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
        numbers_changed=sum(nums(a) != nums(b) for a, b in zip(v, g.text))))
PCHK = pd.DataFrame(rows)
print('\nparaphrases against their verbatim pages (5-gram Jaccard):')
print(PCHK.round(3).to_string(index=False))
assert PCHK.blank.sum() == 0, 'an empty paraphrase'
R['plants'] = dict(check=json.loads(PCHK.to_json(orient='records')))
first = PL0[PL0.benchmark == 'arc'].item_id.iloc[0]
for _, r in PL0[PL0.item_id == first].iterrows():
    print(f'\n{r.form}:\n' + textwrap.indent(textwrap.shorten(
        r.text, 300, placeholder=' …'), '  '))
print('\nItem: ARC-Challenge test, allenai/ai2_arc (CC BY-SA 4.0)')
```

### The Version-1 Manifest
:label:`subsec_lm_dedup_manifest`

<!-- Prose: the teaching pipeline's last three stages on c1: exact duplicates by text hash, near-duplicates by MinHash at the chosen (b, r), decontamination by the 13-gram index (a document is charged to the benchmark of its first hit); the result is c2; the manifest lm-corpus-manifest-v1 records all nine stages from the crawl range on, and its hash is identical when the signatures are computed by one worker or four; the funnel of documents and tokens (the reference tokenizer; the lang_id, nfc and pii token counts are of the c0 text, which is the text after pii); the dataset card generated from the manifest; lm-c2-docs-v1 holds keep and drop decisions with reason codes, no text. -->

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
print(f'manifest SHA-256 with 1 worker {MAN.sha256()[:16]}… '
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
    'output', drops=top_drops(s['drops'])) for s in MAN.stages])
FUNNEL['tokens_out'] = FUNNEL.tokens_out.astype('Int64')
FUNNEL.loc[FUNNEL.tokens_out.isna(), 'tokens_of'] = '—'
print(FUNNEL.to_string(index=False))
print('\n' + MAN.to_card())
C2_KEY, C2_FILE = ARTIFACTS['dedup-c2']
MAN_KEY, MAN_FILE = ARTIFACTS['dedup-manifest']
if 'dedup-c2' in RERUN:
    d2l.lm_save_rerun(C2_FILE, lambda f: d2l.write_docs(f, C2_ROWS))
elif RERUN and C2_KEY not in d2l.DATA_HUB:
    print(f'{C2_KEY} is not registered yet: nothing to compare with')
else:
    with gzip.open(d2l.lm_artifact(C2_KEY), 'rt') as f:
        reg = [json.loads(line) for line in f]
    same = sorted(reg, key=lambda r: r['id']) == sorted(
        C2_ROWS, key=lambda r: r['id'])
    print(f'against {C2_KEY}: {"identical" if same else "DIFFERENT"}')
    assert same
if 'dedup-manifest' in RERUN:
    d2l.lm_save_rerun(MAN_FILE, MAN.save)
elif RERUN and MAN_KEY not in d2l.DATA_HUB:
    print(f'{MAN_KEY} is not registered yet: nothing to compare with')
else:
    with open(d2l.lm_artifact(MAN_KEY), 'rb') as f:
        reg_sha = hashlib.sha256(f.read()).hexdigest()
    print(f'against {MAN_KEY}: SHA-256 '
          f'{"identical" if reg_sha == MAN.sha256() else "DIFFERENT"}')
    assert reg_sha == MAN.sha256()
R['c2'] = dict(manifest_sha256=MAN.sha256(),
               funnel=json.loads(FUNNEL.to_json(orient='records')),
               kept=sum(r['keep'] for r in C2_ROWS))
```

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
    print('a dedup-plants run: the results are written by the next run')
elif 'dedup-results' in RERUN:
    blob = json.dumps(rounded(R), sort_keys=True, indent=1).encode('utf-8')
    d2l.lm_save_rerun(RES_FILE, lambda f: pathlib.Path(f).write_bytes(blob))
elif RERUN and RES_KEY not in d2l.DATA_HUB:
    print(f'{RES_KEY} is not registered yet: nothing to compare with')
else:
    reg = json.load(open(d2l.lm_artifact(RES_KEY)))
    live = rounded(R)
    strict = lambda x: {k: v for k, v in x.items() if k not in LOOSE}
    a = d2l.json_agreement(strict(live), strict(reg))
    bad = a['mismatches'] + a['missing']
    loose = loose_bad({k: live.get(k) for k in LOOSE},
                      {k: reg.get(k) for k in LOOSE})
    print(f'against {RES_KEY}: {a["equal"]}/{a["values"]} values equal, '
          f'largest numeric difference {a["max_diff"]:.2g}; '
          f'{", ".join(LOOSE)} within tolerance: {not loose}')
    for k in (bad + loose)[:20]:
        print('  differs or one-sided:', '/'.join(map(str, k)))
    assert not bad and not loose and a['max_diff'] <= 1e-9
# the kernel's own peak; each of the N_WORKERS spawned workers adds its
# own (a fresh Python that imports d2l), not visible in this number
MEM['peak_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
print('wall clock (s): ' + ', '.join(f'{k} {v:.0f}' for k, v in TIMES.items()))
print(f'memory: shingle sets {MEM["shingle_sets"] / 2**30:.2f} GiB; peak '
      f'RSS of this process {MEM["peak_rss"] / 2**30:.1f} GiB, plus up to '
      f'{N_WORKERS} workers')
```

## Summary

<!-- Prose: Summary (3-6 sentences). -->

## Exercises

<!-- Prose: Exercises (7; spec part 8). -->
