# Chapter 3: Data for Language Models — Blueprint

**Label:** `chap_lm_data` · **Arc:** II. Building a base model · **Sections:** 3.1 `sec_lm_crawl`, 3.2 `sec_lm_quality_filtering`, 3.3 `sec_lm_dedup`, 3.4 `sec_lm_data_mixtures`, 3.5 `sec_lm_synthetic_data`, 3.6 `sec_lm_tokenizers` · **Tiers:** T0 (3.1, 3.3, 3.6 tokenizer work), T1 (3.2, 3.5, 3.6 headline LM), T2 with reduced T1 (3.4) · **Assignment:** A3 · **v5 lines:** 677–784

## Chapter overview

### Draft introduction (150 words)

A base model is a compressed copy of its training text, so everything it can do was decided before the first gradient step. This chapter follows one document from a Common Crawl record to a token in a shard on disk. We extract text from HTML and watch each extractor fail on code, tables and mathematics. We filter with heuristics and with a classifier trained on LLM-written labels, and we audit who the filter removes. We derive MinHash and LSH, tune the S-curve against labeled pairs, and catch benchmark items planted in a training slice. We treat domain weights as hyperparameters fitted with small proxy runs, and we test whether LLM-rephrased text can stand in for fresh text at an equal token budget. Last we choose a tokenizer, measure what it costs each language, and freeze the one that the book's reference model R2 uses. Every step writes to one versioned manifest, so the corpus Chapter 4 trains on can be rebuilt from hashes.

### Learning objectives

After the chapter a reader can:

1. Turn a WARC prefix into a documented corpus: each document carries a record id, URL, language score and a log of every transformation. Measure, per content type (prose, code, table, math), what each of two extractors loses against the FineWeb extraction of the same records.
2. Train a quality classifier on LLM-annotated labels, convert its scores to a keep/drop rule, and report precision, recall and retained-token fraction at the threshold. Audit the removals by language, domain and English variety, with bootstrap intervals from 1.5.
3. Derive P(MinHash agree) = J and the S-curve 1 − (1 − s^r)^b. Pick (b, r) for a threshold and a signature budget, and verify the curve against labeled pairs with a numerical check.
4. Detect verbatim and paraphrased benchmark items in a training slice, and state which detector fails on which kind of contamination.
5. Fit a regression from mixture weights to per-domain loss using proxy runs, and judge it only against a noise floor measured from repeated seeds and against held-out mixtures.
6. Compare original, rephrased and mixed training data at an equal token budget, and state the condition under which recursive training loses the tail.
7. Compute fertility, tokenization premium and bits per byte for BPE and unigram tokenizers on a multilingual slice. Choose a vocabulary size from a cost model. Extend a pretrained vocabulary and compare embedding initializations.
8. Emit tokenized shards plus a manifest that `d2l.TokenShards` reads, and verify them by hash.

### Prerequisites

| Label or section | What is used |
|---|---|
| `sec_text-sequence` | `d2l.BPETokenizer` (train, encode, decode, `GPT2_PATTERN`, `from_tiktoken`), special tokens above the BPE range (`subsec_special-tokens`), fertility (`subsec_fertility-digits-glitch`). Not re-taught. |
| `sec_language-model` | Perplexity and bits per byte (`subsec_bits-per-byte`, `eq_bpb`), natural-language statistics (`subsec_natural-lang-stat`). 3.6 only converts a per-token loss into bpb. |
| `sec_scaling-laws` | 6ND, the miniature scaling study, the repeated-data remark. 3.4 sizes proxy runs with 6ND. |
| `sec_gpt`, `sec_transformer-block` | `d2l.GPT(vocab_size, num_hiddens, num_heads, num_blks, max_len, pos, norm, act, ...)` as the proxy model; forward returns logits with tied embeddings. |
| `sec_scheduler`, `sec_adamw` | Warmup + cosine or WSD schedule and AdamW for proxy runs. |
| `sec_prob`, `sec_mdl-statistics` | Independence and binomial counting for the MinHash derivation; bootstrap confidence intervals. |
| `sec_mdl-information_theory` | Code length; used to read bpb as compression. |
| `sec_environment-and-distribution-shift` | Why a filter keyed on a proxy label shifts the training distribution. 3.2 and 3.4 cite it once. |
| `sec_read_write`, `sec_repro` | `d2l.DATA_HUB`/`d2l.download`, seeds. |
| 1.1 `sec_lm_hf_models`, 1.2 `sec_lm_d2l_checkpoint` | `d2l.load_hf` (SmolLM2 and Qwen loading); `d2l.GPT`. |
| 1.5 `sec_lm_eval_suites` | `d2l.paired_bootstrap(scores_a, scores_b, n_boot=10000, cluster=None, *, seed=0, alpha=0.05) -> BootstrapResult` (read `r.mean`, `r.ci`; not a 3-tuple). |
| 2.1 `sec_lm_word_embeddings`, 2.3 `sec_lm_text_embeddings` | Embedding spaces, frozen embedding plus linear head, Qwen3-Embedding-0.6B usage. 3.6 reuses 2.1's view of new-token initialization as embedding-space placement. |

### Section dependency graph

```
3.1 crawl -> docs c0 --+--> 3.2 filters + classifier -> c1 --> 3.3 dedup + decontam -> c2 --> manifest v1
                       |                                        (3.3 also reads 3.2 embeddings)
FineWeb-Edu (side/val/R2 buckets) ---> 3.4 mixtures (T2/T1) --> 3.5 synthetic (T1)
                                                  \                 \
                                                   +-----------------+--> 3.6 tokenizers --> r2 tokenizer + shards --> Chapter 4
3.4 and 3.5 load the frozen R2 tokenizer (`lm-r2-tokenizer`) that 3.6 teaches and builds.
```

3.1 -> 3.2 -> 3.3 is one pipeline over the crawl slice. 3.4 and 3.5 are experiments on FineWeb-Edu, code and math. 3.6 is read after 3.4/3.5 but builds the tokenizer they load; the notebook for 3.6 verifies that its freshly trained tokenizer reproduces the committed `tokenizer.json` hash.

### The corpus series and R2 data

The chapter keeps two data lines apart on purpose.

- **Teaching pipeline (`c0` -> `c1` -> `c2`).** One WARC prefix from Common Crawl, processed by 3.1–3.3. It is small, fully inspectable, and its documents have FineWeb counterparts for validation. It is not used to train R2.
- **R2 data.** FineWeb-Edu `sample-10BT`, tokenized with the R2 tokenizer and written as shards by 3.6. These are what 4.1 streams and 4.2 trains on. They are FineWeb-Edu "as released" (already filtered and per-dump deduplicated by HuggingFaceFW), plus NFC normalization, hash-bucket splits, and a train/val leakage check run with the 3.3 tools.

**Document buckets (fixed by this chapter, used by 3.4, 3.5, 3.6, 4.1, 4.2).** For a FineWeb-Edu document with id `<urn:uuid:…>`:

```
bucket = int(sha256(id.encode()).hexdigest()[:8], 16) % 256
bucket 0        -> R2 validation
buckets 1..7    -> side pool for 3.4 / 3.5 (web domain train)
bucket 8        -> side validation for 3.4 / 3.5
buckets 9..255  -> R2 train
```

### Shared artifacts: produced and consumed

| Artifact (`DATA_HUB` key) | Produced by | Consumed by |
|---|---|---|
| `lm-cc-warc-256m` (a pin: the WARC URL, the byte range 0..268,435,455 and its SHA-256; the notebook fetches the range from Common Crawl, nothing is mirrored) | 3.1 | 3.1, A3 |
| `lm-c0-docs` (a manifest of the extracted docs: ids, WARC offsets, text hashes, logs, no text; the notebook rebuilds the documents from the pinned range) | 3.1 | 3.2, 3.3, A3 |
| `lm-edu-annot-22k`, `lm-quality-head` | 3.2 | 3.2, A3 |
| `lm-c1-docs`, `lm-c2-docs` (per-document flags and keep/drop lists keyed by c0 id, no text), `lm-corpus-manifest-v1` | 3.2, 3.3 | 3.3, A3 |
| `lm-dedup-pairs`, `lm-contam-plant` | 3.3 | 3.3, A3 |
| `lm-mix-{web,code,math}-{20m,150m}`, `lm-mix-val-{1m,2m}`, `lm-mix-runs-t2` | 3.4 | 3.4, 3.5 |
| `lm-rephrase-fwedu-5m` | 3.5 | 3.5 |
| `lm-r2-tokenizer`, `lm-fwedu-val`, `lm-fwedu-shards-100m`, `lm-fwedu-manifest-4b` | 3.6 | 3.4, 3.5, 4.1, 4.2 |

Exact formats are in 3.6 (shards, tokenizer) and 3.1–3.3 (document JSONL). All `DATA_HUB` entries are `(url, sha1)` pairs. Small artifacts live under `d2l.DATA_URL + 'lm/'`; the R2 tokenizer and the FineWeb-Edu shards of 3.6 live in a d2l-owned Hugging Face repository; each `DATA_HUB` entry is `('https://huggingface.co/<repo>/resolve/<commit>/<file>', sha1)` with a pinned commit, never `main` (`01_interfaces.md` §3). SHA-1 values are filled when the artifacts are published.

**Rights rule for the chapter (editor, 2026-10-29).** Common Crawl content carries no licence of its own, so the book never commits or hosts Common Crawl text: the c0, c1 and c2 artifacts are pins and manifests, and the notebooks fetch the pinned bytes from Common Crawl and rebuild the text. Data from FineWeb-Edu, FineWeb-2 and FineMath (ODC-By 1.0) ships with attribution (the dataset cards carry it): `lm-edu-annot-22k`, `lm-dedup-pairs`' base documents, the R2 shards and the rephrased corpus of 3.5. The other text the chapter hosts is ODC-By (FineWeb-Edu, FineWeb-2, FineMath), permissively licensed code files (3.4's licence filter), the MIT and CC-BY-SA benchmark items of 3.3, or the book's own text under CC BY-SA 4.0.

### Compute plan

| § | Tier | What runs | GPU-hours (full) | Committed artifacts |
|---|---|---|---|---|
| 3.1 | T0 | WARC prefix parse, 2 extractors, fastText LID, NFC, PII regexes | 0 | `lm-cc-warc-256m` (pin), `lm-c0-docs` (manifest), manifest stage records |
| 3.2 | T1 | Embedding 22k + c0 docs with Qwen3-Embedding-0.6B; linear head | ≈ 0.2 (to be measured) | `lm-edu-annot-22k`, `lm-quality-head`, `lm-c1-docs` |
| 3.3 | T0 | MinHash over 20k docs; suffix array over a 2 MB sample; MiniLM embeddings for a 4k-doc semantic subset | 0 | `lm-dedup-pairs`, `lm-contam-plant`, `lm-c2-docs`, `lm-corpus-manifest-v1` |
| 3.4 | T2 (T1) | 36 proxy runs × 2^27 tokens, ~8.1M-parameter `d2l.GPT` (T1: 25 runs × 2^24 tokens, ~4.6M parameters) | T2 ≈ 1.4 (sizing estimate; cap 3); T1 ≤ 0.25 | `lm-mix-runs-t2` (logs), token pools |
| 3.5 | T1 | 120-chunk live rephrase; 15 proxy runs (8.1M-parameter `d2l.GPT`, 2^24 tokens each); precomputed ≈ 4.7M-token rephrase | ≈ 0.3 notebook; authoring generation ≈ 1 h (to be measured) | `lm-rephrase-fwedu-5m` |
| 3.6 | T0 (T1 headline) | Tokenizer training, Viterbi, vocabulary extension; 3-tokenizer proxy LM; shard writer | ≈ 0.3 (to be measured) | `lm-r2-tokenizer`, `lm-fwedu-*` |

Sections per tier: T0 3, T1 2, T2 1 (v5 counts 3.6 at T0; the headline LM at T1 is a deviation recorded in the report).

## Assignment A3: a data pipeline with a checkable manifest

**Label:** `sec_lm_a3`

**Tier:** T0 for everything except the embedding classifier; the classifier has a T0 path. **Estimated student time:** 10–12 hours. **Package:** `a3_data/` with `starter.py`, `tests/`, `grade.py`.

### Tasks

1. **Extract and normalize (T0).** Implement `extract(record) -> Doc | None` over the 128 MiB prefix of the pinned WARC (`lm-cc-warc-256m`, first 128 MiB), using either trafilatura or resiliparse. Apply NFC, drop non-200 responses and non-`text/html`, and run fastText LID (`lid.176.ftz`, `__label__en` with score ≥ 0.65). Emit `docs.jsonl` and a transformation log per document.
2. **Heuristic filters (T0).** Implement the Gopher/C4/FineWeb rule set from 3.2 as `heuristic_filters(doc, cfg) -> list[str]` (empty list means keep). Required: the rules named in 3.2's table, each returning a stable reason code.
3. **MinHash/LSH deduplication (T0).** Implement exactly the signature and banding specified below. Tune `(b, r)` for the signature budget `k = b·r ≤ 128` on `lm-dedup-pairs` (train half), report on the test half. Cluster with union-find and keep the lowest-id document per cluster.
4. **Quality classifier (T0 or T1).** Train a classifier from `lm-edu-annot-22k` (20,000 train, 2,000 test). T1 path: Qwen3-Embedding-0.6B embeddings plus a linear head. T0 path: a hashed word-uni/bigram bag with `nn.EmbeddingBag` (the fastText model) in PyTorch. Choose a threshold; apply to the deduplicated c0 corpus.
5. **Decontamination (T0).** Using `lm-contam-plant`, flag the planted benchmark items with a 13-gram detector. Report recall on verbatim and on paraphrased plants and the false-positive rate on clean documents.
6. **Manifest (T0).** Write `manifest.json` in the `d2l-corpus/1` schema (3.1) listing every stage with counts, hashes and drop reasons.

### MinHash specification (exact, so that signatures are comparable bit for bit)

```
text -> NFC -> lowercase -> regex \w+ tokens -> word 5-gram shingles (joined by one space)
shingle hash  x = int.from_bytes(blake2b(s.encode(), digest_size=8).digest(), 'little')
prime p = 2^61 - 1;  for i in 0..k-1:  a_i, c_i from numpy.random.default_rng(seed=0xD2).integers(1, p, size=(2,k))
signature[i] = min over shingles of ((a_i * (x % p) + c_i) % p)      # use Python ints or uint64 with mulmod
band j = signature[j*r : (j+1)*r]; bucket key = blake2b(band.tobytes(), digest_size=8)
```

Documents with fewer than 5 words have no shingles and are never matched.

### Autograder checks (`grade.py`)

| Check | Rule | Tolerance |
|---|---|---|
| Signature bit-exactness | `signatures(fixture_docs, k=128)` equals the committed array | exact (`np.array_equal`) |
| Estimator unbiasedness | mean of Ĵ − J over the 500 fixture pairs | `abs(mean) ≤ 3·sqrt(J(1−J)/k)/sqrt(500)` evaluated per pair bin, i.e. within 3 standard errors |
| S-curve | empirical candidate rate of `lsh_candidates` for fixture pairs binned by exact J, at the student's (b, r) | each bin within `±(3·sqrt(p(1−p)/n_bin) + 0.01)` of `1 − (1 − J^r)^b` |
| Dedup quality | F1 of "duplicate" on the held-out half of `lm-dedup-pairs`, positives defined as exact 5-gram Jaccard ≥ 0.8 | ≥ (reference-solution F1 − 0.02); the reference F1 is measured by the course staff at release |
| Downstream determinism | `sha256` of the canonical manifest after stages 2–6 starting from the c0 that `d2l.rebuild_c0` rebuilds from the committed `lm-c0-docs` manifest (every document's text SHA-256 verified) | exact equality with the committed `expected_manifest_sha256` |
| Extraction | mean character-5-gram F1 against FineWeb text on 300 gold records | ≥ (reference-solution value − 0.05), measured at release |
| Classifier | binary F1 at score ≥ 3 on the 2,000 test labels | ≥ (T0 reference − 0.02) for the T0 path; ≥ the published 82% only if the student fine-tunes the encoder, which is not required |
| Decontamination | recall on verbatim plants; false positives on clean docs | recall = 1.0; false positives = 0 for 13-grams on 1,000 clean docs |

Thresholds written "reference − margin" are set once from the staff solution and recorded in `grade.py`; they are not a claim about what any method achieves.

### Written analysis (one page each)

1. Using your (b, r), state P(candidate) at J = 0.5, 0.8 and 0.9 and the expected number of candidate pairs among N = 10^6 documents if 0.1% of pairs have J = 0.5. Which term dominates the cost?
2. Which classifier errors on the 2,000 test labels are annotation noise and which are model errors? Support with five hand-read documents.
3. The audit table (3.2) shows removal rates by English variety proxy (ccTLD). Pick one gap, state two explanations, and one measurement that separates them.
4. Which planted paraphrases did your 13-gram detector miss? Propose a detector for them and state its false-positive risk.
5. Your manifest hash is reproducible. Name two sources of nondeterminism you removed and one you could not.

### T1 path

The classifier on Qwen3-Embedding-0.6B (≈ 22k texts truncated to 512 tokens) needs one Ada GPU for minutes; every other task is CPU. The T0 path replaces it with the hashed bag-of-n-grams classifier.

## Teaching sources

Keep all of v5's:

- [CS336](https://cs336.stanford.edu/): data lectures and the Common Crawl assignment. Take the structure of the extraction, filtering and deduplication assignment; the graded quantities (extraction quality against a reference, classifier quality, dedup quality) shape A3.
- [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1): the ablation methodology (small models, equal tokens, per-dump versus global deduplication). Take the experimental discipline, not the numbers.
- [Datatrove](https://github.com/huggingface/datatrove): reference pipeline (readers, filters, MinHash stages). Used as the production oracle in 3.1–3.3.
- [minbpe](https://github.com/karpathy/minbpe): minimal BPE, the from-scratch oracle for 3.6 beside `d2l.BPETokenizer`.
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook): data chapters, for mixtures and annealing.

Added by this blueprint (verified 2026-10-28):

- [FineWeb-2](https://arxiv.org/abs/2506.20920) and its [dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2): the multilingual pipeline whose per-language slices feed 3.6.
- [Mining of Massive Datasets, ch. 3](http://www.mmds.org/): the standard treatment of shingling, MinHash and LSH; the derivation in 3.3 follows it.
- [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) and [Datasheets for Datasets](https://arxiv.org/abs/1803.09010): the documentation standard for 3.1.

## Sections

### 3.1 From crawl to documents

- **Label:** `sec_lm_crawl` · **Tier:** T0 (CPU only; reduced config is the default: `crawl_mib=256`, scale to the whole file with `crawl_mib=1088`) · **Budget:** T0: ≈ 10 min wall-clock with 8 worker processes for 256 MiB, < 3 GiB RAM (both to be measured; extraction is the cost); T1 not needed · **Codex units:** 1.1 (text as data), 1.4 (building a corpus)
- **Builds on (book):** `sec_text-sequence` (bytes versus characters, UTF-8, the `BPETokenizer` byte view) for the Unicode subsection; `sec_read_write` and `d2l.download` for `DATA_HUB` use. Not re-taught: anything about tokenization (3.6 owns the rest).
- **Builds on (this part):** none. 3.1 is the first notebook of the chapter. It defines the manifest that 3.2–3.6 extend.
- **Exports** (`#@save`, all new, none reserved):
  - `d2l.iter_warc(source, n_bytes=None, types=('response',))`: yields `WarcRecord(id, url, date, status, mime, offset, length, headers, body)`. `source` is a URL or a path; for a URL it makes a `Range: bytes=0-(n_bytes-1)` request, decompresses member by member and drops a truncated final member. Contract: yields exactly the records `warcio.ArchiveIterator` yields for the same bytes.
  - `d2l.extract_text(html, method='trafilatura', url=None)`: returns text or `None`. `method` in `{'trafilatura', 'resiliparse', 'naive'}`. `'naive'` is the from-scratch link-density extractor below.
  - `d2l.detect_lang(text, k=1)`: wraps fastText `lid.176.ftz`; returns `[(label, score)]`, labels without the `__label__` prefix, newlines replaced by spaces first (fastText rejects them).
  - `d2l.scrub_pii(text)`: returns `(text, {'email': n, 'ip': n})`.
  - `d2l.Doc`: dataclass `(id, url, date, text, lang, lang_score, src, log, flags)`; `d2l.write_docs(path, docs)` and `d2l.read_docs(path)` for gzip-JSONL.
  - `d2l.CorpusManifest`: `.add_stage(name, params, n_in, n_out, drops, in_hash, out_hash)`, `.canonical_json()`, `.sha256()`, `.save(path)`, `.load(path)`, `.to_card()` (Markdown dataset card). Schema below.
  - `d2l.docs_hash(docs)`: SHA-256 over `sorted((id, sha256(text)))`; the content hash of a document list, independent of order.
  - `d2l.fetch_warc_range(pin='lm-cc-warc-256m', n_bytes=None, cache_dir=None) -> Path`: reads the pin from `DATA_HUB`, makes a `Range` request to Common Crawl for the first `n_bytes` (default the whole 256 MiB range), verifies the SHA-256 (the per-block hashes for a shorter prefix) and returns the cached local path; raises on a mismatch. With `D2L_OFFLINE=1` it uses only an already verified local copy.
  - `d2l.rebuild_c0(manifest='lm-c0-docs', warc_path=None) -> list[Doc]`: rebuilds the c0 documents from the manifest and the verified WARC range. Contract: for every manifest row the rebuilt `Doc` has `text_sha256` equal to the recorded one when the recorded extractor version is installed; rows that differ are returned in `.mismatches` (extractor drift), never silently kept.
- **Data artifacts:** `lm-cc-warc-256m` (a pin, not a mirror: `d2l.DATA_URL + 'lm/cc-main-2025-05-00000-256m.pin.json'` holds the WARC URL below, the byte range `0..268,435,455` (268,435,456 bytes) and its SHA-256, plus the SHA-256 of each 16 MiB block so that a shorter prefix can be verified; hashes to be measured by fetching the range once at authoring); `lm-c0-docs` (a manifest, `c0-manifest.jsonl.gz`: one row per kept document with `id`, `url`, `date`, `src = {warc, offset, length}`, `lang`, `lang_score`, `extractor`, `extractor_version`, `n_chars`, `text_sha256` and the transformation `log`; no text). Common Crawl content has no licence of its own, so no crawled text is committed or hosted: `d2l.rebuild_c0` fetches the pinned range from Common Crawl, re-extracts each listed record by `(offset, length)` with the recorded extractor version and checks `text_sha256`.
- **Models and data:**
  - **WARC (pinned):** `https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-05/segments/1736703361941.29/warc/CC-MAIN-20250126135402-20250126165402-00000.warc.gz`. HEAD on 2026-10-28 and again on 2026-10-29 returned HTTP 200, `accept-ranges: bytes` and `content-length: 1141586184` (≈ 1.06 GiB); the notebook fetches bytes 0..268,435,455 (256 MiB) of it, and its SHA-256 goes into the pin. The first 4 MiB (range request, 2026-10-28) holds 857 complete gzip members: 1 `warcinfo`, 286 `request`, 285 `response`, 285 `metadata`. So ≈ 285 responses per 4 MiB; a 256 MiB prefix therefore holds on the order of 18k responses (extrapolation, to be measured).
  - **WET of the same segment:** `…/wet/CC-MAIN-20250126135402-20250126165402-00000.warc.wet.gz`, 102,325,106 bytes. `conversion` records carry `WARC-Refers-To` (the response record id) and `WARC-Identified-Content-Language`.
  - **FineWeb reference:** `HuggingFaceFW/fineweb`, config `CC-MAIN-2025-05` (license ODC-By 1.0; `datasets-server` size: 265,662,695 rows, 600,030,893,317 parquet bytes), and `HuggingFaceFW/fineweb-edu`, same config (19,287,119 rows, 59,666,840,322 bytes). Columns: `text, id, dump, url, date, file_path, language, language_score, token_count` (+ `score, int_score` in Edu). `id` is a `<urn:uuid:…>` WARC record id; `file_path` is `s3://commoncrawl/crawl-data/CC-MAIN-2025-05/segments/1736703361941.29/warc/CC-MAIN-20250126135402-20250126165402-00000.warc.gz` for the documents from our file. The first 100 rows all have this file path, their `language_score` lies in [0.763, 0.991], and rows inside a file are ordered by URL, not by WARC offset (observed 2026-10-28). Stream rows with `datasets.load_dataset(..., streaming=True)` and stop after the file_path changes.
  - **fastText LID:** `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz` (HTTP 200 verified; the small 176-language model, ≈ 1 MB; license stated on the fastText page: CC-BY-SA 3.0 (verify at authoring)). Alternative with 217 languages: `facebook/fasttext-language-identification` (license CC-BY-NC-4.0, verified; non-commercial, so use only as an aside).
- **Packages:** `warcio>=1.7` (PyPI latest 1.8.2), `trafilatura>=2.0` (latest 2.3.1, 2026-10-06; pin the exact version in the manifest), `resiliparse>=1.0` (latest 1.1.0), `fasttext` (0.9.3; for NumPy 2 use `fasttext-numpy2-wheel` 0.9.2), `datasets>=3`, `regex`. Optional oracle: `datatrove>=0.10`.

#### Narrative arc

1. **The question.** Opening cell: stream the first 4 MiB of the pinned WARC and print the record-type histogram (1 / 286 / 285 / 285, as measured). Then show one response three ways on a Wikipedia-like page from the prefix: raw HTML length, the WET text for the same record, and the trafilatura text. Print the three character counts and the first 300 characters of each. Measured symptom: the byte count of the HTML is an order of magnitude above the text, and the WET text keeps menu lines (FineWeb reports exactly this reason for rejecting WET, so cite the paper, not our number).
2. **3.1.1 What a crawl contains.**
   - WARC record types (`warcinfo`, `request`, `response`, `metadata`), WAT, WET; one gzip member per record, which is what makes `Range` requests legal. `fig_lm_crawl_warc_layout`.
   - Cell: `iter_warc` on the `Range` prefix; table of record counts and median response bytes by MIME type and HTTP status. Cell: arithmetic for the whole file: bytes per response from the prefix, hence the estimated responses in 1,141,586,184 bytes (label the output "extrapolated").
   - What the crawl is not: not the whole web, URL sampling is skewed, robots-excluded and paywalled pages are absent, JavaScript-rendered content is absent. One bullet on `sec_environment-and-distribution-shift`: the corpus is a shifted sample of what people write.
3. **3.1.2 From HTML to text.**
   - Boilerplate: navigation, footers, cookie banners, comment threads. Three extractors: WET text (CC's own, no main-content detection), trafilatura (DOM scoring with fallbacks), Resiliparse (fast; used by DCLM; read DCLM's extraction ablation at authoring and cite its sign, not a number from memory). Plus the from-scratch `naive` extractor so that a reader sees the mechanism: split into blocks, score text density and link density, keep dense blocks.
   - Cell: run all four on the first 2,000 English responses; report mean characters kept, the fraction of empty outputs, and throughput in documents per second per core (all to be measured).
   - Failure on structure: code in `<pre>`, tables, mathematics (MathML with `<annotation encoding="application/x-tex">`, MathJax script nodes, `<img alt="\frac{…}">`). Show 3 concrete pages where each extractor drops a table or flattens a code block, then quantify below.
   - Metric definitions (all computed from the HTML, so independent of any extractor): **code recall** = fraction of non-empty lines inside `<pre>` and `<code>` blocks (whitespace-collapsed) found in the output; **table recall** = fraction of `<td>`/`<th>` strings of length ≥ 3 found in the output; **math recall** = fraction of TeX annotations, MathJax scripts and `alt` strings containing a backslash found in the output; **prose recall** = character-5-gram recall of `<p>` text; **precision proxy** = fraction of output characters that fall inside `<p>`, `<pre>`, `<table>`, `<li>`, `<h1-6>` elements of the source.
4. **3.1.3 Language identification.**
   - fastText `lid.176.ftz`: how the classifier works (hashed n-gram bag, linear), why short text and code are misclassified, and the effect of the threshold. FineWeb keeps English with score ≥ 0.65 (FineWeb paper, section 3.2/3.3); our table sweeps 0.5–0.9.
   - Cell: agreement of fastText with CC's CLD2 label (`WARC-Identified-Content-Language` in WET) on the same records, as a confusion table of the top 8 languages. Cell: retention versus threshold; the retained fraction at 0.65 compared with FineWeb's retained documents from the same file (how many of "our en ≥ 0.65" are present in FineWeb; the ones missing are the rest of FineWeb's pipeline: URL filter, repetition filters, deduplication).
   - Pitfall cell: one 8-word English sentence, the same sentence inside a code comment, and a code-switched sentence; print the scores.
5. **3.1.4 Unicode normalization and what it destroys.**
   - NFC versus NFKC (Unicode TR15). Table `tab_lm_crawl_norm_loss`: for each form, on c0, number of documents changed, characters changed, and 10 examples drawn at random: ligatures (`ﬁ`), superscripts (`x²` to `x2`), full-width Latin, circled digits, mathematical alphanumerics (`𝑥` to `x`), `…`, and `№`. NFKC merges code-relevant distinctions and math symbols; NFC only reorders and composes canonically equivalent sequences (matters for Vietnamese, Hindi, Korean Hangul composition).
   - Whitespace and control characters: NBSP, zero-width space/joiner (the joiner is meaningful in Indic scripts and emoji sequences; stripping it changes meaning), CRLF, form feeds, U+FFFD counts as an encoding-failure signal.
   - **Rule for the book:** NFC on every document; never NFKC on a general corpus; whatever normalization is used is applied once, before any tokenizer, and bits per byte is computed on the bytes of the *normalized* text, identically for every tokenizer being compared (3.6). Round-trip cell: `nfc(nfc(s)) == nfc(s)` on all of c0; count documents whose raw bytes differ from NFC bytes.
6. **3.1.5 Personal data.**
   - What is removable by rule: emails, public IPv4/IPv6 (FineWeb anonymizes these with regexes; the paper states this), phone-like numbers (high false-positive rate on numeric tables). What is not: names, addresses, health information in prose.
   - Cell: `scrub_pii` on c0; counts per document; evaluate on a hand-built fixture of 60 positive and 60 negative strings (committed in the notebook): precision and recall per pattern. Show the failure on version numbers (`10.2.3.4`) and on ISBN/phone digits; show that scrubbing changes bytes (for 3.6 bpb accounting).
7. **3.1.6 Provenance, licenses and opt-outs.**
   - Fields every document carries: `id`, `url`, `date`, `src = {warc, offset, length}`, `lang`, `lang_score`, `extractor`, `log`. Why `offset/length` matters: a document is re-fetchable from the archive by byte range, so a takedown request can be honored by id.
   - What the crawl does and does not license: Common Crawl is an archive, not a license grant; robots.txt, site terms and copyright still apply; *Consent in Crisis* (2407.14933) reports the growth of robots and terms restrictions on crawled domains; StarCoder 2 / The Stack v2 (2402.19173) is the worked example of license detection and an opt-out process for code. Teach that rights metadata are engineering inputs: keep the license field, the opt-out list and the removal log as first-class data. Not legal advice (one sentence).
   - Dataset cards: `CorpusManifest.to_card()` fills sections (motivation, composition, collection, preprocessing, uses, distribution, maintenance) in the *Datasheets for Datasets* order and the Hugging Face card format; the notebook prints the rendered card for c0.
8. **3.1.7 The manifest.**
   - Schema `d2l-corpus/1` (table `tab_lm_crawl_manifest_schema`):
     ```
     {"schema": "d2l-corpus/1", "name": str, "sources": [{"id", "uri", "revision", "license", "n_docs", "bytes"}],
      "stages": [{"name", "impl", "version", "params": {...}, "n_in", "n_out", "tokens_out", "drops": {reason: count},
                  "in_hash", "out_hash"}],
      "tokenizer": null | {"file", "sha256", "vocab_size", "specials": [..]},
      "format":    null | {"dtype": "uint16", "eos_id": 0, "endianness": "little"},
      "splits":    null | {"train": {"n_tokens", "n_docs", "shards": [{"file", "n_tokens", "n_docs", "sha256"}]}, "val": {...}}}
     ```
     The identity `n_in == n_out + sum(drops.values())` is asserted per stage (a document is charged to the *first* rule that drops it; the order is part of `params`). Keys are sorted, floats are rounded to 4 decimals, no timestamps, no absolute paths. `sha256()` hashes the canonical JSON.
   - Cell: build the c0 manifest (`extract`, `lang_id`, `nfc`, `pii` stages), save it, reload it, assert the hash is unchanged, then permute the document order and assert `docs_hash` is unchanged.
   - The funnel `fig_lm_crawl_funnel`: responses -> 200/HTML -> non-empty extraction -> English ≥ 0.65 -> after PII; counts from the manifest.

#### Derivations

None. This section has no derivation; its quantitative content is the range-request arithmetic and the manifest identity above, both exercised by cells.

#### From-scratch implementation

Budget ≈ 140 lines in total.

| Component | Signature | ≈ lines | Test (reference, tolerance, seed) |
|---|---|---|---|
| WARC reader | `iter_warc(source, n_bytes, types)` over `zlib.decompressobj(31)` members: parse the WARC header, HTTP status line and headers, body by `Content-Length` | 35 | Against `warcio.ArchiveIterator` on the 4 MiB prefix: identical record ids, types, offsets, body SHA-256 for all 857 records; histogram equals {warcinfo 1, request 286, response 285, metadata 285} |
| Naive extractor | `extract_naive(html)`: `html.parser` block segmentation (`p, div, li, td, pre, h1-6`), per block text length `t`, link text length `l`; keep a block if `t ≥ 40` and `l/t ≤ 0.33`, plus neighbors of kept blocks within 2 positions if `t ≥ 15` | 45 | On 5 hand-built HTML fixtures (article with sidebar, forum, table, `<pre>`, empty) output equals the golden text; on 200 responses its prose recall is within 0.15 of trafilatura's (a sanity band, not a claim) |
| PII | `scrub_pii`: three compiled regexes (email; IPv4 with octet check and private-range exclusion; IPv6 simple form) | 20 | Fixture of 120 strings: zero false negatives on the positive set by construction; the false-positive list is printed |
| Manifest | `CorpusManifest` with canonical JSON and hash | 40 | Hash stable under key reorder, under save/load, and under document reorder through `docs_hash` (seed 0 permutation) |

#### In production

- `warcio.ArchiveIterator` (stream reader) and `trafilatura.extract(html, url=url, include_tables=True, favor_precision=False)` for the pinned version; `resiliparse.extract.html2text.extract_plain_text(html, main_content=True)`. The notebook runs the production extractors and compares them with the from-scratch pieces: reader equality is exact; the naive extractor is a baseline, not an oracle.
- Datatrove: show a six-line `LocalPipelineExecutor([WarcReader(...), Trafilatura(), LanguageFilter(...), JsonlWriter(...)])` as text only, with the field mapping to our `Doc`; do not run it in the build. Datatrove also stores `metadata` per document, which matches our `src` field.
- The FineWeb reference documents are the oracle for the extraction of the 2025-05 file: our trafilatura output on a matched record is expected to agree closely with FineWeb's text if versions and settings agree; differences are reported as a table of 10 sampled pairs (diff snippets), because FineWeb's exact trafilatura settings and version are not recorded in the dataset (state this).

#### Headline experiment

- **Hypothesis (to test, not assume):** extractors differ mainly on structured content. The three methods have similar prose recall but different code, table and math recall, and WET has lower precision proxy because of menu text.
- **Setup:** all English (fastText ≥ 0.65) responses with HTTP 200 and `text/html` in the first 256 MiB of the pinned WARC, split by content class by structural markers in the HTML (a page can be in several classes): code (has `<pre>` or `<code>` with ≥ 3 lines), table (≥ 1 `<table>` with ≥ 4 cells), math (≥ 1 TeX annotation, MathJax or LaTeX `alt`), prose-only. Four extractors: WET (matched by `WARC-Refers-To`; read the first 128 MiB of WET), `naive`, trafilatura, Resiliparse. Metrics as defined in 3.1.2. No GPU, no seeds.
- **Procedure:** compute per-document recalls and the precision proxy; aggregate per class and extractor; paired bootstrap (`d2l.paired_bootstrap`, `n_boot=10000`, `cluster` = registered domain of the URL so that one site's templated pages count once) for trafilatura versus Resiliparse on each class.
- **Plot (`fig_lm_crawl_extractor_recall`):** grouped bars, x = content class (prose, code, table, math) with n in the tick label, y = mean recall (0–1), one color per extractor, 95% bootstrap interval as error bars. Second panel: precision proxy by extractor.
- **Expected outcome:** direction only. The FineWeb paper reports that trafilatura on WARC gave a more performant ablation model than WET, and rejected WET for boilerplate and menu text; whether structured-content recall differs is to be measured. Classes with n < 30 documents are shown with an "n too small" mark and no interval.
- **Noise check:** extractors are deterministic, so the only noise is document sampling; the bootstrap interval is the noise statement; a difference is reported only if the paired interval excludes 0. Because the FineWeb text is trafilatura-based, agreement with it is reported separately as "same as FineWeb", never as accuracy.
- **T1 reduction:** none needed (T0). Scale by `crawl_mib`.
- **Artifacts committed:** `lm-cc-warc-256m` (the pin), `lm-c0-docs` (the manifest of extracted, normalized, scrubbed documents: ids, offsets, text hashes, logs; no text), the c0 stage manifest, the aggregated metric table as JSON.

#### Figures and tables

- `fig_lm_crawl_warc_layout`: diagram of gzip members (warcinfo, request/response/metadata triples) with a byte axis and a `Range` bracket; built from the prefix scan (offsets and lengths of the first 12 records).
- `fig_lm_crawl_extractor_recall`: described above.
- `fig_lm_crawl_funnel`: horizontal funnel from the manifest counts.
- `tab_lm_crawl_norm_loss`: NFC versus NFKC effects on c0 with examples.
- `tab_lm_crawl_manifest_schema`: the schema with field meanings.
- `tab_lm_crawl_pii`: pattern, precision, recall on the fixture, top false positives.

#### Pitfalls and misconceptions

- Treating WET as "the text of the web": it has no main-content detection and a different record set.
- Dropping a truncated last gzip member silently: count and report it; reading a prefix must not yield a half record.
- Decoding bodies as UTF-8 without charset detection: pages declare latin-1 or Shift-JIS; use the `Content-Type` charset, then `<meta charset>`, then a detector, and log the choice.
- Counting `request`/`metadata` records as documents.
- Joining against FineWeb by URL instead of record id: the same URL is crawled many times; confirm on the prefix that ids match (open risk below).
- LID on short strings and code; treating a LID score as a probability of "being English text".
- NFKC "to clean up": it silently rewrites code and mathematics.
- Putting timestamps or absolute paths into the manifest, which makes the hash irreproducible.
- Believing "public means permitted".

#### Summary (draft)

- A crawl is a sequence of independently gzipped WARC records, so a prefix is a valid sample and a document can be re-fetched from its byte range.
- Text extraction is a modeling step: extractors differ in what they keep from code, tables and math, and the choice shows up in model quality (FineWeb's WET versus trafilatura ablation).
- Language identification, normalization and PII scrubbing are lossy rules; each is logged with counts so its effect on the corpus is measurable.
- NFC once, before tokenization, and bytes for bpb are counted after it.
- Provenance (id, URL, byte range, license, removal log) is data, and a manifest with canonical JSON makes the corpus rebuildable from hashes.

#### Exercises

1. **Check.** The first 4 MiB of the WARC hold 285 response records and 857 records in all. Explain why a prefix of a WARC file can be parsed record by record without reading the rest, why the last record of a range request must be discarded, and why the same does not hold for a prefix of a `.tar.gz`.
   *Sketch:* each record is a separate gzip member, so member boundaries are self-describing; the last member is cut mid-stream and fails the CRC/EOF; a tar.gz is one gzip stream with no independent restart points.
2. **Check.** For each of these characters state whether NFC, NFKC or both change it, and one downstream harm of NFKC: `ﬁ` (U+FB01), `x²`, `é` written as `e` + U+0301, `𝑥` (U+1D465), `‍` in `क्‍ष`.
   *Sketch:* NFC composes `e`+U+0301 to U+00E9 only; NFKC additionally maps `ﬁ` to `fi`, `²` to `2`, `𝑥` to `x`; neither removes U+200D (it is not a compatibility character), which is why zero-width stripping is a separate, harmful, rule for Indic text. Harm: `x²` becomes `x2` in math and code.
3. **Derive.** A stage drops documents by an ordered list of rules R1..Rm; a document is charged to the first rule that fires. Let `p_j` be the marginal firing probability of Rj and assume independence. Derive the expected charged fraction of Rj and show that it depends on the order. Give the order that minimizes the compute of evaluating rules when rule j costs `c_j` per document (assume a document stops at the first firing rule).
   *Sketch:* charged_j = p_j ∏_{i<j}(1−p_i). Expected cost = Σ_j c_j ∏_{i<j}(1−p_i); an adjacent swap argument shows ascending `c_j/p_j` is optimal.
4. **Derive.** Let the compressed sizes of consecutive response triples (request, response, metadata) be i.i.d. with mean s and variance v. Show that a prefix of B bytes contains N(B) complete triples with E[N(B)] ≈ B/s and Var[N(B)] ≈ B·v/s³ for large B, and use the measured 285 responses in 4 MiB to give a 95% interval for the number of responses in the whole 1,141,586,184-byte file (state the extra assumption that the prefix is representative, and when it is not).
   *Sketch:* renewal-theory CLT: N(B) is asymptotically normal with mean B/s and variance B·v/s³. The prefix is a time-ordered slice of the segment, so anything correlated with fetch order (hosts, time of day) breaks the assumption; compare the 4 MiB, 64 MiB and 256 MiB prefixes to test it.
5. **Measure.** Predict, then measure, how the number of English documents retained changes as the LID threshold moves from 0.5 to 0.9 in steps of 0.05, and what fraction of the extra documents at 0.5 are code, boilerplate or non-English. Hand-label 50 documents between 0.5 and 0.65.
6. **Measure.** Replace trafilatura's `favor_precision=False` with `favor_precision=True` and `favor_recall=True`. Predict the direction of the change in code, table and math recall and in the precision proxy, then measure with bootstrap intervals.
7. **Open.** Add a third extractor of your choice (for instance a small HTML-to-Markdown converter or a DOM-visual heuristic). Design an evaluation that does not use any extractor's output as reference, and report where yours wins and loses. Include a one-page dataset card for the result.

#### Further reading

- Penedo et al., [FineWeb](https://arxiv.org/abs/2406.17557): extraction (WARC with trafilatura versus WET), language threshold, PII, the pipeline we mirror.
- Penedo et al., [RefinedWeb](https://arxiv.org/abs/2306.01116): the earlier "web data only" pipeline with trafilatura.
- Kreutzer et al., [Quality at a Glance](https://arxiv.org/abs/2103.12028): audit of language labels in crawled multilingual data.
- Dodge et al., [Documenting Large Webtext Corpora](https://arxiv.org/abs/2104.08758): the audit method behind the 3.2 exercise.
- Elazar et al., [What's In My Big Data?](https://arxiv.org/abs/2310.20707): corpus inspection tools and findings.
- Longpre et al., [Consent in Crisis](https://arxiv.org/abs/2407.14933) and Lozhkov et al., [StarCoder 2 and The Stack v2](https://arxiv.org/abs/2402.19173): opt-outs and licenses.
- Gebru et al., [Datasheets for Datasets](https://arxiv.org/abs/1803.09010).
- Tutorials: [Common Crawl, get started](https://commoncrawl.org/get-started); [Trafilatura docs](https://trafilatura.readthedocs.io/) and its [evaluation page](https://trafilatura.readthedocs.io/en/latest/evaluation.html); [fastText language identification](https://fasttext.cc/docs/en/language-identification.html); [Unicode TR15](https://www.unicode.org/reports/tr15/).
- Code: [warcio](https://github.com/webrecorder/warcio), [trafilatura](https://github.com/adbar/trafilatura), [resiliparse](https://github.com/chatnoir-eu/chatnoir-resiliparse), [datatrove](https://github.com/huggingface/datatrove).

#### Risks and open decisions

- **Common Crawl availability or rate limits during builds.** Fallback: a local copy of the verified range (`d2l.fetch_warc_range` caches it; with `D2L_OFFLINE=1` only that copy is used and the notebook stops with fetch instructions if it is missing). The pinned range is immutable, so a retry is safe.
- **FineWeb id join.** FineWeb's `id` is believed to equal the WARC-Record-ID of the response. Verify on the 285 prefix responses at authoring; fallback: join on normalized URL plus `date`, reported as a weaker match.
- **Trafilatura version drift** changes the extracted text and therefore `lm-c0-docs`' hash. Fallback: the manifest records the extractor version and every document's `text_sha256`; `d2l.rebuild_c0` reports mismatches, the notebook environment pins the versions, and the A3 autograder starts from the rebuilt c0 and fails with the version message if any `text_sha256` differs.
- **fastText wheels** (`fasttext` 0.9.3 does not build cleanly against NumPy 2; `fasttext-numpy2-wheel` exists). Fallback: the pure-PyTorch hashed-bag model of 3.2 trained on `lid.176` data is out of scope; instead use `lingua-language-detector` (PyPI 2.2.0, requires Python ≥ 3.12) and say so.
- **Resiliparse wheels** may be missing on the macOS ARM build hosts. Fallback: run trafilatura and `naive` only; mark Resiliparse results as "from the committed table".
- **LID license:** `lid.176` is CC-BY-SA; the 217-language model is CC-BY-NC; use the former in notebooks.
- **Takedown and rights for sampled pages (decided 2026-10-29):** Common Crawl content has no licence of its own, so the book commits and hosts none of it. `lm-c0-docs` is a manifest and the notebook rebuilds the text from the pinned range fetched from Common Crawl; a takedown request is honored by deleting the row for its `id`. No domain-level split of permissively licensed pages is made.

### 3.2 Quality filtering

- **Label:** `sec_lm_quality_filtering` · **Tier:** T1 (embedding passes; a T0 path uses the hashed-bag classifier; reduced config: `audit_docs=5000`, `train_docs=5000`) · **Budget:** T1 ≈ 10–15 min and ≤ ~6 GiB peak (Qwen3-Embedding-0.6B in BF16 is ≈ 1.2 GB of weights; activations at batch 64 × 512 tokens; both to be measured); T0 path ≈ 3 min on CPU · **Codex units:** 1.5 (filtering half)
- **Builds on (book):** `sec_environment-and-distribution-shift` (a filter keyed on a proxy label is a covariate shift; one paragraph, with `:numref:`); `sec_mdl-statistics` (bootstrap, for the audit intervals). Not re-taught: logistic regression and thresholds (cite the Basics linear-classification chapters by one sentence).
- **Builds on (this part):**
  - 3.1: `d2l.Doc`, `d2l.read_docs`, `d2l.CorpusManifest`, `d2l.detect_lang`, `lm-c0-docs` (rebuilt into documents with `d2l.rebuild_c0`).
  - 1.5: `d2l.paired_bootstrap` for classifier comparisons and group differences.
  - 2.3: frozen embedding plus linear head and Qwen3-Embedding-0.6B usage (what pooling and padding it expects); the notebook says "as in 2.3" in one sentence.
  - `assumed:` none from other chapters. The embedding helper below is exported here under a chapter-specific name so it cannot collide with 2.3's.
- **Exports** (`#@save`, new):
  - `d2l.embed_docs(texts, model_id='Qwen/Qwen3-Embedding-0.6B', max_len=512, batch=64, pool='last', device=None)`: returns an `(n, d)` float16 NumPy array of L2-normalized embeddings; sorts by length internally, restores order, left-pads for `pool='last'`. Contract: equals `sentence_transformers.SentenceTransformer(model_id).encode(texts, normalize_embeddings=True)` up to cosine ≥ 0.999 on 64 texts (BF16).
  - `d2l.GOPHER_CFG`, `d2l.heuristic_filters(text, cfg=None)`: returns the ordered list of reason codes that fire (empty means keep).
  - `d2l.fit_quality_head(X, y, kind='regression'|'logistic', epochs=..., lr=..., seed=0)`: returns a `torch.nn.Linear(d, 1)` and its training curve.
  - `d2l.HashedBagClassifier(num_buckets=2**20, dim=64, ngrams=(1, 2))`: the fastText model in PyTorch (`nn.EmbeddingBag` over hashed word n-grams, mean-pooled, linear output).
  - `d2l.audit_removals(df, keep, by, ref=None, n_boot=2000, cluster=None, seed=0)`: DataFrame of group, n, removal rate, 95% bootstrap interval, difference to `ref` group with interval.
- **Data artifacts:** `lm-edu-annot-22k` (22,000 FineWeb-Edu annotation rows: `id`-free JSONL with `text` truncated to 3,000 characters, `score` 0–5, `split` in {train 20,000, test 2,000}; sampled with seed 0 from the 467,424 rows, uniformly so the label prior is preserved); `lm-quality-head` (weights of the head fitted on Qwen3-Embedding-0.6B embeddings, `.npz`); `lm-c1-docs` (c0 after `heuristics` and `edu_score`: a JSONL manifest keyed by c0 document id with `flags.edu_score` rounded to 3 decimals and the heuristic reason codes; no text, which comes from rebuilding c0). `lm-edu-annot-22k` ships with attribution: its rows are FineWeb-Edu annotation text (ODC-By 1.0, `HuggingFaceFW/fineweb-edu-llama3-annotations`), and the dataset card carries the attribution.
- **Models and data:**
  - `HuggingFaceFW/fineweb-edu-llama3-annotations` (license ODC-By; 467,424 rows, 1,590,654,012 parquet bytes in 8 files of ≈ 199 MB; columns `text`, `metadata{date, dump, file_path, language, language_score, token_count, url}`, `prompt`, `score` int 0–5). Labels come from Llama-3-70B-Instruct; the dataset card states the additive 0–5 rubric. Licence recorded 2026-10-29: `https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations` reports `license: odc-by`, so the 22k-row sample and the head fitted on it ship with attribution.
  - `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0; 595,776,512 parameters, 1024-d output). Pin the revision at authoring.
  - Comparison classifier: `HuggingFaceFW/fineweb-edu-classifier` (Apache-2.0; 109,483,009 parameters), the frozen `Snowflake/snowflake-arctic-embed-m` (Apache-2.0, 108,891,648 parameters) plus a regression head. The model card says the embedding and encoder layers were frozen, the head trained for 20 epochs at learning rate 3e-4 on 450k annotations, and reports a binary F1 of 82% at threshold 3 (read the exact card line at authoring); its hold-out class supports are 5,694 / 26,512 / 10,322 / 3,407 / 807 / 125 for scores 0–5 (46,867 total), so the share with score ≥ 3 is 9.26%.
  - Perplexity-filter model: `HuggingFaceTB/SmolLM2-135M` (Apache-2.0, 134,515,008 parameters).
  - Audit streams: `HuggingFaceFW/fineweb`, config `CC-MAIN-2025-05`, first `audit_docs=20000` rows streamed (rows come ordered by URL within a WARC file, so shuffle with a seeded buffer of 2,000 or take every 50th row of the first 1,000,000 to avoid a lexicographic sample; decide at authoring); `HuggingFaceFW/fineweb-2` configs `deu_Latn, hin_Deva, cmn_Hani, jpn_Jpan, tha_Thai, swh_Latn` (500 documents each, license ODC-By; columns include `text, id, url, language_script, minhash_cluster_size`).
  - Reference decisions: `HuggingFaceFW/fineweb-edu` config `CC-MAIN-2025-05` (19,287,119 rows, `score`, `int_score`): 19,287,119 / 265,662,695 = 7.26% of the FineWeb documents of this dump survive the educational filter.
- **Packages:** `torch`, `transformers>=4.51` (Qwen3 support; pin), `sentence-transformers` (oracle only), `datatrove>=0.10` (oracle only), `scikit-learn` for metrics only, `pandas`.

#### Narrative arc

1. **The question.** Opening cell: print the two numbers 19,287,119 and 265,662,695 and their ratio, 7.26%: the FineWeb-Edu filter throws away 93% of the documents of one dump. Then print five c0 documents that pass every heuristic rule and have an LLM-annotation-style score of 0, next to five that a heuristic rule rejects but a human would keep. The symptom: rules are cheap and blunt, and classifiers inherit the annotator's taste.
2. **3.2.1 Heuristic rules.** A table (`tab_lm_quality_rules`) of rule families with their source and what they catch:
   - Gopher quality rules (Rae et al.; thresholds transcribed from the paper's appendix and unit-tested against its table): word count window, mean word length window, symbol-to-word ratio for `#` and `...`, fraction of lines starting with bullets, fraction ending with an ellipsis, fraction of words with an alphabetic character, required stop words.
   - Gopher repetition rules: duplicate-line and duplicate-paragraph fractions (by count and by characters), top n-gram fractions for n = 2, 3, 4, duplicated n-gram fractions for n = 5…10.
   - C4 rules: terminal-punctuation lines, minimum sentences, `javascript`, "lorem ipsum", curly-brace lines, policy/cookie lines, bad-words list.
   - FineWeb's custom filters (FineWeb paper): fraction of lines ending in terminal punctuation ≤ 0.12 removes 10.14% of tokens; fraction of characters in duplicated lines ≥ 0.1 removes 12.47%; fraction of lines shorter than 30 characters ≥ 0.67 removes 3.73%; together about 22% of tokens, with an aggregate benchmark gain of about 1% (all from the paper). FineWeb applies all C4 filters except terminal punctuation because that one removed about 30% of tokens.
   - Cells: run `heuristic_filters` on c0 (about thousands of documents) and print a reason-code histogram and the funnel (first-firing attribution as in 3.1's exercise 3); show the 3 most common rule pairs that co-fire.
   - Language dependence cell: apply the same rules to the six FineWeb-2 languages; Chinese, Japanese and Thai text has no spaces, so "words" per document is 1 and the word-count rule removes nearly everything. Print the removal rate per language (to be measured). This is the first audit finding.
3. **3.2.2 Perplexity filters.** CCNet's idea (a language model trained on a clean corpus scores documents; keep the low-perplexity buckets; KenLM on Wikipedia in the paper). Cell: per-document mean NLL under SmolLM2-135M on the first 512 tokens for 3,000 c0 documents; scatter of NLL against classifier score later; list the 10 lowest-NLL documents (boilerplate, repeated text) and the 10 highest (code, tables, non-standard orthography, names). Message: perplexity measures similarity to the reference LM's training text, not quality; very low values mean repetition.
4. **3.2.3 Classifiers on LLM-annotated labels.**
   - Recipe: an LLM scores a sample with a rubric; a cheap model learns to reproduce the score; thresholds select the corpus (FineWeb-Edu: Llama-3-70B-Instruct scores, frozen Snowflake encoder plus regression head, keep `int_score ≥ 3`, giving 1.3T tokens versus 5.4T at threshold 2, both from the dataset card). DCLM's variant: a fastText classifier whose positives are instruction-style data and ELI5 answers and whose negatives are random web text, keeping the top fraction (read DCLM §4 for the exact data and fraction at authoring). Nemotron-CC: an ensemble of classifiers, re-bucketed by quality, which keeps more tokens than a single classifier (read the paper; cite its direction, not a number from memory).
   - Cells: (a) build `lm-edu-annot-22k` (code shown, artifact loaded in the build); label histogram; (b) `embed_docs` on 22,000 texts; (c) `fit_quality_head` regression (MSE on the 0–5 score) and logistic (target ≥ 3); (d) `HashedBagClassifier` T0 path; (e) the released FineWeb-Edu classifier on the same 2,000 test rows.
   - Metrics: test F1 at score ≥ 3, precision–recall curve, ROC-AUC, mean absolute error of the regression score, confusion matrix of the rounded score against the label. Paired bootstrap over the 2,000 test rows between any two classifiers.
   - Label-noise reading: compare the confusion matrix with the annotator's rubric; the model card says most errors are annotation noise. Section 3.2.5's derivation turns that into an F1 ceiling.
5. **3.2.4 Choosing a threshold under a token budget.**
   - Retained fraction ρ(t) and precision P(t) against t, on the test set and on the 20k audit stream. Convert ρ into tokens for a stated corpus size. Link: if the budget is data-limited (4.7), a low threshold keeps more unique tokens; if compute-limited, a higher threshold. Show `fig_lm_quality_pr_retention`.
   - The classifier gives a score, not a decision: keep the continuous `edu_score` in the manifest and apply the threshold as a parameter of the stage. This lets 3.4 use score bands as mixture sources.
6. **3.2.5 Auditing removals.**
   - What an audit is: a table of removal rates by group with intervals, and a reading of examples. Groups: language (the six FineWeb-2 languages, heuristics only; the classifier is English-only and says so), domain (host class by rule: `wikipedia.org`, `blogspot|wordpress|medium`, news, forum/Q&A by path keyword, `.edu`, `.gov`, shopping by keyword; list the rule), English variety proxied by ccTLD (`.uk, .au, .ca, .in, .ng, .za, .ph, .ie, .nz` against `.com` as reference; state that ccTLD is a weak proxy for dialect).
   - Why this matters: Gururangan et al. show that quality filters keyed on a prestige corpus favor certain sociolects and sources; Dodge et al. show that C4's blocklist removes text by and about particular groups; both link back to `sec_environment-and-distribution-shift`.
   - Cell: `audit_removals` for heuristics alone, classifier alone and both, cluster-bootstrapped by registered domain; forest plot.
   - False-removal review: sample 40 documents removed by the classifier whose `edu_score` is within 0.3 of the threshold, plus 40 removed by heuristics only; print excerpts and a column for a human label (the notebook ships 40 pre-labeled rows by the authors so the reader can compare; the labeling rubric is two lines). A second annotator, the part's judge `Qwen/Qwen3.5-9B` (bf16, non-thinking) with a judge prompt, is an optional T1 cell that claims a whole 24 GB GPU: report the agreement between judge and human on the 80 rows with Cohen's kappa.
7. **3.2.6 Putting it in the pipeline.** Stage records for `heuristics` (drops by reason) and `edu_score` (no drops; a threshold parameter and the head's hash); write the `lm-c1-docs` flags manifest; extend the c0 manifest.

#### Derivations

**D1. F1 of a thresholded classifier from prior, recall and retention.**

- *Statement.* Let π = P(y = 1), TPR the true-positive rate, ρ the retained fraction. Then precision = π·TPR/ρ and F1 = 2π·TPR/(π + ρ). Retention is ρ = π·TPR + (1 − π)·FPR.
- *Assumptions.* A binary label at threshold 3 and a binary decision at threshold t; i.i.d. documents.
- *Steps.* (1) ρ = P(pred = 1) = π·TPR + (1 − π)·FPR by total probability. (2) Precision = P(y = 1 | pred = 1) = π·TPR/ρ by Bayes. (3) Recall = TPR. (4) F1 = 2PR/(P + R) with P = π·TPR/ρ gives numerator 2π·TPR²/ρ and denominator TPR(π + ρ)/ρ. (5) Cancel: F1 = 2π·TPR/(π + ρ).
- *Reading.* F1 depends on t only through TPR and ρ. For a fixed π, F1 is largest when ρ is close to π and TPR close to 1. A filter that keeps ρ ≪ π has high precision and low recall; the curve in `fig_lm_quality_pr_retention` is this identity traced over t.
- *Numerical check.* On the 2,000 test labels and our classifier scores, compute F1 by the direct formula and by `2*pi*TPR/(pi+rho)` at 50 thresholds; assert equality to 1e-12. Print π (expected ≈ 0.09 from the card's hold-out supports; measured on our sample).

**D2. F1 ceiling under label noise.**

- *Statement.* Suppose the true binary quality is y*, the annotator flips it with probability η in each direction, and a classifier predicts y* perfectly. The observed precision is 1 − η, the observed recall is π*(1 − η)/(π*(1 − η) + (1 − π*)η), and F1 is their harmonic mean.
- *Assumptions.* Symmetric, label-independent noise; true prior π*; perfect classifier.
- *Steps.* (1) P(y_obs = 1 | pred = 1) = P(y_obs = 1 | y* = 1) = 1 − η. (2) P(pred = 1 | y_obs = 1) = P(y* = 1 | y_obs = 1) = π*(1 − η)/(π*(1 − η) + (1 − π*)η) by Bayes. (3) Combine in F1.
- *Reading.* With π* = 0.0926 (the card's prior, used as π*): η = 0.02 gives F1 = 0.901, η = 0.05 gives 0.779, η = 0.10 gives 0.625 (closed-form values, recomputed by the check). Measured classifier F1s can be compared with this curve to bound the annotator's noise rate; the card's 82% would be below the η = 0.05 ceiling.
- *Numerical check.* Monte Carlo (seed 0, 10^6 draws): simulate y*, noisy label, perfect predictor, compute F1, assert agreement with the closed form within 3 binomial standard errors.

#### From-scratch implementation

Budget ≈ 150 lines.

| Component | Signature | ≈ lines | Test |
|---|---|---|---|
| Heuristic rules | `heuristic_filters(text, cfg)`: a list of `(code, fn)` pairs, repetition statistics by `collections.Counter`, word split by whitespace and a `regex` fallback for CJK | 60 | 12 hand-written strings, one per rule code, each firing exactly its rule (golden); agreement with `datatrove` filters (`GopherQualityFilter`, `GopherRepetitionFilter`, `C4QualityFilter`, `FineWebQualityFilter`) on 2,000 c0 documents, with every disagreement printed and explained (tokenizer differences expected) |
| Hashed-bag classifier | `HashedBagClassifier`: word uni/bigrams hashed with `hash(x) % buckets` through a fixed `zlib.crc32` | 35 | Overfits a 64-document toy set to 100% train accuracy (seed 0); on the 2,000 test labels its F1 is reported next to the embedding head |
| Linear head | `fit_quality_head`: full-batch AdamW on cached embeddings | 25 | On a synthetic linearly separable set (seed 0) recovers accuracy ≥ 0.99; regression loss decreases monotonically for a small learning rate |
| Audit | `audit_removals`: cluster bootstrap by resampling domains | 30 | On synthetic groups with known removal rates (0.1 vs 0.3, n = 1,000 each, seed 0) the 95% interval covers the truth in 1 − 0.05 ± 0.04 of 200 repeats |

#### In production

- Datatrove's filter blocks (as above) are the oracle for the rules; the released `HuggingFaceFW/fineweb-edu-classifier` through `AutoModelForSequenceClassification` (the model card's snippet) is the oracle for our own head: its scores on the 2,000 test rows are the second row of the comparison table.
- fastText `train_supervised` as the T0 reference for the hashed-bag classifier (same features); the notebook shows the three-line call and compares F1, not implementation details.
- Production scale: score billions of documents by batching on GPU with length-sorted batches (state the rule; no code run). Mention that the released classifier truncates to 512 tokens, as we do, so long documents are scored on their head.

#### Headline experiment

- **Hypothesis.** The removal rate of the combined filter varies across English-variety groups by more than sampling noise, and the classifier contributes to the gap beyond the heuristics. This is a claim to be tested.
- **Setup.** Audit stream: 20,000 documents of FineWeb `CC-MAIN-2025-05` (reduced config 5,000). Filters: heuristics only; classifier (`edu_score` ≥ 3) only; both. Groups by ccTLD (reference `.com`) and by host class. Threshold from 3.2.4. Our head versus the released classifier for the classifier-only arm.
- **Procedure.** Compute decisions; `audit_removals` with `cluster` = registered domain, `n_boot=2000`; classifier comparison on the 2,000 test labels with `d2l.paired_bootstrap` (n_boot = 10000).
- **Metrics.** Removal rate per group with interval; difference to reference with interval; test F1 for each classifier; Cohen's kappa between our classifier and the released classifier's decisions on the audit stream.
- **Plot (`fig_lm_quality_audit`).** Forest plot: y = group (with n), x = removal-rate difference to `.com` (percentage points) with 95% interval; three marker colors for heuristics, classifier, both. Left panel: the same for the six languages under heuristics only (x = removal rate).
- **Expected outcome.** Direction only: Gururangan et al. and Dodge et al. report that quality filters change group composition, so a gap is plausible; the magnitude is to be measured. For the classifier comparison, a larger frozen embedder may or may not beat the 109M-parameter encoder; the label-noise ceiling (D2) limits any difference, so the paired interval decides.
- **Noise check.** Cluster bootstrap by domain (documents from one site are correlated); groups with fewer than 100 documents or fewer than 10 domains are marked "too few" and excluded from conclusions; a second seed for the stream shuffle (seed 1) must give the same sign for every group reported.
- **T1 reduction.** `train_docs=5000`, `audit_docs=5000`, embedding batch 32. The full configuration (22,000 + 20,000 + c0) fits T1 as well; the reduction is for time.
- **Artifacts committed.** `lm-quality-head`, `lm-edu-annot-22k` (with ODC-By attribution), `lm-c1-docs` (flags manifest, no text), the audit table JSON.

#### Figures and tables

- `fig_lm_quality_pr_retention`: precision, recall and F1 against retained fraction for our head, the released classifier and the hashed-bag classifier on the test labels, with the D1 identity marked at two thresholds.
- `fig_lm_quality_audit`: forest plot described above.
- `fig_lm_quality_ppl_vs_score`: scatter of SmolLM2-135M NLL against `edu_score` with marginal histograms.
- `tab_lm_quality_rules`: rule families, sources, reason codes, c0 removal counts.
- `tab_lm_quality_clf`: F1 (with paired intervals), AUC, MAE, parameters and latency per document for each classifier.

#### Pitfalls and misconceptions

- Evaluating a filter by its classification F1 instead of by what it does to model training (3.4's proxy runs and `sec_scaling-laws` are the right yardstick; F1 is only a proxy).
- Word-based rules on unspaced scripts; treating a high removal rate on Thai as "low quality".
- Reading perplexity as quality; short repetitive text has very low perplexity.
- Selecting the threshold on the test labels; use a validation split of the train set (the notebook carves 2,000 rows of the 20,000).
- Treating LLM labels as ground truth: the annotator's rubric (grade-school and middle-school educational value, per the dataset card) defines "quality", including its prestige bias toward academic style; the model card warns it may overfit to academic-looking content.
- Evaluating a classifier on documents drawn after the filter (selection bias); the audit stream is drawn before our filters but after FineWeb's.
- Cluster correlation: 3,000 documents from one site are not 3,000 samples.

#### Summary (draft)

- Heuristic filters are cheap, interpretable and language-dependent; each rule needs a reason code so the funnel is attributable.
- A quality classifier distills an LLM's rubric into an embedding plus a linear head; F1 depends on the label noise, bounded by the closed-form ceiling in D2.
- F1 of a threshold is determined by the label prior, recall and retained fraction (D1), so the threshold is a token-budget decision.
- Perplexity filters measure resemblance to a reference corpus, not quality.
- Every filter shifts the corpus; audit removals by language, domain and variety with cluster-aware intervals before training.

#### Exercises

1. **Check.** The model card of the FineWeb-Edu classifier reports per-class F1 of 0.02 for score 5 even though binary F1 at threshold 3 is 0.82. Explain why both can be true and what each number says about using the classifier as a filter.
   *Sketch:* class 5 has 125 of 46,867 hold-out rows and its recall is near zero; threshold-3 pooling absorbs scores 3, 4, 5 so the binary decision only needs the 2/3 boundary right.
2. **Check.** Why is it wrong to report F1 at the threshold that maximizes F1 on the test labels? Name two fixes.
   *Sketch:* optimistic selection bias; choose on a validation split or by cross-validation, and report the test number at that fixed threshold.
3. **Derive.** From D1, show that at fixed TPR, F1 increases when retention ρ decreases toward π and decreases when ρ grows beyond π. For which FPR does ρ = π?
   *Sketch:* F1 = 2π·TPR/(π + ρ) is decreasing in ρ; ρ = π when (1 − π)FPR = π(1 − TPR), i.e. FPR = π(1 − TPR)/(1 − π).
4. **Derive.** Extend D2 to asymmetric noise: the annotator misses a positive with probability η₁ and invents one with probability η₀. Give the F1 ceiling of a perfect predictor of the true label, and find the η₀ at which the ceiling with η₁ = 0.02 equals the symmetric η = 0.05 value (π* = 0.0926).
   *Sketch:* precision = 1 − η₁; recall = π*(1 − η₁)/(π*(1 − η₁) + (1 − π*)η₀); F1 is their harmonic mean. Solving numerically gives η₀ ≈ 0.055 (bisection; the symmetric target is F1 = 0.779).
5. **Measure.** Predict, then measure, the F1 on the 2,000 test labels when the head is trained on 500, 2,000, 5,000 and 20,000 labels. Plot with a paired bootstrap interval at each size. Where does the curve flatten, and is that at the D2 ceiling?
6. **Measure.** Replace `Qwen3-Embedding-0.6B` by `Snowflake/snowflake-arctic-embed-m` as the frozen encoder with the same head and compare to the released classifier; predict whether they coincide.
7. **Open.** Design a quality classifier for code or for mathematics (positives from an annotator rubric you write, negatives from the same source), and an audit table for it. Report what the audit shows about the languages or topics that the classifier dislikes.

#### Further reading

- Rae et al., [Gopher](https://arxiv.org/abs/2112.11446): the quality and repetition rules (appendix) and their thresholds.
- Li et al., [DCLM](https://arxiv.org/abs/2406.11794): the fastText quality classifier and the finding that classifier choice dominates other filters in their testbed (read to confirm wording).
- Su et al., [Nemotron-CC](https://arxiv.org/abs/2412.02595): classifier ensembling and quality buckets.
- Penedo et al., [FineWeb](https://arxiv.org/abs/2406.17557) and the [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu); the [classifier model card](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) and the [annotations](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations).
- Wenzek et al., [CCNet](https://arxiv.org/abs/1911.00359): the perplexity-filter pipeline.
- Gururangan et al., [Whose Language Counts as High Quality?](https://arxiv.org/abs/2201.10474); Dodge et al., [Documenting Large Webtext Corpora](https://arxiv.org/abs/2104.08758); Longpre et al., [A Pretrainer's Guide to Training Data](https://arxiv.org/abs/2305.13169).
- Tutorials: the FineWeb blog post (filters and ablations); [fastText supervised tutorial](https://fasttext.cc/docs/en/supervised-tutorial.html).
- Code: [Datatrove filters](https://github.com/huggingface/datatrove), [fastText](https://github.com/facebookresearch/fastText), [Transformers `AutoModelForSequenceClassification`](https://huggingface.co/docs/transformers/index).

#### Risks and open decisions

- **Embedding throughput on the Ada GPU** for ≈ 47k documents may exceed the 15-minute budget. Fallback: `audit_docs=5000`, `max_len=256`, or `all-MiniLM-L6-v2` (22.7M parameters, Apache-2.0).
- **Qwen3-Embedding in Transformers**: version-sensitive (needs a recent `transformers`; left padding). Fallback: `sentence-transformers` loading of the same id, still pinned.
- **Datatrove filter parity** may fail on tokenization details. Fallback: report the disagreement list and use our implementation as the notebook's definition.
- **Audit validity.** ccTLD is a poor dialect proxy and the stream is post-FineWeb-filter. The notebook states both; the editor may substitute a dialect-labeled corpus if one with a clear license is found.
- **Human labels for the false-removal review** must be produced by the authors; the blueprint does not supply them. If unavailable, drop the human column and keep only the LLM-judge agreement with the released classifier.
- **Annotation license (decided 2026-10-29).** `HuggingFaceFW/fineweb-edu-llama3-annotations` is ODC-By (verified 2026-10-29); the labels were produced by Llama-3-70B-Instruct. The notebook and the dataset card record the dataset licence, the annotator model and the attribution; the head is a linear layer fitted only on these labels and ships under the same attribution.

### 3.3 Deduplication and decontamination

- **Label:** `sec_lm_dedup` · **Tier:** T0 (all CPU; semantic parts use `all-MiniLM-L6-v2` on CPU over a 4,000-document subset, with a config switch to Qwen3-Embedding-0.6B at T1) · **Budget:** T0 ≈ 10 min wall-clock on 8 cores, < 4 GiB RAM for 20,000 documents (to be measured; shingling and signatures dominate) · **Codex units:** 1.5 (dedup half), 15.7 (contamination and memorization)
- **Builds on (book):** `sec_prob` (independent events, binomial counts, the product rule used in the S-curve derivation); `sec_mdl-statistics` (confidence intervals for binomial proportions); `sec_text-sequence` (a "token" for 13-gram overlap is a word or a BPE token; we use words). Not re-taught: Jaccard on sets of boxes (`chapter_computer-vision/anchor.md` defines IoU for areas; one sentence of contrast).
- **Builds on (this part):** 3.1 (`d2l.Doc`, `d2l.CorpusManifest`, `d2l.docs_hash`), 3.2 (`lm-c1-docs`, `d2l.embed_docs` for the semantic step at T1; the 20,000-document audit stream becomes this section's slice), 2.3 (what an embedding model gives for semantic similarity; one sentence). `assumed:` none.
- **Exports** (`#@save`, new):
  - `d2l.shingles(text, n=5)`: list of word n-gram strings of the NFC-lowercased `\w+` tokens; empty if fewer than `n` words.
  - `d2l.minhash_signature(shingle_list, k=128, seed=0xD2)`: `np.uint64` array `(k,)` following the A3 specification (blake2b-8 shingle hash, `(a·x + c) mod (2^61 − 1)` per row).
  - `d2l.lsh_candidates(sigs, b, r)`: set of `(i, j)` candidate index pairs, `i < j`, from band-bucket collisions (buckets with more than `max_bucket` members are subsampled and reported).
  - `d2l.union_find_clusters(n, pairs)`: component labels.
  - `d2l.near_dedup(texts, threshold=0.8, ngram=5, num_perm=128, seed=0) -> list[int]`: indices to keep. Convenience wrapper over `shingles`, `minhash_signature`, `lsh_candidates` and `union_find_clusters` (it picks `(b, r)` with `best_br`); the representative of each cluster is its lowest index. Test: on 300 documents with planted near-duplicates (seed 0), `near_dedup` returns exactly one index per planted cluster, namely the lowest, and returns `list(range(n))` when `threshold=1.0` on distinct texts. Chapter 7.3 uses it.
  - `d2l.jaccard_exact(a, b)`: exact Jaccard of two shingle sets.
  - `d2l.scurve(s, b, r)`: `1 - (1 - s**r)**b`.
  - `d2l.best_br(k, tau, w_fp=1.0, w_fn=1.0)`: the `(b, r)` with `b·r ≤ k` that minimizes the weighted error of D3.
  - `d2l.suffix_array(ids)`, `d2l.lcp_array(ids, sa)`, `d2l.repeated_spans(ids, sa, lcp, min_len)`: prefix-doubling suffix array over an integer array, Kasai LCP, and the maximal repeated spans of length ≥ `min_len`.
  - `d2l.NgramIndex(n=13)`: `.add(text, tag)`, `.query(text) -> [(tag, n_hits)]`; hashed word n-gram index for decontamination.
  - `d2l.make_dup_pairs(docs, spec)`: deterministic generator of the labeled pair set (spec below).
- **Data artifacts:**
  - `lm-dedup-pairs` (JSONL; one row per pair): `pair_id`, `a` (doc id), `b` (doc id or `null`), `op` in `{sub, del, boiler, shuffle, trunc, neg_host, neg_random}`, `params` (rate, seed), `J` (exact 5-gram word Jaccard of the final pair), `C` (containment of the smaller in the larger), `label` = `J ≥ 0.8`, `split` in {train, test} (by `sha256(base_id) % 2`). Design: 2,000 base documents (seed 0), three perturbed variants each (6,000 pairs; edit rates ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.3} for `sub`/`del`; boilerplate blocks of 20–200 words; paragraph shuffles; truncations to 30–90%), plus 6,000 negatives (3,000 different pages from the same host, which share templates; 3,000 random pairs). Texts are regenerated from `(a, op, params)`, the c0 documents rebuilt by `d2l.rebuild_c0` (with the `lm-c1-docs` flags) and the 20k FineWeb stream (ODC-By), so the file stays small and holds no crawled text. Exact counts per op are recorded in the manifest.
  - `lm-contam-plant` (JSONL): 100 items from each of GSM8K test (1,319 rows, MIT), ARC-Challenge test (1,172 rows, CC-BY-SA-4.0) and MMLU test (14,042 rows in config `all`, MIT), each planted in three forms (`verbatim`, `reformat` = casing, whitespace, punctuation and choice-order changes, `paraphrase` = Qwen3.5-4B (non-thinking) paraphrase generated once at authoring with the prompt and revision stored in the row): 900 plants into 900 distinct documents of the 20k slice, plus 1,000 untouched control documents. Fields: `doc_id`, `item_id`, `benchmark`, `form`, `char_start`, `char_end`, `source_hash`.
  - `lm-c2-docs` (c1 after `minhash_dedup` and `decontam`: keep/drop lists with reason codes keyed by c0 document id, no text), `lm-corpus-manifest-v1` (stages `extract`, `lang_id`, `nfc`, `pii`, `heuristics`, `edu_score`, `exact_dedup`, `minhash_dedup`, `decontam`).
- **Models and data:** the 20,000-document FineWeb stream from 3.2 (`HuggingFaceFW/fineweb`, config `CC-MAIN-2025-05`, ODC-By); `openai/gsm8k` (config `main`, MIT), `allenai/ai2_arc` (config `ARC-Challenge`, CC-BY-SA-4.0), `cais/mmlu` (config `all`, MIT); `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 22,713,728 parameters); `Qwen/Qwen3.5-4B` (`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Apache-2.0, 4,659,865,088 parameters including the vision tower; loaded text-only, non-thinking) for authoring the paraphrases only; detection is by n-grams and embeddings, so no LLM grades its own paraphrases.
- **Packages:** `numpy`, `datasketch>=1.6` (oracle; latest 2.0.0), `sentence-transformers`, `scikit-learn` (k-means; or a 20-line NumPy k-means), `pydivsufsort` (optional oracle; 0.0.20), `datasets`.

#### Narrative arc

1. **The question.** Opening cell: pick two c1 pages from the same host that differ only in a date and a sidebar. Print their exact 5-gram Jaccard (expected high), then time the all-pairs Jaccard on 1,000 documents and extrapolate to 20,000 and to 10^9 documents with the quadratic formula (printed as a measured-then-extrapolated number). Symptom: exact comparison is quadratic and hopeless at corpus scale, and exact hashing misses these pairs.
2. **3.3.1 Exact deduplication.**
   - Document hash (SHA-256 of normalized text) and line or paragraph hashes; the boilerplate problem (cookie banners, navigation) is a *span* problem, not a document problem; C4's rule of removing all but one of each repeated 3-sentence span (the FineWeb paper lists it among the C4 filters it did not use; cite the C4 description via RefinedWeb or the T5 paper at authoring).
   - Cell: exact-hash dedup on the 20k slice and on c1; count; show five pairs of "different documents, same boilerplate paragraph".
3. **3.3.2 Similarity as set overlap.**
   - Shingling (word 5-grams as in FineWeb); resemblance J and containment C (Broder 1997). Why J misses a truncated copy: J = |A|/|B| when A ⊂ B.
   - Cell: shingle statistics on the slice: shingles per document, memory of all shingle sets (a number to be measured), and the distribution of exact J for 5,000 random pairs and for 5,000 same-host pairs (random pairs concentrate near 0; same-host pairs have a heavy tail from templates). `fig_lm_dedup_jaccard_hist`.
4. **3.3.3 MinHash.**
   - The signature, the estimator Ĵ, its variance J(1 − J)/k (D1). Hash functions instead of permutations: universal-hash approximation and what breaks (small sets, bias with weak hash families); the exact hash specification above.
   - Cells: signatures for the slice; the estimation error of Ĵ against exact J on the labeled pairs, binned; check the 95% half-width at J = 0.8, k = 128 equals 1.96·sqrt(0.8·0.2/128) = 0.069 (closed form).
5. **3.3.4 LSH banding and the S-curve.**
   - Bands of r rows, b bands, `P(candidate) = 1 − (1 − s^r)^b` (D2). Threshold intuition, steepest point (D2), FineWeb's choice (112 hashes as 14 bands of 8, matched probabilities 56%, 77%, 92% and 98.8% at s = 0.7, 0.75, 0.8, 0.85 as the paper reports; D2's check reproduces them) and RefinedWeb's (9,000 hashes as 450 bands of 20, from FineWeb's appendix figure).
   - Tuning under a signature budget (D3) and the false-positive cost at scale (D4): `fig_lm_dedup_scurve` with theory curves for (14, 8), (9, 13), (20, 5), (8, 16) and (450, 20).
   - Cell: build candidates for the slice at the chosen (b, r); count candidate pairs by bucket size; show the largest buckets (templated pages) and the `max_bucket` safeguard.
6. **3.3.5 From pairs to a deduplicated corpus.**
   - Clusters by union-find (transitivity: A~B and B~C without A~C; the FineWeb paper describes this and keeps one randomly chosen document per cluster; we keep the lowest id for determinism). Which document to keep: first seen, highest quality score (`edu_score` from 3.2), longest; effect on the manifest.
   - Per-dump versus global: the FineWeb paper reports that global MinHash deduplication across dumps gave only a modest uplift, with the retained older-dump data of lower quality; it deduplicates each dump on its own (read §3.4 and Fig. 3 at authoring). The consequence for a teaching pipeline: our corpus is one dump.
   - Cell: apply the pipeline to c1, report the funnel and the five largest clusters.
7. **3.3.6 Substring deduplication with a suffix array.**
   - Lee et al.'s ExactSubstr: find repeated substrings of at least 50 tokens across the corpus and remove all but one copy; why document-level MinHash cannot see a 60-token license block shared by thousands of otherwise different documents.
   - Cells: build a suffix array by prefix doubling over a 2 MB sample (bytes, or BPE ids from `d2l.BPETokenizer`), Kasai LCP, extract spans of length ≥ 50 tokens; print the three longest repeated spans, the covered-token fraction, and the overlap with documents MinHash flagged. Scale statement: the Google implementation is a Rust suffix-array pipeline that needs memory proportional to the corpus size; ours is O(n log² n) NumPy.
8. **3.3.7 Semantic deduplication.**
   - SemDeDup: embed, cluster with k-means, drop items whose cosine similarity within a cluster exceeds 1 − ε. It removes near-paraphrases that share few n-grams; its cost is O(n²/k) per run. Cell: 4,000-document subset embedded with MiniLM (T0; Qwen3-Embedding at T1), k = 64, ε swept; show pairs removed at several ε and how many of them MinHash had already flagged; show that it also removes "same topic, different page" at large ε (`fig_lm_dedup_semantic_eps`).
9. **3.3.8 Decontamination.**
   - What it is: removing training text that overlaps evaluation items. GPT-3 used 13-gram overlap (Brown et al.; confirm the exact rule in the paper's contamination appendix); we implement it as `NgramIndex(n=13)`. Detection by form: verbatim plants (recall expected 1 for items of ≥ 13 words), reformatted plants (casing and punctuation normalized in the index key), paraphrase plants (n-gram overlap fails by construction; Yang et al. show rephrased samples evade it and can still inflate scores).
   - Cells: index the full test sets (1,319 + 1,172 + 14,042 items), scan the 20k slice, report recall by benchmark × form and the false-positive rate on the 1,000 controls; list the items shorter than 13 words that cannot be matched at n = 13, per benchmark. A second detector: top-1 cosine similarity to any benchmark item with MiniLM embeddings; threshold chosen on the controls (false-positive budget 1%); recall on paraphrases.
   - Interpretation rules: overlap is evidence of leakage, not of memorization; a model's score drop after decontamination measures both.
10. **3.3.9 Duplication and memorization.**
    - Lee et al. link duplicated substrings to emitted verbatim text; Carlini et al. (2202.07646) show the extractable fraction grows with model size, with the duplication count of a sequence and with prompt length. State the direction, not numbers, in the notebook; cite magnitudes from the papers at authoring.
    - No training in this notebook; the measure exercise below trains proxies from 3.4 on a corpus with planted duplicates.
11. **3.3.10 The v1 manifest.** Run stages 3.1–3.3 end to end from `lm-c0-docs`, with 1 and with 4 workers, and assert identical `manifest.sha256()`. Print the funnel (`fig_lm_dedup_funnel`: counts and tokens after each stage) and the rendered dataset card from `to_card()`.

#### Derivations

**D1. MinHash collision probability equals Jaccard similarity.**

- *Statement.* Let π be a uniformly random permutation of a finite universe U and h(S) = min_{x∈S} π(x). For finite non-empty sets A, B ⊆ U, P[h(A) = h(B)] = |A ∩ B|/|A ∪ B| = J(A, B). The estimator Ĵ = (1/k)·Σ_i 1[h_i(A) = h_i(B)] with independent permutations has mean J and variance J(1 − J)/k.
- *Assumptions.* Independent uniform permutations; the real implementation replaces them with hash functions whose bias is negligible for the set sizes used.
- *Steps.* (1) Consider the union A ∪ B. The element of A ∪ B with the smallest π-value, call it x*, is uniformly distributed over A ∪ B by symmetry of π. (2) h(A) = h(B) holds exactly when the minimum over A and the minimum over B are attained at the same element. (3) The minimum of A ∪ B is the smaller of h(A) and h(B), so x* is the argmin of A or of B. If x* ∈ A ∩ B it is the argmin of both and h(A) = h(B). If x* ∈ A \ B, then x* is the argmin of A but not of B, so h(A) ≠ h(B) (B's argmin has a larger value). Symmetric for B \ A. (4) Therefore P[h(A) = h(B)] = P[x* ∈ A ∩ B] = |A ∩ B|/|A ∪ B|. (5) Each indicator is Bernoulli(J) and independent across i, so Ĵ has mean J and variance J(1 − J)/k; by the normal approximation, the 95% half-width is 1.96·sqrt(J(1 − J)/k).
- *Numerical check.* Cell: 200 random set pairs with known J ∈ {0.1, …, 0.9} (seed 0), k = 128: print the mean of Ĵ − J (expected 0 within 3 standard errors) and the empirical variance against J(1 − J)/k per bin; also compare `d2l.minhash_signature` with `datasketch.MinHash(num_perm=128)` on the same sets: both within 3σ of the exact J.

**D2. The LSH S-curve, its threshold and its steepest point.**

- *Statement.* Split a k = b·r signature into b bands of r rows; a pair is a candidate if at least one band matches in all r rows. If the pair's similarity is s, P(candidate) = 1 − (1 − s^r)^b. The curve is steepest at s* = ((r − 1)/(b·r − 1))^{1/r}, which is close to (1/b)^{1/r}.
- *Assumptions.* Rows are independent Bernoulli(s) matches (D1); bands use disjoint rows.
- *Steps.* (1) A band matches with probability s^r (r independent row matches). (2) No band matches with probability (1 − s^r)^b since bands are independent. (3) The complement gives P. (4) The derivative is P′(s) = b·r·s^{r−1}(1 − s^r)^{b−1} ≥ 0. (5) Its stationary point satisfies (r − 1)/s = (b − 1)·r·s^{r−1}/(1 − s^r); multiply out to (r − 1)(1 − s^r) = (b − 1)·r·s^r, hence s^r = (r − 1)/(b·r − 1). (6) For large b·r this is ≈ 1/b, so s* ≈ (1/b)^{1/r}.
- *Worked values.* (b, r) = (14, 8): s* = (7/111)^{1/8} ≈ 0.708 versus (1/14)^{1/8} ≈ 0.719; P(0.5) = 0.0533, P(0.7) = 0.5645, P(0.75) = 0.7716, P(0.8) = 0.9235, P(0.85) = 0.9884, which reproduces the 56%, 77%, 92% and 98.8% of the FineWeb paper. (9, 13): P(0.8) = 0.3988, P(0.85) = 0.6864.
- *Numerical check.* Assert `scurve(s, 14, 8)` matches the paper's four values after rounding; Monte Carlo with random sets at J ∈ {0.5, 0.7, 0.8, 0.9} (2,000 pairs each, seeds 0–2) through the full `minhash_signature` and `lsh_candidates` pipeline: the empirical rate is within 3 binomial standard errors of the curve for each (b, r) in the grid.

**D3. Choosing (b, r) for a budget k and a threshold τ.**

- *Statement.* For a signature budget k and a target threshold τ, choose (b, r) with b·r ≤ k to minimize E(b, r) = w_fp·∫₀^τ P(s) ds + w_fn·∫_τ^1 (1 − P(s)) ds. The two weights encode the relative cost of keeping a duplicate and of removing a non-duplicate. The integrals are the areas of the false-positive and false-negative regions of the S-curve against an ideal step at τ.
- *Assumptions.* Pair similarities are uniform on [0, 1] (a modelling choice; replace the measure by the empirical distribution of 5-gram Jaccard from the slice to tune on real data).
- *Steps.* (1) The ideal decision is the step function 1[s ≥ τ]. (2) The loss against it integrates the false-positive area on [0, τ] and the false-negative area on [τ, 1]. (3) Enumerate r from 1 to k, set b = ⌊k/r⌋, evaluate both integrals by the trapezoid rule on 20,001 points, and take the minimizer.
- *Result for k = 128, τ = 0.8, w_fp = w_fn = 1 (computed in the blueprint).* r = 13, b = 9 gives E = 0.0586; r = 14, b = 9 gives 0.0588; r = 12, b = 10 gives 0.0600; r = 8, b = 16 gives 0.1395. With equal weights the optimum puts only P(0.8) = 0.399 at the threshold; if missing duplicates is the costlier error, raise w_fn (the cell sweeps w_fn ∈ {1, 2, 5, 10}).
- *Numerical check.* Assert that `best_br(128, 0.8)` returns (9, 13) and that a brute-force search over a 2-D grid of (b, r) with b·r ≤ 128 finds the same minimum value within 1e-9. Compare to `datasketch.MinHashLSH(threshold=0.8, num_perm=128).b, .r` with equal weights (datasketch's optimizer uses the same objective; report both pairs and any difference).

**D4. Candidate-pair cost at scale.**

- *Statement.* With N documents and a density of pair similarities g(s) over pairs, the expected number of candidate pairs is (N²/2)·∫₀¹ g(s)·P(s) ds. The low-similarity bulk contributes through P(s) at small s, so P(0.3) and P(0.5) matter more than P near the threshold.
- *Assumptions.* Pairwise independence of bucket collisions at the level of expectation (true by linearity).
- *Steps.* (1) Let X_ij indicate that pair (i, j) is a candidate; E[X_ij] = P(s_ij). (2) Sum over pairs; linearity of expectation. (3) Substitute g.
- *Numerical check.* For (14, 8): P(0.3) = 9.2e-4 (1 − (1 − 0.3^8)^14), P(0.5) = 0.0533; with N = 10^6 and 0.1% of pairs at s = 0.5, the expected candidate count is 5·10^11 × 10^-3 × 0.0533 ≈ 2.7·10^7. Print this and compare with the observed candidate count on a 20k slice scaled by (N/20,000)², flagging that real similarity densities are not uniform.

#### From-scratch implementation

Budget ≈ 150 lines.

| Component | Signature | ≈ lines | Test (reference, tolerance, seed) |
|---|---|---|---|
| Shingles, signature | `shingles`, `minhash_signature` (blake2b-8, mulmod with Python ints or `np.uint64` with the `2^61 − 1` trick) | 30 | Bit-exact against the committed fixture array (A3 check); D1 cell vs `datasketch` within 3σ |
| LSH + union-find | `lsh_candidates`, `union_find_clusters` (path compression, union by size) | 35 | Candidate set equals a brute-force enumeration of band collisions on 300 signatures (exact); clusters equal `scipy.sparse.csgraph.connected_components` labels up to renaming |
| Suffix array + LCP | `suffix_array` (prefix doubling with `np.lexsort`), `lcp_array` (Kasai), `repeated_spans` | 50 | Equal to `sorted(range(n), key=lambda i: s[i:])` and to the naive LCP for 20 random byte strings of length ≤ 2,000 over an alphabet of 3 (seed 0); equal to `pydivsufsort.divsufsort` if installed |
| N-gram index | `NgramIndex(n=13)`: Python `hash` of the tuple of word hashes replaced by `blake2b-8` for determinism | 20 | Finds a planted 13-word span inserted at 10 random offsets in 10 random documents (exact); returns nothing for a 12-word span |
| Pair generator | `make_dup_pairs` | 15 (plus the spec) | Regeneration reproduces the stored `J` to 1e-12 for all rows |

#### In production

- `datasketch.MinHash` and `MinHashLSH` for the oracle of signatures and of (b, r) optimization (D3).
- Datatrove's MinHash pipeline: four stages (signatures, bucket matching, clustering, filtering); the notebook shows the four stage names and the mapping onto our functions, with parameters `n_grams=5, num_buckets=14, hashes_per_bucket=8` as in the FineWeb setup; it is not run.
- `google-research/deduplicate-text-datasets` for substring deduplication at scale and `allenai/dolma` for Bloom-filter deduplication (cite as pointers); `pydivsufsort` as the suffix-array oracle.
- FineWeb-2 ships `minhash_cluster_size` per document; the notebook streams 2,000 `deu_Latn` rows and compares our cluster sizes with the shipped column on the subset it covers, as a read-only cross-check (differences expected because FineWeb-2's hash settings differ; cite its pipeline paper, 2506.20920).

#### Headline experiment

- **Hypothesis.** For every (b, r) the empirical candidate rate of the pipeline as a function of exact Jaccard follows `1 − (1 − s^r)^b` within binomial error, and the best (b, r) on held-out labeled pairs under the chosen costs is the D3 optimum or an adjacent point.
- **Setup.** `lm-dedup-pairs` (6,000 perturbed pairs and 6,000 negatives; train/test split by base id). k = 128. Grid: r ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16} with b = ⌊128/r⌋. Signature seeds 0, 1, 2.
- **Procedure.** Compute signatures for every distinct document in the pairs; for each (b, r) evaluate the candidate decision for all pairs; bin by exact J in 0.05-wide bins; compute precision, recall, F1 against `label` (J ≥ 0.8) on the train half to choose (b, r) under w_fn ∈ {1, 5}, then report the same metrics on the test half. Compute the same for FineWeb's (14, 8) with k = 112 and for datasketch's choice.
- **Metrics.** Candidate rate per J bin with binomial 95% intervals; F1, precision, recall; number of candidate pairs per 1,000 documents on the 20k slice; wall-clock.
- **Plot (`fig_lm_dedup_scurve`).** x = exact Jaccard (0–1), y = P(candidate); lines = theory for the five (b, r), markers with intervals = empirical bins for the chosen two; dashed vertical line at τ = 0.8. Second panel (`fig_lm_dedup_f1_grid`): F1 against r for w_fn = 1 and 5, train and test.
- **Expected outcome.** The match with the analytic curve is expected by D1/D2 (assumption: independent rows); departures indicate a flawed hash. The best r is expected to move to smaller values when w_fn grows (D3 reading); magnitudes to be measured.
- **Noise check.** Three hash seeds (0, 1, 2); the empirical-vs-theory comparison uses binomial intervals; F1 differences between adjacent r are reported with a paired bootstrap over the test pairs (`d2l.paired_bootstrap`, clustering by base document) and not interpreted when the interval contains 0.
- **T1 reduction.** None (T0). Scale by `n_docs` for the slice, `k` for the signature.
- **Second table (`tab_lm_dedup_decontam`).** Recall by benchmark × form × detector (13-gram, MiniLM top-1) with false-positive rates on the 1,000 controls; items shorter than 13 words listed separately. No interval claims beyond binomial.
- **Artifacts committed.** `lm-dedup-pairs`, `lm-contam-plant`, `lm-c2-docs`, `lm-corpus-manifest-v1`, the S-curve data JSON.

#### Figures and tables

- `fig_lm_dedup_jaccard_hist`: histogram of exact Jaccard for random versus same-host pairs, log y.
- `fig_lm_dedup_scurve`: described above; `fig_lm_dedup_f1_grid`.
- `fig_lm_dedup_semantic_eps`: removed fraction against ε with the MinHash overlap stacked.
- `fig_lm_dedup_funnel`: documents and tokens after each stage of the pipeline.
- `tab_lm_dedup_methods`: exact hash, MinHash/LSH, suffix array, semantic: unit of match, what it misses, cost, our measured throughput.
- `tab_lm_dedup_decontam`: described above.

#### Pitfalls and misconceptions

- Treating Jaccard as symmetric in what you care about: a short document copied inside a long one has small J and containment 1.
- Using Python's built-in `hash` for shingles (salted per process): signatures change across runs and break the manifest hash.
- Band count too high for the pair density: candidate explosion and giant buckets (templated pages); set `max_bucket` and report it.
- Keeping "one random document per cluster" without a seed.
- Deduplicating after the final tokenization or after splitting into train and validation: validation leakage.
- Assuming n-gram decontamination catches paraphrase, translation or a changed number; assuming 13-grams cover short items.
- Removing every document that overlaps a benchmark when the overlap is a quotation of a famous passage; look at the span, not only the count.
- Deduplicating across dumps by default: the FineWeb paper's per-dump versus global comparison.

#### Summary (draft)

- Exact hashes find identical documents; near-duplicates need a similarity estimate, and MinHash gives an unbiased estimate of Jaccard with variance J(1 − J)/k.
- LSH turns the estimate into a candidate rule with S-curve 1 − (1 − s^r)^b; the budget k = b·r is split to place the steepest point near the threshold, and the cost of the low-similarity bulk decides whether the design scales.
- Substring and semantic methods catch what document-level MinHash cannot (shared spans; paraphrases), at different costs.
- Decontamination by 13-gram overlap catches verbatim and lightly reformatted benchmark text, but not paraphrases or short items; report recall by form.
- Duplicates drive memorization; deduplication is both a quality and a privacy measure.

#### Exercises

1. **Check.** Two documents have 1,000 and 100 five-gram shingles; the smaller is fully contained in the larger. What are J and C? Will MinHash with threshold 0.8 flag them, and what sketch would?
   *Sketch:* J = 100/1000 = 0.1, C = 1; no; a containment estimator such as a bottom-k sketch or MinHash on the smaller set queried against the larger's signature set.
2. **Check.** A teammate computes shingle hashes with Python's `hash()`. List two ways this breaks the pipeline.
   *Sketch:* string hashing is salted by `PYTHONHASHSEED`, so signatures differ across processes and machines; the manifest hash and cached signatures stop matching.
3. **Derive.** Show that for r = 1 the S-curve is 1 − (1 − s)^b and for b = 1 it is s^r. Among (b, r) with b·r = 128 and b a power of two, which has (1/b)^{1/r} closest to 0.8?
   *Sketch:* substitute. Thresholds: (8, 16) gives 0.878, (16, 8) gives 0.707, (32, 4) gives 0.420; (8, 16) is nearest (0.078 away versus 0.093). Compare with D3's optimum (9, 13), whose own s* is ((12)/(116))^{1/13} ≈ 0.84.
4. **Derive.** Show that P(candidate) ≤ b·s^r (union bound over bands) and that the bound is tight when b·s^r ≪ 1. Use it to estimate the expected candidate count of D4 for s = 0.3 and (b, r) = (14, 8), and to explain why lowering r by one while keeping b multiplies the low-similarity candidate load by about 1/s.
   *Sketch:* P(∪ A_j) ≤ Σ P(A_j) = b·s^r; inclusion–exclusion corrections are O((b·s^r)²). For s = 0.3 the bound is 14 × 6.56e-5 = 9.2e-4, matching the exact 9.2e-4. Reducing r by one multiplies s^r by 1/s.
5. **Measure.** Predict, then measure, the number of candidate pairs and the F1 on the test half as k ∈ {64, 128, 256} with the best (b, r) from D3 at each. Report F1 with a paired bootstrap interval.
6. **Measure.** Train a proxy model (using 3.4's `d2l.train_proxy_lm`) on 20M tokens of the slice where a random 5% of documents are duplicated {1, 10, 100} times, and measure the NLL of the duplicated documents on a held-out copy of their prefixes versus the NLL on unduplicated ones. Predict the direction; compare to Lee et al. and Carlini et al.
7. **Open.** Build a detector for paraphrased benchmark items that keeps the false-positive rate on the 1,000 controls below 1% and recovers at least 80% of the paraphrase plants (your threshold, your justification), then break it by translating the plants into German with a local model.

#### Further reading

- Broder, "On the resemblance and containment of documents" (1997; by title, no stable open landing page) and [Mining of Massive Datasets, ch. 3](http://www.mmds.org/): MinHash, LSH, the S-curve and optimization of bands.
- Lee et al., [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499): NearDup and ExactSubstr, effects on memorization and perplexity.
- Abbas et al., [SemDeDup](https://arxiv.org/abs/2303.09540); Tirumala et al., [D4](https://arxiv.org/abs/2308.12284): semantic deduplication and diversification.
- Carlini et al., [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) and [Quantifying Memorization Across Neural Language Models](https://arxiv.org/abs/2202.07646).
- Yang et al., [Rethinking Benchmark and Contamination for Language Models with Rephrased Samples](https://arxiv.org/abs/2311.04850); Brown et al., [GPT-3](https://arxiv.org/abs/2005.14165) (contamination appendix); Mirzadeh et al., [GSM-Symbolic](https://arxiv.org/abs/2410.05229) and [LiveCodeBench](https://livecodebench.github.io/) for perturbation and temporal splits.
- Tutorials: the FineWeb blog post (deduplication ablations); the Datatrove MinHash [examples](https://github.com/huggingface/datatrove/tree/main/examples).
- Code: [datasketch](https://github.com/ekzhu/datasketch), [datatrove](https://github.com/huggingface/datatrove), [deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets), [dolma](https://github.com/allenai/dolma).

#### Risks and open decisions

- **Runtime** of shingling 20,000 documents in pure Python may exceed the T0 budget. Fallback: `n_docs=10000`, or vectorize the signature loop with NumPy over all shingles of a document at once; the A3 autograder uses a 2,000-document fixture.
- **uint64 modular arithmetic** overflows in NumPy for `a·x mod (2^61 − 1)`. Implementation options: Python ints (slow but exact), or split multiplication into 32-bit halves. The blueprint fixes the semantics, not the speed trick; the bit-exact test enforces equality with the Python-int version.
- **Paraphrase plants** are generated once with a local model and committed; if the model revision changes, `lm-contam-plant` must be regenerated and the A3 autograder constants updated.
- **Benchmark licenses.** GSM8K and MMLU are MIT; ARC is CC-BY-SA-4.0, which propagates a share-alike term to the planted file; keep the plants in a separate artifact with its license stated.
- **Natural near-duplicate labels** do not exist in the stream because FineWeb has already deduplicated per dump; the labeled pairs are synthetic by construction and the notebook says so. Natural pairs in c1 are inspected, not scored.
- **MiniLM on CPU** over 4,000 documents may exceed the budget; fallback: 2,000 documents and `max_len=128`.
- **Suffix array memory**: 2 MB bytes means n log² n NumPy work with int64 arrays of ≈ 16 MB; fine. If the editor wants BPE ids, cap at 1M ids.

### 3.4 Data mixtures and curricula

- **Label:** `sec_lm_data_mixtures` · **Tier:** T2 (reduced T1: `config='t1'`, same code) · **Budget:** T2 ≤ ~3 h on 1× A100 40 GB for 36 proxy runs (sizing assumption: 15% model-FLOPs utilization on an 8.1M-parameter model, so ≈ 140 s per run, ≈ 1.4 h; to be measured; peak memory ≤ 10 GB expected, to be measured; it fits the 40 GB card with room for two runs at once); T1 ≤ ~15 min and ≤ 15 GiB for 25 runs (≈ 25–35 s each, to be measured) · **Codex units:** 1.6
- **Builds on (book):** `sec_scaling-laws` (6ND to size runs; the repeated-data remark; the miniature-study method); `sec_scheduler` (WSD: the stable-then-decay shape that annealing uses); `sec_gpt` (`d2l.GPT` as the proxy); `sec_adamw`; `sec_mdl-statistics` (noise floors). Not re-taught: scaling-law fitting, WSD derivation.
- **Builds on (this part):**
  - 3.6 (forward reference): the frozen R2 tokenizer `lm-r2-tokenizer` and the flat `uint16` token-file format; 3.4 only *loads* the pools.
  - 3.2: `edu_score` bands are one way to define the high-quality annealing source (mentioned, not required).
  - 1.5: `d2l.paired_bootstrap` (used for the regression metrics; seed-paired arm comparisons use raw differences because n = 3).
  - 1.2: `d2l.GPT`. 1.4: `d2l.bits_per_byte` (imported, not redefined). `d2l.load_hf` is not needed. No other assumed names.
- **Exports** (`#@save`, new; 3.5 and 3.6 import them):
  - `d2l.ProxyCfg`: dataclass `(num_hiddens, num_heads, num_blks, seq_len=1024, batch=64, lr, warmup_frac=0.02, schedule='cosine'|'wsd', decay_frac=0.2, final_lr_frac=0.1, weight_decay=0.1, dtype='bfloat16')`.
  - `d2l.train_proxy_lm(pools, weights, tokens, cfg, seed=0, eval_sets=None, stages=None, log_every=50)`: `pools` is `{domain: np.memmap uint16}`; `weights` is `{domain: float}` summing to 1; `stages` is an optional list of `(fraction_of_tokens, weights)` that overrides `weights` per stage; returns `dict(log=list of (step, tokens, train_loss), final={domain: {'nll': float, 'bpb': float}}, throughput=tokens/s, n_params=int)`. Contract: the data stream depends only on `(seed, weights, tokens)`.
  - `d2l.bpb_from_nll(sum_nll_nats, n_bytes)`: `sum_nll_nats/(n_bytes·ln 2)`. `train_proxy_lm` reports per-domain nll and bytes, not a model and texts, so 3.4 converts with this helper. It does not redefine `d2l.bits_per_byte`, which 1.4 owns with signature `(model, tokenizer, texts, ctx_len=2048, stride=None, batch_size=4) -> float`.
  - `d2l.sample_mixtures(n, domains, concentration, seed, max_weight=0.9, min_weight=0.0)`: weights drawn from a Dirichlet.
  - `d2l.temperature_weights(sizes, alpha)`: `p_i ∝ n_i^α`.
  - `d2l.fit_mixture_regression(W, L, kind='ridge'|'gbm'|'mixlaw', **kw)`: returns a predictor `f(w) -> per-domain loss vector`; `'mixlaw'` fits the exponential mixing law per domain by least squares.
  - `d2l.doremi_weights(ref_nll, domains_batches_fn, model, steps, eta=1.0, smooth=1e-3)`: runs group-DRO on the proxy and returns the averaged domain weights (miniature DoReMi).
  - `d2l.effective_tokens(unique, epochs, r_star=15.4)`: `U + U·R*·(1 − exp(−(E − 1)/R*))`.
- **Data artifacts** (flat little-endian `uint16` `.bin` files plus a `meta.json` with `{n_tokens, n_bytes_nfc, n_docs, tokenizer_sha256, source, revision, bucket_rule}`; all use the R2 tokenizer, so the vocabulary is 32,768):
  - Training pools `lm-mix-web-150m`, `lm-mix-code-150m`, `lm-mix-math-150m` (T2: 150M tokens each = 300 MB each) and `lm-mix-web-20m`, `lm-mix-code-20m`, `lm-mix-math-20m` (T1: 20M tokens each = 40 MB each). Pool size ≥ the largest weight × the run budget (0.9 × 2^27 = 121M tokens at T2; 0.9 × 2^24 = 15M at T1), so no run repeats a token.
  - Validation `lm-mix-val-2m` (T2) and `lm-mix-val-1m` (T1): 2M and 1M tokens per domain, from documents disjoint from the pools.
  - `lm-mix-runs-t2`: JSON with, for each of the 36 T2 runs, `(run_id, arm, weights, seed, tokens, steps, per-domain nll and bpb, throughput, n_params, log)`; analysis cells run on it.
- **Models and data:**
  - Web: `HuggingFaceFW/fineweb-edu` config `sample-10BT` (ODC-By; 9,672,101 rows). Pool = documents with `bucket ∈ {1..7}`, validation = `bucket 8` (the chapter's bucket rule), stream order, NFC.
  - Code: `codeparrot/codeparrot-clean` (Python files from GitHub; the card describes exact-match deduplication and filters on line length and alphanumeric fraction; per-file `license` column; no dataset-level license in the metadata). Keep files whose `license` is one of `mit, apache-2.0, bsd-2-clause, bsd-3-clause, isc`; shuffle with a seeded buffer; take 150M tokens for the pool and 2M for validation from disjoint `hash(path) % 64` buckets. The Stack v2 (`bigcode/the-stack-v2`, gated; file contents live in Software Heritage, not on the Hub) and Stack-Edu / `smollm-corpus` `python-edu` (blob ids and scores only, no content: verified columns `blob_id, repo_name, path, length_bytes, score, int_score`) are the production references and cannot be fetched by a notebook without credentials.
  - Math: `HuggingFaceTB/finemath` config `finemath-4plus` (ODC-By; 6,699,493 rows; columns include `text, url, token_count, score, int_score`), first documents in a seeded-shuffle buffer.
  - Proxy model: `d2l.GPT(vocab_size=32768, num_hiddens=192, num_heads=3, num_blks=4, max_len=1024, pos='rope', norm='rms', act='swiglu')` (T2) with about 8.1M parameters if the block uses ≈ 12·d² parameters (4·d² attention + SwiGLU at 8·d²) plus 32,768 × 192 = 6.3M tied embedding parameters (count with `sum(p.numel())`; the blueprint's number is arithmetic, to be confirmed); T1: `num_hiddens=128, num_heads=2, num_blks=2` (≈ 4.6M by the same arithmetic).
  - Budget per run: T2 `tokens = 2^27 = 134,217,728` (≈ 16.6 tokens per parameter on total parameters); T1 `tokens = 2^24 = 16,777,216`. FLOP sizing with `6ND`: T2 run ≈ 6.5·10^15 FLOP (arithmetic: 6 × 8.1·10^6 × 1.34·10^8). A100 dense BF16 peak is 312 TFLOPS (NVIDIA A100 datasheet), so the 15% utilization assumption gives ≈ 140 s per run.
- **Packages:** `torch`, `numpy`, `scikit-learn` (ridge, gradient boosting), `scipy` (least squares for the mixing law), `pandas`, `datasets` (pool construction only).

#### Narrative arc

1. **The question.** Opening cell: load the three T1 pools and the validation sets; train three tiny proxies for 2^24 tokens on pure web, pure code, pure math, then evaluate each on all three validation sets. Print the 3 × 3 matrix of bits per byte. Symptom: each model is best on its own domain and bad elsewhere, so *a mixture is a choice about which row of this matrix you care about*, and nothing says the choice is linear. (Magnitudes: to be measured.)
2. **3.4.1 Domain weights are hyperparameters.**
   - Mixture = vector w on the simplex over domains; each batch draws a domain with probability w. Distinguish two objectives: average held-out loss over domains, and a target loss Σ_i t_i·bpb_i with a deployment distribution t (the section uses t_avg = (1/3, 1/3, 1/3) and t_dep = (0.7, 0.15, 0.15) over web, code, math; these are stated as illustrative).
   - Compare losses in bits per byte (`d2l.bpb_from_nll` on pool-level sums; `d2l.bits_per_byte` from 1.4 when a model and texts are at hand), never per token, because tokens per byte differ across domains (preview of 3.6; one cell prints bytes per token per domain from the pool metadata).
   - What to hold fixed when attributing a gain to the mixture: tokenizer, model, total tokens, schedule, evaluation sets, data order seed.
3. **3.4.2 Temperature sampling across sources and languages.**
   - `p_i(α) ∝ n_i^α`, α = 1 proportional, α → 0 uniform; the common multilingual choice of α around 0.3 (cite XLM-R and mT5 as practice; confirm values at authoring). Epochs per source e_i = p_i·D/n_i (D1).
   - Cell (CPU): language document counts for ten FineWeb-2 languages fetched through the datasets-server `size` endpoint at authoring and committed (`data/fw2_rows.json`; values read on 2026-10-28: `deu_Latn` 496,029,661; `rus_Cyrl` 699,118,293; `arb_Arab` 62,019,605; `hin_Deva` 22,151,227; `cmn_Hani` 636,092,280; `jpn_Jpan` 400,164,452; `tha_Thai` 35,921,239; `swh_Latn` 1,218,969; `kor_Hang` 60,904,429; English from FineWeb-Edu is not in FineWeb-2); show p_i and e_i for α ∈ {1, 0.7, 0.3, 0} at D = 10^10 tokens-equivalent and flag where e_i exceeds 4 (the repeated-data limit below). `fig_lm_mix_temperature`.
4. **3.4.3 Repetition has a price.**
   - Muennighoff et al.: up to about four epochs of repetition changes loss negligibly compared with unique data; beyond that the value of repeated tokens decays; the effective-data formula with a fitted constant R*_D ≈ 15.4 (paper's fit; confirm in its §5). D2 derives the saturation and `effective_tokens` implements it. Cell: effective tokens versus epochs for U = 1B, and the break-even epoch count at which half the repeated tokens are wasted.
   - Link to 3.5: synthetic or rephrased data as a way to buy unique tokens.
5. **3.4.4 Proxy runs and regression.**
   - The recipe (RegMix): sample many mixtures, train small models, fit a regression from mixture to loss, optimize the mixture on the fitted function, train the optimum at larger scale. We make the transfer check explicit: do proxy rankings transfer to a somewhat larger model? (Exercise 6.)
   - Run design cells: `sample_mixtures` with 8 mixtures from Dirichlet(1) and 8 from Dirichlet(0.3) for training (T2: 16 train + 8 held-out test from a different seed, Dirichlet(1); T1: 12 train + 4 test), `max_weight=0.9`. Show the design on a ternary plot (`fig_lm_mix_simplex`).
   - Train all runs with `train_proxy_lm` (T2: in a `%%writefile` script run in the background; T1: inline); cache results to JSON. Noise floor: three seeds of the uniform mixture (w = 1/3 each) give σ_seed per domain.
   - Fit: ridge on w (features sum to 1, so drop one or use the intercept-free form; D4), gradient boosting (the RegMix-style learner), and the mixing law `L_i(w) = c_i + k_i·exp(Σ_j t_ij·w_j)` of Ye et al. (arXiv 2403.16952), fitted per domain by least squares from several random initializations. Metrics on the 8 (T1: 4) held-out mixtures: per-domain MAE in bits per byte against σ_seed·sqrt(1 + p/n) (D4's floor), and Spearman ρ.
6. **3.4.5 Choosing the mixture.**
   - Optimize the fitted function over a 5,000-point random grid (and a Nelder–Mead polish) for t_avg and t_dep; report the selected weights and the predicted bpb with an interval from bootstrapping the training runs.
   - DoReMi in miniature (D3): reference model = a uniform-mixture proxy; domain weights from group DRO on a proxy; averaged over steps. Report the weights and note that DoReMi's objective is worst-case excess loss, not average loss, so it optimizes something different from RegMix. The trained arms below use them as given.
   - Arms (each T2: 3 seeds; T1: 3 seeds): **uniform**, **regression-selected for t_dep**, **DoReMi weights**, **staged** (WSD, 80% stable at uniform, last 20% in decay on the regression-selected mixture for t_dep; the other arms use WSD as well so that only the mixture differs).
7. **3.4.6 Annealing and curricula.**
   - Why the last tokens matter: during the LR decay the model fits the data it sees; Smol Training Playbook and OLMo 2's staged recipe (mid-training on high-quality and instruction-like data) are the practice, SmolLM2's staged mixtures with web, code, math stages are the example. Distinguish ordering within a stage (a curriculum) from changing the mixture at a stage boundary.
   - Cell: loss curves of the staged versus single-stage arm with a vertical line at the boundary; show that per-domain bpb of the decayed-on domain drops during the decay.
8. **3.4.7 Reading the results honestly.**
   - Cell: the noise table (σ_seed per domain from the uniform seeds) and the comparison of each arm against uniform, per seed, in the same data order seed. `fig_lm_mix_arms`.
   - What transfers: ranking across mixtures at proxy scale versus the effect at the scale of R2 (4.2); state the extrapolation limits (tokens per parameter, tokenizer, model size) in one paragraph.

#### Derivations

**D1. Temperature sampling and epochs per source.**

- *Statement.* With source sizes n_i (tokens), exponent α ∈ [0, 1] and total token budget D, p_i = n_i^α/Σ_j n_j^α, and source i is read e_i = p_i·D/n_i = D·n_i^{α−1}/Σ_j n_j^α times. For α < 1, e_i decreases with n_i: the smallest sources are repeated most.
- *Steps.* (1) By definition source i contributes p_i·D tokens. (2) Divide by n_i. (3) Since n_i^{α−1} is decreasing in n_i for α < 1, the ordering follows. (4) At α = 1 all e_i are equal to D/Σ n_j; at α = 0, e_i = D/(m·n_i).
- *Budget at which a source reaches E epochs:* D_E,i = E·n_i^{1−α}·Σ_j n_j^α.
- *Numerical check.* Toy sizes n = (1000, 100, 10) at α = 0.3: assert Σp_i = 1, assert the epochs from the formula equal those from simulating the sampler for 10^5 tokens within 3 binomial standard errors, assert e_3 > e_2 > e_1.

**D2. Effective tokens under repetition (data-constrained scaling).**

- *Statement.* Let U be the number of unique tokens, a run of E epochs (R = E − 1 repeats). Muennighoff et al. model the effective unique-data size as D′ = U + U·R*·(1 − e^{−R/R*}) with a fitted constant R*. For small R, D′ ≈ U(1 + R) = U·E (repeats are as good as fresh data); for R ≫ R*, D′ → U(1 + R*) (extra repeats add nothing).
- *Assumption.* The paper's functional form, taken as given; the derivation shows the form is the solution of a decay ODE, not that it holds.
- *Steps.* (1) Posit that the k-th repeat of a token is worth a fraction that decays as the repeated mass grows: dD′/dR = U·e^{−R/R*}. (2) Integrate from 0: D′ − U = U·R*·(1 − e^{−R/R*}). (3) Expand for R ≪ R*: ≈ U·R. (4) Take R → ∞: U·R*.
- *Numerical check.* For U = 10^9 and R* = 15.4: assert D′(R = 3)/(4U) = 0.932 ± 0.005 (four epochs retain about 93% of the nominal value under this fit, consistent with the paper's "negligible change up to about four epochs" within its noise; do not read the fit as exact) and D′(R → 1000) within 0.1% of U(1 + R*). Print where the first-order approximation U·E overstates D′ by 10% (R ≈ 3.2).

**D3. DoReMi as multiplicative weights on the simplex.**

- *Statement.* DoReMi solves min_θ max_{α∈Δ} Σ_i α_i·(ℓ_i(θ) − ℓ_ref,i). The α-step of one iteration, α_{t+1} = argmax_{α∈Δ} ⟨α, g_t⟩ − (1/η)·KL(α‖α_t) with g_{t,i} = max(ℓ_i(θ_t) − ℓ_ref,i, 0), is α_{t+1,i} ∝ α_{t,i}·exp(η·g_{t,i}); the algorithm then smooths with the uniform distribution, α ← (1 − c)α + c/m, and returns the average of the iterates.
- *Assumptions.* Per-domain losses are evaluated on the minibatch; clipping at 0 is DoReMi's choice (only domains that are worse than the reference gain weight).
- *Steps.* (1) Lagrangian for the simplex: L(α, λ) = ⟨α, g⟩ − (1/η)·Σ α_i log(α_i/α_{t,i}) − λ(Σ α_i − 1). (2) Stationarity: g_i − (1/η)(log(α_i/α_{t,i}) + 1) − λ = 0. (3) Solve α_i = α_{t,i}·exp(η·g_i − 1 − ηλ). (4) The constant is fixed by normalization: α_{t+1,i} = α_{t,i}·exp(η g_i)/Σ_j α_{t,j}·exp(η g_j).
- *Numerical check.* A 3-domain quadratic toy: ℓ_i(θ) = (θ − c_i)² with c = (0, 1, 3), reference ℓ_ref = 0, θ updated by gradient descent on Σ α_i ℓ_i (seed 0). Check (a) the closed-form α-step equals a numerical maximization of the KL-regularized objective on the simplex (scipy, tolerance 1e-8); (b) the averaged α converges to the weights that equalize the three excess losses at the saddle (compare with a direct solve of the minimax by `scipy.optimize.linprog` on a grid of θ, tolerance 0.02).

**D4. How well can a regression of mixture to loss do, given noise?**

- *Statement.* Fit L = Xβ + ε by OLS with n training runs, p parameters (including the intercept) and noise ε ~ (0, σ²). At a new point x the expected squared prediction error against a *noisy* test observation is σ²·(1 + x′(X′X)⁻¹x), and its average over the training design points is σ²·(1 + p/n). So the test RMSE cannot be below σ·sqrt(1 + p/n) however good the model class.
- *Assumptions.* Correct specification, homoscedastic noise; for a misspecified model add the bias term.
- *Steps.* (1) β̂ − β = (X′X)⁻¹X′ε. (2) Prediction error at x: ŷ − y = x′(β̂ − β) − ε_new. (3) Variance: σ²·x′(X′X)⁻¹x + σ². (4) Average over the n rows of X: tr(X(X′X)⁻¹X′)/n = p/n, giving σ²(1 + p/n). (5) Collinearity: for weights on the simplex, the columns of [1, w₁, w₂, w₃] satisfy w₁ + w₂ + w₃ = 1, so X′X is singular; use two free weights plus the intercept (p = 3) or ridge.
- *Numerical check.* Simulate (seed 0): n = 16 mixtures from the simplex, linear truth β, σ = 0.02; fit OLS on two free weights; over 5,000 replicates compare the empirical mean squared error at new Dirichlet points with σ²(1 + x′(X′X)⁻¹x) (tolerance 3% relative) and report σ²(1 + p/n). Then use the measured σ_seed from the three uniform runs to print the floor for the real regression.

#### From-scratch implementation

Budget ≈ 150 lines.

| Component | Signature | ≈ lines | Test (reference, tolerance, seed) |
|---|---|---|---|
| Mixture sampler | `sample_mixtures`, `temperature_weights` | 20 | Weights sum to 1 (1e-12); max weight ≤ 0.9; D1 check |
| Proxy loop | `train_proxy_lm`: per-domain permuted block indices (non-overlapping 1024-token windows), domain draw by `np.random.default_rng(seed)`, AdamW, warmup + cosine or WSD, bf16 autocast, final per-domain evaluation | 70 | Determinism: the same `(seed, weights, tokens)` gives the same first 10 batches' token hashes and the same loss at step 20 (fp32, CPU, 1e-6); on a single-domain pool with a 2-layer model the loss decreases monotonically on average over 100 steps; empirical domain frequencies within 3σ of w over 10^4 draws |
| bpb conversion | `bpb_from_nll` | 5 | A model that assigns uniform probability over 32,768 tokens gives bpb = 15/bytes-per-token to 1e-9 for a pool whose bytes per token is known |
| Mixture regression | `fit_mixture_regression` (ridge in closed form, GBM from scikit-learn, mixing law with `scipy.optimize.least_squares`) | 35 | D4 simulation; on noiseless data from the mixing law the fit recovers predictions to 1e-6 |
| DoReMi weights | `doremi_weights` | 25 | D3 toy |
| Effective tokens | `effective_tokens` | 5 | D2 check |

#### In production

- Mixture and proxy pipelines in practice: OLMo 2 (Dolmino mid-training mixes), SmolLM2 (staged data), the RegMix and DoReMi code releases; Datatrove for building domain pools; `nanotron`/TorchTitan data loaders (4.1) take mixture weights as sampling probabilities over shard sets.
- Our from-scratch sampler is the oracle for 4.1's weighted streams: a unit test compares domain frequencies and determinism between `train_proxy_lm`'s sampler and `d2l.PackedStream` when both are given the same weights (to be added by the 4.1 author; named here as a cross-chapter test).
- `torch.compile` of the proxy for T2 if it helps throughput (measure; the book's `sec_compilation` owns the technique).

#### Headline experiment

- **Hypothesis (to test).** A regression fitted on proxy runs predicts per-domain bits per byte for held-out mixtures within the seed-noise floor, and the mixture it selects for t_dep achieves lower target-weighted bits per byte than the uniform mixture and than DoReMi's weights.
- **Setup.** Domains web/code/math; R2 tokenizer; proxy model and budget as in "Models and data" (T2: 8.1M parameters, 2^27 tokens; T1: 4.6M parameters, 2^24 tokens). AdamW (β = (0.9, 0.95), weight decay 0.1), batch 64 × 1024 tokens, peak learning rate chosen once on the uniform mixture from {5e-4, 1e-3, 2e-3} at authoring and then fixed (no per-mixture tuning), 2% warmup, cosine to 10% (WSD for the annealing comparison with the same peak). bf16. Seeds: data order seed s ∈ {0, 1, 2}; mixtures as in 3.4.4.
- **Runs.** T2 (36 runs): 16 train mixtures + 8 test mixtures (seed 0 each) + 3 uniform seeds (the noise floor; also arm "uniform") + 3 regression-selected + 3 DoReMi + 3 staged. T1 (25 runs): 12 + 4 + 3 + 3 + 3 (DoReMi and staged at T1 use 2 seeds if the time budget requires; state it). Total budget per run fixed; throughput logged.
- **Metrics.** Per-domain validation bpb; target-weighted bpb for t_avg and t_dep; held-out regression MAE and Spearman ρ; effective epochs per domain for each run.
- **Plots.** `fig_lm_mix_pred_vs_meas`: three panels (web, code, math), x = predicted bpb, y = measured bpb on the held-out mixtures, identity line, a band of ±2σ_seed·sqrt(1 + p/n), one marker style per regressor. `fig_lm_mix_simplex`: ternary scatter of all runs coloured by t_dep bpb, the selected mixture as a star. `fig_lm_mix_arms`: for each arm, the three seed points and the mean of the target-weighted bpb, with a horizontal line at the uniform mean and a grey band of ±2σ_seed.
- **Expected outcome.** Direction only: RegMix reports that regression on many small proxy runs finds mixtures that beat uniform and that proxy rankings transfer to larger models (read its transfer claim and cite its scale); our run is much smaller and uses 16 training mixtures, so the regression may only reach the noise floor, which is itself a finding. DoReMi optimizes worst-case excess loss and need not win on t_dep. Annealing on a higher-quality mixture is expected to lower the bpb of the domains it up-weights (the practice of OLMo 2 and SmolLM2); the amount is to be measured.
- **Noise check.** σ_seed from three uniform seeds per domain; D4's floor; arms are compared seed-paired (same data-order seed) and reported as three differences per comparison. A difference is called real only if all three have the same sign and the mean exceeds 2σ_seed. With n = 3 no bootstrap interval is reported for the arm comparison. For regression metrics the 8 (T1: 4) held-out mixtures are the sample; Spearman ρ with n = 8 is reported without a significance claim.
- **T1 reduction.** `config='t1'`: pools of 20M tokens, model 4.6M, D = 2^24, 12 + 4 + 9 runs. Same code and same functions; only `ProxyCfg` and pool keys change.
- **Artifacts committed for T2.** `lm-mix-runs-t2` (all run logs and per-domain results), the fitted regressors' parameters, the selected mixtures.

#### Figures and tables

- `fig_lm_mix_matrix`: heat map of the 3 × 3 bpb matrix (train domain × eval domain) from the opening cell.
- `fig_lm_mix_temperature`: sampling probability and epochs per language against α.
- `fig_lm_mix_effective_tokens`: D′ against epochs for R* = 15.4 with the linear reference.
- `fig_lm_mix_simplex`, `fig_lm_mix_pred_vs_meas`, `fig_lm_mix_arms`: described above.
- `fig_lm_mix_staged_curves`: validation bpb against tokens for uniform versus staged arms with the stage boundary.
- `tab_lm_mix_noise`: σ_seed per domain; D4 floor; regression MAE per regressor.
- `tab_lm_mix_arms`: weights of each arm, per-domain bpb (mean over seeds ± the three raw values), target-weighted bpb.

#### Pitfalls and misconceptions

- Comparing per-token losses across domains with different bytes per token.
- Reading a regression's R² on training mixtures as success; only held-out mixtures count, and the floor is the seed noise.
- Tuning the learning rate per mixture; this conflates the mixture with the optimizer.
- Choosing the target weights after seeing the results.
- Using a pool smaller than the run needs: repeated tokens quietly change the experiment (D2).
- Sampling mixtures only from the interior: the optimum often lies near an edge of the simplex; hence Dirichlet(0.3) draws.
- Extrapolating a proxy ranking to a model 100× larger without a transfer check; DoReMi and RegMix both report transfer for their settings, not in general.
- Treating annealing on high-quality data as free: it also changes what the model has seen last, so benchmark gains can be recency, not capability.
- Forgetting that the validation sets are from disjoint buckets and must stay disjoint from every pool.

#### Summary (draft)

- A mixture is a point on a simplex and a hyperparameter; its value is defined only relative to a target distribution.
- Temperature sampling and the repetition law (D1, D2) tell how many epochs each source gets and what repeated tokens are worth.
- Proxy runs plus a regression turn the search into data analysis; the achievable accuracy is bounded below by the seed noise (D4), so state the floor before claiming a fit.
- DoReMi is multiplicative weights on worst-case excess loss; it solves a different problem from regression-based selection.
- Annealing on high-quality data during the learning-rate decay is a mixture change in time; report it as one.

#### Exercises

1. **Check.** A source has 2 B tokens and gets weight 0.2 in a 100 B-token run. How many epochs does it see? According to D2 with R* = 15.4, what fraction of the repeated tokens' nominal value remains?
   *Sketch:* epochs = 0.2·100/2 = 10; D′ = 2(1 + 15.4(1 − e^{−9/15.4})) ≈ 2 × (1 + 15.4 × 0.442) = 15.6 B effective versus 20 B nominal, so ≈ 78%.
2. **Check.** Why can't a weight vector on the simplex be regressed with an intercept and all m weights by OLS? What does ridge do about it?
   *Sketch:* the columns sum to the intercept column, so X′X is singular; ridge picks the minimum-norm solution among the infinitely many exact fits.
3. **Derive.** Show that DoReMi's multiplicative update with excess losses g_i ≥ 0 never decreases the weight of a domain whose g_i exceeds the weighted mean excess loss Σ_j α_j g_j, and give the fixed points of the unclipped dynamics.
   *Sketch:* α_i′/α_i = e^{ηg_i}/Σ_j α_j e^{ηg_j} ≥ 1 iff e^{ηg_i} ≥ Σ_j α_j e^{ηg_j}; for small η this is g_i ≥ Σ_j α_j g_j; fixed points have all g_i with α_i > 0 equal.
4. **Derive.** Extend D4 to ridge regression with penalty λ: give the bias and variance of the prediction at x and the λ minimizing the average MSE when β ~ N(0, τ²I).
   *Sketch:* β̂ = (X′X + λI)⁻¹X′y; variance σ²x′(X′X + λI)⁻¹X′X(X′X + λI)⁻¹x; bias −λx′(X′X + λI)⁻¹β; with the Gaussian prior the optimum is λ = σ²/τ² (posterior mean).
5. **Measure.** Predict, then measure, how the held-out MAE of the ridge and GBM regressors changes as the number of training mixtures goes from 6 to 12 to 16 (T1: 4, 8, 12). Plot with the noise floor.
6. **Measure.** Transfer check: train five of the mixtures at T2 scale (8.1M parameters) and at a model four times larger (`num_hiddens=384`, `num_blks=6`) at the same tokens; compute Spearman ρ between the two rankings on t_dep bpb and the regret of choosing the proxy's best mixture at the larger scale.
7. **Open.** Add a fourth domain (for instance the Wikipedia slice `wikimedia/wikipedia` 20231101.en or a multilingual FineWeb-2 slice) and a fifth mixture-selection method of your choice (such as the mixing law with interaction terms, or a bandit over domains during training). Report whether it beats the regression and the uniform baseline under the same noise criterion.

#### Further reading

- Xie et al., [DoReMi](https://arxiv.org/abs/2305.10429): group DRO for domain weights and their transfer to a large model.
- Liu et al., [RegMix](https://arxiv.org/abs/2407.01492): regression on many small proxy runs.
- Ye et al., [Data Mixing Laws](https://arxiv.org/abs/2403.16952): exponential mixing laws with scale extrapolation.
- Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264): repetition, effective data, the four-epoch regime.
- OLMo Team, [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) (staged data, mid-training); Allal et al., [SmolLM2](https://arxiv.org/abs/2502.02737) (staged mixtures with web, code, math); [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (experiment planning and mixture discussion).
- Tutorials: CS336's data and scaling lectures; the [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) for ablation discipline.
- Code: [RegMix](https://github.com/sail-sg/regmix), [DoReMi](https://github.com/sangmichaelxie/doremi), [OLMo](https://github.com/allenai/OLMo).

#### Risks and open decisions

- **Throughput of tiny models** on an A100 is often kernel-launch-bound, not FLOP-bound; the 15% utilization assumption may be optimistic or pessimistic. Fallback: `torch.compile`, a larger micro-batch with gradient accumulation, or fewer runs (24); the T2 cap is 3 h.
- **Statistical power at T1.** Twelve training mixtures and four test mixtures are too few for a firm regression claim; the notebook says so and reports the floor. If T1 results are uninformative, publish the T1 notebook as a pipeline demonstration and put the conclusions on the T2 logs.
- **Code-domain license filter.** `codeparrot-clean` `license` values are per file and may be noisy; permissive-only filtering reduces the pool. Fallback: `bigcode/starcoderdata` (gated, needs a token) or a smaller T1 pool.
- **FineMath's `token_count`** is counted with another tokenizer; the pool uses R2 token counts from the metadata written at build time.
- **Learning-rate choice** at one value for all mixtures may favor some mixtures; mitigated by choosing on the uniform mixture and reporting a single sensitivity check (two other values on the best and worst mixture).
- **Forward reference to 3.6.** The proxies need `lm-r2-tokenizer`. Alternative if the editor dislikes forward references: build the pools with the GPT-2 tokenizer from `tiktoken` (vocabulary 50,257, larger embedding cost) and state it; this changes the pool artifacts only.

### 3.5 Synthetic and rephrased data

- **Label:** `sec_lm_synthetic_data` · **Tier:** T1 · **Budget:** ≈ 12–15 min on one Ada GPU and ≤ 15 GiB (Qwen3.5-2B in BF16, text-only, is 3.8 GB of weights plus a batch-32 KV cache, which only 6 of its 24 layers keep, and a fixed linear-attention state of about 19 MB per sequence; 15 proxy runs of 2^24 tokens each; all to be measured). The 4.7M-token rephrased corpus is precomputed once at authoring (≈ 1 h with batched generation; to be measured) and loaded from `DATA_HUB`. · **Codex units:** 1.7
- **Builds on (book):** `sec_decoding` (temperature and sampling for generation diversity; not re-taught); `sec_scaling-laws` (repeated data); `sec_mdl-statistics` (noise); `sec_gpt` and `sec_kv-cache` (the generation loop and why batching helps; one sentence).
- **Builds on (this part):**
  - 3.3: `d2l.shingles`, `d2l.minhash_signature`, `d2l.lsh_candidates`, `d2l.union_find_clusters` (near-duplicate rate of generated text).
  - 3.4: `d2l.train_proxy_lm`, `d2l.ProxyCfg`, `d2l.bpb_from_nll`, `d2l.effective_tokens`, the pools `lm-mix-web-20m` and `lm-mix-val-1m`, and the noise-floor method; 3.6's frozen tokenizer `lm-r2-tokenizer`.
  - 1.1: `d2l.load_hf` (loads the generator); 1.2: `d2l.generate` is *not* used (batching and stopping need Transformers' `generate`; the notebook says why in one sentence).
  - `assumed:` none.
- **Exports** (`#@save`, new):
  - `d2l.REPHRASE_PROMPTS`: dict of style name to chat-template prompt; styles `wiki` (rewrite as a clear encyclopedic passage) and `qa` (rewrite as a short conversational question–answer exchange), each with the instruction "preserve every fact, name and number; add nothing".
  - `d2l.rephrase_docs(chunks, style, model_id='Qwen/Qwen3.5-2B', max_new_tokens=512, batch=32, temperature=0.7, top_p=0.9, seed=0, backend='hf')`: returns a list of `{'src': chunk, 'out': text, 'n_out': int, 'finish': 'stop'|'length'}`; Qwen3 non-thinking mode (`enable_thinking=False` in the chat template); drops outputs with `finish='length'`; `backend='vllm'` is used if `vllm` is importable.
  - `d2l.synth_metrics(src, out)`: dict `(copy_rate, num_fidelity, len_ratio)`; `copy_rate` is the character-5-gram Jaccard between source and output, `num_fidelity` is the fraction of digit strings in `src` that appear in `out`, `len_ratio` is `len(out)/len(src)`.
  - `d2l.distinct_n(texts, n=4)`: fraction of distinct word n-grams over all n-grams.
  - `d2l.collapse_sim(n, generations, mode='replace'|'accumulate', reps=1000, seed=0)`: the Gaussian refit simulator of D1; returns per-generation mean and variance of the fitted `(μ, σ²)` across repetitions.
  - `d2l.chunk_doc(token_ids, tokenizer, max_tokens=400)`: splits a document at sentence boundaries into chunks of at most `max_tokens` tokens.
- **Data artifacts:**
  - `lm-rephrase-fwedu-5m` (tar): `train_src_S1.bin`, `train_src_S2.bin`, `val_src.bin` (flat `uint16` R2-tokenized source chunks, each chunk terminated by EOS id 0), `train_reph_S1.bin`, `train_reph_S2.bin`, `val_reph.bin` (their rephrasings, same format), and `rephrase.jsonl` with one row per chunk `{chunk_id, src_doc_id, style, model_id, model_revision, prompt_id, seed, n_src, n_out, copy_rate, num_fidelity, len_ratio, finish, src_text, out_text}`: the corpus ships as text (the source chunk and its rephrasing) alongside the token-id files, which are the R2-tokenized form of the same text. The source is FineWeb-Edu (ODC-By 1.0) and the rephraser is Apache-2.0 (`Qwen/Qwen3.5-2B`, checked 2026-10-29), so the artifact carries the FineWeb-Edu attribution in its dataset card and `rephrase.jsonl` records source ids. Sizes: S1 and S2 each 2^21 = 2,097,152 source tokens, val source 2^19 = 524,288 tokens; rephrased token counts are measured and recorded (expected close to the source counts; to be measured). About 2 bytes × 4.7M ≈ 9.4 MB per side plus the JSONL.
  - The manifest stage record `rephrase` (model id and revision, prompt text hash, seed, drops by reason) appended to `lm-corpus-manifest-v1`'s chapter-level example manifest for the synthetic slice.
- **Models and data:**
  - Generator: `Qwen/Qwen3.5-2B` (`15852e8c16360a2fea060d615a32b45270f8a8fc`, Apache-2.0; 2,274,069,824 parameters, 1,881,825,088 in the language model; loaded text-only, non-thinking with `enable_thinking=False`). Optional stronger generator for a measure exercise: `Qwen/Qwen3.5-4B` (`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Apache-2.0; 8.4 GB of text-only weights in BF16, fits the 15 GiB slot with a smaller batch). The factuality judge is the part's Qwen3.5-9B, a different model from the rephraser.
  - Source text: the web pool `lm-mix-web-20m` of 3.4 (FineWeb-Edu `sample-10BT`, buckets 1–7), decoded with the R2 tokenizer (lossless apart from NFC); S1 = chunks of the first documents until 2^21 tokens, S2 = the next 2^21 tokens; validation source = bucket-8 chunks until 2^19 tokens.
  - Inspection set: `HuggingFaceTB/cosmopedia` (Apache-2.0; configs `auto_math_text`, `khanacademy`, `openstax`, `stanford`, `stories`, `web_samples_v1`, `web_samples_v2`, `wikihow`; columns `prompt, text, seed_data, format, audience, text_token_length`; `web_samples_v2` has 10,345,867 rows by the datasets-server size endpoint); stream a few rows only.
  - Proxy model: the T2 proxy of 3.4 (`num_hiddens=192, num_heads=3, num_blks=4`, ≈ 8.1M parameters) at 2^24 tokens per run (≈ 8·10^14 FLOP by 6ND; arithmetic).
- **Packages:** `transformers>=4.51` (Qwen3 chat template), `torch`, optional `vllm` (pin at authoring), `datasets`, `numpy`.

#### Narrative arc

1. **The question.** Opening cell: stream three Cosmopedia rows from different configs and print `format`, `audience`, `seed_data` and the first 200 characters of `text`; then print the number of rows of `web_samples_v2` (10,345,867). Symptom: a large corpus of fluent text exists that no human wrote, and its diversity was *engineered through the prompt* (format, audience, seed), not left to sampling. Then show the cheap failure: the same prompt sampled 9 times at temperature 0.7 gives 9 near-identical paragraphs (`distinct_n` printed).
2. **3.5.1 What counts as synthetic data.**
   - Four uses: generate new documents from seeds (Cosmopedia, phi), rephrase existing text (WRAP, Nemotron-CC), produce demonstrations for post-training (7.3), and produce labels (the 3.2 quality annotations are synthetic). This chapter covers the first two as pretraining data.
   - The evaluation rule from v5's Codex plan: judge synthetic data by its incremental effect on a held-out distribution, not by fluency. One bullet on "textbook-quality" claims (phi-1 in *Textbooks Are All You Need*) and their limits: a small code model with filtered web plus generated textbooks and exercises; the paper's claims are for its setting.
3. **3.5.2 Rephrasing the web.**
   - WRAP (*Rephrasing the Web*): rephrase web chunks in styles with an instruction-tuned model; train on a 1:1 mix of real and rephrased text; the paper reports pretraining speedups at equal compute (read its headline numbers at authoring and quote them with the setting; do not paraphrase from memory). Nemotron-CC: use classifiers to split data, rephrase low-quality text into cleaner forms and generate tasks from high-quality text; the paper reports gains on a long-horizon token budget (read its style list and numbers at authoring).
   - Pipeline cells: chunk → prompt → generate → filter. Chunking by sentences (`chunk_doc`); prompts printed; generation with `rephrase_docs` on 120 chunks (live demo); filters: truncated outputs, length ratio outside [0.5, 1.5], numeric fidelity < 1.0 flagged, copy rate > 0.8 flagged (the thresholds are tunable parameters shown in the manifest, not claims).
   - `synth_metrics` over the 120 live outputs; histogram of copy rate and numeric fidelity (`fig_lm_synth_fidelity`); five side-by-side examples including the worst numeric-fidelity case.
4. **3.5.3 Generating documents from seeds.**
   - Cosmopedia's recipe: seed samples (web text, textbooks' outlines), a target format (textbook, blog post, wikiHow, story) and audience (young children, high-school students, professionals); the prompt is the diversity knob. Cell: 3 audiences × 3 formats for one seed (9 generations of ≤ 300 tokens with `rephrase_docs`-style batching) versus 9 samples of an unconditioned prompt; compare `distinct_n(4)` and the mean pairwise 5-gram Jaccard from 3.3's shingles; `fig_lm_synth_diversity`.
   - Quality versus diversity as a trade: temperature and top-p raise diversity and error together; prompt-level conditioning raises diversity without that cost but needs a prompt budget.
5. **3.5.4 Checking generated data.**
   - Dedup: run the 3.3 MinHash at (b, r) = (9, 13) on the rephrased chunks; report the near-duplicate rate (generators repeat themselves). Contamination: 13-gram check against the benchmark index from 3.3 (a generator may reproduce benchmark items from its training data).
   - Factuality: numeric fidelity as a cheap proxy; what it misses (changed entities, hallucinated additions); an optional LLM-judge cell on 60 pairs with the part's judge, `Qwen/Qwen3.5-9B` (bf16, non-thinking; a different model from the rephraser, so it does not grade its own output), asks "does the output add any claim absent from the source?" and reports the judge's yes-rate with a binomial interval, plus the agreement with `num_fidelity` flags.
6. **3.5.5 Training on it: equal budgets and fair accounting.**
   - Arms (5 arms × 3 seeds, 2^24 tokens each; the model and training recipe are the 3.4 proxy's):
     - **A** real only: S1 (U = 2^21 tokens) repeated 8 epochs;
     - **B** real + rephrased, same source chunks: S1 and rephrase(S1), 4 epochs each (the WRAP-style augmentation of a fixed source set);
     - **C** rephrased only: rephrase(S1), 8 epochs;
     - **D** fresh real: 2^24 unique tokens from the pool (an upper reference: what more real data would give);
     - **E** real + rephrased from different chunks: S1 and rephrase(S2), 4 epochs each (new information in rephrased form).
   - The accounting: nominal tokens are equal (2^24); the repeated-data law of 3.4 (D2, `effective_tokens`) predicts the value of A's 8 epochs at U(1 + R*(1 − e^{−7/R*})) with R* = 15.4, about 6.6U of nominal 8U (a fit to another setting; use as a reference, not a prediction).
   - Evaluate bits per byte on the real validation set and on the rephrased validation set (the style gap). Compare arms in the 3.4 way: paired by data-order seed; three raw differences; noise from within-arm seed standard deviations.
7. **3.5.6 Recursive training and collapse.**
   - The Gaussian toy (D1): each generation trains on samples from the previous generation's fitted model. Replace mode loses variance exponentially and its mean random-walks; accumulate mode (keep earlier data) stays bounded. Cell: `collapse_sim` for n = 100, 50 generations, 1,000 repetitions; compare with D1; `fig_lm_synth_collapse`.
   - A discrete toy (CPU): fit a bigram language model on 200k words, sample a same-size corpus, refit; track distinct word types and the mass of the rarest 10% of types over 10 generations in both modes. Show the tail vanishing (the "curse of recursion" of Shumailov et al.) and that accumulation prevents it; cite Gerstgrasser et al. (*Is Model Collapse Inevitable?*) for the accumulate condition.
   - What it does and does not say: the toy assumes the next model is trained only on its predecessor's output with no fresh real data; real pipelines mix, filter and verify. The notebook states this as a limit of the argument, not as a prediction about frontier practice.
8. **3.5.7 Licenses, provenance and disclosure.**
   - What a generated token's rights depend on: the source text's license (rephrasing does not remove it; the legal position is unsettled; not legal advice), the generator's license (Qwen3: Apache-2.0, verified on the model card; Cosmopedia was generated with Mixtral-8x7B-Instruct and is released under Apache-2.0; some model licenses restrict how outputs may be used or require attribution in derived model names; some API terms restrict training competing models; check each at use).
   - Record in the manifest: generator id and revision, prompt text hash, seed, source ids, filters. A dataset card line "contains synthetic text, share X%". Contamination risk: the generator has seen the benchmarks.

#### Derivations

**D1. Model collapse in the Gaussian refit toy.**

- *Statement.* Generation 0 draws n i.i.d. samples from N(μ, σ²) and fits (μ̂₀, σ̂₀²) by maximum likelihood. In **replace** mode generation t draws n samples from N(μ̂_{t−1}, σ̂²_{t−1}) and refits. Let q = (n − 1)/n. Then E[σ̂²_T] = q^T σ² and Var(μ̂_T) = σ²(1 − q^T). In **accumulate** mode, generation t refits on the pooled data of all generations so far (N_t = n(t + 1) samples); then Var(μ̂_T) = (σ²/n)(1 + Σ_{t=1}^{T} 1/(t + 1)²) → (π²/6)·σ²/n, bounded in T.
- *Assumptions.* Gaussian family, MLE, independent draws within a generation; accumulate mode samples each new batch from the current fit.
- *Steps (replace).* (1) Given σ̂²_{t−1}, the MLE variance of n draws has conditional mean (n − 1)/n·σ̂²_{t−1}. (2) Iterate: E[σ̂²_T] = q^T σ². (3) μ̂_t = μ̂_{t−1} + (σ̂_{t−1}/√n)·z_t with z_t standard normal independent of the past, so increments are uncorrelated with conditional variance σ̂²_{t−1}/n. (4) Var(μ̂_T) = (1/n)Σ_{t=0}^{T−1} E[σ̂_t²] = (σ²/n)·Σ q^t = σ²(1 − q^T) (generation 0 contributes σ²/n for μ̂₀).
- *Steps (accumulate).* (1) μ̂_t = (N_{t−1}μ̂_{t−1} + Σ new samples)/N_t. (2) New samples have mean μ̂_{t−1} + σ̂·z/√n, so μ̂_t = μ̂_{t−1} + (n/N_t)(σ̂/√n)·z_t = μ̂_{t−1} + (σ̂√n/N_t)·z_t. (3) With σ̂ ≈ σ and N_t = n(t + 1): increment variance σ²/(n(t + 1)²). (4) Sum: Var(μ̂_T) = (σ²/n)(1 + Σ_{t=1}^{T} (t + 1)^{−2}), and Σ_{t≥1}(t + 1)^{−2} = π²/6 − 1.
- *Numerical check (seed 0, 4,000 repetitions, n = 100, T = 50).* Replace: E[σ̂²_T] ≈ q^T = 0.605 (one run of this blueprint's check gave 0.587, within Monte-Carlo error) and Var(μ̂_T) ≈ 0.395 (one run gave 0.381). Accumulate: Var(μ̂_T) ≈ 0.01645 (one run gave 0.01644; the bound (π²/6)/n = 0.016449) and E[σ̂²_T] ≈ 0.99. The cell asserts agreement within 3 Monte-Carlo standard errors.

**D2. Equal nominal tokens are not equal effective tokens.**

- *Statement.* For arm A (U unique tokens, E epochs), the effective unique-token budget is D′(U, E) = U + U·R*(1 − e^{−(E−1)/R*}) (3.4's D2). For arm B (U real tokens plus their rephrasing, each pair seen E/2 times) the effective budget is bracketed: D′(U, E) ≤ D′_B ≤ D′(2U, E/2). The lower bound holds if rephrasing adds nothing; the upper bound holds if rephrased text is as informative as fresh text.
- *Assumptions.* The functional form and constant R* of 3.4's D2, taken from another setting; surface variation never hurts (lower bound).
- *Steps.* (1) Rephrasing a chunk adds no facts, so it cannot exceed fresh data of the same size: D′_B ≤ D′(2U, E/2). (2) If surface variation is never harmful, B is at least as good as repeating the real text E times: D′_B ≥ D′(U, E). (3) Evaluate both bounds for U = 2^21, E = 8, R* = 15.4.
- *Numerical check.* Compute the bounds in units of U: D′(U, 8) = 1 + 15.4(1 − e^{−7/15.4}) = 6.62; D′(2U, 4) = 2·(1 + 15.4(1 − e^{−3/15.4})) = 7.45. Assert both numbers, and print the implied maximum improvement of arm B over arm A in effective-token terms (about 12% under this fit); compare with the measured gap relative to the noise floor. The bounds depend on a constant fitted for another setting and are a reference line, not a prediction.

#### From-scratch implementation

Budget ≈ 140 lines.

| Component | Signature | ≈ lines | Test (reference, tolerance, seed) |
|---|---|---|---|
| Chunker | `chunk_doc(token_ids, tokenizer, max_tokens)`: split at sentence ends (`.`, `?`, `!` followed by a space or newline), greedy fill | 15 | Concatenating chunks reproduces the document exactly; no chunk exceeds `max_tokens` except a single over-long sentence (hard split); 20 random documents, seed 0 |
| Rephrase loop | `rephrase_docs`: apply the chat template with `enable_thinking=False`, left-pad, `generate` with `max_new_tokens`, stop at EOS, decode only the new tokens, record `finish` | 45 | Output equals a second call with the same seed (greedy at `temperature=0` exactly; sampled with `torch.manual_seed`); batch of 1 versus batch of 8 gives the same greedy outputs up to padding effects (report any difference) |
| Metrics | `synth_metrics`, `distinct_n` | 30 | Golden strings: identical text gives copy_rate = 1.0 and num_fidelity = 1.0; a digit changed lowers num_fidelity by exactly 1/#numbers; `distinct_n` of a repeated sentence equals 1/k |
| Collapse sim | `collapse_sim` plus the bigram toy | 50 | D1 closed forms within 3 Monte-Carlo standard errors |

#### In production

- vLLM or SGLang (13.2) serve the generator at throughput scales that Transformers' `generate` cannot; the notebook's `backend='vllm'` path is the production variant and the Transformers path is the oracle for output equality under greedy decoding.
- Datatrove has a synthetic-data/inference stage (check the current release at authoring) and the Cosmopedia pipeline is documented in the [Cosmopedia blog post](https://huggingface.co/blog/cosmopedia); the notebook points to them without running them.
- Quality-conditioned rephrasing at scale (Nemotron-CC) and verification of math and code outputs (execution, answer checking) are the production controls that this notebook only names; verifiers are the subject of 9.4.

#### Headline experiment

- **Hypotheses (to test).** H1: B (real + rephrased, same chunks) has lower real-validation bits per byte than A (real only, repeated). H2: E (new information in rephrased form) beats B. H3: C (rephrased only) is worse than A on real validation but better on rephrased validation (a distribution gap). D is the reference bound.
- **Setup.** Proxy: `d2l.GPT(vocab_size=32768, num_hiddens=192, num_heads=3, num_blks=4, max_len=1024)`, AdamW, peak LR as chosen in 3.4, 2% warmup then cosine to 10%, 2^24 tokens per run (batch 64 × 1024; 256 steps), bf16, data order seeds {0, 1, 2}. Arms A–E as in 3.5.5. Validation sets: real (1M tokens, `lm-mix-val-1m` web) and rephrased (`val_reph.bin`, ≈ 0.5M tokens), both in bits per byte using bytes of the NFC text.
- **Procedure.** Build the arms from the artifact files; train 15 runs; evaluate at the end and at 25%, 50%, 75% of tokens for curves.
- **Metrics.** bpb real, bpb rephrased; generalization gap curves (validation minus training bpb) to show overfit under repetition; `synth_metrics` summaries of the rephrased training chunks.
- **Plots.** `fig_lm_synth_arms`: two panels (real val, rephrased val), x = arm (A–E), y = bpb, the three seed points and their mean, a grey band of ±2σ (σ = pooled within-arm seed standard deviation, 10 degrees of freedom). `fig_lm_synth_curves`: validation bpb on real data against training tokens for A, B, D with the repeat boundaries marked.
- **Expected outcome.** Direction only. WRAP and Nemotron-CC report that mixing rephrased with real data helps at their scales (read their settings and quote them); the repeated-data law predicts that eight epochs of 2M tokens waste about 17% of the nominal budget (D2 under a fit from another setting), so any B-over-A gain is bounded to the order of that waste. A null result at this scale is a legitimate outcome and the notebook is written to report it.
- **Noise check.** Three seeds per arm; a difference between arms is reported only if all three paired differences have the same sign and their mean exceeds 2σ_pooled; otherwise the text says "no difference detected at this scale". The 15 runs also give 10 degrees of freedom for σ.
- **T1 reduction.** This notebook is T1 already. Scale knob: `scale=k` multiplies U and the token budget by k (k = 4 fits T2; the committed rephrase must then be regenerated).
- **Artifacts committed for T2+.** `lm-rephrase-fwedu-5m`; the regeneration script `data/make_rephrase.py` with the prompt hash and generator revision.

#### Figures and tables

- `fig_lm_synth_fidelity`: histograms of copy rate and numeric fidelity for the live 120 outputs, with the flag thresholds marked.
- `fig_lm_synth_diversity`: bars of `distinct_n(4)` and mean pairwise Jaccard for unconditioned sampling, audience × format conditioning and (real) Cosmopedia samples.
- `fig_lm_synth_arms`, `fig_lm_synth_curves`: described above.
- `fig_lm_synth_collapse`: variance (log y) and mean spread against generation for replace and accumulate, with the D1 curves overlaid; second panel for the bigram toy (distinct types against generation).
- `tab_lm_synth_styles`: style, prompt hash, mean length ratio, mean copy rate, flagged fraction.
- `tab_lm_synth_arms`: arm, unique tokens, epochs, effective tokens per D2, bpb on both validation sets (mean and the three raw values).

#### Pitfalls and misconceptions

- Judging a rephrase by fluency; a fluent paragraph can change a number or add a claim.
- Letting the generator see the benchmark: its outputs can reproduce items (check with the 3.3 index).
- Comparing arms at equal *nominal* tokens when they repeat different amounts (D2).
- Reporting only the real validation loss: a rephrased-only model looks worse on real text and better on its own style; report both.
- Using a thinking-mode model without disabling thinking: the `<think>` text lands in the data (check `enable_thinking=False` and strip any residue).
- Sampling every output at one temperature and one prompt (diversity collapse); relying on temperature as the only diversity source.
- Treating the Gaussian toy as a forecast; it shows a mechanism under a stated training protocol.
- Assuming generated text is license-free.

#### Summary (draft)

- Synthetic pretraining data comes in two forms: text generated from seeds and text rephrased from existing documents; prompt design, not sampling temperature, is the main diversity control.
- Rephrased output must be checked for copy rate, numeric fidelity, length drift, near-duplicates and benchmark overlap, and every parameter goes in the manifest.
- Equal nominal tokens are not equal effective tokens; repetition has a price and rephrasing buys surface variety, not new facts, unless the source is also new.
- In a Gaussian refit, replacing data with generated data loses variance exponentially, while accumulating keeps the error bounded; real pipelines mix and filter, so this is a mechanism, not a forecast.
- Rights attach to the source text and to the generator; record both.

#### Exercises

1. **Check.** A rephrased chunk has `copy_rate = 0.95` and `num_fidelity = 1.0`. Is it good data? What about `copy_rate = 0.05` and `num_fidelity = 0.5`?
   *Sketch:* the first is almost a copy (little added diversity, a near-duplicate of the source for 3.3's purposes); the second changed half the numbers, so it fails fidelity regardless of fluency.
2. **Check.** Why does the accumulate mode of D1 avoid the variance collapse? What would break it?
   *Sketch:* each generation's new samples are a bounded fraction n/N_t of the data, so the sample-size growth makes the increments shrink like 1/N_t; discarding the original data or fitting each generation on the last batch only restores the replace dynamics.
3. **Derive.** For replace mode, show that after T generations the probability that σ̂²_T < ε·σ² for a fixed ε tends to 1 as T → ∞, and find the number of generations at which E[σ̂²_T] = σ²/2 for n = 100.
   *Sketch:* σ̂² is a positive martingale-like product that shrinks in expectation by q each step and converges to 0 a.s. (log σ̂²_t is a random walk with negative drift by Jensen); q^T = 1/2 gives T = ln 2/ ln(100/99) ≈ 69.
4. **Derive.** In replace mode show that Var(μ̂_T) → σ² as T → ∞ while E[σ̂²_T] → 0, and interpret: what distribution does the limiting model have?
   *Sketch:* from D1, Var(μ̂_T) = σ²(1 − q^T) → σ²; σ̂²_T → 0 in expectation and almost surely. The limit is a point mass at a location distributed N(μ, σ²): the final model is as informative as one draw from the original distribution, and it has lost all spread.
5. **Measure.** Predict, then measure, how the A-versus-D gap changes when A repeats the same U = 2^21 tokens for 2, 4, 8 and 16 epochs (2^22 … 2^25 budget). Compare the measured gap with the D2 effective-token loss at each epoch count.
6. **Measure.** Swap the generator for `Qwen/Qwen3.5-4B` on S1 only and compare B with the original B using `synth_metrics` and bits per byte (3 seeds). Predict whether a stronger generator changes copy rate, numeric fidelity or the arm gap.
7. **Open.** Design a third rephrasing style (for example question–answer pairs distilled from the chunk, or a "knowledge list") and a metric that tells whether it adds new information rather than a new surface; test it on arm E.

#### Further reading

- Gunasekar et al., [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) and the [Phi-4 technical report](https://arxiv.org/abs/2412.08905): synthetic data at the center of the training mix, with each paper's scope.
- Maini et al., [Rephrasing the Web](https://arxiv.org/abs/2401.16380) (WRAP); Su et al., [Nemotron-CC](https://arxiv.org/abs/2412.02595).
- [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (dataset card and generation prompts).
- Shumailov et al., [The Curse of Recursion](https://arxiv.org/abs/2305.17493) (the model-collapse paper); Gerstgrasser et al., [Is Model Collapse Inevitable?](https://arxiv.org/abs/2404.01413).
- Tutorials: [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) (synthetic data and rephrasing discussion); CS336 data lectures.
- Code: [Datatrove](https://github.com/huggingface/datatrove), [vLLM](https://github.com/vllm-project/vllm), the Qwen3.5 chat-template documentation (thinking and non-thinking modes) on the [model card](https://huggingface.co/Qwen/Qwen3.5-2B).

#### Risks and open decisions

- **Generation throughput** on the Ada GPU: if the one-time 4.7M-token generation takes too long, reduce U to 2^20 and the proxy tokens to 2^23, or generate at T2.
- **Qwen3 chat template** and `enable_thinking` flag are version-sensitive; the notebook asserts that no `<think>` tag appears in outputs.
- **Statistical power.** Three seeds and an expected gap bounded by the 12% effective-token margin may not separate arms; the notebook's null-result wording is part of the design. If the editor wants a firmer claim, raise to five seeds at T2 scale.
- **License language.** The statements in 3.5.7 are descriptive, not legal advice; the notebook says so once and cites each licence text.
- **Source rights of rephrased output (decided 2026-10-29).** `lm-rephrase-fwedu-5m` derives from FineWeb-Edu (ODC-By) text rephrased by an Apache-2.0 model; it ships as text with attribution. Its size is to be measured now that it carries text.

### 3.6 Tokenizers at scale

- **Label:** `sec_lm_tokenizers` · **Tier:** T0 for tokenizer training, Viterbi, fertility, the vocabulary-size cost model and the shard writer; T1 for the headline small-LM comparison and the vocabulary-extension runs (v5 lists the section at T0; this is a recorded deviation) · **Budget:** T0 ≈ 15 min on 8 cores and < 8 GiB (HF `tokenizers` BPE on 50 MB and SentencePiece on 50 MB; to be measured); T1 ≈ 12–15 min and ≤ 15 GiB for 6 small-LM runs of one 45 MB epoch each plus 8 continued-training runs of SmolLM2-135M (to be measured; reduce `steps`). The R2 tokenizer and shards are built once at authoring and loaded from `DATA_HUB` · **Codex units:** 1.2 (tokenization at scale), 1.3 (changing the representation)
- **Builds on (book):** `sec_text-sequence` (`d2l.BPETokenizer`, the merge rule, pre-tokenization regexes, special tokens above the BPE range, fertility, digits and glitch tokens; recapped in one paragraph, not re-derived); `sec_language-model` (`subsec_bits-per-byte`, `eq_bpb`); `sec_mdl-information_theory` (code length and why a unigram model of tokens is a compressor); `sec_gpt` (`d2l.GPT`).
- **Builds on (this part):**
  - 3.1: NFC rule, `d2l.CorpusManifest`, `d2l.docs_hash`.
  - 3.4: `d2l.train_proxy_lm`, `d2l.ProxyCfg`, `d2l.bpb_from_nll`, `d2l.temperature_weights`, the bucket rule (chapter overview).
  - 2.1 `sec_lm_word_embeddings`: embedding geometry; the initialization of new rows is the embedding-space placement problem of 2.1 (one sentence).
  - 1.1: `d2l.load_hf` to load SmolLM2-135M and Qwen3 tokenizers.
  - `assumed:` none.
- **Exports** (`#@save`, new unless marked):
  - `d2l.UnigramTokenizer(vocab_size, seed_size=100_000, em_iters=4, prune_frac=0.8)`: `.train(sentences)`, `.encode(text)` (Viterbi), `.encode_sample(text, alpha=0.2, seed=0)` (subword regularization via forward-filtering backward-sampling), `.pieces` (dict piece to log-probability).
  - `d2l.viterbi_segment(text, logp, max_len)`: best segmentation of a string given piece log-probabilities.
  - `d2l.fertility_table(encoders, parallel)`: DataFrame with, per language and encoder: bytes per token, tokens per character, tokens per passage, premium relative to English (tokens ratio on the same passages), share of byte-fallback tokens.
  - `d2l.R2_SPLIT_REGEX`: the pre-tokenization regex (below); `d2l.train_hf_bpe(texts, vocab_size, specials, regex=d2l.R2_SPLIT_REGEX)`: wrapper around Hugging Face `tokenizers`.
  - `d2l.r2_tokenizer()`: loads `lm-r2-tokenizer` from `DATA_HUB` and returns a `tokenizers.Tokenizer`; asserts its SHA-256 against the manifest.
  - `d2l.extend_vocab(model, tokenizer, new_pieces, init='random'|'mean'|'decomp', seed=0)`: returns `(model, tokenizer)` with `len(new_pieces)` added tokens, embedding rows initialized by the chosen rule (tied or untied models).
  - `d2l.write_shards(docs, tokenizer, out_dir, split, shard_tokens=100_000_000, eos_id=0, meta_fn=None, manifest=None)` and `d2l.TokenShards(root, split='train', fetch=False, max_tokens=None)`: the shard writer and the reader that Chapter 4 consumes (format below).
- **Data artifacts and the exact formats** (normative; 4.1's `d2l.PackedStream` and 4.2's reference-model training read these):
  - **Tokenizer, key `lm-r2-tokenizer`:** `tokenizer.json` in the Hugging Face `tokenizers` format. Byte-level BPE, vocabulary size exactly 32,768 = 2^15. Normalizer: NFC (Qwen3's tokenizer carries the same normalizer, verified in its `tokenizer.json`). Pre-tokenizer: `Sequence([Split(R2_SPLIT_REGEX, 'isolated'), ByteLevel(add_prefix_space=False, use_regex=False)])` where `R2_SPLIT_REGEX = (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`. This is the pattern in Qwen3's `tokenizer.json` (verified 2026-10-28): contractions, letters with an optional leading non-letter, **single digits** (`\p{N}`, not groups of three), punctuation runs, newlines, whitespace. Decoder: ByteLevel. Special tokens: ids 0–31 hold 32 specials, passed to the trainer in this order so their ids are fixed: `<|endoftext|>` = 0 (document delimiter and EOS; there is no BOS), `<|pad|>` = 1, `<|im_start|>` = 2, `<|im_end|>` = 3 (for the 7.2 chat template), and `<|reserved_4|>` … `<|reserved_31|>` = 4–31, placeholders that a later chapter can rename (tool calls, thinking delimiters, fill-in-the-middle) without retraining the tokenizer or resizing the embedding. They cost 28 merges, 0.09% of V. Only `<|endoftext|>` occurs in pretraining; the other 31 embedding rows are trained only through the softmax normalizer, so a chapter that starts using one initializes it first (for example to the mean embedding). Ids 32..32,767 are the byte alphabet and merges, in trainer order (do not assume that id − 32 equals a byte value). Trained on 200,000 documents from the R2 train buckets (≈ 1 GB of text; to be measured), disjoint from validation. File size ≈ 2–3 MB (to be measured).
  - **Shards (root directory):**
    - `manifest.json`: schema `d2l-corpus/1` (3.1) with `tokenizer = {file, sha256, vocab_size: 32768, specials: [...]}`, `format = {dtype: "uint16", eos_id: 0, endianness: "little", doc_end: "eos"}`, `sources` (FineWeb-Edu `sample-10BT` with the dataset revision hash, license ODC-By, row count), `stages` (`nfc`, `bucket_split`, `shuffle(seed=0, buffer=500000)`, `tokenize`), and `splits` with per-shard `{file, n_tokens, n_docs, sha256}`.
    - `{split}-{k:05d}.bin`: raw little-endian token ids, no header; the dtype is `uint16` when the vocabulary is at most 65,536 (every id ≤ 65,535) and `uint32` otherwise (e.g. Qwen's 151,936 in 4.5), recorded in `manifest.format.dtype`; R2 (V = 32,768) uses `uint16`. `write_shards` appends EOS (id 0) after every document. Each document is its token ids followed by one `<|endoftext|>` (id 0); id 0 appears nowhere else in the file; documents never span shards. A shard is closed at the first document boundary at or after `shard_tokens` (default 10^8 tokens, i.e. ≈ 200 MB per file).
    - `{split}-{k:05d}.idx`: raw little-endian `uint32` array of length `n_docs + 1`; `offsets[i]` is the index of the first token of document i in the shard, `offsets[n_docs]` equals the shard's token count; document i is `bin[offsets[i]:offsets[i+1]]` and ends with the EOS id.
    - `{split}-{k:05d}.meta.parquet` (audits and filtering; not needed for training): `doc_id` (string, FineWeb `<urn:uuid:…>`), `n_tokens` (int32, including EOS), `bucket` (int16), `edu_score` (float32), `src_row` (int64, row index in the source stream).
    - Document order: source stream order after the seeded shuffle; split by the bucket rule (bucket 0 = validation, buckets 9–255 = train); NFC applied before tokenization.
    - Reader contract (`TokenShards`): `.n_tokens`, `.n_docs`, `.n_shards`, `.vocab_size`, `.eos_id`; `.shard(k)` returns a read-only `np.memmap` of the `.bin`; `.offsets(k)` returns the `.idx` array; `.doc(i)` returns the i-th document of the split in global order as `np.int32`; `.iter_docs(start=0)` yields `(i, array)`; `.locate(i)` returns `(shard, local_doc, token_offset)` (the cursor 4.1 stores for exact resumption); `.verify()` recomputes the shard hashes; `max_tokens` truncates the split at a shard boundary for reduced configurations; `fetch=True` downloads missing shards from `DATA_HUB` entries listed in the manifest.
  - **`DATA_HUB` keys and sizes** (SHA-1 values filled at publication). The tokenizer goes in the d2l-owned Hugging Face model repository that also holds the R2 checkpoints (4.2); the shards, validation set and manifest go in a d2l-owned dataset repository (`https://huggingface.co/datasets/<repo>/resolve/<commit>/<file>`) whose card carries the FineWeb-Edu ODC-By 1.0 attribution. Entries pin a commit hash, never `main`:

    | Key | Content | Size |
    |---|---|---|
    | `lm-r2-tokenizer` | `tokenizer.json` | ≈ 2–3 MB |
    | `lm-fwedu-val` | tar: `manifest.json`, `val-00000.{bin,idx,meta.parquet}`, `val.jsonl.gz` (the same documents as NFC text) | ≥ 2^23 = 8,388,608 tokens ≈ 16.8 MB of `.bin`, + text |
    | `lm-fwedu-shards-100m` | tar: `manifest.json` (+ tokenizer), `train-00000.{bin,idx,meta.parquet}` | ≈ 100M tokens = 200 MB `.bin`; `.idx` ≈ 4 bytes per document (≈ 0.4 MB if documents average about 1,000 tokens; the average is measured, see below) |
    | `lm-fwedu-manifest-4b` | `manifest.json` for 40 train shards of 10^8 tokens (≈ 4.0·10^9 tokens, ≈ 8.0 GB of `.bin`) with their URLs and hashes; shards are fetched on demand by `TokenShards(fetch=True)` | manifest ≈ 20 KB; shards 40 × 200 MB |

    Sizing note for 4.2: by 6ND a 150M-parameter model trained on 2·10^9 tokens costs 1.8·10^18 FLOP; the manifest provides up to 4.0·10^9 tokens so that 4.2 can choose its budget with `max_tokens`. The expected documents per shard are ≈ 10^5 if the average length stays near the nominal 10^10 / 9,672,101 ≈ 1,034 GPT-2 tokens per document (arithmetic from the dataset name and row count; the real R2-token average is measured at build time and written to the manifest).
- **Models and data:**
  - Multilingual slice (10 languages): English from `HuggingFaceFW/fineweb-edu` `sample-10BT` side buckets (ODC-By); `deu_Latn, rus_Cyrl, arb_Arab, hin_Deva, cmn_Hani, jpn_Jpan, tha_Thai, swh_Latn, kor_Hang` from `HuggingFaceFW/fineweb-2` (ODC-By; train split; row counts read on 2026-10-28 in the 3.4 entry; `swh_Latn` has 1,218,969 rows and `tha_Thai` 35,921,239, a 29-fold spread). Each language contributes `mb_per_lang = 5` MB of text (50 MB total, config), taken from the first rows of a seeded shuffle buffer; 90% for tokenizer and LM training, 10% held out.
  - Parallel evaluation text: `facebook/belebele` (CC-BY-SA-4.0; ungated), column `flores_passage` (the FLORES passage; 900 rows per language, of which 488 distinct passages by the dataset's design (verify the number at authoring)), configs `eng_Latn, deu_Latn, rus_Cyrl, arb_Arab, hin_Deva, zho_Hans, jpn_Jpan, tha_Thai, swh_Latn, kor_Hang` (note `zho_Hans` here corresponds to FineWeb-2's `cmn_Hani`; all names verified to exist). Passages are translations of the same text, so token counts are comparable.
  - Reference tokenizers: GPT-2 and `o200k_base` through `tiktoken`; `Qwen/Qwen3-0.6B` (Apache-2.0; `tokenizer.json`: BPE, 151,643 base entries plus 26 added tokens, NFC normalizer, single-digit regex; verified 2026-10-28); `HuggingFaceTB/SmolLM2-135M` (Apache-2.0; 134,515,008 parameters; `vocab_size` 49,152, tied embeddings, hidden size 576, 30 layers; tokenizer: `Digits(individual_digits=True)` then `ByteLevel` with the GPT-2 regex, no normalizer; 48,900 merges and 17 added tokens; verified).
  - Hindi data for extension: `HuggingFaceFW/fineweb-2` `hin_Deva` (22,151,227 rows), first 30,000 documents (config `n_docs`; size to be measured) plus 500 held-out documents; Belebele `hin_Deva` passages as a parallel dev set.
  - English retention set: the first 200 documents of `lm-fwedu-val` text.
- **Packages:** `tokenizers>=0.21` (PyPI latest 0.23.3, 2026-10-09), `sentencepiece>=0.2` (latest 0.2.2), `tiktoken`, `regex`, `transformers>=4.51`, `numpy`, `pandas`, `pyarrow`, `datasets`.

#### Narrative arc

1. **The question.** Opening cell: encode the 488 distinct Belebele passages in ten languages with GPT-2, `o200k_base`, Qwen3 and SmolLM2 tokenizers and print `d2l.fertility_table`: bytes per token and the premium over English (tokens for the same passages). Symptom (to be measured): the same content costs several times more tokens in Hindi, Thai or Arabic under an English-centered tokenizer, which means several times less context, compute and (through the loss) less learning per byte. One paragraph recaps `sec_text-sequence` (byte-level BPE, pre-tokenization, specials) and states what is new: unigram models, what the training mixture does to a tokenizer, the size of the vocabulary, changing a trained model's vocabulary, and the frozen R2 tokenizer.
2. **3.6.1 Unigram tokenization.**
   - The model: a piece set V with probabilities p(x); a segmentation s = (x₁, …, x_m) of a string has probability ∏ p(x_i); the string's probability sums over all segmentations; the tokenizer returns the argmax (Viterbi) or a sample (subword regularization, Kudo 2018). BPE is deterministic given its merge list; unigram is probabilistic and can produce multiple segmentations.
   - Training (D2): seed vocabulary of frequent substrings (all single characters plus the most frequent substrings up to a length cap), EM on expected piece counts by forward–backward on the lattice, prune the pieces whose removal costs least likelihood (keep all single characters), repeat until the vocabulary size; SentencePiece's engineering (suffix array seed, `shrinking_factor`, `num_sub_iterations`, byte fallback, `character_coverage`).
   - Cells: from-scratch `UnigramTokenizer` on 1 MB of English; compare its segmentations with SentencePiece's on 1,000 held-out sentences when SentencePiece's pieces and scores are loaded into `viterbi_segment` (exact equality expected for Viterbi; EM-trained vocabularies differ; report the overlap of vocabularies); show five words with two plausible segmentations and their log-probabilities; sample five segmentations of one sentence with `encode_sample` (alpha = 0.2).
   - SentencePiece pitfalls shown by cell: the default normalization rule `nmt_nfkc` rewrites text (compare `identity`); `▁` marks word boundaries instead of byte-level space handling; `byte_fallback` removes `<unk>` for unseen characters; `character_coverage=0.9995` versus 1.0 for CJK.
3. **3.6.2 Training data decides who pays.**
   - Fertility, tokenization premium, parity: Petrov et al. and Ahia et al. document premiums across languages; Rust et al. connect tokenizer quality to monolingual performance. Our measurement uses parallel passages so premium is a clean ratio.
   - Experiment cell (T0): train an HF byte-level BPE (V = 16,384, R2 regex) on the 10-language slice under three language weightings, α ∈ {1, 0.3, 0} (`temperature_weights` on the FineWeb-2 row counts, with English assigned the size of `deu_Latn` as a stated assumption); report the premium per language on the held-out Belebele passages. `fig_lm_tok_premium`. State that a tokenizer's language mix and the model's training mix are separate choices and should be consistent.
4. **3.6.3 Vocabulary size.**
   - The cost model: the output projection and embedding cost V·d parameters and 6·V·d training FLOP per token, while a larger V shortens sequences by the compression b̄(V) bytes per token (D3). Show the measured bytes per token of one BPE run read at prefix sizes V ∈ {1K, 2K, …, 64K} (the prefix property of `sec_text-sequence`) on English and on the 10-language slice. `fig_lm_tok_bytes_vs_vocab`.
   - Evaluate D3: the per-byte training-compute curve for a reference transformer (d = 1024, L = 24, non-embedding ≈ 12·L·d² ≈ 302M parameters) and where its minimum falls on the measured b̄(V); the elasticity condition; the limits of this cost model (ignores inference memory, KV cache, the effect of the vocabulary on loss itself). Tao et al. (*Scaling Laws with Vocabulary*) fit optimal vocabulary sizes with a unit-normalized loss and report that larger models deserve larger vocabularies; the notebook quotes their fitted relation at authoring and compares the direction with D3's.
   - Practical rules: round V to a multiple of 64 or 128 for matmul efficiency (tensor-core tile alignment; the book's `sec_hardware` owns the reason), reserve ids for specials and for later chat tokens (R2 reserves 32), untied versus tied embeddings.
5. **3.6.4 Numbers, whitespace and code.**
   - Digits: GPT-2 lets BPE carve numbers by frequency; Llama-3 and GPT-4 tokenizers cap digit runs at three; Qwen3 and R2 split every digit (the regexes are in the `tokenizer.json` files). Cell: `12345678`, `3.14159`, `1,234,567` under GPT-2, `o200k_base`, Qwen3, R2; and a column-addition cell showing which digits share a token. The arithmetic consequences are in `sec_text-sequence`'s exercises; here the point is the design choice and its token cost (digit-heavy text: tokens per digit).
   - Code: indentation runs, tabs, newlines; tokens per line and bytes per token on 2,000 permissive `codeparrot-clean` files for GPT-2, `o200k_base`, Qwen3 and R2; show that GPT-2 has no merge for long whitespace runs while the R2 regex isolates `\s*[\r\n]+` and whitespace runs so BPE can learn them.
6. **3.6.5 Extending a trained model's vocabulary.**
   - Why: a model whose tokenizer fragments a language (SmolLM2 on Hindi) pays in context and compute; adding tokens shortens sequences, but the new embedding rows have never been trained.
   - Procedure: (1) train a Hindi byte-level BPE of 8,192 pieces on the Hindi slice; (2) keep pieces that decode to complete Unicode strings with at least two Devanagari characters and are absent from the SmolLM2 vocabulary: the first 4,096 by merge order become `new_pieces` (the count is measured); (3) `tokenizer.add_tokens(new_pieces)` (added tokens are matched before BPE, so ASCII text is segmented exactly as before; test it); (4) `model.resize_token_embeddings(len(tokenizer))` through `d2l.extend_vocab`, which then writes the new rows. Alternative of editing merge lists is in the exercises.
   - Initializations (all for the tied matrix of SmolLM2; for untied models the same rule fills the output matrix): **random**: N(0, σ²) with σ = the config's `initializer_range` (0.0417 for SmolLM2-135M); **mean**: the mean of the old embedding rows (plus independent noise of 1% of the old rows' standard deviation so rows are not identical); **decomposition**: the mean of the old embeddings of the pieces that the old tokenizer assigns to the new token's string. The 2.1 view: each rule places the new vector at a different point of the embedding space. FOCUS (Dobler and de Melo) and WECHSEL (Minixhofer et al.) use cross-lingual similarity to place vectors; the notebook describes them and does not implement them (they need external embeddings). Hewitt's blog post on initializing new word embeddings is the analysis of the mean-versus-sampled issue (cite by title; the URL did not respond during verification).
   - Diagnostics at step 0 (no training): (a) probability mass on the new tokens for 200 English contexts (should be ≈ 0; large values mean the new rows hijack the softmax); (b) Hindi bits per byte on the held-out documents and on Belebele `hin_Deva`; (c) English bits per byte on the retention set (must equal the unextended model's value up to the tokenizer being unchanged on ASCII: assert equality of token ids on English text).
7. **3.6.6 Byte patches as an aside.** Why a tokenizer is a fixed, frozen compression decision; Byte Latent Transformer replaces fixed tokens by entropy-driven patches (fig and one paragraph; architecture in 14.8). Toy cell (CPU, 15 lines): a character 3-gram model's entropy on Hindi text sets patch boundaries where entropy jumps; plot patch lengths against BPE token lengths on the same text. Not a claim about BLT's quality.
8. **3.6.7 Freezing the tokenizer and writing shards.**
   - Train `lm-r2-tokenizer` (HF BPE, V = 32,768, the specials above, 200,000 train-bucket documents); round-trip test on 10,000 documents; bytes per token on FineWeb-Edu compared with GPT-2 (using the dataset's own `token_count` column as the GPT-2 count: ratio of R2 tokens to GPT-2 tokens, measured).
   - The bucket rule and the shard writer: tokenize with `Tokenizer.encode_batch` (multi-threaded Rust), append EOS, write `.bin`/`.idx`/`.meta.parquet`, hash, update the manifest. Run for 100M train tokens and the validation set at T0 (CPU; throughput to be measured; the T2 build of 4.0·10^9 tokens is a script `data/build_r2_shards.py`, committed with its manifest).
   - `TokenShards` demo: random document access, `locate`, a doc-boundary check that every EOS in the `.bin` equals an offset; the leakage check from the 3.3 tools between validation and train (exact 50-gram hash overlap on a 10^5-document sample: expected zero by construction of buckets; report).
   - The chapter-level manifest: print the final manifest, its SHA-256, and the sizes table.

#### Derivations

**D1. Bits per byte from per-token loss, and why it is the comparable quantity.**

- *Statement.* A model that assigns a text of B UTF-8 bytes (after the fixed normalization) a sequence of N tokens with mean negative log-likelihood L nats per token has bits per byte bpb = N·L/(B·ln 2) = L/(b̄·ln 2), where b̄ = B/N is the bytes per token. For a *fixed text*, bpb is the code length per byte of the model's probability for that text under the tokenization; two tokenizers are compared by bpb because both are code lengths of the same bytes.
- *Assumptions.* The probability of the text is the product of token conditionals for the tokenization that the tokenizer returns (a deterministic tokenizer); decoding the tokens returns exactly the same bytes (lossless), which is why byte-level tokenizers are compared on normalized text. For a stochastic tokenizer the marginal over segmentations is the right quantity; a single segmentation gives an upper bound on the code length.
- *Steps.* (1) Total code length in nats = N·L. (2) Convert to bits: divide by ln 2. (3) Normalize by the number of bytes B. (4) Substitute N = B/b̄. (5) Perplexity per token exp(L) is not comparable across tokenizers because b̄ differs: with the same bpb, a tokenizer with larger b̄ has larger L.
- *Numerical check.* On 100 held-out documents, compute bpb three ways for the headline LM: from the mean token loss and b̄, from the summed loss over the total byte count, and from per-document values weighted by bytes; assert equality to 1e-9. Show that a byte-fallback character-level model and a BPE model on the same text have different perplexities but bpb within a stated gap.

**D2. Unigram language-model tokenization: forward–backward, Viterbi, M-step.**

- *Statement.* For a string s of length n over a piece set V with probabilities p(x) (Σ p = 1) let S(s) be the set of segmentations. Define forward α₀ = 1, α_j = Σ_{i<j: s[i:j]∈V} α_i·p(s[i:j]); backward β_n = 1, β_i = Σ_{j>i: s[i:j]∈V} p(s[i:j])·β_j. Then P(s) = α_n, the posterior expected count of piece x in s is c_x(s) = Σ_{(i,j): s[i:j]=x} α_i·p(x)·β_j / P(s), and the EM update p_new(x) ∝ Σ_s c_x(s) maximizes the expected complete-data log-likelihood. The Viterbi segmentation replaces sums by maxima.
- *Assumptions.* Pieces are generated i.i.d. (a unigram model over pieces); the string is the concatenation of the pieces; V contains every single character so S(s) is non-empty.
- *Steps.* (1) The segmentations are paths in the DAG over positions 0..n whose edge (i, j) has weight p(s[i:j]); P(s) is the sum of path weights, computed by the forward recursion. (2) The posterior probability that a path uses edge (i, j) is α_i·p(s[i:j])·β_j/P(s). (3) The expected-complete-data objective is Σ_x c_x log p(x) with the constraint Σ_x p(x) = 1; a Lagrange multiplier gives p(x) = c_x/Σ c. (4) EM never decreases log P(s) summed over the corpus. (5) Viterbi: δ_j = max_{i<j} δ_i·p(s[i:j]); backpointers recover the path. (6) Pruning: the loss of removing piece x is the drop in total log-likelihood when every occurrence is re-segmented without x (Kudo's approximation uses the Viterbi paths); keep the single characters; drop the 20% lowest-loss pieces per round.
- *Numerical check.* (a) For 200 random strings of length ≤ 12 over a 3-letter alphabet with a random piece set, compare the forward–backward expected counts with brute-force enumeration of all segmentations to 1e-9; (b) the Viterbi path equals the argmax over the enumerated segmentations (ties broken by fewest pieces); (c) over 6 EM iterations on a 5,000-string toy corpus the corpus log-likelihood is non-decreasing (tolerance 1e-9); (d) with SentencePiece's learned pieces and scores loaded, `viterbi_segment` matches `sp.encode(..., out_type=str)` on 1,000 sentences (report mismatches; ties and normalization are the expected causes).

**D3. Vocabulary size from a per-byte compute model.**

- *Statement.* With non-embedding parameters A (compute 6A per token), embedding dimension d and vocabulary V, the training compute per byte is C(V) = 6(A + V·d)/b̄(V). If b̄(V) = c·V^ε locally, the minimum satisfies ε = V·d/(A + V·d), i.e. V* = ε·A/(d(1 − ε)). The condition says: grow the vocabulary until the elasticity of compression with respect to V falls to the share of the softmax in per-token compute.
- *Assumptions.* Compute per token is 6 × (parameters touched by every token); attention FLOP and KV-cache are ignored; b̄ depends on V alone; loss per byte is unchanged by V (this last assumption is what the scaling-law paper relaxes).
- *Steps.* (1) C(V) ∝ f(V) = (A + dV)/b̄(V). (2) f′(V) = [d·b̄ − (A + dV)·b̄′]/b̄². (3) Set f′ = 0: d·b̄ = (A + dV)·b̄′. (4) Divide by b̄ and multiply by V: V·d/(A + V·d) = V·b̄′/b̄ = d ln b̄/d ln V = ε. (5) For constant ε solve for V.
- *Worked example (arithmetic).* d = 1024 and A = 12·24·1024² = 301,989,888 (≈ 302M): ε = 0.10 gives V* = 0.1·302M/(1024·0.9) ≈ 32,768 by the formula (ε = 0.10 is an illustrative value, not a measurement). The notebook then measures ε(V) from the BPE prefix curve and reports where the elasticity crosses the softmax share.
- *Numerical check.* A grid search of C(V) over V ∈ [1,000, 200,000] with a synthetic b̄ = c·V^0.10 reproduces V* to within 1%; with the measured b̄(V) the grid minimum and the elasticity-crossing point agree within one grid step.

#### From-scratch implementation

Budget ≈ 150 lines.

| Component | Signature | ≈ lines | Test (reference, tolerance, seed) |
|---|---|---|---|
| Unigram trainer | `UnigramTokenizer.train`: seed vocabulary by frequent substrings (cap 12 characters, `seed_size`), EM with forward–backward in NumPy log-space, loss-based pruning, final vocabulary of `vocab_size` | 65 | D2 checks (a)–(c); trains on 1 MB English in a stated number of seconds (to be measured) |
| Viterbi and sampling | `viterbi_segment`, `encode_sample` (forward filtering, backward sampling with temperature alpha) | 25 | D2 check (b), (d); sampled segmentations have empirical frequencies within 3σ of their posterior probabilities for a 6-character string over 10,000 draws (seed 0) |
| Extension and init | `extend_vocab` with three rules; mean/decomposition computed from the embedding matrix | 30 | English token ids unchanged for 1,000 ASCII sentences after extension (exact); with `init='decomp'` the step-0 hidden-state logit of a new token equals the mean of its pieces' logits (tied embeddings, linear in the embedding) to 1e-5 |
| Shard writer and reader | `write_shards`, `TokenShards` | 50 | Round trip: tokens of 1,000 documents written then read back equal the direct `encode` of the texts; every document ends with exactly one EOS; `offsets[-1]` equals the file's token count; SHA-256 stored equals recomputed; `.idx` as `uint32` matches `np.cumsum` of lengths |

Oracles: `sentencepiece` (unigram and BPE), Hugging Face `tokenizers` (byte-level BPE), `d2l.BPETokenizer` (merge-by-merge comparison on a 1 MB sample at V = 1,024 with the same regex; report the number of identical merges among the first 256, differences expected only at count ties), `minbpe` (reference BPE).

#### In production

- SentencePiece (`spm.SentencePieceTrainer.train(model_type='unigram'|'bpe', vocab_size=…, byte_fallback=True, normalization_rule_name='identity', split_digits=True, character_coverage=…)`) and Hugging Face `tokenizers` (`BpeTrainer`, `Split` and `ByteLevel` pre-tokenizers, `Tokenizer.encode_batch`) are the two tools the notebook uses; Transformers' `PreTrainedTokenizerFast`, `add_tokens` and `resize_token_embeddings` for the extension; `tiktoken` for reading OpenAI encodings.
- At corpus scale: tokenize in parallel across shards, write `uint16`, memory-map at training time (the pattern of nanoGPT/build-nanogpt and of Megatron's indexed datasets); keep the tokenizer's SHA-256 in the manifest so that a model checkpoint and its data cannot be paired with a different vocabulary.

#### Headline experiment

- **Hypothesis (to test).** At a fixed vocabulary size and equal training bytes, the tokenizer's *premium profile* (tokens per passage relative to English) predicts the per-language bits per byte of a small LM trained on that tokenization, and unigram and BPE differ less across tokenizers than the premium spread across languages does.
- **Setup.** 10-language slice (50 MB, 90% train, 10% held out). Three tokenizers at V = 16,384: SentencePiece unigram, SentencePiece BPE (both `byte_fallback=True`, `identity` normalization after our NFC, `split_digits=True`, `character_coverage=0.9995`), and Hugging Face byte-level BPE with the R2 regex. Model: `d2l.GPT(vocab_size=16384, num_hiddens=256, num_heads=4, num_blks=4, max_len=512)` (≈ 7.3M parameters by the arithmetic 12·d²·L + V·d; to be measured), the 3.4 training recipe (AdamW, 2% warmup, cosine to 10%), a fixed *byte* budget instead of a token budget (each run trains for one epoch on the same 45 MB of text, so the token count differs by tokenizer and is recorded), bf16, seeds {0, 1}.
- **Procedure.** Train tokenizers (T0); encode; train six LMs (3 tokenizers × 2 seeds); evaluate bpb per language on the held-out 10% and on Belebele passages; compute fertility and premium on Belebele.
- **Metrics.** bpb per language and macro-average; bytes per token; premium; vocabulary overlap between the three tokenizers per language; unknown or byte-fallback rate.
- **Plot (`fig_lm_tok_bpb_premium`).** x = premium (tokens relative to English) per language, y = bits per byte, one point per (language, tokenizer), marker by tokenizer, lines connecting a language across tokenizers; second panel (`fig_lm_tok_bpb_bars`): per-language bpb by tokenizer with the two-seed range.
- **Expected outcome.** Direction only. Unigram and BPE are usually close at the same vocabulary size in the literature (Bostrom and Durrett, *Byte Pair Encoding is Suboptimal for Language Model Pretraining*, report a unigram advantage on downstream tasks; cite at authoring and state that the claim is about their setting, not ours); a premium spread of several times is expected for non-Latin scripts under any tokenizer trained on a mix skewed to English. Bits per byte differences between tokenizers are to be measured against the seed-to-seed range.
- **Noise check.** Two seeds per tokenizer give a range per language; differences smaller than the larger of the two ranges are not interpreted; evaluation passages are bootstrapped (by passage) for the macro-average with `d2l.paired_bootstrap` between tokenizers; with so few seeds the notebook states that rankings within the seed range are "not distinguished".
- **T1 reduction.** `mb_per_lang=5`, `vocab=16384`, `num_blks=4`; the full variant (`mb_per_lang=50`, V = 32,768, `num_blks=8`) is T2 and not part of this blueprint's committed artifacts.
- **Second experiment (T1): vocabulary extension.** Arms: unextended SmolLM2-135M with its own tokenizer (control), and the extended model with `random`, `mean` and `decomp` initialization; continued training on Hindi with AdamW (peak LR 1e-4, 10 warmup steps, cosine, batch 16 × 1024 tokens, `steps = 150` at T1 and 2,000 at T2), two data-order seeds (the random-init arm also varies its initialization with the seed). Metrics: Hindi bpb (held-out FineWeb-2 documents and Belebele passages) at steps {0, 25, 50, 100, 150}; English bpb on the retention set; step-0 new-token mass; Hindi bytes per token before and after. Plot `fig_lm_tok_extend_curves` (x = step, y = Hindi bpb, one line per arm, seed points) and `tab_lm_tok_extend`. Hypothesis: informed initializations reach a lower Hindi bpb at step 0 and early steps than random, and the arms converge as training proceeds; the size of the step-0 gap and the English change are to be measured. Noise: step-0 values are deterministic and bootstrapped over passages with `d2l.paired_bootstrap`; trained values are reported as the two seed points. T2: `steps = 2000` and three seeds.
- **Artifacts committed.** `lm-r2-tokenizer`, `lm-fwedu-val`, `lm-fwedu-shards-100m`, `lm-fwedu-manifest-4b`, and the two result tables as JSON.

#### Figures and tables

- `fig_lm_tok_premium`: premium per language for four reference tokenizers and for the three α-weighted tokenizers.
- `fig_lm_tok_unigram_lattice`: a lattice diagram of segmentations of one word with edge probabilities and the Viterbi path (drawn from a real run of `UnigramTokenizer`).
- `fig_lm_tok_bytes_vs_vocab`: bytes per token against V for English and the 10-language slice (log x); `fig_lm_tok_cost_vs_vocab`: per-byte compute against V with the D3 minimum and the elasticity curve.
- `fig_lm_tok_bpb_premium`, `fig_lm_tok_bpb_bars`: headline experiment.
- `fig_lm_tok_extend_curves`: extension experiment.
- `fig_lm_tok_patching`: patch boundaries versus BPE token boundaries on a Hindi sentence.
- `tab_lm_tok_fertility`: the opening table.
- `tab_lm_tok_digits_code`: digits and code token statistics per tokenizer.
- `tab_lm_tok_shard_format`: the normative format of the shards (as above).
- `tab_lm_tok_extend`: arms, step-0 mass on new tokens, Hindi and English bpb, bytes per token.

#### Pitfalls and misconceptions

- Comparing perplexity across tokenizers (D1); counting bytes before normalization for one tokenizer and after for another.
- SentencePiece's default normalization silently applies NFKC-like rules (`nmt_nfkc`).
- Training the tokenizer on text the model never sees (glitch tokens: `sec_text-sequence`), or on a language mix unlike the model's.
- Choosing V by habit; the softmax and embedding share and the compression elasticity (D3) are measurable.
- Treating the number of new tokens as the benefit: shorter sequences help only if the new embeddings are trained enough.
- Mean initialization of new rows making all new tokens identical and hijacking the softmax at step 0 (check the new-token mass).
- Forgetting the output matrix when embeddings are untied; forgetting `model.config.vocab_size`.
- Adding tokens that contain ASCII and thereby changing English segmentation.
- Writing shards with `uint16` when V > 65,536 (assert; use `uint32`), or with a header the reader does not expect; letting EOS appear inside a document.
- Pairing a checkpoint with a different tokenizer file of the same vocabulary size.

#### Summary (draft)

- A tokenizer is a compressor with a training set: the language mix of its data decides the premium each language pays in sequence length.
- Unigram tokenization is a probabilistic model over segmentations trained by EM and decoded by Viterbi; BPE is a deterministic merge list; both need byte coverage and a stated normalization.
- Compare tokenizers by bits per byte on normalized text, never by token perplexity (D1).
- Vocabulary size trades softmax cost against compression; D3 gives the optimum where the compression elasticity equals the softmax share.
- Extending a trained model's vocabulary needs informed initialization of the new rows and diagnostics at step 0.
- The R2 tokenizer (32,768 byte-level BPE, single-digit regex, EOS id 0) and the `uint16` shard format with `.idx` offsets are the contract between this chapter and Chapter 4.

#### Exercises

1. **Check.** Two tokenizers give an LM the same bits per byte on a text but different per-token perplexities. Which has the larger bytes per token and by what factor does ln-perplexity differ?
   *Sketch:* bpb = L/(b̄ ln 2), so L₁/b̄₁ = L₂/b̄₂ and L₁/L₂ = b̄₁/b̄₂; the one with larger b̄ has larger L (in nats per token) by that ratio.
2. **Check.** Why must `viterbi_segment` keep every single character in the piece set? What does byte fallback add?
   *Sketch:* so that every string has at least one segmentation (P(s) > 0); byte fallback covers characters never seen in training by their bytes, giving coverage without `<unk>`.
3. **Derive.** Show that the EM update of D2 never decreases the log-likelihood of the corpus.
   *Sketch:* standard EM argument: log P(s) = Q(θ|θ_t) + H(θ|θ_t) with H maximized at θ = θ_t by Gibbs' inequality, and the M-step maximizes Q.
4. **Derive.** Generalize D3 to untied input and output embeddings (parameters 2Vd, compute still 6Vd per token) and to a model whose non-embedding compute per token is A + a·T for context length T. Find the new optimality condition.
   *Sketch:* if only the output matrix costs FLOP, the compute is unchanged (6Vd), while parameter memory doubles; with an attention term, A is replaced by A + aT and the same elasticity condition holds with the new A; discuss what the untied matrix changes (memory, not FLOP).
5. **Measure.** Predict, then measure, the Hindi premium (tokens per passage relative to English) of the three α-weighted BPE tokenizers, and the break-even α at which Hindi's premium equals Arabic's. Plot.
6. **Measure.** Repeat the extension experiment with 1,024, 4,096 and 8,192 new Hindi tokens (decomposition init). Predict where the Hindi bits per byte after 150 steps stops improving and what happens to English bpb.
7. **Open.** Replace the added-token mechanism by an edit of the BPE merge list (new merges before the old ones for pure-Devanagari pieces) and show that English segmentation is unchanged; compare the two extensions on Hindi bytes per token and bpb. Or implement SuperBPE-style multiword merges by training BPE on a pre-tokenizer that does not split on spaces after a first stage, and report compression and the effect on a small LM.

#### Further reading

- Kudo and Richardson, [SentencePiece](https://aclanthology.org/D18-2012/); Kudo, [Subword Regularization](https://aclanthology.org/P18-1007/); Sennrich et al., [BPE for NMT](https://aclanthology.org/P16-1162/).
- Tao et al., [Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623): optimal vocabulary size as a function of compute.
- Petrov et al., [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425); Ahia et al., [Do All Languages Cost the Same?](https://arxiv.org/abs/2305.13707); Rust et al., [How Good is Your Tokenizer?](https://arxiv.org/abs/2012.15613); Bostrom and Durrett, [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://arxiv.org/abs/2004.03720); Ali et al., [Tokenizer Choice for LLM Training: Negligible or Crucial?](https://arxiv.org/abs/2310.08754).
- Minixhofer et al., [WECHSEL](https://aclanthology.org/2022.naacl-main.293/); Dobler and de Melo, [FOCUS](https://arxiv.org/abs/2305.14481): embedding initialization for new vocabularies.
- Liu et al., [SuperBPE](https://arxiv.org/abs/2503.13423) and its [project page](https://superbpe.github.io/); Pagnoni et al., [Byte Latent Transformer](https://arxiv.org/abs/2412.09871).
- Tutorials: [Hugging Face tokenizers pipeline](https://huggingface.co/docs/tokenizers/pipeline) and [tokenization algorithms summary](https://huggingface.co/docs/transformers/tokenizer_summary); CS336's tokenization lecture; Karpathy's minbpe video.
- Code: [minbpe](https://github.com/karpathy/minbpe), [SentencePiece](https://github.com/google/sentencepiece), [tokenizers](https://huggingface.co/docs/tokenizers/index).

#### Risks and open decisions

- **R2 tokenizer choice is a one-way door.** Every R2 checkpoint, shard and downstream notebook in Chapters 4, 6, 11, 12 is tied to it. Decided 2026-10-28: V = 32,768, the single-digit regex, and 32 special ids (four named, 28 reserved). Rejected alternative: reuse GPT-2 (`d2l.BPETokenizer.from_tiktoken('gpt2')`, vocabulary 50,257, and FineWeb's own GPT-2 token counts) with no tokenizer training at all; it costs a 53% larger embedding for a 150M model and worse digit and whitespace handling.
- **Forward reference.** 3.4 and 3.5 load `lm-r2-tokenizer` before 3.6 teaches it. Mitigation: 3.6's notebook reproduces the tokenizer hash; if the editor prefers, move the tokenizer-training cell into a short shared "setup" notebook.
- **uint16 range.** Valid only for V ≤ 65,536; the writer picks `uint16` below that and `uint32` otherwise (e.g. Qwen's 151,936 in 4.5), recording it in `format.dtype`; Chapter 4 relies on both this rule and EOS (id 0) after every document.
- **Hosting size.** The 4.0·10^9-token shard set is ≈ 8 GB; if the D2L bucket cannot host it, ship only the manifest and the build script and let T2 users regenerate (hours of tokenization; to be measured), or host 10^9 tokens.
- **SentencePiece on NumPy 2 / Python versions**: wheels exist (0.2.2 verified on PyPI); fallback is the pure-Python `UnigramTokenizer`.
- **Belebele passage count** (488 distinct) is from the dataset's design; verify at authoring and use whatever the count is.
- **Token-extension method.** `add_tokens` is simple but its segmentation is not BPE-consistent; the merge-edit variant is an exercise. If the editor wants the cleaner method in the main text, budget +40 lines.
- **Licenses.** CC-BY-SA-4.0 (Belebele) and ODC-By (FineWeb-2, FineWeb-Edu) travel with derived artifacts, which ship with attribution on their dataset cards; the shard manifest records source licenses; fastText LID licensing is in 3.1.
- **The tokenizer-training slice** (200,000 train-bucket documents) is also training data of R2; no leakage into validation because of the bucket rule.
