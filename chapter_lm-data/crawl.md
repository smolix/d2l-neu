# Document Extraction from Web Crawls
:label:`sec_lm_crawl`

<!-- Author brief, copied verbatim from the Language Models part plan
(docs/llm/language_models_part_plan.md, "3.1 Document Extraction from Web Crawls · T0").
Blueprint: docs/llm/blueprints/ch03_data.md section 3.1.
The plan is a specification, not book text; this comment is replaced
when the section is written.

**Idea:** Extraction decides what the model can ever learn.
**Opening:** One crawl record holds a navigation menu, a cookie banner and an article. Two extractors render it differently: one keeps the article and loses the code block inside it, the other keeps the code and half the menu. The section follows the record from the archive format to a documented, normalized document, measures what each extractor loses by page class, and closes on the question of what "quality" should mean once the text is clean.
**Teaches:**
- WARC and WET records (`d2l.iter_warc` over a pinned byte range of one archive file); why WET is not the text of the web (no main-content detection, a different record set); charset detection before decoding.
- HTML extraction (a from-scratch link-density extractor against two libraries) and how it fails on code, tables and mathematics; the FineWeb text of the same records as a reference, reported as agreement, never as accuracy.
- Language identification with a small released classifier and what a score is not; Unicode normalization (NFC) and what NFKC silently rewrites in code and mathematics.
- Personal data scrubbing, provenance, licences and opt-outs as engineering inputs; a manifest with record ids, offsets, extractor versions, text hashes and a transformation log per document (`d2l.CorpusManifest`, with a dataset card generated from it); FineWeb-2 as the multilingual pipeline.
**Derivation:** none.
**Notebook:**
1. Fetch the pinned 256 MiB range of one Common Crawl archive file (`d2l.fetch_warc_range`), verify its per-block SHA-256, and iterate the records; count responses, requests and metadata records and the truncated final member.
2. Keep English (score at least 0.65), HTTP 200, `text/html` responses; extract with the naive extractor, trafilatura and Resiliparse, and join WET conversions by record id; apply NFC and the PII scrubber; log every transformation.
3. Classify pages by structural markers (code, table, mathematics, prose-only) and compute per-document recall of each class's content against the FineWeb text of the same record, plus a precision proxy for boilerplate.
4. Plot recall by class and extractor with a paired bootstrap clustered by registered domain; write the manifest and rebuild the documents from it (`d2l.rebuild_c0`) to show the hash round trip.
5. Claim that survives noise: extractor disagreement is concentrated in code and table pages, with prose recall alike across extractors; classes with fewer than 30 documents are marked and carry no interval. Extractors are deterministic, so the bootstrap over documents is the whole noise statement.
6. Reused pieces: `d2l.download`, `d2l.paired_bootstrap`; the byte view of `sec_text-sequence` for the Unicode paragraph; new `d2l.iter_warc`, `d2l.extract_text`, `d2l.detect_lang`, `d2l.scrub_pii`, `d2l.Doc`, `d2l.write_docs`, `d2l.read_docs`, `d2l.CorpusManifest`, `d2l.docs_hash`, `d2l.fetch_warc_range`, `d2l.rebuild_c0`.
**Figures:** `fig_lm_crawl_pipeline` (one record through the stages, with the manifest row it leaves behind at each stage).
**Restate / forward:** `sec_text-sequence` (a tokenizer sees bytes, so the normalization chosen here decides which byte sequences exist). Forward: 3.2 (the filter that decides which documents stay).
**Exercises:** 5 to 7. Load-bearing: measure extraction loss on one page class of your choice against a hand-labelled sample of 20 pages (measure); derive the expected number of documents per language under an identifier with a stated error matrix and a threshold, and the threshold that maximizes retained English documents at a target purity (derive); check that the manifest hash is unchanged by re-ordering the input records and changed by one edited byte (check); open: draft the dataset card for the extracted documents from the manifest alone and list what a reader cannot learn from it.
**Writer notes:**
- The pin: one archive file of the `CC-MAIN-2025-05` crawl (URL in blueprint ch03 3.1), bytes 0 to 268,435,455, with per-16 MiB block hashes so a shorter prefix verifies; the first 4 MiB held 285 complete responses at blueprint time, so the range holds on the order of 18,000 responses (to be measured). The SHA-256 is measured by one download at authoring. `D2L_OFFLINE=1` uses only a verified local copy and stops with fetch instructions if it is missing.
- Exports with contracts: `d2l.iter_warc(source, n_bytes=None, types=('response',))` yields exactly the records `warcio.ArchiveIterator` yields for the same bytes (the oracle test) and drops a truncated final member with a count; `d2l.extract_text(html, method='trafilatura'|'resiliparse'|'naive', url=None)`; `d2l.detect_lang(text, k=1)` wraps the 176-language fastText model with newlines replaced first; `d2l.CorpusManifest.add_stage(name, params, n_in, n_out, drops, in_hash, out_hash)`, `.canonical_json()`, `.sha256()`, `.to_card()`; `d2l.docs_hash(docs)` is the SHA-256 over sorted `(id, sha256(text))`, order-independent; `d2l.rebuild_c0` returns `.mismatches` on extractor drift, never silently.
- Silent breakages: decoding bodies as UTF-8 without the `Content-Type` charset, the `<meta charset>` and a detector in that order (log the choice); counting request and metadata records as documents; joining FineWeb by URL instead of record id (the same URL is crawled many times; confirm on the prefix that FineWeb's `id` equals the WARC record id, else join on normalized URL plus date and report it as weaker); timestamps or absolute paths in the manifest, which break its hash; NFKC "to clean up".
- Packages and wheels: `warcio`, `trafilatura` (pin the exact version in the manifest; extractor drift changes every text hash), `resiliparse` (wheels may be missing on macOS ARM build hosts: run the other two and mark its rows as from the committed table), `fasttext` (does not build cleanly against NumPy 2; use the NumPy-2 wheel or `lingua-language-detector` and say so); the 217-language fastText model is CC BY-NC and is only an aside.
- Reference: the FineWeb and FineWeb-Edu configs of the same crawl, streamed with `datasets` and stopped when the `file_path` changes; rows inside a file are ordered by URL, not offset.
- Name FineWeb-2 as the multilingual corpus; keep the manifest and the rights rule; the section stays T0.
- Committed artifacts: `lm-cc-warc-256m` (the pin), `lm-c0-docs` (manifest rows with ids, offsets, hashes and logs; no text), the stage manifest, the aggregated metric table as JSON.
- Do not re-teach: tokenization (3.6 owns the rest), the byte view beyond one paragraph.
**Sources:**
- Papers: [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) (WARC with trafilatura against WET, the language threshold, PII; the pipeline this section mirrors); [FineWeb2: One Pipeline to Scale Them All](https://arxiv.org/abs/2506.20920) (the multilingual counterpart, per-language thresholds); [Quality at a Glance](https://arxiv.org/abs/2103.12028) (an audit of language labels in crawled multilingual data); [Documenting Large Webtext Corpora](https://arxiv.org/abs/2104.08758) (the audit method the dataset-card exercise follows); [Consent in Crisis](https://arxiv.org/abs/2407.14933) (opt-outs and the shrinking crawlable web); [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (the documentation standard).
- Code and documentation: [Common Crawl, get started](https://commoncrawl.org/get-started) (access and record formats); [warcio](https://github.com/webrecorder/warcio) (the iterator oracle); [Trafilatura documentation](https://trafilatura.readthedocs.io/) (extraction options and its evaluation page); [fastText language identification](https://fasttext.cc/docs/en/language-identification.html) (the 176-language model and its licence); [Unicode Standard Annex 15](https://www.unicode.org/reports/tr15/) (NFC and NFKC, with test cases); [Datatrove](https://github.com/huggingface/datatrove) (readers and extraction stages); [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards).
- Courses, tutorials and blogs: [CS336](https://cs336.stanford.edu/) (data sources lecture and assignment A4); [FineWeb blog post](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) (the extraction ablation).
-->

<!-- Placeholder (stage 2, cells step): the cells follow the section
specification's cell table (S/spec/20.1.md part 3); one-line comments mark
where prose goes. Rights rule P1: no cell prints text that we extracted from
the crawl (HTML, WET text, extractor output); displayed text is FineWeb's
(ODC-By) for the same record id, everything else is counts, code points,
tag skeletons and hashes. -->

<!-- Prose: the opening (one record holds a menu, a cookie banner and an article); the question of the section. -->

```{.python .input #crawl-imports}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #crawl-extras}
%%tab pytorch
import collections
import concurrent.futures
import gzip
import hashlib
import importlib.metadata
import io
import itertools
import json
import multiprocessing
import os
import pathlib
import re
import unicodedata
import warnings
import pandas as pd
from warcio.archiveiterator import ArchiveIterator
```

<!-- Prose: what the notebook computes and how long it takes; every result is recomputed in each build (the steps are deterministic), and D2L_RERUN only selects which files are written. -->

```{.python .input #crawl-config}
%%tab pytorch
CRAWL_MIB = 256  # bytes of the pinned WARC range read (multiple of 16)
LID_MIN = 0.65   # FineWeb's English threshold on the fastText score
N_WORKERS = min(8, os.cpu_count())
OFFLINE = os.environ.get('D2L_OFFLINE') == '1'  # only verified local bytes
ARTIFACTS = {  # name: (registered key, file a regeneration writes)
    'pin': ('lm-cc-warc-256m-v1', 'lm-cc-warc-256m-v1.pin.json'),
    'c0': ('lm-c0-docs-v1', 'lm-c0-docs-v1.jsonl.gz'),
    'crawl-results': ('lm-data-crawl-results-v1',
                      'lm-data-crawl-results-v1.json')}
RERUN = d2l.lm_rerun(ARTIFACTS)
print(f'CRAWL_MIB={CRAWL_MIB}, LID_MIN={LID_MIN}, N_WORKERS={N_WORKERS}, '
      f'OFFLINE={OFFLINE}, writing: {", ".join(sorted(RERUN)) or "none"}')
```

<!-- Prose: the registry; the crawl range is a pin (URLs and hashes), never a copy of Common Crawl bytes. -->

<!-- Pending registrations (cells step, 2026-11-15): `lm-data-fineweb-ref-v1` (builder, W2; read by the opening, fineweb-agree and lid cells), `lm-c0-docs-v1` and `lm-data-crawl-results-v1` (regeneration `c0,crawl-results`, W6) join this cell when uploaded; until then a normal build stops at the opening cell with a KeyError. -->

```{.python .input #crawl-data-hub}
%%tab pytorch
#@save
d2l.DATA_HUB['lm-cc-warc-256m-v1'] = (  # pin only: URLs, sizes, SHA-256s
    d2l.LM_DATA_URL + 'lm-cc-warc-256m-v1.pin.json',
    '9444225da3dd569d00b801ad9ae1455431dc1b01')
d2l.DATA_HUB['lid.176.ftz'] = (  # fastText, 176 languages, CC BY-SA 3.0
    'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz',
    '86d1b630ba55a5040231eda9fe24a7befdc411f2')
```

```{.python .input #crawl-sources}
%%tab pytorch
#@save
LM_DATA_SOURCES = {  # name: (kind, Hub repository, revision, licence)
    'fineweb': ('dataset', 'HuggingFaceFW/fineweb',
                '9bb295ddab0e05d785b879661af7260fed5140fc', 'odc-by'),
    'fineweb-edu': ('dataset', 'HuggingFaceFW/fineweb-edu',
                    '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9', 'odc-by'),
    'fineweb-2': ('dataset', 'HuggingFaceFW/fineweb-2',
                  'af9c13333eb981300149d5ca60a8e9d659b276b9', 'odc-by'),
    'edu-annotations': ('dataset', 'HuggingFaceFW/fineweb-edu-llama3-annotations',
                        '72df4c92fb1b48beceb16016e8f695ec40a6c3a5', 'odc-by'),
    'finemath': ('dataset', 'HuggingFaceTB/finemath',
                 'e92b25a616738fe95dc186b64dfb19f9c8525594', 'odc-by'),
    'codeparrot-clean': ('dataset', 'codeparrot/codeparrot-clean',
                         '35a59fb025bc0a102f7d96eac09d145b896d487b', 'per-file'),
    'belebele': ('dataset', 'facebook/belebele',
                 '7899cdfa4e1e0d733fd77c848e2c273cb1d32be2', 'cc-by-sa-4.0'),
    'cosmopedia': ('dataset', 'HuggingFaceTB/cosmopedia',
                   '0ae6ec63f91742bd2d1eaef4f02232c55d719385', 'apache-2.0'),
    'mmlu': ('dataset', 'cais/mmlu',
             'c30699e8356da336a370243923dbaf21066bb9fe', 'mit'),
    'fineweb-edu-classifier': ('model', 'HuggingFaceFW/fineweb-edu-classifier',
                               '284663cbb2dabf9bda30d8f8cc49601251ee1631',
                               'apache-2.0')}
```

## What a Crawl Contains
:label:`subsec_lm_crawl_warc`

<!-- Prose: WARC record types (warcinfo, request, response, metadata), WAT and WET; one gzip member per record, so a byte prefix is a valid sample and Range requests are legal. -->

```{.python .input #crawl-warc-def}
%%tab pytorch
#@save
WarcRecord = collections.namedtuple(
    'WarcRecord', 'id type url date status mime offset length headers body')

#@save
def iter_warc(source, n_bytes=None, types=('response',), *, stats=None):
    """Yield the WARC records of the given types (None: all) from a local
    path or a URL, reading only the first n_bytes (a Range request for a
    URL). Each gzip member is decompressed on its own; offset and length
    are the member's compressed byte range. A member cut off by the end of
    the input is not yielded: it is counted in stats['truncated'], beside
    one count per record type (stats is a dict the caller passes)."""
    for offset, length, raw in d2l._gzip_members(
            d2l._warc_chunks(source, n_bytes)):
        if raw is None:
            if stats is not None:
                stats['truncated'] = stats.get('truncated', 0) + 1
            continue
        rec = d2l._warc_parse(raw, offset, length)
        if stats is not None:
            stats[rec.type] = stats.get(rec.type, 0) + 1
        if types is None or rec.type in types:
            yield rec
```

<!-- Prose: the pinned range, fetched block by block and checked against the pin; D2L_OFFLINE. -->

```{.python .input #crawl-fetch-def}
%%tab pytorch
#@save
def fetch_warc_range(pin='lm-cc-warc-256m-v1', n_bytes=None, cache_dir=None,
                     *, which='warc'):
    """Local path of the pinned Common Crawl bytes: which='warc' gives the
    first n_bytes of the pinned range (a multiple of the 16 MiB block; None
    for the whole range), fetched with Range requests and checked block by
    block against the pin's SHA-256s; which='wet' gives the whole WET file
    of the same segment, checked as one file. With D2L_OFFLINE=1 only a
    verified local copy is used. The bytes stay in cache_dir (default
    ../data/lm/cc); they are never committed or uploaded."""
    import pathlib
    with open(d2l.download_keyed(pin)) as f:
        spec = json.load(f)
    cache = pathlib.Path(cache_dir or os.path.join('..', 'data', 'lm', 'cc'))
    cache.mkdir(parents=True, exist_ok=True)
    offline = os.environ.get('D2L_OFFLINE') == '1'
    if which == 'wet':
        if n_bytes is not None:
            raise ValueError('the WET file is fetched and checked whole')
        return d2l._fetch_whole(spec['wet'], cache, offline)
    if which != 'warc':
        raise ValueError(f"which is 'warc' or 'wet', not {which!r}")
    warc, block = spec['warc'], spec['warc']['block_bytes']
    n_bytes = warc['n_bytes'] if n_bytes is None else n_bytes
    if n_bytes % block or not 0 < n_bytes <= warc['n_bytes']:
        raise ValueError(f'n_bytes must be a multiple of {block} up to '
                         f'{warc["n_bytes"]}')
    path = cache / f'{pin}.warc.gz'
    have = d2l._verified_blocks(path, warc['block_sha256'], block,
                                n_bytes // block)
    if have * block < n_bytes:
        if offline:
            raise RuntimeError(
                f'D2L_OFFLINE=1 and {path} holds {have} verified blocks of '
                f'{n_bytes // block}: run once with network access, or fetch '
                f'bytes 0-{n_bytes - 1} of {warc["url"]} into {path}')
        d2l._fetch_blocks(warc, path, have, n_bytes // block)
    return path
```

<!-- Prose: the opening record, three ways: its HTML as a tag skeleton, the WET text as counts, FineWeb's text for the same record id. -->

```{.python .input #crawl-opening}
%%tab pytorch
path = d2l.fetch_warc_range(n_bytes=16 * 2**20)  # block 0, SHA-256 checked
head_stats = {}
head = list(d2l.iter_warc(path, n_bytes=4 * 2**20, types=None,
                          stats=head_stats))
print('records in the first 4 MiB:', head_stats)
fineweb = pd.read_parquet(d2l.download_keyed('lm-data-fineweb-ref-v1')
                          ).set_index('id')

def pick(records):
    """First HTTP-200 HTML response with a <pre> or <table> element, at
    least 2,000 characters of trafilatura text and a FineWeb row."""
    for r in records:
        if (r.type, r.status, r.mime) != ('response', 200, 'text/html') or \
                r.id not in fineweb.index:
            continue
        page = d2l.decode_body(r.body, r.headers)[0]
        text = (d2l.extract_text(page, url=r.url)
                if re.search(r'<(pre|table)\b', page, re.I) else None)
        if text and len(text) >= 2000:
            return r, page, text

found, where = pick(head), 'the first 4 MiB'
if found is None:
    found, where = (pick(d2l.iter_warc(path, n_bytes=16 * 2**20)),
                    'the first 16 MiB block')
record, page, ours_text = found
print(f'rule: first HTTP-200 text/html response with <pre> or <table>, >= 2,000 '
      f'characters of trafilatura text and a FineWeb row; found in {where}')
print(f'record {record.id}, {record.url}\n')
print(f'(1) HTML: {len(record.body):,} bytes; first 25 elements of <body>:')
print('\n'.join(d2l.html_skeleton(page, 25)))
wet_record = next(r for r in d2l.iter_warc(d2l.fetch_warc_range(which='wet'),
                                    types=('conversion',))
           if r.headers.get('WARC-Refers-To') == record.id)
lines = wet_record.body.decode('utf-8').splitlines()
short = [len(l.strip()) < 30 for l in lines]
run = max(len(list(g)) for k, g in itertools.groupby(short) if k) \
    if any(short) else 0
print(f'\n(2) WET text: {sum(map(len, lines)):,} characters in {len(lines)} '
      f'lines, {np.mean(short):.0%} of them under 30 characters, longest run '
      f'of short lines (a menu signature): {run}')
fw = fineweb.loc[record.id, 'text']
mask = lambda t: re.sub(r'\S+@\S+|\b(?:\d{1,3}\.){3}\d{1,3}\b', '<PII>',
                        t or '')  # FineWeb replaced emails and IPv4s too
f1 = d2l.char_ngram_f1(mask(ours_text), mask(fw))[2]
print(f'\n(3) FineWeb text: {len(fw):,} characters; our trafilatura text: '
      f'{len(ours_text):,} characters, character-5-gram F1 against FineWeb {f1:.2f}')
print(f'first 300 characters of FineWeb\'s text:\n  {fw[:300]!r}')
print('(text: FineWeb, HuggingFaceFW/fineweb CC-MAIN-2025-05, ODC-By 1.0)')
```

<!-- Prose: iter_warc is checked against warcio, the reference reader, on the same bytes. -->

```{.python .input #crawl-warcio-oracle}
%%tab pytorch
with open(path, 'rb') as f:
    it = ArchiveIterator(io.BytesIO(f.read(4 * 2**20)))
    theirs = []
    try:
        for r in it:
            body = hashlib.sha256(r.raw_stream.read()).hexdigest()
            offset = it.get_record_offset()
            it.read_to_end(r)
            theirs.append((r.rec_headers.get_header('WARC-Record-ID'),
                           r.rec_type, offset, it.get_record_length(), body))
    except Exception:  # warcio may raise inside the cut-off member
        pass
mine = [(r.id, r.type, r.offset, r.length, hashlib.sha256(r.body).hexdigest())
        for r in head]
same = sum(a == b for a, b in zip(mine, theirs))
print(f'{same}/{len(mine)} records identical (id, type, offset, length, '
      f'body SHA-256); warcio also yields {len(theirs) - len(mine)} partial '
      f'record(s) from the cut-off member, which iter_warc counts as '
      f'truncated ({head_stats.get("truncated", 0)})')
assert same == len(mine) and len(theirs) - len(mine) == head_stats.get(
    'truncated', 0)
```

<!-- Prose: the whole range: record counts, the truncated member, responses by MIME type and status; the arithmetic for the whole file is an extrapolation. -->

```{.python .input #crawl-scan}
%%tab pytorch
path = d2l.fetch_warc_range(n_bytes=CRAWL_MIB * 2**20)
stats, index = {}, []
for r in d2l.iter_warc(path, n_bytes=CRAWL_MIB * 2**20, types=None,
                       stats=stats):
    index.append((r.id, r.type, r.url, r.date, r.status, r.mime, r.offset,
                  r.length, len(r.body)))
index = pd.DataFrame(index, columns=['id', 'type', 'url', 'date', 'status',
                                     'mime', 'offset', 'length', 'n_body'])
print(f'{CRAWL_MIB} MiB: {len(index):,} complete records, '
      f'{stats.get("truncated", 0)} truncated member at the cut')
print(index.type.value_counts().to_string(), '\n')
resp = index[index.type == 'response']
by_mime = resp.groupby(resp.mime.fillna('(none)')).agg(
    responses=('id', 'size'), status_200=('status', lambda s: (s == 200).sum()),
    median_body_bytes=('n_body', 'median'))
print(by_mime.sort_values('responses', ascending=False).head(6).to_string())
per_mib = len(resp) / CRAWL_MIB
print(f'\n{per_mib:.1f} responses per MiB; extrapolated to the whole '
      f'1,141,586,184-byte file: about {per_mib * 1141586184 / 2**20:,.0f} '
      f'responses (extrapolated, assumes the prefix is representative)')
print('\nfirst 12 records (type, offset, compressed length):')
print(index[['type', 'offset', 'length']].head(12).to_string(index=False))
```

### Charsets and Decoding
:label:`subsec_lm_crawl_charset`

<!-- Prose: the charset chain (Content-Type header, then <meta charset>, then a detector), logged per document; decode_body is in the library. -->

## From HTML to Text
:label:`subsec_lm_crawl_extract`

<!-- Prose: boilerplate; WET has no main-content detection; trafilatura scores the DOM; Resiliparse (DCLM's extractor); the naive link-density extractor written out below. -->

```{.python .input #crawl-extract-def}
%%tab pytorch
#@save
NAIVE_BLOCK_TAGS = {'p', 'div', 'li', 'td', 'th', 'tr', 'pre', 'h1', 'h2',
                    'h3', 'h4', 'h5', 'h6', 'section', 'article', 'main',
                    'blockquote', 'ul', 'ol', 'table', 'header', 'footer',
                    'nav', 'aside', 'form'}

#@save
def naive_blocks(page):
    """The text blocks of a page, split at block-level tags, as (text,
    characters of link text) pairs; a <pre> block keeps its lines.
    Comments, <head>, scripts and styles are removed first."""
    import html
    page = re.sub(r'<!--.*?-->|<(head|script|style|noscript|template)\b'
                  r'.*?</\1\s*>', ' ', page, flags=re.S | re.I)
    blocks, parts, in_a, pre = [], [], 0, 0  # parts: (string, in a link)

    def flush():
        raw = ''.join(s for s, _ in parts)
        text = ('\n'.join(l.rstrip() for l in raw.strip('\n').splitlines())
                if pre else ' '.join(raw.split()))
        links = sum(len(s.strip()) for s, link in parts if link)
        if text:
            blocks.append((text, min(links, len(text))))
        parts.clear()

    for m in re.finditer(r'<(/?)([A-Za-z][\w:-]*)[^>]*>|(<|[^<]+)', page):
        close, tag, data = m.group(1), (m.group(2) or '').lower(), m.group(3)
        if data is not None:
            parts.append((html.unescape(data), in_a > 0))
        elif tag in d2l.NAIVE_BLOCK_TAGS:
            flush()  # a block tag, opening or closing, ends a block
            pre = max(0, pre + (tag == 'pre') * (-1 if close else 1))
        elif tag == 'a':
            in_a = max(0, in_a + (-1 if close else 1))
        elif tag == 'br':
            parts.append(('\n', False))
    flush()
    return blocks

#@save
def extract_text(html, method='trafilatura', url=None):
    """The main text of a page, or None. 'naive': blocks with at least 40
    characters of which at most a third is link text, plus their
    neighbours within two blocks that have at least 15 characters and pass
    the same link-density test; 'trafilatura' and 'resiliparse': the two
    libraries' main-content extraction."""
    import logging
    if method == 'naive':
        blocks = d2l.naive_blocks(html)
        dense = [len(t) >= 15 and l <= 0.33 * len(t) for t, l in blocks]
        core = [i for i, (t, l) in enumerate(blocks)
                if len(t) >= 40 and dense[i]]
        keep = sorted({j for i in core for j in range(i - 2, i + 3)
                       if 0 <= j < len(blocks) and dense[j]})
        text = '\n'.join(blocks[j][0] for j in keep)
    elif method == 'trafilatura':
        import trafilatura
        logging.getLogger('trafilatura').setLevel(logging.CRITICAL)
        text = trafilatura.extract(html, url=url, include_tables=True,
                                   favor_precision=False)
    elif method == 'resiliparse':
        from resiliparse.extract.html2text import extract_plain_text
        text = extract_plain_text(html, main_content=True)
    else:
        raise ValueError(f'unknown method {method!r}')
    return text if text and text.strip() else None
```

<!-- Prose: five hand-built pages fix what the naive extractor keeps. -->

```{.python .input #crawl-naive-fixtures}
%%tab pytorch
#@save
NAIVE_FIXTURES = {  # name: (HTML, the naive extractor's expected output)
    'article with sidebar': (
        '<html><body><nav><a href="/">Home</a> <a href="/news">News</a> '
        '<a href="/about">About us</a></nav><article><h1>Rain returns to '
        'the valley</h1><p>After eleven dry months the first storm of the '
        'season reached the valley on Tuesday night.</p><p>Farmers said '
        'the rain came too late for the barley but in time for the '
        'vines.</p></article><aside><ul><li><a href="/a">Most read: the '
        'new bridge opens</a></li><li><a href="/b">Weather for the '
        'weekend ahead</a></li></ul></aside></body></html>',
        'Rain returns to the valley\nAfter eleven dry months the first '
        'storm of the season reached the valley on Tuesday night.\nFarmers '
        'said the rain came too late for the barley but in time for the '
        'vines.'),
    'forum': (
        '<html><body><div class="post"><div class="user"><a href="/u/7">'
        'mira_k</a></div><div class="msg">Has anyone managed to compile '
        'the driver on the new kernel? It fails at the linking step for '
        'me.</div></div><div class="post"><div class="user"><a href="/u/9">'
        'ole</a></div><div class="msg">Yes, you need the headers package '
        'for your exact kernel version, then rebuild.</div></div>'
        '</body></html>',
        'Has anyone managed to compile the driver on the new kernel? It '
        'fails at the linking step for me.\nYes, you need the headers '
        'package for your exact kernel version, then rebuild.'),
    'table': (
        '<html><body><p>Monthly rainfall at the station, in millimetres, '
        'for the first quarter:</p><table><tr><th>Month</th><th>Rain</th>'
        '</tr><tr><td>January</td><td>82</td></tr><tr><td>February</td>'
        '<td>61</td></tr></table></body></html>',
        'Monthly rainfall at the station, in millimetres, for the first '
        'quarter:'),  # the cells are too short: the table is lost
    'pre': (
        '<html><body><p>The loop below sums the first hundred integers '
        'and prints the result:</p><pre>total = 0\nfor i in range(100):\n'
        '    total += i\nprint(total)</pre></body></html>',
        'The loop below sums the first hundred integers and prints the '
        'result:\ntotal = 0\nfor i in range(100):\n    total += i\n'
        'print(total)'),
    'empty': ('<html><head><title>x</title></head><body></body></html>',
              None)}

for name, (page, want) in d2l.NAIVE_FIXTURES.items():
    got = d2l.extract_text(page, method='naive')
    print(f'{name:22s} {"as expected" if got == want else "DIFFERS"}')
    assert got == want, (name, got)
```

<!-- Prose: the run over every HTTP-200 HTML response of the range, three extractors per page in worker processes, WET joined by WARC-Refers-To; page classes come from markers in the HTML. -->

```{.python .input #crawl-extract}
%%tab pytorch
html_rows = index[(index.type == 'response') & (index.status == 200) &
                  (index.mime == 'text/html')]
jobs = [(str(path), int(o), int(l))
        for o, l in zip(html_rows.offset, html_rows.length)]
d2l.download_keyed('lid.176.ftz')  # once, before the workers start
with concurrent.futures.ProcessPoolExecutor(
        N_WORKERS, mp_context=multiprocessing.get_context('spawn')) as ex:
    pages = list(ex.map(d2l.extract_page, jobs, chunksize=200))
wet, cld2 = {}, {}  # WET text and Common Crawl's CLD2 label by record id
ids = {p['id'] for p in pages}
for r in d2l.iter_warc(d2l.fetch_warc_range(which='wet'),
                       types=('conversion',)):
    rid = r.headers.get('WARC-Refers-To')
    if rid in ids:
        wet[rid] = r.body.decode('utf-8')
        cld2[rid] = r.headers.get('WARC-Identified-Content-Language', '')
for p in pages:
    p['texts']['wet'] = wet.get(p['id'])
EXTRACTORS = ['wet', 'naive', 'trafilatura', 'resiliparse']
CLASSES = ['prose', 'code', 'table', 'math']
sources = collections.Counter(p['charset_source'] for p in pages)
print(f'{len(pages):,} HTTP-200 HTML responses; charset from: '
      + ', '.join(f'{k} {v:,}' for k, v in sources.most_common()))
print('pages per class (a page may be in several):',
      {c: sum(c in p['struct']['classes'] for p in pages) for c in CLASSES})
print(pd.DataFrame({
    'empty output': [np.mean([not p['texts'][m] for p in pages])
                     for m in EXTRACTORS],
    'documents/s per core': [None] + [len(pages) / sum(
        p['secs'][m] for p in pages) for m in EXTRACTORS[1:]]},
    index=EXTRACTORS).round(3).to_string())
print(f'(WET has no conversion record for {len(pages) - len(wet):,} pages)')
```

### Code, Tables and Mathematics
:label:`subsec_lm_crawl_structure`

<!-- Prose: the metrics, defined from the HTML alone so that no extractor is the reference: code-line, table-cell and math recall, prose 5-gram recall, and a precision proxy for boilerplate. Say plainly that the definitions (a prose paragraph has at least 100 characters outside nav/header/footer/aside/form, a prose page at least 200 such characters, the precision proxy's content set) were set by looking at pages of this same sample, so they are not held out; the naive extractor's thresholds (40, a third, 15, two blocks) come from the plan. The precision proxy matches a line anywhere in the content text, so short menu lines can count as content: it flatters WET and the naive extractor, which only strengthens their low scores. -->

```{.python .input #crawl-metrics-def}
%%tab pytorch
#@save
def structure_recall(html, text):
    """Recall of a page's structural content in an extracted text: the
    share of code lines, table cells and mathematics strings found in it,
    the character-5-gram recall of the <p> text ('prose'), and a precision
    proxy, the share of the text's characters on lines found in the page's
    content elements. html is a page or its html_structure; a class absent
    from the page gives None, and an empty text recalls nothing."""
    s = html if isinstance(html, dict) else d2l.html_structure(html)
    text = text or ''
    flat, nows = d2l._ws(text), d2l._nows(text)
    out = {}
    out['code'] = (np.mean([l in flat for l in s['code']])
                   if s['code'] else None)
    for c in ('table', 'math'):
        out[c] = np.mean([x in nows for x in s[c]]) if s[c] else None
    grams = lambda t: {t[i:i + 5] for i in range(len(t) - 4)}
    P = grams(s['prose'].replace('\n', ''))
    out['prose'] = len(P & grams(nows)) / len(P) if P else None
    lines = [d2l._nows(l) for l in text.splitlines()]
    total = sum(map(len, lines))
    out['precision'] = (sum(len(l) for l in lines if l in s['content'])
                        / total if total else None)
    return {k: (None if v is None else float(v)) for k, v in out.items()}
```

<!-- Prose: the headline measurement: recall by page class and extractor, with intervals from a bootstrap over registered domains (one site's templated pages count once). The pages are chosen without any compared extractor (fastText on the HTML's content text, the same 0.65 threshold), and a missing or empty output scores 0 for every extractor; choosing them by language of trafilatura's text, as the c0 funnel does, would drop exactly the pages trafilatura fails on. Precision is compared on the pages where all four outputs are non-empty. -->

```{.python .input #crawl-recall}
%%tab pytorch
ev = [p for p in pages if p['html_lang'][0] == 'en'  # language of the HTML,
      and p['html_lang'][1] >= LID_MIN]             # not of any extractor
rows = []
for p in ev:
    dom = d2l.registered_domain(p['url'])
    for m in EXTRACTORS:  # a missing or empty output recalls nothing
        rows.append(dict(id=p['id'], domain=dom, extractor=m,
                         classes=p['struct']['classes'],
                         **d2l.structure_recall(p['struct'], p['texts'][m])))
rec = pd.DataFrame(rows)
in_c0 = sum(p['c0']['drop'] is None for p in ev)
print(f'{len(ev):,} pages English by their HTML ({in_c0:,} of them in c0); '
      'missing or empty output, scored 0: ' + ', '.join(
          f'{m} {sum(not p["texts"][m] for p in ev)}' for m in EXTRACTORS))

def interval(values, cluster, paired=None):
    """Mean with a 95% domain-clustered interval, or None (n too small)."""
    if len(values) < 30 or len(set(cluster)) < 30:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if paired is None:
            return d2l.bootstrap_ci(values, cluster=cluster)
        return d2l.paired_bootstrap(values, paired, cluster=cluster)

recall, deltas = [], []
for c in CLASSES:
    sub = rec[rec.classes.map(lambda cl: c in cl)]
    wide = sub.pivot(index='id', columns='extractor', values=c)
    doms = sub.drop_duplicates('id').set_index('id').domain
    for m in EXTRACTORS:
        v = wide[m].dropna()
        ci = interval(v.to_numpy(), doms[v.index].to_numpy())
        recall.append(dict(cls=c, extractor=m, n=len(v),
                           domains=doms[v.index].nunique(), mean=v.mean(),
                           lo=ci.ci[0] if ci else np.nan,
                           hi=ci.ci[1] if ci else np.nan))
    for a, b in [('trafilatura', 'resiliparse'), ('naive', 'trafilatura')]:
        both = wide[[a, b]].dropna()
        d = interval(both[a].to_numpy(), doms[both.index].to_numpy(),
                     paired=both[b].to_numpy())
        deltas.append(dict(cls=c, pair=f'{a} - {b}', n=len(both),
                           diff=both[a].mean() - both[b].mean(),
                           lo=d.ci[0] if d else np.nan,
                           hi=d.ci[1] if d else np.nan,
                           p=d.p_two_sided if d else np.nan))
recall, deltas = pd.DataFrame(recall), pd.DataFrame(deltas)
full = {p['id'] for p in ev  # pages where every extractor returned text
        if all(p['texts'][m] and p['texts'][m].strip() for m in EXTRACTORS)}
prec = []
for m in EXTRACTORS:
    v = rec[(rec.extractor == m) & rec.id.isin(full)].dropna(
        subset=['precision'])
    ci = interval(v.precision.to_numpy(), v.domain.to_numpy())
    prec.append(dict(extractor=m, n=len(v), mean=ci.mean, lo=ci.ci[0],
                     hi=ci.ci[1]))
prec = pd.DataFrame(prec)

fig, axes = d2l.plt.subplots(1, 2, figsize=(9, 3.4),
                             gridspec_kw=dict(width_ratios=[2, 1]))
shown = recall.assign(lo=recall.lo.fillna(recall['mean']),  # n too small:
                      hi=recall.hi.fillna(recall['mean']))  # no interval
ticks = []
for c in CLASSES:
    r = recall[(recall.cls == c) & (recall.extractor == 'trafilatura')]
    ticks.append(f'{c} (n={r.n.iloc[0]}'
                 + (', too small)' if r.lo.isna().iloc[0] else ')'))
d2l.plot_intervals(ticks, {m: tuple(
    shown[shown.extractor == m].set_index('cls').loc[CLASSES, k].to_numpy()
    for k in ('mean', 'lo', 'hi')) for m in EXTRACTORS},
    'page class', 'recall', axes=axes[0])
d2l.plot_intervals(EXTRACTORS, {'precision proxy': (prec['mean'], prec.lo,
                                                    prec.hi)},
                   'extractor', 'precision proxy', axes=axes[1])
fig.tight_layout()
print(deltas.round(3).to_string(index=False))
```

<!-- Prose: agreement with FineWeb's text for the same records; agreement, never accuracy. FineWeb's text comes from datatrove's Trafilatura step, whose defaults (checked in the datatrove 0.10.1 source) are favor_precision=True, deduplicate=True, include_comments=False; ours uses favor_precision=False and trafilatura's own defaults (comments kept), so even our trafilatura text is not FineWeb's. FineWeb then replaced emails and public IPv4 addresses by fixed placeholders (datatrove's PIIFormatter: email@example.com, firstname.lastname@example.org, six fixed IPs); both sides are masked alike before comparing, so a scrubbed address does not count as a lost line. A page that the WET file lacks counts as empty. -->

```{.python .input #crawl-fineweb-agree}
%%tab pytorch
by_id = {p['id']: p for p in pages}
in_range = fineweb[fineweb.url.isin(set(resp.url))]  # rows of the scanned part
ref, how = in_range.text[in_range.index.isin(list(by_id))], 'record id'
if len(ref) < 0.99 * len(in_range):  # fallback, a weaker match
    key = {(p['url'], p['c0']['row']['date']): p['id'] for p in pages}
    ref = pd.Series({key[(u, d)]: t for u, d, t in zip(
        in_range.url, in_range.date, in_range.text) if (u, d) in key})
    how = 'URL and date (weaker)'
joined = sorted(ref.index)  # our record ids of the pages FineWeb kept
print(f'{len(joined):,} of {len(in_range):,} FineWeb rows of this range '
      f'joined by {how}')
agree, masked = [], {}
for i in joined:
    p = by_id[i]
    cls = p['struct']['classes'] or ['none']
    for m in EXTRACTORS:  # a missing output (no WET record) is empty
        masked[i, m] = mask(p['texts'][m])
        prec_, rec_, f1 = d2l.char_ngram_f1(masked[i, m], mask(ref[i]))
        agree.append(dict(id=i, domain=d2l.registered_domain(p['url']),
                          extractor=m, cls=cls, f1=f1,
                          fineweb_only=1 - rec_, ours_only=1 - prec_))
agree = pd.DataFrame(agree)
summary = []
for m in EXTRACTORS:
    v = agree[agree.extractor == m]
    ci = interval(v.f1.to_numpy(), v.domain.to_numpy())
    summary.append(dict(extractor=m, n=len(v), f1=ci.mean, lo=ci.ci[0],
                        hi=ci.ci[1]))
summary = pd.DataFrame(summary)
print(summary.round(3).to_string(index=False))
split = (agree.explode('cls').groupby(['extractor', 'cls'])
         [['fineweb_only', 'ours_only']].mean().unstack('cls'))
print('\nshare of 5-grams only in FineWeb / only in ours, by page class:')
print(split.reindex(EXTRACTORS).round(2).to_string())
print('\nlines FineWeb kept and an extractor lost (FineWeb\'s text, ODC-By; '
      'addresses masked on both sides):')
for m in ['naive', 'resiliparse', 'trafilatura']:
    for i in joined:
        if not {'code', 'table'} & set(by_id[i]['struct']['classes']):
            continue
        out = d2l._ws(masked[i, m])
        lost = [l for l in mask(ref[i]).splitlines()
                if len(l) >= 40 and d2l._ws(l) not in out]
        if lost:
            print(f'  {m}: {lost[0][:160]!r}')
            break
print('(text: FineWeb, HuggingFaceFW/fineweb CC-MAIN-2025-05, ODC-By 1.0)')
```

## Language Identification
:label:`subsec_lm_crawl_lid`

<!-- Prose: fastText lid.176 is a linear model on hashed character n-grams; its score is a softmax over 176 labels, not the probability that a page is English text; FineWeb keeps English at 0.65. -->

```{.python .input #crawl-lid-def}
%%tab pytorch
#@save
_LID_MODEL = []  # the fastText model, loaded once per process

#@save
def detect_lang(text, k=1):
    """The top k (language, score) pairs of fastText's 176-language
    identifier (lid.176.ftz) for a text; newlines become spaces first, as
    fastText reads one line."""
    if not d2l._LID_MODEL:
        import fasttext
        fasttext.FastText.eprint = lambda *args, **kwargs: None  # quiet load
        d2l._LID_MODEL.append(fasttext.load_model(
            str(d2l.download_keyed('lid.176.ftz'))))
    labels, scores = d2l._LID_MODEL[0].predict(text.replace('\n', ' '), k=k)
    return [(l.removeprefix('__label__'), min(float(s), 1.0))
            for l, s in zip(labels, scores)]
```

<!-- Prose: our identifier against Common Crawl's CLD2 labels on the same records, the retention curve against the threshold, and what FineWeb kept of our English documents. -->

```{.python .input #crawl-lid}
%%tab pytorch
lid = {p['id']: d2l.detect_lang(p['texts']['trafilatura'])[0]
       for p in pages if p['texts']['trafilatura']}
assert all(lid[p['id']][0] == p['c0']['row']['lang'] and abs(
    lid[p['id']][1] - p['c0']['row']['lang_score']) < 1e-4
           for p in pages if p['id'] in lid)  # the workers agree
cld = {i: d2l._ISO639_3TO1.get(cld2[i].split(',')[0], cld2[i].split(',')[0]
                               or 'none') for i in lid if i in cld2}
top = [l for l, _ in collections.Counter(cld.values()).most_common(8)]
conf = pd.crosstab(pd.Series([cld[i] for i in cld], name='CLD2 (WET)'),
                   pd.Series([lid[i][0] if lid[i][0] in top else 'other'
                              for i in cld], name='fastText'))
print(conf.reindex(index=top, columns=top + ['other'], fill_value=0
                   ).to_string(), '\n')
en_scores = pd.Series({i: s for i, (l, s) in lid.items() if l == 'en'})
thresholds = np.round(np.arange(0.50, 0.91, 0.05), 2)
kept_en = [(en_scores >= t).sum() for t in thresholds]
in_fw = [en_scores[en_scores >= t].index.isin(joined).sum()
         for t in thresholds]
d2l.plot(thresholds, [kept_en, in_fw], 'fastText threshold on "en"',
         'English documents', legend=['kept', 'kept and in FineWeb'],
         legend_loc='upper right')
d2l.plt.axvline(LID_MIN, color='black', linestyle='--', linewidth=0.8)
k = list(thresholds).index(LID_MIN)
print(f'at {LID_MIN}: {kept_en[k]:,} English documents, {in_fw[k]:,} '
      f'({in_fw[k] / kept_en[k]:.0%}) of them in FineWeb')
for name, s in [
        ('sentence', 'The weather was cold so we stayed inside all day.'),
        ('code comment', 'x = load(path)  # The weather was cold so we '
                         'stayed inside all day\ny = x.sum()'),
        ('code-switched', 'The weather was cold, así que nos quedamos en '
                          'casa todo el día.')]:
    print(f'{name:14s}', [(l, round(sc, 2)) for l, sc in d2l.detect_lang(
        s, k=2)])
```

### Unicode Normalization
:label:`subsec_lm_crawl_unicode`

<!-- Prose: NFC versus NFKC (TR15), what NFKC rewrites in code and mathematics, the zero-width joiner, NBSP and U+FFFD; the rule: NFC once, before any tokenizer, and bytes for bits per byte counted after it. -->

```{.python .input #crawl-unicode}
%%tab pytorch
en = [p for p in pages if p['c0']['drop'] is None]  # the c0 documents
docs_en = [p['texts']['trafilatura'] for p in en]
loss, changed = {}, {}
for form in ('NFC', 'NFKC'):
    changed[form] = [s for s in docs_en
                     if not unicodedata.is_normalized(form, s)]
    loss[form] = dict(  # code points: those the form maps on their own
        documents_changed=len(changed[form]),
        share=len(changed[form]) / len(docs_en),
        code_points_rewritten=sum(unicodedata.normalize(form, c) != c
                                  for s in changed[form] for c in s),
        length_change=sum(len(unicodedata.normalize(form, s)) - len(s)
                          for s in changed[form]))
print(pd.DataFrame(loss).T.round(3).to_string(), '\n')
nfkc = collections.Counter(c for s in changed['NFKC'] for c in s
                           if unicodedata.normalize('NFKC', c) != c)
print(pd.DataFrame([dict(
    code_point=f'U+{ord(c):04X} {unicodedata.name(c, "?")}',
    nfkc=repr(unicodedata.normalize('NFKC', c)), count=n,
    category=unicodedata.category(c)) for c, n in nfkc.most_common(10)]
    ).to_string(index=False), '\n')
nfc = lambda s: unicodedata.normalize('NFC', s)
assert all(nfc(nfc(s)) == nfc(s) for s in docs_en)  # NFC is idempotent
for name, ch in [('U+FFFD replacement', '�'), ('U+00A0 NBSP', '\xa0'),
                 ('U+200B zero-width space', '​'),
                 ('U+200D zero-width joiner', '‍'),
                 ('U+FEFF byte-order mark', '﻿')]:
    print(f'{name:26s} in {sum(ch in s for s in docs_en):5,} documents, '
          f'{sum(s.count(ch) for s in docs_en):7,} times')
```

## Personal Data, Rights and the Manifest
:label:`subsec_lm_crawl_rights`

<!-- Prose: what rules can remove (emails, public IP addresses) and what they cannot (names, addresses, health information in prose). -->

```{.python .input #crawl-pii-def}
%%tab pytorch
#@save
PII_PATTERNS = {
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+'
                        r'[A-Za-z]{2,}\b'),
    'ip': re.compile(r'(?<![\w.:])(?:(?:\d{1,3}\.){3}\d{1,3}|'
                     r'(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|'
                     r'(?:[0-9A-Fa-f]{1,4}:){1,6}:'
                     r'(?:[0-9A-Fa-f]{1,4}:){0,5}[0-9A-Fa-f]{1,4})'
                     r'(?![\w:]|\.\d)')}  # a sentence may end in a dot

#@save
def scrub_pii(text):
    """Replace email addresses by <EMAIL> and public IPv4 and IPv6
    addresses by <IP> (an address must parse; private, loopback,
    documentation and multicast ranges are kept); returns the text and
    the counts."""
    import ipaddress
    counts = {'email': 0, 'ip': 0}

    def email(m):
        counts['email'] += 1
        return '<EMAIL>'

    def ip(m):
        try:
            a = ipaddress.ip_address(m.group())
            public = a.is_global and not a.is_multicast
        except ValueError:
            return m.group()
        counts['ip'] += public
        return '<IP>' if public else m.group()

    text = d2l.PII_PATTERNS['email'].sub(email, text)
    return d2l.PII_PATTERNS['ip'].sub(ip, text), counts
```

<!-- Prose: precision and recall per pattern on a fixture of 60 strings that hold personal data and 60 that do not; the false positives are the lesson. The fixture was written together with the rules (the sentence-final dot was added while writing it), so it specifies what the rules match and miss; its recall of 1.0 holds by construction and is never the scrubber's recall on web text. The c0 counts are rule matches, not checked by hand. -->

```{.python .input #crawl-pii}
%%tab pytorch
#@save
PII_FIXTURE = [  # (string, kind it holds: 'email', 'ip' or None)
    ('write to anna.berg@example.org for the slides', 'email'),
    ('Contact: j.smith+news@mail.example.co.uk', 'email'),
    ('send it to OPS@EXAMPLE.COM today', 'email'),
    ('reach me at li_wei99@uni-example.edu.cn', 'email'),
    ('Fragen an info@beispiel.de bitte', 'email'),
    ('(maria.garcia@correo.example.es)', 'email'),
    ('mailto:help-desk@example.net', 'email'),
    ('cc: bob@sub.domain.example.io, alice@example.ai', 'email'),
    ('Author: k.tanaka@example.jp', 'email'),
    ('reply to noreply@notifications.example.com', 'email'),
    ('email: first.last@company-example.com.', 'email'),
    ('<a href="mailto:press@example.org">press</a>', 'email'),
    ('orders: sales@shop.example.store', 'email'),
    ('my address is pat_2001@example.travel', 'email'),
    ('Ping x@example.dev if the build breaks', 'email'),
    ('Webmaster - webmaster@example.museum', 'email'),
    ('send CVs to jobs@example.careers', 'email'),
    ('Report abuse: abuse@isp.example.net', 'email'),
    ('questions -> q.and.a@forum.example.org', 'email'),
    ('Kontakt: m.mueller@firma-beispiel.at', 'email'),
    ('his email (tom.o@example.ie) was on the poster', 'email'),
    ('chair: prof.chen@cs.example.edu', 'email'),
    ('Email:support@example.com', 'email'),
    ('a.b.c.d@example.co', 'email'),
    ('Subscribe: list-request@lists.example.org', 'email'),
    ('billing@example.biz handles refunds', 'email'),
    ('user%host@example.com is an old relay form', 'email'),
    ('Send photos to editor@news.example.com.au', 'email'),
    ('Tel. 555-0100, mail: office@example.fr', 'email'),
    ('Questions? Ask ivan.petrov@example.ru', 'email'),
    ('server at 8.8.8.8 answered', 'ip'),
    ('Client IP: 203.0.114.7', 'ip'),
    ('ssh root@93.184.216.34', 'ip'),
    ('blocked 45.33.32.156 after 5 tries', 'ip'),
    ('origin=151.101.1.69;', 'ip'),
    ('ping 1.1.1.1 -c 4', 'ip'),
    ('from host 81.2.69.160 port 443', 'ip'),
    ('X-Forwarded-For: 66.249.66.1', 'ip'),
    ('the attack came from 185.220.101.4.', 'ip'),
    ('[23.45.67.89]', 'ip'),
    ('IP 104.16.132.229 resolved', 'ip'),
    ('nameserver 9.9.9.9', 'ip'),
    ('allowlist: 52.95.110.1', 'ip'),
    ('log line 17.253.144.10 - - [26/Jan/2025]', 'ip'),
    ('192.30.255.112 is listed', 'ip'),
    ('visit http://200.147.67.142/index', 'ip'),
    ('my VPS is 139.59.21.43', 'ip'),
    ('peer: 77.88.55.60', 'ip'),
    ('gateway (31.13.71.36)', 'ip'),
    ('upstream 140.82.112.3', 'ip'),
    ('AAAA 2a00:1450:4001:82a::200e', 'ip'),
    ('v6: 2606:4700:4700::1111', 'ip'),
    ('host 2620:fe::fe answers', 'ip'),
    ('2a03:2880:f12f:83:face:b00c:0:25de', 'ip'),
    ('from 2001:4860:4860::8888', 'ip'),
    ('addr 2a01:4f8:c17:1234::1', 'ip'),
    ('[2400:cb00:2048:1::c629:d7a2]', 'ip'),
    ('client 2a02:6b8::2:242 connected', 'ip'),
    ('route via 2600:1f18:6a3:4b00::12', 'ip'),
    ('ns 2001:500:2f::f', 'ip'),
    ('our router is at 192.168.1.1', None),
    ('the cluster uses 10.0.0.0/8 internally', None),
    ('localhost is 127.0.0.1', None),
    ('documentation range 192.0.2.15', None),
    ('link-local fe80::1ff:fe23:4567:890a', None),
    ('test net 198.51.100.7', None),
    ('private 172.16.254.3', None),
    ('loopback ::1 only', None),
    ('unique local fd12:3456:789a:1::1', None),
    ('example 2001:db8::ff00:42:8329', None),
    ('Upgrade to version 4.2.1.0 now', None),
    ('Changelog for 1.2.3.4 (stable)', None),
    ('build 5.15.0.91 of the kernel', None),
    ('SNMP OID 1.3.6.1.4.1.2021', None),
    ('section 3.4.1.2 of the standard', None),
    ('ISBN 978-0-306-40615-7', None),
    ('ISBN 0-306-40615-2 (paperback)', None),
    ('phone +1 555 0100 222', None),
    ('the score was 3.14.15.92 in the puzzle', None),
    ('logo@2x.png in the assets folder', None),
    ('icon@3x.webp for retina screens', None),
    ('follow @d2l_ai for updates', None),
    ('write to anna dot berg at example dot org', None),
    ('the domain example.com hosts it', None),
    ('price 1.299,00 EUR', None),
    ('coordinates 52.5200, 13.4050', None),
    ('date 2025.01.26', None),
    ('time 13:54:02', None),
    ('MAC 00:1A:2B:3C:4D:5E', None),
    ('C++ uses std::vector<int>', None),
    ('the ratio 16:9 screen', None),
    ('PEP 8 says @decorator first', None),
    ('the user name is root', None),
    ('256.300.1.2 is not an address', None),
    ('999.1.1.1 overflows', None),
    ('vector 1:2:3 notation', None),
    ('npm i pkg@1.2.3', None),
    ('git@github.com:org/repo.git', None),
    ('the hash a3f2:9c1e', None),
    ('Bible verse John 3:16', None),
    ('temperature 21.5 C', None),
    ('a.b@c', None),
    ('foo@bar', None),
    ('10.10.10.10 on the lab switch', None),
    ('the meeting is at 10.30', None),
    ('patent US 7,654,321 B2', None),
    ('IPv4 has 4.3 billion addresses', None),
    ('the CIDR 0.0.0.0/0 means everything', None),
    ('multicast 224.0.0.251 for mDNS', None),
    ('broadcast 255.255.255.255', None),
    ('shared address space 100.64.0.1', None),
    ('benchmark range 198.18.0.1', None),
    ('version v2.0.0.1-beta', None),
    ('model 7.1.2.3 of the router firmware', None),
    ('ISSN 0317-8471', None),
    ('the regex \\d+@\\w+', None),
    ('emoji :smile: in chat', None),
    ('call 0800 123 4567 for help', None),
    ('IBAN DE89 3704 0044 0532 0130 00', None),
    ('ISBN-13 978.3.16.148410.0', None)]

kinds = ['email', 'ip']
pred = [d2l.scrub_pii(s)[1] for s, _ in d2l.PII_FIXTURE]
table = []
for k in kinds:
    tp = sum(p[k] > 0 and t == k for p, (_, t) in zip(pred, d2l.PII_FIXTURE))
    fp = sum(p[k] > 0 and t != k for p, (_, t) in zip(pred, d2l.PII_FIXTURE))
    pos = sum(t == k for _, t in d2l.PII_FIXTURE)
    table.append(dict(pattern=k, positives=pos, recall=tp / pos,
                      precision=tp / max(tp + fp, 1), false_positives=fp))
print(f'{len(d2l.PII_FIXTURE)} fixture strings, '
      f'{sum(t is not None for _, t in d2l.PII_FIXTURE)} with personal data')
print(pd.DataFrame(table).round(3).to_string(index=False))
print('false positives:', [s for p, (s, t) in zip(pred, d2l.PII_FIXTURE)
                           if t is None and any(p.values())])
scrub = [d2l.scrub_pii(unicodedata.normalize('NFC', p['texts']['trafilatura']))
         for p in en]
assert all(c == p['c0']['pii'] for (_, c), p in zip(scrub, en))
n_email = sum(c['email'] > 0 for _, c in scrub)
n_ip = sum(c['ip'] > 0 for _, c in scrub)
delta = sum(len(t.encode()) - len(unicodedata.normalize(
    'NFC', p['texts']['trafilatura']).encode())
            for (t, _), p in zip(scrub, en))
print(f'c0 ({len(en):,} English documents), rule matches (not checked by '
      f'hand): emails in {n_email} documents, public IP addresses in {n_ip}; '
      f'{sum(c["email"] for _, c in scrub)} and '
      f'{sum(c["ip"] for _, c in scrub)} replaced; UTF-8 bytes changed by '
      f'{delta:+,}')
```

<!-- Prose: provenance fields; Common Crawl is an archive, not a licence; Consent in Crisis; opt-outs (The Stack v2); why this section shows FineWeb's text and never its own extraction of a crawl page; not legal advice; FineWeb-2 as the multilingual pipeline. -->

### The Manifest
:label:`subsec_lm_crawl_manifest`

<!-- Prose: the schema d2l-corpus/1 (table tab_lm_crawl_manifest_schema, static Markdown, written with the prose); a document and its provenance. -->

```{.python .input #crawl-doc-def}
%%tab pytorch
#@save
Doc = collections.namedtuple(
    'Doc', 'id url date text lang lang_score src log flags')

#@save
def write_docs(path, docs):
    """Write documents (Doc or dicts with Doc's fields) as gzip-compressed
    JSON lines, keys sorted and no time stamp in the gzip header, so equal
    documents give equal bytes."""
    with open(path, 'wb') as raw, gzip.GzipFile(
            filename='', mode='wb', fileobj=raw, mtime=0) as f:
        for d in docs:
            row = d._asdict() if hasattr(d, '_asdict') else dict(d)
            f.write((json.dumps(row, ensure_ascii=False, sort_keys=True)
                     + '\n').encode('utf-8'))

#@save
def read_docs(path):
    """The documents of a file written by write_docs, as a list of Doc
    (fields missing from a row are None)."""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    return [d2l.Doc(**{k: r.get(k) for k in d2l.Doc._fields}) for r in rows]

#@save
def docs_hash(docs):
    """SHA-256 over the sorted (id, SHA-256 of the text) pairs: the content
    of a document set, independent of its order. Takes Doc-like objects,
    dicts or (id, text) pairs."""
    def pair(d):
        if isinstance(d, dict):
            return d['id'], d['text']
        if hasattr(d, 'text'):
            return d.id, d.text
        return d[0], d[1]
    keys = sorted((str(i), hashlib.sha256((t or '').encode('utf-8'))
                   .hexdigest()) for i, t in map(pair, docs))
    h = hashlib.sha256()
    for i, t in keys:
        h.update(f'{i}\t{t}\n'.encode('utf-8'))
    return h.hexdigest()
```

<!-- Prose: the stage identity n_in = n_out + sum(drops) with first-rule charging; canonical JSON and its hash. -->

```{.python .input #crawl-manifest-def}
%%tab pytorch
#@save
class CorpusManifest:
    """The provenance of a corpus (schema d2l-corpus/1): its sources, the
    stages that turned them into documents, and, once tokenized, the
    tokenizer, the shard format and the splits. Each stage satisfies
    n_in == n_out + sum(drops), a document being charged to the first rule
    that drops it. The canonical JSON has sorted keys, no spaces, floats
    rounded to 4 decimals and no absolute paths; sha256() hashes it."""
    SCHEMA = 'd2l-corpus/1'

    def __init__(self, name, sources=None, stages=None, tokenizer=None,
                 format=None, splits=None):
        self.name, self.tokenizer, self.format = name, tokenizer, format
        self.sources, self.stages = list(sources or []), list(stages or [])
        self.splits = splits

    def add_source(self, id, uri, revision=None, license=None, n_docs=None,
                   bytes=None, **extra):
        self.sources.append(dict(id=id, uri=uri, revision=revision,
                                 license=license, n_docs=n_docs,
                                 bytes=bytes, **extra))
        return self

    def add_stage(self, name, params, n_in, n_out, drops, in_hash, out_hash,
                  *, impl=None, version=None, tokens_out=None):
        drops = {str(k): int(v) for k, v in (drops or {}).items()}
        if n_in != n_out + sum(drops.values()):
            raise ValueError(f'stage {name}: n_in {n_in} != n_out {n_out} '
                             f'+ drops {sum(drops.values())}')
        self.stages.append(dict(
            name=name, impl=impl, version=version, params=params,
            n_in=int(n_in), n_out=int(n_out), tokens_out=tokens_out,
            drops=drops, in_hash=in_hash, out_hash=out_hash))
        return self

    def to_dict(self):
        return dict(schema=self.SCHEMA, name=self.name, sources=self.sources,
                    stages=self.stages, tokenizer=self.tokenizer,
                    format=self.format, splits=self.splits)

    def canonical_json(self):
        def clean(x, where):
            if isinstance(x, float):
                if x != x or x in (float('inf'), float('-inf')):
                    raise ValueError(f'{where}: {x} has no JSON form')
                return round(x, 4)
            if isinstance(x, (np.integer, np.floating, np.bool_)):
                return clean(x.item(), where)
            if isinstance(x, str) and (x.startswith('/') or
                                       re.match(r'[A-Za-z]:\\', x)):
                raise ValueError(f'{where}: absolute path {x!r}')
            if isinstance(x, dict):
                return {str(k): clean(v, f'{where}.{k}')
                        for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [clean(v, f'{where}[{i}]') for i, v in enumerate(x)]
            return x
        return json.dumps(clean(self.to_dict(), 'manifest'), sort_keys=True,
                          separators=(',', ':'), ensure_ascii=False)

    def sha256(self):
        return hashlib.sha256(self.canonical_json().encode('utf-8')
                              ).hexdigest()

    def save(self, path):
        """Write the canonical JSON, so the file's SHA-256 is sha256()."""
        with open(path, 'wb') as f:
            f.write(self.canonical_json().encode('utf-8'))
        return path

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        if d.get('schema') != cls.SCHEMA:
            raise ValueError(f'{path}: schema {d.get("schema")!r}, '
                             f'expected {cls.SCHEMA}')
        return cls(d['name'], d.get('sources'), d.get('stages'),
                   d.get('tokenizer'), d.get('format'), d.get('splits'))

    def to_card(self):
        """A Markdown dataset card generated from the manifest alone."""
        lines = [f'# Dataset card: {self.name}', '',
                 f'Schema `{self.SCHEMA}`, manifest SHA-256 '
                 f'`{self.sha256()[:16]}…`.', '', '## Sources', '',
                 '| id | uri | revision | license | documents | bytes |',
                 '|---|---|---|---|---|---|']
        for s in self.sources:
            lines.append('| ' + ' | '.join(str(s.get(k, '')) for k in (
                'id', 'uri', 'revision', 'license', 'n_docs', 'bytes')) + ' |')
        lines += ['', '## Processing stages', '',
                  '| stage | implementation | in | out | dropped (reason: count) |',
                  '|---|---|---|---|---|']
        for s in self.stages:
            impl = ' '.join(str(v) for v in (s.get('impl'), s.get('version'))
                            if v)
            drops = ', '.join(f'{k}: {v}' for k, v in s['drops'].items()
                              ) or '—'
            lines.append(f'| {s["name"]} | {impl or "—"} | {s["n_in"]} | '
                         f'{s["n_out"]} | {drops} |')
        if self.tokenizer:
            lines += ['', '## Tokenizer', '', '```',
                      json.dumps(self.tokenizer, sort_keys=True), '```']
        if self.format:
            lines += ['', '## Format', '', '```',
                      json.dumps(self.format, sort_keys=True), '```']
        if self.splits:
            lines += ['', '## Splits', '', '| split | documents | tokens | shards |',
                      '|---|---|---|---|']
            for k, v in sorted(self.splits.items()):
                lines.append(f'| {k} | {v.get("n_docs")} | {v.get("n_tokens")}'
                             f' | {len(v.get("shards", []))} |')
        return '\n'.join(lines) + '\n'
```

<!-- Prose: the c0 manifest: four stages, the funnel they record, and the hash round trip. -->

```{.python .input #crawl-c0}
%%tab pytorch
with open(d2l.download_keyed('lm-cc-warc-256m-v1')) as f:
    pin = json.load(f)
resp = index[index.type == 'response']
c0 = {p['id']: p['c0'] for p in pages}
empty = [i for i, o in c0.items() if o['drop'] == 'empty']
nonempty = [p for p in pages if c0[p['id']]['drop'] != 'empty']
kept = [p for p in nonempty if c0[p['id']]['drop'] is None]
raw = lambda ps: [(p['id'], p['texts']['trafilatura']) for p in ps]
nfc_docs = [(p['id'], unicodedata.normalize('NFC', p['texts']['trafilatura']))
            for p in kept]
docs = [d2l.Doc(o['row']['id'], o['row']['url'], o['row']['date'], o['text'],
                o['row']['lang'], o['row']['lang_score'], o['row']['src'],
                o['row']['log'], None) for o in (c0[p['id']] for p in kept)]
m = d2l.CorpusManifest('c0')
m.add_source('cc-main-2025-05-00000', pin['warc']['url'],
             revision=pin['warc']['sha256'], n_docs=len(resp),
             bytes=CRAWL_MIB * 2**20, license='none granted (Common Crawl '
             'terms of use); text not redistributed')
m.add_stage('extract', dict(rules=['not_200', 'not_html', 'empty'],
                            charset_chain=['header', 'meta', 'detector',
                                           'fallback'],
                            include_tables=True, favor_precision=False,
                            versions={k: importlib.metadata.version(k) for k in
                                      ('lxml', 'justext', 'charset-normalizer')}),
            len(resp), len(nonempty), dict(
                not_200=int((resp.status != 200).sum()),
                not_html=int(((resp.status == 200) &
                              (resp.mime != 'text/html')).sum()),
                empty=len(empty)), pin['warc']['sha256'],
            d2l.docs_hash(raw(nonempty)), impl='trafilatura',
            version=importlib.metadata.version('trafilatura'))
m.add_stage('lang_id', dict(rules=['not_en', 'low_score'], label='en',
                            min_score=LID_MIN, model='lid.176.ftz',
                            model_sha1=d2l.DATA_HUB['lid.176.ftz'][1]),
            len(nonempty), len(kept), collections.Counter(
                c0[p['id']]['drop'] for p in nonempty if c0[p['id']]['drop']),
            d2l.docs_hash(raw(nonempty)), d2l.docs_hash(raw(kept)),
            impl='fasttext',
            version=importlib.metadata.version('fasttext-numpy2-wheel'))
m.add_stage('nfc', dict(form='NFC'), len(kept), len(kept), {},
            d2l.docs_hash(raw(kept)), d2l.docs_hash(nfc_docs),
            impl='unicodedata', version=unicodedata.unidata_version)
m.add_stage('pii', dict(patterns=sorted(d2l.PII_PATTERNS),
                        replace=dict(email='<EMAIL>', ip='<IP>')),
            len(kept), len(docs), {}, d2l.docs_hash(nfc_docs),
            d2l.docs_hash(docs), impl='d2l.scrub_pii')
print(pd.DataFrame([dict(stage=s['name'], n_in=s['n_in'], n_out=s['n_out'],
                         dropped=', '.join(f'{k} {v:,}' for k, v in
                                           s['drops'].items()) or '-')
                    for s in m.stages]).to_string(index=False))

cache = os.path.join('..', 'data', 'lm', 'c0')
os.makedirs(cache, exist_ok=True)
m.save(os.path.join(cache, 'c0-manifest.json'))
again = d2l.CorpusManifest.load(os.path.join(cache, 'c0-manifest.json'))
order = np.random.default_rng(0).permutation(len(docs))
last = docs[1].text[-1]
edited = docs[:1] + [docs[1]._replace(  # one character changed
    text=docs[1].text[:-1] + ('#' if last != '#' else '%'))] + docs[2:]
print(f'manifest {m.sha256()[:16]}…, after save and load '
      f'{again.sha256()[:16]}…; docs_hash {d2l.docs_hash(docs)[:16]}…, '
      f'permuted {d2l.docs_hash([docs[i] for i in order])[:16]}…, '
      f'one byte edited {d2l.docs_hash(edited)[:16]}…')
assert again.sha256() == m.sha256()
assert d2l.docs_hash([docs[i] for i in order]) == d2l.docs_hash(docs)
assert d2l.docs_hash(edited) != d2l.docs_hash(docs)

rows = [c0[p['id']]['row'] for p in kept]  # WARC order, no text
C0_KEY, C0_FILE = ARTIFACTS['c0']
if 'c0' in RERUN:
    d2l.lm_save_rerun(C0_FILE, lambda f: d2l.write_docs(f, rows))
else:
    with gzip.open(d2l.lm_artifact(C0_KEY), 'rt', encoding='utf-8') as f:
        reg = [json.loads(line) for line in f]
    same = [(r['id'], r['text_sha256']) for r in reg] == [
        (r['id'], r['text_sha256']) for r in rows]
    print(f'against {C0_KEY}: {len(reg):,} registered rows, {len(rows):,} '
          f'now; ids and text hashes {"identical" if same else "DIFFER"}')
    assert same
```

<!-- Prose: the c0 rows hold no text; rebuild_c0 recovers every document from the pinned range and checks its hash, so a takedown is honoured by deleting a row. -->

```{.python .input #crawl-rebuild-def}
%%tab pytorch
#@save
def rebuild_c0(manifest='lm-c0-docs-v1', warc_path=None, *, cache_dir=None,
               workers=None):
    """Rebuild the c0 documents from their manifest rows (a DATA_HUB key or
    a path): each record is read from the verified WARC range by offset
    and length and passed through the recorded stages again. A document is
    kept only if its text hashes to the recorded text_sha256; the others
    are listed in .mismatches (extractor drift), never kept. The texts are
    written to cache_dir (default ../data/lm/c0), never uploaded."""
    import concurrent.futures
    import multiprocessing
    import pathlib
    path = (d2l.download_keyed(manifest) if manifest in d2l.DATA_HUB
            else manifest)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    warc_path = str(warc_path or d2l.fetch_warc_range())
    d2l.download_keyed('lid.176.ftz')  # once, before the workers
    with concurrent.futures.ProcessPoolExecutor(
            workers or min(8, os.cpu_count()),
            mp_context=multiprocessing.get_context('spawn')) as ex:
        outs = list(ex.map(d2l._c0_rebuild_one,
                           [(warc_path, r) for r in rows], chunksize=200))
    docs = d2l.DocList()
    docs.mismatches = []
    for row, out in zip(rows, outs):
        got = out['row']
        if out['drop'] or got['text_sha256'] != row['text_sha256']:
            docs.mismatches.append(dict(
                id=row['id'], drop=out['drop'], recorded=row['text_sha256'],
                rebuilt=got['text_sha256'],
                versions=(row['extractor_version'],
                          got['extractor_version'])))
        else:
            docs.append(d2l.Doc(row['id'], row['url'], row['date'],
                                out['text'], row['lang'], row['lang_score'],
                                row['src'], row['log'], None))
    cache = pathlib.Path(cache_dir or os.path.join('..', 'data', 'lm', 'c0'))
    cache.mkdir(parents=True, exist_ok=True)
    d2l.write_docs(cache / 'c0-docs.jsonl.gz', docs)
    return docs
```

```{.python .input #crawl-rebuild}
%%tab pytorch
rebuilt = d2l.rebuild_c0(d2l.lm_artifact(
    C0_KEY, C0_FILE if 'c0' in RERUN else None), path, workers=N_WORKERS)
print(f'{len(rebuilt):,} documents rebuilt from the rows and the pinned '
      f'range, {len(rebuilt.mismatches)} mismatches; docs_hash '
      f'{d2l.docs_hash(rebuilt)[:16]}… (manifest out_hash '
      f'{m.stages[-1]["out_hash"][:16]}…)')
for x in rebuilt.mismatches[:5]:
    print('  mismatch:', x)
assert not rebuilt.mismatches
assert d2l.docs_hash(rebuilt) == m.stages[-1]['out_hash']
```

<!-- Prose: a dataset card generated from the manifest alone, and what it cannot say. -->

```{.python .input #crawl-card}
%%tab pytorch
print(m.to_card())
```

<!-- Prose: the results this section registers, and their agreement with the registered copies. -->

```{.python .input #crawl-results}
%%tab pytorch
def rounded(x):
    """Floats to 4 decimals, numpy scalars to Python, NaN to None."""
    if isinstance(x, dict):
        return {str(k): rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        x = x.item()
    if isinstance(x, float):
        return None if x != x else round(x, 4)
    return x

results = rounded(dict(
    crawl_mib=CRAWL_MIB, lid_min=LID_MIN, records=stats,
    c0_manifest=m.to_dict(), recall=recall.to_dict('records'),
    paired=deltas.to_dict('records'), precision=prec.to_dict('records'),
    evaluation=dict(pages=len(ev), in_c0=in_c0, precision_pages=len(full)),
    fineweb=dict(joined=len(joined), rows=len(in_range), how=how,
                 f1=summary.to_dict('records')),
    lid=dict(thresholds=thresholds.tolist(), kept=kept_en, in_fineweb=in_fw,
             confusion=conf.reindex(index=top, columns=top + ['other'],
                                    fill_value=0).to_dict()),
    unicode=loss, pii=table))
blob = json.dumps(results, sort_keys=True, indent=1).encode('utf-8')
R_KEY, R_FILE = ARTIFACTS['crawl-results']
if 'crawl-results' in RERUN:
    d2l.lm_save_rerun(R_FILE, lambda f: pathlib.Path(f).write_bytes(blob))
else:
    with open(d2l.lm_artifact(R_KEY)) as f:
        reg = json.load(f)
    a = d2l.json_agreement(results, reg)
    same = reg['c0_manifest'] == results['c0_manifest']
    print(f'against {R_KEY}: {a["equal"]}/{a["values"]} values equal, '
          f'largest numeric difference {a["max_diff"]:.2g}; manifest '
          f'{"identical" if same else "DIFFERS"}')
    assert same and a['equal'] == a['values']  # deterministic given the pins
P_KEY, P_FILE = ARTIFACTS['pin']
if 'pin' in RERUN:  # a consistency check, not a new pin: fetch_warc_range
    # verifies the bytes against the registered pin's block hashes, so this
    # reproduces that pin or fails; a pin for another range starts from
    # d2l._cc_pin on a download that nothing has verified yet
    new_pin = d2l._cc_pin_bytes(d2l._cc_pin(
        d2l.fetch_warc_range(), d2l.fetch_warc_range(which='wet')))
    d2l.lm_save_rerun(P_FILE, lambda f: pathlib.Path(f).write_bytes(new_pin))
    with open(d2l.download_keyed(P_KEY), 'rb') as f:
        same = f.read() == new_pin
    print(f'pin {"identical to" if same else "DIFFERS from"} {P_KEY}')
```

## Summary

<!-- Prose: Summary (3-6 sentences). -->

## Exercises

<!-- Prose: Exercises (7; spec part 8). -->
