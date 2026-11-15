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

<!-- Placeholder (stage 2, infra step): the cells below are the section's
shared library, written before the section; the prose around them comes with
the section. Cell ids follow the section specification. -->

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
import gzip
import hashlib
import json
import os
import re
```

<!-- Placeholder (stage 2): the pin of the crawl range and the language identifier; the chapter's other artifacts join this registry as they are built. -->

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

<!-- Placeholder (stage 2): every external dataset and model the chapter reads, at a pinned revision, with its licence. -->

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

<!-- Placeholder (stage 2): a WARC file is a sequence of gzip members, one record each, so any byte prefix parses record by record; the last member of a cut-off prefix is incomplete and is dropped. -->

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

<!-- Placeholder (stage 2): the pinned range is fetched once, block by block, and every 16 MiB block is checked against the SHA-256 in the pin before it is kept. -->

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

<!-- Placeholder (stage 2): a document is a record with its provenance; the content hash of a set of documents does not depend on their order. -->

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

<!-- Placeholder (stage 2): the corpus manifest `d2l-corpus/1`; every stage satisfies n_in = n_out + sum(drops), and the canonical JSON (sorted keys, no time stamps, no absolute paths) is what the hash covers. -->

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

