# Document Extraction: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It holds the library code that
:numref:`sec_lm_crawl` explains in a sentence each but does not show:

- the gzip-member reader and the record parser behind `iter_warc`
  (`_gzip_members`, `_warc_chunks`, `_warc_parse`);
- the block-checked fetch behind `fetch_warc_range` (`_verified_blocks`,
  `_fetch_blocks`, `_fetch_whole`, `_sha256_file`) and the pin itself
  (`_CC_PIN_SOURCE`, `_cc_pin`, `_cc_pin_bytes`), which the section's
  regeneration `pin` writes;
- registered domains for cluster bootstraps (`registered_domain`, tldextract
  with its bundled suffix list, so no network).

Every block is tagged `%%tab pytorch` and imports what it needs beyond the
library preamble (`zlib`, `pathlib`, `tldextract`) inside its functions: a
module-level import would join the `_preamble` shard and move the freshness
fingerprint of every notebook in the book.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_crawl`

```{.python .input #crawl-lib-warc}
%%tab pytorch
#@save
def _warc_chunks(source, n_bytes=None, chunk=1 << 16):
    """The first n_bytes (None: all) of a local file or URL in chunks of
    64 KiB: small chunks keep zlib's copy of the unused input short."""
    if str(source).startswith(('http://', 'https://')):
        headers = {} if n_bytes is None else {
            'Range': f'bytes=0-{n_bytes - 1}'}
        with requests.get(str(source), headers=headers, stream=True,
                          timeout=60) as r:
            r.raise_for_status()
            if n_bytes is not None and r.status_code != 206:
                raise RuntimeError(f'{source}: the server ignored the Range')
            yield from r.iter_content(chunk)
        return
    left = float('inf') if n_bytes is None else n_bytes
    with open(source, 'rb') as f:
        while left > 0:
            b = f.read(int(min(chunk, left)))
            if not b:
                return
            left -= len(b)
            yield b

#@save
def _gzip_members(chunks):
    """(offset, length, data) of each gzip member in a byte stream, offset
    and length in compressed bytes; a member cut off by the end of the
    stream comes last, with data None."""
    import zlib
    chunks = (c for c in chunks if c)
    buf, offset = b'', 0
    while True:
        if not buf:
            buf = next(chunks, None)
            if buf is None:
                return
        d, out, used = zlib.decompressobj(31), [], 0
        while True:
            out.append(d.decompress(buf))
            if d.eof:
                used += len(buf) - len(d.unused_data)
                buf = d.unused_data
                break
            used += len(buf)
            buf = next(chunks, None)
            if buf is None:
                yield offset, used, None
                return
        yield offset, used, b''.join(out)
        offset += used

#@save
def _warc_parse(raw, offset, length):
    """The WarcRecord in the decompressed bytes of one member. headers holds
    the WARC header fields; for an HTTP request, response or revisit it also
    holds headers['HTTP'], the HTTP header fields with lower-cased names,
    and body is the HTTP payload, else the whole content block."""
    head, _, rest = raw.partition(b'\r\n\r\n')
    lines = head.split(b'\r\n')
    if not lines[0].startswith(b'WARC/'):
        raise ValueError(f'no WARC record at offset {offset}')
    headers = {}
    for line in lines[1:]:
        k, _, v = line.decode('utf-8', 'replace').partition(':')
        headers[k.strip()] = v.strip()
    block = rest[:int(headers.get('Content-Length', len(rest)))]
    kind = headers.get('WARC-Type')
    ctype = headers.get('Content-Type', '')
    status, mime, body = None, ctype.split(';')[0].strip().lower(), block
    if (kind in ('response', 'request', 'revisit')
            and ctype.startswith('application/http')):
        http, _, body = block.partition(b'\r\n\r\n')
        http = http.decode('iso-8859-1').split('\r\n')
        fields = {}
        for line in http[1:]:
            k, _, v = line.partition(':')
            k = k.strip().lower()
            fields[k] = (fields[k] + ', ' if k in fields else '') + v.strip()
        headers['HTTP'] = fields
        mime = fields.get('content-type', '').split(';')[0].strip().lower()
        parts = http[0].split()
        if kind != 'request' and len(parts) > 1 and parts[1].isdigit():
            status = int(parts[1])
    return d2l.WarcRecord(headers.get('WARC-Record-ID'), kind,
                          headers.get('WARC-Target-URI'),
                          headers.get('WARC-Date'), status, mime or None,
                          offset, length, headers, body)
```

```{.python .input #crawl-lib-fetch}
%%tab pytorch
#@save
def _sha256_file(path, n_bytes=None):
    """SHA-256 of the first n_bytes (None: all) of a file."""
    h, left = hashlib.sha256(), float('inf') if n_bytes is None else n_bytes
    with open(path, 'rb') as f:
        while left > 0:
            b = f.read(int(min(1 << 24, left)))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()

#@save
def _verified_blocks(path, block_sha256, block, n_blocks):
    """How many leading blocks of the file match the pinned SHA-256s (up to
    n_blocks). The file is cut back to them, so a later fetch appends."""
    if not path.exists():
        return 0
    good = 0
    with open(path, 'rb') as f:
        while good < n_blocks:
            b = f.read(block)
            if len(b) < block or (hashlib.sha256(b).hexdigest()
                                  != block_sha256[good]):
                break
            good += 1
    if path.stat().st_size > good * block and good < n_blocks:
        with open(path, 'r+b') as f:
            f.truncate(good * block)
    return good

#@save
def _fetch_blocks(warc, path, start, stop):
    """Append blocks start..stop-1 of the pinned range to path, each one
    checked against its SHA-256 before it is written."""
    block = warc['block_bytes']
    print(f'fetching bytes {start * block}-{stop * block - 1} of '
          f'{warc["url"]}')
    rng = f'bytes={start * block}-{stop * block - 1}'
    with requests.get(warc['url'], headers={'Range': rng}, stream=True,
                      timeout=60) as r, open(path, 'ab') as f:
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError(f'{warc["url"]}: the server ignored the Range')
        k, buf = start, bytearray()
        for c in r.iter_content(1 << 20):
            buf += c
            while len(buf) >= block and k < stop:
                got = hashlib.sha256(buf[:block]).hexdigest()
                if got != warc['block_sha256'][k]:
                    raise RuntimeError(f'block {k} of {warc["url"]}: '
                                       f'SHA-256 {got}, the pin says '
                                       f'{warc["block_sha256"][k]}')
                f.write(buf[:block])
                del buf[:block]
                k += 1
    if k != stop:
        raise RuntimeError(f'{warc["url"]}: got {k - start} of '
                           f'{stop - start} blocks')

#@save
def _fetch_whole(spec, cache, offline):
    """A whole file named in the pin, fetched once and checked by SHA-256."""
    path = cache / spec['url'].rsplit('/', 1)[1]
    if path.exists() and d2l._sha256_file(path) == spec['sha256']:
        return path
    if offline:
        raise RuntimeError(f'D2L_OFFLINE=1 and no verified {path}: run once '
                           f'with network access, or fetch {spec["url"]}')
    print(f'fetching {spec["url"]}')
    tmp = path.with_suffix(path.suffix + '.part')
    with requests.get(spec['url'], stream=True, timeout=60) as r, \
            open(tmp, 'wb') as f:
        r.raise_for_status()
        for c in r.iter_content(1 << 20):
            f.write(c)
    got = d2l._sha256_file(tmp)
    if got != spec['sha256']:
        tmp.unlink()
        raise RuntimeError(f'{spec["url"]}: SHA-256 {got}, the pin says '
                           f'{spec["sha256"]}')
    tmp.replace(path)
    return path
```

```{.python .input #crawl-lib-pin}
%%tab pytorch
#@save
_CC_PIN_SOURCE = dict(  # one archive file of the CC-MAIN-2025-05 crawl
    crawl='CC-MAIN-2025-05',
    warc_url='https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-05/'
             'segments/1736703361941.29/warc/'
             'CC-MAIN-20250126135402-20250126165402-00000.warc.gz',
    warc_size=1141586184, n_bytes=256 * 2**20, block_bytes=16 * 2**20,
    wet_url='https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-05/'
            'segments/1736703361941.29/wet/'
            'CC-MAIN-20250126135402-20250126165402-00000.warc.wet.gz',
    wet_size=102325106)

#@save
def _cc_pin(warc_path, wet_path, src=None):
    """The pin of the crawl range from verified local copies: URLs, sizes,
    the SHA-256 of the range, of each of its blocks and of the WET file.
    It names Common Crawl bytes; it holds none."""
    src = src or d2l._CC_PIN_SOURCE
    n, block = src['n_bytes'], src['block_bytes']
    if os.path.getsize(warc_path) < n:
        raise ValueError(f'{warc_path} holds fewer than {n} bytes')
    blocks = []
    with open(warc_path, 'rb') as f:
        for _ in range(n // block):
            blocks.append(hashlib.sha256(f.read(block)).hexdigest())
    if os.path.getsize(wet_path) != src['wet_size']:
        raise ValueError(f'{wet_path} is not {src["wet_size"]} bytes')
    _, cc, path = src['warc_url'].partition('commoncrawl.org/')
    return dict(  # FineWeb's file_path column names the file this way
        schema='d2l-cc-pin/1', crawl=src['crawl'],
        fineweb_file_path='s3://commoncrawl/' + path if cc else None,
        warc=dict(url=src['warc_url'], file_size=src['warc_size'],
                  range=[0, n - 1], n_bytes=n,
                  sha256=d2l._sha256_file(warc_path, n), block_bytes=block,
                  block_sha256=blocks),
        wet=dict(url=src['wet_url'], file_size=src['wet_size'],
                 sha256=d2l._sha256_file(wet_path)))

#@save
def _cc_pin_bytes(pin):
    """The pin file's exact bytes: sorted keys, one space indent, newline."""
    return (json.dumps(pin, sort_keys=True, indent=1) + '\n').encode('utf-8')
```

```{.python .input #crawl-lib-domain}
%%tab pytorch
#@save
_TLD_EXTRACT = []  # one tldextract instance per process, made on first use

#@save
def registered_domain(url):
    """The registered domain of a URL (example.co.uk for
    a.b.example.co.uk), from tldextract's bundled public-suffix snapshot
    (no network); the lower-cased host when there is none (an IP
    address, localhost)."""
    import urllib.parse
    if not d2l._TLD_EXTRACT:
        import tldextract
        d2l._TLD_EXTRACT.append(tldextract.TLDExtract(
            suffix_list_urls=(), cache_dir=None))
    ext = d2l._TLD_EXTRACT[0](url)
    dom = ext.top_domain_under_public_suffix
    return (dom or urllib.parse.urlsplit(url).hostname or '').lower()
```
