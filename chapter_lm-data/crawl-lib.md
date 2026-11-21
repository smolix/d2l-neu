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

```{.python .input #crawl-lib-decode}
%%tab pytorch
#@save
_CHARSET_ALIASES = {  # WHATWG: these labels mean windows-1252 in a browser
    'iso-8859-1': 'cp1252', 'iso8859-1': 'cp1252', 'latin1': 'cp1252',
    'latin-1': 'cp1252', 'l1': 'cp1252', 'us-ascii': 'cp1252',
    'ascii': 'cp1252'}

#@save
def decode_body(body, headers):
    """(text, charset, source) of an HTTP payload. The charset comes from
    the first source that decodes it: the Content-Type header ('header'),
    a <meta charset> or http-equiv tag in the first 4 KiB ('meta'),
    charset_normalizer's detector ('detector'), else UTF-8 with
    replacement characters ('fallback'). A declared charset is accepted if
    it decodes strictly or with at most 0.1% replacement characters (the
    U+FFFD count of a page is a symptom of a wrong or mixed encoding)."""
    import codecs

    def attempt(label):
        label = label.strip().strip('"\'').lower()
        try:
            name = codecs.lookup(d2l._CHARSET_ALIASES.get(label, label)).name
        except LookupError:
            return None
        try:
            return body.decode(name), name
        except UnicodeDecodeError:
            text = body.decode(name, 'replace')
            return (text, name) if text.count('�') <= 1e-3 * max(
                len(text), 1) else None

    http = headers.get('HTTP', {}) if headers else {}
    m = re.search(r'charset\s*=\s*([^\s;,]+)', http.get('content-type', ''),
                  re.I)
    if m and (got := attempt(m.group(1))):
        return got[0], got[1], 'header'
    m = re.search(rb'<meta[^>]+charset\s*=\s*["\']?([\w.:-]+)', body[:4096],
                  re.I)
    if m and (got := attempt(m.group(1).decode('ascii', 'replace'))):
        return got[0], got[1], 'meta'
    from charset_normalizer import from_bytes
    best = from_bytes(body).best()
    if best is not None:
        return str(best), codecs.lookup(best.encoding).name, 'detector'
    return body.decode('utf-8', 'replace'), 'utf-8', 'fallback'
```

```{.python .input #crawl-lib-structure}
%%tab pytorch
#@save
def _ws(s):
    """Whitespace runs collapsed to one space, ends stripped."""
    return ' '.join(s.split())

#@save
def _nows(s):
    """All whitespace removed: inline tags split or join words differently
    in every extractor, so short strings are compared without it."""
    return re.sub(r'\s+', '', s)

#@save
_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]')

#@save
def _lxml_document(html):
    """The lxml document of a page, or None: an XML declaration (which
    lxml refuses in a str) and control characters are removed first."""
    import lxml.etree
    import lxml.html
    html = _CONTROL.sub(' ', re.sub(r'^\s*<\?xml[^>]*\?>', '', html))
    try:
        return lxml.html.document_fromstring(html)
    except (ValueError, lxml.etree.ParserError):
        return None

#@save
_MATH_LOADERS = re.compile(r'mathjax|katex', re.I)

#@save
_TEX_SPANS = re.compile(r'\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$', re.S)

#@save
def html_structure(html):
    """The structural content of a page, read from its HTML alone (no
    extractor involved): code lines of <pre>/<code> blocks with at least 3
    lines of at least 3 characters; table cells (<td>/<th> without a
    nested table, 3-200 characters); mathematics (MathML TeX annotations,
    MathJax <script type="math/tex">, <img> alt text with a backslash, and
    \\(..\\), \\[..\\], $$..$$ spans when the page loads MathJax or KaTeX);
    the prose; and the content text behind the precision proxy (<p>,
    <pre>, <table>, <li>, <h1>-<h6> outside <nav>, <header>, <footer>,
    <aside> and <form>, a <li> only if under half of its text is link
    text). Strings are whitespace-free except code lines (whitespace
    collapsed); text holds the same content elements with whitespace
    collapsed, one per line, the input of a language identifier that no
    extractor has touched. classes lists 'code', 'table', 'math' when present, and
    'prose' for a page with none of them and at least 200 characters of
    prose. Prose is the text of the <p> elements with at least 100
    characters outside <nav>, <header>, <footer>, <aside> and <form>:
    short paragraphs are mostly form messages, banners and captions."""
    out = dict(code=[], table=[], math=[], prose='', content='', text='',
               classes=[])
    root = _lxml_document(html)
    if root is None:
        return out
    loads_math = any(_MATH_LOADERS.search(s.get('src') or '')
                     for s in root.iter('script'))
    for a in root.xpath('//*[local-name()="annotation"]'
                        '[@encoding="application/x-tex"]'):
        out['math'].append(_nows(a.text_content()))
    for s in root.xpath('//script[starts-with(@type, "math/tex")]'):
        out['math'].append(_nows(s.text_content()))
    for img in root.iter('img'):
        if '\\' in (img.get('alt') or ''):
            out['math'].append(_nows(img.get('alt')))
    for bad in root.xpath('//script|//style|//noscript|//template'):
        bad.drop_tree()
    if loads_math:
        body = root.find('body')
        for m in _TEX_SPANS.finditer(body.text_content() if body is not None
                                     else ''):
            out['math'].append(_nows(next(g for g in m.groups() if g)))
    out['math'] = [m for m in out['math'] if len(m) >= 2]
    for el in root.xpath('//pre|//code[not(ancestor::pre)]'):
        lines = [_ws(l) for l in el.text_content().splitlines()]
        lines = [l for l in lines if len(l) >= 3]
        if len(lines) >= 3:
            out['code'] += lines
    for table in root.iter('table'):
        cells = [_nows(' '.join(c.itertext()))
                 for c in table.xpath('.//td[not(.//table)]|'
                                      './/th[not(.//table)]')]
        cells = [c for c in cells if 3 <= len(c) <= 200]
        if len(cells) >= 4:
            out['table'] += cells
    outside = ('not(ancestor::nav or ancestor::header or ancestor::footer '
               'or ancestor::aside or ancestor::form)')
    paras = [_nows(' '.join(p.itertext()))
             for p in root.xpath(f'//p[{outside}]')]
    out['prose'] = '\n'.join(p for p in paras if len(p) >= 100)
    content = []
    for el in root.xpath(f'//*[self::p or self::pre or self::table or '
                         f'self::li or self::h1 or self::h2 or self::h3 or '
                         f'self::h4 or self::h5 or self::h6][{outside}]'):
        text = _ws(' '.join(el.itertext()))
        if el.tag == 'li':
            links = sum(len(_nows(' '.join(a.itertext())))
                        for a in el.iter('a'))
            if links > 0.5 * len(_nows(text)):
                continue
        content.append(text)
    out['content'] = '\n'.join(map(_nows, content))
    out['text'] = '\n'.join(t for t in content if t)
    out['classes'] = [c for c in ('code', 'table', 'math') if out[c]]
    if not out['classes'] and len(out['prose']) >= 200:
        out['classes'] = ['prose']
    return out

#@save
def html_skeleton(html, n=25):
    """The first n elements of a page's <body> as indented tag names, each
    text node replaced by its character count: the page's shape without
    its words (and without attributes)."""
    root = _lxml_document(html)
    body = None if root is None else root.find('body')
    lines = []

    def walk(el, depth):
        if len(lines) >= n or not isinstance(el.tag, str):
            return
        text = len((el.text or '').strip())
        lines.append('  ' * depth + el.tag + (f'  [text: {text} chars]'
                                              if text else ''))
        for child in el:
            walk(child, depth + 1)

    if body is not None:
        walk(body, 0)
    return lines

#@save
def char_ngram_f1(a, b, n=5):
    """(precision, recall, F1) of the character n-grams of text a against
    those of the reference b, as sets, after collapsing whitespace; 0.0
    for an empty side."""
    A, B = (set(s[i:i + n] for i in range(len(s) - n + 1))
            for s in (_ws(a or ''), _ws(b or '')))
    hit = len(A & B)
    p, r = (hit / len(A) if A else 0.0), (hit / len(B) if B else 0.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

#@save
_HOST_CLASSES = (  # first matching rule wins; tested on the host name
    ('edu_gov', r'(\.edu|\.gov|\.mil|\.ac\.[a-z]{2}|\.edu\.[a-z]{2}|'
                r'\.gov\.[a-z]{2})$'),
    ('wiki', r'wiki'),
    ('forum', r'forum|board|community|discuss|answers|stackexchange|'
              r'stackoverflow|reddit'),
    ('news', r'news|times|post|herald|tribune|journal|gazette|daily'),
    ('blog', r'blog|wordpress|medium\.com|substack|tumblr'),
    ('shop', r'shop|store|market|buy|cart|deals|amazon|ebay|etsy'))

#@save
def host_class(url):
    """A coarse class of the host (edu_gov, wiki, forum, news, blog, shop,
    other) from the rules in _HOST_CLASSES, matched on the lower-cased
    host name: a cheap grouping for removal audits, not a site type."""
    import urllib.parse
    host = (urllib.parse.urlsplit(url).hostname or '').lower()
    for name, rule in d2l._HOST_CLASSES:
        if re.search(rule, host):
            return name
    return 'other'
```

```{.python .input #crawl-lib-c0}
%%tab pytorch
#@save
_ISO639_3TO1 = dict(  # CLD2's WET labels (ISO 639-3) to fastText's codes
    eng='en', deu='de', fra='fr', spa='es', rus='ru', jpn='ja', zho='zh',
    ita='it', por='pt', pol='pl', nld='nl', tur='tr', ces='cs', vie='vi',
    ind='id', kor='ko', swe='sv', fas='fa', ara='ar', ukr='uk', hun='hu',
    ell='el', dan='da', fin='fi', ron='ro', nor='no', nob='no', heb='he',
    tha='th', hin='hi', bul='bg', slk='sk', cat='ca', hrv='hr', lit='lt',
    slv='sl', srp='sr', est='et', lav='lv', msa='ms', ben='bn', tam='ta')

#@save
def warc_record_at(path, offset, length):
    """The WarcRecord stored in bytes offset..offset+length-1 of a local
    WARC file: one gzip member, decompressed on its own."""
    import zlib
    with open(path, 'rb') as f:
        f.seek(offset)
        raw = f.read(length)
    d = zlib.decompressobj(31)
    data = d.decompress(raw)
    if not d.eof:
        raise ValueError(f'{path}: no complete gzip member at {offset}')
    return d2l._warc_parse(data, offset, length)

#@save
def _c0_extract_one(record, extractor='trafilatura', lid_min=0.65):
    """One response record through the c0 stages: status and MIME,
    decode_body, extract_text, detect_lang, NFC, scrub_pii. Returns a dict
    with the manifest row (no text), the final text, the extracted text
    before NFC and PII ('raw'), the decoded HTML, the reason the record
    was dropped (None if kept, else the first rule that fired: not_200,
    not_html, empty, not_en, low_score) and the extraction time. The
    section's extract cell and rebuild_c0 both call it, so a rebuild
    repeats the build exactly."""
    import importlib.metadata
    import time
    import unicodedata
    if not isinstance(record, d2l.WarcRecord):  # (path, offset, length)
        record = d2l.warc_record_at(*record)
    src = dict(warc=d2l._CC_PIN_SOURCE['warc_url'].partition(
        'commoncrawl.org/')[2], offset=record.offset, length=record.length)
    row = dict(id=record.id, url=record.url, date=record.date, src=src,
               charset=None, charset_source=None, lang=None, lang_score=None,
               extractor=extractor,
               extractor_version=(importlib.metadata.version(extractor)
                                  if extractor != 'naive' else 'd2l'),
               n_chars=0, text_sha256=None, log=[])
    out = dict(row=row, text=None, raw=None, html=None, drop=None, secs=0.0,
               pii={'email': 0, 'ip': 0})
    if record.status != 200:
        out['drop'] = 'not_200'
        return out
    if record.mime != 'text/html':
        out['drop'] = 'not_html'
        return out
    html, row['charset'], row['charset_source'] = d2l.decode_body(
        record.body, record.headers)
    out['html'] = html
    row['log'].append(f'decode:{row["charset"]}/{row["charset_source"]}')
    t = time.perf_counter()
    raw = d2l.extract_text(html, method=extractor, url=record.url)
    out['secs'] = time.perf_counter() - t
    out['raw'] = raw
    row['log'].append(f'extract:{extractor} {row["extractor_version"]}')
    if not raw:
        out['drop'] = 'empty'
        return out
    lang, score = d2l.detect_lang(raw)[0]
    row['lang'], row['lang_score'] = lang, round(score, 4)
    row['log'].append(f'lang:{lang} {row["lang_score"]:.4f}')
    if lang != 'en':
        out['drop'] = 'not_en'
        return out
    if score < lid_min:
        out['drop'] = 'low_score'
        return out
    text = unicodedata.normalize('NFC', raw)
    row['log'].append('nfc:' + ('changed' if text != raw else 'unchanged'))
    text, out['pii'] = d2l.scrub_pii(text)
    row['log'].append('pii:' + ','.join(f'{k}={v}'
                                        for k, v in out['pii'].items()))
    row['n_chars'] = len(text)
    row['text_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
    out['text'] = text
    return out

#@save
def extract_page(job):
    """The extract cell's worker for one HTML response job = (path, offset,
    length): the c0 pipeline with trafilatura (_c0_extract_one), the
    naive and Resiliparse extractions of the same decoded HTML with their
    times, html_structure, and html_lang, the top (language, score) of
    detect_lang on the structure's text ((None, 0.0) if it has none): a
    label that depends on no extractor. The decoded HTML stays in the
    worker."""
    import time
    c0 = d2l._c0_extract_one(d2l.warc_record_at(*job))
    html = c0.pop('html') or ''
    texts, secs = {'trafilatura': c0.pop('raw')}, {'trafilatura': c0['secs']}
    for m in ('naive', 'resiliparse'):
        t = time.perf_counter()
        texts[m] = d2l.extract_text(html, method=m)
        secs[m] = time.perf_counter() - t
    row, struct = c0['row'], d2l.html_structure(html)
    lang, score = (d2l.detect_lang(struct['text'])[0] if struct['text']
                   else (None, 0.0))
    return dict(id=row['id'], url=row['url'], offset=job[1],
                n_html=len(html), charset=row['charset'],
                charset_source=row['charset_source'], texts=texts, secs=secs,
                struct=struct, html_lang=(lang, round(score, 4)), c0=c0)

#@save
class DocList(list):
    """A list of Doc with the rows that failed to rebuild in .mismatches."""
    mismatches = ()

#@save
def _c0_rebuild_one(job):
    """rebuild_c0's worker: job = (path, manifest row)."""
    path, row = job
    src = row['src']
    out = d2l._c0_extract_one((path, src['offset'], src['length']),
                              row['extractor'])
    out.pop('html')
    return out

#@save
def json_agreement(new, old):
    """How far two JSON-like results agree: the number of leaf values in
    new, how many equal old's value at the same path, the largest
    absolute difference between numeric leaves present in both, the paths
    of unequal non-numeric leaves (strings, booleans, None, or a number
    against a non-number) and the paths present on one side only."""
    def leaves(x, path=()):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from leaves(v, path + (str(k),))
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                yield from leaves(v, path + (i,))
        else:
            yield path, x
    def is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    a, b = dict(leaves(new)), dict(leaves(old))
    num = [max(v - b[k], b[k] - v)  # d2l.abs is torch.abs
           for k, v in a.items() if k in b and is_num(v) and is_num(b[k])]
    other = [k for k, v in a.items() if k in b and b[k] != v
             and not (is_num(v) and is_num(b[k]))]
    return dict(values=len(a), equal=sum(k in b and b[k] == v
                                         for k, v in a.items()),
                max_diff=max(num, default=0.0), mismatches=other,
                missing=sorted(set(a) ^ set(b), key=str))
```

<!-- Quality filtering (:numref:`sec_lm_quality_filtering`): the statistics
behind `heuristic_filters`, the agreement and interval helpers, the batching
of `edu_label_scores`, the annotation sample and the released classifier.
The label below sets the "Defined in" suffix of the blocks after it. -->
:label:`sec_lm_quality_filtering`

```{.python .input #crawl-lib-quality-rules}
%%tab pytorch
#@save
# Word and punctuation tokens, close to NLTK's English word_tokenize (which
# datatrove uses): a run of word characters joined by inner apostrophes,
# hyphens, periods, commas, colons or slashes (don't, 1,000, 3.5, 10:30),
# or a run of other non-space characters. A run of Han, kana or Thai
# characters is one "word": the rules see unspaced scripts this way.
_RULE_WORD = re.compile(r"\w+(?:[.,:/'’-]\w+)*|[^\w\s]+")
# Terminal punctuation for FineWeb's line rule: the common marks of the
# scripts in this chapter (datatrove 0.10.1 lists 159 code points)
_TERMINAL_PUNCT = tuple('.!?。！？．｡؟।॥')
_C4_POLICY = ('terms of use', 'privacy policy', 'cookie policy',
              'uses cookies', 'use of cookies', 'use cookies')

#@save
def _rule_words(text):
    """The word and punctuation tokens of text (see _RULE_WORD)."""
    return d2l._RULE_WORD.findall(text)

#@save
def _rule_is_symbol(word):
    """True if a token has no letter or digit (punctuation, symbols)."""
    return not any(ch.isalnum() for ch in word)

#@save
def _rule_dups(items):
    """(number, characters) of the items that repeat an earlier item."""
    seen, n, chars = set(), 0, 0
    for x in items:
        if x in seen:
            n, chars = n + 1, chars + len(x)
        else:
            seen.add(x)
    return n, chars

#@save
def _rule_top_ngram(words, n):
    """Characters covered by the most frequent word n-gram (joined by
    spaces): its length times its count, as in Gopher's repetition rule."""
    grams = collections.Counter(' '.join(words[i:i + n])
                                for i in range(len(words) - n + 1))
    if not grams:
        return 0
    gram, count = grams.most_common(1)[0]
    return len(gram) * count

#@save
def _rule_dup_ngram(words, n):
    """Characters in word n-grams (joined without spaces) that repeat an
    earlier n-gram, skipping past each repeat, as in Gopher's rule."""
    seen, chars, i = set(), 0, 0
    while i < len(words) - n + 1:
        gram = ''.join(words[i:i + n])
        if gram in seen:
            chars, i = chars + len(gram), i + n
        else:
            seen.add(gram)
            i += 1
    return chars

#@save
def _rule_c4(text, c):
    """True if C4's document rules drop text: 'lorem ipsum' or a curly
    brace on a line that survives the line rules, or fewer than
    c['min_sentences'] sentences on those lines. As in FineWeb's run, a
    line without terminal punctuation is not removed; sentences are split
    at . ! ? followed by space (NLTK's punkt in datatrove)."""
    sentences = 0
    for line in text.splitlines():
        line = line.strip()
        words = line.split()
        if any(len(w) > c['max_word_len'] for w in words):
            continue
        line = re.sub(r'\[\d*\]|\[edit\]|\[citation needed\]', '', line)
        if len(words) < c['min_line_words']:
            continue
        low = line.lower()
        if 'lorem ipsum' in low:
            return True
        if 'javascript' in low:
            continue
        if '{' in line:
            return True
        if any(p in low for p in d2l._C4_POLICY):
            continue
        sentences += len([s for s in re.split(r'(?<=[.!?])\s+', line)
                          if s.strip()])
    return sentences < c['min_sentences']
```

```{.python .input #crawl-lib-quality-stats}
%%tab pytorch
#@save
def cohen_kappa(a, b):
    """Cohen's kappa of two label arrays: agreement beyond what the two
    marginal distributions give by chance; nan if chance agreement is 1."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else float('nan')

#@save
def quadratic_kappa(a, b, k=6):
    """Quadratic-weighted kappa of two integer scores in 0..k-1: one minus
    the observed mean squared disagreement over the one expected from the
    marginals."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    O = np.zeros((k, k))
    np.add.at(O, (a, b), 1)
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    W = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    return float(1 - (W * O).sum() / (W * E).sum())

#@save
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k / n."""
    if n == 0:
        return float('nan'), float('nan')
    p, z2 = k / n, z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return mid - half, mid + half

#@save
def binary_f1(y, pred):
    """F1 of boolean predictions against boolean labels (0 if none)."""
    y, pred = np.asarray(y, dtype=bool), np.asarray(pred, dtype=bool)
    tp = np.sum(y & pred)
    return float(2 * tp / (y.sum() + pred.sum())) if tp else 0.0

#@save
def rank_auc(y, s):
    """ROC AUC of scores s for boolean labels y (Mann-Whitney, ties at
    half)."""
    y, s = np.asarray(y, dtype=bool), np.asarray(s, dtype=float)
    r = pd.Series(s).rank().to_numpy()  # average ranks for ties
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

#@save
def _edu_batches(lengths, batch_size, max_batch_tokens):
    """Batches of indices, longest first: at most batch_size rows and at
    most max_batch_tokens tokens with left padding (rows times the first,
    longest, row's length); a row longer than the cap is alone."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches = []
    for i in order:
        b = batches[-1] if batches else None
        if (b is None or len(b) == batch_size or
                lengths[b[0]] * (len(b) + 1) > max_batch_tokens):
            batches.append([i])
        else:
            b.append(i)
    return batches
```

```{.python .input #crawl-lib-quality-data}
%%tab pytorch
#@save
_EDU_ANNOT_ROWS = 467_424  # rows of the eight released files
_EDU_ANNOT_SPLITS = (18_000, 2_000, 2_000)  # train, val, test

#@save
# The rows of the released annotations that the released FineWeb-Edu
# classifier did not train on (its card's hold-out of 46,867, reproduced
# from its training code; 46,744 survive in the release):
# tools/build_lm_edu_holdout.py
d2l.DATA_HUB['lm-edu-holdout-v1'] = (
    d2l.LM_DATA_URL + 'lm-edu-holdout-v1.json.gz',
    'dc1759992f17805864e08da5c5dcce51928e1c03')

#@save
def _edu_annot_sample():
    """The 22,000 annotation rows of lm-edu-annot-22k-v1 before labelling,
    in the order of p = default_rng(0).permutation(467,424) over the rows
    of the eight released files read in order: train is p[:18,000]; val and
    test are the next 2,000 and 2,000 rows of p[18,000:] that lie in the
    released classifier's hold-out (lm-edu-holdout-v1), so its scores on
    them are out-of-sample. Columns row, split, text, prompt (the
    labeller's input as released), llama_score, url, dump; text and prompt
    as released."""
    import huggingface_hub
    import pyarrow.parquet as pq
    _, repo, rev, _ = d2l.LM_DATA_SOURCES['edu-annotations']
    with gzip.open(d2l.download_keyed('lm-edu-holdout-v1'), 'rt') as f:
        hold = json.load(f)
    assert hold['release_revision'] == rev, 'another annotation revision'
    n_train, n_val, n_test = d2l._EDU_ANNOT_SPLITS
    p = np.random.default_rng(0).permutation(d2l._EDU_ANNOT_ROWS)
    rest = p[n_train:]
    rest = rest[np.isin(rest, hold['rows'])][:n_val + n_test]
    perm = np.concatenate([p[:n_train], rest])
    split = np.repeat(['train', 'val', 'test'], d2l._EDU_ANNOT_SPLITS)
    frames, start = [], 0
    for k in range(8):
        path = huggingface_hub.hf_hub_download(
            repo, f'data/train-{k:05d}-of-00008.parquet', revision=rev,
            repo_type='dataset')
        f = pq.ParquetFile(path)
        n = f.metadata.num_rows
        pos = np.flatnonzero((perm >= start) & (perm < start + n))
        t = f.read(columns=['text', 'prompt', 'score', 'metadata']).take(
            perm[pos] - start).to_pandas()
        frames.append(pd.DataFrame(dict(
            pos=pos, row=perm[pos], split=split[pos], text=t.text,
            prompt=t.prompt, llama_score=t.score.astype(int),
            url=[m['url'] for m in t.metadata],
            dump=[m['dump'] for m in t.metadata])))
        start += n
    assert start == d2l._EDU_ANNOT_ROWS, start
    df = pd.concat(frames).sort_values('pos', ignore_index=True)
    return df.drop(columns='pos')

#@save
# The quality labeller (from 2026-11-20; Qwen3.5-4B before): Qwen3.8-27B,
# the judge of :numref:`sec_lm_rag`, in RedHat's W4A16 build (Apache-2.0),
# one copy per 24 GB GPU served by vLLM, which lives in an environment of
# its own (D2L_VLLM); 16.7 GiB of weights per card
EDU_LABELLER_LOCAL = {'model': 'RedHatAI/Qwen3.8-27B-INT4',
                      'revision': '71dface38d4cbfa34121f1d446d4a0fa1025f13c',
                      'vllm': '0.33.0'}

#@save
def edu_label_servers(gpus=None, vllm=None):
    """One d2l.LocalChatServer per GPU (default: every visible GPU), each
    serving EDU_LABELLER_LOCAL with vLLM on that GPU alone, offline from
    the local Hub cache, from the executable vllm (default: the
    environment variable D2L_VLLM, else `vllm` on the PATH). At the
    defaults (0.9 of the memory, 64 sequences) vLLM refuses to start:
    the linear-attention layers' state cache holds 54 sequences. A server
    takes 0.95 of an idle card; on a card that already holds memory (this
    process's own, say) it takes what is free less 1.5 GiB."""
    import shutil  # not in the d2l library preamble
    import subprocess
    exe = vllm or os.environ.get('D2L_VLLM') or shutil.which('vllm')
    if exe is None:
        raise RuntimeError('no vLLM: install it in its own environment and '
                           'set D2L_VLLM to its vllm executable')
    if gpus is None:
        vis = os.environ.get('CUDA_VISIBLE_DEVICES')
        gpus = (vis.split(',') if vis else
                [str(i) for i in range(torch.cuda.device_count())])
    servers, j = [], d2l.EDU_LABELLER_LOCAL
    for g in gpus:  # nvidia-smi reads memory without opening a CUDA context
        used, total = map(int, subprocess.run(
            ['nvidia-smi', '-i', str(g), '--query-gpu=memory.used,'
             'memory.total', '--format=csv,noheader,nounits'], check=True,
            capture_output=True, text=True).stdout.split(','))
        frac = 0.95 if used <= 100 else min(0.95, (total - used - 1536)
                                            / total)
        servers.append(d2l.LocalChatServer([
            'env', f'CUDA_VISIBLE_DEVICES={g}', 'HF_HUB_OFFLINE=1', exe,
            'serve', j['model'], '--revision', j['revision'],
            '--max-model-len', '16384', '--max-num-seqs', '32',
            '--gpu-memory-utilization', f'{frac:.3f}', '--language-model-only',
            '--limit-mm-per-prompt', '{"image": 0, "video": 0}',
            '--reasoning-parser', 'qwen3', '--seed', '0', '--host',
            '127.0.0.1', '--port', '{port}']))
    return servers

#@save
def _vllm_post(url, path, body, timeout=600.0, attempts=3):
    """The JSON reply of an OpenAI-compatible server to a POST of body at
    url + path, with up to attempts tries five seconds apart."""
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url + path, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    for k in range(attempts):
        try:
            with opener.open(req, timeout=timeout) as r:
                return json.load(r)
        except OSError:
            if k == attempts - 1:
                raise
            time.sleep(5)

#@save
def _edu_label_ids(tokenizer, prompts, prefix, extra_space=False):
    """The prompts as edu_label_scores scores them: (token ids, the ids of
    '0' ... '5', the id of one more space or None). Each prompt is a user
    turn of the chat template (non-thinking) followed by prefix, tokenized
    as if the digit followed, then cut, so a prefix that ends in spaces
    splits as in the model's own text."""
    digits = [tokenizer.encode(str(d), add_special_tokens=False)
              for d in range(6)]
    if any(len(t) != 1 for t in digits):
        raise ValueError(f'the digits are not single tokens: {digits}')
    digits = [t[0] for t in digits]
    ids, space = [], None
    for p in prompts:
        chat = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) + prefix
        x = tokenizer(chat + '0', add_special_tokens=False)['input_ids']
        if x[-1] != digits[0]:
            raise ValueError(f'the digit after {prefix!r} merges with it')
        ids.append(x[:-1])
        if extra_space:  # the same text with one more space
            y = tokenizer(chat + ' 0', add_special_tokens=False)['input_ids']
            if y[:len(x) - 1] != x[:-1] or len(y) != len(x) + 1:
                raise ValueError('one more space does not add one token')
            if space not in (None, y[-2]):
                raise ValueError('the extra space is not one token id')
            space = y[-2]
    return ids, digits, space

#@save
def edu_label_scores_served(urls, tokenizer, prompts, *, model=None,
                            prefix='Educational score: ', extra_space=False,
                            concurrency=1):
    """edu_label_scores computed by OpenAI-compatible vLLM servers at urls,
    each serving model (default EDU_LABELLER_LOCAL's): the same token ids,
    sent as the prompt of a one-token greedy completion with the top 20
    log-probabilities; a digit outside them counts as 0. With extra_space,
    the probability of digit d also counts the answers that write one more
    space first: p(d) + p(space) p(d | space). Rows are spread over the
    servers in turn, at most concurrency requests in flight per server;
    they come back in input order. One at a time (the default), every
    server returns the same bits for a prompt; a batch of concurrent
    requests changes them (by up to about 0.08 for Qwen3.8-27B-INT4)."""
    import concurrent.futures
    import threading
    model = model or d2l.EDU_LABELLER_LOCAL['model']
    ids, digits, space = d2l._edu_label_ids(tokenizer, prompts, prefix,
                                            extra_space)
    want = digits + ([space] if extra_space else [])
    gates = [threading.Semaphore(concurrency) for _ in urls]

    def top(i, x):
        k = i % len(urls)
        with gates[k]:
            d = d2l._vllm_post(urls[k], '/completions', dict(
                model=model, prompt=x, max_tokens=1, temperature=0,
                logprobs=20, return_tokens_as_token_ids=True))
        lp = d['choices'][0]['logprobs']['top_logprobs'][0]
        lp = {int(t.rsplit(':', 1)[-1]): v for t, v in lp.items()}
        return [math.exp(lp[t]) if t in lp else 0.0 for t in want]

    with concurrent.futures.ThreadPoolExecutor(
            concurrency * len(urls)) as ex:
        P = np.array(list(ex.map(top, range(len(ids)), ids)))
        if not extra_space:
            return P
        more = np.flatnonzero(P[:, 6] > 0)  # one more space, then a digit
        Q = np.array(list(ex.map(top, more, [ids[i] + [space]
                                             for i in more])))
    out = P[:, :6].copy()
    if len(more):
        out[more] += P[more, 6:7] * Q[:, :6]
    return out

#@save
def _stream_half(domain):
    """The audit stream's half ('A' or 'B') of a registered domain: the
    parity of the first 32 bits of its SHA-256, so the halves share no
    domain (the bootstrap's cluster)."""
    h = int(hashlib.sha256(domain.encode('utf-8')).hexdigest()[:8], 16)
    return 'A' if h % 2 == 0 else 'B'

#@save
def _cctld(url):
    """The last label of the host's public suffix with a dot ('.uk' for
    example.co.uk, '.com'), from the same tldextract snapshot as
    registered_domain; '' for an IP address."""
    d2l.registered_domain('http://example.com')  # makes the extractor
    suffix = d2l._TLD_EXTRACT[0](url).suffix
    return '.' + suffix.rsplit('.', 1)[-1].lower() if suffix else ''

#@save
def released_edu_scores(texts, batch=64, device=None, dtype=None):
    """Scores of the released FineWeb-Edu classifier (a regression head on
    a frozen Snowflake arctic-embed-m encoder; Apache-2.0) at its pinned
    revision: the first 512 tokens of each text, length-sorted batches,
    float32 scores in input order. bf16 on a GPU, fp32 on the CPU."""
    import transformers
    _, repo, rev, _ = d2l.LM_DATA_SOURCES['fineweb-edu-classifier']
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = dtype or (torch.bfloat16 if device == 'cuda' else torch.float32)
    tok = transformers.AutoTokenizer.from_pretrained(repo, revision=rev)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        repo, revision=rev, dtype=dtype).to(device).eval()
    ids = tok(list(texts), truncation=True, max_length=512)['input_ids']
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
    out = np.empty(len(ids), dtype=np.float32)
    for j in range(0, len(order), batch):
        rows = order[j:j + batch]
        enc = tok.pad({'input_ids': [ids[i] for i in rows]},
                      return_tensors='pt').to(device)
        with torch.inference_mode():
            out[rows] = model(**enc).logits[:, 0].float().cpu().numpy()
    del model
    d2l.lm_free()
    return out

#@save
def c0_documents(manifest='lm-c0-docs-v1', cache_dir=None):
    """The c0 documents: the cached rebuild in cache_dir (default
    ../data/lm/c0) if its ids and text hashes equal the manifest rows,
    else rebuild_c0 (which writes that cache)."""
    import pathlib
    cache = pathlib.Path(cache_dir or os.path.join('..', 'data', 'lm', 'c0'))
    path = (d2l.download_keyed(manifest) if manifest in d2l.DATA_HUB
            else manifest)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        want = {r['id']: r['text_sha256'] for r in map(json.loads, f)}
    if (cache / 'c0-docs.jsonl.gz').exists():
        docs = d2l.read_docs(cache / 'c0-docs.jsonl.gz')
        got = {d.id: hashlib.sha256(d.text.encode('utf-8')).hexdigest()
               for d in docs}
        if got == want:
            return docs
    docs = d2l.rebuild_c0(manifest, cache_dir=cache)
    assert not docs.mismatches, f'{len(docs.mismatches)} c0 rows drifted'
    return docs
```

<!-- Deduplication and decontamination (:numref:`sec_lm_dedup`): the
vectorized MinHash rows behind `minhash_signature` and their parallel
driver, the perturbations behind `make_dup_pairs` (with the boilerplate
blocks), the Poisson-binomial interval of the S-curve check, the benchmark
items, the planted documents and the embedding windows of the
decontamination detectors. The label below sets the "Defined in" suffix
of the blocks after it. -->
:label:`sec_lm_dedup`

```{.python .input #crawl-lib-dedup-minhash}
%%tab pytorch
#@save
_MINHASH_PARAMS = {}  # (k, seed) -> the rows' (a, c), drawn once

#@save
def _shingle_hashes(shingle_list):
    """The 64-bit shingle hashes of the MinHash specification: blake2b with
    an 8-byte digest of the UTF-8 shingle, read little-endian (uint64)."""
    digests = b''.join(hashlib.blake2b(s.encode('utf-8'), digest_size=8)
                       .digest() for s in shingle_list)
    return np.frombuffer(digests, dtype='<u8').astype(np.uint64)

#@save
def _minhash_params(k, seed):
    """The k rows' coefficients (a, c), each uint64 in [1, p), from
    numpy.random.default_rng(seed) as the specification draws them."""
    if (k, seed) not in d2l._MINHASH_PARAMS:
        a, c = np.random.default_rng(seed).integers(
            1, d2l.MINHASH_PRIME, size=(2, k), dtype=np.uint64)
        d2l._MINHASH_PARAMS[k, seed] = (a, c)
    return d2l._MINHASH_PARAMS[k, seed]

#@save
def _mulmod61(a, x):
    """(a * x) mod p for p = 2^61 - 1 and uint64 arrays a, x < 2^61
    (broadcast), exact. The product needs 122 bits, so both factors are
    split into 32-bit halves, whose partial products fit in 64 bits, and
    the high bits fold back because 2^61 = 1 (mod p)."""
    p, m32 = np.uint64(d2l.MINHASH_PRIME), np.uint64(0xFFFFFFFF)
    a_hi, a_lo = a >> np.uint64(32), a & m32
    x_hi, x_lo = x >> np.uint64(32), x & m32
    lo = a_lo * x_lo                     # < 2^64
    mid = a_hi * x_lo + a_lo * x_hi      # < 2^62, weight 2^32
    hi = a_hi * x_hi                     # < 2^58, weight 2^64 = 8 (mod p)
    s = ((lo & p) + (lo >> np.uint64(61))
         + ((mid & np.uint64((1 << 29) - 1)) << np.uint64(32))
         + (mid >> np.uint64(29)) + (hi << np.uint64(3)))   # < 2^63
    s = (s & p) + (s >> np.uint64(61))
    return np.where(s >= p, s - p, s)

#@save
def _minhash_rows(x, a, c, chunk=4096):
    """min over the hashes x (mod p already) of (a_i x + c_i) mod p for
    each row i: the signature, computed chunk by chunk of shingles."""
    p = np.uint64(d2l.MINHASH_PRIME)
    sig = np.full(len(a), p, dtype=np.uint64)
    for j in range(0, len(x), chunk):
        v = d2l._mulmod61(a[:, None], x[None, j:j + chunk]) + c[:, None]
        v = np.where(v >= p, v - p, v)
        sig = np.minimum(sig, v.min(axis=1))
    return sig

#@save
def _minhash_chunk(job):
    """_minhash_texts' worker: job = (texts, k, seeds, ngram); one list of
    signatures (None for fewer than ngram words) per seed."""
    texts, k, seeds, ngram = job
    out = {s: [] for s in seeds}
    for t in texts:
        sh = d2l.shingles(t, ngram)
        x = (np.unique(d2l._shingle_hashes(sh)) % np.uint64(
            d2l.MINHASH_PRIME)) if sh else None
        for s in seeds:
            out[s].append(None if x is None else
                          d2l._minhash_rows(x, *d2l._minhash_params(k, s)))
    return out

#@save
def _minhash_texts(texts, k=128, seeds=(0xD2,), ngram=5, workers=1):
    """{seed: [signature or None per text]} for many texts, each text
    shingled and hashed once for all seeds; with workers > 1 in separate
    processes (chunks of 500 texts, results in input order, so the output
    does not depend on workers)."""
    import concurrent.futures
    import multiprocessing
    texts = [str(t) for t in texts]
    jobs = [(texts[i:i + 500], k, tuple(seeds), ngram)
            for i in range(0, len(texts), 500)]
    if workers <= 1 or len(jobs) < 2:
        outs = map(d2l._minhash_chunk, jobs)
    else:
        ex = concurrent.futures.ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context('spawn'))
        with ex:
            outs = list(ex.map(d2l._minhash_chunk, jobs))
    res = {s: [] for s in seeds}
    for out in outs:
        for s in seeds:
            res[s] += out[s]
    return res

#@save
def _sig_matrix(sigs, k):
    """Signatures (a list with None for empty documents) as a (n, k) uint64
    array and the boolean mask of documents that have one."""
    ok = np.array([s is not None for s in sigs])
    S = np.zeros((len(sigs), k), dtype=np.uint64)
    if ok.any():
        S[ok] = np.stack([s for s in sigs if s is not None])
    return S, ok

#@save
def _band_match(SA, SB, b, r):
    """Per pair, whether the signature rows SA[i] and SB[i] agree in all r
    rows of at least one of the first b bands: lsh_candidates' decision
    for one known pair, vectorized (equal up to 8-byte key collisions)."""
    eq = SA[:, :b * r] == SB[:, :b * r]
    return eq.reshape(len(SA), b, r).all(axis=2).any(axis=1)
```

```{.python .input #crawl-lib-dedup-pairs}
%%tab pytorch
#@save
DUP_BOILERPLATE = (  # 40 generic template blocks for the 'boiler' perturbation
    'We use cookies to improve your experience on our site. By continuing '
    'to browse you agree to our use of cookies. Learn more in our cookie '
    'policy.',
    'Subscribe to our newsletter and get the latest news, offers and '
    'updates delivered straight to your inbox every week. You can '
    'unsubscribe at any time.',
    'Home | About Us | Services | Blog | Contact | Privacy Policy | Terms '
    'and Conditions | Sitemap',
    'Share this article: Facebook Twitter LinkedIn Pinterest Email Print',
    'All rights reserved. No part of this website may be reproduced, '
    'distributed or transmitted in any form without the prior written '
    'permission of the publisher.',
    'This post may contain affiliate links. If you buy something through '
    'one of these links we may earn a small commission at no extra cost '
    'to you.',
    'Leave a Reply. Your email address will not be published. Required '
    'fields are marked. Name Email Website Comment',
    'Related posts: you might also like these articles from our archive, '
    'chosen for readers who enjoyed this one.',
    'Free shipping on all orders over fifty dollars. Easy returns within '
    'thirty days. Secure checkout with all major cards.',
    'Disclaimer: the information on this page is for general purposes only '
    'and does not constitute professional advice. Consult a qualified '
    'expert before acting on it.',
    'Click here to download the free guide. Enter your name and email '
    'address below and we will send it right away.',
    'Posted in Uncategorized | Tagged news, updates, tips | Leave a comment',
    'Previous article Next article Back to top',
    'Skip to main content Search this site Menu Close menu Log in Register',
    'Copyright the authors. Content on this site is licensed for personal '
    'use only; please contact us for permission to republish.',
    'Follow us on social media for daily updates, behind the scenes photos '
    'and special announcements from our team.',
    'Your cart is empty. Continue shopping or sign in to see items you '
    'saved during an earlier visit.',
    'This website uses analytics to understand how visitors use it. We do '
    'not sell your personal data. Read our privacy notice for details.',
    'Accept all cookies Reject non-essential cookies Manage preferences',
    'Sign up for a free account to save your favourite pages, comment on '
    'articles and receive personalised recommendations.',
    'Advertisement. Continue reading below.',
    'Was this page helpful? Yes No. Thank you for your feedback.',
    'Terms of use: by accessing this site you accept these terms in full. '
    'If you disagree with any part of them you must not use the site.',
    'Contact us: send us a message using the form below and a member of '
    'our team will get back to you within two working days.',
    'Recent Posts Archives Categories Meta Log in Entries feed Comments feed',
    'Print this page Email this page Bookmark this page',
    'Last updated on the date shown above. Prices and availability are '
    'subject to change without notice.',
    'Join thousands of readers who already get our weekly digest. No spam, '
    'we promise. Your details are safe with us.',
    'You are here: Home > Articles > Current page',
    'Read more Show less Load more comments',
    'Our support team is available Monday to Friday from nine to five. '
    'Call us or open a ticket in the help centre.',
    'Images are for illustration purposes only and may not show the exact '
    'product described on this page.',
    'Tags: guide, how to, beginners, tips and tricks, frequently asked '
    'questions',
    'Download our app for a faster experience on your phone. Available for '
    'all major platforms.',
    'Please enable JavaScript in your browser to view this content and use '
    'all the features of this site.',
    'Rate this article: one star two stars three stars four stars five '
    'stars. Average rating based on reader votes.',
    'About the author: a writer and editor who covers everyday topics for '
    'a general audience. Views expressed are their own.',
    'Back to the list of all articles in this category.',
    'Sponsored content. This article was produced in partnership with an '
    'advertiser and reviewed by our editorial team.',
    'Thank you for visiting. Come back soon for new articles every week.')

#@save
def _dup_vocab(texts):
    """The substitution vocabulary of make_dup_pairs: the sorted distinct
    lowercased \\w+ words of texts."""
    import regex
    return sorted({w for t in texts for w in regex.findall(r'\w+',
                                                            t.lower())})

#@save
def _dup_perturb(text, op, params, vocab=None):
    """A perturbed copy of text by op (sub, del, boiler, shuffle, trunc)
    with params; every random choice comes from default_rng(params['seed'])."""
    import regex
    rng = np.random.default_rng(params['seed'])
    if op in ('sub', 'del'):
        pat = r'\w+' if op == 'sub' else r'\w+[^\S\n]*'
        spans = [m.span() for m in regex.finditer(pat, text)]
        m = max(1, round(params['rate'] * len(spans)))
        pick = set(rng.choice(len(spans), m, replace=False).tolist())
        if op == 'sub' and vocab is None:
            raise ValueError("'sub' needs the substitution vocabulary")
        out, at = [], 0
        for i, (s, e) in enumerate(spans):
            if i in pick:
                out.append(text[at:s])
                if op == 'sub':
                    out.append(vocab[int(rng.integers(len(vocab)))])
                at = e
        return ''.join(out) + text[at:]
    paras = text.split('\n')
    if op == 'boiler':
        words = []
        for j in rng.permutation(len(d2l.DUP_BOILERPLATE)):
            words += d2l.DUP_BOILERPLATE[j].split()
            if len(words) >= params['words']:
                break
        paras.insert(int(rng.integers(len(paras) + 1)),
                     ' '.join(words[:params['words']]))
        return '\n'.join(paras)
    if op == 'shuffle':
        return '\n'.join(paras[i] for i in rng.permutation(len(paras)))
    if op == 'trunc':
        ends = [m.end() for m in regex.finditer(r'\w+', text)]
        return text[:ends[max(1, round(params['frac'] * len(ends))) - 1]]
    raise ValueError(f'unknown op {op!r}')

#@save
def _url_host(url):
    """The lowercased host name of a URL ('' if it has none)."""
    import urllib.parse
    return (urllib.parse.urlsplit(url).hostname or '').lower()

#@save
def _pb_interval(p, level=0.95):
    """Central interval (lo, hi) of a Poisson-binomial count, the number of
    successes of independent trials with probabilities p: lo is the
    smallest count with P(X <= lo) > (1 - level) / 2 and hi the smallest
    with P(X <= hi) >= (1 + level) / 2, from the exact distribution."""
    p = np.asarray(p, dtype=float)
    ones, p = int(np.sum(p >= 1.0)), p[(p > 0.0) & (p < 1.0)]
    pmf = np.ones(1)
    for q in p:
        pmf = np.append(pmf * (1 - q), 0.0) + np.append(0.0, pmf * q)
    cdf = np.cumsum(pmf)
    alpha = (1 - level) / 2
    lo = int(np.searchsorted(cdf, alpha, side='right'))
    hi = int(np.searchsorted(cdf, 1 - alpha - 1e-12))
    return lo + ones, hi + ones
```

```{.python .input #crawl-lib-dedup-contam}
%%tab pytorch
#@save
def benchmark_items():
    """The test items of GSM8K (1,319), ARC-Challenge (1,172) and MMLU
    (14,042) at their pinned files: dicts with item_id, benchmark,
    question, choices (labels and texts, or None), gold (answer label or
    final number) and text, the indexed form: question and gold answer
    text."""
    items = []
    gsm = pd.read_parquet(d2l.download_keyed('gsm8k-test'))
    for i, row in gsm.iterrows():
        items.append(dict(item_id=f'gsm8k-{i}', benchmark='gsm8k',
                          question=row.question.strip(), choices=None,
                          gold=row.answer.split('####')[-1].strip(),
                          text=row.question.strip() + '\n' + row.answer))
    arc = pd.read_parquet(d2l.download_keyed('arc-challenge-test'))
    for _, row in arc.iterrows():
        labels = [str(x) for x in row.choices['label']]
        texts = [str(x) for x in row.choices['text']]
        gold = texts[labels.index(str(row.answerKey))]
        items.append(dict(item_id=f'arc-{row.id}', benchmark='arc',
                          question=row.question.strip(),
                          choices=list(zip(labels, texts)),
                          gold=str(row.answerKey),
                          text=row.question.strip() + '\n' + gold))
    rows = d2l._eval_load_rows(d2l.EVAL_TASKS['mmlu'], 'test')
    for i, row in enumerate(rows):
        texts = [str(x) for x in row['choices']]
        items.append(dict(item_id=f'mmlu-{i}', benchmark='mmlu',
                          question=row['question'].strip(),
                          choices=list(zip('ABCD', texts)),
                          gold='ABCD'[int(row['answer'])],
                          text=row['question'].strip() + '\n'
                          + texts[int(row['answer'])]))
    return items

#@save
def _plant_text(item, form):
    """A benchmark item as a page would show it: the question, the lettered
    options (ARC, MMLU) and an answer line. form 'reformat' changes it by
    rule: lower case, one space for every whitespace run, commas and
    semicolons before a space dropped, a space before '?', the options in
    reverse order with their letters reassigned."""
    choices = item['choices'] or []
    gold = item['gold']
    if form == 'reformat' and choices:
        old = dict(choices)[gold]
        choices = [(lab, t) for (lab, _), (_, t) in zip(
            choices, reversed(choices))]
        gold = next(lab for lab, t in choices if t == old)
    fmt = '({}) {}' if item['benchmark'] == 'arc' else '{}. {}'
    lines = [item['question']] + [fmt.format(lab, t) for lab, t in choices]
    text = '\n'.join(lines + [f'Answer: {gold}'])
    if form == 'reformat':
        text = re.sub(r'[,;](?=\s)', '', text.lower())
        text = re.sub(r'\s+', ' ', text.replace('?', ' ?')).strip()
    return text

#@save
def _sentence_starts(text):
    """Positions where a sentence or a line starts (plant insertion points),
    with 0 and len(text)."""
    import regex
    pos = {0, len(text)}
    pos.update(m.end() for m in regex.finditer(r'(?<=[.!?])\s+|\n', text))
    return sorted(pos)

#@save
def make_plants(items, stream, paraphrase, n_per=100, n_controls=1000,
                seed=0, min_words=13, max_words=180, prompt_sha256=None,
                model_revision=None):
    """The planted decontamination test set: n_per items of each benchmark
    (items whose question has at least min_words words and whose page form
    at most max_words), each in three forms (verbatim, reformat,
    paraphrase = paraphrase(list of verbatim texts) -> list of texts),
    inserted at a sentence start of distinct stream documents (stream: a
    DataFrame with id and text), plus n_controls untouched documents. One
    row per document: doc_id, item_id, benchmark, form ('control' for the
    controls), text (the planted text), char_start, char_end (its span in
    the planted document), prompt_sha256, model_revision and source_hash
    (SHA-256 of the document before planting). All choices from
    default_rng(seed)."""
    import regex
    rng = np.random.default_rng(seed)
    nwords = lambda t: len(regex.findall(r'\w+', t))
    chosen = []
    for bench in ('gsm8k', 'arc', 'mmlu'):
        ok = [it for it in items if it['benchmark'] == bench and
              nwords(it['question']) >= min_words and
              nwords(d2l._plant_text(it, 'verbatim')) <= max_words]
        chosen += [ok[i] for i in rng.choice(len(ok), n_per, replace=False)]
    verbatim = [d2l._plant_text(it, 'verbatim') for it in chosen]
    para = list(paraphrase(verbatim))
    assert len(para) == len(chosen)
    plants = []
    for it, v, ph in zip(chosen, verbatim, para):
        plants += [(it, 'verbatim', v),
                   (it, 'reformat', d2l._plant_text(it, 'reformat')),
                   (it, 'paraphrase', ph)]
    order = rng.permutation(len(stream))
    assert len(order) >= len(plants) + n_controls
    rows = []
    for k, i in enumerate(order[:len(plants) + n_controls]):
        doc = stream.iloc[int(i)]
        row = dict(doc_id=doc.id, item_id=None, benchmark=None,
                   form='control', text=None, char_start=None, char_end=None,
                   prompt_sha256=None, model_revision=None,
                   source_hash=hashlib.sha256(doc.text.encode('utf-8'))
                   .hexdigest())
        if k < len(plants):
            it, form, text = plants[k]
            at = d2l._sentence_starts(doc.text)
            pos = at[int(rng.integers(len(at)))]
            row.update(item_id=it['item_id'], benchmark=it['benchmark'],
                       form=form, text=text, char_start=pos + 1,
                       char_end=pos + 1 + len(text))
            if form == 'paraphrase':
                row.update(prompt_sha256=prompt_sha256,
                           model_revision=model_revision)
        rows.append(row)
    return rows

#@save
def contam_documents(plants, stream):
    """The documents of the planted set, doc_id -> text: each planted text
    inserted as its own line at char_start - 1 of the stream document
    (document[:char_start - 1] + newline + text + newline + rest), the
    controls unchanged; the source document's SHA-256 is checked."""
    text = dict(zip(stream.id, stream.text))
    out = {}
    for row in plants:
        t = text[row['doc_id']]
        got = hashlib.sha256(t.encode('utf-8')).hexdigest()
        assert got == row['source_hash'], f'{row["doc_id"]} changed'
        if row['form'] != 'control':
            at = row['char_start'] - 1
            t = t[:at] + '\n' + row['text'] + '\n' + t[at:]
            assert t[row['char_start']:row['char_end']] == row['text']
        out[row['doc_id']] = t
    return out

#@save
def word_windows(text, size=128, stride=64):
    """Windows of size whitespace-separated words every stride words, the
    last one ending at the text's end: [(first word index, text)]."""
    words = text.split()
    starts = list(range(0, max(len(words) - size, 0) + 1, stride))
    if starts[-1] + size < len(words):
        starts.append(len(words) - size)
    return [(s, ' '.join(words[s:s + size])) for s in starts]
```
