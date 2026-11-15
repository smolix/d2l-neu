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
    new, how many equal old's value at the same path, and the largest
    absolute difference between numeric leaves present in both."""
    def leaves(x, path=()):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from leaves(v, path + (str(k),))
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                yield from leaves(v, path + (i,))
        else:
            yield path, x
    a, b = dict(leaves(new)), dict(leaves(old))
    num = [max(v - b[k], b[k] - v)  # d2l.abs is torch.abs
           for k, v in a.items() if k in b
           and isinstance(v, (int, float)) and not isinstance(v, bool)
           and isinstance(b[k], (int, float))]
    return dict(values=len(a), equal=sum(k in b and b[k] == v
                                         for k, v in a.items()),
                max_diff=max(num, default=0.0))
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
def _edu_annot_sample():
    """The 22,000 annotation rows of lm-edu-annot-22k-v1 before labelling:
    rows default_rng(0).permutation(467,424)[:22,000] of the eight released
    files read in order; the first 18,000 are train, then 2,000 val and
    2,000 test. Columns row, split, text, prompt (the labeller's input as
    released), llama_score, url, dump; text and prompt as released."""
    import huggingface_hub
    import pyarrow.parquet as pq
    _, repo, rev, _ = d2l.LM_DATA_SOURCES['edu-annotations']
    perm = np.random.default_rng(0).permutation(d2l._EDU_ANNOT_ROWS)[:22_000]
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
