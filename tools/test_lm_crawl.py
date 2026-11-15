#!/usr/bin/env python3
"""CPU tests for sec_lm_crawl (chapter_lm-data/crawl.md), the shared part
written before the section: `iter_warc` against warcio on a synthetic
gzip-member WARC with a truncated tail (ids, types, offsets, lengths and
bodies exact; the cut member counted), the HTTP header parse, a byte-prefix
read; `fetch_warc_range` against a fixture pin with 1 KiB blocks (verified
cache, a corrupted block cut back, D2L_OFFLINE, the WET file); the pin
builder; `Doc` I/O, `docs_hash`; `CorpusManifest` (identity assertion,
canonical JSON under key order, rounding, absolute paths, save/load, card);
`registered_domain`.

Gated: the 4 MiB prefix of the pinned WARC against warcio runs when the
range is in data/lm/cc/ (no network); D2L_TEST_NET=1 fetches the first
16 MiB block from Common Crawl into a temporary directory."""
import contextlib
import gzip
import hashlib
import io
import json
import os
import pathlib
import tempfile
import unittest

from d2l import torch as d2l

ROOT = pathlib.Path(__file__).resolve().parents[1]
NET = os.environ.get('D2L_TEST_NET') == '1'
CC_CACHE = ROOT / 'data' / 'lm' / 'cc' / 'lm-cc-warc-256m-v1.warc.gz'


def warc_record(kind, rid, block, content_type, uri=None):
    head = [b'WARC/1.0', f'WARC-Type: {kind}'.encode(),
            f'WARC-Record-ID: {rid}'.encode(),
            b'WARC-Date: 2025-01-26T13:54:02Z']
    if uri:
        head.append(f'WARC-Target-URI: {uri}'.encode())
    head += [f'Content-Type: {content_type}'.encode(),
             f'Content-Length: {len(block)}'.encode()]
    return b'\r\n'.join(head) + b'\r\n\r\n' + block + b'\r\n\r\n'


def synthetic_warc(truncate=True):
    """Five gzip members (warcinfo, request, response, metadata, a second
    response) and, if truncate, the first half of a sixth member."""
    html = '<html><body><p>Grüße, ¿qué tal?</p></body></html>'.encode()
    resp = (b'HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=UTF-8\r\n'
            b'X-Test: a\r\nx-test: b\r\n\r\n' + html)
    recs = [
        warc_record('warcinfo', '<urn:uuid:0>', b'software: test\r\n',
                    'application/warc-fields'),
        warc_record('request', '<urn:uuid:1>',
                    b'GET / HTTP/1.1\r\nHost: example.com\r\n\r\n',
                    'application/http; msgtype=request', 'https://example.com/'),
        warc_record('response', '<urn:uuid:2>', resp,
                    'application/http; msgtype=response', 'https://example.com/'),
        warc_record('metadata', '<urn:uuid:3>', b'fetchTimeMs: 12\r\n',
                    'application/warc-fields', 'https://example.com/'),
        warc_record('response', '<urn:uuid:4>',
                    b'HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n'
                    b'\r\nmissing', 'application/http; msgtype=response',
                    'https://example.com/x'),
        warc_record('response', '<urn:uuid:5>', resp + b'tail' * 200,
                    'application/http; msgtype=response', 'https://e.org/')]
    members = [gzip.compress(r, mtime=0) for r in recs]
    if truncate:
        members[-1] = members[-1][:len(members[-1]) // 2]
    else:
        members = members[:-1]
    return b''.join(members), [len(m) for m in members]


def warcio_records(data):
    from warcio.archiveiterator import ArchiveIterator
    out, it = [], ArchiveIterator(io.BytesIO(data))
    with contextlib.suppress(Exception):
        for rec in it:
            body = rec.raw_stream.read()
            rid = rec.rec_headers.get_header('WARC-Record-ID')
            offset = it.get_record_offset()
            it.read_to_end(rec)
            out.append((rid, rec.rec_type, offset, it.get_record_length(),
                        body))
    return out


class IterWarcTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data, self.sizes = synthetic_warc()
        self.path = os.path.join(self.tmp.name, 'x.warc.gz')
        with open(self.path, 'wb') as f:
            f.write(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def test_against_warcio(self):
        stats = {}
        ours = list(d2l.iter_warc(self.path, types=None, stats=stats))
        self.assertEqual(len(ours), 5)
        self.assertEqual(stats, {'warcinfo': 1, 'request': 1, 'response': 2,
                                 'metadata': 1, 'truncated': 1})
        theirs = warcio_records(self.data)
        # warcio also yields the cut-off sixth record, partially
        self.assertGreaterEqual(len(theirs), 5)
        for r, t in zip(ours, theirs):
            self.assertEqual((r.id, r.type, r.offset, r.length, r.body), t)
        offsets = [0]
        for n in self.sizes[:-1]:
            offsets.append(offsets[-1] + n)
        self.assertEqual([r.offset for r in ours], offsets[:5])
        self.assertEqual([r.length for r in ours], self.sizes[:5])

    def test_fields(self):
        resp = list(d2l.iter_warc(self.path))  # responses only by default
        self.assertEqual([r.id for r in resp], ['<urn:uuid:2>', '<urn:uuid:4>'])
        r = resp[0]
        self.assertEqual((r.status, r.mime, r.url), (200, 'text/html',
                                                     'https://example.com/'))
        self.assertEqual(r.headers['HTTP']['content-type'],
                         'text/html; charset=UTF-8')
        self.assertEqual(r.headers['HTTP']['x-test'], 'a, b')
        self.assertEqual(r.headers['WARC-Type'], 'response')
        self.assertTrue(r.body.startswith(b'<html>'))
        self.assertEqual((resp[1].status, resp[1].body), (404, b'missing'))
        info, req = list(d2l.iter_warc(self.path, types=('warcinfo',
                                                         'request')))
        self.assertEqual((info.status, info.mime, info.body),
                         (None, 'application/warc-fields',
                          b'software: test\r\n'))
        self.assertIsNone(req.status)

    def test_prefix(self):
        cut = sum(self.sizes[:3]) + 10  # inside the fourth member
        stats = {}
        recs = list(d2l.iter_warc(self.path, n_bytes=cut, types=None,
                                  stats=stats))
        self.assertEqual([r.type for r in recs], ['warcinfo', 'request',
                                                  'response'])
        self.assertEqual(stats['truncated'], 1)
        whole, _ = synthetic_warc(truncate=False)
        with open(self.path, 'wb') as f:
            f.write(whole)
        stats = {}
        self.assertEqual(len(list(d2l.iter_warc(self.path, types=None,
                                                stats=stats))), 5)
        self.assertNotIn('truncated', stats)

    @unittest.skipUnless(CC_CACHE.exists(), 'pinned WARC range not cached')
    def test_pinned_prefix_against_warcio(self):
        n = 4 * 2**20
        with open(CC_CACHE, 'rb') as f:
            data = f.read(n)
        stats = {}
        ours = list(d2l.iter_warc(CC_CACHE, n_bytes=n, types=None,
                                  stats=stats))
        theirs = warcio_records(data)
        self.assertEqual(stats['truncated'], 1)
        self.assertEqual(len(ours), 857)
        for r, t in zip(ours, theirs):
            self.assertEqual((r.id, r.type, r.offset, r.length, r.body), t)


class FetchTest(unittest.TestCase):
    """fetch_warc_range against a fixture pin with 1 KiB blocks; the pin
    is placed where download_keyed finds it, so nothing is downloaded."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = pathlib.Path(self.tmp.name)
        self.work = t / 'work'
        self.work.mkdir()
        self.cache = t / 'cc'
        self.warc_bytes = bytes(range(256)) * 16  # 4 KiB = 4 blocks
        self.wet_bytes = b'wet file'
        src = dict(crawl='TEST', warc_url='https://example.invalid/w.warc.gz',
                   warc_size=10_000, n_bytes=4096, block_bytes=1024,
                   wet_url='https://example.invalid/w.warc.wet.gz',
                   wet_size=len(self.wet_bytes))
        (t / 'w.warc.gz').write_bytes(self.warc_bytes)
        (t / 'w.warc.wet.gz').write_bytes(self.wet_bytes)
        pin = d2l._cc_pin(t / 'w.warc.gz', t / 'w.warc.wet.gz', src)
        self.pin = pin
        pin_bytes = d2l._cc_pin_bytes(pin)
        key_dir = t / 'data' / 'lm' / 'test-pin-v1'
        key_dir.mkdir(parents=True)
        (key_dir / 'test-pin-v1.pin.json').write_bytes(pin_bytes)
        d2l.DATA_HUB['test-pin-v1'] = (
            'https://example.invalid/test-pin-v1.pin.json',
            hashlib.sha1(pin_bytes).hexdigest())
        self.cwd = os.getcwd()
        os.chdir(self.work)
        self.env = os.environ.get('D2L_OFFLINE')
        os.environ['D2L_OFFLINE'] = '1'

    def tearDown(self):
        os.chdir(self.cwd)
        del d2l.DATA_HUB['test-pin-v1']
        if self.env is None:
            os.environ.pop('D2L_OFFLINE', None)
        else:
            os.environ['D2L_OFFLINE'] = self.env
        self.tmp.cleanup()

    def test_pin(self):
        w = self.pin['warc']
        self.assertEqual(w['range'], [0, 4095])
        self.assertEqual(len(w['block_sha256']), 4)
        self.assertEqual(w['block_sha256'][1], hashlib.sha256(
            self.warc_bytes[1024:2048]).hexdigest())
        self.assertEqual(w['sha256'],
                         hashlib.sha256(self.warc_bytes).hexdigest())
        self.assertEqual(self.pin['wet']['sha256'],
                         hashlib.sha256(self.wet_bytes).hexdigest())
        self.assertIsNone(self.pin['fineweb_file_path'])
        real = d2l._CC_PIN_SOURCE['warc_url']
        self.assertTrue(real.startswith('https://data.commoncrawl.org/'))
        b = d2l._cc_pin_bytes(self.pin)
        self.assertEqual(b, d2l._cc_pin_bytes(json.loads(b)))
        self.assertTrue(b.endswith(b'}\n'))

    def test_offline(self):
        self.cache.mkdir()
        path = self.cache / 'test-pin-v1.warc.gz'
        path.write_bytes(self.warc_bytes[:2048])
        got = d2l.fetch_warc_range('test-pin-v1', n_bytes=2048,
                                   cache_dir=self.cache)
        self.assertEqual(got, path)
        with self.assertRaisesRegex(RuntimeError, 'D2L_OFFLINE'):
            d2l.fetch_warc_range('test-pin-v1', cache_dir=self.cache)
        with self.assertRaises(ValueError):
            d2l.fetch_warc_range('test-pin-v1', n_bytes=1000,
                                 cache_dir=self.cache)
        # a corrupted second block: the file is cut back to the first
        bad = bytearray(self.warc_bytes)
        bad[1500] ^= 1
        path.write_bytes(bytes(bad))
        with self.assertRaisesRegex(RuntimeError, '1 verified blocks'):
            d2l.fetch_warc_range('test-pin-v1', n_bytes=2048,
                                 cache_dir=self.cache)
        self.assertEqual(path.stat().st_size, 1024)
        path.write_bytes(self.warc_bytes)
        self.assertEqual(d2l.fetch_warc_range('test-pin-v1',
                                              cache_dir=self.cache), path)

    def test_wet(self):
        self.cache.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'D2L_OFFLINE'):
            d2l.fetch_warc_range('test-pin-v1', which='wet',
                                 cache_dir=self.cache)
        (self.cache / 'w.warc.wet.gz').write_bytes(self.wet_bytes)
        self.assertEqual(d2l.fetch_warc_range('test-pin-v1', which='wet',
                                              cache_dir=self.cache),
                         self.cache / 'w.warc.wet.gz')
        with self.assertRaises(ValueError):
            d2l.fetch_warc_range('test-pin-v1', which='wat',
                                 cache_dir=self.cache)


@unittest.skipUnless(NET, 'network test: set D2L_TEST_NET=1')
class FetchNetTest(unittest.TestCase):
    def test_first_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(ROOT / 'chapter_lm-data')
            try:
                path = d2l.fetch_warc_range(n_bytes=16 * 2**20,
                                            cache_dir=tmp)
                with open(d2l.download_keyed('lm-cc-warc-256m-v1')) as f:
                    pin = json.load(f)
            finally:
                os.chdir(cwd)
            self.assertEqual(d2l._sha256_file(path),
                             pin['warc']['block_sha256'][0])


class DocsTest(unittest.TestCase):
    def docs(self):
        return [d2l.Doc(f'<urn:uuid:{i}>', f'https://e.org/{i}', '2025-01-26',
                        f'text {i} é', 'en', 0.9, dict(warc='w', offset=i,
                                                       length=1),
                        ['nfc'], {}) for i in range(5)]

    def test_round_trip_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, 'a.jsonl.gz'), os.path.join(tmp, 'b.gz')
            d2l.write_docs(a, self.docs())
            d2l.write_docs(b, [d._asdict() for d in self.docs()])
            with open(a, 'rb') as f, open(b, 'rb') as g:
                self.assertEqual(f.read(), g.read())
            self.assertEqual(d2l.read_docs(a), self.docs())
            with gzip.open(a, 'wt') as f:
                f.write(json.dumps({'id': 'x', 'text': 't'}) + '\n')
            (doc,) = d2l.read_docs(a)
            self.assertEqual((doc.id, doc.text, doc.lang), ('x', 't', None))

    def test_docs_hash(self):
        docs = self.docs()
        h = d2l.docs_hash(docs)
        self.assertEqual(h, d2l.docs_hash(docs[::-1]))
        self.assertEqual(h, d2l.docs_hash([d._asdict() for d in docs]))
        self.assertEqual(h, d2l.docs_hash([(d.id, d.text) for d in docs]))
        edited = docs[:2] + [docs[2]._replace(text='text 2 è')] + docs[3:]
        self.assertNotEqual(h, d2l.docs_hash(edited))
        self.assertNotEqual(h, d2l.docs_hash(docs[:4]))


class ManifestTest(unittest.TestCase):
    def manifest(self, order=1):
        m = d2l.CorpusManifest('c0')
        m.add_source('cc', 'https://data.commoncrawl.org/x.warc.gz',
                     'CC-MAIN-2025-05', 'none', 10, 1024)
        params = {'threshold': 0.65, 'order': ['empty', 'lang']}
        if order < 0:
            params = dict(reversed(list(params.items())))
        m.add_stage('lang_id', params, 10, 7, {'empty': 1, 'lang': 2},
                    'a' * 64, 'b' * 64, impl='fasttext', version='lid.176')
        return m

    def test_identity(self):
        m = self.manifest()
        with self.assertRaisesRegex(ValueError, 'n_in'):
            m.add_stage('pii', {}, 7, 7, {'pii': 1}, 'b', 'c')

    def test_canonical(self):
        a, b = self.manifest(), self.manifest(-1)
        self.assertEqual(a.canonical_json(), b.canonical_json())
        self.assertEqual(a.sha256(), b.sha256())
        s = a.canonical_json()
        self.assertNotIn(' ', s.split('"uri"')[0])
        m = d2l.CorpusManifest('x').add_stage('s', {'w': 1 / 3}, 1, 1, {},
                                              'a', 'b')
        self.assertIn('"w":0.3333', m.canonical_json())
        m.add_stage('t', {'path': '/home/user/data'}, 1, 1, {}, 'a', 'b')
        with self.assertRaisesRegex(ValueError, 'absolute path'):
            m.canonical_json()

    def test_save_load_card(self):
        m = self.manifest()
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'manifest.json')
            m.save(p)
            self.assertEqual(d2l._sha256_file(p), m.sha256())
            m2 = d2l.CorpusManifest.load(p)
            self.assertEqual(m2.sha256(), m.sha256())
            with open(p, 'w') as f:
                json.dump({'schema': 'other', 'name': 'x'}, f)
            with self.assertRaises(ValueError):
                d2l.CorpusManifest.load(p)
        card = m.to_card()
        self.assertIn('lang_id', card)
        self.assertIn('lang: 2', card)
        self.assertIn(m.sha256()[:16], card)


class DomainTest(unittest.TestCase):
    def test_registered_domain(self):
        cases = {'https://a.b.example.co.uk/x': 'example.co.uk',
                 'https://Docs.Python.org/3/': 'python.org',
                 'http://192.168.0.1/admin': '192.168.0.1',
                 'http://localhost:8000/': 'localhost',
                 'https://www.bbc.co.uk/news': 'bbc.co.uk'}
        for url, dom in cases.items():
            self.assertEqual(d2l.registered_domain(url), dom, url)


if __name__ == '__main__':
    unittest.main()
