#!/usr/bin/env python3
"""Build the reference tokenizer and its FineWeb-Edu shards (chapter 20.6).

The one-way doors of the Language Models part (chapter 20 specification,
part 1.6), read exactly by chapter 21:

    lm-r2-tokenizer-v1.json      tokenizer.json, to_str(pretty=False) bytes
    lm-fwedu-val-v1.tar          manifest.json, val-00000.{bin,idx,meta.parquet},
                                 val.jsonl.gz (doc_id, text; NFC)
    lm-fwedu-shards-100m-v1.tar  manifest.json, tokenizer.json,
                                 train-00000.{bin,idx,meta.parquet}
    lm-fwedu-manifest-4b-v1.json the manifest of the 40 training shards and the
                                 validation shard, with each file's url (under
                                 LM_DATA_URL + 'fwedu-r2-v1/'), sha1 and SHA-256

Source: FineWeb-Edu sample/10BT at the pinned revision of
d2l.LM_DATA_SOURCES['fineweb-edu'] (ODC-By), read in the chapter's file order
(d2l._FWEDU_FILES; files 005, 012 and 013 are left to chapter 21).

Stages (run in order; each writes into --work and reads what the earlier ones
wrote; all deterministic):

  scope      released metadata only (id, token_count): the validation source
             files (the shortest prefix of the order whose bucket-0 GPT-2
             tokens reach 1.5 x 2^23), the train scope (the shortest prefix
             whose bucket 9-255 GPT-2 tokens reach 1.3 x 4e9) and the
             N_TOK_DOCS pre-check (the first 80,000 documents of the shuffled
             train stream hold at most 1e8 / 1.15 GPT-2 tokens).
  stream     NFC and empty-document drops over the train scope; the
             validation candidates (bucket 0 of the validation files, file
             order); val_dedup: a candidate is dropped when its NFC text's
             SHA-256 (first 16 bytes), else its normalize_url(url), occurs
             among the train scope's bucket 9-255 documents (first-rule
             charging). Writes val-docs.jsonl.gz and stream.json.
  tokenizer  the first N_TOK_DOCS documents of the shuffled train stream;
             train_hf_bpe(texts, 32768, R2_SPECIALS) twice, in two processes
             with different RAYON_NUM_THREADS (identical bytes required);
             decode(encode(t)) == t for every training text.
  val        one permutation default_rng(1) of the deduplicated candidates,
             tokenized and cut at the first document boundary at or after
             2^23 tokens; one shard val-00000.
  train      the shuffled train stream (train scope, file order, NFC,
             buckets 9-255, shuffle buffer of 500,000 documents with
             default_rng(0)) into train-{k:05d} shards of at least 1e8
             tokens; --n-shards 1 for the 100M set, 40 for the 4B set (whose
             first shard is byte-identical to the 100M set's).
  pack       the release files above, in --work/release/, with their sha1.

    PYTHONPATH=$PWD HF_HOME=... .venv-lm/bin/python tools/build_r2_shards.py \\
        --stage scope|stream|tokenizer|val|train|pack [--n-shards 40]

The helpers (shuffle_buffer, val_split, ...) are imported by
tools/test_lm_tokenizers.py.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from d2l import torch as d2l  # noqa: E402

VOCAB = 32_768
N_TOK_DOCS = 80_000
VAL_TOKENS = 2**23
SHARD_TOKENS = 100_000_000
SHUFFLE_SEED, SHUFFLE_BUFFER, VAL_SEED = 0, 500_000, 1
VAL_GPT2_QUOTA = 1.5 * 2**23
TRAIN_GPT2_QUOTA = 1.3 * 4.0e9
TOK_GPT2_MAX = 1e8 / 1.15
RATIO_MAX = 1.3  # reference tokens per GPT-2 token on the 100M shard
TRAIN_BUCKETS = (9, 255)
REPO = 'HuggingFaceFW/fineweb-edu'
SOURCE_URI = 'hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT'
URL_PREFIX = 'fwedu-r2-v1/'
COLUMNS = ['id', 'text', 'url', 'dump', 'score', 'int_score', 'token_count']
SHUFFLE_RULE = ('documents enter a buffer until it holds `buffer`; then each '
                'incoming document replaces the element at '
                'rng.integers(buffer), which is emitted; at the end the '
                'buffer is emitted in rng.permutation order')


# ---------------------------------------------------------------- helpers
def shuffle_buffer(items, seed=SHUFFLE_SEED, size=SHUFFLE_BUFFER):
    """The seeded shuffle buffer of the train stream (rule above)."""
    rng, buf = np.random.default_rng(seed), []
    for x in items:
        if len(buf) < size:
            buf.append(x)
            continue
        j = int(rng.integers(size))
        yield buf[j]
        buf[j] = x
    for j in rng.permutation(len(buf)):
        yield buf[int(j)]


def text_key(text):
    """The validation dedup key of an NFC text: its SHA-256's first 16
    bytes."""
    return hashlib.sha256(text.encode('utf-8')).digest()[:16]


def val_split(candidates, train_text_keys, train_urls, count_tokens,
              quota=VAL_TOKENS, seed=VAL_SEED):
    """The validation split from its candidates (doc_id, text, meta) in file
    order: val_dedup drops a candidate whose text key is among the train
    keys ('text_hash'), else whose normalized URL is among the train URLs
    ('url'); the rest is permuted with default_rng(seed) and cut at the
    first document boundary at or after quota tokens (count_tokens(texts)
    gives each text's tokens without its EOS). Returns (kept, drops,
    n_dedup, n_tokens): drops by reason, n_dedup documents after
    val_dedup."""
    drops, kept = {'text_hash': 0, 'url': 0}, []
    for d in candidates:
        if text_key(d[1]) in train_text_keys:
            drops['text_hash'] += 1
        elif d2l.normalize_url(d[2]['url']) in train_urls:
            drops['url'] += 1
        else:
            kept.append(d)
    n_dedup = len(kept)
    kept = [kept[int(i)] for i in np.random.default_rng(seed).permutation(
        n_dedup)]
    lens = np.array([n + 1 for n in count_tokens([d[1] for d in kept])])
    total = np.cumsum(lens)
    if not len(total) or total[-1] < quota:
        raise ValueError(f'the validation candidates hold '
                         f'{int(total[-1]) if len(total) else 0} tokens, '
                         f'fewer than {quota}: add the next file (risk R10)')
    cut = int(np.searchsorted(total, quota)) + 1
    return kept[:cut], drops, n_dedup, int(total[cut - 1])


def docs_hash_of_pairs(pairs):
    """d2l.docs_hash from (doc_id, SHA-256 hex of the text) pairs."""
    h = hashlib.sha256()
    for i, t in sorted((str(i), t) for i, t in pairs):
        h.update(f'{i}\t{t}\n'.encode('utf-8'))
    return h.hexdigest()


def write_tar(path, root, names):
    """A deterministic USTAR archive of root/name for each name."""
    tmp = path + '.part'
    with tarfile.open(tmp, 'w', format=tarfile.USTAR_FORMAT) as tf:
        for n in names:
            with open(os.path.join(root, n), 'rb') as f:
                data = f.read()
            ti = tarfile.TarInfo(n)
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            ti.uname = ti.gname = ''
            tf.addfile(ti, io.BytesIO(data))
    os.replace(tmp, path)


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- sources
def revision():
    return d2l.LM_DATA_SOURCES['fineweb-edu'][2]


def local_file(name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, name, revision=revision(),
                           repo_type='dataset')


def file_rows(name, columns, batch=8192):
    """(row, record) of one source file, in file order."""
    import pyarrow.parquet as pq
    pf, row = pq.ParquetFile(local_file(name)), 0
    for b in pf.iter_batches(columns=columns, batch_size=batch):
        cols = [b.column(c).to_pylist() for c in columns]
        for vals in zip(*cols):
            yield row, dict(zip(columns, vals))
            row += 1


def meta_of(r, bucket, name, row):
    return dict(url=r['url'], dump=r['dump'], bucket=bucket,
                edu_score=r['score'], int_score=r['int_score'],
                gpt2_tokens=r['token_count'], src_file=name, src_row=row)


def train_docs(files, stats=None, rows_of=None):
    """(doc_id, text, meta) of the train stream before the shuffle: the
    files in order, rows in file order, NFC, empty documents dropped,
    buckets 9-255 kept. rows_of(name) yields a file's (row, record)."""
    lo, hi = TRAIN_BUCKETS
    rows_of = rows_of or (lambda name: file_rows(name, COLUMNS))
    for name in files:
        for row, r in rows_of(name):
            text = unicodedata.normalize('NFC', r['text'] or '')
            if stats is not None:
                stats['rows'] = stats.get('rows', 0) + 1
            if not text:
                if stats is not None:
                    stats['empty'] = stats.get('empty', 0) + 1
                continue
            b = d2l.fwedu_bucket(r['id'])
            if lo <= b <= hi:
                yield r['id'], text, meta_of(r, b, name, row)


def train_stream(files, stats=None, rows_of=None, size=SHUFFLE_BUFFER):
    return shuffle_buffer(train_docs(files, stats, rows_of), size=size)


# ---------------------------------------------------------------- stages
def stage_scope(work):
    """Validation source files, train scope and the N_TOK_DOCS pre-check,
    from the released id and token_count columns."""
    import pyarrow.parquet as pq
    per_file, val_files, scope, acc0, acc9 = {}, None, None, 0, 0
    train_counts = []  # GPT-2 tokens of the train-bucket rows, file order
    for name in d2l._FWEDU_FILES:
        t = pq.read_table(local_file(name), columns=['id', 'token_count'])
        ids, tc = t.column('id').to_pylist(), t.column(
            'token_count').to_numpy()
        b = np.array([d2l.fwedu_bucket(i) for i in ids])
        tr = (b >= TRAIN_BUCKETS[0]) & (b <= TRAIN_BUCKETS[1])
        per_file[name] = dict(rows=len(ids), bucket0_docs=int((b == 0).sum()),
                              bucket0_gpt2=int(tc[b == 0].sum()),
                              train_docs=int(tr.sum()),
                              train_gpt2=int(tc[tr].sum()))
        acc0 += per_file[name]['bucket0_gpt2']
        acc9 += per_file[name]['train_gpt2']
        if scope is None:
            train_counts.append(tc[tr])
        log(name, per_file[name])
        if val_files is None and acc0 >= VAL_GPT2_QUOTA:
            val_files = list(d2l._FWEDU_FILES[:len(per_file)])
        if scope is None and acc9 >= TRAIN_GPT2_QUOTA:
            scope = list(d2l._FWEDU_FILES[:len(per_file)])
        if val_files and scope:
            break
    if not (val_files and scope):
        raise ValueError('the chapter files cannot meet the quotas')
    head = list(itertools_islice(shuffle_buffer(
        np.concatenate(train_counts).tolist()), N_TOK_DOCS))
    out = dict(revision=revision(), files=per_file, val_files=val_files,
               train_scope=scope, val_gpt2_quota=VAL_GPT2_QUOTA,
               train_gpt2_quota=TRAIN_GPT2_QUOTA,
               tok_docs_gpt2=int(sum(head)), tok_gpt2_max=TOK_GPT2_MAX)
    log('val files', val_files, 'train scope', scope)
    log(f'first {N_TOK_DOCS:,} shuffled train documents: {sum(head):,} '
        f'GPT-2 tokens (at most {TOK_GPT2_MAX:,.0f})')
    if sum(head) > TOK_GPT2_MAX:
        raise ValueError('N_TOK_DOCS pre-check failed: a new N_TOK_DOCS is '
                         'an author step')
    with open(os.path.join(work, 'scope.json'), 'w') as f:
        json.dump(out, f, indent=1, sort_keys=True)


def itertools_islice(it, n):
    import itertools
    return itertools.islice(it, n)


def _dedup_file(args):
    """Per train-scope file: counts, and which validation keys occur among
    its bucket 9-255 documents."""
    name, val_keys, val_urls = args
    lo, hi = TRAIN_BUCKETS
    hit_k, hit_u, n = set(), set(), dict(rows=0, empty=0, bucket0=0,
                                         side=0, train=0)
    for row, r in file_rows(name, ['id', 'text', 'url']):
        n['rows'] += 1
        text = unicodedata.normalize('NFC', r['text'] or '')
        if not text:
            n['empty'] += 1
            continue
        b = d2l.fwedu_bucket(r['id'])
        if b == 0:
            n['bucket0'] += 1
        elif b < lo:
            n['side'] += 1
        elif b <= hi:
            n['train'] += 1
            k = text_key(text)
            if k in val_keys:
                hit_k.add(k)
            u = d2l.normalize_url(r['url'])
            if u in val_urls:
                hit_u.add(u)
    return name, n, hit_k, hit_u


def stage_stream(work, procs):
    from multiprocessing import Pool
    with open(os.path.join(work, 'scope.json')) as f:
        scope = json.load(f)
    cand, empty0 = [], 0
    for name in scope['val_files']:
        for row, r in file_rows(name, COLUMNS):
            if d2l.fwedu_bucket(r['id']) != 0:
                continue
            text = unicodedata.normalize('NFC', r['text'] or '')
            if not text:
                empty0 += 1
                continue
            cand.append((r['id'], text, meta_of(r, 0, name, row)))
    log(f'{len(cand):,} validation candidates ({empty0} empty dropped)')
    keys = {text_key(t) for _, t, _ in cand}
    urls = {d2l.normalize_url(m['url']) for _, _, m in cand}
    hit_k, hit_u, counts = set(), set(), {}
    with Pool(procs) as pool:
        for name, n, hk, hu in pool.imap(
                _dedup_file, [(f, keys, urls) for f in scope['train_scope']]):
            counts[name] = n
            hit_k |= hk
            hit_u |= hu
            log(name, n, len(hk), len(hu))
    with gzip.GzipFile(os.path.join(work, 'val-candidates.jsonl.gz'), 'wb',
                       mtime=0) as g:
        for i, t, m in cand:
            g.write((json.dumps(dict(doc_id=i, text=t, meta=m),
                                ensure_ascii=False, sort_keys=True)
                     + '\n').encode('utf-8'))
    out = dict(counts=counts, n_candidates=len(cand), val_empty=empty0,
               hits_text=sorted(k.hex() for k in hit_k),
               hits_url=sorted(hit_u))
    with open(os.path.join(work, 'stream.json'), 'w') as f:
        json.dump(out, f, indent=1, sort_keys=True)
    log('val_dedup hits: text', len(hit_k), 'url', len(hit_u))


def read_candidates(work):
    with gzip.open(os.path.join(work, 'val-candidates.jsonl.gz'), 'rt',
                   encoding='utf-8') as f:
        return [(r['doc_id'], r['text'], r['meta']) for r in map(json.loads,
                                                                  f)]


def train_once(work, out_path):
    """Child process: train the tokenizer on tok-train.jsonl.gz."""
    with gzip.open(os.path.join(work, 'tok-train.jsonl.gz'), 'rt',
                   encoding='utf-8') as f:
        texts = [json.loads(line)['text'] for line in f]
    tok = d2l.train_hf_bpe(texts, VOCAB, d2l.R2_SPECIALS)
    with open(out_path, 'wb') as f:
        f.write(tok.to_str(pretty=False).encode('utf-8'))


def stage_tokenizer(work, threads=(8, 32)):
    with open(os.path.join(work, 'scope.json')) as f:
        scope = json.load(f)
    docs = list(itertools_islice(train_stream(scope['train_scope']),
                                 N_TOK_DOCS))
    gpt2 = sum(m['gpt2_tokens'] for _, _, m in docs)
    log(f'{len(docs):,} training documents, {gpt2:,} GPT-2 tokens, '
        f'{sum(len(t) for _, t, _ in docs):,} characters')
    assert len(docs) == N_TOK_DOCS and gpt2 == scope['tok_docs_gpt2'], (
        'the stream differs from the scope stage (empty documents?)')
    with gzip.GzipFile(os.path.join(work, 'tok-train.jsonl.gz'), 'wb',
                       mtime=0) as g:
        for i, t, _ in docs:
            g.write((json.dumps(dict(doc_id=i, text=t), ensure_ascii=False)
                     + '\n').encode('utf-8'))
    shas = []
    for n in threads:
        path = os.path.join(work, f'tokenizer-rayon{n}.json')
        env = dict(os.environ, RAYON_NUM_THREADS=str(n))
        tic = time.time()
        subprocess.run([sys.executable, __file__, '--work', work, '--stage',
                        'tokenizer-child', '--child-out', path], env=env,
                       check=True)
        with open(path, 'rb') as f:
            shas.append(hashlib.sha256(f.read()).hexdigest())
        log(f'RAYON_NUM_THREADS={n}: {time.time() - tic:.0f} s, SHA-256 '
            f'{shas[-1]}')
    if len(set(shas)) != 1:
        raise ValueError('the two trainings differ (risk R1)')
    final = os.path.join(work, 'tokenizer.json')
    os.replace(path, final)
    tok = d2l._r2_from_file(final, raw=True, sha256=shas[0])
    assert tok.get_vocab_size() == VOCAB
    assert [tok.id_to_token(i) for i in range(32)] == d2l.R2_SPECIALS
    texts = [t for _, t, _ in docs]
    enc = tok.encode_batch(texts, add_special_tokens=False)
    assert tok.decode_batch([e.ids for e in enc],
                            skip_special_tokens=False) == texts
    assert all(min(e.ids, default=32) >= 32 for e in enc)
    log(f'round trip of all {len(texts):,} training texts; SHA-256 '
        f'{shas[0]}')
    with open(os.path.join(work, 'tokenizer.sha256'), 'w') as f:
        f.write(shas[0])


def tokenizer_info(work):
    with open(os.path.join(work, 'tokenizer.sha256')) as f:
        sha = f.read().strip()
    return dict(file='tokenizer.json', sha256=sha, vocab_size=VOCAB,
                specials=list(d2l.R2_SPECIALS), split_special_tokens=True)


def r2(work):
    return d2l._r2_from_file(os.path.join(work, 'tokenizer.json'), raw=True,
                             sha256=tokenizer_info(work)['sha256'])


def base_manifest(work, name):
    with open(os.path.join(work, 'scope.json')) as f:
        scope = json.load(f)
    m = d2l.CorpusManifest(name, tokenizer=tokenizer_info(work))
    read = scope['train_scope']
    m.add_source('fineweb-edu', SOURCE_URI, revision=scope['revision'],
                 license='odc-by',
                 n_docs=sum(scope['files'][f]['rows'] for f in read),
                 files=[dict(file=f, rows=scope['files'][f]['rows'])
                        for f in read],
                 val_files=scope['val_files'], train_scope=read,
                 reserved=['sample/10BT/005_00000.parquet',
                           'sample/10BT/012_00000.parquet',
                           'sample/10BT/013_00000.parquet'])
    return m, scope


def stage_val(work):
    with open(os.path.join(work, 'stream.json')) as f:
        st = json.load(f)
    hits_text = {bytes.fromhex(k) for k in st['hits_text']}
    hits_url = set(st['hits_url'])
    cand = read_candidates(work)
    tok = r2(work)
    kept, drops, n_dedup, n_tok = val_split(
        cand, hits_text, hits_url,
        lambda texts: map(len, d2l._doc_token_ids(tok, texts)))
    log(f'val_dedup {drops}; {n_dedup:,} documents; cut at {len(kept):,} '
        f'documents, {n_tok:,} tokens')
    m, scope = base_manifest(work, 'lm-fwedu-val-v1')
    c = [st['counts'][f] for f in scope['val_files']]
    rows, empty = sum(x['rows'] for x in c), sum(x['empty'] for x in c)
    assert len(cand) == sum(x['bucket0'] for x in c)
    pairs_in = [(i, hashlib.sha256(t.encode()).hexdigest())
                for i, t, _ in cand]
    m.add_stage('nfc', dict(form='NFC', files='val_files'), rows,
                rows - empty, {'empty': empty}, None, None)
    m.add_stage('bucket_split', dict(val=[0]), rows - empty, len(cand),
                {'other_buckets': rows - empty - len(cand)}, None,
                docs_hash_of_pairs(pairs_in))
    keep_ids = {i for i, _, _ in kept}
    dedup_ids = set()
    for i, t, mm in cand:
        if not (text_key(t) in hits_text
                or d2l.normalize_url(mm['url']) in hits_url):
            dedup_ids.add(i)
    sha = dict(pairs_in)
    m.add_stage('val_dedup', dict(
        rules=['text_hash', 'url'], key_text='sha256(nfc text)[:16]',
        key_url='normalize_url', against='train scope, buckets 9-255',
        charging='first rule'), len(cand), n_dedup, drops,
        docs_hash_of_pairs(pairs_in),
        docs_hash_of_pairs([(i, sha[i]) for i in dedup_ids]))
    m.add_stage('permute', dict(seed=VAL_SEED, rng='numpy default_rng',
                                cut_tokens=VAL_TOKENS,
                                cut='first document boundary at or after'),
                n_dedup, len(kept), {'beyond_cut': n_dedup - len(kept)},
                docs_hash_of_pairs([(i, sha[i]) for i in dedup_ids]),
                docs_hash_of_pairs([(i, sha[i]) for i in keep_ids]))
    out = os.path.join(work, 'val')
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    entry = d2l.write_shards(kept, tok, out, 'val', shard_tokens=VAL_TOKENS,
                             manifest=m)
    assert len(entry['shards']) == 1 and entry['n_tokens'] == n_tok
    m.add_stage('tokenize', dict(tokenizer='lm-r2-tokenizer-v1',
                                 eos_after_each_document=True),
                len(kept), entry['n_docs'], {},
                docs_hash_of_pairs([(i, sha[i]) for i in keep_ids]),
                docs_hash_of_pairs([(i, sha[i]) for i in keep_ids]),
                tokens_out=entry['n_tokens'])
    m.save(os.path.join(out, 'manifest.json'))
    with gzip.GzipFile(os.path.join(out, 'val.jsonl.gz'), 'wb',
                       mtime=0) as g:
        for i, t, _ in kept:
            g.write((json.dumps(dict(doc_id=i, text=t), ensure_ascii=False)
                     + '\n').encode('utf-8'))
    assert d2l.TokenShards(out, 'val').verify()
    log('val written', entry['n_docs'], entry['n_tokens'])


def stage_train(work, n_shards, batch_docs=8192):
    m, scope = base_manifest(work, 'lm-fwedu-shards-100m-v1' if n_shards == 1
                             else 'lm-fwedu-manifest-4b-v1')
    with open(os.path.join(work, 'stream.json')) as f:
        st = json.load(f)
    counts = [st['counts'][x] for x in scope['train_scope']]
    tok = r2(work)
    out = os.path.join(work, f'train-{n_shards}')
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    stats, pairs, gpt2 = {}, [], [0]
    tic = [time.time(), 0]

    def meta_fn(doc_id, text, meta):
        pairs.append((doc_id, hashlib.sha256(text.encode()).hexdigest()))
        gpt2[0] += meta['gpt2_tokens']
        if len(pairs) % 200_000 == 0:
            log(f'{len(pairs):,} documents, {time.time() - tic[0]:.0f} s')
        return meta

    entry = d2l.write_shards(train_stream(scope['train_scope'], stats), tok,
                             out, 'train', shard_tokens=SHARD_TOKENS,
                             meta_fn=meta_fn, manifest=m,
                             batch_docs=batch_docs, max_shards=n_shards)
    if len(entry['shards']) != n_shards:
        raise ValueError(f'the train scope ran out after '
                         f'{len(entry["shards"])} shards')
    s0 = entry['shards'][0]
    assert s0['n_docs'] >= N_TOK_DOCS, 'train-00000 must hold the ' \
        'tokenizer documents'
    n_written = entry['n_docs']
    ratio = (entry['n_tokens'] - n_written) / gpt2[0]
    log(f'{n_shards} shards, {entry["n_tokens"]:,} tokens, {n_written:,} '
        f'documents; reference/GPT-2 tokens {ratio:.4f}; read {stats}')
    assert ratio <= RATIO_MAX
    total_train = sum(x['train'] for x in counts)
    rows, empty = sum(x['rows'] for x in counts), sum(x['empty']
                                                      for x in counts)
    m.stages = []
    m.add_stage('nfc', dict(form='NFC', files='train_scope'), rows,
                rows - empty, {'empty': empty}, None, None)
    m.add_stage('bucket_split', dict(train=list(TRAIN_BUCKETS)),
                rows - empty, total_train,
                {'val_bucket_0': sum(x['bucket0'] for x in counts),
                 'side_buckets_1_8': sum(x['side'] for x in counts)},
                None, None)
    m.add_stage('shuffle', dict(seed=SHUFFLE_SEED, buffer=SHUFFLE_BUFFER,
                                rng='numpy default_rng', rule=SHUFFLE_RULE),
                total_train, total_train, {}, None, None)
    out_hash = docs_hash_of_pairs(pairs)
    m.add_stage('tokenize', dict(tokenizer='lm-r2-tokenizer-v1',
                                 shard_tokens=SHARD_TOKENS,
                                 n_shards=n_shards,
                                 eos_after_each_document=True,
                                 gpt2_tokens=gpt2[0],
                                 tokens_per_gpt2_token=round(ratio, 4)),
                total_train, n_written,
                {'beyond_last_shard': total_train - n_written}, None,
                out_hash, tokens_out=entry['n_tokens'])
    m.save(os.path.join(out, 'manifest.json'))
    with open(os.path.join(out, 'pairs.json'), 'w') as f:
        json.dump(dict(gpt2=gpt2[0], ratio=ratio,
                       n0=s0['n_docs']), f)
    log('train written', out, f'{time.time() - tic[0]:.0f} s')


def stage_pack(work, n_big):
    rel = os.path.join(work, 'release')
    os.makedirs(rel, exist_ok=True)
    shutil = __import__('shutil')
    shutil.copyfile(os.path.join(work, 'tokenizer.json'),
                    os.path.join(rel, 'lm-r2-tokenizer-v1.json'))
    val = os.path.join(work, 'val')
    write_tar(os.path.join(rel, 'lm-fwedu-val-v1.tar'), val,
              ['manifest.json', 'val.jsonl.gz', 'val-00000.bin',
               'val-00000.idx', 'val-00000.meta.parquet'])
    one = os.path.join(work, 'train-1')
    shutil.copyfile(os.path.join(work, 'tokenizer.json'),
                    os.path.join(one, 'tokenizer.json'))
    m1 = d2l.CorpusManifest.load(os.path.join(one, 'manifest.json'))
    assert d2l.TokenShards(one).verify()
    write_tar(os.path.join(rel, 'lm-fwedu-shards-100m-v1.tar'), one,
              ['manifest.json', 'tokenizer.json', 'train-00000.bin',
               'train-00000.idx', 'train-00000.meta.parquet'])
    files = {}
    if n_big:
        big = os.path.join(work, f'train-{n_big}')
        mb = d2l.CorpusManifest.load(os.path.join(big, 'manifest.json'))
        assert mb.splits['train']['shards'][0] == m1.splits['train'][
            'shards'][0], 'shard 00000 differs from the 100M set'
        mv = d2l.CorpusManifest.load(os.path.join(val, 'manifest.json'))
        mb.splits = dict(train=mb.splits['train'], val=mv.splits['val'])
        mb.stages = ([dict(s, split='train') for s in mb.stages]
                     + [dict(s, split='val') for s in mv.stages])
        for split, root in (('train', big), ('val', val)):
            for s in mb.splits[split]['shards']:
                s['url'], s['sha1'] = {}, {}
                for ext, e in d2l._SHARD_EXT.items():
                    p = os.path.join(root, s['file'] + e)
                    s['sha1'][ext] = sha1_file(p)
                    s['url'][ext] = d2l.LM_DATA_URL + URL_PREFIX + s[
                        'file'] + e
                    files[s['file'] + e] = p
        mb.save(os.path.join(rel, 'lm-fwedu-manifest-4b-v1.json'))
        with open(os.path.join(rel, 'shard-files.txt'), 'w') as f:
            f.writelines(p + '\n' for _, p in sorted(files.items()))
    for n in sorted(os.listdir(rel)):
        p = os.path.join(rel, n)
        if os.path.isfile(p) and not n.endswith('.txt'):
            log(n, os.path.getsize(p), sha1_file(p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--work', default=os.path.join(ROOT, 'data', 'lm',
                                                   'build-r2-v1'))
    ap.add_argument('--stage', required=True,
                    choices=['scope', 'stream', 'tokenizer', 'val', 'train',
                             'pack', 'tokenizer-child'])
    ap.add_argument('--n-shards', type=int, default=1)
    ap.add_argument('--big', type=int, default=0,
                    help='pack: the shard count of the large set (0: none)')
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--child-out')
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    if a.stage == 'scope':
        stage_scope(a.work)
    elif a.stage == 'stream':
        stage_stream(a.work, a.procs)
    elif a.stage == 'tokenizer':
        stage_tokenizer(a.work)
    elif a.stage == 'tokenizer-child':
        train_once(a.work, a.child_out)
    elif a.stage == 'val':
        stage_val(a.work)
    elif a.stage == 'train':
        stage_train(a.work, a.n_shards)
    elif a.stage == 'pack':
        stage_pack(a.work, a.big)


if __name__ == '__main__':
    main()
