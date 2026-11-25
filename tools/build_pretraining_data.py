#!/usr/bin/env python3
"""Build the token shards of sec_lm_continued_pretraining (21.4) and
sec_lm_long_context (21.5).

Sources (pinned; peS2o and FineWeb-Edu are ODC-By, attribution in every
manifest):
  * domain: peS2o v2 (allenai/peS2o @636a503e), train file 00000 of 20, read
    in file order. The whole file is s2ag/train (titles and abstracts;
    3,058,025 records, counted 2026-11-25): "scientific prose".
  * domain validation: peS2o v2 validation file 00000 of 2 (s2ag/valid, the
    same kind of text, a different split).
  * replay: FineWeb-Edu sample/10BT/005_00000.parquet (@87f09149), a file
    the reference model's shards never read (d2l._FWEDU_FILES).
  * general validation: the reference model's FineWeb-Edu validation texts
    (data/lm/fwedu-r2-v1/val/val.jsonl.gz, i.e. d2l.reference_shards('val')).
  * the book's train and held-out sections are built by the library
    (d2l._book_shards) on first use; this script builds them too so that the
    T2 manifest can list them.

Stages:
  --stage cpt-t1   the R2-tokenized (uint16) T1 slices, hosted:
                   data/lm/cpt-t1-v1/{pes2o,val_domain,replay}/ and the
                   upload tars data/lm/rerun/lm-pes2o-12m-v1.tar,
                   lm-pes2o-val-v1.tar, lm-fwedu-replay-12m-v1.tar (each a
                   shard set with its manifest.json at the root; sha1 and
                   size printed for DATA_HUB).
  --stage cpt-qwen the Qwen3-tokenized (uint32) T2 shards, local only:
                   data/lm/cpt-qwen-v1/{pes2o,replay,val_domain,val_general}/
                   (about 0.5 GB) and the committed summary
                   data/lm/rerun/lm-cpt-qwen-manifest-v1.json.gz (json-table:
                   the stream table, each stream's layout hash, the sources
                   and the attribution).
  --check          verify every shard set of the stage (SHA-256 and EOS
                   invariants), the token targets, and the disjointness of
                   train and evaluation documents (peS2o ids; FineWeb-Edu
                   text hashes against the general validation texts).

  --stage lc-long  21.5's long half (R2-tokenized, uint16, hosted): PG-19
                   train books (emozilla/pg19 @d2f3c93e, the parquet
                   conversion of deepmind/pg19; public-domain books) in file
                   order up to 50M tokens, then peS2o v2 full-text papers
                   (validation file 1 of 2, which is all s2orc/valid) of at
                   least 8,192 tokens in file order up to 10M tokens, in
                   shards of at least 8M tokens (the first shard is the T1
                   prefix): data/lm/lc-long-v1/ and the upload tar
                   data/lm/rerun/lm-lc-long-60m-v1.tar.
  --stage pg19-test 21.5's test books: the first 64 PG-19 test books (file
                   order) of at least 16,385 R2 tokens, each cut to its first
                   16,385 tokens (one evaluation window of 16,384 targets):
                   data/lm/pg19-test-v1/ and data/lm/rerun/lm-pg19-test-v1.tar.

Deterministic: the same inputs give the same bytes (no timestamps, files read
in order, one shard per stream). Run from the repository root:

    HF_HOME=/home/smola/.cache/hf-lm .venv-lm/bin/python \\
        tools/build_pretraining_data.py --stage cpt-t1 [--check]

Expected time: cpt-t1 about 2 minutes, cpt-qwen about 8 minutes (CPU,
tokenization of about 110M tokens), lc-long about 5 minutes, pg19-test
under a minute; a few GB of RAM.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER = os.path.join(ROOT, 'chapter_lm-pretraining')
PES2O_TRAIN = 'data/v2/train-00000-of-00020.json.gz'
PES2O_VAL = 'data/v2/validation-00000-of-00002.json.gz'
FWEDU_REPLAY = 'sample/10BT/005_00000.parquet'

# stage: {stream: (token target, None = all)}
TARGETS = {
    'cpt-t1': {'pes2o': 12_000_000, 'val_domain': 2_200_000,
               'replay': 12_000_000},
    'cpt-qwen': {'pes2o': 80_000_000, 'replay': 24_000_000,
                 'val_domain': 2_200_000, 'val_general': None}}
UPLOADS = {'pes2o': 'lm-pes2o-12m-v1', 'val_domain': 'lm-pes2o-val-v1',
           'replay': 'lm-fwedu-replay-12m-v1'}
LC = {  # 21.5: the long half's two parts, their token targets, the windows
    'pg19': 50_000_000, 'pes2o': 10_000_000, 'min_paper': 8_192,
    'shard': 8_000_000, 'test_books': 64, 'test_tokens': 16_385}
PES2O_LONG = 'data/v2/validation-00001-of-00002.json.gz'  # all s2orc/valid
LC_ATTRIBUTION = (
    'PG-19 (deepmind/pg19, Apache-2.0; public-domain books from Project '
    'Gutenberg, read from the parquet conversion emozilla/pg19) and peS2o '
    '(allenai/peS2o, ODC-By 1.0; full-text papers derived from S2ORC, Allen '
    'Institute for AI). Token shards derived for Dive into Deep Learning, '
    'section "Long-Context Training".')
ATTRIBUTION = (
    'peS2o (allenai/peS2o, ODC-By 1.0; derived from S2ORC/S2AG, Allen '
    'Institute for AI) and FineWeb-Edu (HuggingFaceFW/fineweb-edu, ODC-By '
    '1.0; Hugging Face). Token shards derived for Dive into Deep Learning, '
    'section "Continued Pretraining and Mid-Training".')


def d2l_module():
    sys.path.insert(0, ROOT)
    from d2l import torch as d2l
    return d2l


def pes2o_docs(d2l, path):
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES['pes2o']
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(repo, path, repo_type='dataset', revision=rev)
    with gzip.open(local, 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            yield f'pes2o:{d["id"]}', d['text']


def general_hashes(d2l):
    """sha1 of every general-validation text."""
    return {hashlib.sha1(t.encode()).hexdigest()
            for _, t in val_general_docs(d2l)}


def fwedu_docs(d2l, exclude=frozenset()):
    """FineWeb-Edu file 005 in order, without the documents whose text is
    also a general-validation text (FineWeb-Edu has exact duplicates
    across its files: 2 in the first 24M Qwen tokens)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    _, repo, rev, _ = d2l.LM_DATA_SOURCES['fineweb-edu']
    local = hf_hub_download(repo, FWEDU_REPLAY, repo_type='dataset',
                            revision=rev)
    pf = pq.ParquetFile(local)
    for batch in pf.iter_batches(batch_size=2048, columns=['id', 'text']):
        for i, t in zip(batch.column('id').to_pylist(),
                        batch.column('text').to_pylist()):
            h = hashlib.sha1(t.encode()).hexdigest()
            if h not in exclude:
                yield f'fwedu:{i}', t, {'sha1': h}


def val_general_docs(d2l):
    root = d2l.reference_shards('val').root
    with gzip.open(os.path.join(root, 'val.jsonl.gz'), 'rt') as f:
        for k, line in enumerate(f):
            d = json.loads(line)
            yield str(d.get('doc_id', d.get('id', f'val:{k}'))), d['text']


def sources(d2l, stage):
    out = {'pes2o': lambda: pes2o_docs(d2l, PES2O_TRAIN),
           'val_domain': lambda: pes2o_docs(d2l, PES2O_VAL),
           'replay': lambda: fwedu_docs(d2l, general_hashes(d2l))}
    if stage == 'cpt-qwen':
        out['val_general'] = lambda: val_general_docs(d2l)
    return out


def tokenizer(d2l, stage):
    os.chdir(CHAPTER)  # the library's relative data paths
    return d2l._cpt_tokenizer('t1' if stage == 'cpt-t1' else 't2')


def stage_dir(d2l, stage):
    return os.path.abspath(d2l._CPT_DIRS['t1' if stage == 'cpt-t1' else 't2'])


def build_stream(d2l, tok, docs, target, out):
    """One shard: documents in order until the shard holds target tokens."""
    import shutil
    tmp = out + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    kw = (dict(shard_tokens=target, max_shards=1) if target else
          dict(shard_tokens=10 ** 12))
    d2l.write_shards(docs, tok, tmp, 'train', **kw)
    with open(os.path.join(tmp, 'manifest.json')) as f:
        m = json.load(f)
    m['attribution'] = ATTRIBUTION
    with open(os.path.join(tmp, 'manifest.json'), 'w') as f:
        json.dump(m, f, indent=1, sort_keys=True)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return d2l.TokenShards(out, 'train')


def deterministic_tar(folder, path):
    """A tar of folder's files at the archive root, sorted, with fixed
    owners and times (the same shards give the same bytes)."""
    with tarfile.open(path + '.tmp', 'w', format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(os.listdir(folder)):
            info = tar.gettarinfo(os.path.join(folder, name), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            with open(os.path.join(folder, name), 'rb') as f:
                tar.addfile(info, f)
    os.replace(path + '.tmp', path)
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest(), os.path.getsize(path)


def pg19_books(d2l, split):
    """(index, title, text) of the PG-19 books of one split, file order."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES['pg19']
    files = sorted(f for f in HfApi().list_repo_files(
        repo, repo_type='dataset', revision=rev)
        if f.startswith(f'data/{split}-'))
    k = 0
    for name in files:
        local = hf_hub_download(repo, name, repo_type='dataset', revision=rev)
        for batch in pq.ParquetFile(local).iter_batches(
                batch_size=64, columns=['short_book_title', 'text']):
            for title, text in zip(batch.column(0).to_pylist(),
                                   batch.column(1).to_pylist()):
                yield k, title, text
                k += 1


def lc_long_docs(d2l, tok):
    """The long half's documents in order, each with its part: PG-19 train
    books up to LC['pg19'] tokens, then peS2o papers of at least
    LC['min_paper'] tokens up to LC['pes2o'] tokens."""
    n = 0
    for k, title, text in pg19_books(d2l, 'train'):
        if n >= LC['pg19']:
            break
        n += len(tok(text, add_special_tokens=False)['input_ids'])
        yield f'pg19-train:{k}', text, {'part': 'pg19', 'title': title}
    n = 0
    for doc_id, text in pes2o_docs(d2l, PES2O_LONG):
        if n >= LC['pes2o']:
            break
        m = len(tok(text, add_special_tokens=False)['input_ids'])
        if m >= LC['min_paper']:
            n += m
            yield doc_id, text, {'part': 'pes2o', 'title': ''}


class PreTokenized:
    """A tiktoken-like encoding whose 'texts' are keys of precomputed ids
    (write_shards then stores exactly those ids)."""
    special_tokens_set = {'<|endoftext|>'}

    def __init__(self, ids, n_vocab):
        self.ids, self.n_vocab = ids, n_vocab

    def encode_single_token(self, s):
        return 0

    def encode_ordinary(self, key):
        return self.ids[key]

    def encode_ordinary_batch(self, keys):
        return [self.ids[k] for k in keys]


def pg19_test_docs(d2l, tok):
    """The first LC['test_books'] test books of at least LC['test_tokens']
    tokens, cut to that many: (doc_id, key, meta) and the ids by key."""
    docs, ids = [], {}
    for k, title, text in pg19_books(d2l, 'test'):
        x = tok(text, add_special_tokens=False)['input_ids']
        if len(x) >= LC['test_tokens']:
            key = f'pg19-test:{k}'
            ids[key] = x[:LC['test_tokens']]
            docs.append((key, key, {'part': 'test', 'title': title}))
        if len(docs) == LC['test_books']:
            break
    return docs, ids


def lc_dirs(stage):
    root = os.path.join(ROOT, 'data', 'lm')
    return {'lc-long': (os.path.join(root, 'lc-long-v1'),
                        'lm-lc-long-60m-v1'),
            'pg19-test': (os.path.join(root, 'pg19-test-v1'),
                          'lm-pg19-test-v1')}[stage]


def build_lc(stage):
    """21.5's two shard sets (R2 tokenizer, uint16) and their upload tars."""
    import shutil
    d2l = d2l_module()
    os.chdir(CHAPTER)
    tok = d2l.r2_tokenizer()
    out, name = lc_dirs(stage)
    tmp = out + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    meta = lambda doc_id, text, m: m
    if stage == 'lc-long':
        d2l.write_shards(lc_long_docs(d2l, tok), tok, tmp, 'train',
                         shard_tokens=LC['shard'], meta_fn=meta)
    else:
        docs, ids = pg19_test_docs(d2l, tok)
        d2l.write_shards(docs, PreTokenized(ids, len(tok)), tmp, 'train',
                         shard_tokens=10 ** 12, eos_id=0, meta_fn=meta)
    with open(os.path.join(tmp, 'manifest.json')) as f:
        m = json.load(f)
    m['attribution'] = LC_ATTRIBUTION
    m['sources'] = {'pg19': list(d2l.PRETRAIN_SOURCES['pg19'][1:3]),
                    'pes2o': list(d2l.PRETRAIN_SOURCES['pes2o'][1:3]) + [
                        PES2O_LONG]}
    m['tokenizer'] = dict(m.get('tokenizer') or {}, name='lm-r2-tokenizer-v1')
    with open(os.path.join(tmp, 'manifest.json'), 'w') as f:
        json.dump(m, f, indent=1, sort_keys=True)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    shards = d2l.TokenShards(out, 'train')
    rerun = os.path.abspath(d2l._rerun_dir())
    os.makedirs(rerun, exist_ok=True)
    sha1, size = deterministic_tar(out, os.path.join(rerun, name + '.tar'))
    print(f'{stage}: {shards.n_docs:,} documents, {shards.n_tokens:,} '
          f'tokens in {shards.n_shards} shards')
    print(f"DATA_HUB['{name}'] = ('{name}.tar', '{sha1}')  "
          f'# {size / 2**20:.1f} MB')


def check_lc(stage):
    import pandas as pd
    d2l = d2l_module()
    os.chdir(CHAPTER)
    out, _ = lc_dirs(stage)
    s = d2l.TokenShards(out, 'train')
    s.verify()
    meta = pd.concat([s.meta(k) for k in range(s.n_shards)])
    tokens = meta.groupby('part')['n_tokens'].sum()
    ok = True
    if stage == 'lc-long':
        ok &= bool(tokens['pg19'] >= LC['pg19'] and tokens['pes2o'] >=
                   LC['pes2o'])
        ok &= bool(meta.loc[meta.part == 'pes2o', 'n_tokens'].min() >
                   LC['min_paper'])
        ok &= bool(s.shards[0]['n_tokens'] >= LC['shard'])
        ok &= bool((meta['part'].iloc[:(meta.part == 'pg19').sum()] ==
                    'pg19').all())  # PG-19 first: the T1 prefix
    else:
        ok &= len(meta) == LC['test_books'] and bool(
            (meta['n_tokens'] == LC['test_tokens'] + 1).all())
    print(f'{stage}: verified; tokens by part {tokens.to_dict()}')
    print('check passed' if ok else 'CHECK FAILED')
    return ok


def build(stage):
    d2l = d2l_module()
    tok = tokenizer(d2l, stage)
    root = stage_dir(d2l, stage)
    os.makedirs(root, exist_ok=True)
    shards = {}
    for key, make in sources(d2l, stage).items():
        shards[key] = build_stream(d2l, tok, make(), TARGETS[stage][key],
                                   os.path.join(root, key))
        print(f'{key}: {shards[key].n_docs:,} documents, '
              f'{shards[key].n_tokens:,} tokens')
    rerun = os.path.abspath(d2l._rerun_dir())
    os.makedirs(rerun, exist_ok=True)
    if stage == 'cpt-t1':
        for key, name in UPLOADS.items():
            sha1, size = deterministic_tar(os.path.join(root, key),
                                           os.path.join(rerun, name + '.tar'))
            print(f"DATA_HUB['{name}'] = ('{name}.tar', '{sha1}')  "
                  f'# {size / 2**20:.1f} MB')
        return
    shards.update(d2l._book_shards('t2'))
    table = d2l._shard_table(shards)
    layout = {k: d2l._layout_hash(s) for k, s in shards.items()}
    pins = {k: d2l.PRETRAIN_SOURCES['pes2o'][1:3] for k in ('pes2o',
                                                             'val_domain')}
    pins['replay'] = d2l.LM_DATA_SOURCES['fineweb-edu'][1:3] + (
        FWEDU_REPLAY,)
    name = 'Qwen/Qwen3-0.6B-Base'
    result = dict(table=table, layout=layout, sources=pins,
                  tokenizer=[name, d2l.HF_REVISIONS[name]],
                  attribution=ATTRIBUTION)
    path = os.path.join(rerun, 'lm-cpt-qwen-manifest-v1.json.gz')
    d2l._write_result(path, 'json-table', result,
                      dict(config=None, label=''))
    with open(path, 'rb') as f:
        print(f'wrote {path}, sha1 {hashlib.sha1(f.read()).hexdigest()}')
    print(table.to_string(index=False))


def check(stage):
    d2l = d2l_module()
    os.chdir(CHAPTER)
    root = stage_dir(d2l, stage)
    ok = True
    shards = {k: d2l.TokenShards(os.path.join(root, k), 'train')
              for k in TARGETS[stage]}
    for key, s in shards.items():
        s.verify()
        target = TARGETS[stage][key]
        good = target is None or s.n_tokens >= target
        ok &= good
        print(f'{key}: verified, {s.n_tokens:,} tokens'
              f'{"" if good else f" < target {target:,}"}')
    import pandas as pd
    meta = lambda s: pd.concat([pd.read_parquet(os.path.join(
        s.root, sh['file'] + '.meta.parquet')) for sh in s.shards])
    train, val = set(meta(shards['pes2o'])['doc_id']), set(
        meta(shards['val_domain'])['doc_id'])
    print(f'peS2o train/validation documents shared: {len(train & val)}')
    ok &= not train & val
    general = general_hashes(d2l)
    replay = set(meta(shards['replay'])['sha1'])  # recorded at build time
    print(f'replay/general-validation texts shared: {len(replay & general)}')
    ok &= not replay & general
    print('check passed' if ok else 'CHECK FAILED')
    return ok


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--stage', choices=sorted(TARGETS) + ['lc-long',
                                                         'pg19-test'],
                   required=True)
    p.add_argument('--check', action='store_true',
                   help='verify the built shards (after building, or alone '
                        'with --check-only)')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    lc = a.stage in ('lc-long', 'pg19-test')
    if not a.check_only:
        (build_lc if lc else build)(a.stage)
    if a.check or a.check_only:
        sys.exit(0 if (check_lc if lc else check)(a.stage) else 1)
