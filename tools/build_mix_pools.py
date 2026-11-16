#!/usr/bin/env python3
"""Build the token pools of the data-mixtures section (sec_lm_data_mixtures).

For one configuration of d2l.MIX_CONFIGS (t1: 20M tokens per domain and 1M
per validation set; t2: 150M and 2M) this writes, under OUT/<config>/,

  web-<size>.bin, code-<size>.bin, math-<size>.bin   training pools
  web-val-<v>.bin, code-val-<v>.bin, math-val-<v>.bin   validation sets
  <name>.docs.jsonl.gz   one line per document (id or source, tokens)
  meta.json   {<name>.bin: meta} for every file

Each .bin is the flat little-endian uint16 token ids of its documents in
source order, every document followed by one EOS (id 0), tokenized with the
reference tokenizer (lm-r2-tokenizer-v1; --tokenizer names another
tokenizer.json, for tests and smoke runs only). Sources, at the revisions of
d2l.LM_DATA_SOURCES, read through the local Hugging Face cache:

  web   FineWeb-Edu sample/10BT, files in d2l._FWEDU_FILES order: pool =
        buckets 1-7, validation = bucket 8 (d2l.fwedu_bucket of the id)
  math  FineMath finemath-4plus, files in order: validation when
        sha256(url) % 64 == 0, else pool
  code  CodeParrot-clean, files in order, licences mit, apache-2.0,
        bsd-2-clause, bsd-3-clause and isc only: validation when
        sha256(repo_name/path) % 64 == 0, else pool

Texts are NFC-normalized; empty documents are skipped. A pool closes at the
first document boundary at or after its quota. The build is deterministic:
--check rebuilds into a temporary directory and compares every file's
SHA-256. --tokens POOL,VAL builds smaller pools under the same names (smoke
runs only). --tar DIR also writes the upload tars of the registered keys (web
and math pools, their validation sets, and the code meta: CodeParrot's files
are not hosted, so the code key holds the meta and the per-file attribution
only).

Usage (from the repository, .venv-lm, HF_HOME set):
  python tools/build_mix_pools.py --config t1 [--out data/lm/mix] [--check]
      [--tar data/lm/mix/upload] [--tokenizer PATH] [--domains web,math]
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import pathlib
import sys
import tarfile
import tempfile
import time
import unicodedata

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d2l import torch as d2l  # noqa: E402

PERMISSIVE = ('mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'isc')
CODE_FILES = tuple(f'file-{k:012d}.json.gz' for k in range(1, 54))
MATH_FILES = tuple(f'finemath-4plus/train-{k:05d}-of-00064.parquet'
                   for k in range(64))


def sizes(name, tokens=None):
    """The configuration, with smaller quotas (pool, val) for smoke runs:
    the files keep the configuration's names."""
    c = dict(d2l.MIX_CONFIGS[name])
    if tokens is not None:
        c['pool_tokens'], c['val_tokens'] = tokens
    return c


def mod64(s):
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 64


def hub_file(source, filename):
    """Local path of a file of a pinned dataset (downloaded once)."""
    from huggingface_hub import hf_hub_download
    _, repo, revision, _ = d2l.LM_DATA_SOURCES[source]
    return hf_hub_download(repo, filename, revision=revision,
                           repo_type='dataset')


def web_docs():
    import pyarrow.parquet as pq
    for name in d2l._FWEDU_FILES:
        pf = pq.ParquetFile(hub_file('fineweb-edu', name))
        row = 0
        for batch in pf.iter_batches(columns=['id', 'text'], batch_size=4096):
            for doc_id, text in zip(batch.column('id').to_pylist(),
                                    batch.column('text').to_pylist()):
                b = d2l.fwedu_bucket(doc_id)
                split = ('pool' if b in d2l.FWEDU_BUCKETS['side'] else
                         'val' if b in d2l.FWEDU_BUCKETS['side_val'] else None)
                if split:
                    yield split, text, dict(doc_id=doc_id, src_file=name,
                                            src_row=row)
                row += 1


def math_docs():
    import pyarrow.parquet as pq
    for name in MATH_FILES:
        pf = pq.ParquetFile(hub_file('finemath', name))
        row = 0
        for batch in pf.iter_batches(columns=['url', 'text'],
                                     batch_size=4096):
            for url, text in zip(batch.column('url').to_pylist(),
                                 batch.column('text').to_pylist()):
                split = 'val' if mod64(url) == 0 else 'pool'
                yield split, text, dict(doc_id=f'{name}#{row}', url=url,
                                        src_file=name, src_row=row)
                row += 1


def code_docs():
    for name in CODE_FILES:
        with gzip.open(hub_file('codeparrot-clean', name), 'rt',
                       encoding='utf-8') as f:
            for row, line in enumerate(f):
                r = json.loads(line)
                if r['license'] not in PERMISSIVE:
                    continue
                key = f'{r["repo_name"]}/{r["path"]}'
                split = 'val' if mod64(key) == 0 else 'pool'
                yield split, r['content'], dict(
                    doc_id=key, repo_name=r['repo_name'], path=r['path'],
                    license=r['license'], src_file=name, src_row=row,
                    sha256=hashlib.sha256(
                        r['content'].encode('utf-8')).hexdigest())


SOURCES = {  # domain: (documents, LM_DATA_SOURCES name, split rule)
    'web': (web_docs, 'fineweb-edu', {
        'pool': 'fwedu_bucket(id) in 1-7', 'val': 'fwedu_bucket(id) == 8'}),
    'math': (math_docs, 'finemath', {
        'pool': 'sha256(url) % 64 != 0', 'val': 'sha256(url) % 64 == 0'}),
    'code': (code_docs, 'codeparrot-clean', {
        'pool': f'license in {PERMISSIVE}, sha256(repo_name/path) % 64 != 0',
        'val': f'license in {PERMISSIVE}, sha256(repo_name/path) % 64 == 0'})}


def build_domain(domain, c, tok, tok_sha, byte_len, special, out):
    """Tokenize one domain's documents into its pool and validation set."""
    docs, source, rule = SOURCES[domain]
    quota = {'pool': c['pool_tokens'], 'val': c['val_tokens']}
    names = {'pool': f'{domain}-{c["pool"]}', 'val': f'{domain}-val-{c["val"]}'}
    parts = {s: [] for s in quota}
    index = {s: [] for s in quota}
    n_tok, n_nfc = dict.fromkeys(quota, 0), dict.fromkeys(quota, 0)
    files = {s: {} for s in quota}
    empty, batch = 0, []

    def flush():
        ids = d2l._doc_token_ids(tok, [t for _, t, _ in batch])
        for (split, text, info), x in zip(batch, ids):
            if n_tok[split] >= quota[split]:
                continue
            x = np.asarray(list(x) + [0], dtype=np.uint16)  # EOS = id 0
            assert not np.isin(x[:-1], special).any(), info
            parts[split].append(x)
            index[split].append(dict(info, n_tokens=len(x)))
            n_tok[split] += len(x)
            n_nfc[split] += len(text.encode('utf-8'))
            f = files[split]
            f[info['src_file']] = max(f.get(info['src_file'], 0),
                                      info['src_row'] + 1)
        batch.clear()

    for split, text, info in docs():
        if all(n_tok[s] >= quota[s] for s in quota):
            break
        if n_tok[split] >= quota[split]:
            continue
        text = unicodedata.normalize('NFC', text)
        if not text:
            empty += 1
            continue
        batch.append((split, text, info))
        if len(batch) == 1000:
            flush()
    flush()
    meta = {}
    _, repo, revision, licence = d2l.LM_DATA_SOURCES[source]
    for split in quota:
        if n_tok[split] < quota[split]:
            raise RuntimeError(f'{domain} {split}: only {n_tok[split]:,} '
                               f'tokens in the files read')
        ids = np.concatenate(parts[split])
        stem = names[split]
        (out / f'{stem}.bin').write_bytes(ids.astype('<u2').tobytes())
        docs_bytes = gzip.compress(''.join(
            json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n'
            for r in index[split]).encode('utf-8'), mtime=0)
        (out / f'{stem}.docs.jsonl.gz').write_bytes(docs_bytes)
        n_scored = int(byte_len[ids].sum())
        assert n_scored == n_nfc[split], (domain, split, n_scored,
                                          n_nfc[split])
        meta[f'{stem}.bin'] = dict(
            domain=domain, split=split, n_tokens=int(len(ids)),
            n_docs=len(index[split]), n_bytes_nfc=n_nfc[split],
            n_bytes_scored=n_scored, dtype='uint16', eos_id=0,
            vocab_size=len(byte_len), tokenizer_sha256=tok_sha,
            source=repo, revision=revision, license=licence,
            split_rule=rule[split], files=files[split],
            empty_skipped=empty, docs=f'{stem}.docs.jsonl.gz',
            docs_sha256=hashlib.sha256(docs_bytes).hexdigest(),
            sha256=hashlib.sha256(ids.astype('<u2').tobytes()).hexdigest())
        print(f'{stem}: {len(ids):,} tokens, {len(index[split]):,} docs, '
              f'{n_nfc[split] / len(ids):.3f} bytes per token, files '
              f'{files[split]}', flush=True)
    return meta


def load_tokenizer(path):
    if path is None:
        tok = d2l.r2_tokenizer(raw=True)
        path = d2l.download_keyed('lm-r2-tokenizer-v1')
    else:
        tok = d2l._r2_from_file(path, raw=True)
    sha = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    byte_len = d2l.token_byte_lengths(tok)
    special = np.array(sorted(d2l._tok_info(tok)[2]), dtype=np.uint16)
    assert len(byte_len) <= 65536, 'uint16 pools need at most 65,536 ids'
    return tok, sha, byte_len, special


def build(name, out, tokenizer, domains, tokens=None):
    c = sizes(name, tokens)
    out = pathlib.Path(out) / name
    out.mkdir(parents=True, exist_ok=True)
    tok, sha, byte_len, special = load_tokenizer(tokenizer)
    meta_path = out / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for d in domains:
        t0 = time.time()
        meta.update(build_domain(d, c, tok, sha, byte_len, special, out))
        print(f'{d}: {time.time() - t0:.0f} s', flush=True)
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
    return out


def tar_bytes(members):
    """A deterministic tar of (name, bytes) pairs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.USTAR_FORMAT) as t:
        for name, data in sorted(members):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def write_tars(name, built, dest):
    """The upload tars of the registered keys of the configuration."""
    c = sizes(name)
    meta = json.loads((built / 'meta.json').read_text())
    groups = {}
    for stem, key in c['keys'].items():
        groups.setdefault(key, []).append(stem)
    groups[c['code_meta']] = [f'code-{c["pool"]}', f'code-val-{c["val"]}']
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for key, stems in groups.items():
        sub = {f'{s}.bin': meta[f'{s}.bin'] for s in stems}
        members = [('meta.json', (json.dumps(sub, indent=1, sort_keys=True)
                                  + '\n').encode())]
        for s in stems:
            members.append((f'{s}.docs.jsonl.gz',
                            (built / f'{s}.docs.jsonl.gz').read_bytes()))
            if key != c['code_meta']:
                members.append((f'{s}.bin', (built / f'{s}.bin').read_bytes()))
        data = tar_bytes(members)
        (dest / f'{key}.tar').write_bytes(data)
        print(f'{key}.tar: {len(data) / 2**20:.1f} MiB, sha1 '
              f'{hashlib.sha1(data).hexdigest()}')


def file_hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(pathlib.Path(folder).iterdir()) if p.is_file()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'lm' / 'mix'))
    ap.add_argument('--tokenizer', default=None,
                    help='tokenizer.json in place of the reference one '
                    '(tests and smoke runs only)')
    ap.add_argument('--domains', default=','.join(d2l.MIX_DOMAINS))
    ap.add_argument('--tokens', default=None,
                    help='POOL,VAL token quotas in place of the '
                    "configuration's (smoke runs only)")
    ap.add_argument('--check', action='store_true',
                    help='rebuild into a temporary directory and compare')
    ap.add_argument('--tar', default=None, help='write the upload tars here')
    args = ap.parse_args()
    domains = args.domains.split(',')
    tokens = args.tokens and tuple(int(x) for x in args.tokens.split(','))
    built = build(args.config, args.out, args.tokenizer, domains, tokens)
    if args.check:
        with tempfile.TemporaryDirectory(dir=args.out) as tmp:
            again = build(args.config, tmp, args.tokenizer, domains,
                          tokens)
            a, b = file_hashes(built), file_hashes(again)
            bad = [k for k in b if a.get(k) != b[k]]
            if bad:
                sys.exit(f'--check: {bad} differ between two builds')
            print(f'--check: {len(b)} files identical')
    if args.tar:
        write_tars(args.config, built, args.tar)


if __name__ == '__main__':
    main()
