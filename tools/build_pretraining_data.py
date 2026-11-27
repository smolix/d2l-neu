#!/usr/bin/env python3
"""Build the token shards of sec_lm_continued_pretraining (21.4) and
sec_lm_long_context (21.5).

21.4's sources (pinned; attribution in every manifest):
  * domain: reasoning traces, OpenThoughts3-1.2M
    (open-thoughts/OpenThoughts3-1.2M @61bcf9d4, Apache-2.0; the licence is
    re-read from the card at build and the build stops if it changed). The
    pinned subset is every 10th of its 120 parquet files
    (d2l.PRETRAIN_SOURCE_FILES['openthoughts3']: 12 files, 2.85 GB, 120,000
    conversations, the dataset's domain order spanned). One document = one
    conversation (one user turn, one assistant turn that opens a <think>
    block). Most assistant turns stop at the annotator's generation cap
    (about 16K tokens): about two thirds of the selected documents never
    close the <think> block, and some that close it are cut in the answer.
    Every document gets an ending, recorded per split: 'open' (no
    </think>), 'capped' (closed, but the assistant turn holds at least
    TRACES['near_cap'] tokens, where answers start being cut) or
    'complete'; TRACES['endings'] names the endings the training and
    held-out selections keep (all three: the dataset as published). Its id is the sha1 of its
    ChatML text (d2l.chatml); its
    question id the sha1 of the user turn after NFKC, lower case and one
    space per whitespace run. Exact duplicates keep one copy.
    Decontamination first: a document whose user turn shares a word
    13-gram (NFKC, lower case, \\w+ words) with a GSM8K test question or a
    MATH-500 problem (HuggingFaceH4/MATH-500 @6e4ed1a2) is dropped, as is
    one whose text holds one of Qwen3's special-token strings. Then, in id
    order: the training selection takes documents until it holds
    TRACES['train'] Qwen tokens (EOS included); the held-out set takes the
    next documents whose question the selection does not ask, one per
    question, dropping (and recording) those whose 50-token-gram overlap
    with the selection exceeds 20%, until it holds 500 conversations;
    val_domain is its first 2.2M tokens of whole conversations.
  * replay: FineWeb-Edu sample/10BT/005_00000.parquet (@87f09149), a file
    the reference model's shards never read (d2l._FWEDU_FILES), without
    texts of the general validation set.
  * general validation: the reference model's FineWeb-Edu validation texts
    (data/lm/fwedu-r2-v1/val/val.jsonl.gz, i.e. d2l.reference_shards('val')).
  * the book's held-out sections are built by the library (d2l._book_shards)
    on first use; this script builds them too so that the T2 manifest can
    list them. No arm trains on the book.

Stages:
  --stage cpt-qwen-traces  the Qwen3-tokenized (uint32) T2 shards, local only
                   (about 0.5 GB): data/lm/cpt-qwen-traces-v1/{traces,
                   val_domain,replay,val_general}/ (traces in ChatML, EOS
                   151,643 after each document), heldout.jsonl.gz (the 500
                   held-out conversations, the SmolLM3 cell's "traces
                   (seen)" windows), split.json (the selection: ids, file
                   and row of every training and held-out document, the
                   counts) and the committed summary
                   data/lm/rerun/<PRETRAIN_ARTIFACTS['cpt-qwen-manifest']>
                   .json.gz (json-table: the stream table, each stream's
                   layout hash, sources and file list, licence, attribution,
                   decontamination, overlap, per-domain and per-ending
                   counts, the sha1 of the training and held-out id lists).
                   The licence is re-read from the card on every build,
                   also when split.json is reused. Replay and general validation
                   are rebuilt byte-equal to cpt-qwen-v1's (layout hashes
                   checked against lm-cpt-qwen-manifest-v1).
  --stage cpt-t1-traces  the R2-tokenized (uint16) T1 slices, hosted: the
                   same selection as plain text ("User: ...\\nAssistant:
                   ..."), training documents in id order to 12M tokens and
                   held-out ones to 2.2M, plus the unchanged replay slice:
                   data/lm/cpt-t1-traces-v1/{traces,val_domain,replay}/ and
                   the upload tars data/lm/rerun/lm-traces-12m-v1.tar and
                   lm-traces-val-v1.tar (sha1 and size printed for
                   DATA_HUB); the replay tar must equal the hosted
                   lm-fwedu-replay-12m-v1. Reads split.json (computes it
                   when missing).
  --check          verify every shard set of the stage (SHA-256 and EOS
                   invariants), the token targets (the arms' windows within
                   one epoch: T2 305 x 128 x 2,048, T1 183 x 64 x 1,024),
                   the file list against the pin, the endings recounted,
                   ids and question ids recomputed from the parquet, the
                   tokens re-encoded, training and held-out questions and
                   texts disjoint, decontamination, the held-out overlap
                   target (under 2% of held-out 50-grams, none over 20%).

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

  --stage domains  21.2's per-domain validation texts (lm-fwedu-val-domains-v1,
                   jsonl.gz with doc_id, domain, url_host, text): FineWeb-Edu
                   sample/10BT/012_00000.parquet (@87f09149; a file the
                   reference model's shards never read) by URL host into
                   wikipedia, edu (.edu, .edu.xx, .ac.xx), gov (.gov,
                   .gov.xx), org and other, each filled to 2,000,000 R2 tokens
                   in a fixed pseudo-random order (sha1 of the document id),
                   plus the book's held-out sections (book_split(): every
                   fifth section; CC BY-SA 4.0) as domain book:
                   data/lm/rerun/lm-fwedu-val-domains-v1.jsonl.gz.
                   --check: the token targets, no text shared with the
                   general validation texts, every host in its domain.

Deterministic: the same inputs give the same bytes (no timestamps, files read
in order, one shard per stream), except the gzip header of the json-table
manifest, which d2l._write_result writes with the file's time and temporary
name (its content is the same). Run from the repository root:

    HF_HOME=/home/smola/.cache/hf-lm .venv-lm/bin/python \\
        tools/build_pretraining_data.py --stage cpt-qwen-traces [--check]

Expected time (measured 2026-11-27 on 6 workers): cpt-qwen-traces about
4 minutes including its --check, after the 2.8 GB download;
cpt-t1-traces under a minute; lc-long about 5 minutes, pg19-test under a
minute; up to 10 GB of RAM.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import tarfile
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER = os.path.join(ROOT, 'chapter_lm-pretraining')
FWEDU_REPLAY = 'sample/10BT/005_00000.parquet'

TRACES = dict(  # 21.4's reasoning-trace corpus (tokens include each EOS)
    train=84_000_000,  # Qwen tokens of the training selection
    seq_len=2048, steps=305, batch=128,  # the arms: 305 x 128 windows
    heldout=500,  # held-out conversations, one per question
    near_cap=15_500,  # assistant tokens from which closed answers get cut
    endings=('complete', 'capped', 'open'),  # the endings selected (a
    # complete-only selection reads ids up to 3d0c...: prefetch '5')
    val=2_200_000,  # val_domain: whole held-out conversations, id order
    gram=50, max_overlap=0.20, overlap_target=0.02,  # held-out vs train
    contam=13,  # word n-grams shared with GSM8K test or MATH-500
    t1_train=12_000_000, t1_val=2_200_000,  # R2 tokens (T1)
    t1_seq_len=1024, t1_steps=183, t1_batch=64,  # the T1 arms' windows
    replay_t1=12_000_000, replay_t2=24_000_000,
    prefetch='2',  # texts kept by the scan for ids below this (1/8)
    workers=6)
TRACE_UPLOADS = {'traces': 'lm-traces-12m-v1',
                 'val_domain': 'lm-traces-val-v1'}
REPLAY_T1 = 'lm-fwedu-replay-12m-v1'  # hosted; rebuilt byte-equal
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
FWEDU_ATTRIBUTION = (
    'FineWeb-Edu (HuggingFaceFW/fineweb-edu, ODC-By 1.0; Hugging Face). '
    'Token shards derived for Dive into Deep Learning, section "Continued '
    'Pretraining and Mid-Training".')
TRACES_ATTRIBUTION = (
    'OpenThoughts3-1.2M (open-thoughts/OpenThoughts3-1.2M, Apache-2.0; '
    'OpenThoughts team, Guha et al. 2025), a pinned subset of its files. '
    'Token shards derived for Dive into Deep Learning, section "Continued '
    'Pretraining and Mid-Training".')


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


def tokenizer(d2l, tier):
    os.chdir(CHAPTER)  # the library's relative data paths
    return d2l._cpt_tokenizer(tier)


def build_stream(d2l, tok, docs, target, out, attribution=FWEDU_ATTRIBUTION,
                 eos_id=None, meta_fn=None):
    """One shard: documents in order until the shard holds target tokens
    (None: all of them)."""
    import shutil
    tmp = out + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    kw = (dict(shard_tokens=target, max_shards=1) if target else
          dict(shard_tokens=10 ** 12))
    d2l.write_shards(docs, tok, tmp, 'train', eos_id=eos_id,
                     meta_fn=meta_fn, **kw)
    with open(os.path.join(tmp, 'manifest.json')) as f:
        m = json.load(f)
    m['name'] = os.path.basename(out)  # not the temporary folder's name
    m['attribution'] = attribution
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
    (write_shards then stores exactly those ids); eos is its only special
    token."""
    special_tokens_set = {'<|endoftext|>'}

    def __init__(self, ids, n_vocab, eos=0):
        self.ids, self.n_vocab, self.eos = ids, n_vocab, eos

    def encode_single_token(self, s):
        return self.eos

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
    m['name'] = os.path.basename(out)  # not the temporary folder's name
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


FWEDU_DOMAINS = 'sample/10BT/012_00000.parquet'
DOMAIN_TOKENS = 2_000_000
DOMAINS_KEY = 'lm-fwedu-val-domains-v1'


def url_domain(host):
    """The domain class of a URL host."""
    if host == 'wikipedia.org' or host.endswith('.wikipedia.org'):
        return 'wikipedia'
    if host.endswith('.edu') or '.edu.' in host or '.ac.' in host:
        return 'edu'
    if host.endswith('.gov') or '.gov.' in host:
        return 'gov'
    return 'org' if host.endswith('.org') else 'other'


def domains_path(d2l):
    return os.path.join(os.path.abspath(d2l._rerun_dir()),
                        DOMAINS_KEY + '.jsonl.gz')


def build_domains():
    """FineWeb-Edu file 012 by URL host, 2M R2 tokens per domain, and the
    book's held-out sections; one deterministic jsonl.gz."""
    import urllib.parse
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    d2l = d2l_module()
    tok = tokenizer(d2l, 't1')  # the reference tokenizer; chdir
    _, repo, rev, _ = d2l.LM_DATA_SOURCES['fineweb-edu']
    local = hf_hub_download(repo, FWEDU_DOMAINS, repo_type='dataset',
                            revision=rev)
    table = pq.read_table(local, columns=['id', 'url', 'text'])
    ids = table.column('id').to_pylist()
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha1(
        ids[i].encode()).hexdigest())
    urls, texts = table.column('url').to_pylist(), table.column('text')
    general = general_hashes(d2l)
    rows, counts = [], {}
    for i in order:
        host = (urllib.parse.urlsplit(urls[i]).hostname or '').lower()
        dom = url_domain(host)
        if counts.get(dom, 0) >= DOMAIN_TOKENS:
            if len(counts) == 5 and min(counts.values()) >= DOMAIN_TOKENS:
                break
            continue
        text = texts[i].as_py()
        if hashlib.sha1(text.encode()).hexdigest() in general:
            continue
        counts[dom] = counts.get(dom, 0) + len(tok.encode(text))
        rows.append(dict(doc_id=f'fwedu:{ids[i]}', domain=dom,
                         url_host=host, text=text))
    rows.sort(key=lambda r: (r['domain'], r['doc_id']))
    _, held = d2l.book_split()
    book = 0
    for doc_id, text in held:
        if book >= DOMAIN_TOKENS:
            break
        book += len(tok.encode(text))
        rows.append(dict(doc_id=f'book:{doc_id}', domain='book',
                         url_host='d2l.ai', text=text))
    counts['book'] = book
    path = domains_path(d2l)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                           mtime=0) as f:
            for r in rows:
                f.write((json.dumps(r, ensure_ascii=False) + '\n').encode())
    with open(path, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    for dom in sorted(counts):
        n = sum(r['domain'] == dom for r in rows)
        print(f'{dom}: {n:,} documents, {counts[dom]:,} R2 tokens')
    print(f"DATA_HUB['{DOMAINS_KEY}'] = ('{DOMAINS_KEY}.jsonl.gz', "
          f"'{sha1}')  # {os.path.getsize(path) / 2**20:.1f} MB")


def check_domains():
    d2l = d2l_module()
    tok = tokenizer(d2l, 't1')
    with gzip.open(domains_path(d2l), 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    general = general_hashes(d2l)
    ok, counts = True, {}
    for r in rows:
        counts[r['domain']] = counts.get(r['domain'], 0) + len(
            tok.encode(r['text']))
        if r['domain'] != 'book':
            ok &= url_domain(r['url_host']) == r['domain']
        ok &= hashlib.sha1(r['text'].encode()).hexdigest() not in general
    ids = [r['doc_id'] for r in rows]
    ok &= len(ids) == len(set(ids))
    for dom, n in sorted(counts.items()):
        good = dom == 'book' or n >= DOMAIN_TOKENS
        ok &= good
        print(f'{dom}: {n:,} R2 tokens{"" if good else " < target"}')
    print('check passed' if ok else 'CHECK FAILED')
    return ok


# ---------------------------------------------------------------------------
# 21.4's reasoning traces (OpenThoughts3-1.2M)

def trace_words(text):
    """Lower-case NFKC word tokens (the decontamination's unit)."""
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).lower())


def word_ngrams(text, n):
    w = trace_words(text)
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}


def question_id(user):
    """sha1 of a user turn after NFKC, lower case and one space per
    whitespace run: every annotation of a question shares it."""
    norm = ' '.join(unicodedata.normalize('NFKC', user).lower().split())
    return hashlib.sha1(norm.encode()).hexdigest()


def trace_messages(conv):
    """OpenThoughts3's turns ({from: human|gpt, value}) as role/content."""
    roles = {'human': 'user', 'gpt': 'assistant'}
    return [dict(role=roles[m['from']], content=m['value']) for m in conv]


def trace_plain(messages):
    """The T1 rendering for the reference tokenizer (no chat tokens)."""
    names = {'user': 'User', 'assistant': 'Assistant'}
    return '\n'.join(f'{names[m["role"]]}: {m["content"]}' for m in messages)


def trace_record(d2l, messages, grams, n, specials):
    """A conversation's record: doc_id (sha1 of its ChatML text),
    question_id, the benchmarks whose word n-grams its user turn shares
    (contam), whether a message holds a special-token string, and the
    messages."""
    user = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
    g = word_ngrams(user, n)
    return dict(
        doc_id=hashlib.sha1(d2l.chatml(messages).encode()).hexdigest(),
        question_id=question_id(user),
        contam=sorted(name for name, bench in grams.items() if g & bench),
        special=any(s in m['content'] for m in messages for s in specials),
        messages=messages)


ENDINGS = ('complete', 'capped', 'open')


def trace_ending(record, n_assistant, near_cap):
    """'open' when the assistant turn never closes its <think> block (the
    generation stopped inside it), 'capped' when it closes it but holds at
    least near_cap tokens (n_assistant; the answer may be cut at the same
    cap), else 'complete'."""
    reply = ''.join(m['content'] for m in record['messages']
                    if m['role'] == 'assistant')
    if '</think>' not in reply:
        return 'open'
    return 'capped' if n_assistant >= near_cap else 'complete'


def assistant_tokens(im_start):
    """ids -> the number of tokens after the last <|im_start|> (id
    im_start): the assistant turn with its role line and end marker."""
    def count(ids):
        at = np.flatnonzero(np.asarray(ids) == im_start)
        return len(ids) - int(at[-1]) - 1 if len(at) else len(ids)
    return count


def endings_count(pairs, ending):
    """{ending: dict(documents, tokens)} of (record, ids) pairs (one EOS per
    document counted); ending(record, ids) -> one of ENDINGS."""
    out = {e: dict(documents=0, tokens=0) for e in ENDINGS}
    for rec, ids in pairs:
        c = out[ending(rec, ids)]
        c['documents'] += 1
        c['tokens'] += len(ids) + 1
    return out


def scan_traces_file(job):
    """The records of one parquet file in row order, with file (its index
    in the pin), row, domain, source and difficulty; messages are kept only
    for ids below prefix (the start of the id order, all a selection
    reads)."""
    import pyarrow.parquet as pq
    k, path, grams, n, specials, prefix = job
    d2l = d2l_module()
    out = []
    pf = pq.ParquetFile(path)
    for g in range(pf.num_row_groups):
        for d in pf.read_row_group(g).to_pylist():
            r = trace_record(d2l, trace_messages(d['conversations']), grams,
                             n, specials)
            r.update(file=k, row=len(out), domain=d['domain'],
                     source=d['source'], difficulty=d['difficulty'])
            if r['doc_id'] >= prefix:
                r['messages'] = None
            out.append(r)
    return out


def dedupe_sort(records):
    """Records sorted by doc_id, one per id (an exact duplicate keeps the
    copy first in (domain, source, difficulty) order, so the result does
    not depend on the order of the files), and the number dropped."""
    key = lambda r: (r['doc_id'], r['domain'], r['source'], r['difficulty'],
                     r['file'], r['row'])
    out = []
    for r in sorted(records, key=key):
        if not out or out[-1]['doc_id'] != r['doc_id']:
            out.append(r)
    return out, len(records) - len(out)


def ngram_hashes(d2l, ids, n):
    return d2l._rolling_ngram_hashes(ids, n)[0]


def select_traces(d2l, docs, token_ids, train_tokens, n_heldout, gram,
                  max_overlap, batch=256, keep=None):
    """The training selection and the held-out set from records sorted by
    doc_id. Usable records (no benchmark n-gram, no special-token string)
    are tokenized in id order, in batches, by token_ids(records) -> id
    arrays; with keep(record, ids) given, only those it accepts are used
    (both selections). Training takes them until it holds train_tokens tokens (one EOS
    per document counted). Held-out takes the following ones whose question
    training does not ask, one try per question, and drops (records) a
    document more than max_overlap of whose gram-token windows occur in
    the training documents. Returns dict(train, heldout: lists of (record,
    ids); overlap: (hits, windows) per held-out document; dropped ids)."""
    usable = [r for r in docs if not r['contam'] and not r['special']]

    def stream():
        for i in range(0, len(usable), batch):
            part = usable[i:i + batch]
            if any(r['messages'] is None for r in part):
                raise RuntimeError('the scan kept too few texts; raise '
                                   "TRACES['prefetch']")
            for rec, ids in zip(part, token_ids(part)):
                if keep is None or keep(rec, ids):
                    yield rec, ids

    it, train, n = stream(), [], 0
    for rec, ids in it:
        train.append((rec, ids))
        n += len(ids) + 1
        if n >= train_tokens:
            break
    else:
        raise RuntimeError(f'only {n:,} tokens for a {train_tokens:,} target')
    known = np.unique(np.concatenate(
        [ngram_hashes(d2l, ids, gram) for _, ids in train]))
    asked = {r['question_id'] for r, _ in train}
    heldout, overlap, dropped = [], [], []
    for rec, ids in it:
        if rec['question_id'] in asked:
            continue
        asked.add(rec['question_id'])
        h = ngram_hashes(d2l, ids, gram)
        pos = np.minimum(np.searchsorted(known, h), len(known) - 1)
        hits = int((known[pos] == h).sum()) if len(known) else 0
        if hits > max_overlap * len(h):
            dropped.append(rec['doc_id'])
            continue
        heldout.append((rec, ids))
        overlap.append((hits, len(h)))
        if len(heldout) == n_heldout:
            break
    else:
        raise RuntimeError(f'only {len(heldout)} held-out conversations')
    return dict(train=train, heldout=heldout, overlap=overlap,
                dropped=dropped)


def prefix_docs(pairs, tokens):
    """The shortest prefix of (record, ids) pairs holding at least tokens
    tokens (one EOS per document counted)."""
    n = 0
    for k, (_, ids) in enumerate(pairs):
        n += len(ids) + 1
        if n >= tokens:
            return pairs[:k + 1]
    raise RuntimeError(f'only {n:,} tokens for a {tokens:,} target')


def count_by(records, field='domain'):
    out = {}
    for r in records:
        out[r[field]] = out.get(r[field], 0) + 1
    return dict(sorted(out.items()))


def split_record(d2l, records, sel, duplicates, targets, ending):
    """The selection as stored in split.json: the pin, the targets, the
    entries [doc_id, question_id, domain, source, file, row] of training
    and held-out documents (held-out ones with their overlap hits and
    windows), the dropped ids and the counts (with each split's endings:
    ending(record, ids) -> 'complete', 'capped' or 'open')."""
    entry = lambda r: [r['doc_id'], r['question_id'], r['domain'],
                       r['source'], r['file'], r['row']]
    rtrain = [r for r, _ in sel['train']]
    rheld = [r for r, _ in sel['heldout']]
    hits = sum(h for h, _ in sel['overlap'])
    windows = sum(w for _, w in sel['overlap'])
    val = prefix_docs(sel['heldout'], targets['val'])
    tokens = lambda pairs: sum(len(i) + 1 for _, i in pairs)
    counts = dict(
        rows=len(records) + duplicates, duplicates=duplicates,
        documents=len(records),
        questions=len({r['question_id'] for r in records}),
        by_domain=count_by(records),
        contaminated={b: sum(b in r['contam'] for r in records)
                      for b in ('gsm8k', 'math500')},
        special=sum(r['special'] for r in records),
        train=dict(documents=len(rtrain), tokens=tokens(sel['train']),
                   questions=len({r['question_id'] for r in rtrain}),
                   by_domain=count_by(rtrain),
                   endings=endings_count(sel['train'], ending)),
        heldout=dict(documents=len(rheld), tokens=tokens(sel['heldout']),
                     by_domain=count_by(rheld),
                     endings=endings_count(sel['heldout'], ending),
                     overlap=hits / max(windows, 1),
                     max_overlap=max(h / max(w, 1)
                                     for h, w in sel['overlap']),
                     dropped=len(sel['dropped'])),
        val_domain=dict(documents=len(val), tokens=tokens(val),
                        by_domain=count_by([r for r, _ in val]),
                        endings=endings_count(val, ending)))
    return dict(files=list(d2l.PRETRAIN_SOURCE_FILES['openthoughts3']),
                pin=list(d2l.PRETRAIN_SOURCES['openthoughts3'][1:3]),
                targets=dict(targets),
                train=[entry(r) for r in rtrain],
                heldout=[entry(r) + list(o)
                         for r, o in zip(rheld, sel['overlap'])],
                dropped=sel['dropped'], counts=counts)


def trace_licence(d2l):
    """The licence read from the dataset card at the pin; exits if it is
    not the pinned one (or the pin does not resolve to itself)."""
    from huggingface_hub import HfApi
    _, repo, rev, licence = d2l.PRETRAIN_SOURCES['openthoughts3']
    info = HfApi().dataset_info(repo, revision=rev)
    found = (info.card_data or {}).get('license')
    if info.sha != rev or found != licence:
        sys.exit(f'{repo}@{rev}: licence {found!r}, pinned {licence!r}: '
                 'stop and ask before building')
    return found


def trace_paths(d2l):
    """The pinned subset's local parquet paths (downloaded once) and the
    licence re-read from the card (trace_licence)."""
    from huggingface_hub import hf_hub_download
    found = trace_licence(d2l)
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES['openthoughts3']
    return [hf_hub_download(repo, f, repo_type='dataset', revision=rev)
            for f in d2l.PRETRAIN_SOURCE_FILES['openthoughts3']], found


def benchmark_grams(d2l, n):
    """Word n-grams of the GSM8K test questions and MATH-500 problems."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    gsm = pd.read_parquet(d2l.download_keyed('gsm8k-test'))
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES['math500']
    path = hf_hub_download(repo, d2l.PRETRAIN_SOURCE_FILES['math500'][0],
                           repo_type='dataset', revision=rev)
    with open(path, encoding='utf-8') as f:
        math = [json.loads(line)['problem'] for line in f]
    grams = lambda texts: set().union(*(word_ngrams(t, n) for t in texts))
    return {'gsm8k': grams(gsm['question']), 'math500': grams(math)}


def special_strings(tok):
    """The special-token strings of a Transformers tokenizer."""
    return sorted({t.content for t in tok.added_tokens_decoder.values()
                   if t.special})


def chatml_ids(d2l, tok):
    """records -> their ChatML token ids (the chat markers become the
    tokenizer's special tokens; the special-string filter keeps such
    strings out of the contents)."""
    def ids(records):
        out = tok([d2l.chatml(r['messages']) for r in records],
                  add_special_tokens=False, verbose=False)['input_ids']
        return [np.asarray(x, dtype=np.int64) for x in out]
    return ids


def qwen_ending(tok, near_cap):
    """ending(record, ids) for ChatML ids under a Qwen tokenizer."""
    count = assistant_tokens(tok.convert_tokens_to_ids('<|im_start|>'))
    return lambda rec, ids: trace_ending(rec, count(ids), near_cap)


def traces_split(d2l, tok, root):
    """(split, selection): split.json's content, read when root holds it,
    else computed (scan, decontamination, selection) and written; the
    selection's (record, ids) pairs when computed now, else None. Either
    way the licence is re-read from the card first, and a split.json
    written under another licence, near_cap or set of endings is
    refused."""
    path = os.path.join(root, 'split.json')
    if os.path.exists(path):
        licence = trace_licence(d2l)
        with open(path) as f:
            split = json.load(f)
        got = {k: split['targets'].get(k) for k in ('near_cap', 'endings')}
        want = dict(near_cap=TRACES['near_cap'],
                    endings=list(TRACES['endings']))
        if split.get('licence') != licence or got != want:
            sys.exit(f'{path}: written for licence {split.get("licence")!r}'
                     f' and {got}, now {licence!r} and {want}: remove it '
                     'and rebuild')
        return split, None
    import multiprocessing
    paths, licence = trace_paths(d2l)
    grams = benchmark_grams(d2l, TRACES['contam'])
    jobs = [(k, p, grams, TRACES['contam'], special_strings(tok),
             TRACES['prefetch']) for k, p in enumerate(paths)]
    with multiprocessing.get_context('fork').Pool(TRACES['workers']) as pool:
        records = [r for part in pool.map(scan_traces_file, jobs)
                   for r in part]
    docs, duplicates = dedupe_sort(records)
    ending = qwen_ending(tok, TRACES['near_cap'])
    keep = lambda rec, ids: ending(rec, ids) in TRACES['endings']
    sel = select_traces(d2l, docs, chatml_ids(d2l, tok), TRACES['train'],
                        TRACES['heldout'], TRACES['gram'],
                        TRACES['max_overlap'], keep=keep)
    split = split_record(d2l, docs, sel, duplicates, TRACES, ending)
    split.update(licence=licence, tokenizer=[
        'Qwen/Qwen3-0.6B-Base', d2l.HF_REVISIONS['Qwen/Qwen3-0.6B-Base']])
    os.makedirs(root, exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(split, f, indent=0)
    os.replace(path + '.tmp', path)
    return split, sel


def fetch_messages(d2l, entries):
    """doc_id -> messages for split entries, read from the pinned parquet
    files by (file, row); each id is checked against its text."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES['openthoughts3']
    files = d2l.PRETRAIN_SOURCE_FILES['openthoughts3']
    want = {}
    for e in entries:
        want.setdefault(e[4], []).append((e[5], e[0]))
    out = {}
    for k, rows in sorted(want.items()):
        pf = pq.ParquetFile(hf_hub_download(repo, files[k],
                                            repo_type='dataset',
                                            revision=rev))
        starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows
                                  for g in range(pf.num_row_groups)])
        for g in sorted({int(np.searchsorted(starts, r, 'right')) - 1
                         for r, _ in rows}):
            col = pf.read_row_group(g, columns=['conversations']).column(0)
            for r, doc_id in rows:
                if starts[g] <= r < starts[g + 1]:
                    msgs = trace_messages(col[r - starts[g]].as_py())
                    got = hashlib.sha1(d2l.chatml(msgs).encode()).hexdigest()
                    if got != doc_id:
                        raise ValueError(f'{files[k]} row {r}: id {got}, '
                                         f'expected {doc_id}')
                    out[doc_id] = msgs
    return out


def entry_records(entries, messages):
    return [dict(doc_id=e[0], question_id=e[1], domain=e[2], source=e[3],
                 messages=messages[e[0]]) for e in entries]


def write_traces(d2l, pairs, out, n_vocab, eos):
    """(record, ids) pairs as one shard set holding exactly those ids, EOS
    after each document, meta question_id and domain."""
    key = {r['doc_id']: ids for r, ids in pairs}
    docs = [(r['doc_id'], r['doc_id'], dict(question_id=r['question_id'],
                                            domain=r['domain']))
            for r, _ in pairs]
    return build_stream(d2l, PreTokenized(key, n_vocab, eos), docs, None,
                        out, TRACES_ATTRIBUTION, eos_id=eos,
                        meta_fn=lambda doc_id, text, meta: meta)


def write_heldout(records, path):
    """The held-out conversations as a deterministic jsonl.gz."""
    with open(path + '.tmp', 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                           mtime=0) as f:
            for r in records:
                row = {k: r[k] for k in ('doc_id', 'question_id', 'domain',
                                         'source', 'messages')}
                f.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
    os.replace(path + '.tmp', path)


def read_heldout(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def build_traces_qwen():
    """--stage cpt-qwen-traces (see the module docstring)."""
    d2l = d2l_module()
    tok = tokenizer(d2l, 't2')
    root = os.path.abspath(d2l._CPT_DIRS['t2'])
    split, sel = traces_split(d2l, tok, root)
    if sel is None:  # split.json was there: re-read and re-encode
        msgs = fetch_messages(d2l, split['train'] + split['heldout'])
        sel = {}
        for part in ('train', 'heldout'):
            recs = entry_records(split[part], msgs)
            sel[part] = list(zip(recs, chatml_ids(d2l, tok)(recs)))
    eos = d2l._lm_boundary_id(tok)
    shards = {
        'traces': write_traces(d2l, sel['train'],
                               os.path.join(root, 'traces'), len(tok), eos),
        'val_domain': write_traces(
            d2l, prefix_docs(sel['heldout'], TRACES['val']),
            os.path.join(root, 'val_domain'), len(tok), eos),
        'replay': build_stream(d2l, tok, fwedu_docs(d2l, general_hashes(d2l)),
                               TRACES['replay_t2'],
                               os.path.join(root, 'replay')),
        'val_general': build_stream(d2l, tok, val_general_docs(d2l), None,
                                    os.path.join(root, 'val_general'))}
    write_heldout([r for r, _ in sel['heldout']],
                  os.path.join(root, 'heldout.jsonl.gz'))
    for key, s in shards.items():
        print(f'{key}: {s.n_docs:,} documents, {s.n_tokens:,} tokens')
    shards['val_book'] = d2l._book_shards('t2')['val_book']
    table = d2l._shard_table(shards, 't2')
    layout = {k: d2l._layout_hash(s) for k, s in shards.items()}
    _, repo, rev, _ = d2l.PRETRAIN_SOURCES['openthoughts3']
    name = 'Qwen/Qwen3-0.6B-Base'
    ids_sha1 = lambda part: hashlib.sha1(' '.join(
        e[0] for e in split[part]).encode()).hexdigest()
    result = dict(
        table=table, layout=layout,
        sources=dict(
            traces=[repo, rev] + split['files'],
            replay=list(d2l.LM_DATA_SOURCES['fineweb-edu'][1:3]) +
            [FWEDU_REPLAY],
            gsm8k=d2l.DATA_HUB['gsm8k-test'][0],
            math500=list(d2l.PRETRAIN_SOURCES['math500'][1:3]) +
            list(d2l.PRETRAIN_SOURCE_FILES['math500'])),
        licence=dict(traces=split['licence'], replay='odc-by',
                     book='cc-by-sa-4.0'),
        tokenizer=[name, d2l.HF_REVISIONS[name]],
        attribution=TRACES_ATTRIBUTION + ' ' + FWEDU_ATTRIBUTION,
        counts=split['counts'], targets=split['targets'],
        ids_sha1=dict(train=ids_sha1('train'), heldout=ids_sha1('heldout')))
    rerun = os.path.abspath(d2l._rerun_dir())
    os.makedirs(rerun, exist_ok=True)
    key = d2l.PRETRAIN_ARTIFACTS['cpt-qwen-manifest'][0]
    path = os.path.join(rerun, key + '.json.gz')
    d2l._write_result(path, 'json-table', result,
                      dict(config=None, label=''))
    with open(path, 'rb') as f:
        print(f"DATA_HUB['{key}'] = ('{key}.json.gz', "
              f"'{hashlib.sha1(f.read()).hexdigest()}')  "
              f'# {os.path.getsize(path) / 2**10:.1f} KB')
    print(table.to_string(index=False))
    print(json.dumps(split['counts'], indent=1))


def build_traces_t1():
    """--stage cpt-t1-traces (see the module docstring)."""
    d2l = d2l_module()
    qwen = tokenizer(d2l, 't2')
    split, _ = traces_split(d2l, qwen, os.path.abspath(d2l._CPT_DIRS['t2']))
    tok = tokenizer(d2l, 't1')
    root = os.path.abspath(d2l._CPT_DIRS['t1'])
    os.makedirs(root, exist_ok=True)
    msgs = fetch_messages(d2l, split['train'] + split['heldout'])
    docs = lambda part: ((e[0], trace_plain(msgs[e[0]]),
                          dict(question_id=e[1], domain=e[2]))
                         for e in split[part])
    meta = lambda doc_id, text, m: m
    shards = {
        'traces': build_stream(d2l, tok, docs('train'), TRACES['t1_train'],
                               os.path.join(root, 'traces'),
                               TRACES_ATTRIBUTION, meta_fn=meta),
        'val_domain': build_stream(d2l, tok, docs('heldout'),
                                   TRACES['t1_val'],
                                   os.path.join(root, 'val_domain'),
                                   TRACES_ATTRIBUTION, meta_fn=meta),
        'replay': build_stream(d2l, tok, fwedu_docs(d2l, general_hashes(d2l)),
                               TRACES['replay_t1'],
                               os.path.join(root, 'replay'))}
    for key, s in shards.items():
        print(f'{key}: {s.n_docs:,} documents, {s.n_tokens:,} tokens')
    rerun = os.path.abspath(d2l._rerun_dir())
    os.makedirs(rerun, exist_ok=True)
    for key, name in TRACE_UPLOADS.items():
        sha1, size = deterministic_tar(os.path.join(root, key),
                                       os.path.join(rerun, name + '.tar'))
        print(f"DATA_HUB['{name}'] = ('{name}.tar', '{sha1}')  "
              f'# {size / 2**20:.1f} MB')


def check_trace_shards(d2l, split, shards, targets, tier):
    """The structural invariants of a trace stage's output: [(ok, line)].
    split: split.json's content; shards: {'traces', 'val_domain'} (and the
    rest) as TokenShards; targets: TRACES."""
    import pandas as pd
    out = [(split['files'] == list(d2l.PRETRAIN_SOURCE_FILES['openthoughts3'])
            and split['pin'] == list(d2l.PRETRAIN_SOURCES['openthoughts3'][
                1:3]), f'file list equals the pin ({len(split["files"])} '
            'files)')]
    for k, s in shards.items():
        try:
            s.verify()  # SHA-256 of every file, EOS and offsets
            out.append((True, f'{k}: verified, {s.n_docs:,} documents, '
                              f'{s.n_tokens:,} tokens'))
        except (ValueError, AssertionError) as e:
            out.append((False, f'{k}: {e}'))
    got = {k: list(pd.concat([shards[k].meta(i) for i in range(
        shards[k].n_shards)])['doc_id']) for k in ('traces', 'val_domain')}
    train = [e[0] for e in split['train']]
    held = [e[0] for e in split['heldout']]
    held_q = {e[1] for e in split['heldout']}
    out.append((got['traces'] == train[:len(got['traces'])],
                'training documents: the selection, in id order'))
    out.append((got['val_domain'] == held[:len(got['val_domain'])],
                'val_domain: the first held-out conversations, id order'))
    out.append((train == sorted(train) and held == sorted(held) and
                (not train or not held or train[-1] < held[0]),
                'ids increase; held-out ids follow the training ones'))
    out.append((not {e[1] for e in split['train']} & held_q and
                not set(train) & set(held),
                'training and held-out questions and texts disjoint'))
    out.append((len(held) == targets['heldout'] == len(held_q),
                f'{len(held)} held-out conversations, one per question'))
    counts = split['counts']
    shown = []
    for part in ('train', 'heldout', 'val_domain'):
        e = counts[part]['endings']
        shown.append(f'{part} ' + ' / '.join(
            f'{e[k]["documents"]:,}' for k in ENDINGS))
    out.append((all(counts[p]['endings'][k]['documents'] == 0
                    for p in ('train', 'heldout', 'val_domain')
                    for k in ENDINGS if k not in targets['endings']),
                'endings complete / capped / open (selected: ' +
                ', '.join(targets['endings']) + '): ' + '; '.join(shown)))
    c = counts['heldout']
    out.append((c['overlap'] < targets['overlap_target'] and
                c['max_overlap'] <= targets['max_overlap'],
                f'held-out {targets["gram"]}-gram overlap {c["overlap"]:.2%} '
                f'(largest {c["max_overlap"]:.1%}; {c["dropped"]} dropped)'))
    if tier == 't2':
        T, B = targets['seq_len'], targets['batch']
        steps = (shards['traces'].n_tokens - 1) // T // B
        out.append((shards['traces'].n_tokens >= targets['train'] and
                    steps >= targets['steps'],
                    f'traces: {steps} steps of {B} x {T} per epoch '
                    f'(the arms take {targets["steps"]}: no repeats)'))
        out.append((shards['val_domain'].n_tokens >= targets['val'],
                    f'val_domain: >= {targets["val"]:,} tokens'))
    else:
        T, B = targets['t1_seq_len'], targets['t1_batch']
        steps = (shards['traces'].n_tokens - 1) // T // B
        out.append((shards['traces'].n_tokens >= targets['t1_train'] and
                    steps >= targets['t1_steps'],
                    f'traces: {steps} steps of {B} x {T} per epoch '
                    f'(the T1 arms take {targets["t1_steps"]}: no repeats)'))
        out.append((shards['val_domain'].n_tokens >= targets['t1_val'],
                    f'val_domain: >= {targets["t1_val"]:,} tokens'))
    return out


def check_traces(stage):
    """--check of a trace stage: prints each invariant, returns ok."""
    d2l = d2l_module()
    qwen = tokenizer(d2l, 't2')
    t2 = os.path.abspath(d2l._CPT_DIRS['t2'])
    with open(os.path.join(t2, 'split.json')) as f:
        split = json.load(f)
    tier = 't1' if stage == 'cpt-t1-traces' else 't2'
    root = os.path.abspath(d2l._CPT_DIRS[tier])
    keys = ['traces', 'val_domain', 'replay'] + (
        ['val_general'] if tier == 't2' else [])
    shards = {k: d2l.TokenShards(os.path.join(root, k), 'train')
              for k in keys}
    lines = check_trace_shards(d2l, split, shards, TRACES, tier)
    if tier == 't2':
        held = read_heldout(os.path.join(t2, 'heldout.jsonl.gz'))
        lines.append(([r['doc_id'] for r in held] == [
            e[0] for e in split['heldout']] and all(
            hashlib.sha1(d2l.chatml(r['messages']).encode()).hexdigest() ==
            r['doc_id'] for r in held), 'heldout.jsonl.gz: the held-out '
            'ids, each the sha1 of its ChatML text'))
        msgs = fetch_messages(d2l, split['train'] + split['heldout'])
        grams = benchmark_grams(d2l, TRACES['contam'])
        specials = special_strings(qwen)
        recs = [trace_record(d2l, m, grams, TRACES['contam'], specials)
                for m in msgs.values()]
        lines.append((not any(r['contam'] or r['special'] for r in recs),
                      'no selected user turn shares a 13-gram with GSM8K '
                      'test or MATH-500; no special-token strings'))
        q = {e[0]: e[1] for e in split['train'] + split['heldout']}
        lines.append((all(r['question_id'] == q[r['doc_id']] for r in recs),
                      'question ids recomputed from the texts'))
        enc, same, pairs = chatml_ids(d2l, qwen), True, {}
        for k, part in (('traces', 'train'), ('val_domain', 'heldout')):
            s, entries = shards[k], split[part]
            pairs[part] = []
            for i in range(0, len(entries), 256):
                batch = entry_records(entries[i:i + 256], msgs)
                for j, want in enumerate(enc(batch)):
                    pairs[part].append((batch[j], want))
                    if i + j < s.n_docs:
                        same &= np.array_equal(s.doc(i + j)[:-1], want)
            same &= s.n_docs <= len(entries)
        lines.append((same, 'shard tokens equal the re-encoded ChatML'))
        ending = qwen_ending(qwen, split['targets']['near_cap'])
        n_val = shards['val_domain'].n_docs
        got = {p: endings_count(x, ending) for p, x in (
            ('train', pairs['train']), ('heldout', pairs['heldout']),
            ('val_domain', pairs['heldout'][:n_val]))}
        lines.append((all(got[p] == split['counts'][p]['endings']
                          for p in got), 'endings recounted from the texts'))
        if 'lm-cpt-qwen-manifest-v1' in d2l.DATA_HUB:
            with d2l._quiet():
                v1 = d2l._read_result(d2l.download_keyed(
                    'lm-cpt-qwen-manifest-v1'), 'json-table')
            for k in ('replay', 'val_general'):
                lines.append((d2l._layout_hash(shards[k]) == v1.layout[k],
                              f'{k}: bytes equal to cpt-qwen-v1'))
    else:
        with d2l._quiet():
            hosted = d2l._layout_hash(d2l.load_artifact(
                'cpt-fwedu-replay-12m'))
        lines.append((d2l._layout_hash(shards['replay']) == hosted,
                      f'replay: bytes equal to the hosted {REPLAY_T1}'))
    for good, text in lines:
        print(('ok    ' if good else 'FAIL  ') + text)
    ok = all(good for good, _ in lines)
    print('check passed' if ok else 'CHECK FAILED')
    return ok


STAGES = {  # stage: (build, check)
    'cpt-qwen-traces': (build_traces_qwen, lambda: check_traces(
        'cpt-qwen-traces')),
    'cpt-t1-traces': (build_traces_t1, lambda: check_traces('cpt-t1-traces')),
    'domains': (build_domains, check_domains),
    'lc-long': (lambda: build_lc('lc-long'), lambda: check_lc('lc-long')),
    'pg19-test': (lambda: build_lc('pg19-test'),
                  lambda: check_lc('pg19-test'))}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--stage', choices=sorted(STAGES), required=True)
    p.add_argument('--check', action='store_true',
                   help='verify the built shards (after building, or alone '
                        'with --check-only)')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    build, check = STAGES[a.stage]
    if not a.check_only:
        build()
    if a.check or a.check_only:
        sys.exit(0 if check() else 1)
