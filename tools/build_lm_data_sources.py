#!/usr/bin/env python3
"""Build the experiment slices of chapter 20 from pinned Hugging Face
revisions (chapter 20 specification, part 1.4, "Experiment slices").

Sources built here (one subcommand each; more join as their sections are
finished):

  fineweb-ref lm-data-fineweb-ref-v1.parquet, 20.1's reference text: the
              FineWeb rows (CC-MAIN-2025-05, data/CC-MAIN-2025-05/
              000_00000.parquet at the pinned revision) whose file_path is
              the pinned WARC file and whose id is a response record in the
              pinned 256 MiB range. Columns id, url, date, text,
              language_score, token_count; sorted by id; zstd parquet. The
              build checks that the file's rows are complete: they are
              contiguous, followed by another file's rows in the same shard,
              and absent from the next shard.

  fw2-slice   lm-data-fw2-slice-v1.tar, the ten-language tokenizer slice of
              20.6: slice.parquet, hindi.parquet, meta.json, README.md.

lm-data-fw2-slice-v1
  slice.parquet  lang, part ('train' | 'heldout'), doc_id, text (NFC),
                 src_row. Nine FineWeb-2 configs (deu_Latn, rus_Cyrl,
                 arb_Arab, hin_Deva, cmn_Hani, jpn_Jpan, tha_Thai, swh_Latn,
                 kor_Hang): the rows of data/<lang>/train/000_00000.parquet
                 in order until 5,000,000 UTF-8 bytes of NFC text (the
                 document that crosses the budget is included); English:
                 the rows of FineWeb-Edu sample/10BT/000_00000.parquet in
                 buckets 1-7 (d2l.fwedu_bucket), in order, same budget.
                 'heldout' = every document whose index i within its
                 language has i % 10 == 9 (10% by index).
  hindi.parquet  part ('ext_train' | 'ext_heldout'), doc_id, text, src_row:
                 the next 30,500 non-empty hin_Deva documents after the
                 slice's; the first 30,000 are ext_train, the last 500
                 ext_heldout.
  meta.json      sources (repository, revision, files, licence), the
                 budget, fw2_docs (documents per FineWeb-2 config, from the
                 dataset card at the pinned revision; 20.6's temperature
                 weights), the English size rule (English is given
                 deu_Latn's count), per-language counts and bytes.
  Empty NFC documents are skipped and counted. Parquet with zstd level 9;
  a deterministic USTAR archive (mtime 0).

    PYTHONPATH=$PWD HF_HOME=... .venv-lm/bin/python \\
        tools/build_lm_data_sources.py fw2-slice --out data/lm/build-sources-v1
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from d2l import torch as d2l  # noqa: E402

BUDGET = 5_000_000
FW2_LANGS = ('deu_Latn', 'rus_Cyrl', 'arb_Arab', 'hin_Deva', 'cmn_Hani',
             'jpn_Jpan', 'tha_Thai', 'swh_Latn', 'kor_Hang')
HINDI_TRAIN, HINDI_HELD = 30_000, 500
README = """# lm-data-fw2-slice-v1

The ten-language tokenizer slice of *Dive into Deep Learning*, section
"Tokenizers at Scale" (chapter 20.6), and its Hindi extension files.

* FineWeb-2 (`HuggingFaceFW/fineweb-2` at revision `{rev2}`): the first
  documents of `data/<config>/train/000_00000.parquet` for nine configs.
* FineWeb-Edu (`HuggingFaceFW/fineweb-edu` at revision `{revE}`): English
  documents of `sample/10BT/000_00000.parquet` in buckets 1-7.

Both datasets are released by Hugging Face under the Open Data Commons
Attribution License (ODC-By) v1.0; their use is also subject to Common
Crawl's Terms of Use. The text is copied unchanged apart from Unicode NFC
normalization. Built by `tools/build_lm_data_sources.py fw2-slice`; the rules
are in `meta.json`.
"""


def card_documents(rev2):
    """Documents per config from the FineWeb-2 dataset card (README.md) at
    the pinned revision: the table row of each config's `<config>` cell."""
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download('HuggingFaceFW/fineweb-2', 'README.md',
                              revision=rev2, repo_type='dataset'),
              encoding='utf-8') as f:
        card = f.read()
    out = {}
    for lang in FW2_LANGS:
        m = re.search(r'\| `' + lang + r'` \| ([\d,]+) \| ([\d,]+) \|', card)
        if not m:
            raise ValueError(f'{lang}: no row in the dataset card')
        out[lang] = int(m.group(2).replace(',', ''))
    return out


def rows(pf, keep=None):
    """(row, id, text) of a parquet file in order; keep(id) filters."""
    r = 0
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=['id', 'text'])
        for i, x in zip(t.column('id').to_pylist(),
                        t.column('text').to_pylist()):
            if keep is None or keep(i):
                yield r, i, x
            r += 1


def take(lang, it, hindi=None):
    """The slice documents of one language, then (for Hindi) the extension
    documents."""
    n, b, empty, out = 0, 0, 0, []
    for r, i, x in it:
        x = unicodedata.normalize('NFC', x or '')
        if not x:
            empty += 1
            continue
        if b < BUDGET:
            out.append(dict(lang=lang, part='heldout' if n % 10 == 9
                            else 'train', doc_id=i, text=x, src_row=r))
            n, b = n + 1, b + len(x.encode('utf-8'))
            continue
        if hindi is None:
            break
        hindi.append(dict(part='ext_train' if len(hindi) < HINDI_TRAIN
                          else 'ext_heldout', doc_id=i, text=x, src_row=r))
        if len(hindi) == HINDI_TRAIN + HINDI_HELD:
            break
    if hindi is not None and len(hindi) < HINDI_TRAIN + HINDI_HELD:
        raise ValueError('the Hindi file holds too few documents')
    return out, dict(n_docs=n, bytes=b, empty_skipped=empty)


def parquet_bytes(records):
    import pyarrow as pa
    import pyarrow.parquet as pq
    b = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(records), b, compression='zstd',
                   compression_level=9)
    return b.getvalue()


def fw2_slice(out_dir):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download
    rev2 = d2l.LM_DATA_SOURCES['fineweb-2'][2]
    revE = d2l.LM_DATA_SOURCES['fineweb-edu'][2]
    fw2_docs = card_documents(rev2)
    fileE = 'sample/10BT/000_00000.parquet'
    pfE = pq.ParquetFile(hf_hub_download('HuggingFaceFW/fineweb-edu', fileE,
                                         revision=revE, repo_type='dataset'))
    recs, hindi, stats = [], [], {}
    out, stats['eng_Latn'] = take('eng_Latn', rows(
        pfE, keep=lambda i: 1 <= d2l.fwedu_bucket(i) <= 7))
    recs += out
    print('eng_Latn', stats['eng_Latn'], flush=True)
    fs = HfFileSystem()
    for lang in FW2_LANGS:
        path = (f'datasets/HuggingFaceFW/fineweb-2@{rev2}/data/{lang}/train/'
                '000_00000.parquet')
        with fs.open(path, 'rb', block_size=8 << 20) as f:
            out, stats[lang] = take(lang, rows(pq.ParquetFile(f)),
                                    hindi if lang == 'hin_Deva' else None)
        recs += out
        print(lang, stats[lang], flush=True)
    stats['hindi_extension'] = dict(
        n_docs=len(hindi), bytes=sum(len(h['text'].encode()) for h in hindi))
    meta = dict(
        schema='d2l-fw2-slice/1', budget_bytes=BUDGET,
        heldout_rule='document index % 10 == 9',
        sources={'fineweb-2': dict(
            repo='HuggingFaceFW/fineweb-2', revision=rev2, license='odc-by',
            files={l: f'data/{l}/train/000_00000.parquet'
                   for l in FW2_LANGS}),
            'fineweb-edu': dict(
                repo='HuggingFaceFW/fineweb-edu', revision=revE,
                license='odc-by', files={'eng_Latn': fileE}, buckets=[1, 7])},
        fw2_docs=fw2_docs,
        fw2_docs_source='FineWeb-2 dataset card (README.md at the pinned '
                        'revision), documents column',
        english_size_rule="eng_Latn is given deu_Latn's count",
        langs=['eng_Latn', *FW2_LANGS], stats=stats,
        hindi=dict(n_train=HINDI_TRAIN, n_heldout=HINDI_HELD,
                   rule='the next 30,500 non-empty hin_Deva documents after '
                        'the slice'))
    files = {'slice.parquet': parquet_bytes(recs),
             'hindi.parquet': parquet_bytes(hindi),
             'meta.json': json.dumps(meta, indent=1, sort_keys=True,
                                     ensure_ascii=False).encode('utf-8'),
             'README.md': README.format(rev2=rev2, revE=revE).encode('utf-8')}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'lm-data-fw2-slice-v1.tar')
    with tarfile.open(path + '.part', 'w', format=tarfile.USTAR_FORMAT) as tf:
        for name in sorted(files):
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(files[name]), 0, 0o644
            ti.uname = ti.gname = ''
            tf.addfile(ti, io.BytesIO(files[name]))
    os.replace(path + '.part', path)
    print('wrote', path, os.path.getsize(path), flush=True)


FW_FILE = 'data/CC-MAIN-2025-05/000_00000.parquet'
FW_COLS = ('id', 'url', 'date', 'text', 'language_score', 'token_count')


def fineweb_ref(out_dir):
    """lm-data-fineweb-ref-v1.parquet (see the module docstring)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download
    rev = d2l.LM_DATA_SOURCES['fineweb'][2]
    out_dir = os.path.abspath(out_dir)
    # The library's data paths are ../data/lm/..., relative to a notebook's
    # folder; from tools/ they name this checkout's data/lm.
    os.chdir(os.path.join(ROOT, 'tools'))
    with open(d2l.download_keyed('lm-cc-warc-256m-v1')) as f:
        pin = json.load(f)
    want = pin['fineweb_file_path']
    warc = d2l.fetch_warc_range()
    resp = {r.id: r for r in d2l.iter_warc(warc)}
    print('responses in the range', len(resp), flush=True)
    pf = pq.ParquetFile(hf_hub_download('HuggingFaceFW/fineweb', FW_FILE,
                                        revision=rev, repo_type='dataset'))
    rows, pos, first, last, after = [], 0, None, None, None
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=[*FW_COLS, 'file_path']).to_pylist()
        for i, r in enumerate(t):
            if r['file_path'] == want:
                if last is not None and last != pos + i - 1:
                    raise AssertionError('the file\'s rows are not contiguous')
                first = pos + i if first is None else first
                last = pos + i
                if r['id'] in resp:
                    rows.append(r)
            elif last is not None and after is None:
                after = (pos + i, r['file_path'])
        pos += len(t)
        if after is not None:
            break
    if first is None or after is None:
        raise AssertionError('the file\'s rows do not end inside the shard')
    print(f'{want}: rows {first}-{last} of {FW_FILE}, followed by '
          f'{after[1]} at row {after[0]}', flush=True)
    fs = HfFileSystem()
    nxt = FW_FILE.replace('000_00000', '000_00001')
    with fs.open(f'datasets/HuggingFaceFW/fineweb@{rev}/{nxt}', 'rb',
                 block_size=8 << 20) as f:
        paths = set(pq.ParquetFile(f).read(columns=['file_path'])
                    .column('file_path').to_pylist())
    if want in paths:
        raise AssertionError(f'{nxt} also holds rows of {want}')
    print(f'{nxt}: no row of the file ({len(paths)} other files)',
          flush=True)
    bad = [(r['id'], k) for r in rows for k in ('url', 'date')
           if r[k] != getattr(resp[r['id']], k)]
    if bad:
        raise AssertionError(f'url/date differ from the WARC: {bad[:5]}')
    rows.sort(key=lambda r: r['id'])
    schema = pa.schema([('id', pa.string()), ('url', pa.string()),
                        ('date', pa.string()), ('text', pa.string()),
                        ('language_score', pa.float64()),
                        ('token_count', pa.int64())])
    table = pa.Table.from_pylist([{k: r[k] for k in FW_COLS} for r in rows],
                                 schema=schema)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'lm-data-fineweb-ref-v1.parquet')
    pq.write_table(table, path + '.part', compression='zstd',
                   compression_level=3, row_group_size=65536,
                   use_dictionary=False, write_statistics=False)
    os.replace(path + '.part', path)
    print(json.dumps(dict(path=path, rows=len(rows), file_rows=last - first
                          + 1, bytes=os.path.getsize(path),
                          sha1=hashlib.sha1(open(path, 'rb').read())
                          .hexdigest())), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('source', choices=['fw2-slice', 'fineweb-ref'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'lm',
                                                  'build-sources-v1'))
    a = ap.parse_args(argv)
    if a.source == 'fw2-slice':
        fw2_slice(a.out)
    elif a.source == 'fineweb-ref':
        fineweb_ref(a.out)


if __name__ == '__main__':
    main()
