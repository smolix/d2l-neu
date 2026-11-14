"""Build the book corpus artifact `lm-book-r3-v1` (chapter 19, "the book corpus").

    python tools/build_r3.py --commit <sha> --out <dir> [--zip] [--check <zip>]
    python tools/build_r3.py --root <dir> --worktree --out <dir>   # files on disk

Run under .venv-lm with PYTHONPATH=<repo> after `make lib`: the cleaning,
the chunkers and the corpus assembly are library code (`d2l.clean_d2l_markdown`,
`d2l.chunk_sections`, `d2l.chunk_fixed`, `d2l._r3_build`, in
chapter_lm-encoders/retrieval-lib.md), so the book and this builder share one
source of truth. Sources are read with `git show <sha>:<path>` (no checkout),
so any tree can rebuild any commit. Tokens are counted with Qwen/Qwen3-0.6B at
its pin in d2l.HF_REVISIONS.

Deterministic: sorted paths, JSON with sorted keys, gzip with mtime 0, zip
entries dated 1980-01-01 and stored. The manifest records the sha256 of every
uncompressed member, and `--check <zip>` rebuilds and compares them (and the
zip bytes) with a released zip.
"""
import argparse
import gzip
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import zipfile

MEMBERS = ('docs.jsonl.gz', 'chunks_sec512.jsonl.gz',
           'chunks_fix256o0.jsonl.gz', 'chunks_fix256o64.jsonl.gz')
ZIP_NAME = 'lm-book-r3-v1.zip'


def dumps(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'))


def jsonl(records):
    return ''.join(dumps(r) + '\n' for r in records).encode('utf-8')


def builder_code_sha256(d2l):
    """sha256 of the library code the corpus depends on."""
    names = ['clean_d2l_markdown', '_r3_label_titles', '_r3_rewrite_refs',
             '_r3_caption_head', '_r3_sentences', '_r3_parse_blocks',
             '_r3_ntokens', '_r3_split_block', '_r3_chunk_record',
             '_r3_heading_path', 'chunk_sections', 'chunk_fixed',
             '_r3_quarto_files', '_r3_split_docs', '_r3_build']
    src = ''.join(inspect.getsource(getattr(d2l, n)) for n in names)
    src += repr(d2l._R3_EXCLUDED) + repr(d2l._R3_DIRECTIVES)
    return hashlib.sha256(src.encode()).hexdigest()


def build(root, commit, worktree=False):
    """(manifest, {member: uncompressed bytes})."""
    from d2l import torch as d2l
    import transformers
    if worktree:
        def read(p):
            with open(os.path.join(root, p), encoding='utf-8') as f:
                return f.read()
        commit = 'worktree'
    else:
        commit = subprocess.run(['git', '-C', root, 'rev-parse', commit],
                                capture_output=True, text=True,
                                check=True).stdout.strip()
        read = lambda p: subprocess.run(
            ['git', '-C', root, 'show', f'{commit}:{p}'],
            capture_output=True, check=True).stdout.decode('utf-8')
    rev = d2l.HF_REVISIONS[d2l.R3_TOKENIZER]
    tok = transformers.AutoTokenizer.from_pretrained(d2l.R3_TOKENIZER,
                                                     revision=rev)
    manifest, docs, chunkings = d2l._r3_build(read, commit, tok, rev)
    members = {'docs.jsonl.gz': jsonl(docs)}
    for tag, chunks in chunkings.items():
        members[f'chunks_{tag}.jsonl.gz'] = jsonl(chunks)
    manifest['builder_code_sha256'] = builder_code_sha256(d2l)
    manifest['members'] = {name: hashlib.sha256(data).hexdigest()
                           for name, data in sorted(members.items())}
    return manifest, members


def zip_bytes(manifest, members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_STORED) as z:
        files = [('manifest.json', (json.dumps(
            manifest, sort_keys=True, ensure_ascii=False, indent=1)
            + '\n').encode('utf-8'))]
        files += [(name, gzip.compress(members[name], 9, mtime=0))
                  for name in MEMBERS]
        for name, data in files:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr = 3, 0o644 << 16
            z.writestr(info, data)
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--commit', default='origin/main')
    ap.add_argument('--worktree', action='store_true',
                    help='read the files under --root, not a commit')
    ap.add_argument('--out', required=True)
    ap.add_argument('--zip', action='store_true')
    ap.add_argument('--check', metavar='ZIP')
    args = ap.parse_args(argv)
    if args.check:
        with zipfile.ZipFile(args.check) as z:
            released = json.loads(z.read('manifest.json'))
        commit = released['book_commit']
        manifest, members = build(args.root, commit,
                                  args.worktree or commit == 'worktree')
        bad = [m for m in MEMBERS if manifest['members'][m]
               != released['members'][m]]
        same_manifest = manifest == released
        with open(args.check, 'rb') as f:
            same_bytes = zip_bytes(manifest, members) == f.read()
        for m in MEMBERS:
            print(f'{m}: {"same" if m not in bad else "DIFFERENT"} '
                  f'({manifest["members"][m][:12]})')
        print(f'manifest {"same" if same_manifest else "DIFFERENT"}; '
              f'zip bytes {"same" if same_bytes else "DIFFERENT"}')
        return 0 if not bad and same_manifest else 1
    manifest, members = build(args.root, args.commit, args.worktree)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, sort_keys=True, ensure_ascii=False, indent=1)
        f.write('\n')
    for name in MEMBERS:
        with open(os.path.join(args.out, name), 'wb') as f:
            f.write(gzip.compress(members[name], 9, mtime=0))
    if args.zip:
        path = os.path.join(args.out, ZIP_NAME)
        data = zip_bytes(manifest, members)
        with open(path, 'wb') as f:
            f.write(data)
        print(f'wrote {path}: {len(data):,} bytes, '
              f'sha1 {hashlib.sha1(data).hexdigest()}')
    c = manifest['counts']
    print(f'commit {manifest["book_commit"]}: {c["docs"]} documents '
          f'({c["docs_train"]} train, {c["docs_dev"]} dev, '
          f'{c["docs_test"]} test), {c["tokens"]:,} tokens; chunks: '
          + ', '.join(f'{t} {c["chunks_" + t]:,}'
                      for t in ('sec512', 'fix256o0', 'fix256o64')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
