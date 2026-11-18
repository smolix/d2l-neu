#!/usr/bin/env python3
"""Make lm-heldout-texts-v1.json, the held-out texts of sec_lm_scoring.

Bits per byte on wikitext-2 partly measures memorization: the text is 2016
Wikipedia and in every model's pretraining data. Cell `scoring-bpb-models`
therefore also scores two texts that neither Qwen3 (2025-04-29) nor Qwen3.5
(2026-02-28) can have seen, because the book wrote them later:

* prose: the running text of chapter_computational-performance/
  performance-model.md and compilation.md, both added on 2026-07-20 in the
  chapter 13 rebuild (new text: no line of 40 characters or more is shared with
  the first-edition files that rebuild deleted), joined by a blank line;
  tools/lm/book_text.prose strips code, comments, images, directives and the
  slide section; CC BY-SA 4.0;
* code: tools/gen_mdl_diffusion_figures.py, added on 2026-09-26, verbatim; the
  book's sample-code licence (MIT-0).

Both are read at COMMIT (a commit of main) with `git show`, so the output is
byte-identical on every rerun. The script asserts that each file was first
added after the later model release.

    python3 tools/oneshot/make_lm_heldout_texts.py [--out PATH]
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/lm'))
import book_text  # noqa: E402

COMMIT = 'b4d28000a49956d76ef1a8c2f5a0d7a135609efc'  # main, 2026-11-12
CUTOFF = '2026-02-28'  # Qwen3.5 release, the later of the two families
PROSE = ['chapter_computational-performance/performance-model.md',
         'chapter_computational-performance/compilation.md']
CODE = 'tools/gen_mdl_diffusion_figures.py'


def added(path):
    """Commit and date at which path was first added (following renames)."""
    out = subprocess.run(
        ['git', '-C', str(ROOT), 'log', '--diff-filter=A', '--follow',
         '--format=%H %cI', COMMIT, '--', path],
        check=True, capture_output=True, text=True).stdout.split()
    sha, date = out[-2], out[-1]
    utc = datetime.datetime.fromisoformat(date).astimezone(
        datetime.timezone.utc).date().isoformat()
    assert utc > CUTOFF, (path, utc)
    return sha, utc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=str(
        ROOT / 'data/lm/authoring/lm-heldout-texts-v1.json'))
    args = ap.parse_args()
    commit, date = book_text.git_commit(ROOT, COMMIT)
    prose = '\n'.join(book_text.prose(book_text.git_show(ROOT, commit, p))
                      for p in PROSE)
    code = book_text.git_show(ROOT, commit, CODE)
    texts = [
        {'name': 'prose', 'source': PROSE, 'commit': commit,
         'added': {p: added(p)[1] for p in PROSE},
         'licence': book_text.PROSE_LICENCE,
         'processing': 'tools/lm/book_text.prose of each file, joined by a '
                       'blank line', 'text': prose},
        {'name': 'code', 'source': [CODE], 'commit': commit,
         'added': {CODE: added(CODE)[1]}, 'licence': book_text.CODE_LICENCE,
         'processing': 'verbatim', 'text': code},
    ]
    for t in texts:
        t['n_chars'] = len(t['text'])
        t['n_bytes'] = len(t['text'].encode('utf-8'))
    doc = {'name': 'lm-heldout-texts-v1',
           'producer': 'tools/oneshot/make_lm_heldout_texts.py',
           'commit': commit, 'commit_date': date,
           'note': f'every source file was first added after {CUTOFF}, the '
                   'Qwen3.5 release; Qwen3 was released on 2025-04-29',
           'texts': texts}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1, ensure_ascii=False) + '\n').encode()
    out.write_bytes(data)
    print(f'wrote {out} ({len(data)} bytes, sha1 '
          f'{hashlib.sha1(data).hexdigest()}); ' + ', '.join(
              f"{t['name']} {t['n_bytes']} bytes" for t in texts))


if __name__ == '__main__':
    main()
