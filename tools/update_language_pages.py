#!/usr/bin/env python3
"""Record which pages a language's site serves, in the shared languages.json.

The navbar's language menu (_d2l-languages.html) reads
https://d2l-data.smola.org/site/languages.json and links to the same page in
another language only when that language's `pages` list has it, otherwise to
that language's home page. No 404 page or error rule is involved (the zone's
plan has none).

tools/upload_r2.sh calls this after every real upload to a production
language bucket (d2l-<code>): it lists the HTML pages of the uploaded book
directory, replaces that language's `pages` in the live languages.json and
uploads it again. The language list itself (names, order, old editions) is
owned by d2l-localization (`l10n menu`); a language missing from it is not
added here, only reported.

Usage: update_language_pages.py <code> <book dir> [--dry-run]
Credentials: ~/.config/cloudflare/d2l-data.env (override D2L_DATA_ENV);
never printed.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = 'https://d2l-data.smola.org/site/languages.json'
KEY = 's3://d2l-data/site/languages.json'
# Directories of a build that are not book pages with the language menu.
SKIP_DIRS = {'slides', 'site_libs', 'notebooks', 'pdf', 'static', 'img', '_i18n'}


def book_pages(book: Path) -> list:
    pages = []
    for p in book.rglob('*.html'):
        rel = p.relative_to(book)
        if rel.parts[0] in SKIP_DIRS or rel.name == '404.html':
            continue
        pages.append(rel.as_posix())
    return sorted(pages)


def fetch() -> dict:
    req = urllib.request.Request(URL, headers={'User-Agent': 'd2l-language-pages',
                                               'Cache-Control': 'no-cache'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def upload(data: dict, dry_run: bool):
    env_file = Path(os.environ.get('D2L_DATA_ENV',
                                   '~/.config/cloudflare/d2l-data.env')).expanduser()
    if not env_file.exists():
        raise SystemExit(f'update_language_pages: {env_file} not found; menu pages not updated')
    env = dict(os.environ)
    for line in env_file.read_text().splitlines():
        k, sep, v = line.strip().partition('=')
        if sep and not k.startswith('#'):
            env[k] = v.strip().strip('"\'')
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
        tmp = f.name
    cmd = ['aws', 's3', 'cp', tmp, KEY, '--content-type', 'application/json; charset=utf-8',
           '--cache-control', 'public, max-age=300', '--region', 'auto',
           '--endpoint-url', env['AWS_ENDPOINT_URL_S3']]
    if dry_run:
        cmd.append('--dryrun')
    rc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL).returncode
    os.unlink(tmp)
    if rc:
        raise SystemExit(f'update_language_pages: upload failed (rc={rc})')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('code')
    ap.add_argument('book', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    pages = book_pages(args.book)
    if not pages:
        raise SystemExit(f'update_language_pages: no HTML pages under {args.book}')
    data = fetch()
    entry = next((l for l in data.get('languages', []) if l.get('code') == args.code), None)
    if entry is None:
        print(f'update_language_pages: {args.code!r} is not in languages.json '
              '(add it in d2l-localization languages.yml, then `l10n menu --upload`)',
              file=sys.stderr)
        return
    old = entry.get('pages') or []
    entry['pages'] = pages
    upload(data, args.dry_run)
    print(f'languages.json: {args.code} serves {len(pages)} pages '
          f'(+{len(set(pages) - set(old))} / -{len(set(old) - set(pages))})'
          + (' [dry run]' if args.dry_run else ''))


if __name__ == '__main__':
    main()
