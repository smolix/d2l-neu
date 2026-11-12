#!/usr/bin/env python3
"""Upload data artifacts to the d2l-data R2 bucket (public DATA_HUB files).

Registered artifacts (`d2l.DATA_HUB['lm-...-v1']`, rule D1 of the Language
Models part: every key and file name carries its version) are served from the
Cloudflare R2 bucket `d2l-data` at https://d2l-data.smola.org/<key>. This
script puts local files there under a prefix and prints, for each, the public
URL, the size and the sha1 that the registering `DATA_HUB` entry pins.

    make upload-data FILES="data/lm/rerun/lm-x-v1.json" PREFIX=lm/
    python3 tools/upload_data_r2.py --prefix lm/ FILE... [--dry-run] [--force]
    python3 tools/upload_data_r2.py --delete lm/_test/x.txt

Rules:
  * The object key is PREFIX + the file's base name; the name must already
    carry its version (`-v1`). An existing key is never overwritten: a key
    with the same bytes (same MD5 ETag) is reported as already uploaded, a key
    with other bytes is refused unless --force. Publish a new version instead.
  * --dry-run reads no file into the bucket: it prints the plan (and, when the
    credentials file is present, whether each key exists).
  * --delete removes keys. Keys outside a `_test/` folder also need --force:
    a published key may be pinned by a DATA_HUB entry somewhere.

Credentials: the file ~/.config/d2l/r2-data.env (override with
D2L_R2_DATA_ENV) holds CF_API_TOKEN, a Cloudflare API token scoped to the
bucket. The S3 credentials are derived from it as Cloudflare documents for R2:
access key id = the token's id (GET /user/tokens/verify), secret access key =
sha256 hex of the token. The token and the secret are passed to `aws` through
its environment only; they are never printed, logged or written anywhere.
Needs the AWS CLI (`aws`, also used by tools/upload_r2.sh).
"""

import argparse
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

BUCKET = 'd2l-data'
ENDPOINT = 'https://1712830f746ac9278e7db6de736dc430.r2.cloudflarestorage.com'
PUBLIC_URL = 'https://d2l-data.smola.org/'
ENV_FILE = Path(os.environ.get('D2L_R2_DATA_ENV',
                               '~/.config/d2l/r2-data.env')).expanduser()
VERIFY_URL = 'https://api.cloudflare.com/client/v4/user/tokens/verify'

# Types mimetypes does not know (or guesses differently across hosts).
CONTENT_TYPES = {
    '.json': 'application/json',
    '.jsonl': 'application/x-ndjson',
    '.csv': 'text/csv; charset=utf-8',
    '.tsv': 'text/tab-separated-values; charset=utf-8',
    '.txt': 'text/plain; charset=utf-8',
    '.md': 'text/markdown; charset=utf-8',
    '.parquet': 'application/vnd.apache.parquet',
    '.safetensors': 'application/octet-stream',
    '.pt': 'application/octet-stream',
    '.npz': 'application/octet-stream',
    '.zip': 'application/zip',
    '.gz': 'application/gzip',
    '.tar': 'application/x-tar',
}

KEY_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._/-]*')


class UploadError(Exception):
    pass


def content_type(path):
    ext = Path(path).suffix.lower()
    if ext in CONTENT_TYPES:
        return CONTENT_TYPES[ext]
    guess, _ = mimetypes.guess_type(str(path))
    return guess or 'application/octet-stream'


def digests(path):
    """(sha1 hex, md5 hex, size) of a file, read in 1 MiB chunks."""
    sha1, md5, n = hashlib.sha1(), hashlib.md5(), 0
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            sha1.update(chunk)
            md5.update(chunk)
            n += len(chunk)
    return sha1.hexdigest(), md5.hexdigest(), n


def object_key(prefix, path):
    key = prefix + Path(path).name
    if not KEY_RE.fullmatch(key) or '//' in key or '/../' in f'/{key}/':
        raise UploadError(f'unsafe object key {key!r}')
    return key


def normalise_prefix(prefix):
    prefix = prefix.strip()
    if prefix.startswith('/'):
        raise UploadError('PREFIX must be relative (lm/, not /lm/)')
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    return prefix


def read_token(env_file=ENV_FILE):
    """CF_API_TOKEN from the credentials file (KEY=VALUE lines)."""
    if not env_file.exists():
        raise UploadError(f'credentials file {env_file} not found')
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[len('export '):].strip()
        name, sep, value = line.partition('=')
        if sep and name.strip() == 'CF_API_TOKEN':
            value = value.strip().strip('"').strip("'")
            if value:
                return value
    raise UploadError(f'no CF_API_TOKEN in {env_file}')


def derive_s3_credentials(token):
    """(access key id, secret access key) of an R2 API token."""
    req = urllib.request.Request(
        VERIFY_URL, headers={'Authorization': f'Bearer {token}',
                             'User-Agent': 'd2l-upload-data'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        # The message names the status only; the request (and its token)
        # is never printed.
        raise UploadError(f'token verification failed: HTTP {e.code}') from None
    except urllib.error.URLError as e:
        raise UploadError(f'token verification failed: {e.reason}') from None
    result = body.get('result') or {}
    if not body.get('success') or not result.get('id'):
        raise UploadError('token verification returned no token id')
    if result.get('status') not in (None, 'active'):
        raise UploadError(f"token status is {result.get('status')!r}")
    return result['id'], hashlib.sha256(token.encode()).hexdigest()


class Bucket:
    """The few S3 calls we need, through the AWS CLI with the derived
    credentials in its environment (never on its command line)."""

    def __init__(self, access_key, secret_key):
        if not shutil.which('aws'):
            raise UploadError('aws CLI not found')
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith('AWS_')}
        self.env.update({
            'AWS_ACCESS_KEY_ID': access_key,
            'AWS_SECRET_ACCESS_KEY': secret_key,
            'AWS_DEFAULT_REGION': 'auto',
            'AWS_EC2_METADATA_DISABLED': 'true',
            # Do not let a profile in ~/.aws pick other credentials.
            'AWS_CONFIG_FILE': os.devnull,
            'AWS_SHARED_CREDENTIALS_FILE': os.devnull,
        })

    def _run(self, *args):
        cmd = ['aws', 's3api', *args, '--bucket', BUCKET,
               '--endpoint-url', ENDPOINT, '--region', 'auto',
               '--output', 'json']
        return subprocess.run(cmd, env=self.env, capture_output=True,
                              text=True, timeout=600)

    def head(self, key):
        """Object metadata, or None if the key does not exist."""
        r = self._run('head-object', '--key', key)
        if r.returncode == 0:
            return json.loads(r.stdout)
        if 'Not Found' in r.stderr or '404' in r.stderr:
            return None
        raise UploadError(f'head-object {key}: {r.stderr.strip()[-300:]}')

    def put(self, key, path, ctype, sha1):
        r = self._run('put-object', '--key', key, '--body', str(path),
                      '--content-type', ctype, '--metadata', f'sha1={sha1}')
        if r.returncode != 0:
            raise UploadError(f'put-object {key}: {r.stderr.strip()[-300:]}')

    def delete(self, key):
        r = self._run('delete-object', '--key', key)
        if r.returncode != 0:
            raise UploadError(f'delete-object {key}: {r.stderr.strip()[-300:]}')


def connect():
    return Bucket(*derive_s3_credentials(read_token()))


def etag_md5(meta):
    return (meta.get('ETag') or '').strip('"')


def upload(files, prefix, dry_run=False, force=False):
    plan = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            raise UploadError(f'{f}: not a file')
        sha1, md5, size = digests(p)
        plan.append((p, object_key(prefix, p), content_type(p), sha1, md5,
                     size))
    keys = [k for _, k, *_ in plan]
    if len(set(keys)) != len(keys):
        raise UploadError('two files map to the same key')

    bucket = None
    try:
        bucket = connect()
    except UploadError as e:
        if not dry_run:
            raise
        print(f'note: {e}; existence of the keys not checked')

    failed = 0
    for p, key, ctype, sha1, md5, size in plan:
        url = PUBLIC_URL + key
        meta = bucket.head(key) if bucket else None
        if meta is not None and etag_md5(meta) == md5:
            state = 'already uploaded (identical bytes)'
        elif meta is not None and not force:
            print(f'REFUSED {key}: exists with other bytes (size '
                  f"{meta.get('ContentLength')}); publish a new version "
                  f'(-v<N+1>) or pass --force', file=sys.stderr)
            failed += 1
            continue
        elif dry_run:
            state = ('would overwrite (--force)' if meta is not None
                     else 'would upload' if bucket else 'would upload (unchecked)')
        else:
            bucket.put(key, p, ctype, sha1)
            after = bucket.head(key)
            if (after is None or etag_md5(after) != md5
                    or int(after.get('ContentLength', -1)) != size):
                raise UploadError(f'{key}: verification after upload failed')
            state = 'overwritten (--force)' if meta is not None else 'uploaded'
        print(f'{state}: {p}\n'
              f'  key      {key}\n'
              f'  type     {ctype}\n'
              f'  size     {size} bytes\n'
              f'  sha1     {sha1}\n'
              f'  url      {url}\n'
              f"  DATA_HUB d2l.DATA_HUB['{Path(key).stem}'] = (\n"
              f"               '{url}',\n"
              f"               '{sha1}')")
    return failed


def delete(keys, dry_run=False, force=False):
    bucket = connect()
    failed = 0
    for key in keys:
        if not KEY_RE.fullmatch(key):
            raise UploadError(f'unsafe object key {key!r}')
        if '/_test/' not in f'/{key}' and not force:
            print(f'REFUSED {key}: not under a _test/ folder; a published '
                  f'key may be pinned by a DATA_HUB entry (pass --force)',
                  file=sys.stderr)
            failed += 1
            continue
        if bucket.head(key) is None:
            print(f'absent: {key}')
            continue
        if dry_run:
            print(f'would delete: {key}')
            continue
        bucket.delete(key)
        if bucket.head(key) is not None:
            raise UploadError(f'{key}: still present after delete')
        print(f'deleted: {key}')
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Upload versioned data artifacts to the d2l-data R2 '
                    'bucket and print their public URL and sha1.')
    ap.add_argument('files', nargs='*', help='local files to upload')
    ap.add_argument('--prefix', default='lm/',
                    help='key prefix (folder) in the bucket (default: lm/)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan, change nothing')
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing key with other bytes; delete '
                         'outside _test/')
    ap.add_argument('--delete', nargs='+', metavar='KEY',
                    help='delete these object keys instead of uploading')
    args = ap.parse_args(argv)
    try:
        if args.delete:
            if args.files:
                ap.error('--delete takes keys, not files')
            failed = delete(args.delete, args.dry_run, args.force)
        else:
            if not args.files:
                ap.error('no files given (make upload-data FILES=...)')
            failed = upload(args.files, normalise_prefix(args.prefix),
                            args.dry_run, args.force)
    except UploadError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
