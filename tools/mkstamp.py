#!/usr/bin/env python3
"""Content-hash stamp writer.

Touches a stamp file only when its tracked content actually changed,
preserving the previous mtime otherwise. Used at boundaries where Make's
default mtime model is misleading (recipes that run regardless of content,
batch generators that rewrite many files, venv syncs that touch site-
packages metadata even on a no-op resolve).

Usage:
    # Stamp depends on the content of one-or-more files. mkstamp.py reads
    # them all, hashes them, compares against a sidecar `.hash` written
    # last time. If unchanged, leaves the stamp's mtime alone. Otherwise
    # touches the stamp and rewrites the sidecar.
    python tools/mkstamp.py <stamp-path> <input-file> [<input-file> ...]

    # Stamp depends on the content of a single literal string (e.g. the
    # output of `uv pip list` or `nvidia-smi`):
    python tools/mkstamp.py --stdin-hash <stamp-path>
    # Reads bytes from stdin, hashes, behaves as above.

Exit code: 0 always (on real errors raises an exception).
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path


def _hash_files(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(b'\x00')
        h.update(str(p).encode('utf-8'))
        h.update(b'\x00')
        try:
            h.update(Path(p).read_bytes())
        except FileNotFoundError:
            h.update(b'<missing>')
    return h.hexdigest()


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _stamp_unchanged(stamp_path, new_digest):
    """Return True if a stored hash equals new_digest."""
    sidecar = stamp_path.with_suffix(stamp_path.suffix + '.hash')
    if not sidecar.exists():
        return False
    try:
        old = sidecar.read_text().strip()
    except Exception:
        return False
    return old == new_digest


def _write_stamp(stamp_path, new_digest):
    """Touch stamp and update sidecar."""
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = stamp_path.with_suffix(stamp_path.suffix + '.hash')
    stamp_path.touch()
    sidecar.write_text(new_digest + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stamp', type=Path,
                        help='Stamp file path (e.g. .venv-pt/.synced.stamp)')
    parser.add_argument('inputs', nargs='*', type=Path,
                        help='Files whose content determines the stamp')
    parser.add_argument('--stdin-hash', action='store_true',
                        help='Hash bytes read from stdin instead of files')
    args = parser.parse_args()

    if args.stdin_hash:
        new_digest = _hash_bytes(sys.stdin.buffer.read())
    else:
        if not args.inputs:
            print(f'{sys.argv[0]}: at least one input file required '
                  '(or --stdin-hash)', file=sys.stderr)
            sys.exit(2)
        new_digest = _hash_files(args.inputs)

    if _stamp_unchanged(args.stamp, new_digest):
        # No mtime bump.
        print(f'{args.stamp}: unchanged')
        return
    _write_stamp(args.stamp, new_digest)
    print(f'{args.stamp}: updated')


if __name__ == '__main__':
    main()
