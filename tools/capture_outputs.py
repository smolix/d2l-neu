#!/usr/bin/env python3
"""Capture executed notebook outputs into the committed `outputs/` store.

This is the "bless" step of the decoupled build (see docs/build-system.md). It
distills executed notebooks — which are scratch, gitignored, and expensive to
produce — into a small, committed, cell-ID-keyed store that site rendering reads
instead. Run it after `make run-notebooks-*`.

For each `_notebooks/<fw>/<chapter>/<stem>.ipynb` it writes:

  outputs/<fw>/<chapter>/<stem>.json      per-notebook MANIFEST (plain git)
  outputs/<fw>/<chapter>/<stem>/<id>-<n>.png|svg   image ASSETS (Git LFS)

The split: text outputs (stream / text/plain) live inline in the manifest;
image/binary outputs become asset files. See docs/build-system.md §2.2.

The manifest is a pure function of the executed notebook + repo state — no
wall-clock or HEAD-commit fields — so re-capturing an unchanged notebook produces
a byte-identical file (capture twice → empty `git diff`).

Usage:
    python tools/capture_outputs.py                       # all notebooks, all fw
    python tools/capture_outputs.py --frameworks pytorch  # one framework
    python tools/capture_outputs.py FILES=chapter_x/foo.md ...   # subset (Make passes this)
    python tools/capture_outputs.py chapter_x/foo.md      # subset (positional)
"""

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'mxnet']
FRAMEWORK_PKG = {
    'pytorch': 'torch',
    'tensorflow': 'tensorflow',
    'jax': 'jax',
    'mxnet': 'mxnet',
}
SCHEMA = 1


def _sha(b: bytes) -> str:
    return 'sha256:' + hashlib.sha256(b).hexdigest()


def normalize_source(src) -> str:
    """Canonical text of a notebook code cell, for the code fingerprint.

    The cell id lives in metadata.id, never in the source, so it doesn't enter
    the hash. We rstrip each line so trailing-whitespace churn is ignored.
    """
    text = ''.join(src) if isinstance(src, list) else (src or '')
    return '\n'.join(ln.rstrip() for ln in text.split('\n'))


def _join(val) -> str:
    return ''.join(val) if isinstance(val, list) else (val or '')


# ── framework version (notebook-level invalidation key) ──────────

_VERSION_CACHE = {}


def public_version(spec: str) -> str:
    """Strip the PEP 440 *local* version segment from a 'pkg==X.Y.Z+local'
    provenance string, leaving 'pkg==X.Y.Z'.

    The local segment (e.g. '+cu128', '+cpu') identifies the *platform wheel*,
    not the framework version: torch 2.11.0 is torch 2.11.0 whether it's the
    CUDA-12.8 Linux build or the arm64 macOS CPU/MPS build. Keying freshness on
    it wrongly invalidates the *entire* committed store the instant you audit
    from a different machine — defeating the whole point of committing
    pre-executed outputs (a portable baseline that lets you re-run only the
    CPU-capable notebooks you touched, anywhere, instead of a full multi-GPU
    re-run). So freshness keys on the public version only; a genuine
    version change (2.11.0 → 2.12.0) still invalidates. See docs/build-system.md §3.
    """
    name, sep, ver = spec.partition('==')
    if not sep:
        return spec
    return f'{name}=={ver.split("+", 1)[0]}'


def framework_version(repo_root: Path, fw: str) -> str:
    if fw in _VERSION_CACHE:
        return _VERSION_CACHE[fw]
    pkg = FRAMEWORK_PKG[fw]
    py = repo_root / f'.venv-{fw}' / 'bin' / 'python'
    ver = 'unknown'
    if py.exists():
        try:
            r = subprocess.run(
                [str(py), '-c', f'import {pkg}; print({pkg}.__version__)'],
                capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and r.stdout.strip():
                ver = public_version(f'{pkg}=={r.stdout.strip()}')
        except Exception:
            pass
    _VERSION_CACHE[fw] = ver
    return ver


# ── d2l lib fingerprint (from the notebook's .d file) ────────────

def d2l_lib_fingerprint(repo_root: Path, d_path: Path) -> str:
    """Hash exactly the d2l/_blocks/<fw>/*.py this notebook depends on.

    The .d file (emitted by tools/scan_d2l_usage.py) lists the per-symbol block
    files as Make prerequisites. A notebook that uses no d2l symbols hashes the
    empty set.
    """
    blocks = []
    if d_path.exists():
        for tok in d_path.read_text(encoding='utf-8').split():
            if tok.startswith('d2l/_blocks/'):
                blocks.append(tok)
    h = hashlib.sha256()
    for rel in sorted(set(blocks)):
        h.update(rel.encode())
        h.update(b'\0')
        p = repo_root / rel
        h.update(p.read_bytes() if p.exists() else b'<missing>')
        h.update(b'\0')
    return 'sha256:' + h.hexdigest()


# ── output extraction ────────────────────────────────────────────

def _cap_text(text: str, max_inline: int) -> str:
    """Safety valve for pathological text dumps. 0 = unlimited (the default,
    which preserves exact render parity; inject_outputs does display trimming)."""
    if not max_inline:
        return text
    raw = text.encode('utf-8')
    if len(raw) <= max_inline:
        return text
    lines = text.split('\n')
    half = max(1, len(lines) // 4)
    return '\n'.join(
        lines[:half] + [f'...[truncated {len(raw)} bytes]...'] + lines[-half:])


def extract_cell_outputs(cell, cellid, stem, asset_dir: Path, max_inline):
    """Return (list-of-manifest-output-entries, kind).

    Output-type precedence matches inject_outputs.format_cell_output:
    image/png > image/svg+xml > stream > text/plain. Anything else (text/html,
    application/*, error) is dropped — format_cell_output ignores it too.
    """
    entries = []
    n = 0
    has_text = has_asset = False
    for out in cell.get('outputs', []):
        otype = out.get('output_type', '')
        if otype == 'error':
            continue
        if otype == 'stream':
            entries.append({
                'type': 'stream',
                'name': out.get('name', 'stdout'),
                'text': _cap_text(_join(out.get('text')), max_inline),
            })
            has_text = True
            continue
        data = out.get('data', {})
        if 'image/png' in data:
            n += 1
            raw = base64.b64decode(_join(data['image/png']))
            fname = f'{cellid}-{n}.png'
            (asset_dir / fname).write_bytes(raw)
            entries.append({
                'type': otype, 'mime': 'image/png',
                'asset': f'{stem}/{fname}',
                'bytes': len(raw), 'sha256': _sha(raw),
            })
            has_asset = True
        elif 'image/svg+xml' in data:
            n += 1
            svg = _join(data['image/svg+xml'])
            fname = f'{cellid}-{n}.svg'
            (asset_dir / fname).write_text(svg, encoding='utf-8')
            raw = svg.encode('utf-8')
            entries.append({
                'type': otype, 'mime': 'image/svg+xml',
                'asset': f'{stem}/{fname}',
                'bytes': len(raw), 'sha256': _sha(raw),
            })
            has_asset = True
        elif 'text/plain' in data:
            entries.append({
                'type': otype,
                'text': _cap_text(_join(data['text/plain']), max_inline),
            })
            has_text = True
        # else: not renderable by inject_outputs → drop
    kind = 'mixed' if (has_text and has_asset) else ('asset' if has_asset else 'text')
    return entries, kind


# ── fingerprints / provenance (shared with audit_outputs.py) ─────
#
# These MUST be identical for capture and audit, or audit reports false
# staleness. Both import these helpers; never duplicate the logic.

def code_cells_of(nb):
    return [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']


def cell_id_of(cell):
    return cell.get('id') or cell.get('metadata', {}).get('id')


def prefix_fingerprints(code_cells):
    """{cell_id: sha256 of normalized source of cells 0..i} (prefix-inclusive).

    Untagged cells still advance the running hash (they build state) but get no
    entry, so editing them invalidates downstream tagged cells.
    """
    running = hashlib.sha256()
    fps = {}
    order = []
    for cell in code_cells:
        running.update(normalize_source(cell.get('source')).encode('utf-8'))
        running.update(b'\0')
        cid = cell_id_of(cell)
        if cid:
            fps[cid] = 'sha256:' + running.copy().hexdigest()
            order.append(cid)
    return fps, order


def build_provenance(repo_root, fw, d_path, max_inline=None):
    prov = {
        'framework_version': framework_version(repo_root, fw),
        'd2l_lib_fingerprint': d2l_lib_fingerprint(repo_root, d_path),
    }
    if max_inline is not None:
        prov['max_inline_bytes'] = max_inline
    return prov


# ── per-notebook capture ─────────────────────────────────────────

def capture_notebook(repo_root, store_dir, nb_path, fw, chapter, stem,
                     max_inline):
    nb = json.loads(nb_path.read_bytes())
    code_cells = code_cells_of(nb)
    fps, _ = prefix_fingerprints(code_cells)

    store_chapter = store_dir / fw / chapter
    asset_dir = store_chapter / stem
    # Rewrite assets from scratch so a cell that stops emitting an image leaves
    # no orphan file. Unchanged plots are rewritten byte-identically → no git diff.
    if asset_dir.exists():
        shutil.rmtree(asset_dir)

    cells = {}
    pending_assets = False
    for cell in code_cells:
        cid = cell_id_of(cell)
        if not cid:
            continue  # untagged cell: contributes to prefix hash, not to the store
        if not pending_assets:
            asset_dir.mkdir(parents=True, exist_ok=True)
            pending_assets = True
        outs, kind = extract_cell_outputs(cell, cid, stem, asset_dir, max_inline)
        cells[cid] = {
            'code_fingerprint': fps[cid],
            'kind': kind,
            'outputs': outs,
        }

    # Drop the asset dir if nothing was written to it.
    if asset_dir.exists() and not any(asset_dir.iterdir()):
        asset_dir.rmdir()

    manifest = {
        'schema': SCHEMA,
        'source': f'{chapter}/{stem}.md',
        'framework': fw,
        'provenance': build_provenance(
            repo_root, fw, nb_path.with_suffix('.d'), max_inline),
        'cells': cells,
    }
    store_chapter.mkdir(parents=True, exist_ok=True)
    out_path = store_chapter / f'{stem}.json'
    # indent=2, no sort_keys: dicts are built in a fixed (notebook) order, so the
    # output is deterministic AND diffs read top-to-bottom.
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    return len(cells), sum(
        1 for c in cells.values() for o in c['outputs'] if 'asset' in o)


# ── enumeration / filtering ──────────────────────────────────────

def parse_files_filter(tokens):
    """Accept `FILES=a.md,b.md`, `FILES=a.md`, or bare `chapter_x/foo.md`.
    Return a set of (chapter, stem) pairs, or None for 'all'."""
    pairs = set()
    for tok in tokens:
        if tok.startswith('FILES='):
            tok = tok[len('FILES='):]
        for item in tok.replace(',', ' ').split():
            p = Path(item)
            stem = p.stem
            chapter = p.parent.name if p.parent.name else None
            if chapter:
                pairs.add((chapter, stem))
            else:
                pairs.add((None, stem))
    return pairs or None


def matches(pairs, chapter, stem):
    if pairs is None:
        return True
    return (chapter, stem) in pairs or (None, stem) in pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*',
                    help='Optional subset: FILES=chapter_x/foo.md or chapter_x/foo.md')
    ap.add_argument('--project-dir', default='.')
    ap.add_argument('--notebooks-dir', default='_notebooks')
    ap.add_argument('--store-dir', default='outputs')
    ap.add_argument('--frameworks', default=','.join(FRAMEWORKS),
                    help='Comma-separated subset of frameworks')
    ap.add_argument('--max-inline-bytes', type=int, default=0,
                    help='Safety cap on inline text (0 = unlimited; preserves parity)')
    args = ap.parse_args()

    repo_root = Path(args.project_dir).resolve()
    nb_root = repo_root / args.notebooks_dir
    store_dir = repo_root / args.store_dir
    frameworks = [f.strip() for f in args.frameworks.split(',') if f.strip()]
    pairs = parse_files_filter(args.files)

    total_nb = total_cells = total_assets = 0
    for fw in frameworks:
        fw_dir = nb_root / fw
        if not fw_dir.exists():
            continue
        nbs = sorted(fw_dir.glob('chapter_*/*.ipynb'))
        n_fw = 0
        for nb_path in nbs:
            chapter = nb_path.parent.name
            stem = nb_path.stem
            if not matches(pairs, chapter, stem):
                continue
            c, a = capture_notebook(repo_root, store_dir, nb_path, fw,
                                    chapter, stem, args.max_inline_bytes)
            total_nb += 1
            n_fw += 1
            total_cells += c
            total_assets += a
        if n_fw:
            print(f'  {fw}: captured {n_fw} notebooks')

    print(f'Captured {total_nb} notebooks → {args.store_dir}/ '
          f'({total_cells} cells, {total_assets} image assets)')
    if total_nb == 0:
        print('  (nothing matched — check FILES= filter or that notebooks are '
              'executed)', file=sys.stderr)


if __name__ == '__main__':
    main()
