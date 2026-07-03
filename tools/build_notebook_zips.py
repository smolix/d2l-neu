#!/usr/bin/env python3
"""Build one downloadable .zip of executed notebooks per framework.

This is the notebook analog of the per-framework PDF/slide outputs: a reader can
grab all of one framework's notebooks — WITH cell outputs baked in — from the
site's "Notebooks" menu. Like the rest of the decoupled build (docs/build-
system.md), it needs NO GPU and NO framework venv: the *code* comes from the
generated `_notebooks/<fw>/` tree (`make notebooks`, CPU-only) and the *outputs*
are injected from the committed `outputs/` store, keyed by cell id — the same
seam `inject_outputs.py` uses for html/slides/pdf.

For each framework it writes:

    <out-dir>/d2l-<fw>.zip        →  d2l-<fw>/<chapter>/<stem>.ipynb  (+ README.md)

Only computed cell outputs (plots, printed values) are embedded — those live in
the notebook. Illustrative figures included via `![](../img/…svg)` are NOT
bundled (img/ is ~118 MB); they render on the website. Notebooks import the
`d2l` package (`pip install d2l`), exactly like the d2l.ai downloads.

The zip is a deterministic function of the notebooks + store: fixed entry
timestamps and sorted order, so an unchanged build re-produces a byte-identical
zip (the hash-based R2 upload then skips it). Re-run after any recapture.

Usage:
    python tools/build_notebook_zips.py                       # all fw → _book/notebooks
    python tools/build_notebook_zips.py --frameworks pytorch  # one framework
    python tools/build_notebook_zips.py --out-dir _book/notebooks
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject_outputs import index_store_by_id  # store → {cell_id: [outputs]}

FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'mxnet']

# Fixed zip entry timestamp (1980-01-01, the zip epoch) so an unchanged build is
# byte-identical → the content-hash R2 sync doesn't re-upload it.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

_README = """\
# Dive into Deep Learning — {fw} notebooks

These are the executed {fw} notebooks from the book, with computed cell outputs
(plots, printed values) included.

## Running them

    pip install d2l

then open any notebook with Jupyter. Notebooks import the backend as, e.g.:

    from d2l import {mod} as d2l

Schematic/illustrative figures referenced as `../img/*.svg` are not bundled and
render on the website; every *computed* output is embedded here.
"""

_MOD = {'pytorch': 'torch', 'tensorflow': 'tensorflow', 'jax': 'jax',
        'mxnet': 'mxnet'}


def _code_cells(nb):
    return [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']


def _cell_id(cell):
    return cell.get('id') or cell.get('metadata', {}).get('id')


def normalize(nb):
    """Make the notebook validate cleanly as nbformat 4.4.

    gen_notebooks emits `nbformat_minor: 4` but also a *top-level* cell `id`
    (duplicated in `metadata.id`) — a combination strict `nbformat.validate`
    rejects, since top-level cell ids require minor >= 5 (which in turn requires
    ids on *every* cell, incl. markdown/raw, which these lack). Downloadable
    notebooks should open without warnings, so drop the redundant top-level id
    and keep `metadata.id`. Must run AFTER injection, which reads the id."""
    nb['nbformat'] = 4
    nb['nbformat_minor'] = 4
    for cell in nb.get('cells', []):
        cell.pop('id', None)
    return nb


def inject_notebook(nb, store_ids):
    """Return `nb` with each code cell's outputs replaced by the store's outputs
    for that cell id (empty if absent). execution_count is renumbered 1..N over
    code cells so the notebook reads as a clean top-to-bottom run."""
    n = 0
    for cell in _code_cells(nb):
        n += 1
        cell['execution_count'] = n
        cid = _cell_id(cell)
        outs = list(store_ids.get(cid, [])) if store_ids else []
        # execute_result outputs must carry execution_count to be valid nbformat.
        for o in outs:
            if o.get('output_type') == 'execute_result' and 'execution_count' not in o:
                o['execution_count'] = n
        cell['outputs'] = outs
    return nb


def build_zip(repo_root, notebooks_dir, store_dir, fw, out_dir):
    """Write <out-dir>/d2l-<fw>.zip. Returns (n_notebooks, n_with_outputs, bytes)."""
    fw_root = notebooks_dir / fw
    nbs = sorted(p for p in fw_root.rglob('*.ipynb')
                 if '.ipynb_checkpoints' not in p.parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f'd2l-{fw}.zip'

    n_nb = n_out = 0
    top = f'd2l-{fw}'
    # Deterministic: sorted entries, fixed timestamps, fixed compression.
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        readme = _README.format(fw=fw, mod=_MOD[fw])
        _writestr(z, f'{top}/README.md', readme)
        for nb_path in nbs:
            chapter = nb_path.parent.name
            stem = nb_path.stem
            nb = json.loads(nb_path.read_bytes())
            res = index_store_by_id(store_dir, fw, chapter, stem)
            store_ids = res[0] if res else {}
            inject_notebook(nb, store_ids)
            normalize(nb)
            if any(c.get('outputs') for c in _code_cells(nb)):
                n_out += 1
            text = json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
            _writestr(z, f'{top}/{chapter}/{stem}.ipynb', text)
            n_nb += 1
    return n_nb, n_out, zip_path.stat().st_size


def _writestr(z, arcname, text):
    """Add a text entry with a fixed timestamp (deterministic archive)."""
    info = zipfile.ZipInfo(arcname, date_time=_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16  # regular file, rw-r--r--
    z.writestr(info, text.encode('utf-8'))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project-dir', default='.')
    ap.add_argument('--notebooks-dir', default='_notebooks')
    ap.add_argument('--store-dir', default='outputs')
    ap.add_argument('--out-dir', default='_book/notebooks')
    ap.add_argument('--frameworks', default=','.join(FRAMEWORKS))
    args = ap.parse_args()

    repo_root = Path(args.project_dir).resolve()
    notebooks_dir = repo_root / args.notebooks_dir
    store_dir = repo_root / args.store_dir
    out_dir = repo_root / args.out_dir
    frameworks = [f.strip() for f in args.frameworks.split(',') if f.strip()]

    if not store_dir.exists():
        print(f'No store at {args.store_dir}/ — run `make capture-outputs` first',
              file=sys.stderr)
        return 1

    total = 0
    MB = 1048576
    for fw in frameworks:
        if not (notebooks_dir / fw).is_dir():
            print(f'  {fw}: no generated notebooks — run `make notebooks` (skipped)',
                  file=sys.stderr)
            continue
        n_nb, n_out, size = build_zip(repo_root, notebooks_dir, store_dir, fw, out_dir)
        total += 1
        print(f'  d2l-{fw}.zip: {n_nb} notebooks ({n_out} with outputs), '
              f'{size / MB:.1f} MB')
    if total == 0:
        print('  (nothing built — check --frameworks / that notebooks are generated)',
              file=sys.stderr)
        return 1
    print(f'Wrote {total} notebook zip(s) → {args.out_dir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
