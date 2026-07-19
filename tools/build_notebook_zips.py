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

    <out-dir>/d2l-<fw>.zip        →  d2l-<fw>/<chapter>/<stem>.ipynb
                                     d2l-<fw>/img/…
                                     d2l-<fw>/d2l/…
                                     d2l-<fw>/pylock.{cpu,gpu}.toml
                                     d2l-<fw>/{pyproject.toml,README.md}

Computed cell outputs (plots, printed values) live inline in the notebook; the
illustrative figures a notebook pulls in via `![](../img/…)` are bundled under
`d2l-<fw>/img/` — only the subset that framework actually references, not all of
img/ (~118 MB). The matching `d2l` source and pinned uv CPU/GPU environments are
bundled too, so the extracted archive is a runnable project. Datasets remain
on-demand downloads and are not copied into the archive.

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
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inject_outputs import index_store_by_id  # store → {cell_id: [outputs]}

FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'mxnet']

# Fixed zip entry timestamp (1980-01-01, the zip epoch) so an unchanged build is
# byte-identical → the content-hash R2 sync doesn't re-upload it.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

# Illustrative figures a notebook pulls in with `![](../img/…)`. A notebook lives
# at d2l-<fw>/<chapter>/<stem>.ipynb, so `../img/` resolves to d2l-<fw>/img/ —
# bundling the referenced files there makes the download self-contained.
_IMGREF = re.compile(r'\.\./img/([\w./-]+\.(?:svg|png|jpg|jpeg|gif))')

_README = """\
# Dive into Deep Learning — {fw} notebooks

These are the executed {fw} notebooks from the book. Computed outputs and the
figures referenced by the notebooks are included. The matching `d2l` source and
pinned environments are included as well.

The latest source is at <https://github.com/smolix/d2l-neu>.

## CPU setup (recommended first)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

    uv venv --python 3.12
    uv pip sync pylock.cpu.toml
    uv run --no-sync jupyter lab

{cpu_note}## NVIDIA GPU setup

On Linux/x86-64 with a compatible NVIDIA driver, replace the CPU lock above:

    uv pip sync pylock.gpu.toml
    uv run --no-sync jupyter lab

The GPU lock is framework-specific; check that `nvidia-smi` works before using
it. The notebooks import the backend as, e.g.:

    from d2l import {mod} as d2l

Notebook outputs can be read offline. Executing notebooks that use datasets or
pretrained models downloads those assets on first use and therefore needs an
internet connection and additional disk space.
"""

_CPU_NOTE = {
    'pytorch': '',
    'tensorflow': '',
    'jax': '',
    'mxnet': ('On Apple Silicon, use `pylock.cpu-macos.toml` instead of '
              '`pylock.cpu.toml`.\n\n'),
}

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


def build_zip(repo_root, notebooks_dir, store_dir, img_dir, env_dir,
              project_file, package_dir, fw, out_dir):
    """Write <out-dir>/d2l-<fw>.zip. Returns a stats dict."""
    fw_root = notebooks_dir / fw
    nbs = sorted(p for p in fw_root.rglob('*.ipynb')
                 if '.ipynb_checkpoints' not in p.parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f'd2l-{fw}.zip'

    n_nb = n_out = 0
    img_refs = set()  # relpaths under img/ that this framework's notebooks use
    top = f'd2l-{fw}'
    # Deterministic: sorted entries, fixed timestamps, fixed compression.
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        readme = _README.format(fw=fw, mod=_MOD[fw], cpu_note=_CPU_NOTE[fw])
        _write(z, f'{top}/README.md', readme.encode('utf-8'))
        _write(z, f'{top}/.python-version', b'3.12\n')
        _write(z, f'{top}/pyproject.toml', project_file.read_bytes())

        fw_env = env_dir / fw
        locks = sorted(fw_env.glob('pylock.*.toml'))
        if not locks:
            raise FileNotFoundError(f'no uv locks found under {fw_env}')
        for lock in locks:
            lock_text = lock.read_text()
            old = 'directory = { path = "../../", editable = true }'
            new = 'directory = { path = ".", editable = true }'
            if lock_text.count(old) != 1:
                raise ValueError(f'{lock}: expected one editable ../.. d2l path')
            lock_text = lock_text.replace(old, new)
            _write(z, f'{top}/{lock.name}', lock_text.encode('utf-8'))

        package_files = sorted(p for p in package_dir.glob('*.py') if p.is_file())
        if not package_files:
            raise FileNotFoundError(f'no Python package files under {package_dir}')
        for src in package_files:
            _write(z, f'{top}/d2l/{src.name}', src.read_bytes())

        for nb_path in nbs:
            chapter = nb_path.parent.name
            stem = nb_path.stem
            raw = nb_path.read_bytes()
            img_refs.update(_IMGREF.findall(raw.decode('utf-8', 'replace')))
            nb = json.loads(raw)
            res = index_store_by_id(store_dir, fw, chapter, stem)
            store_ids = res[0] if res else {}
            inject_notebook(nb, store_ids)
            normalize(nb)
            if any(c.get('outputs') for c in _code_cells(nb)):
                n_out += 1
            text = json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
            _write(z, f'{top}/{chapter}/{stem}.ipynb', text.encode('utf-8'))
            n_nb += 1

        # Bundle the referenced illustrative figures so `../img/…` resolves
        # inside the extracted tree. Sorted → deterministic; missing files are
        # reported, not fatal.
        n_img = img_bytes = missing = 0
        for rel in sorted(img_refs):
            src = img_dir / rel
            if not src.is_file():
                missing += 1
                continue
            _write(z, f'{top}/img/{rel}', src.read_bytes())
            n_img += 1
            img_bytes += src.stat().st_size

    return {'notebooks': n_nb, 'with_outputs': n_out, 'images': n_img,
            'img_bytes': img_bytes, 'img_missing': missing,
            'locks': len(locks), 'package_files': len(package_files),
            'zip_bytes': zip_path.stat().st_size}


def _write(z, arcname, data: bytes):
    """Add an entry with a fixed timestamp (deterministic archive)."""
    info = zipfile.ZipInfo(arcname, date_time=_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16  # regular file, rw-r--r--
    z.writestr(info, data)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project-dir', default='.')
    ap.add_argument('--notebooks-dir', default='_notebooks')
    ap.add_argument('--store-dir', default='outputs')
    ap.add_argument('--img-dir', default='img')
    ap.add_argument('--env-dir', default='notebook_envs')
    ap.add_argument('--project-file', default='pyproject.toml')
    ap.add_argument('--package-dir', default='d2l')
    ap.add_argument('--out-dir', default='_book/notebooks')
    ap.add_argument('--frameworks', default=','.join(FRAMEWORKS))
    args = ap.parse_args()

    repo_root = Path(args.project_dir).resolve()
    notebooks_dir = repo_root / args.notebooks_dir
    store_dir = repo_root / args.store_dir
    img_dir = repo_root / args.img_dir
    env_dir = repo_root / args.env_dir
    project_file = repo_root / args.project_file
    package_dir = repo_root / args.package_dir
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
        s = build_zip(repo_root, notebooks_dir, store_dir, img_dir, env_dir,
                      project_file, package_dir, fw, out_dir)
        total += 1
        miss = f', {s["img_missing"]} img MISSING' if s['img_missing'] else ''
        print(f'  d2l-{fw}.zip: {s["notebooks"]} notebooks '
              f'({s["with_outputs"]} with outputs) + {s["images"]} images '
              f'({s["img_bytes"] / MB:.1f} MB raw) + {s["locks"]} uv locks + '
              f'{s["package_files"]} d2l modules{miss} → '
              f'{s["zip_bytes"] / MB:.1f} MB')
    if total == 0:
        print('  (nothing built — check --frameworks / that notebooks are generated)',
              file=sys.stderr)
        return 1
    print(f'Wrote {total} notebook zip(s) → {args.out_dir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
