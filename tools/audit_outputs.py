#!/usr/bin/env python3
"""Audit the committed `outputs/` store: freshness + referential integrity.

See docs/build-system.md §3 (freshness) and §4.3 (integrity).

Freshness is **code-provenance driven, never output-equality driven** — so the
nondeterminism of training (loss curves, sampled plots) never false-flags a
notebook. A notebook is STALE iff, relative to the current source/environment:
  - its `framework_version` changed, OR
  - its `d2l_lib_fingerprint` changed (a #@save symbol it uses was edited), OR
  - any cell's prefix-inclusive `code_fingerprint` changed (its code, or an
    earlier cell's code, was edited).

Severity follows the storage split (§3.3):
  - a stale cell with an INLINE (text) output is a HARD error — the printed value
    would be wrong beside its code;
  - a stale cell with only ASSET (image) outputs is a SOFT warning — an old but
    valid sample.

Integrity (store ↔ current notebook):
  - manifest entry whose cell id is gone from the notebook  → ORPHANED (error)
  - id'd notebook cell with no manifest entry               → MISSING  (warning:
    needs capture)

Modes:
  (default)        human report; exit non-zero on integrity errors.
  --verify-fresh   render gate: exit non-zero on any HARD-stale (inline) cell or
                   integrity error; SOFT (asset) staleness only warns.
  --stale          print the minimal re-execution set (one source path per line).
  --json           machine-readable report to stdout.

Requires the *generated* notebooks (`_notebooks/<fw>/...`) to exist so current
fingerprints can be computed — generation is CPU-only (`make notebooks`); it does
not execute anything.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import capture_outputs as cap
from scan_notebook_manifests import source_execution_class

FRAMEWORKS = cap.FRAMEWORKS


def host_gpu_count():
    """Number of NVIDIA GPUs visible on this host (0 if none / no nvidia-smi).

    Used to make the freshness gate host-capability-aware: a stale notebook
    cannot be re-executed on a box with fewer GPUs than it needs (a GPU notebook
    on Apple Silicon, a multi-GPU notebook on a single-GPU box), so there is
    nothing the author could do *here* to fix it. Hard-failing the render on
    such a box would defeat the point of committing pre-executed outputs — a
    portable baseline that lets any machine render the whole book and re-run
    only the notebooks its hardware can actually execute. So those stay a
    WARNING (render from store); a host that DOES have enough GPUs still
    hard-fails and refreshes them. See docs/build-system.md §3.
    """
    exe = shutil.which('nvidia-smi')
    if not exe:
        return 0
    try:
        r = subprocess.run([exe, '-L'], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return 0
        return sum(1 for ln in r.stdout.splitlines() if ln.strip().startswith('GPU '))
    except Exception:
        return 0


def audit_manifest(repo_root, nb_root, manifest_path, fw):
    """Return a dict describing one notebook's freshness + integrity."""
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    chapter = manifest_path.parent.name
    stem = manifest_path.stem
    source = manifest.get('source', f'{chapter}/{stem}.md')

    rec = {
        'framework': fw, 'source': source,
        'manifest': str(manifest_path),
        'stale': False, 'hard_stale': False, 'reasons': [],
        'orphaned': [], 'missing': [], 'notebook_present': True,
    }

    nb_path = nb_root / fw / chapter / f'{stem}.ipynb'
    if not nb_path.exists():
        rec['notebook_present'] = False
        rec['reasons'].append('generated notebook absent (run `make notebooks`)')
        return rec

    nb = json.loads(nb_path.read_bytes())
    cur_fps, _ = cap.prefix_fingerprints(cap.code_cells_of(nb))
    cur_prov = cap.build_provenance(repo_root, fw, nb_path.with_suffix('.d'))
    old_prov = manifest.get('provenance', {})

    # Notebook-level invalidation. framework_version compares the PEP 440
    # *public* version only — the platform/build local segment ('+cu128',
    # '+cpu') identifies the wheel, not the framework version, and must not
    # invalidate a store captured on a different machine (cap.public_version
    # also normalizes legacy '+cu128' manifests captured before this change).
    prov_drift = False
    for key in ('framework_version', 'd2l_lib_fingerprint'):
        old_val, cur_val = old_prov.get(key), cur_prov.get(key)
        if key == 'framework_version':
            old_val = cap.public_version(old_val) if old_val else old_val
            cur_val = cap.public_version(cur_val) if cur_val else cur_val
        if old_val != cur_val:
            prov_drift = True
            rec['reasons'].append(
                f'{key}: {old_prov.get(key)} → {cur_prov.get(key)}')

    cells = manifest.get('cells', {})
    cur_ids = set(cur_fps)
    man_ids = set(cells)

    rec['orphaned'] = sorted(man_ids - cur_ids)
    rec['missing'] = sorted(cur_ids - man_ids)

    def has_inline(entry):
        return any('asset' not in o for o in entry.get('outputs', []))

    stale_cells = []
    for cid, entry in cells.items():
        if cid not in cur_fps:
            continue  # orphaned — handled in integrity
        drift = prov_drift or (entry.get('code_fingerprint') != cur_fps[cid])
        if drift:
            stale_cells.append(cid)
            if has_inline(entry):
                rec['hard_stale'] = True
    if stale_cells:
        rec['stale'] = True
        if prov_drift and 'provenance drift' not in rec['reasons']:
            pass  # reasons already list the specific keys
        elif not prov_drift:
            rec['reasons'].append(
                f'{len(stale_cells)} cell(s) with changed code')
    return rec


def collect(repo_root, store_dir, nb_root, frameworks):
    records = []
    for fw in frameworks:
        fw_dir = store_dir / fw
        if not fw_dir.exists():
            continue
        for manifest_path in sorted(fw_dir.glob('chapter_*/*.json')):
            records.append(
                audit_manifest(repo_root, nb_root, manifest_path, fw))
    return records


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project-dir', default='.')
    ap.add_argument('--notebooks-dir', default='_notebooks')
    ap.add_argument('--store-dir', default='outputs')
    ap.add_argument('--frameworks', default=','.join(FRAMEWORKS))
    ap.add_argument('--verify-fresh', action='store_true',
                    help='render gate: fail on hard-stale (inline) or integrity errors')
    ap.add_argument('--stale', action='store_true',
                    help='print minimal re-execution set (source paths)')
    ap.add_argument('--json', action='store_true', help='machine-readable output')
    args = ap.parse_args()

    repo_root = Path(args.project_dir).resolve()
    store_dir = repo_root / args.store_dir
    nb_root = repo_root / args.notebooks_dir
    frameworks = [f.strip() for f in args.frameworks.split(',') if f.strip()]

    if not store_dir.exists():
        print(f'No store at {args.store_dir}/ — run `make capture-outputs` first',
              file=sys.stderr)
        return 1

    records = collect(repo_root, store_dir, nb_root, frameworks)

    stale = [r for r in records if r['stale']]
    hard = [r for r in records if r['hard_stale']]
    orphaned = [r for r in records if r['orphaned']]
    missing = [r for r in records if r['missing']]
    no_nb = [r for r in records if not r['notebook_present']]
    # `no_nb` is a warning, NOT an integrity error: a pure CPU-only render box has
    # the committed store but no generated _notebooks/, and must still render. The
    # freshness gate only bites where the notebooks exist (an author's dev box, or
    # CI after `make notebooks`). Orphaned ids are always a real error.
    integrity_err = bool(orphaned)

    if args.json:
        print(json.dumps({
            'total': len(records),
            'stale': [r['source'] + f" [{r['framework']}]" for r in stale],
            'hard_stale': [r['source'] + f" [{r['framework']}]" for r in hard],
            'orphaned': {f"{r['framework']}:{r['source']}": r['orphaned']
                         for r in orphaned},
            'missing': {f"{r['framework']}:{r['source']}": r['missing']
                        for r in missing},
        }, indent=2))
        return 1 if integrity_err else 0

    if args.stale:
        # minimal re-execution set — unique source paths, one per line
        for src in sorted({r['source'] for r in stale}):
            print(src)
        return 0

    # ── human report ──
    print(f'Audited {len(records)} manifests across {len(frameworks)} framework(s).')

    if no_nb:
        print(f'\n  ⚠ {len(no_nb)} manifest(s) with no generated notebook '
              f'(run `make notebooks`):')
        for r in no_nb[:10]:
            print(f'      {r["framework"]:11} {r["source"]}')

    if orphaned:
        print(f'\n  ✗ {len(orphaned)} manifest(s) with ORPHANED cell ids '
              f'(in store, gone from notebook):')
        for r in orphaned[:10]:
            print(f'      {r["framework"]:11} {r["source"]}: {", ".join(r["orphaned"][:6])}')

    if missing:
        print(f'\n  ⚠ {len(missing)} manifest(s) MISSING cells '
              f'(in notebook, not captured — run capture):')
        for r in missing[:10]:
            print(f'      {r["framework"]:11} {r["source"]}: {", ".join(r["missing"][:6])}')

    if stale:
        print(f'\n  ● {len(stale)} STALE notebook(s) '
              f'({len(hard)} hard / inline, {len(stale) - len(hard)} soft / asset-only):')
        for r in stale[:30]:
            tag = 'HARD' if r['hard_stale'] else 'soft'
            print(f'      [{tag}] {r["framework"]:11} {r["source"]}'
                  f'  — {"; ".join(r["reasons"])}')
        if len(stale) > 30:
            print(f'      ... and {len(stale) - 30} more')
    else:
        print('  ✓ no stale notebooks.')

    if not (stale or integrity_err or missing):
        print('\n  ✓ store is clean and fresh.')

    if args.verify_fresh:
        gpus = host_gpu_count()
        # GPUs each resource class needs to *execute*. A stale notebook the
        # current host cannot run (it has fewer GPUs than the class needs) is
        # deferred to a warning and rendered from the committed store — nothing
        # the author could do here would fix it. One the host CAN run still
        # hard-fails. This tiers cleanly: a CPU box (0) defers gpu+multi-gpu, a
        # single-GPU box (1) defers only multi-gpu, a multi-GPU box (>=2) blocks
        # on everything (the strict canonical gate).
        NEED = {'cpu': 0, 'gpu': 1, 'multi-gpu': 2}
        blocked, deferred = [], []
        for r in hard:
            rel = Path(r['source'])
            cls = source_execution_class(repo_root / rel, rel)
            (deferred if gpus < NEED.get(cls, 0) else blocked).append((r, cls))
        if deferred:
            print(f'\n  ⚠ {len(deferred)} stale notebook(s) need more GPUs than '
                  f'this host has ({gpus}) — rendering from the committed store; '
                  f'refresh them on a host with enough GPUs:', file=sys.stderr)
            for r, cls in deferred:
                print(f'      [{cls}] {r["framework"]:11} {r["source"]}',
                      file=sys.stderr)
        if blocked:
            print(f'\nFAIL (verify-fresh): {len(blocked)} notebook(s) have stale '
                  f'INLINE outputs you can re-run on this host. Re-run + capture '
                  f'them, or use `make render-fresh`.', file=sys.stderr)
            return 1
        if integrity_err:
            print('\nFAIL (verify-fresh): integrity errors above.', file=sys.stderr)
            return 1
        return 0

    return 1 if integrity_err else 0


if __name__ == '__main__':
    sys.exit(main() or 0)
