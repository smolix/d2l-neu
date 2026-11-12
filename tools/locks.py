#!/usr/bin/env python3
"""Notebook locks: reuse captured outputs instead of re-executing.

Some notebooks are expensive to re-execute (paid API calls, A100 or multi-node
training). Once their outputs are captured in the committed `outputs/` store,
the author LOCKS them with a one-line directive anywhere in the source `.md`
(outside code fences):

    <!-- execution: locked; requires: openrouter -->
    <!-- execution: locked; requires: 8xa100; frameworks: pytorch,jax -->

`requires:` is a free-form cost class used for reporting (`openrouter`,
`a100`, `8xa100`, `multinode`, ...). `frameworks:` limits the lock to some of
the frameworks the file has; without it every framework is locked. One
directive per file; anything that starts like a directive but does not parse
is an error.

What a lock does (docs/build-system.md, "Locked notebooks"):

  * Builds never execute a locked notebook. Its `.executed` stamp is satisfied
    by MATERIALIZING the executed notebook: the stored outputs are injected
    into the freshly generated `.ipynb` (`materialize`), which is marked so
    that capture never blesses it and an unlock re-executes it. The scheduler
    reports it as `LOCKED (skipped)`; a direct `make <stamp>` materializes too
    (tools/run_one_notebook.py).
  * Only `make rerun-locked FILES=...` executes it (D2L_RERUN_LOCKED=1), and
    only `make capture-outputs FILES=... UNLOCK=1` may overwrite its manifest.
  * The freshness gate tolerates prose edits and lib/version drift (warning)
    but fails on a code-cell change (tools/audit_outputs.py).

This module is the ONE parser of the directive. The scheduler, the audit,
capture, run_one_notebook, the MANIFEST.mk scan, the source linter, the
preprocessors (which strip the line) and `make locks` all use it.

CLI:
    python3 tools/locks.py list      # `make locks`: every lock + its status
    python3 tools/locks.py check     # validate every directive (exit 1 on error)
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FRAMEWORKS = ('pytorch', 'tensorflow', 'jax', 'mxnet')

# A well-formed directive line. The body is parsed by `_parse_body`.
_DIRECTIVE_RE = re.compile(r'^\s*<!--\s*execution\s*:(?P<body>.*?)-->\s*$')
# A comment that starts as the directive (`<!-- execution:`, any case) but
# does not parse: a hard error. Mere look-alikes (`<!-- executon: ...`) are
# only a lint warning (`looks_like_directive`), never a build failure.
_NEAR_RE = re.compile(r'^\s*<!--\s*execution\s*:', re.IGNORECASE)
_FIRST_WORD_RE = re.compile(r'^\s*<!--\s*([A-Za-z]+)')
_FENCE_RE = re.compile(r'^\s*(```|~~~)')
_TOKEN_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.+-]*$')
_KEYS = ('requires', 'frameworks')


class LockError(ValueError):
    """A malformed or duplicated execution directive."""

    def __init__(self, path, line, msg):
        self.path, self.line, self.msg = str(path), line, msg
        super().__init__(f'{path}:{line}: malformed execution directive: {msg}')


@dataclass(frozen=True)
class Lock:
    source: str                 # path of the .md, as given to the parser
    line: int                   # 1-based line of the directive
    requires: str = None        # cost class, e.g. 'openrouter' (None: unstated)
    frameworks: tuple = None    # locked frameworks (None: all the file has)

    def covers(self, fw):
        return self.frameworks is None or fw in self.frameworks

    @property
    def requires_label(self):
        return self.requires or '-'


def _parse_body(body, path, line):
    parts = [p.strip() for p in body.split(';')]
    if parts and parts[-1] == '':
        parts.pop()                       # tolerate a trailing ';'
    if not parts or parts[0] != 'locked':
        raise LockError(path, line, "the first field must be 'locked' "
                        f"(got {parts[0] if parts else 'nothing'!r})")
    fields = {}
    for part in parts[1:]:
        key, sep, val = part.partition(':')
        key, val = key.strip(), val.strip()
        if not sep or key not in _KEYS:
            raise LockError(path, line, f'unknown field {part!r} '
                            f'(expected one of: {", ".join(_KEYS)})')
        if key in fields:
            raise LockError(path, line, f'field {key!r} given twice')
        if not val:
            raise LockError(path, line, f'field {key!r} is empty')
        fields[key] = val
    requires = fields.get('requires')
    if requires is not None and not _TOKEN_RE.match(requires):
        raise LockError(path, line, f'requires: {requires!r} must be one '
                        'token such as openrouter, a100, 8xa100, multinode')
    frameworks = None
    if 'frameworks' in fields:
        fws = [f.strip() for f in fields['frameworks'].split(',')]
        bad = [f for f in fws if f not in FRAMEWORKS]
        if bad or not fws:
            raise LockError(path, line, f'frameworks: unknown {bad} '
                            f'(expected a comma list of {", ".join(FRAMEWORKS)})')
        if len(set(fws)) != len(fws):
            raise LockError(path, line, 'frameworks: duplicate entry')
        frameworks = tuple(fws)
    return Lock(str(path), line, requires, frameworks)


def _directive_lines(text):
    """Yield (lineno, line) for lines outside code fences."""
    in_fence = None
    for i, line in enumerate(text.split('\n'), start=1):
        m = _FENCE_RE.match(line)
        if m:
            if in_fence is None:
                in_fence = m.group(1)
            elif m.group(1) == in_fence:
                in_fence = None
            continue
        if in_fence is None:
            yield i, line


def parse_text(text, path='<text>'):
    """Return the Lock declared in *text*, or None. Raises LockError on a
    malformed directive or on more than one directive."""
    found = None
    for i, line in _directive_lines(text):
        m = _DIRECTIVE_RE.match(line)
        if not m:
            if _NEAR_RE.match(line):
                raise LockError(path, i, 'expected one line of the form '
                                '<!-- execution: locked; requires: <class> -->')
            continue
        if found is not None:
            raise LockError(path, i, f'second directive (first on line '
                            f'{found.line}); one per file')
        found = _parse_body(m.group('body'), path, i)
    return found


def parse_file(md_path):
    return parse_text(Path(md_path).read_text(encoding='utf-8'), md_path)


def looks_like_directive(text):
    """Line numbers of comments whose first word resembles `execution` but
    that are not the directive (`<!-- executon: locked -->`, `<!-- execution
    locked -->`): tools/lint_source.py warns about them."""
    import difflib
    hits = []
    for i, line in _directive_lines(text):
        if _DIRECTIVE_RE.match(line) or _NEAR_RE.match(line):
            continue
        m = _FIRST_WORD_RE.match(line)
        if m and difflib.SequenceMatcher(
                None, m.group(1).lower(), 'execution').ratio() >= 0.8:
            hits.append(i)
    return hits


def strip_directive(text):
    """Drop the directive line from *text* (preprocessors call this, so the
    line never reaches HTML, PDF, notebooks or slides). Text without a
    directive is returned unchanged."""
    if 'execution' not in text:
        return text
    drop = {i for i, line in _directive_lines(text) if _DIRECTIVE_RE.match(line)}
    if not drop:
        return text
    return '\n'.join(line for i, line in enumerate(text.split('\n'), start=1)
                     if i not in drop)


_CACHE = {}


def lock_for_source(repo_root, source_rel):
    """Lock of `<repo_root>/<source_rel>` (e.g. 'chapter_x/foo.md'), cached
    per process; None when the file is absent or has no directive."""
    path = Path(repo_root) / source_rel
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = (parse_text(path.read_text(encoding='utf-8'), source_rel)
                       if path.exists() else None)
    return _CACHE[key]


def lock_for_notebook(repo_root, fw, chapter, stem):
    """Lock covering framework *fw* of `chapter/stem.md`, or None."""
    lk = lock_for_source(repo_root, f'{chapter}/{stem}.md')
    return lk if lk is not None and lk.covers(fw) else None


def iter_locks(repo_root):
    """Yield (source_rel, Lock) for every locked `chapter_*/*.md`."""
    root = Path(repo_root)
    for md in sorted(root.glob('chapter_*/*.md')):
        lk = lock_for_source(root, str(md.relative_to(root)))
        if lk is not None:
            yield str(md.relative_to(root)), lk


# ── materialization (the .executed stamp of a locked notebook) ─────────

# Notebook-level marker on every notebook whose state came from the lock path
# (store outputs injected, or refused for code drift) rather than from an
# execution. Capture never blesses a marked notebook; once the source is no
# longer locked for that framework, the scheduler re-executes it. gen_notebooks
# carries the marker across a regeneration; run_notebooks clears it when the
# notebook is really executed.
MARKER_KEY = 'd2l'
MARKER = 'materialized_from_store'


def is_materialized(nb):
    """True if notebook dict (or path) *nb* carries the materialization mark."""
    if not isinstance(nb, dict):
        try:
            nb = json.loads(Path(nb).read_bytes())
        except (OSError, ValueError):
            return False
    return bool((nb.get('metadata', {}).get(MARKER_KEY) or {}).get(MARKER))


def mark_materialized(nb):
    nb.setdefault('metadata', {}).setdefault(MARKER_KEY, {})[MARKER] = True


def materialized_flag(nb_path):
    """`<stem>.materialized` next to the notebook: a cheap index of the
    marker (a stat, not a JSON parse), for the scheduler's per-run scan."""
    return Path(nb_path).with_suffix('.materialized')


def clear_materialized(nb_path):
    """Drop the mark from the notebook file (before a real execution).
    Returns True when the file changed."""
    nb_path = Path(nb_path)
    materialized_flag(nb_path).unlink(missing_ok=True)
    try:
        nb = json.loads(nb_path.read_bytes())
    except (OSError, ValueError):
        return False
    meta = nb.get('metadata', {}).get(MARKER_KEY)
    if not meta or MARKER not in meta:
        return False
    del meta[MARKER]
    if not meta:
        del nb['metadata'][MARKER_KEY]
    _write_nb(nb_path, nb)
    return True


def _write_nb(nb_path, nb):
    # Same serialization as tools/gen_notebooks.py.
    Path(nb_path).write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                             encoding='utf-8')


def code_fingerprints(nb):
    import capture_outputs as cap
    return cap.prefix_fingerprints(cap.code_cells_of(nb))[0]


def code_drift(cells, cur_fps):
    """Why the stored *cells* (manifest `cells`) do not belong to the code
    whose fingerprints are *cur_fps*: changed, new or removed code cells.
    Empty when the code matches. The audit uses the same rule."""
    changed = [cid for cid, entry in cells.items()
               if cid in cur_fps and entry.get('code_fingerprint') != cur_fps[cid]]
    new = sorted(set(cur_fps) - set(cells))
    gone = sorted(set(cells) - set(cur_fps))
    why = []
    if changed:
        why.append(f'{len(changed)} code cell(s) changed since capture')
    if new:
        why.append(f'{len(new)} new code cell(s) never captured')
    if gone:
        why.append(f'{len(gone)} captured cell(s) gone')
    return why, changed + new + gone


def sidecar_status(nb_path, nb=None):
    """Compare the execution sidecar of *nb_path* with the notebook's code.

    Returns (status, sidecar) with status
      'none'   no (readable) sidecar: the notebook was not executed;
      'exact'  the sidecar covers exactly the current code cells, with the
               same fingerprints, and every code cell ran: this notebook is a
               real execution of the current code (lib/version may differ);
      'drift'  anything else (a cell added, removed or edited since the run).
    """
    sc_path = Path(nb_path).with_suffix('.provenance.json')
    try:
        sc = json.loads(sc_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return 'none', None
    if nb is None:
        try:
            nb = json.loads(Path(nb_path).read_bytes())
        except (OSError, ValueError):
            return 'drift', sc
    ran = all(c.get('execution_count') is not None
              for c in nb.get('cells', []) if c.get('cell_type') == 'code'
              and ''.join(c.get('source') or '').strip())
    exact = sc.get('code_fingerprints') == code_fingerprints(nb)
    return ('exact' if exact and ran else 'drift'), sc


def materialize(repo_root, fw, nb_path, store_dir=None, touch_stamp=True):
    """Fill the generated notebook *nb_path* with its stored outputs instead
    of executing it. Returns (ok, message); the stamp is touched only on ok.

    1. A real, uncaptured re-run of the current code (`make rerun-locked`
       done, capture not yet: `sidecar_status` 'exact', not marked) is kept
       as it is, even when the lib moved since (lib drift is a warning for a
       locked notebook), so a regeneration does not throw paid outputs away.
    2. Otherwise the stored outputs are injected by cell id, the code cells
       are numbered 1..N like a top-to-bottom run, the notebook is marked
       `metadata.d2l.materialized_from_store` and stamped: everything that
       reads `_notebooks/<fw>/` sees an executed notebook, and capture will
       never bless it.
    3. If the store's cells do not match the current code (a code cell
       changed, appeared or disappeared) or there is no manifest, nothing is
       injected and nothing is stamped: the notebook is only marked, and the
       freshness gate reports the code drift.
    """
    from build_notebook_zips import inject_notebook
    from inject_outputs import index_store_by_id

    repo_root = Path(repo_root)
    nb_path = Path(nb_path)
    store_dir = Path(store_dir) if store_dir else repo_root / 'outputs'
    chapter, stem = nb_path.parent.name, nb_path.stem
    stamp = nb_path.with_suffix('.executed')
    manifest = store_dir / fw / chapter / f'{stem}.json'
    rerun = (f'make rerun-locked FILES={chapter}/{stem}.md, then make '
             f'capture-outputs FILES={chapter}/{stem}.md UNLOCK=1')
    try:
        nb = json.loads(nb_path.read_bytes())
    except OSError:
        return False, f'{nb_path} not generated (run `make notebooks-{fw}`)'
    cur_fps = code_fingerprints(nb)

    status, _ = sidecar_status(nb_path, nb)
    if status == 'exact' and not is_materialized(nb):
        if touch_stamp:
            stamp.touch()
        return True, ('kept an uncaptured re-run (capture it: make '
                      f'capture-outputs FILES={chapter}/{stem}.md UNLOCK=1)')

    def refuse(msg):
        stamp.unlink(missing_ok=True)
        materialized_flag(nb_path).touch()
        if not is_materialized(nb):
            mark_materialized(nb)
            _write_nb(nb_path, nb)
        return False, msg

    if not manifest.exists():
        return refuse(f'no stored outputs ({manifest.relative_to(repo_root)} '
                      f'missing); a build never executes a locked notebook: '
                      f'{rerun}')
    try:
        man = json.loads(manifest.read_text(encoding='utf-8'))
        why, _ = code_drift(man.get('cells', {}), cur_fps)
        if why:
            return refuse(f'code drift ({"; ".join(why)}): not materialized; '
                          f'revert the code edit, or {rerun}')
        ids, _ = index_store_by_id(store_dir, fw, chapter, stem)
        inject_notebook(nb, ids)
    except (OSError, ValueError, KeyError) as e:
        return False, f'cannot materialize from {manifest}: {e}'
    mark_materialized(nb)
    _write_nb(nb_path, nb)
    materialized_flag(nb_path).touch()
    # The notebook no longer holds the outputs a sidecar would describe.
    nb_path.with_suffix('.provenance.json').unlink(missing_ok=True)
    if touch_stamp:
        stamp.touch()
    return True, f'materialized from {manifest.relative_to(repo_root)}'


# ── `make locks` report ────────────────────────────────────────────────

def _git_date(repo_root, path):
    try:
        r = subprocess.run(['git', 'log', '-1', '--format=%cs', '--', str(path)],
                           cwd=repo_root, capture_output=True, text=True,
                           timeout=30)
        return r.stdout.strip() or 'uncommitted'
    except Exception:
        return '?'


def _runtime(seconds):
    if seconds is None:
        return '-'
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h{m:02d}m' if h else (f'{m}m{s:02d}s' if m else f'{s}s')


def lock_rows(repo_root, notebooks_dir='_notebooks', store_dir='outputs'):
    """One row per locked (source, framework): what `make locks` prints."""
    import audit_outputs
    from scan_notebook_manifests import file_frameworks
    root = Path(repo_root)
    nb_root, store = root / notebooks_dir, root / store_dir
    rows = []
    for source, lk in iter_locks(root):
        have = file_frameworks(root / source)
        for fw in [f for f in FRAMEWORKS if f in have and lk.covers(f)]:
            manifest = store / fw / Path(source).with_suffix('.json')
            row = {'source': source, 'framework': fw,
                   'requires': lk.requires_label, 'captured': '-',
                   'runtime': '-', 'status': 'ERROR', 'detail': ''}
            if not manifest.exists():
                row['detail'] = 'never captured (no manifest)'
                rows.append(row)
                continue
            prov = json.loads(manifest.read_text(encoding='utf-8')).get(
                'provenance', {})
            run = prov.get('execution') or {}
            row['captured'] = (run.get('finished') or '')[:10] or \
                _git_date(root, manifest)
            row['runtime'] = _runtime(run.get('runtime_s'))
            rec = audit_outputs.audit_manifest(root, nb_root, manifest, fw,
                                               lock=lk)
            status = rec.get('locked_status')
            row['status'] = {'clean': 'clean', 'lib-drift': 'WARNING',
                             'code-drift': 'ERROR',
                             'unchecked': 'unchecked'}.get(status, status)
            row['detail'] = '; '.join(rec.get('locked_reasons', []))
            rows.append(row)
        for fw in lk.frameworks or ():
            if fw not in have:
                rows.append({'source': source, 'framework': fw,
                             'requires': lk.requires_label, 'captured': '-',
                             'runtime': '-', 'status': 'WARNING',
                             'detail': 'locked framework has no notebook'})
    return rows


def cmd_list(args):
    root = Path(args.project_dir).resolve()
    try:
        rows = lock_rows(root, args.notebooks_dir, args.store_dir)
    except LockError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    if not rows:
        print('No locked notebooks.')
        return 0
    print(f'Locked notebooks ({len(rows)} source x framework):')
    w = max(len(r['source']) for r in rows)
    print(f'  {"source":<{w}}  {"framework":<10}  {"requires":<11}  '
          f'{"captured":<11}  {"runtime":>8}  status')
    for r in rows:
        line = (f'  {r["source"]:<{w}}  {r["framework"]:<10}  '
                f'{r["requires"]:<11}  {r["captured"]:<11}  {r["runtime"]:>8}  '
                f'{r["status"]}')
        if r['detail']:
            line += f' ({r["detail"]})'
        print(line)
    print('\n  clean      outputs match the current code, lib and framework version')
    print('  WARNING    locked, outputs from an older lib/version (rendered as is)')
    print('  ERROR      code cells changed since capture, or never captured:'
          ' the render gate fails.')
    print('             Re-run: make rerun-locked FILES=<chapter/file.md> '
          '[FRAMEWORKS=pytorch]')
    print('             then:   make capture-outputs FILES=<chapter/file.md> '
          'UNLOCK=1')
    print('  unchecked  generated notebook absent (make notebooks)')
    return 1 if any(r['status'] == 'ERROR' for r in rows) else 0


def cmd_check(args):
    root = Path(args.project_dir).resolve()
    files = args.files or sorted(root.glob('chapter_*/*.md'))
    errors = 0
    for f in files:
        try:
            parse_file(f)
        except LockError as e:
            print(e, file=sys.stderr)
            errors += 1
    return 1 if errors else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project-dir', default='.')
    ap.add_argument('--notebooks-dir', default='_notebooks')
    ap.add_argument('--store-dir', default='outputs')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='every locked notebook and its status')
    p = sub.add_parser('check', help='validate the directives')
    p.add_argument('files', nargs='*', type=Path)
    args = ap.parse_args()
    return {'list': cmd_list, 'check': cmd_check}[args.cmd](args)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
