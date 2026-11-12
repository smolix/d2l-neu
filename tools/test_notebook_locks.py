#!/usr/bin/env python3
"""Regression tests for notebook locks (docs/build-system.md "Locked notebooks").

A source `.md` carrying `<!-- execution: locked; requires: <class> -->` must
never be executed by a build: its `.executed` stamp is satisfied from the
committed `outputs/` store, the freshness gate tolerates prose edits and
lib/version drift but fails on code drift, capture refuses to overwrite its
manifest without UNLOCK, and only `make rerun-locked` executes it.

Everything runs on a throwaway fixture under a tmp dir (no GPU, no venv, a few
seconds): `python3 tools/test_notebook_locks.py` or `make test-trap`. The real
store is never touched.
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import locks                           # noqa: E402
import notebook_scheduler as ns        # noqa: E402
import run_notebooks                   # noqa: E402

FW = 'pytorch'
CH = 'chapter_lock_test'
DIRECTIVE = '<!--   execution:  locked ;requires: openrouter  -->'

_passed = 0
_failed = 0


def check(cond, msg):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f'  ok   {msg}')
    else:
        _failed += 1
        print(f'  FAIL {msg}')


# ── fixture ──────────────────────────────────────────────────────────

def md_text(prose='Prose that may change freely.', directive=DIRECTIVE):
    return (f'# Locked fixture\n:label:`sec_lock_fixture`\n\n{directive}\n\n'
            f'{prose}\n\n```{{.python .input #locked-1}}\nimport d2l\n'
            f'print(d2l.Foo.answer)\n```\n\n```{{.python .input #locked-2}}\n'
            f'x = 1\nx\n```\n\n<!-- slides -->\n\n::: {{.slide}}\n## Slide\n'
            f'@locked-1\n:::\n')


CODE = {'locked-1': 'import d2l\nprint(d2l.Foo.answer)', 'locked-2': 'x = 1\nx'}
OUTS = {'locked-1': [{'output_type': 'stream', 'name': 'stdout',
                      'text': ['42\n']}],
        'locked-2': [{'output_type': 'execute_result', 'execution_count': 2,
                      'data': {'text/plain': ['1']}, 'metadata': {}}]}


def notebook(executed, code=CODE):
    cells = [{'cell_type': 'markdown', 'metadata': {},
              'source': ['Prose that may change freely.']}]
    for n, (cid, src) in enumerate(code.items(), start=1):
        cells.append({'cell_type': 'code', 'id': cid, 'metadata': {'id': cid},
                      'source': src, 'execution_count': n if executed else None,
                      'outputs': OUTS.get(cid, []) if executed else []})
    return {'cells': cells, 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}


def nb_path(stem):
    return FIX / '_notebooks' / FW / CH / f'{stem}.ipynb'


def write_nb(stem, executed, code=CODE):
    nb_path(stem).write_text(json.dumps(notebook(executed, code), indent=1),
                             encoding='utf-8')


def build_fixture():
    (FIX / CH).mkdir(parents=True)
    (FIX / CH / 'locked.md').write_text(md_text(), encoding='utf-8')
    (FIX / CH / 'other.md').write_text(
        md_text(directive='<!-- execution: locked; requires: 8xa100 -->'),
        encoding='utf-8')
    (FIX / CH / 'plain.md').write_text(md_text(directive=''), encoding='utf-8')
    shard = FIX / 'd2l' / '_blocks' / FW / 'Foo.py'
    shard.parent.mkdir(parents=True)
    shard.write_text('class Foo:\n    answer = 42\n', encoding='utf-8')
    (FIX / '_notebooks' / FW / CH).mkdir(parents=True)
    for stem in ('locked', 'other', 'plain'):
        write_nb(stem, executed=True)
        deps = f'    d2l/_blocks/{FW}/Foo.py\n' if stem != 'plain' else '\n'
        nb_path(stem).with_suffix('.d').write_text(
            f'_notebooks/{FW}/{CH}/{stem}.executed: \\\n{deps}', encoding='utf-8')


def record_sidecar(stem, elapsed=12.5):
    saved = run_notebooks.NOTEBOOKS_DIR
    try:
        run_notebooks.NOTEBOOKS_DIR = FIX / '_notebooks'
        run_notebooks.write_execution_provenance(nb_path(stem), elapsed=elapsed)
    finally:
        run_notebooks.NOTEBOOKS_DIR = saved


def run_tool(script, *args):
    r = subprocess.run([sys.executable, str(TOOLS / script),
                        '--project-dir', str(FIX), *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def manifest(stem):
    return FIX / 'outputs' / FW / CH / f'{stem}.json'


@contextlib.contextmanager
def fixture_root():
    """Point the scheduler (module constants) at the fixture."""
    saved = ns.ROOT, ns.NOTEBOOKS_DIR
    ns.ROOT, ns.NOTEBOOKS_DIR = FIX, FIX / '_notebooks'
    try:
        yield
    finally:
        ns.ROOT, ns.NOTEBOOKS_DIR = saved


def capture(*args):
    return run_tool('capture_outputs.py', '--frameworks', FW, *args)


# ── tests ────────────────────────────────────────────────────────────

def test_parser():
    print('parser')
    lk = locks.parse_text(md_text(), 'x.md')
    check(lk is not None and lk.requires == 'openrouter' and lk.line == 4
          and lk.frameworks is None and lk.covers('jax'),
          'whitespace-tolerant directive parses; default covers every framework')
    lk = locks.parse_text('<!-- execution: locked; requires: a100; '
                          'frameworks: pytorch, jax -->\n', 'x.md')
    check(lk.frameworks == ('pytorch', 'jax') and lk.covers('jax')
          and not lk.covers('mxnet'), 'frameworks: limits the lock')
    check(locks.parse_text('<!-- execution: locked -->') .requires is None,
          'requires: is optional (reported as "-")')
    check(locks.parse_text('```\n<!-- execution: locked -->\n```\n') is None,
          'a directive inside a code fence is ignored')
    check(locks.parse_text('no directive\n<!-- split: x -->\n') is None,
          'files without the directive are unlocked')
    bad = {
        'state is not locked': '<!-- execution: unlocked -->',
        'unknown field': '<!-- execution: locked; cost: 3 -->',
        'unknown framework': '<!-- execution: locked; frameworks: torch -->',
        'empty field': '<!-- execution: locked; requires: -->',
        'two-token requires': '<!-- execution: locked; requires: big gpu -->',
        'missing -->': '<!-- execution: locked; requires: a100',
        'two directives': '<!-- execution: locked -->\n<!-- execution: locked -->',
    }
    for what, text in bad.items():
        try:
            locks.parse_text(text, 'x.md')
            check(False, f'malformed directive raises ({what})')
        except locks.LockError as e:
            check(e.line >= 1, f'malformed directive raises ({what})')


def test_directive_not_rendered():
    print('directive never reaches a rendered artifact')
    import d2l_preprocess
    import gen_notebooks
    import gen_pdf
    import gen_slides
    md = FIX / CH / 'locked.md'
    qmd = d2l_preprocess.convert_file(str(md), primary=FW)
    check('execution:' not in qmd and 'Prose that may change' in qmd,
          'HTML preprocessor (d2l_preprocess.convert_file) drops it')
    pdf = gen_pdf.convert_file_pdf(str(md), FW)
    check('execution:' not in pdf, 'PDF preprocessor (gen_pdf) drops it')
    nbq = gen_notebooks.convert_file_notebook(str(md), FW)
    check('execution:' not in nbq, 'notebook generator (gen_notebooks) drops it')
    deck = gen_slides.generate_slides_qmd(str(md), FW, []) or ''
    check('execution:' not in deck and 'Slide' in deck,
          'slide generator (gen_slides) never sees it')
    check(locks.strip_directive('a\nb\n') == 'a\nb\n',
          'text without a directive is returned unchanged')


def test_capture_refuses_without_unlock():
    print('capture')
    rc, out, err = capture('FILES=%s/plain.md' % CH)
    check(rc == 0 and manifest('plain').exists(), 'unlocked notebook captures')
    record_sidecar('locked')
    rc, out, err = capture('FILES=%s/locked.md' % CH)
    check(rc == 2 and 'REFUSING' in err and 'LOCKED' in err
          and not manifest('locked').exists(),
          'capture REFUSES a named locked notebook without UNLOCK (nothing written)')
    rc, out, err = capture()
    check(rc == 0 and not manifest('locked').exists() and 'skipped' in err,
          'capture-all skips locked notebooks without UNLOCK')
    rc, out, err = capture('--unlock', 'FILES=%s/locked.md' % CH)
    man = json.loads(manifest('locked').read_text()) if manifest('locked').exists() else {}
    check(rc == 0 and man.get('cells', {}).get('locked-1'),
          'capture --unlock (UNLOCK=1) blesses a re-executed locked notebook')
    check((man.get('provenance', {}).get('execution') or {}).get('runtime_s') == 12.5,
          'the re-run time and runtime land in provenance.execution')
    (nb_path('other').with_suffix('.provenance.json')).unlink(missing_ok=True)
    rc, out, err = capture('--unlock', 'FILES=%s/other.md' % CH)
    check(rc == 2 and not manifest('other').exists(),
          'capture --unlock refuses a locked notebook with no execution provenance')
    record_sidecar('other', elapsed=3600)
    rc, out, err = capture('--unlock', 'FILES=%s/other.md' % CH)
    check(rc == 0 and manifest('other').exists(), 'other locked notebook captured')


def test_scheduler_skips_and_materializes():
    print('scheduler: skip + materialize')
    # A pristine build: `make clean && make notebooks` leaves generated,
    # unexecuted notebooks, no stamps and no sidecars.
    for stem in ('locked', 'other', 'plain'):
        write_nb(stem, executed=False)
        for suf in ('.executed', '.provenance.json'):
            nb_path(stem).with_suffix(suf).unlink(missing_ok=True)
    with fixture_root():
        locked = []
        items = ns.build_worklist([FW], locked=locked)
        check([it.nb.stem for it in items] == ['plain'],
              'only the unlocked notebook is queued for execution')
        check(sorted(it.nb.stem for it, stale, _ in locked) == ['locked', 'other']
              and all(stale for _, stale, _ in locked),
              'both locked notebooks are reported (stale stamp)')
        check([it.nb.stem for it in ns.build_worklist([FW])] == ['plain'],
              'callers that pass no `locked` list never get locked items')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            failed = ns.handle_locked(locked, dry_run=True)
        check(not failed and buf.getvalue().count('LOCKED (skipped)') == 2
              and not nb_path('locked').with_suffix('.executed').exists(),
              '--dry-run lists them as LOCKED (skipped) and writes nothing')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            failed = ns.handle_locked(locked)
    nb = json.loads(nb_path('locked').read_text())
    code = [c for c in nb['cells'] if c['cell_type'] == 'code']
    check(not failed and nb_path('locked').with_suffix('.executed').exists(),
          'materialization writes the .executed stamp')
    check(''.join(code[0]['outputs'][0]['text']) == '42\n'
          and code[1]['outputs'][0]['data']['text/plain'] == '1'
          and [c['execution_count'] for c in code] == [1, 2],
          'the notebook carries the stored outputs, numbered like a run')
    check(nb['cells'][0]['cell_type'] == 'markdown',
          'prose cells of the generated notebook are kept')

    # The scheduler's summary.
    env = {'D2L_GPU_SLOTS_PER': '1', 'D2L_GPU_VRAM_PER': '24564',
           'D2L_CPU_SLOTS': '1', 'D2L_GPU_MIB_PER_SLOT': '7680',
           'D2L_JAX_TOTAL_SLOTS': '1'}
    saved_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    saved_argv = sys.argv
    try:
        for stem in ('locked', 'other'):
            nb_path(stem).with_suffix('.executed').unlink()
        sys.argv = ['notebook_scheduler.py', '--frameworks', FW, '--dry-run']
        buf = io.StringIO()
        with fixture_root(), contextlib.redirect_stdout(buf):
            rc = ns.main()
        out = buf.getvalue()
        check(rc == 0 and '2 LOCKED (skipped)' in out
              and '1/1 ok, 0 failed' in out,
              'summary: 1/1 ok, 0 failed, 2 LOCKED (skipped)')
    finally:
        sys.argv = saved_argv
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # A direct `make <stamp>` reaches run_one_notebook.py: it materializes too.
    import run_one_notebook as r1
    write_nb('other', executed=False)
    saved = r1.NOTEBOOKS_DIR, sys.argv
    r1.NOTEBOOKS_DIR = FIX / '_notebooks'
    sys.argv = ['run_one_notebook.py', FW, str(nb_path('other'))]
    os.environ.pop('D2L_RERUN_LOCKED', None)
    code = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r1.main()
    except SystemExit as e:
        code = e.code
    finally:
        r1.NOTEBOOKS_DIR, sys.argv = saved
    nb = json.loads(nb_path('other').read_text())
    check(code == 0 and any(c.get('outputs') for c in nb['cells']),
          'run_one_notebook.py (direct make <stamp>) materializes, never executes')

    # No stored outputs → materialization fails loudly.
    shutil.move(manifest('other'), FIX / 'other.json.bak')
    ok, msg = locks.materialize(FIX, FW, nb_path('other'))
    check(not ok and 'rerun-locked' in msg,
          'a locked notebook with no manifest fails with a re-run hint')
    shutil.move(FIX / 'other.json.bak', manifest('other'))

    # An uncaptured re-run (fresh sidecar) survives a regeneration.
    write_nb('other', executed=True)
    record_sidecar('other')
    nb_before = nb_path('other').read_text()
    ok, msg = locks.materialize(FIX, FW, nb_path('other'))
    check(ok and 'uncaptured' in msg and nb_path('other').read_text() == nb_before,
          'materialize keeps an uncaptured re-run (fresh sidecar)')
    nb_path('other').with_suffix('.provenance.json').unlink()


def test_rerun_selects_named():
    print('rerun-locked')
    with fixture_root():
        files = ns.parse_files('%s/locked.md' % CH)
        items = ns.build_worklist([FW], files=files, rerun_locked=True,
                                  force_all=True)
        check([it.nb.stem for it in items] == ['locked'],
              'rerun-locked selects exactly the named locked notebook')
        check(ns.build_worklist(['jax'], files=files, rerun_locked=True) == [],
              'a framework without a notebook selects nothing')

        class A:
            dry_run = True
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc = ns.rerun_locked(A, [FW], ns.parse_files('%s/plain.md' % CH))
        check(rc == 2 and 'not locked' in err.getvalue(),
              'rerun-locked refuses an unlocked file')
        with contextlib.redirect_stderr(io.StringIO()):
            rc = ns.rerun_locked(A, [FW], None)
        check(rc == 2, 'rerun-locked refuses an empty FILES')


def gate():
    return run_tool('audit_outputs.py', '--verify-fresh')


def test_gate():
    print('freshness gate')
    rc, out, err = gate()
    check(rc == 0, 'clean locked store passes --verify-fresh')

    (FIX / CH / 'locked.md').write_text(
        md_text(prose='Rewritten prose, a new figure ![x](../img/x.svg), '
                      'a reference :numref:`sec_x`.'), encoding='utf-8')
    rc, out, err = gate()
    check(rc == 0, 'a prose edit in a locked file leaves the gate green')

    shard = FIX / 'd2l' / '_blocks' / FW / 'Foo.py'
    shard.write_text('class Foo:\n    answer = 43  # lib changed\n',
                     encoding='utf-8')
    rc, out, err = gate()
    check(rc == 0 and 'locked, outputs from older lib/version' in err,
          'a lib-fingerprint change gives a WARNING only')
    rc, out, err = run_tool('audit_outputs.py', '--stale')
    check('locked.md' not in out and 'other.md' not in out,
          '--stale (refresh-stale) never lists a locked notebook')
    rc, out, err = run_tool('audit_outputs.py', '--stale-stamps')
    check('locked.executed' not in out, '--stale-stamps never lists one either')
    rc, out, err = run_tool('audit_outputs.py', '--json')
    rep = json.loads(out)
    check(rep.get('locked', {}).get(f'{FW}:{CH}/locked.md') == 'lib-drift',
          '--json reports locked_status lib-drift')

    code = dict(CODE, **{'locked-2': 'x = 2\nx'})
    write_nb('locked', executed=True, code=code)
    rc, out, err = gate()
    check(rc != 0 and 'LOCKED' in err and 'rerun-locked' in err,
          'a code-cell edit in a locked notebook FAILS the gate')
    rc, out, err = run_tool('audit_outputs.py', '--stale')
    check('locked.md' not in out and 'rerun-locked' in err,
          '... and refresh-stale still does not run it (stderr names it)')
    write_nb('locked', executed=True, code=dict(CODE, **{'locked-3': 'y = 3'}))
    rc, out, err = gate()
    check(rc != 0, 'an added code cell is code drift too')
    write_nb('locked', executed=True)

    rc, out, err = run_tool('locks.py', 'list')
    check('chapter_lock_test/locked.md' in out and 'openrouter' in out
          and '8xa100' in out and '1h00m' in out and 'WARNING' in out,
          '`make locks` lists file, requires, runtime and status')

    manifest('other').unlink()
    rc, out, err = gate()
    check(rc != 0 and 'other.md' in err,
          'a locked notebook that was never captured fails the gate')
    rc, out, err = run_tool('audit_outputs.py', '--stale')
    check('other.md' not in out, '... but is not on the refresh-stale list')


# ── review findings (2026-11-12): one small fixture per scenario ─────

LOCK = '<!-- execution: locked; requires: a100 -->'


class Mini:
    """A one-file fixture: chapter `ch`, stem `foo`, frameworks as given."""

    def __init__(self, ch, fws=(FW,)):
        self.ch, self.fws = ch, fws
        (FIX / ch).mkdir(parents=True, exist_ok=True)
        for fw in fws:
            d = FIX / '_notebooks' / fw / ch
            d.mkdir(parents=True, exist_ok=True)
            for stem in ('foo', 'plain'):
                (d / f'{stem}.d').write_text(
                    f'_notebooks/{fw}/{ch}/{stem}.executed: \\\n    '
                    f'd2l/_blocks/{fw}/Foo.py\n', encoding='utf-8')
            shard = FIX / 'd2l' / '_blocks' / fw / 'Foo.py'
            shard.parent.mkdir(parents=True, exist_ok=True)
            if not shard.exists():
                shard.write_text('class Foo:\n    answer = 42\n')

    def md(self, cells, lock=LOCK, stem='foo'):
        body = ''.join(f'```{{.python .input #{cid}}}\n{src}\n```\n\n'
                       for cid, src in cells.items())
        (FIX / self.ch / f'{stem}.md').write_text(
            f'# T\n\n{lock}\n\nProse.\n\n{body}', encoding='utf-8')
        locks._CACHE.clear()

    def nb(self, cells, executed, fw=FW, stem='foo'):
        p = self.path(fw, stem)
        nb = notebook(executed, cells)
        nb['cells'] = [c for c in nb['cells'] if c['cell_type'] == 'code']
        for c in nb['cells']:
            if executed:
                c['outputs'] = [{'output_type': 'stream', 'name': 'stdout',
                                 'text': [f'{c["id"]} {fw}\n']}]
        p.write_text(json.dumps(nb, indent=1), encoding='utf-8')
        return p

    def path(self, fw=FW, stem='foo'):
        return FIX / '_notebooks' / fw / self.ch / f'{stem}.ipynb'

    def run(self, fw=FW, stem='foo'):
        """A real (scheduler) execution: outputs + provenance sidecar."""
        saved = run_notebooks.NOTEBOOKS_DIR
        try:
            run_notebooks.NOTEBOOKS_DIR = FIX / '_notebooks'
            run_notebooks.write_execution_provenance(self.path(fw, stem),
                                                     elapsed=5)
        finally:
            run_notebooks.NOTEBOOKS_DIR = saved

    def manifest(self, fw=FW, stem='foo'):
        return FIX / 'outputs' / fw / self.ch / f'{stem}.json'

    def stored(self, fw=FW, stem='foo'):
        m = json.loads(self.manifest(fw, stem).read_text())
        return {k: [o.get('text') for o in v['outputs']]
                for k, v in m['cells'].items()}

    def remove(self):
        shutil.rmtree(FIX / self.ch, ignore_errors=True)
        for fw in self.fws:
            shutil.rmtree(FIX / '_notebooks' / fw / self.ch, ignore_errors=True)
            shutil.rmtree(FIX / 'outputs' / fw / self.ch, ignore_errors=True)
        locks._CACHE.clear()


def test_finding1_unlock_does_not_launder():
    print('review 1: unlocking never launders store outputs')
    m = Mini('chapter_r1')
    m.md({'r1-1': 'print(1)'})
    m.nb({'r1-1': 'print(1)'}, executed=True)
    m.run()
    rc, _, _ = capture('--unlock', f'FILES={m.ch}/foo.md')
    check(rc == 0, 'locked notebook captured after its re-run')

    # Clean materialization is marked (metadata + flag file) and stamped.
    m.nb({'r1-1': 'print(1)'}, executed=False)
    m.path().with_suffix('.provenance.json').unlink()
    ok, _ = locks.materialize(FIX, FW, m.path())
    check(ok and locks.is_materialized(m.path())
          and locks.materialized_flag(m.path()).exists()
          and m.path().with_suffix('.executed').exists(),
          'a materialized notebook is marked and stamped')
    rc, out, err = capture('--unlock')
    check(rc == 0 and 'materialized from the store' in err,
          'capture never blesses a materialized notebook, even with UNLOCK')

    # gen_notebooks carries the marker through a regeneration.
    import gen_notebooks
    prev = json.loads(m.path().read_text())
    m.nb({'r1-1': 'print(1)'}, executed=False)
    gen_notebooks._merge_outputs_from_prev(m.path(), prev)
    check(locks.is_materialized(m.path()),
          'a regeneration (gen_notebooks merge) keeps the marker')

    # Unlocked now: the stamped, materialized notebook must be re-executed.
    m.md({'r1-1': 'print(1)'}, lock='')
    with fixture_root():
        items = ns.build_worklist([FW], files=ns.parse_files(f'{m.ch}/foo.md'))
    check(len(items) == 1 and items[0].unstamp,
          'after unlocking, the scheduler queues it and drops its stamp')
    before = m.manifest().read_text()
    rc, out, err = capture()
    check(rc == 0 and m.manifest().read_text() == before,
          'capture-all after unlocking leaves the store untouched')
    check(locks.clear_materialized(m.path())
          and not locks.is_materialized(m.path())
          and not locks.materialized_flag(m.path()).exists(),
          'a real execution clears the marker (run_notebooks.execute_notebook)')

    # Code drift: nothing injected, nothing stamped, gate fails.
    m.md({'r1-1': 'print(2)'})
    m.nb({'r1-1': 'print(2)'}, executed=False)
    m.path().with_suffix('.executed').touch()
    ok, msg = locks.materialize(FIX, FW, m.path())
    nb = json.loads(m.path().read_text())
    check(not ok and 'code drift' in msg
          and not m.path().with_suffix('.executed').exists()
          and not nb['cells'][0]['outputs'] and locks.is_materialized(nb),
          'code drift: no outputs injected, stamp removed, notebook marked')
    rc, out, err = gate()
    check(rc != 0 and 'chapter_r1/foo.md' in err, '... and the gate fails')
    m.md({'r1-1': 'print(2)'}, lock='')
    rc, out, err = capture('FILES=%s/foo.md' % m.ch)
    check(rc == 2 and m.manifest().read_text() == before,
          'unlocked after the drift: capture still refuses the marked notebook')
    m.remove()


def test_finding2_partial_lock_capture():
    print('review 2: a partial lock does not block the other frameworks')
    m = Mini('chapter_r2', fws=(FW, 'jax'))
    (FIX / m.ch / 'foo.md').write_text(
        '# T\n\n<!-- execution: locked; requires: a100; frameworks: pytorch -->'
        '\n\n```{.python .input #r2-1}\n%%tab pytorch\nx = 1\n```\n\n'
        '```{.python .input #r2-1}\n%%tab jax\nx = 1\n```\n', encoding='utf-8')
    (FIX / m.ch / 'plain.md').write_text(
        '# P\n\n```{.python .input #r2p-1}\nx = 1\n```\n', encoding='utf-8')
    locks._CACHE.clear()
    for fw in (FW, 'jax'):
        m.nb({'r2-1': 'x = 1'}, executed=True, fw=fw)
        m.run(fw)
    m.nb({'r2p-1': 'x = 1'}, executed=True, stem='plain')
    rc, out, err = run_tool('capture_outputs.py', '--frameworks', 'pytorch,jax',
                            f'FILES={m.ch}/foo.md')
    check(rc == 0 and m.manifest('jax').exists()
          and not m.manifest(FW).exists() and 'LOCKED' in err,
          'FILES on a partly locked file captures jax, skips pytorch, exit 0')
    rc, out, err = run_tool('capture_outputs.py', '--frameworks', 'pytorch',
                            f'FILES={m.ch}/foo.md {m.ch}/plain.md')
    check(rc == 0 and m.manifest(FW, 'plain').exists()
          and not m.manifest(FW).exists(),
          'a FILES list mixing locked and unlocked files captures the unlocked')
    rc, out, err = run_tool('capture_outputs.py', '--frameworks', 'pytorch',
                            f'FILES={m.ch}/foo.md')
    check(rc == 2, '... and refuses only when every requested pair is locked')
    mk = (TOOLS.parent / 'make' / 'store.mk').read_text()
    recipe = mk[mk.index('refresh-stale: notebooks'):mk.index('render-fresh:')]
    check('|| crc=$$?' in recipe and 'rc=$$crc' in recipe,
          'refresh-stale propagates the capture exit code')
    m.remove()


def test_finding3_and_4_rerun_vs_code_and_lib():
    print('review 3+4: a re-run must cover the current code; lib drift keeps it')
    m = Mini('chapter_r3')
    m.md({'r3-1': 'print(1)'})
    m.nb({'r3-1': 'print(1)'}, executed=True)
    m.run()
    capture('--unlock', f'FILES={m.ch}/foo.md')
    stored = m.stored()
    p = m.path()
    # 3: a cell appended after the re-run, before the capture; the
    # regeneration carried the first cell's outputs forward.
    m.md({'r3-1': 'print(1)', 'r3-2': 'print(expensive())'})
    m.nb({'r3-1': 'print(1)', 'r3-2': 'print(expensive())'}, executed=False)
    nb = json.loads(p.read_text())
    nb['cells'][0]['outputs'] = [{'output_type': 'stream', 'name': 'stdout',
                                  'text': ['r3-1 rerun\n']}]
    nb['cells'][0]['execution_count'] = 1
    p.write_text(json.dumps(nb, indent=1))
    check(locks.sidecar_status(p)[0] == 'drift',
          'a sidecar that lacks a current code cell is code drift')
    rc, out, err = capture('--unlock', f'FILES={m.ch}/foo.md')
    check(rc == 2 and 'CODE DRIFT' in err and m.stored() == stored,
          'capture UNLOCK refuses a re-run that misses an appended cell')
    ok, msg = locks.materialize(FIX, FW, p)
    check(not ok and not p.with_suffix('.executed').exists()
          and 'kept' not in msg,
          'materialize neither keeps nor stamps such a re-run')

    # 4: the lib moved between the re-run and the capture.
    m.md({'r3-1': 'print(1)'})
    m.nb({'r3-1': 'print(1)'}, executed=True)
    m.run()
    old_fp = json.loads(p.with_suffix('.provenance.json').read_text())[
        'provenance']['d2l_lib_fingerprint']
    shard = FIX / 'd2l' / '_blocks' / FW / 'Foo.py'
    saved = shard.read_text()
    shard.write_text('class Foo:\n    answer = 44  # r4\n')
    nb_before = p.read_text()
    ok, msg = locks.materialize(FIX, FW, p)
    check(ok and 'kept' in msg and p.read_text() == nb_before
          and p.with_suffix('.provenance.json').exists(),
          'lib drift after a re-run: materialize keeps the paid outputs')
    rc, out, err = capture('--unlock', f'FILES={m.ch}/foo.md')
    man = json.loads(m.manifest().read_text())
    check(rc == 0 and 'predates the current d2l lib' in err
          and man['provenance']['d2l_lib_fingerprint'] == old_fp,
          "capture UNLOCK accepts it with a warning, under the run's own lib")
    rc, out, err = run_tool('audit_outputs.py', '--json')
    check(json.loads(out).get('locked', {}).get(f'{FW}:{m.ch}/foo.md')
          == 'lib-drift', '... so the audit still reports lib-drift')
    shard.write_text(saved)
    m.remove()


def test_small_issues():
    print('review: timeout, hosted sweep, parser near-misses')
    import run_one_notebook as r1
    from runtime_env import NOTEBOOK_TIMEOUTS
    check(r1.default_timeout('chapter_x/foo.ipynb', False, {}) == 3600
          and r1.default_timeout('chapter_x/foo.ipynb', True, {}) == 21600
          and r1.default_timeout('chapter_x/foo.ipynb', True,
                                 {'D2L_LOCKED_TIMEOUT': '99'}) == 99,
          'locked re-runs default to 6 h, LOCKED_TIMEOUT overrides')
    if NOTEBOOK_TIMEOUTS:
        rel, t = next(iter(NOTEBOOK_TIMEOUTS.items()))
        check(r1.default_timeout(rel, True, {'D2L_LOCKED_TIMEOUT': '99'}) == t,
              'a NOTEBOOK_TIMEOUTS entry still wins')
    import run_hosted_full as hosted
    run, skipped = hosted.drop_locked(FIX, FW, [f'{CH}/locked.ipynb',
                                                f'{CH}/plain.ipynb'])
    check(run == [f'{CH}/plain.ipynb'] and skipped == [f'{CH}/locked.ipynb'],
          'the hosted sweep skips locked notebooks (unless --include-locked)')
    check(locks.parse_text('<!-- executive summary: costs -->\n') is None,
          '`<!-- executive summary: ... -->` is not a directive and no error')
    try:
        locks.parse_text('<!-- Execution: lockd -->\n')
        check(False, 'a malformed `<!-- execution:` comment is a hard error')
    except locks.LockError:
        check(True, 'a malformed `<!-- execution:` comment is a hard error')
    text = '<!-- executon: locked -->\n<!-- executive summary: x -->\n'
    check(locks.parse_text(text) is None
          and locks.looks_like_directive(text) == [1],
          'a typo (`executon:`) is only a lint warning')


def main():
    global FIX
    FIX = Path(tempfile.mkdtemp(prefix='d2l-locks-test-'))
    try:
        build_fixture()
        test_parser()
        test_directive_not_rendered()
        test_capture_refuses_without_unlock()
        test_scheduler_skips_and_materializes()
        test_rerun_selects_named()
        test_gate()
        test_finding1_unlock_does_not_launder()
        test_finding2_partial_lock_capture()
        test_finding3_and_4_rerun_vs_code_and_lib()
        test_small_issues()
    finally:
        shutil.rmtree(FIX, ignore_errors=True)
    print(f'\n{_passed} passed, {_failed} failed')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
