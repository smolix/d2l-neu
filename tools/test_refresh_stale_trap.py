#!/usr/bin/env python3
"""End-to-end regression test for the freshness-disagreement ("refresh-stale") trap.

Background (docs/build-system.md §3.3): the committed `outputs/` store keys each
notebook's outputs on fingerprints (code_fingerprint + d2l_lib_fingerprint), while
the scheduler tracks execution with mtime-based `.executed` stamps. These can
DISAGREE: if a `#@save` library symbol a notebook imports is edited but the
notebook's own SOURCE is unchanged, the stamp stays "fresh" (byte-identical
`.ipynb`, newer stamp) so the scheduler never re-runs it — yet its store outputs
are now lib-stale. A naive `make capture-outputs` then blesses the OLD outputs
under the NEW fingerprint and `audit_outputs --verify-fresh` passes: a FALSE GREEN.
This actually shipped stale word-level BLEU into the attention chapter once.

This test reproduces the trap on a throwaway temp fixture (no GPU, no venv, fast)
and asserts the three defenses hold together:

  (a) `audit_outputs.py` flags the lib-stale-but-stamp-fresh notebook
      (--stale-stamps lists its stamp, --json marks it hard_stale, --verify-fresh
      exits non-zero);
  (b) the `refresh-stale` mechanic — `rm` of the `--stale-stamps` paths — removes
      the stamp so a subsequent run is FORCED to re-execute;
  (c) the capture-side guard REFUSES to bless the stale outputs (non-zero exit,
      store left untouched), and — the zero-false-positive half — ACCEPTS the
      capture once the notebook is re-executed under the current lib.

Run directly (`python tools/test_refresh_stale_trap.py`) or via `make test-trap`.
The real committed store is never touched — everything happens under a tmp dir.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import capture_outputs as cap          # noqa: E402
import run_notebooks                   # noqa: E402

FW = 'pytorch'
CHAPTER = 'chapter_trap_test'
STEM = 'foo'
CELL_ID = 'foo-lib-stale-cell-1'

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


def run_tool(script, *args):
    """Run a repo tool against the fixture; return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(TOOLS / script),
           '--project-dir', str(FIX), *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def build_fixture(root: Path, shard_body: str):
    """Lay down a minimal one-notebook fixture: source .md, generated .ipynb + .d,
    a d2l shard the notebook depends on, and a fresh `.executed` stamp."""
    (root / CHAPTER).mkdir(parents=True, exist_ok=True)
    # Plain CPU source (no GPU keywords) so verify-fresh classifies it 'cpu'.
    (root / CHAPTER / f'{STEM}.md').write_text(
        '# Trap fixture\n\nA tiny notebook that imports a d2l symbol.\n',
        encoding='utf-8')

    shard = root / 'd2l' / '_blocks' / FW / 'MTFraEng.py'
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(shard_body, encoding='utf-8')

    nb_dir = root / '_notebooks' / FW / CHAPTER
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb = {
        'cells': [{
            'cell_type': 'code',
            'id': CELL_ID,
            'metadata': {},
            'source': ['import d2l\n', 'print(d2l.MTFraEng.__name__)\n'],
            'outputs': [{
                'output_type': 'stream', 'name': 'stdout',
                'text': ['MTFraEng\n'],
            }],
            'execution_count': 1,
        }],
        'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5,
    }
    (nb_dir / f'{STEM}.ipynb').write_text(
        json.dumps(nb, indent=1), encoding='utf-8')
    # .d file: the notebook depends on the MTFraEng shard (Make-prereq syntax).
    (nb_dir / f'{STEM}.d').write_text(
        f'_notebooks/{FW}/{CHAPTER}/{STEM}.executed: \\\n'
        f'    d2l/_blocks/{FW}/MTFraEng.py\n', encoding='utf-8')
    (nb_dir / f'{STEM}.executed').touch()


def record_sidecar(root: Path):
    """Simulate a real execution recording its provenance sidecar, exercising the
    actual writer (tools/run_notebooks.write_execution_provenance) against the
    fixture by pointing its NOTEBOOKS_DIR at the tmp tree."""
    nb_path = root / '_notebooks' / FW / CHAPTER / f'{STEM}.ipynb'
    saved = run_notebooks.NOTEBOOKS_DIR
    try:
        run_notebooks.NOTEBOOKS_DIR = root / '_notebooks'
        run_notebooks.write_execution_provenance(nb_path)
    finally:
        run_notebooks.NOTEBOOKS_DIR = saved
    return nb_path.with_suffix('.provenance.json')


def manifest_path(root: Path):
    return root / 'outputs' / FW / CHAPTER / f'{STEM}.json'


def lib_fp_of(root: Path):
    d = root / '_notebooks' / FW / CHAPTER / f'{STEM}.d'
    return cap.build_provenance(root, FW, d)['d2l_lib_fingerprint']


def main():
    global FIX
    tmp = Path(tempfile.mkdtemp(prefix='d2l-trap-test-'))
    FIX = tmp
    try:
        # ── Phase A: fixture at lib V1; capture + record sidecar under V1 ──
        build_fixture(tmp, shard_body='class MTFraEng:  # V1 word-level\n    pass\n')
        fp_v1 = lib_fp_of(tmp)

        rc, out, err = run_tool('capture_outputs.py', '--frameworks', FW,
                                'FILES=%s/%s.md' % (CHAPTER, STEM))
        check(rc == 0 and manifest_path(tmp).exists(),
              'initial capture under lib V1 succeeds (no sidecar yet → back-compat)')
        man_v1 = manifest_path(tmp).read_text()
        check(json.loads(man_v1)['provenance']['d2l_lib_fingerprint'] == fp_v1,
              'store manifest recorded under lib V1 fingerprint')

        sidecar = record_sidecar(tmp)
        check(sidecar.exists()
              and json.loads(sidecar.read_text())['provenance']['d2l_lib_fingerprint'] == fp_v1,
              'execution recorded a provenance sidecar under lib V1')

        # Sanity: with a MATCHING sidecar, re-capture is clean (no false positive).
        rc, out, err = run_tool('capture_outputs.py', '--frameworks', FW,
                                'FILES=%s/%s.md' % (CHAPTER, STEM))
        check(rc == 0 and manifest_path(tmp).read_text() == man_v1,
              'fresh sidecar → re-capture passes and is byte-idempotent')

        # ── Phase B: edit the d2l lib (V1 → V2); notebook SOURCE untouched ──
        (tmp / 'd2l' / '_blocks' / FW / 'MTFraEng.py').write_text(
            'class MTFraEng:  # V2 byte-level BPE — behaviour changed!\n    pass\n',
            encoding='utf-8')
        fp_v2 = lib_fp_of(tmp)
        check(fp_v2 != fp_v1, 'lib edit moved the d2l fingerprint (V1 != V2)')

        stamp = tmp / '_notebooks' / FW / CHAPTER / f'{STEM}.executed'
        check(stamp.exists(), 'the .executed stamp is still present (mtime-fresh) — '
                              'the scheduler would SKIP re-execution: the trap')

        # (a) audit flags the lib-stale-but-stamp-fresh notebook.
        rc, out, err = run_tool('audit_outputs.py', '--stale-stamps')
        stamps = [ln for ln in out.splitlines() if ln.strip()]
        want_stamp = f'_notebooks/{FW}/{CHAPTER}/{STEM}.executed'
        check(any(s.endswith(want_stamp) for s in stamps),
              '(a) audit --stale-stamps lists the stale stamp')

        rc, out, err = run_tool('audit_outputs.py', '--json')
        report = json.loads(out)
        check(any(STEM in h for h in report['hard_stale']),
              '(a) audit --json marks it hard_stale (inline output would be wrong)')

        rc, out, err = run_tool('audit_outputs.py', '--verify-fresh')
        check(rc != 0, '(a) audit --verify-fresh exits non-zero (render gate blocks)')

        # (b) refresh-stale mechanic: rm the audit's stamp → re-execution forced.
        for s in stamps:
            p = tmp / s
            if p.exists():
                p.unlink()
        check(not stamp.exists(),
              '(b) removing the --stale-stamps path clears the stamp (forces re-run)')

        # (c) capture-side guard: the sidecar still says V1, current lib is V2 →
        # blessing would write the new fingerprint over old-lib outputs → REFUSE.
        before = manifest_path(tmp).read_text()
        rc, out, err = run_tool('capture_outputs.py', '--frameworks', FW,
                                'FILES=%s/%s.md' % (CHAPTER, STEM))
        check(rc != 0, '(c) capture REFUSES the stale re-bless (non-zero exit)')
        check('REFUSING' in err or 'REFUSING' in out,
              '(c) capture prints an explicit refusal')
        check(manifest_path(tmp).read_text() == before,
              '(c) store manifest left UNTOUCHED by the refused capture')

        # (c-force) the documented escape hatch still lets you override.
        rc, out, err = run_tool('capture_outputs.py', '--frameworks', FW, '--force',
                                'FILES=%s/%s.md' % (CHAPTER, STEM))
        check(rc == 0, '(c) --force overrides the guard (escape hatch)')
        # restore the pre-force manifest so the next phase starts from V1 outputs
        manifest_path(tmp).write_text(before)

        # ── Phase C: genuine re-execution under V2 → sidecar rewritten → passes ──
        record_sidecar(tmp)   # lib is now V2, so the sidecar records V2
        check(json.loads(sidecar.read_text())['provenance']['d2l_lib_fingerprint'] == fp_v2,
              'a real re-execution rewrites the sidecar under lib V2')
        rc, out, err = run_tool('capture_outputs.py', '--frameworks', FW,
                                'FILES=%s/%s.md' % (CHAPTER, STEM))
        check(rc == 0, '(c) capture ACCEPTS the re-blessed outputs (no false positive)')
        check(json.loads(manifest_path(tmp).read_text())['provenance']['d2l_lib_fingerprint'] == fp_v2,
              '(c) store manifest now carries the current lib V2 fingerprint')

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f'\n{_passed} passed, {_failed} failed')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
