#!/usr/bin/env python3
"""Regression tests for the library-dependency side of the freshness gate.

docs/build-system.md §3.4 (#94). Three defects let a library or source edit
leave a notebook marked fresh:

  1. Shard naming: `build_lib.py` named a shard after the alphabetically first
     name a block defined, nested helpers included, so the `plot` block landed
     in `has_one_axis.py` and no notebook depended on `d2l.plot`.
  2. Transitivity: `scan_d2l_usage.py` listed only the symbols a notebook names
     itself, so a notebook calling `d2l.Trainer` did not depend on the shards
     `Trainer` calls (`ProgressBoard`, the JAX `_trainer_train_step`, ...).
  3. No manifest: `audit_outputs.py` iterated manifests only, so a notebook
     with code cells that was never captured was never reported.
  4. Unsharded code: the hand-written library header, import-only #@save
     blocks, `d2l.DATA_HUB[...]` entries, config.ini aliases and
     d2l/nnx_resnet.py had no shard, so editing them flagged nothing.

Everything runs on throwaway fixtures under a tmp dir (no GPU, no venv):
`python3 tools/test_freshness_deps.py` or `make test-trap`.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import build_lib                     # noqa: E402
import scan_d2l_usage as scan        # noqa: E402

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


PLOT_BLOCK = '''\
def plot(X, Y=None, axes=None):
    """Plot data points."""

    def has_one_axis(X):
        return hasattr(X, "ndim") and X.ndim == 1

    if has_one_axis(X):
        X = [X]
    set_axes(axes)'''

TWO_TOP_LEVEL = '''\
def train_lm(model, data):
    def loss_fn(params):
        return 0.0
    return loss_fn

@jax.jit
def recall_step(model, X):
    return model(X)
WIKI_URL: str = 'https://example.org'
'''


def test_shard_names():
    print('shard naming (build_lib._block_shard_names)')
    check(build_lib._block_shard_names(PLOT_BLOCK) == ['plot'],
          'nested helper does not name the shard: plot block -> [plot]')
    check(build_lib._block_shard_names(TWO_TOP_LEVEL)
          == ['train_lm', 'recall_step', 'WIKI_URL'],
          'every module-level def/class/assignment gets a shard, in order')
    check(build_lib._block_shard_names('class A(d2l.Module):\n    x = 1\n')
          == ['A'], 'class attributes are not module-level names')


TRAINER_SHARD = '''\
class Trainer(d2l.HyperParameters):
    def fit(self, model, data):
        board = ProgressBoard()          # bare module global
        for batch in data:
            _trainer_train_step(model, batch)
        plot = 3                         # a local that shadows a symbol
        return plot + Averager().n + d2l.num_gpus()
'''


def test_shard_references():
    print('shard references (scan_d2l_usage.shard_references)')
    refs = scan.shard_references(TRAINER_SHARD)
    for name in ('HyperParameters', 'ProgressBoard', '_trainer_train_step',
                 'Averager', 'num_gpus'):
        check(name in refs, f'Trainer shard references {name}')
    check('plot' not in refs, 'a local variable is not a reference')
    check('d2l' not in refs, 'the module alias d2l is not a symbol')
    check('model' not in refs and 'batch' not in refs,
          'parameters and loop variables are not references')


def test_closure_end_to_end(tmp):
    print('transitive .d files (scan_d2l_usage main)')
    shards = tmp / 'd2l' / '_blocks' / 'pytorch'
    shards.mkdir(parents=True)
    (shards / 'Trainer.py').write_text(TRAINER_SHARD)
    (shards / 'ProgressBoard.py').write_text(
        'class ProgressBoard:\n    def draw(self):\n        set_axes(None)\n')
    (shards / 'set_axes.py').write_text('def set_axes(axes):\n    pass\n')
    (shards / 'Averager.py').write_text('class Averager:\n    n = 0\n')
    (shards / 'unrelated.py').write_text('def unrelated():\n    pass\n')
    # The preamble shard: every notebook depends on it, and its own names
    # (here a call to `unrelated`) do not pull anything in.
    (shards / '_preamble.py').write_text('import os\nunrelated()\n')
    ch = tmp / 'chapter_fixture'
    ch.mkdir()
    (ch / 'nb.md').write_text(
        '# Fixture\n\n```{.python .input #nb-1}\n%%tab pytorch\n'
        'trainer = d2l.Trainer(max_epochs=1)\n```\n')
    r = subprocess.run(
        [sys.executable, str(TOOLS / 'scan_d2l_usage.py'), '--source', str(tmp),
         '--output-dir', str(tmp / '_notebooks'),
         '--shard-dir', str(tmp / 'd2l' / '_blocks')],
        capture_output=True, text=True)
    check(r.returncode == 0, f'scanner runs ({r.stderr.strip()[-200:]})')
    d = (tmp / '_notebooks' / 'pytorch' / 'chapter_fixture' / 'nb.d')
    deps = {Path(t).stem for t in d.read_text().split()
            if '_blocks' in t} if d.exists() else set()
    check(deps == {'Trainer', 'ProgressBoard', 'set_axes', 'Averager',
                   '_preamble'},
          f'.d holds the closure, two levels deep, plus the preamble, '
          f'nothing else: {sorted(deps)}')


def test_build_lib_extra_shards(tmp):
    print('preamble, DATA_HUB, alias and module shards (build_lib)')
    src = tmp / 'src'
    (src / 'chapter_x').mkdir(parents=True)
    (src / 'chapter_x' / 'a.md').write_text(
        '# A\n\n```{.python .input #a-1}\n#@save\nimport regex\n```\n\n'
        '```{.python .input #a-2}\n'
        "d2l.DATA_HUB['toy'] = (d2l.DATA_URL + 'toy.zip',  #@save\n"
        "                       '0' * 40)\n```\n\n"
        '```{.python .input #a-3}\n'
        'def download(name):  #@save\n    return DATA_HUB[name]\n```\n')
    out = tmp / 'out' / 'd2l'
    out.mkdir(parents=True)
    lib = out / 'torch.py'
    lib.write_text("DATA_HUB = dict()\nimport logging  # hand-written header\n")
    mod = tmp / 'nnx_resnet.py'
    mod.write_text('class ResNet50:\n    pass\n')
    build_lib.build_library(
        src, lib, 'pytorch', {'lib_name': 'torch', 'simple_alias': 'reshape'},
        ['chapter_x/a.md'], [mod])
    shards = out / '_blocks' / 'pytorch'
    names = {p.stem for p in shards.glob('*.py')}
    check({'_preamble', 'DATA_HUB', 'download', 'reshape', 'nnx_resnet'}
          <= names, f'shards written: {sorted(names)}')
    pre = (shards / '_preamble.py').read_text()
    check('hand-written header' in pre and 'import regex' in pre,
          'preamble shard holds the header and the import-only block')
    check("DATA_HUB['toy']" in (shards / 'DATA_HUB.py').read_text(),
          'a DATA_HUB entry lands in the DATA_HUB shard')
    g = scan.shard_graph(shards)
    check('DATA_HUB' in scan.closure({'download'}, g),
          'download reaches the DATA_HUB entries through the closure')
    check('reshape = torch.reshape' in (shards / 'reshape.py').read_text(),
          'a config.ini alias gets its own shard')


def test_unmanifested_notebook(tmp):
    print('notebook with code cells but no manifest (audit_outputs)')
    ch = 'chapter_fixture'
    (tmp / ch).mkdir(parents=True, exist_ok=True)
    (tmp / ch / 'new.md').write_text(
        '# New section\n\n```{.python .input #new-1}\n%%tab pytorch\n'
        'print(1 + 1)\n```\n')
    # A section with prose only, and one with code for another framework
    # only: neither needs a pytorch manifest.
    (tmp / ch / 'prose.md').write_text('# Prose\n\nNo code here.\n')
    (tmp / ch / 'jaxonly.md').write_text(
        '# J\n\n```{.python .input #j-1}\n%%tab jax\nprint(2)\n```\n')
    # No _notebooks/ at all: the rule reads the sources (fresh clone).
    (tmp / 'outputs' / 'pytorch').mkdir(parents=True, exist_ok=True)

    def run(*args):
        return subprocess.run(
            [sys.executable, str(TOOLS / 'audit_outputs.py'),
             '--project-dir', str(tmp), '--frameworks', 'pytorch', *args],
            capture_output=True, text=True)

    r = run('--stale')
    lines = r.stdout.split()
    check(lines == [f'{ch}/new.md'],
          f'--stale lists the uncaptured notebook only, without any '
          f'generated notebooks: {lines}')
    r = run('--json')
    rep = json.loads(r.stdout)
    check(rep['no_manifest'] == [f'{ch}/new.md [pytorch]'],
          '--json reports it under no_manifest')
    r = run('--verify-fresh')
    check(r.returncode != 0,
          '--verify-fresh fails on it (a cpu notebook: runnable on any host)')


def main():
    test_shard_names()
    test_shard_references()
    tmp = Path(tempfile.mkdtemp(prefix='d2l-freshness-deps-'))
    try:
        test_closure_end_to_end(tmp / 'closure')
        test_build_lib_extra_shards(tmp / 'buildlib')
        test_unmanifested_notebook(tmp / 'audit')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{_passed} passed, {_failed} failed')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
