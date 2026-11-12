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
  5. One registry shard: every `DATA_HUB` entry sat in one `DATA_HUB.py`
     shard reached through `download`, so registering one new dataset staled
     every notebook that downloads anything (57 PyTorch notebooks for the
     Language Models part's keys). Each key now has its own shard, and the
     scan follows string-literal keys.

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
from capture_outputs import d2l_lib_fingerprint  # noqa: E402

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
    check({'_preamble', 'DATA_HUB.toy', 'download', 'reshape', 'nnx_resnet'}
          <= names and 'DATA_HUB' not in names,
          f'shards written: {sorted(names)}')
    pre = (shards / '_preamble.py').read_text()
    check('hand-written header' in pre and 'import regex' in pre,
          'preamble shard holds the header and the import-only block')
    check("DATA_HUB['toy']" in (shards / 'DATA_HUB.toy.py').read_text(),
          'a DATA_HUB entry lands in its own key shard')
    g = scan.shard_graph(shards)
    check(not any(s.startswith('DATA_HUB') for s in scan.closure({'download'}, g)),
          'download itself reaches no DATA_HUB entry (keys are followed '
          'from literals)')
    check('reshape = torch.reshape' in (shards / 'reshape.py').read_text(),
          'a config.ini alias gets its own shard')

    # Moving a chapter reorders the registry entries and import blocks it
    # contributes; the shards (fingerprint inputs) must not change.
    (src / 'chapter_y').mkdir()
    (src / 'chapter_y' / 'b.md').write_text(
        '# B\n\n```{.python .input #b-1}\n#@save\nimport zlib\n```\n\n'
        '```{.python .input #b-2}\n'
        "d2l.DATA_HUB['abc'] = (d2l.DATA_URL + 'abc.zip',  #@save\n"
        "                       '1' * 40)\n```\n")

    def shard_bytes(order):
        build_lib.build_library(
            src, lib, 'pytorch',
            {'lib_name': 'torch', 'simple_alias': 'reshape'}, order, [mod])
        return {p.name: p.read_bytes() for p in shards.glob('*.py')}
    before = shard_bytes(['chapter_x/a.md', 'chapter_y/b.md'])
    after = shard_bytes(['chapter_y/b.md', 'chapter_x/a.md'])
    check(before == after,
          'chapter order does not change the shards (registry/import blocks sorted)')


REGISTRY_SRC = (
    '# Data\n\n'
    # A pure registry block: two entries, one shard each.
    '```{.python .input #r-1}\n#@save\n'
    "d2l.DATA_HUB['alpha'] = (d2l.DATA_URL + 'alpha.zip',\n"
    "                         '1' * 40)\n"
    "d2l.DATA_HUB['beta'] = (d2l.DATA_URL + 'beta.zip', '2' * 40)\n```\n\n"
    # A named block that also registers a key (the GPT2_URL pattern).
    '```{.python .input #r-2}\n#@save\n'
    "GAMMA_URL = 'https://example.org/'\n"
    "d2l.DATA_HUB['gamma'] = (GAMMA_URL + 'g.bin', '3' * 40)\n```\n\n"
    # Another registry keeps one shard per container.
    '```{.python .input #r-3}\n#@save\n'
    "d2l.HF_REVISIONS['org/model'] = '4' * 40\n```\n\n"
    # A key that is not a plain file name.
    '```{.python .input #r-4}\n#@save\n'
    "d2l.DATA_HUB['odd key/1'] = (d2l.DATA_URL + 'odd.zip', '5' * 40)\n```\n\n"
    '```{.python .input #r-5}\n'
    'def download(name, folder=None):  #@save\n'
    '    return DATA_HUB[name]\n\n'
    'def download_extract(name, folder=None):  #@save\n'
    '    return download(name)\n\n'
    'def load_gamma():  #@save\n'
    "    return d2l.download('gamma')\n\n"
    'class Embedding:  #@save\n'
    '    def __init__(self, name):\n'
    '        self.path = d2l.download_extract(name)\n```\n')

REGISTRY_NOTEBOOKS = {
    # literal key in a download call
    'nb_alpha': "d2l.download('alpha')",
    # key reached through a library helper's literal
    'nb_gamma': 'd2l.load_gamma()',
    # literal passed to a class whose shard downloads a variable key
    'nb_embed': "emb = d2l.Embedding('beta')",
    # a URL download names no key
    'nb_url': "d2l.download(d2l.DATA_URL + 'timemachine.txt')",
    # reads the other registry
    'nb_hf': "rev = d2l.HF_REVISIONS['org/model']",
}


def test_data_hub_key_shards(tmp):
    print('per-key DATA_HUB shards (build_lib + scan_d2l_usage)')
    check(build_lib.registry_shard_name('DATA_HUB', 'glove.6b.100d')
          == 'DATA_HUB.glove.6b.100d', 'a plain key names its shard as is')
    odd = build_lib.registry_shard_name('DATA_HUB', 'odd key/1')
    check(odd.startswith('DATA_HUB.odd_key_1-')
          and odd != build_lib.registry_shard_name('DATA_HUB', 'odd_key_1'),
          f'an unsafe key is sanitised and hashed: {odd}')
    entries, rest = build_lib.registry_entries(
        "GAMMA_URL = 'u'\nd2l.DATA_HUB['gamma'] = (GAMMA_URL, 's')\n")
    check([n for n, _ in entries] == ['DATA_HUB.gamma'] and rest,
          'registry_entries splits a named block into entry + rest')
    check(build_lib.registry_entries("d2l.DATA_HUB[k] = ('u', 's')")
          == ([], True), 'a computed key is not split off')

    src = tmp / 'src'
    (src / 'chapter_lib').mkdir(parents=True)
    (src / 'chapter_lib' / 'data.md').write_text(REGISTRY_SRC)
    lib = tmp / 'd2l' / 'torch.py'
    lib.parent.mkdir(parents=True)
    lib.write_text('DATA_HUB = dict()\nHF_REVISIONS = dict()\n')
    files = ['chapter_lib/data.md']

    def build():
        build_lib.build_library(src, lib, 'pytorch', {'lib_name': 'torch'},
                                files)
    build()
    shards = tmp / 'd2l' / '_blocks' / 'pytorch'
    names = {p.stem for p in shards.glob('*.py')}
    want = {'DATA_HUB.alpha', 'DATA_HUB.beta', 'DATA_HUB.gamma', odd,
            'GAMMA_URL', 'HF_REVISIONS', 'download', 'download_extract',
            'load_gamma', 'Embedding'}
    check(want <= names and 'DATA_HUB' not in names,
          f'one shard per key, none for the container: {sorted(names)}')
    alpha = (shards / 'DATA_HUB.alpha.py').read_text()
    check("['alpha']" in alpha and "['beta']" not in alpha,
          'a two-entry block is split into two shards')
    check("['gamma']" in (shards / 'DATA_HUB.gamma.py').read_text()
          and 'GAMMA_URL = ' in (shards / 'GAMMA_URL.py').read_text(),
          'a named block keeps its shard; its entry gets a key shard')

    for stem, line in REGISTRY_NOTEBOOKS.items():
        (src / 'chapter_lib' / f'{stem}.md').write_text(
            f'# {stem}\n\n```{{.python .input #{stem}-1}}\n%%tab pytorch\n'
            f'{line}\n```\n')

    def scan_deps():
        r = subprocess.run(
            [sys.executable, str(TOOLS / 'scan_d2l_usage.py'),
             '--source', str(src), '--output-dir', '_notebooks',
             '--shard-dir', 'd2l/_blocks'],
            cwd=tmp, capture_output=True, text=True)
        check(r.returncode == 0, f'scanner runs ({r.stderr.strip()[-200:]})')
        out = {}
        for stem in REGISTRY_NOTEBOOKS:
            d = tmp / '_notebooks' / 'pytorch' / 'chapter_lib' / f'{stem}.d'
            out[stem] = (
                {Path(t).name[:-3] for t in d.read_text().split()
                 if '_blocks' in t},
                d2l_lib_fingerprint(tmp, d))
        return out
    deps = scan_deps()

    def keys(stem):
        return {n for n in deps[stem][0] if n.startswith('DATA_HUB')}
    check(keys('nb_alpha') == {'DATA_HUB.alpha'},
          f"download('alpha') depends on that key only: {keys('nb_alpha')}")
    check(keys('nb_gamma') == {'DATA_HUB.gamma'}
          and 'GAMMA_URL' in deps['nb_gamma'][0],
          'a helper\'s literal key is followed through the closure, '
          'and the entry pulls in GAMMA_URL')
    check(keys('nb_embed') == {'DATA_HUB.beta'},
          'a literal handed to a class that downloads a variable key counts')
    check(keys('nb_url') == set(), 'a URL download depends on no key')
    check('HF_REVISIONS' in deps['nb_hf'][0],
          'other registries keep one shard per container')

    # The point of it all: registering a new key changes no existing shard
    # and no fingerprint of a notebook that does not name it.
    before = {p.name: p.read_bytes() for p in shards.glob('*.py')}
    (src / 'chapter_lib' / 'more.md').write_text(
        '# More\n\n```{.python .input #m-1}\n%%tab pytorch\n#@save\n'
        "d2l.DATA_HUB['delta'] = (d2l.DATA_URL + 'delta.zip', '6' * 40)\n"
        '```\n')
    files.append('chapter_lib/more.md')
    build()
    after = {p.name: p.read_bytes() for p in shards.glob('*.py')}
    check(set(after) - set(before) == {'DATA_HUB.delta.py'}
          and all(after[n] == b for n, b in before.items()),
          'a new key adds one shard and leaves every other shard byte-identical')
    deps2 = scan_deps()
    check(all(deps2[s][1] == deps[s][1] for s in REGISTRY_NOTEBOOKS),
          'no existing notebook fingerprint changes when a key is added')
    # ... and editing an entry stales exactly the notebooks that name it.
    (src / 'chapter_lib' / 'data.md').write_text(
        REGISTRY_SRC.replace("'2' * 40", "'7' * 40"))
    build()
    deps3 = scan_deps()
    changed = {s for s in REGISTRY_NOTEBOOKS if deps3[s][1] != deps[s][1]}
    check(changed == {'nb_embed'},
          f'editing the beta entry stales only its user: {sorted(changed)}')


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
        test_data_hub_key_shards(tmp / 'registry')
        test_unmanifested_notebook(tmp / 'audit')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{_passed} passed, {_failed} failed')
    return 1 if _failed else 0


if __name__ == '__main__':
    sys.exit(main())
