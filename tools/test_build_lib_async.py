#!/usr/bin/env python3
"""`async def` in #@save blocks (tools/build_lib.py).

18.7 saves coroutine functions (`ChatAPI.achat`, `chat_batch`). Every place
that recognises `def` must recognise `async def` too: block collection
(standalone marker, `@d2l.add_to_class`, inline marker), the "Defined in"
docstring label, the defined-name and shard-name scans, the class-method
dedup and the block dedup. A missed site truncates a body, files the block in
the `_preamble` shard, or leaves a method outside its class.

Runs on a throwaway fixture (no GPU, no venv):
`python3 tools/test_build_lib_async.py`, or `make test-lm`.
"""

import ast
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import build_lib  # noqa: E402

LABEL = 'sec_async_fixture'

SOURCE = f'''\
# Async fixture
:label:`{LABEL}`

```{{.python .input #async-class}}
%%tab pytorch
class C:  #@save
    """A client."""
    def __init__(self):
        self.n = 0
```

```{{.python .input #async-standalone}}
%%tab pytorch
#@save
async def f(x):
    """Standalone marker before an async def."""
    import asyncio  # not in the d2l library preamble

    await asyncio.sleep(0)
    return x + 1
```

```{{.python .input #async-method}}
%%tab pytorch
@d2l.add_to_class(C)  #@save
async def m(self, x):
    """A coroutine method added to C."""
    import asyncio  # not in the d2l library preamble

    await asyncio.sleep(0)
    return x * self.n
```

```{{.python .input #async-inline}}
%%tab pytorch
async def g(xs):  #@save
    """Inline marker on an async def."""
    import asyncio  # not in the d2l library preamble

    return await asyncio.gather(*(f(x) for x in xs))
```
'''

# The same definitions as plain functions: the async support must not change
# what a `def` produces.
SYNC_SOURCE = SOURCE.replace('async def', 'def').replace(
    'await asyncio.sleep(0)', 'asyncio.sleep(0)').replace(
    'return await asyncio.gather', 'return asyncio.gather')


class AsyncDefTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='d2l-build-lib-async-'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, text, name='lib'):
        root = self.tmp / name
        (root / 'chapter_x').mkdir(parents=True)
        (root / 'chapter_x' / 'a.md').write_text(text)
        lib = root / 'd2l' / 'torch.py'
        lib.parent.mkdir(parents=True)
        lib.write_text('import os\n')
        build_lib.build_library(root, lib, 'pytorch', {'lib_name': 'torch'},
                                ['chapter_x/a.md'])
        shards = root / 'd2l' / '_blocks' / 'pytorch'
        return lib.read_text(), {p.stem: p.read_text()
                                 for p in shards.glob('*.py')}

    def test_save_blocks_collects_whole_bodies(self):
        code = '\n'.join(b for b, _, _ in build_lib.extract_save_blocks(
            self._write(SOURCE), 'pytorch'))
        for needle in ('return x + 1', 'return x * self.n',
                       'return await asyncio.gather'):
            self.assertIn(needle, code, f'body cut short before {needle!r}')

    def test_library_and_shards(self):
        lib, shards = self.build(SOURCE)
        tree = ast.parse(lib)  # the emitted module is valid Python
        top = {n.name: n for n in tree.body
               if isinstance(n, (ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef))}
        self.assertIsInstance(top.get('f'), ast.AsyncFunctionDef)
        self.assertIsInstance(top.get('g'), ast.AsyncFunctionDef)
        self.assertNotIn('m', top, 'the method stayed outside its class')
        methods = {n.name: n for n in top['C'].body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIsInstance(methods.get('m'), ast.AsyncFunctionDef,
                              'add_to_class merged m into C')
        for name in ('f', 'g', 'C'):
            self.assertIn(name, shards, f'{name} has its own shard')
        self.assertIn('async def m', shards['C'])
        pre = shards.get('_preamble', '')
        self.assertNotIn('async def', pre,
                         'an async block fell into the _preamble shard')
        for fn in (top['f'], top['g'], methods['m'], top['C']):
            doc = ast.get_docstring(fn)
            self.assertIn(f'Defined in :numref:`{LABEL}`', doc or '',
                          f'{fn.name} docstring carries the label')

    def test_sync_output_unchanged(self):
        # A plain `def` source builds exactly as the pattern change predicts:
        # the async build differs from it only by the `async`/`await` words.
        lib_async, shards_async = self.build(SOURCE, 'a')
        lib_sync, shards_sync = self.build(SYNC_SOURCE, 's')
        strip = (lambda t: t.replace('async def', 'def')
                 .replace('await asyncio.sleep(0)', 'asyncio.sleep(0)')
                 .replace('return await asyncio.gather',
                          'return asyncio.gather'))
        self.assertEqual(strip(lib_async), lib_sync)
        self.assertEqual({k: strip(v) for k, v in shards_async.items()},
                         shards_sync)

    def test_name_scans(self):
        block = 'async def h():\n    pass\n\nclass D:\n    pass\n'
        self.assertEqual(build_lib._block_shard_names(block), ['h', 'D'])
        self.assertEqual(build_lib._block_defined_names(block), {'h', 'D'})

    def test_method_dedup_keeps_last(self):
        cls = ('class E:\n'
               '    async def run(self):\n'
               '        return 1\n\n'
               '    def other(self):\n'
               '        return 0\n\n'
               '    async def run(self):\n'
               '        return 2\n')
        out = build_lib._dedup_class_methods(cls)
        self.assertEqual(out.count('def run'), 1)
        self.assertIn('return 2', out)
        self.assertNotIn('return 1', out)

    def test_block_dedup_keeps_last(self):
        blocks = [('async def k():\n    return 1', '', 'a'),
                  ('async def k():\n    return 2', '', 'b')]
        out = build_lib.deduplicate_blocks(blocks)
        self.assertEqual([c for c, _, _ in out], ['async def k():\n    return 2'])

    def test_drop_unresolved_names_async(self):
        blocks = [('@d2l.add_to_class(d2l.Missing)\nasync def z(self):\n'
                   '    pass', '', 'x.md')]
        self.assertEqual(build_lib.drop_unresolved_blocks(blocks, set()), [])

    def _write(self, text):
        p = self.tmp / 'src.md'
        p.write_text(text)
        return p


if __name__ == '__main__':
    unittest.main()
