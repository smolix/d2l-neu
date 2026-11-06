#!/usr/bin/env python3
"""The standard notebook preamble: one imports cell per framework, book-wide.

Single source of truth for the "teach the imports once, then hide them"
convention (docs/code-guide.md §2.1, docs/build-system.md §6.10):

- Every notebook's first code cell, per framework, is exactly
  ``CANONICAL[fw]`` (byte-identical). Imports the notebook needs beyond it
  go in a second, short "extras" cell right after it.
- The book renders (HTML via ``d2l_preprocess.emit_qmd``, PDF via
  ``gen_pdf.emit_pdf_qmd``) drop a cell whose text equals ``CANONICAL[fw]``;
  the notebooks (``gen_notebooks.py``, hence Colab) keep it. A cell that
  differs in any byte still shows: unique preambles stay visible.
- The installation page (``TEACH_FILE``, section ``TEACH_LABEL``) shows the
  four blocks once and explains them. ``lint_source.py`` checks that the
  page quotes them verbatim.

The blocks were chosen from a survey of every imports cell in the book
(2026-11): each holds what most notebooks of that framework import, plus
NumPy and ``math``. ``np`` is NumPy in every framework; MXNet's array module
is ``mnp`` (docs/code-guide.md §2). ``%matplotlib inline`` is not needed:
the Jupyter kernel's default backend (``matplotlib_inline``) is the inline
one, in JupyterLab, Colab, and the ``nbconvert`` execution path alike.

Also used by ``normalize_imports.py`` (the rewrite) and ``lint_source.py``.
"""

import ast

CANONICAL = {
    'pytorch': '''\
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l''',
    'jax': '''\
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l''',
    'tensorflow': '''\
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l''',
    'mxnet': '''\
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()''',
}

# Statements other than imports that a canonical block runs, per framework.
CANONICAL_CALLS = {'mxnet': {'npx.set_np()'}}

# `from d2l import <module> as d2l` → framework.
D2L_MODULE_TO_FW = {'torch': 'pytorch', 'jax': 'jax',
                    'tensorflow': 'tensorflow', 'mxnet': 'mxnet'}

# Where the book shows and explains the preamble (never hidden there: the
# page quotes the blocks in non-executed ```py fences).
TEACH_FILE = 'chapter_installation/index.md'
TEACH_LABEL = 'sec_preamble'

# Imports cells that are deliberately not the standard preamble, as
# `<cell id>@<framework>` (or a bare `<cell id>` for every framework's
# variant). lint_source.py --preamble does not warn on them; the book shows
# them, as it shows any cell that differs from the preamble. Empty: settings
# that must precede the framework's import go in a setup cell before the
# preamble, and MXNet code writes its arrays as `mnp` (docs/code-guide.md
# §2.1).
UNIQUE_PREAMBLES = set()

# Lines a normalizable imports cell may hold besides import statements.
DROPPABLE_LINES = {'%matplotlib inline'}

def _strip(text):
    return text.strip('\n')


def is_unique(cell_id, fw):
    """True iff the imports cell `cell_id` of framework `fw` is registered
    in UNIQUE_PREAMBLES."""
    return (cell_id in UNIQUE_PREAMBLES
            or f'{cell_id}@{fw}' in UNIQUE_PREAMBLES)


def is_canonical(code, fw=None):
    """True iff `code` (a cell body without its %%tab line, as a str or a
    list of lines) is exactly a canonical block — of `fw`, or of any
    framework when fw is None. Leading/trailing blank lines are ignored;
    everything else must match byte for byte."""
    if isinstance(code, (list, tuple)):
        code = '\n'.join(code)
    code = _strip(code)
    if fw is not None:
        return code == CANONICAL[fw]
    return code in CANONICAL.values()


def imports_cell_framework(code):
    """Framework of an imports cell (one whose top level imports
    `from d2l import <fw> as d2l`), else None. Cells run through a cell
    magic (`%%writefile` …) are never imports cells: the import there is
    the text of a script, not this notebook's."""
    if isinstance(code, (list, tuple)):
        code = '\n'.join(code)
    for line in code.split('\n'):
        s = line.strip()
        if s.startswith('%%') and not s.startswith('%%tab'):
            return None
    try:
        tree = ast.parse(_without_magics(code))
    except SyntaxError:
        return None
    for node in tree.body:
        if (isinstance(node, ast.ImportFrom) and node.module == 'd2l'
                and node.level == 0):
            for a in node.names:
                if a.asname == 'd2l' and a.name in D2L_MODULE_TO_FW:
                    return D2L_MODULE_TO_FW[a.name]
    return None


def _without_magics(code):
    return '\n'.join('' if l.lstrip().startswith(('%', '!')) else l
                     for l in code.split('\n'))


# ──────────────────────────────────────────────────────────
# Import bindings
# ──────────────────────────────────────────────────────────

class Binding:
    """One name an import statement binds.

    kind 'import'  — `import mod` / `import mod as alias`
    kind 'from'    — `from mod import name` / `... as alias`
    `target` is the dotted object the name refers to, so equivalent spellings
    compare equal: `import jax.numpy as jnp` and `from jax import numpy as
    jnp` both bind jnp → 'jax.numpy'.
    """
    __slots__ = ('kind', 'module', 'name', 'asname', 'bound', 'target')

    def __init__(self, kind, module, name, asname):
        self.kind, self.module, self.name, self.asname = (
            kind, module, name, asname)
        if kind == 'import':
            self.bound = asname or module.split('.')[0]
            self.target = module if asname else module.split('.')[0]
        else:
            self.bound = asname or name
            self.target = f'{module}.{name}'

    def loads_submodule(self):
        """`import a.b` (no alias): binds a, and also loads a.b, which no
        other spelling of the binding replaces."""
        return self.kind == 'import' and not self.asname and '.' in self.module


def bindings(code):
    """List of Binding for the import statements in `code`; raises
    SyntaxError."""
    tree = ast.parse(_without_magics(code))
    out = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append(Binding('import', a.name, None, a.asname))
        elif isinstance(node, ast.ImportFrom):
            mod = '.' * node.level + (node.module or '')
            for a in node.names:
                out.append(Binding('from', mod, a.name, a.asname))
    return out


CANONICAL_BINDINGS = {fw: {b.bound: b.target for b in bindings(text)}
                      for fw, text in CANONICAL.items()}


def render_imports(binds):
    """Sorted import statements for a list of Binding: plain imports first,
    then from-imports (names of one module merged on one line), each group
    alphabetical by module."""
    plain = sorted({(b.module, b.asname or '') for b in binds
                    if b.kind == 'import'})
    lines = [f'import {m} as {a}' if a else f'import {m}' for m, a in plain]
    by_mod = {}
    for b in binds:
        if b.kind == 'from':
            by_mod.setdefault(b.module, set()).add((b.name, b.asname or ''))
    for mod in sorted(by_mod):
        names = ', '.join(f'{n} as {a}' if a else n
                          for n, a in sorted(by_mod[mod]))
        lines.append(f'from {mod} import {names}')
    return lines


class Classification:
    """Result of classify(): status is 'canonical', 'normalizable' (with
    `extras`, possibly empty) or 'manual' (with `reasons`)."""
    __slots__ = ('status', 'extras', 'reasons', 'dropped')

    def __init__(self, status, extras=(), reasons=(), dropped=()):
        self.status = status
        self.extras = list(extras)
        self.reasons = list(reasons)
        self.dropped = list(dropped)


def classify(code, fw):
    """Decide how an imports cell of framework `fw` normalizes.

    Normalizable: the cell holds only import statements, blank lines,
    `%matplotlib inline`, and the canonical block's own calls (MXNet's
    `npx.set_np()`); no name it imports is bound by the canonical block to a
    different object. Its extras are the imports the canonical block does
    not already provide. Anything else (comments, `#@save`, environment
    settings, seeds, other statements, conflicting names) is 'manual'.
    """
    if isinstance(code, (list, tuple)):
        code = '\n'.join(code)
    if is_canonical(code, fw):
        return Classification('canonical')
    reasons, dropped = [], []
    calls = CANONICAL_CALLS.get(fw, set())
    import_src = []
    for line in code.split('\n'):
        s = line.strip()
        if not s:
            continue
        if '#@save' in s:
            reasons.append('#@save line')
        elif s in DROPPABLE_LINES:
            dropped.append(s)
            continue
        elif s.startswith('#'):
            reasons.append(f'comment: {s[:60]}')
        elif s.startswith(('%', '!')):
            reasons.append(f'magic: {s[:60]}')
        elif s in calls:
            continue
        import_src.append(line)
    try:
        tree = ast.parse('\n'.join(import_src))
    except SyntaxError as e:
        return Classification('manual', reasons=[f'syntax error: {e.msg}'])
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment('\n'.join(import_src), node) or ''
            reasons.append(f'statement: {seg.splitlines()[0][:60]}')
        elif isinstance(node, ast.ImportFrom) and any(
                a.name == '*' for a in node.names):
            reasons.append('star import')
    if reasons:
        return Classification('manual', reasons=_dedup(reasons))
    canon = CANONICAL_BINDINGS[fw]
    extras = []
    for b in bindings('\n'.join(import_src)):
        if b.bound in canon:
            if canon[b.bound] != b.target:
                reasons.append(f'`{b.bound}` is {b.target} here '
                               f'but {canon[b.bound]} in the preamble')
                continue
            if not b.loads_submodule():
                continue
        extras.append(b)
    if reasons:
        return Classification('manual', reasons=_dedup(reasons))
    # One name bound twice in the extras (e.g. `keras` from two places):
    # keep the last, as Python would.
    seen = {}
    for b in extras:
        seen[b.bound] = b
    return Classification('normalizable', extras=list(seen.values()),
                          dropped=dropped)


def _dedup(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out
