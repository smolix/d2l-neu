#!/usr/bin/env python3
"""Scan source .md files and emit per-notebook Make dep fragments (.d).

For each (framework, source.md) that produces a notebook, walks the
framework-specific code blocks and records every `d2l.<symbol>` reference.
Writes `_notebooks/<fw>/<rel>.d` whose content is a Make-includable line:

    _notebooks/<fw>/<rel>.executed: \
        d2l/_blocks/<fw>/Symbol1.py \
        d2l/_blocks/<fw>/Symbol2.py \
        ...

The per-notebook .executed Make rule `-include`s this file, so editing
one #@save block invalidates only the notebooks whose .d list it. Symbols
that exist in the source but not in `d2l/_blocks/<fw>/` are silently
dropped (e.g. references to symbols defined locally in the same notebook).

Usage:
    python tools/scan_d2l_usage.py --source . --output-dir _notebooks

The notebook scan is textual: it matches `d2l.<symbol>` patterns in the
notebook's code cells. False positives (e.g. `d2l.something_not_a_function`)
are filtered against the real shard set, so we don't add ghost deps.

The dependency list is TRANSITIVE over the library's own call graph: a
notebook that calls `d2l.Trainer` also depends on every shard the `Trainer`
shard references (`ProgressBoard`, `Module`, the JAX `_trainer_train_step`,
...), and so on recursively. Library code reaches other symbols both as
`d2l.X` and as bare module globals (`_trainer_train_step(...)` inside
`Trainer.fit_epoch`); `shard_references` finds both with `ast`/`symtable`,
counting only names that resolve to module globals (a local variable that
happens to share a symbol's name is not a reference).

Every notebook also depends on the framework's `_preamble` shard: the
hand-written header of `d2l/<fw>.py` plus the import-only #@save blocks,
which run on `import d2l` whatever the notebook calls.
"""
import argparse
import ast
import os
import re
import symtable
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scan_notebook_manifests import FRAMEWORKS, file_frameworks
from build_lib import LIB_ONLY_FILES


# Match `d2l.<identifier>`. The identifier rule is greedy enough for the
# typical usage in source code and prose-as-code blocks.
D2L_REF_RE = re.compile(r'\bd2l\.([A-Za-z_][A-Za-z_0-9]*)')

# Match tab headers so we can scope d2l refs to the framework that produces
# the relevant notebook. `all` and unqualified blocks attribute to every fw.
TAB_RE = re.compile(r'^(?:%%tab|#@tab)\s+([^\n]+)$', re.MULTILINE)

# Match fenced code blocks `\`\`\`{.python ...}` ... `\`\`\``.
CODE_BLOCK_RE = re.compile(
    r'```\{\.python[^}]*\}\n(.*?)```', re.DOTALL)


def _block_tabs(block):
    """Return the set of frameworks this code block applies to.

    Tab markers appear inside the block as `#@tab <fws>` or `%%tab <fws>`.
    A block with no tab marker applies to all frameworks ('all').
    """
    fws = set()
    has_any = False
    for m in TAB_RE.finditer(block):
        has_any = True
        spec = m.group(1).strip()
        if spec == 'all':
            return set(FRAMEWORKS)
        for t in (t.strip() for t in spec.split(',')):
            if t in FRAMEWORKS:
                fws.add(t)
    if not has_any:
        return set(FRAMEWORKS)
    return fws


def d2l_refs_per_framework(md_path):
    """Return {framework: set(symbol)} of `d2l.X` references in md_path."""
    try:
        text = md_path.read_text(encoding='utf-8')
    except Exception:
        return {fw: set() for fw in FRAMEWORKS}

    refs = {fw: set() for fw in FRAMEWORKS}
    for cm in CODE_BLOCK_RE.finditer(text):
        block = cm.group(1)
        block_fws = _block_tabs(block)
        for rm in D2L_REF_RE.finditer(block):
            sym = rm.group(1)
            for fw in block_fws:
                refs[fw].add(sym)
    return refs


def shard_references(code, filename='<shard>'):
    """Return the module-level names one shard's code refers to.

    Two kinds of reference count: attribute access `d2l.X`, and a bare name
    that resolves to a module global in its scope (Python's own scoping rules,
    via `symtable`; parameters, locals and closure variables are excluded).
    The module alias `d2l` itself is not a symbol. On a parse error, fall back
    to every identifier in the text (over-approximates; never misses).
    """
    try:
        tree = ast.parse(code, filename)
        table = symtable.symtable(code, filename, 'exec')
    except SyntaxError:
        return set(re.findall(r'\b[A-Za-z_]\w*\b', code)) - {'d2l'}
    refs = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == 'd2l'}
    stack = [table]
    while stack:
        scope = stack.pop()
        is_module = str(scope.get_type()).lower().endswith('module')
        for sym in scope.get_symbols():
            if sym.is_referenced() and (is_module or sym.is_global()):
                refs.add(sym.get_name())
        stack.extend(scope.get_children())
    refs.discard('d2l')
    return refs


# build_lib.py's shard for the hand-written library header, the import-only
# #@save blocks and nothing else: every notebook of the framework imports it.
PREAMBLE = '_preamble'


def shard_graph(shard_dir):
    """{symbol: set(symbols its shard references)} for one framework."""
    paths = {p.stem: p for p in Path(shard_dir).glob('*.py')}
    names = set(paths)
    graph = {}
    for name, p in paths.items():
        try:
            code = p.read_text(encoding='utf-8')
        except OSError:
            code = ''
        graph[name] = (shard_references(code, str(p)) & names) - {name}
    # Every notebook depends on the preamble shard (see main); its own
    # imports and module-level names are not calls into the library, so it
    # gets no outgoing edges (otherwise every notebook would inherit them).
    if PREAMBLE in graph:
        graph[PREAMBLE] = set()
    return graph


def closure(roots, graph):
    """Every symbol reachable from *roots* in *graph* (roots included)."""
    seen = set()
    todo = [r for r in roots if r in graph]
    while todo:
        s = todo.pop()
        if s in seen:
            continue
        seen.add(s)
        todo.extend(graph.get(s, ()))
    return seen


def _write_if_changed(path, content_bytes):
    if path.exists() and path.read_bytes() == content_bytes:
        # Content unchanged: keep the file byte-identical (no git churn, no
        # downstream rebuild) but bump its mtime to *now*. This `.d` is pulled
        # in via `-include`; GNU Make re-makes generated include files and then
        # RESTARTS, and it keeps restarting as long as the include is older than
        # its prerequisites (the source .md). Without this touch the .d stays
        # stale forever and make spins in an infinite restart loop (observed
        # under GNU Make 4.x on macOS; 3.81 masked it via its grouped-target
        # misparse). Safe because .executed depends on the *shards named inside*
        # the .d, not on the .d file's own timestamp.
        os.utime(path, None)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content_bytes)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('.'),
                        help='Source dir (default: cwd)')
    parser.add_argument('--output-dir', type=Path, default=Path('_notebooks'),
                        help='Destination _notebooks dir')
    parser.add_argument('--shard-dir', type=Path, default=Path('d2l/_blocks'),
                        help='Where per-symbol shards live; '
                             'used to drop ghost references')
    args = parser.parse_args()

    # The shards that exist for each framework, and the references between
    # them (for the transitive closure).
    graphs = {fw: shard_graph(args.shard_dir / fw) for fw in FRAMEWORKS}

    n_written = 0
    lib_only = set(LIB_ONLY_FILES)
    for md in sorted(args.source.glob('chapter_*/*.md')):
        rel = md.relative_to(args.source)
        if str(rel) in lib_only:
            # Build-only lib-extraction source: never produces a notebook.
            continue
        produces = file_frameworks(md)
        if not produces:
            continue
        refs_per_fw = d2l_refs_per_framework(md)

        for fw in produces:
            # The notebook's executed-stamp depends on every shard for
            # symbols it consumes (and present in this framework's shard
            # dir), plus everything those shards reach in turn. All-frameworks
            # refs (no tab marker) end up under every fw; framework-specific
            # tabs scope correctly.
            consumed = closure(refs_per_fw[fw] | {PREAMBLE}, graphs[fw])
            stamp = (args.output_dir / fw /
                     rel).with_suffix('.executed')
            depfile = (args.output_dir / fw /
                       rel).with_suffix('.d')
            shards = sorted(args.shard_dir / fw / f'{s}.py' for s in consumed)
            if shards:
                lines = [f'{stamp}: \\']
                for i, s in enumerate(shards):
                    suffix = ' \\' if i < len(shards) - 1 else ''
                    lines.append(f'    {s}{suffix}')
                content = ('# Auto-generated by tools/scan_d2l_usage.py — '
                           'do not edit.\n' + '\n'.join(lines) + '\n')
            else:
                content = ('# Auto-generated by tools/scan_d2l_usage.py — '
                           'do not edit.\n# (no d2l symbols referenced)\n')
            if _write_if_changed(depfile, content.encode('utf-8')):
                n_written += 1

    print(f'scan_d2l_usage: updated {n_written} .d file(s)')


if __name__ == '__main__':
    main()
