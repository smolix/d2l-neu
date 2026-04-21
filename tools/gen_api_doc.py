#!/usr/bin/env python3
"""Generate API documentation for the d2l package.

Reads each framework's d2l/<fw>.py, extracts class and function signatures
and docstrings using the ast module, and injects per-framework tabbed Markdown
into the d2l.qmd file's empty Classes and Functions sections.

Usage:
    python tools/gen_api_doc.py [--lib-dir d2l] [--qmd chapter_.../d2l.qmd]
"""

import ast
import re
import argparse
from pathlib import Path

FRAMEWORKS = [
    ('pytorch', 'PyTorch', 'torch.py'),
    ('tensorflow', 'TensorFlow', 'tensorflow.py'),
    ('jax', 'JAX', 'jax.py'),
    ('mxnet', 'MXNet', 'mxnet.py'),
]


def _get_signature(node):
    """Reconstruct a function/method signature from AST."""
    args = node.args
    parts = []

    positionals = args.args
    defaults = args.defaults
    n_defaults = len(defaults)
    n_positionals = len(positionals)

    for i, arg in enumerate(positionals):
        if arg.arg in ('self', 'cls'):
            continue
        di = i - (n_positionals - n_defaults)
        if di >= 0:
            try:
                default = ast.unparse(defaults[di])
            except Exception:
                default = '...'
            parts.append(f'{arg.arg}={default}')
        else:
            parts.append(arg.arg)

    if args.vararg:
        parts.append(f'*{args.vararg.arg}')

    kw_defaults = args.kw_defaults
    for i, kwarg in enumerate(args.kwonlyargs):
        d = kw_defaults[i] if i < len(kw_defaults) else None
        if d is not None:
            try:
                default = ast.unparse(d)
            except Exception:
                default = '...'
            parts.append(f'{kwarg.arg}={default}')
        else:
            parts.append(kwarg.arg)

    if args.kwarg:
        parts.append(f'**{args.kwarg.arg}')

    return ', '.join(parts)


def _get_class_init_sig(node):
    """Get __init__ signature for a class."""
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
            return _get_signature(item)
    return ''


def _get_bases(node):
    """Get base class names for a class."""
    bases = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            pass
    return bases


def _format_docstring(docstring):
    """Extract the first line and Defined-in reference from a docstring."""
    if not docstring:
        return '', ''
    lines = docstring.strip().split('\n')
    first_line = lines[0].strip()

    defined_in = ''
    for line in lines:
        m = re.search(r'Defined in :numref:`([^`]+)`', line)
        if m:
            defined_in = m.group(1)
            break

    return first_line, defined_in


def _numref_to_link(label):
    """Convert a :numref: label to a Quarto @-reference."""
    return f'@{label.replace("_", "-")}'


def extract_api(lib_path):
    """Extract classes and functions from a d2l library file."""
    source = Path(lib_path).read_text(encoding='utf-8')
    tree = ast.parse(source)

    classes = []
    functions = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.startswith('_'):
                continue
            docstring = ast.get_docstring(node) or ''
            first_line, defined_in = _format_docstring(docstring)
            if not defined_in:
                continue
            bases = _get_bases(node)
            init_sig = _get_class_init_sig(node)

            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                    m_doc = ast.get_docstring(item) or ''
                    m_first, m_def = _format_docstring(m_doc)
                    m_sig = _get_signature(item)
                    methods.append({
                        'name': item.name,
                        'signature': m_sig,
                        'description': m_first,
                        'defined_in': m_def,
                    })

            classes.append({
                'name': node.name,
                'bases': bases,
                'init_signature': init_sig,
                'description': first_line,
                'defined_in': defined_in,
                'methods': methods,
            })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith('_'):
                continue
            docstring = ast.get_docstring(node) or ''
            first_line, defined_in = _format_docstring(docstring)
            if not defined_in:
                continue
            sig = _get_signature(node)
            functions.append({
                'name': node.name,
                'signature': sig,
                'description': first_line,
                'defined_in': defined_in,
            })

    classes.sort(key=lambda c: c['name'])
    functions.sort(key=lambda f: f['name'])
    return classes, functions


def _format_entry_desc(description, defined_in):
    """Format description + cross-reference line."""
    desc = description or ''
    ref = _numref_to_link(defined_in) if defined_in else ''
    if desc and ref:
        return f'{desc} {ref}\n'
    if desc:
        return f'{desc}\n'
    if ref:
        return f'{ref}\n'
    return '\n'


def _emit_classes(classes):
    """Emit Markdown for a list of classes (no section heading)."""
    lines = []
    for cls in classes:
        bases_str = ', '.join(cls['bases'])
        if bases_str:
            header = f'`class {cls["name"]}({bases_str})`'
        else:
            header = f'`class {cls["name"]}`'

        lines.append(f'#### {header} {{.unnumbered .unlisted}}\n')

        if cls['init_signature']:
            lines.append(f'```python\n{cls["name"]}({cls["init_signature"]})\n```\n')

        lines.append(_format_entry_desc(cls['description'], cls['defined_in']))

        if cls['methods']:
            for method in cls['methods']:
                parts = [f'`{method["name"]}({method["signature"]})`']
                if method['description']:
                    parts.append(f' — {method["description"]}')
                if method['defined_in']:
                    parts.append(f' {_numref_to_link(method["defined_in"])}')
                lines.append(f'- {"".join(parts)}')
            lines.append('')
    return lines


def _emit_functions(functions):
    """Emit Markdown for a list of functions (no section heading)."""
    lines = []
    for func in functions:
        lines.append(f'#### `{func["name"]}` {{.unnumbered .unlisted}}\n')
        lines.append(f'```python\n{func["name"]}({func["signature"]})\n```\n')
        lines.append(_format_entry_desc(func['description'], func['defined_in']))
    return lines


def generate_markdown(all_apis):
    """Generate tabbed Markdown for all frameworks.

    all_apis: list of (display_name, classes, functions) tuples
    """
    lines = []

    # Classes section with per-framework tabs
    lines.append('## Classes\n')
    lines.append(':::: {.panel-tabset group="framework"}\n')
    for display_name, classes, _ in all_apis:
        lines.append(f'### {display_name}\n')
        if classes:
            lines.extend(_emit_classes(classes))
        else:
            lines.append('*No documented classes for this framework.*\n')
    lines.append('::::\n')

    # Functions section with per-framework tabs
    lines.append('\n## Functions\n')
    lines.append(':::: {.panel-tabset group="framework"}\n')
    for display_name, _, functions in all_apis:
        lines.append(f'### {display_name}\n')
        if functions:
            lines.extend(_emit_functions(functions))
        else:
            lines.append('*No documented functions for this framework.*\n')
    lines.append('::::\n')

    return '\n'.join(lines)


def inject_into_qmd(qmd_path, markdown_content):
    """Replace the empty Classes/Functions sections in the .qmd file."""
    text = Path(qmd_path).read_text(encoding='utf-8')

    # Clean up stale :begin_tab: / panel-tabset artifacts between the intro
    # paragraph and the Classes heading (leftover from eval_rst currentmodule
    # directives that the preprocessor couldn't fully convert).
    text = re.sub(
        r'(::: \{\.panel-tabset[^}]*\}.*?:::\s*\n)',
        '', text, count=1, flags=re.DOTALL)
    text = re.sub(r':begin_tab:`[^`]*`\s*\n', '', text)
    text = re.sub(r':end_tab:\s*\n', '', text)

    pattern = re.compile(
        r'(## Classes\s*\n)(.*?)(## Functions\s*\n)(.*?)(\Z)',
        re.DOTALL)

    m = pattern.search(text)
    if not m:
        print(f'WARNING: Could not find Classes/Functions sections in {qmd_path}')
        return False

    new_text = text[:m.start()] + markdown_content
    Path(qmd_path).write_text(new_text, encoding='utf-8')
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate d2l API documentation')
    parser.add_argument('--lib-dir', type=Path, default=Path('d2l'),
                        help='Directory containing framework .py files')
    parser.add_argument('--qmd', type=Path,
                        default=Path('chapter_appendix-tools-for-deep-learning/d2l.qmd'),
                        help='Path to d2l.qmd file to inject into')
    args = parser.parse_args()

    if not args.qmd.exists():
        print(f'ERROR: QMD file {args.qmd} not found. Run d2l_preprocess.py first.')
        return 1

    all_apis = []
    for fw_key, display_name, filename in FRAMEWORKS:
        lib_path = args.lib_dir / filename
        if not lib_path.exists():
            print(f'  SKIP {display_name}: {lib_path} not found')
            continue
        classes, functions = extract_api(lib_path)
        print(f'  {display_name}: {len(classes)} classes, {len(functions)} functions')
        all_apis.append((display_name, classes, functions))

    if not all_apis:
        print('ERROR: No library files found. Run build_lib.py first.')
        return 1

    markdown = generate_markdown(all_apis)
    if inject_into_qmd(args.qmd, markdown):
        print(f'Injected API documentation into {args.qmd}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
