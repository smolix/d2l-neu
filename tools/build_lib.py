#!/usr/bin/env python3
"""Build the d2l Python package from d2l-en source files.

Extracts code blocks marked with #@save from each framework's notebook source,
merges @d2l.add_to_class methods into their target classes, adds docstring
references, and appends framework-specific aliases.

Usage:
    python tools/build_lib.py ../d2l-en d2l/ [--frameworks pytorch jax]
"""

import re
import ast
import sys
import argparse
import configparser
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import (
    CHAPTER_NUMBERING, FRAMEWORKS,
    parse_blocks, extract_tab, is_boilerplate, is_python_block,
    clean_save_markers, CodeBlock, MarkdownBlock, CodeTabSet, TocBlock,
)

# ──────────────────────────────────────────────────────────
# Block extraction
# ──────────────────────────────────────────────────────────

def _collect_tab_body(lines, start, body_indent=None):
    """Collect indented body lines starting at *start*. Returns (body_lines, next_i)."""
    body_lines = []
    i = start
    while i < len(lines):
        l = lines[i]
        if l.strip() == '':
            body_lines.append(l)
            i += 1
            continue
        current_indent = len(l) - len(l.lstrip())
        if body_indent is None:
            body_indent = current_indent
        if current_indent >= body_indent:
            body_lines.append(l)
            i += 1
        else:
            break
    return body_lines, i, body_indent


def _resolve_nested_tabs(body_lines, framework):
    """Resolve nested tab.selected() inside a collected body, inlining or
    dropping in place (without changing indentation).  This ensures that
    when the caller de-indents the body, nested content lands at the
    correct depth relative to surrounding code (e.g. inside __init__)."""
    result = []
    i = 0
    while i < len(body_lines):
        line = body_lines[i]
        m = re.match(r'^(\s*)if (tab\.selected\(.+)\):\s*$', line)
        if m:
            fws = re.findall(r"'(\w+)'", m.group(2))
            keep = framework in fws
            inner_body, i, _ = _collect_tab_body(body_lines, i + 1)
            if keep:
                result.extend(inner_body)
        else:
            result.append(line)
            i += 1
    return result


def flatten_tab_branches(code, framework):
    """Flatten tab.selected() branches, keeping only the target framework.

    Transforms:
        if tab.selected('pytorch'):
            x = torch.tensor(1)
        if tab.selected('tensorflow'):
            x = tf.constant(1)

    Into (for framework='pytorch'):
        x = torch.tensor(1)

    Handles nested tab.selected() inside an outer block correctly by
    resolving inner branches *before* de-indenting, so content like
    ``self.training = None`` stays inside the preceding method body.
    """
    lines = code.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(\s*)if (tab\.selected\(.+)\):\s*$', line)
        if m:
            indent = m.group(1)
            fws = re.findall(r"'(\w+)'", m.group(2))
            keep = framework in fws

            body_lines, i, body_indent = _collect_tab_body(lines, i + 1)

            if keep and body_lines:
                # Resolve any nested tab branches in-place first
                body_lines = _resolve_nested_tabs(body_lines, framework)
                # De-indent: strip body_indent chars, prepend outer indent
                for bl in body_lines:
                    if bl.strip():
                        result.append(indent + bl[body_indent:])
                    else:
                        result.append(bl)
        else:
            result.append(line)
            i += 1

    out = '\n'.join(result)
    if 'tab.selected(' in out:
        return flatten_tab_branches(out, framework)
    return out


def save_blocks(source):
    """Extract all code blocks following #@save markers.

    Returns a list of extracted code strings (one per #@save marker).
    """
    lines = source.split('\n')
    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r'#\s*@save', line)
        if not m:
            i += 1
            continue

        block_lines = []
        # Keep code BEFORE the marker on the same line
        prefix = line[:m.start()].rstrip()
        if prefix:
            block_lines.append(prefix)

        # Collect the block that follows
        i += 1
        if not prefix:
            # Standalone #@save: peek at the first non-blank line
            first_real = ''
            for j in range(i, len(lines)):
                if lines[j].strip():
                    first_real = lines[j]
                    break
            if (first_real.startswith('def ') or
                    first_real.startswith('class ') or
                    first_real.startswith('@')):
                # def/class definition — use indentation-based collection
                while i < len(lines):
                    l = lines[i]
                    if (l.startswith(' ') or l.startswith('\t') or
                            l.strip() == '' or l.startswith('def ') or
                            l.startswith('class ') or l.startswith('@')):
                        block_lines.append(l)
                        i += 1
                    else:
                        break
            else:
                # Top-level statements (imports, assignments).
                # Collect until a blank line.
                while i < len(lines):
                    l = lines[i]
                    if l.strip() == '':
                        break
                    block_lines.append(l)
                    i += 1
        else:
            # Inline #@save (e.g. "def func():  #@save" or "@decorator  #@save")
            # Collect any extra decorators, then the def/class and its body.
            got_def = prefix.startswith('def ') or prefix.startswith('class ')
            while i < len(lines):
                l = lines[i]
                if l.startswith(' ') or l.startswith('\t') or l.strip() == '':
                    block_lines.append(l)
                    i += 1
                elif not got_def and (l.startswith('def ') or
                                      l.startswith('class ')):
                    block_lines.append(l)
                    got_def = True
                    i += 1
                elif not got_def and l.startswith('@'):
                    block_lines.append(l)
                    i += 1
                else:
                    break

        text = '\n'.join(block_lines).rstrip()
        if text:
            results.append(text)

    return results


def find_section_label(text_before_code):
    """Find the most recent :label:`sec_*` in preceding markdown text."""
    labels = re.findall(r':label:`(sec_[^`]+)`', text_before_code)
    if labels:
        return labels[-1]
    # Also check Quarto-style labels
    labels = re.findall(r'\{#(sec-[^}]+)\}', text_before_code)
    if labels:
        return labels[-1]
    return ''


def extract_save_blocks(src_path, framework):
    """Extract all #@save blocks from a source .md file for a given framework.

    Parses the raw .md file directly (does NOT use the preprocessor's
    parse_blocks, which strips #@save markers).

    Returns list of (code_block, section_label, source_file) tuples.
    """
    text = Path(src_path).read_text(encoding='utf-8')
    lines = text.split('\n')

    saved = []
    markdown_so_far = ''
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for code fence start
        m = re.match(r'^```(.*)$', line)
        if m and not line.startswith('````'):
            info = m.group(1).strip()
            code_lines = []
            i += 1
            while i < len(lines) and not re.match(r'^```\s*$', lines[i]):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```

            if not is_python_block(info):
                continue
            if is_boilerplate(code_lines):
                continue

            # Extract tab marker
            tab, cleaned = extract_tab(code_lines)

            # Check if this block is for our framework
            if tab is not None and tab != 'all':
                tabs = [t.strip() for t in tab.split(',')]
                if framework not in tabs:
                    continue

            # Look for #@save in the code (using CLEANED lines, minus tab marker)
            code = '\n'.join(cleaned)
            if '#@save' not in code and '# @save' not in code:
                continue

            # Flatten tab.selected() branches for the target framework
            code = flatten_tab_branches(code, framework)

            for extracted in save_blocks(code):
                label = find_section_label(markdown_so_far)
                saved.append((extracted, label, str(src_path)))
        else:
            markdown_so_far += line + '\n'
            i += 1

    return saved


# ──────────────────────────────────────────────────────────
# Block refactoring
# ──────────────────────────────────────────────────────────

def add_docstring_labels(blocks):
    """Add 'Defined in :numref:`sec_*`' to function/class docstrings."""
    result = []
    for code, label, source in blocks:
        if not label:
            result.append((code, label, source))
            continue

        lines = code.split('\n')
        new_lines = []
        label_added = False

        for i, line in enumerate(lines):
            new_lines.append(line)
            # Look for docstring opening after def/class
            if not label_added and (line.strip().startswith('def ') or
                                     line.strip().startswith('class ')):
                # Find the docstring
                for j in range(i + 1, min(i + 5, len(lines))):
                    if '"""' in lines[j]:
                        indent = len(lines[j]) - len(lines[j].lstrip())
                        spaces = ' ' * indent
                        # Single-line docstring: """text"""
                        if lines[j].count('"""') >= 2:
                            # Insert label before closing """
                            doc = lines[j].rstrip()
                            if doc.endswith('"""'):
                                doc = doc[:-3].rstrip()
                                lines[j] = (f'{doc}\n\n'
                                           f'{spaces}Defined in :numref:`{label}`"""')
                                label_added = True
                        else:
                            # Multi-line: find closing """
                            for k in range(j + 1, len(lines)):
                                if '"""' in lines[k]:
                                    lines[k] = (f'\n{spaces}Defined in '
                                               f':numref:`{label}`"""')
                                    label_added = True
                                    break
                        break

        result.append(('\n'.join(lines) if label_added else code, label, source))
    return result


def _block_defined_names(code):
    """Return the top-level names a block defines (class/def/assignment)."""
    names = set()
    for line in code.split('\n'):
        m = re.match(r'(class|def)\s+(\w+)', line.lstrip())
        if m:
            names.add(m.group(2))
            continue
        m = re.match(r'(\w+)\s*=', line)
        if m:
            names.add(m.group(1))
    return names


def _block_d2l_references(code):
    """Return the set of d2l.X symbols this block references (bases + calls).

    Keeps the analysis shallow: matches class-base lists and decorator calls,
    which is where an unresolved reference at import time would blow up.
    """
    refs = set()
    for m in re.finditer(r'^class\s+\w+\(([^)]*)\):', code, re.MULTILINE):
        for b in m.group(1).split(','):
            b = b.strip()
            if b.startswith('d2l.'):
                refs.add(b[4:].split('(', 1)[0])
    for m in re.finditer(r'@d2l\.(\w+)', code):
        refs.add(m.group(1))
    return refs


def drop_unresolved_blocks(blocks, extra_names):
    """Drop #@save blocks whose bases/decorators reference d2l.X that nothing
    in this framework defines.

    A class like ``SuccessiveHalvingScheduler(d2l.HPOScheduler)`` lands in a
    framework's save-blocks even when its parent is gated to `%%tab pytorch`;
    left in, it raises at import time. Iterate until stable because dropping
    one block may orphan others (e.g. subclasses of the dropped class).
    """
    blocks = list(blocks)
    while True:
        defined = set(extra_names)
        for code, _, _ in blocks:
            defined |= _block_defined_names(code)
        drop = []
        for i, (code, _, source) in enumerate(blocks):
            refs = _block_d2l_references(code)
            missing = refs - defined
            if missing:
                drop.append((i, missing, source))
        if not drop:
            return blocks
        for i, missing, source in drop:
            top = re.search(r'^(class|def)\s+(\w+)', blocks[i][0], re.MULTILINE)
            name = top.group(2) if top else '(anonymous)'
            print(f'    drop {name}: unresolved {sorted(missing)} '
                  f'(from {Path(source).name})')
        blocks = [b for i, b in enumerate(blocks) if i not in {d[0] for d in drop}]


def merge_add_to_class(blocks):
    """Merge @d2l.add_to_class(ClassName) methods into class bodies."""
    # First pass: find all class definitions and their indices
    class_indices = {}
    for i, (code, label, source) in enumerate(blocks):
        m = re.match(r'class\s+(\w+)', code)
        if m:
            class_indices[m.group(1)] = i

    # Second pass: find @d2l.add_to_class decorators and merge
    merged = list(blocks)
    to_remove = set()

    for i, (code, label, source) in enumerate(blocks):
        m = re.match(r'@d2l\.add_to_class\((?:d2l\.)?(\w+)\)', code)
        if m:
            class_name = m.group(1)
            if class_name in class_indices:
                # Extract the method (skip the decorator line)
                method_lines = code.split('\n')
                method_start = 1  # skip decorator line
                method_code = '\n'.join(method_lines[method_start:])
                # Indent the method with 4 spaces
                indented = '\n'.join(
                    ('    ' + l if l.strip() else l)
                    for l in method_code.split('\n'))

                # Append to the class block
                ci = class_indices[class_name]
                class_code, cl, cs = merged[ci]
                merged[ci] = (class_code + '\n\n' + indented, cl, cs)
                to_remove.add(i)

    return [b for i, b in enumerate(merged) if i not in to_remove]


def _dedup_class_methods(code):
    """Remove duplicate method defs within a class body, keeping the last."""
    lines = code.split('\n')
    method_spans = []
    i = 0
    while i < len(lines):
        m = re.match(r'^(    )def (\w+)\(', lines[i])
        if m:
            name = m.group(2)
            start = i
            while start > 0 and re.match(r'^    @', lines[start - 1]):
                start -= 1
            i += 1
            while i < len(lines) and (lines[i].startswith('        ')
                                       or lines[i].strip() == ''):
                i += 1
            end = i
            while end > start and lines[end - 1].strip() == '':
                end -= 1
            method_spans.append((name, start, end))
        else:
            i += 1

    seen_last = {}
    for idx, (name, start, end) in enumerate(method_spans):
        seen_last[name] = idx

    drop_lines = set()
    for idx, (name, start, end) in enumerate(method_spans):
        if seen_last[name] != idx:
            for j in range(start, end):
                drop_lines.add(j)

    if not drop_lines:
        return code
    kept = [l for i, l in enumerate(lines) if i not in drop_lines]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(kept))


def deduplicate_blocks(blocks):
    """Remove earlier blocks when a later block redefines the same name.

    Also deduplicates methods within class bodies (e.g. placeholder stubs
    replaced by real implementations via add_to_class merging).
    """
    result = []
    for code, label, source in blocks:
        if re.match(r'class\s+\w+', code):
            code = _dedup_class_methods(code)
        result.append((code, label, source))

    name_to_last = {}
    for i, (code, _, _) in enumerate(result):
        names = set()
        for m in re.finditer(r'^(?:def|class)\s+(\w+)', code, re.MULTILINE):
            names.add(m.group(1))
        if len(names) == 1:
            name = next(iter(names))
            name_to_last[name] = i

    drop = set()
    seen = {}
    for i, (code, _, _) in enumerate(result):
        names = set()
        for m in re.finditer(r'^(?:def|class)\s+(\w+)', code, re.MULTILINE):
            names.add(m.group(1))
        if len(names) == 1:
            name = next(iter(names))
            if name in seen and name_to_last.get(name) == i:
                drop.add(seen[name])
            seen[name] = i

    return [b for i, b in enumerate(result) if i not in drop]


# ──────────────────────────────────────────────────────────
# Alias generation
# ──────────────────────────────────────────────────────────

def parse_alias_config(config_str, split_by_comma=True):
    """Parse alias config string into (from, to) pairs."""
    if not config_str:
        return []

    if split_by_comma:
        items = [s.strip() for s in config_str.replace('\n', ',').split(',')]
        items = [s for s in items if s]
    else:
        items = [s.strip() for s in config_str.strip().split('\n')]
        items = [s for s in items if s]

    pairs = []
    for item in items:
        if '->' in item:
            a, b = item.split('->', 1)
            pairs.append((a.strip(), b.strip()))
        else:
            pairs.append((item.strip(), item.strip()))
    return pairs


def generate_aliases(lib_config):
    """Generate alias code from a library config section."""
    lines = []
    lib_name = lib_config.get('lib_name', '')

    # Simple aliases: name = lib_name.name (or name = lib_name.target)
    if 'simple_alias' in lib_config:
        for src, dst in parse_alias_config(lib_config['simple_alias']):
            # Handle exp( pattern (trailing paren in config)
            src = src.rstrip('(')
            dst = dst.rstrip('(')
            lines.append(f'{src} = {lib_name}.{dst}')

    # Fluent aliases: name = lambda x, *a, **k: x.method(*a, **k)
    if 'fluent_alias' in lib_config:
        for src, dst in parse_alias_config(lib_config['fluent_alias']):
            chain = '.'.join(f'{part}(*args, **kwargs)'
                           if i == len(dst.split('.')) - 1
                           else part
                           for i, part in enumerate(dst.split('.')))
            lines.append(
                f'{src} = lambda x, *args, **kwargs: x.{chain}')

    # Custom alias code
    if 'alias' in lib_config:
        for line in lib_config['alias'].strip().split('\n'):
            line = line.strip()
            if line:
                lines.append(line)

    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────
# Library file generation
# ──────────────────────────────────────────────────────────

HEADER = """\
#################   WARNING   ################
# The below part is generated automatically through:
#    python tools/build_lib.py
# Don't edit it directly

import sys
d2l = sys.modules[__name__]
"""


def build_library(src_dir, output_file, framework, lib_config, files):
    """Build a framework-specific library file."""
    print(f'  Extracting #@save blocks for {framework}...')

    # Extract all save blocks from all source files
    all_blocks = []
    for rel in files:
        src_file = src_dir / rel
        if not src_file.exists():
            continue
        blocks = extract_save_blocks(src_file, framework)
        all_blocks.extend(blocks)

    print(f'  Found {len(all_blocks)} #@save blocks')

    # Refactor: add docstring labels, merge add_to_class methods, deduplicate
    all_blocks = add_docstring_labels(all_blocks)
    all_blocks = merge_add_to_class(all_blocks)
    all_blocks = deduplicate_blocks(all_blocks)

    # Compute names provided by the preamble + alias config so we can detect
    # blocks whose d2l.X references have no definition in this framework.
    preamble_for_scan = ''
    output_path = Path(output_file)
    if output_path.exists():
        existing = output_path.read_text()
        preamble_for_scan = (existing[:existing.index('#####')]
                             if 'WARNING' in existing else existing)
    alias_code = generate_aliases(lib_config)
    extra_names = _block_defined_names(preamble_for_scan) | _block_defined_names(alias_code)

    before = len(all_blocks)
    all_blocks = drop_unresolved_blocks(all_blocks, extra_names)
    if len(all_blocks) != before:
        print(f'  Dropped {before - len(all_blocks)} orphan blocks')

    print(f'  After merging: {len(all_blocks)} blocks')

    # Read existing preamble (custom header before WARNING)
    preamble = ''
    output_path = Path(output_file)
    if output_path.exists():
        existing = output_path.read_text()
        if 'WARNING' in existing:
            preamble = existing[:existing.index('#####')]
        elif existing.strip():
            preamble = existing

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        if preamble:
            f.write(preamble)
        f.write(HEADER + '\n')

        for code, label, source in all_blocks:
            f.write(code + '\n\n')

        # Append aliases
        aliases = generate_aliases(lib_config)
        if aliases:
            f.write('\n' + aliases + '\n')

    print(f'  Written to {output_file}')


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build the d2l Python package from source')
    parser.add_argument('source', type=Path, help='Source d2l-en directory')
    parser.add_argument('output', type=Path, help='Output d2l/ package directory')
    parser.add_argument('--frameworks', nargs='*',
                        default=['pytorch', 'tensorflow', 'jax', 'mxnet'],
                        help='Frameworks to build (default: all)')
    parser.add_argument('--config', type=Path, default=None,
                        help='Path to config.ini (default: <source>/config.ini)')
    args = parser.parse_args()

    config_path = args.config or args.source / 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    files = list(CHAPTER_NUMBERING.keys())

    # Copy __init__.py
    init_src = args.source / 'd2l' / '__init__.py'
    init_dst = args.output / '__init__.py'
    init_dst.parent.mkdir(parents=True, exist_ok=True)
    if init_src.exists():
        init_dst.write_text(init_src.read_text())
        print(f'Copied __init__.py')

    # Build each framework
    for fw in args.frameworks:
        fw_tab = fw
        section = f'library-{fw}'
        if not config.has_section(section):
            print(f'\nSKIP {fw}: no [{section}] in config.ini')
            continue

        lib_config = dict(config[section])
        lib_file = lib_config.get('lib_file', f'd2l/{fw}.py')
        output_file = args.output / Path(lib_file).name

        print(f'\n=== {fw} → {output_file} ===')

        # Use existing preamble from the current library file
        existing_lib = args.source / lib_file
        if existing_lib.exists():
            # Copy preamble (everything before the WARNING header)
            existing = existing_lib.read_text()
            if '#####' in existing:
                preamble = existing[:existing.index('#####')]
            else:
                preamble = existing
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(preamble)

        build_library(args.source, output_file, fw_tab, lib_config, files)

    # Copy setup.py
    setup_src = args.source / 'setup.py'
    setup_dst = args.output.parent / 'setup.py'
    if setup_src.exists() and not setup_dst.exists():
        setup_dst.write_text(setup_src.read_text())
        print(f'\nCopied setup.py')

    print('\nDone.')


if __name__ == '__main__':
    main()
