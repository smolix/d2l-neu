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

def flatten_tab_branches(code, framework):
    """Flatten tab.selected() branches, keeping only the target framework.

    Transforms:
        if tab.selected('pytorch'):
            x = torch.tensor(1)
        if tab.selected('tensorflow'):
            x = tf.constant(1)

    Into (for framework='pytorch'):
        x = torch.tensor(1)
    """
    lines = code.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(\s*)if tab\.selected\(([^)]+)\):', line)
        if m:
            indent = m.group(1)
            frameworks_str = m.group(2)
            # Parse the framework list from the if condition
            fws = re.findall(r"'(\w+)'", frameworks_str)
            keep = framework in fws

            # Collect the indented body
            body_lines = []
            i += 1
            body_indent = None
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

            if keep and body_lines:
                # De-indent: remove the body indent relative to the if statement
                dedent = body_indent - len(indent) if body_indent else 4
                for bl in body_lines:
                    if bl.strip():
                        result.append(indent + bl[body_indent:])
                    else:
                        result.append(bl)
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


def save_block(source):
    """Extract the code block following a #@save marker.

    Returns the extracted code, or empty string if no #@save found.
    """
    lines = source.split('\n')
    block_lines = []
    for i, line in enumerate(lines):
        m = re.search(r'#\s*@save', line)
        if m:
            # Keep code BEFORE the marker on the same line
            prefix = line[:m.start()].rstrip()
            if prefix:
                block_lines.append(prefix)
            # Collect the indented block that follows
            for j in range(i + 1, len(lines)):
                l = lines[j]
                if l.startswith(' ') or l.startswith('\t') or l.strip() == '':
                    block_lines.append(l)
                else:
                    # Hit a non-indented non-empty line → start of next block
                    # Include this line only if it's a def/class (continuation)
                    if l.startswith('def ') or l.startswith('class ') or l.startswith('@'):
                        block_lines.append(l)
                        # Continue collecting indented body
                        for k in range(j + 1, len(lines)):
                            l2 = lines[k]
                            if l2.startswith(' ') or l2.startswith('\t') or l2.strip() == '':
                                block_lines.append(l2)
                            else:
                                break
                    break
            break

    return '\n'.join(block_lines).rstrip() if block_lines else ''


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

            extracted = save_block(code)
            if extracted:
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

    # Refactor: add docstring labels, merge add_to_class methods
    all_blocks = add_docstring_labels(all_blocks)
    all_blocks = merge_add_to_class(all_blocks)

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
                preamble = ''
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
