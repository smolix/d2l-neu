#!/usr/bin/env python3
"""Generate per-framework Jupyter notebooks from d2l-en source.

For each framework (pytorch, tensorflow, jax, mxnet), generates a set of
single-framework .qmd files, then converts them to .ipynb via `quarto convert`.

Usage:
    python tools/gen_notebooks.py <source_dir> <output_dir> [--frameworks pytorch jax]
"""

import re
import argparse
import subprocess
from pathlib import Path

# Reuse the preprocessor's parsing and directive translation
import sys
sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import (
    FRAMEWORKS, FRAMEWORK_DISPLAY, CHAPTER_NUMBERING,
    parse_blocks, extract_tab, is_boilerplate, is_python_block,
    clean_save_markers, translate_directives, CodeBlock, MarkdownBlock,
    CodeTabSet, TocBlock,
)


def convert_prose_tabs_single(text, framework):
    """Convert :begin_tab:/:end_tab: by keeping only the target framework."""
    tab_pattern = re.compile(
        r':begin_tab:`([^`]+)`\s*\n(.*?)\n?:end_tab:', re.DOTALL)

    matches = list(tab_pattern.finditer(text))
    if not matches:
        return text

    groups = []
    current_group = [matches[0]]
    for i in range(1, len(matches)):
        between = text[matches[i-1].end():matches[i].start()]
        if between.strip() == '':
            current_group.append(matches[i])
        else:
            groups.append(current_group)
            current_group = [matches[i]]
    groups.append(current_group)

    parts = []
    last_end = 0
    for group in groups:
        group_start = group[0].start()
        parts.append(text[last_end:group_start])

        # Find content for the target framework
        for match in group:
            fw_key = match.group(1)
            content = match.group(2).strip()
            fws = [f.strip() for f in fw_key.split(',')]
            if framework in fws:
                parts.append(content + '\n')
                break
        # If framework not found in group, skip entirely

        last_end = group[-1].end()

    parts.append(text[last_end:])
    return ''.join(parts)


def emit_notebook_qmd(blocks, framework):
    """Emit single-framework .qmd for notebook conversion."""
    parts = []

    for block in blocks:
        if isinstance(block, MarkdownBlock):
            text = '\n'.join(block.lines)
            text = translate_directives(text)
            parts.append(text)

        elif isinstance(block, CodeBlock):
            code = '\n'.join(block.lines)

            if not is_python_block(block.info) and block.tab is None:
                lang = block.info or ''
                parts.append(f'\n```{lang}\n{code}\n```\n')

            elif block.tab is None or block.tab == 'all':
                parts.append(f'\n```{{python}}\n{code}\n```\n')

            elif framework in (block.tab or ''):
                parts.append(f'\n```{{python}}\n{code}\n```\n')
            # else: skip (different framework)

        elif isinstance(block, CodeTabSet):
            if framework in block.tabs:
                code = '\n'.join(block.tabs[framework])
                parts.append(f'\n```{{python}}\n{code}\n```\n')
            # else: skip (framework has no code for this section)

        elif isinstance(block, TocBlock):
            pass

    output = '\n'.join(parts)
    output = re.sub(r'\n{4,}', '\n\n\n', output)
    return output


def convert_file_notebook(src_path, framework):
    """Convert a d2l .md file to a single-framework .qmd for notebooks."""
    text = Path(src_path).read_text(encoding='utf-8')

    # Convert prose tabs: keep only target framework
    text = convert_prose_tabs_single(text, framework)

    # Parse and group code blocks (reuses preprocessor logic)
    blocks = parse_blocks(text)

    # Group framework-specific code blocks
    from d2l_preprocess import group_code_tabs
    blocks = group_code_tabs(blocks, framework)

    # Emit single-framework output
    output = emit_notebook_qmd(blocks, framework)

    # Add minimal front matter for Jupyter
    front_matter = '---\njupyter: python3\nexecute:\n  enabled: false\n---\n'
    output = front_matter + output

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-framework Jupyter notebooks from d2l-en')
    parser.add_argument('source', type=Path, help='Source d2l-en directory')
    parser.add_argument('output', type=Path, help='Output directory')
    parser.add_argument('--frameworks', nargs='*',
                        default=['pytorch', 'tensorflow', 'jax', 'mxnet'],
                        help='Frameworks to generate (default: all)')
    parser.add_argument('--convert', action='store_true',
                        help='Also run quarto convert to produce .ipynb files')
    args = parser.parse_args()

    src = args.source
    files = list(CHAPTER_NUMBERING.keys())

    for fw in args.frameworks:
        fw_dir = args.output / fw
        print(f'\n=== {FRAMEWORK_DISPLAY.get(fw, fw)} ===')

        converted = 0
        for rel in files:
            src_file = src / rel
            if not src_file.exists():
                continue

            dst_file = fw_dir / rel.replace('.md', '.qmd')
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            output = convert_file_notebook(src_file, fw)

            # Skip files that have no code at all (pure prose chapters)
            has_code = '```{python}' in output
            dst_file.write_text(output, encoding='utf-8')
            converted += 1

        print(f'  Generated {converted} .qmd files in {fw_dir}')

        if args.convert:
            print(f'  Converting to .ipynb...')
            nb_count = 0
            for qmd in sorted(fw_dir.rglob('*.qmd')):
                try:
                    result = subprocess.run(
                        ['quarto', 'convert', str(qmd)],
                        capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        nb_count += 1
                        # Remove .qmd after conversion
                        qmd.unlink()
                    else:
                        print(f'    WARN: {qmd.name}: {result.stderr.strip()[:80]}')
                except subprocess.TimeoutExpired:
                    print(f'    WARN: {qmd.name}: timeout')
            print(f'  Produced {nb_count} .ipynb files')

    print(f'\nDone. Notebooks in {args.output}/')


if __name__ == '__main__':
    main()
