#!/usr/bin/env python3
"""Generate per-framework Jupyter notebooks from d2l-en source.

For each framework (pytorch, tensorflow, jax, mxnet), generates a set of
single-framework .qmd files, then converts them to .ipynb via `quarto convert`.

Usage:
    python tools/gen_notebooks.py <source_dir> <output_dir> [--frameworks pytorch jax]
"""

import re
import json
import shutil
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
from build_lib import flatten_tab_branches


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
            id_prefix = f'#| label: {block.cell_id}\n' if block.cell_id else ''

            if not is_python_block(block.info) and block.tab is None:
                lang = block.info or ''
                parts.append(f'\n```{lang}\n{code}\n```\n')

            elif block.tab is None or block.tab == 'all':
                code = flatten_tab_branches(code, framework)
                parts.append(f'\n```{{python}}\n{id_prefix}{code}\n```\n')

            elif framework in (block.tab or ''):
                code = flatten_tab_branches(code, framework)
                parts.append(f'\n```{{python}}\n{id_prefix}{code}\n```\n')
            # else: skip (different framework)

        elif isinstance(block, CodeTabSet):
            if framework in block.tabs:
                code = '\n'.join(block.tabs[framework])
                code = flatten_tab_branches(code, framework)
                cid = block.ids.get(framework)
                id_prefix = f'#| label: {cid}\n' if cid else ''
                parts.append(f'\n```{{python}}\n{id_prefix}{code}\n```\n')
            # else: skip (framework has no code for this section)

        elif isinstance(block, TocBlock):
            pass

    output = '\n'.join(parts)
    output = re.sub(r'\n{4,}', '\n\n\n', output)
    return output


def file_supports_framework(src_path, framework):
    """Return True iff the source .md should produce a notebook for *framework*.

    A file supports framework X if:
    1. A code block is explicitly tagged for X (#@tab X or %%tab X), OR
    2. tab.interact_select lists X, OR
    3. The file has code but no framework-specific tags at all (only #@tab all
       / %%tab all, or bare blocks) — generated as pytorch notebooks only.
    """
    text = Path(src_path).read_text(encoding='utf-8')

    has_any_specific = False
    for m in re.finditer(r'^(?:%%tab|#@tab)\s+([^\n]+)$', text, re.MULTILINE):
        fws = [t.strip() for t in m.group(1).split(',')]
        if framework in fws:
            return True
        if fws != ['all']:
            has_any_specific = True

    for m in re.finditer(r'tab\.interact_select\(([^)]+)\)', text):
        fws = re.findall(r"['\"](\w+)['\"]", m.group(1))
        if framework in fws:
            return True
        if fws:
            has_any_specific = True

    if not has_any_specific and framework == 'pytorch':
        if re.search(r'```\{\.python', text):
            return True

    return False


_LABEL_LINE_RE = re.compile(r'^#\|\s*label:\s*([a-z][a-z0-9-]*)\s*$')


def patch_ipynb_ids(ipynb_path: Path):
    """Post-process a .ipynb produced by `quarto convert`: extract
    `#| label: <id>` lines from code cells and turn them into proper
    nbformat IDs (top-level cell.id and metadata.id), then strip the
    label line from cell.source.

    Idempotent. Safe to run on .ipynb files without label lines.
    """
    text = ipynb_path.read_text(encoding='utf-8')
    nb = json.loads(text)
    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source')
        if not source:
            continue
        # cell.source is a list of lines (with trailing \n) per nbformat.
        first = source[0] if isinstance(source, list) else source.split('\n', 1)[0]
        m = _LABEL_LINE_RE.match(first.rstrip('\n'))
        if not m:
            continue
        cid = m.group(1)
        cell['id'] = cid
        cell.setdefault('metadata', {})['id'] = cid
        # Strip the label line from source. The cell may begin with a
        # blank trailing-newline line if the label was the only line; in
        # practice it is always followed by code, so just drop the first.
        if isinstance(source, list):
            cell['source'] = source[1:]
        else:
            cell['source'] = source.split('\n', 1)[1] if '\n' in source else ''
        changed = True
    if changed:
        ipynb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                              encoding='utf-8')


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

        # Wipe stale chapter output — skip the img/ and data/ symlinks that
        # build.sh recreates, since those are cross-run infrastructure.
        if fw_dir.exists():
            for child in fw_dir.iterdir():
                if child.is_symlink() or child.name in ('img', 'data'):
                    continue
                shutil.rmtree(child) if child.is_dir() else child.unlink()

        converted = 0
        skipped = 0
        for rel in files:
            src_file = src / rel
            if not src_file.exists():
                continue

            if not file_supports_framework(src_file, fw):
                skipped += 1
                continue

            dst_file = fw_dir / rel.replace('.md', '.qmd')
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            output = convert_file_notebook(src_file, fw)

            # Skip files that have no code at all (pure prose chapters)
            has_code = '```{python}' in output
            dst_file.write_text(output, encoding='utf-8')
            converted += 1

        print(f'  Generated {converted} .qmd files in {fw_dir}'
              + (f' (skipped {skipped} files with no {fw} tabs)' if skipped else ''))

        if args.convert:
            print(f'  Converting to .ipynb...')
            qmd_files = sorted(fw_dir.rglob('*.qmd'))

            def convert_one(qmd):
                try:
                    result = subprocess.run(
                        ['quarto', 'convert', str(qmd)],
                        capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        ipynb = qmd.with_suffix('.ipynb')
                        if ipynb.exists():
                            patch_ipynb_ids(ipynb)
                        qmd.unlink()
                        return True
                    else:
                        return False
                except subprocess.TimeoutExpired:
                    return False

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=10) as pool:
                results = list(pool.map(convert_one, qmd_files))

            nb_count = sum(results)
            print(f'  Produced {nb_count} .ipynb files')

    print(f'\nDone. Notebooks in {args.output}/')


if __name__ == '__main__':
    main()
