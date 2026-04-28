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
    """Convert :begin_tab:/:end_tab: by keeping only the target framework.

    Wraps the kept tab content in `<!--d2l:tab fw-->...<!--/d2l:tab-->`
    sentinels so emit_notebook_qmd can later set the prose header's
    `fw=` field to `<framework>` (for sync_back), distinguishing
    begin_tab-derived prose from regular shared prose (which gets
    `fw=all`).
    """
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

        for match in group:
            fw_key = match.group(1)
            content = match.group(2).strip()
            fws = [f.strip() for f in fw_key.split(',')]
            if framework in fws:
                parts.append(
                    f'<!--d2l:tab {fw_key}-->\n{content}\n<!--/d2l:tab-->\n')
                break

        last_end = group[-1].end()

    parts.append(text[last_end:])
    return ''.join(parts)


def emit_notebook_qmd(blocks, framework, file_slug='cell'):
    """Emit single-framework .qmd for notebook conversion.

    Each markdown block is prepended with a hidden HTML comment header
    `<!-- d2l:prose id=<file>-md-<seq> fw=<framework> -->`. The header
    is invisible in rendered output but serves as a stable anchor for
    sync_back.py to find the corresponding source location.
    """
    parts = []
    md_seq = 0

    # Match begin_tab sentinels emitted by convert_prose_tabs_single:
    #   <!--d2l:tab pytorch-->...<!--/d2l:tab-->
    # Markdown blocks containing one or more sentinels are split into
    # multiple emitted cells: each sentinel chunk becomes its own cell
    # with fw=<framework>; the surrounding shared prose becomes
    # separate fw=all cells.
    tab_sentinel_re = re.compile(
        r'<!--d2l:tab\s+([^>]+?)-->\s*(.*?)\s*<!--/d2l:tab-->',
        re.DOTALL)

    def emit_md_cell(text, header_fw):
        nonlocal md_seq
        text = text.strip()
        if not text:
            return
        md_seq += 1
        header = (f'<!-- d2l:prose id={file_slug}-md-{md_seq} '
                  f'fw={header_fw} -->')
        parts.append(f'{header}\n\n{text}')

    for block in blocks:
        if isinstance(block, MarkdownBlock):
            text = '\n'.join(block.lines)
            text = translate_directives(text)

            # Split at sentinel boundaries
            cursor = 0
            for m in tab_sentinel_re.finditer(text):
                pre = text[cursor:m.start()]
                if pre.strip():
                    emit_md_cell(pre, 'all')
                fw_key = m.group(1).strip()
                inner = m.group(2)
                if inner.strip():
                    emit_md_cell(inner, fw_key)
                cursor = m.end()
            tail = text[cursor:]
            if tail.strip():
                emit_md_cell(tail, 'all')

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


_PROSE_HEADER_LINE_RE = re.compile(
    r'^<!--\s*d2l:prose\s+id=([^\s]+)\s+fw=([^\s]+)\s*-->\s*$')


def _split_prose_cell(cell):
    """If a markdown cell has multiple `<!-- d2l:prose -->` headers
    (because quarto convert merged consecutive md cells), split it
    into multiple cells. Returns list[cell-dict].

    Each split cell carries one header. The header stays as the first
    line of source so sync_back can read it.
    """
    src = cell.get('source')
    if isinstance(src, list):
        lines = [ln.rstrip('\n') for ln in src]
    elif isinstance(src, str):
        lines = src.split('\n')
    else:
        return [cell]

    # Find header indices
    headers = [i for i, ln in enumerate(lines)
               if _PROSE_HEADER_LINE_RE.match(ln)]
    if len(headers) <= 1:
        return [cell]

    out = []
    for k, start in enumerate(headers):
        end = headers[k + 1] if k + 1 < len(headers) else len(lines)
        chunk = lines[start:end]
        # Strip trailing blank lines
        while chunk and not chunk[-1].strip():
            chunk.pop()
        if not chunk:
            continue
        new_cell = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [ln + '\n' for ln in chunk[:-1]] + [chunk[-1]],
        }
        out.append(new_cell)
    return out


def patch_ipynb_ids(ipynb_path: Path, framework: str):
    """Post-process a .ipynb produced by `quarto convert`:

    - Extract `#| label: <id>` lines from code cells and turn them into
      proper nbformat IDs (top-level cell.id and metadata.id), then
      strip the label line from cell.source.
    - Split markdown cells that contain multiple `<!-- d2l:prose -->`
      headers (quarto convert merges consecutive md cells; we split
      them back so each prose cell stands alone).
    - Set notebook-level metadata.kernelspec to point at the per-
      framework kernel (`d2l-<fw>`), so VS Code auto-selects the
      right interpreter.

    Idempotent.
    """
    text = ipynb_path.read_text(encoding='utf-8')
    nb = json.loads(text)

    # Pass 1: split merged markdown cells
    new_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            new_cells.extend(_split_prose_cell(cell))
        else:
            new_cells.append(cell)
    nb['cells'] = new_cells

    # Pass 2: extract `#| label:` from code cells
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source')
        if not source:
            continue
        first = source[0] if isinstance(source, list) else source.split('\n', 1)[0]
        m = _LABEL_LINE_RE.match(first.rstrip('\n'))
        if not m:
            continue
        cid = m.group(1)
        cell['id'] = cid
        cell.setdefault('metadata', {})['id'] = cid
        if isinstance(source, list):
            cell['source'] = source[1:]
        else:
            cell['source'] = source.split('\n', 1)[1] if '\n' in source else ''

    # Notebook-level kernelspec
    nb_meta = nb.setdefault('metadata', {})
    nb_meta['kernelspec'] = {
        'name': f'd2l-{framework}',
        'display_name': f'd2l ({FRAMEWORK_DISPLAY.get(framework, framework)})',
        'language': 'python',
    }

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

    # Derive a file slug for prose-cell IDs
    src_path = Path(src_path)
    if src_path.stem == 'index':
        slug = src_path.parent.name.removeprefix('chapter_')
    else:
        slug = src_path.stem

    # Emit single-framework output
    output = emit_notebook_qmd(blocks, framework, file_slug=slug)

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
                            patch_ipynb_ids(ipynb, fw)
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
