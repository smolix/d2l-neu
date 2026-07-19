#!/usr/bin/env python3
"""Generate per-framework Jupyter notebooks from d2l-en source.

For each framework (pytorch, tensorflow, jax, mxnet), generates a set of
single-framework .qmd files, then converts them to .ipynb via `quarto convert`.

Usage:
    python tools/gen_notebooks.py <source_dir> <output_dir> [--frameworks pytorch jax]
"""

import re
import os
import json
import shutil
import argparse
import subprocess
import time
from pathlib import Path


def _write_if_changed(path, content_bytes):
    """Write content to path only if it differs; preserves mtime otherwise."""
    if path.exists() and path.read_bytes() == content_bytes:
        return False
    path.write_bytes(content_bytes)
    return True


def _normalize_source(src):
    """Return cell source as a single string for stable code-equality
    comparisons. nbformat stores source as either a list of lines or a
    flat string."""
    if isinstance(src, list):
        return ''.join(src)
    return src or ''


def _merge_outputs_from_prev(ipynb_path, prev_nb):
    """Restore executed outputs from `prev_nb` into the just-regenerated
    notebook at `ipynb_path` for any code cell whose id and source still
    match. Returns True if any cell's code changed (so the file genuinely
    needs to be re-executed).

    Why this exists: `quarto convert` always emits a fresh, never-executed
    notebook, so a naive "byte-equal?" check after regeneration always
    fails and downstream `.executed` stamps get invalidated — even when
    the source .md didn't change a single character of code. By copying
    outputs forward per cell we keep already-executed notebooks usable
    across no-op regenerations and across regenerations that only edited
    prose. Cells with edited code, or new cells, get no outputs back —
    which is correct, since their outputs would be stale.
    """
    new_nb = json.loads(ipynb_path.read_text(encoding='utf-8'))
    prev_by_id = {}
    for cell in prev_nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        cid = cell.get('id') or cell.get('metadata', {}).get('id')
        if cid:
            prev_by_id[cid] = cell
    any_code_changed = False
    for cell in new_nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        cid = cell.get('id') or cell.get('metadata', {}).get('id')
        prev = prev_by_id.get(cid) if cid else None
        if prev is None:
            # New cell or unidentifiable cell — no outputs to restore.
            if _normalize_source(cell.get('source')).strip():
                any_code_changed = True
            continue
        if _normalize_source(cell.get('source')) != _normalize_source(prev.get('source')):
            any_code_changed = True
            continue
        # Code is unchanged → copy execution state forward.
        if 'outputs' in prev:
            cell['outputs'] = prev['outputs']
        if 'execution_count' in prev:
            cell['execution_count'] = prev['execution_count']
    ipynb_path.write_text(
        json.dumps(new_nb, indent=1, ensure_ascii=False),
        encoding='utf-8')
    return any_code_changed

# Reuse the preprocessor's parsing and directive translation
import sys
sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import (
    FRAMEWORKS, FRAMEWORK_DISPLAY, CHAPTER_NUMBERING,
    parse_blocks, extract_tab, is_boilerplate, is_python_block,
    clean_save_markers, translate_directives, CodeBlock, MarkdownBlock,
    CodeTabSet, TocBlock,
)
from build_lib import flatten_tab_branches, LIB_ONLY_FILES


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


# Quarto-only constructs that leak into notebook markdown cells via the
# shared .md → .qmd directive translation, but that a plain notebook
# renderer (VS Code / Jupyter markdown-it + KaTeX) cannot parse:
#   - `{#eq-…}` / `{#sec-…}` ID attributes. After a `$$…$$` display
#     equation the trailing `{#eq-…}` breaks KaTeX's delimiter matching
#     (it swallows following `$`), which then mangles later equations.
#   - `@eq-…` / `@sec-…` cross-references render as broken `@…` literals.
# Notebooks are not Quarto, so we strip the IDs and render the refs as
# readable words. Equation numbering/cross-refs remain correct in the
# Quarto HTML/PDF build (which sees the un-stripped .qmd); this only
# affects the notebook view used for authoring/review.
_QUARTO_ID_RE = re.compile(r'[ \t]*\{#(?:eq|sec|fig|tbl|lst)-[^}]*\}')
_QUARTO_XREF_RE = re.compile(r'@(eq|sec|fig|tbl)-[A-Za-z0-9_-]+')
_XREF_WORD = {'eq': 'the equation', 'sec': 'that section',
              'fig': 'the figure', 'tbl': 'the table'}


def notebookify_markdown(text):
    """Strip Quarto-only IDs/cross-refs from a notebook markdown cell so it
    renders cleanly in VS Code / Jupyter. Idempotent."""
    text = _QUARTO_ID_RE.sub('', text)
    text = _QUARTO_XREF_RE.sub(lambda m: _XREF_WORD[m.group(1)], text)
    return text


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

    # Pass 1b: strip Quarto-only IDs/cross-refs from markdown cells so they
    # render in a plain notebook frontend (VS Code/Jupyter) without breaking
    # KaTeX. (Preserves code cells and their outputs untouched.)
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'markdown':
            continue
        src = cell.get('source')
        joined = ''.join(src) if isinstance(src, list) else (src or '')
        cell['source'] = notebookify_markdown(joined).splitlines(keepends=True)

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
    from d2l_preprocess import strip_slide_divs
    text = Path(src_path).read_text(encoding='utf-8')
    text = strip_slide_divs(text)

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
    parser.add_argument('--files', nargs='*', default=None,
                        help='Source .md files to regenerate, relative to source root')
    args = parser.parse_args()

    src = args.source
    if args.files is not None:
        files = args.files
    else:
        # Numbered book pages plus code-bearing standalone notebooks.  The
        # manifest scanner and Makefile already discover every chapter_*/*.md;
        # keeping the same source universe prevents batch cleanup from deleting
        # notebook-only supplements that intentionally stay out of the book TOC.
        standalone = sorted(str(path.relative_to(src))
                            for path in src.glob("chapter_*/*.md"))
        files = list(dict.fromkeys([*CHAPTER_NUMBERING.keys(), *standalone]))
        # Build-only lib-extraction sources (e.g. legacy-attention-lib.md) carry
        # #@save blocks for `make lib` but are never rendered or executed as
        # notebooks — their cells reference nn/tf/… with no imports cell. Keep
        # them out of the notebook set so they are neither generated nor run.
        files = [f for f in files if f not in set(LIB_ONLY_FILES)]

    # In --files mode (per-notebook Make invocation), suppress the verbose
    # batch-style banner / counts. Each per-notebook rebuild fires gen
    # independently; the existing recipe `tee -a`s into one log per run, so
    # without this gating the log would contain N copies of the same
    # "=== PyTorch === / Generated 1 / Converting / Produced 1 / Done."
    # block for an N-notebook rebuild. The batch path (no --files) keeps
    # the original output.
    quiet = args.files is not None

    for fw in args.frameworks:
        fw_dir = args.output / fw
        if not quiet:
            print(f'\n=== {FRAMEWORK_DISPLAY.get(fw, fw)} ===')

        # Cleanup pass: remove orphans (output .ipynb / .qmd / .executed
        # whose source .md no longer exists) and any leftover .qmd files
        # from previous runs (.qmd is a transient byproduct of `quarto
        # convert`). We deliberately KEEP existing .ipynb files for files
        # that are still part of the current source set, so the per-cell
        # output-preserving merge below can copy executed outputs forward.
        # Wiping them indiscriminately (as the previous full-gen path did)
        # made every regen reset all executions to scratch.
        if fw_dir.exists() and args.files is None:
            expected_stems = set()
            for rel in files:
                src_file = src / rel
                if not src_file.exists():
                    continue
                if not file_supports_framework(src_file, fw):
                    continue
                expected_stems.add(str(Path(rel).with_suffix('')))
            for child in fw_dir.iterdir():
                if child.is_symlink() or child.name in ('img', 'data'):
                    continue
                if child.is_dir():
                    for sub in child.iterdir():
                        if sub.is_dir():
                            # Sub-sub directories (unexpected) → leave alone.
                            continue
                        rel_stem = str(sub.relative_to(fw_dir).with_suffix(''))
                        suf = sub.suffix
                        # .qmd is always rebuilt from source; safe to nuke.
                        if suf == '.qmd':
                            sub.unlink()
                            continue
                        # Drop output / stamp files whose source has gone.
                        if suf in ('.ipynb', '.executed') \
                                and rel_stem not in expected_stems:
                            sub.unlink()
                else:
                    # Top-level transient files (e.g. .generated stamp from
                    # an old layout). Keep MANIFEST.mk and dotfiles.
                    if child.suffix == '.qmd':
                        child.unlink()

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
            # Content-aware write: keep mtime if file is byte-identical so
            # downstream per-notebook .executed stamps don't get invalidated
            # by a no-op regeneration.
            _write_if_changed(dst_file, output.encode('utf-8'))
            converted += 1

        if not quiet:
            print(f'  Generated {converted} .qmd files in {fw_dir}'
                  + (f' (skipped {skipped} files with no {fw} tabs)' if skipped else ''))

        if args.convert:
            if not quiet:
                print(f'  Converting to .ipynb...')
            if args.files is not None:
                # Per-file mode: convert only the qmds we just emitted for
                # the requested sources. Without this, a per-notebook Make
                # invocation would still re-convert every leftover qmd in
                # the framework tree from a previous full run, defeating
                # the granularity.
                qmd_files = []
                for rel in args.files:
                    qmd = fw_dir / rel.replace('.md', '.qmd')
                    if qmd.exists():
                        qmd_files.append(qmd)
            else:
                qmd_files = sorted(fw_dir.rglob('*.qmd'))

            def convert_one(qmd):
                ipynb = qmd.with_suffix('.ipynb')
                try:
                    # quarto writes .ipynb next to .qmd. quarto convert
                    # produces a fresh notebook with no outputs, so a naive
                    # byte-compare always sees the previously-executed file
                    # as "different" and wipes outputs. Instead, snapshot
                    # the old notebook and, after regen, copy outputs back
                    # into any code cell whose id+source matches.
                    prev_nb = None
                    prev_stat = None
                    if ipynb.exists():
                        try:
                            prev_nb = json.loads(ipynb.read_text(
                                encoding='utf-8'))
                            prev_stat = ipynb.stat()
                        except Exception:
                            prev_nb = None
                    result = subprocess.run(
                        ['quarto', 'convert', str(qmd)],
                        capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        status = 'new' if prev_nb is None else 'updated'
                        if ipynb.exists():
                            patch_ipynb_ids(ipynb, fw)
                            if prev_nb is not None:
                                changed = _merge_outputs_from_prev(
                                    ipynb, prev_nb)
                                status = 'code-changed' if changed else 'unchanged'
                                # Code cells unchanged → we just produced
                                # an identical .ipynb. Advance BOTH the
                                # .ipynb and its .executed sibling (if it
                                # exists) to the same fresh mtime so:
                                #   - make sees .ipynb as up-to-date wrt
                                #     its source .md (no rebuild loop on
                                #     prose-only or no-op edits),
                                #   - any previously-valid .executed stamp
                                #     stays at-or-above the .ipynb mtime
                                #     so make doesn't re-execute.
                                if not changed:
                                    now_ns = time.time_ns()
                                    os.utime(ipynb, ns=(now_ns, now_ns))
                                    executed = ipynb.with_suffix('.executed')
                                    if executed.exists():
                                        os.utime(executed, ns=(now_ns, now_ns))
                        qmd.unlink()
                        if quiet:
                            rel = ipynb.relative_to(fw_dir)
                            print(f'[{fw}] {rel}: {status}')
                        return True
                    else:
                        if quiet:
                            rel = ipynb.relative_to(fw_dir)
                            print(f'[{fw}] {rel}: FAIL (quarto convert rc={result.returncode})')
                        return False
                except subprocess.TimeoutExpired:
                    if quiet:
                        rel = ipynb.relative_to(args.output)
                        print(f'[{fw}] {rel}: FAIL (timeout)')
                    return False

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=10) as pool:
                results = list(pool.map(convert_one, qmd_files))

            nb_count = sum(results)
            if not quiet:
                print(f'  Produced {nb_count} .ipynb files')

        # Note: per-framework MANIFEST.mk is written by
        # tools/scan_notebook_manifests.py at Make-parse time, not here.

    if not quiet:
        print(f'\nDone. Notebooks in {args.output}/')


if __name__ == '__main__':
    main()
