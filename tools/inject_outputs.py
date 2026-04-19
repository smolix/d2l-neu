#!/usr/bin/env python3
"""Inject executed notebook outputs into .qmd files for rendering.

Reads executed .ipynb files from _notebooks/{fw}/ and injects cell outputs
into .qmd files so Quarto renders the book with visible outputs.

Two modes:
  html  - Inject all frameworks' outputs into multi-framework chapter_*/*.qmd
  pdf   - Inject single-framework outputs into single-framework _pdf/*.qmd

Usage:
    python tools/inject_outputs.py html [--notebooks-dir _notebooks] [--img-dir img/outputs]
    python tools/inject_outputs.py pdf --framework pytorch [--pdf-dir _pdf] [--notebooks-dir _notebooks] [--img-dir img/outputs]
"""

import re
import os
import json
import base64
import hashlib
import argparse
from pathlib import Path

FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'mxnet']
FRAMEWORK_DISPLAY = {
    'pytorch': 'PyTorch',
    'tensorflow': 'TensorFlow',
    'jax': 'JAX',
    'mxnet': 'MXNet',
}
DISPLAY_TO_FW = {v: k for k, v in FRAMEWORK_DISPLAY.items()}
FALLBACK_ORDER = ['pytorch', 'mxnet', 'tensorflow', 'jax']

OUTPUT_START = '<!-- d2l:output -->'
OUTPUT_END = '<!-- /d2l:output -->'
MAX_TEXT_LINES = 20


# ── QMD Parsing ──────────────────────────────────────────────

class QmdCell:
    __slots__ = ('start_line', 'end_line', 'source', 'framework')

    def __init__(self, start_line, end_line, source, framework):
        self.start_line = start_line
        self.end_line = end_line
        self.source = source
        self.framework = framework  # None = shared / tab-all


def parse_qmd_cells(text):
    """Parse .qmd text and return ordered list of Python code cells.

    Tracks framework tab context via panel-tabset divs so each cell
    knows which framework it belongs to (or None for shared cells).
    """
    lines = text.split('\n')
    cells = []
    i = 0
    div_stack = []  # True = fw-tabset, False = other div
    current_fw = None

    while i < len(lines):
        line = lines[i]

        # Fenced div opener: ::: {.attrs}
        if re.match(r'^:{3,}\s*\{', line):
            is_fw_tabset = ('panel-tabset' in line
                            and 'group="framework"' in line)
            div_stack.append(is_fw_tabset)
            if is_fw_tabset:
                current_fw = None
            i += 1
            continue

        # Fenced div closer: bare :::
        if re.match(r'^:{3,}\s*$', line) and div_stack:
            was_fw_tabset = div_stack.pop()
            if was_fw_tabset:
                current_fw = None
            i += 1
            continue

        # Framework tab heading within a tabset
        if any(div_stack):
            m = re.match(r'^##\s+(PyTorch|TensorFlow|JAX|MXNet)\s*$', line)
            if m and True in div_stack:
                current_fw = DISPLAY_TO_FW[m.group(1)]
                i += 1
                continue

        # Python code fence: ```{python} or ```python
        if re.match(r'^```(\{python\}|python)\s*$', line):
            start = i
            code_lines = []
            i += 1
            while i < len(lines) and not re.match(r'^```\s*$', lines[i]):
                code_lines.append(lines[i])
                i += 1
            end = i  # closing ```

            fw = current_fw if (True in div_stack) else None
            cells.append(QmdCell(start, end, '\n'.join(code_lines), fw))
            i += 1
            continue

        # Skip non-Python code blocks
        if re.match(r'^```', line) and not line.startswith('````'):
            i += 1
            while i < len(lines) and not re.match(r'^```\s*$', lines[i]):
                i += 1
            i += 1
            continue

        i += 1

    return cells


def build_cell_map(cells, framework):
    """Map qmd cell index → ipynb cell index for a given framework.

    A cell belongs to a framework if it's shared (framework=None) or
    explicitly tagged for that framework.
    """
    mapping = {}
    ipynb_idx = 0
    for i, cell in enumerate(cells):
        if cell.framework is None or cell.framework == framework:
            mapping[i] = ipynb_idx
            ipynb_idx += 1
    return mapping


# ── Notebook Output Loading ──────────────────────────────────

def load_ipynb_outputs(path):
    """Load per-code-cell outputs from an executed .ipynb.

    Returns list of list[dict] — one entry per code cell,
    each containing the cell's raw output dicts.
    """
    nb = json.loads(Path(path).read_bytes())
    return [
        cell.get('outputs', [])
        for cell in nb['cells']
        if cell['cell_type'] == 'code'
    ]


# ── Output Rendering ─────────────────────────────────────────

def _output_fingerprint(raw_outputs):
    """Hash the content of outputs for deduplication across frameworks."""
    h = hashlib.sha256()
    for out in raw_outputs:
        otype = out.get('output_type', '')
        if otype == 'error':
            continue
        data = out.get('data', {})
        if 'image/png' in data:
            h.update(base64.b64decode(data['image/png']))
        elif otype == 'stream':
            h.update(''.join(out.get('text', [])).encode())
        elif 'text/plain' in data:
            plain = data['text/plain']
            h.update((plain if isinstance(plain, str)
                       else ''.join(plain)).encode())
    return h.hexdigest()


def format_cell_output(raw_outputs, img_dir, cell_id, qmd_parent, mode):
    """Render notebook outputs as Quarto markdown.

    Args:
        raw_outputs: list of output dicts from the .ipynb cell
        img_dir: absolute Path to save images into
        cell_id: unique string for image filenames
        qmd_parent: absolute Path of the .qmd file's directory
        mode: 'html' or 'pdf'

    Returns:
        Quarto markdown string, or '' if no renderable output.
    """
    if not raw_outputs:
        return ''

    parts = []
    img_idx = 0

    for out in raw_outputs:
        otype = out.get('output_type', '')
        if otype == 'error':
            continue

        data = out.get('data', {})

        # Image output
        if 'image/png' in data:
            img_idx += 1
            fname = f'{cell_id}-{img_idx}.png'
            fpath = Path(img_dir) / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(base64.b64decode(data['image/png']))
            rel = os.path.relpath(fpath, qmd_parent)
            parts.append(
                f'::: {{.cell-output-display}}\n![{cell_id}]({rel})\n:::')
            continue

        if 'image/svg+xml' in data:
            img_idx += 1
            svg = data['image/svg+xml']
            if isinstance(svg, list):
                svg = ''.join(svg)
            fname = f'{cell_id}-{img_idx}.svg'
            fpath = Path(img_dir) / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(svg, encoding='utf-8')
            rel = os.path.relpath(fpath, qmd_parent)
            parts.append(
                f'::: {{.cell-output-display}}\n![{cell_id}]({rel})\n:::')
            continue

        # Text output
        if otype == 'stream':
            text = ''.join(out.get('text', []))
        elif 'text/plain' in data:
            plain = data['text/plain']
            text = plain if isinstance(plain, str) else ''.join(plain)
        else:
            continue

        text = text.rstrip('\n')
        if not text:
            continue

        text_lines = text.split('\n')
        n = len(text_lines)

        if n > MAX_TEXT_LINES:
            if mode == 'pdf':
                half = MAX_TEXT_LINES // 2
                text = '\n'.join(
                    text_lines[:half] + ['...'] + text_lines[-half:])
                parts.append(
                    f'::: {{.cell-output .cell-output-stdout}}\n'
                    f'```\n{text}\n```\n:::')
            else:
                parts.append(
                    f'::: {{.cell-output .cell-output-stdout '
                    f'.d2l-output-scroll}}\n```\n{text}\n```\n:::')
        else:
            parts.append(
                f'::: {{.cell-output .cell-output-stdout}}\n'
                f'```\n{text}\n```\n:::')

    return '\n'.join(parts)


# ── Output Stripping ─────────────────────────────────────────

def strip_outputs(text):
    """Remove previously injected output blocks for idempotent re-injection."""
    return re.sub(
        rf'\n?{re.escape(OUTPUT_START)}.*?{re.escape(OUTPUT_END)}\n?',
        '', text, flags=re.DOTALL)


# ── HTML Injection ───────────────────────────────────────────

def inject_file_html(qmd_path, notebooks_dir, img_base, project_root):
    """Inject all-framework outputs into one multi-framework .qmd file.

    Returns number of cells injected, or 0 if nothing changed.
    """
    text = qmd_path.read_text(encoding='utf-8')
    text = strip_outputs(text)
    cells = parse_qmd_cells(text)
    if not cells:
        return 0

    chapter = qmd_path.parent.name
    stem = qmd_path.stem
    rel_nb = Path(chapter) / f'{stem}.ipynb'

    # Load outputs + cell maps for all available frameworks
    fw_outputs = {}
    fw_maps = {}
    for fw in FRAMEWORKS:
        nb_path = Path(notebooks_dir) / fw / rel_nb
        if nb_path.exists():
            fw_outputs[fw] = load_ipynb_outputs(nb_path)
            fw_maps[fw] = build_cell_map(cells, fw)

    if not fw_outputs:
        return 0

    img_dir = Path(img_base) / chapter
    qmd_parent = qmd_path.parent.resolve()
    injections = []  # (after_line, markdown)

    for i, cell in enumerate(cells):
        if cell.framework is not None:
            # Framework-specific cell: inject that framework's output
            fw = cell.framework
            if fw not in fw_maps or i not in fw_maps[fw]:
                continue
            idx = fw_maps[fw][i]
            if idx >= len(fw_outputs.get(fw, [])):
                continue
            outs = fw_outputs[fw][idx]
            if not outs:
                continue
            rendered = format_cell_output(
                outs, img_dir, f'{stem}-c{idx}-{fw}',
                qmd_parent, 'html')
            if rendered:
                injections.append((cell.end_line, rendered))
        else:
            # Shared cell: collect outputs from all frameworks
            fw_rendered = {}
            fw_fingerprints = {}
            for fw in FRAMEWORKS:
                if fw not in fw_maps or i not in fw_maps[fw]:
                    continue
                idx = fw_maps[fw][i]
                if idx >= len(fw_outputs.get(fw, [])):
                    continue
                outs = fw_outputs[fw][idx]
                if not outs:
                    continue
                rendered = format_cell_output(
                    outs, img_dir, f'{stem}-c{idx}-{fw}',
                    qmd_parent, 'html')
                if rendered:
                    fw_rendered[fw] = rendered
                    fw_fingerprints[fw] = _output_fingerprint(outs)

            if not fw_rendered:
                continue

            unique_fps = set(fw_fingerprints.values())
            if len(unique_fps) <= 1:
                # All frameworks produce identical output — no tabset needed
                injections.append(
                    (cell.end_line, next(iter(fw_rendered.values()))))
            else:
                # Different outputs per framework — wrap in synced tabset
                parts = ['::: {.panel-tabset group="framework" '
                         '.d2l-output-tabs}']
                for fw in FRAMEWORKS:
                    if fw in fw_rendered:
                        parts.append(
                            f'\n## {FRAMEWORK_DISPLAY[fw]}\n')
                        parts.append(fw_rendered[fw])
                parts.append('\n:::')
                injections.append((cell.end_line, '\n'.join(parts)))

    if not injections:
        return 0

    # Apply injections bottom-to-top so line numbers stay valid
    lines = text.split('\n')
    for after_line, markdown in sorted(injections, key=lambda x: -x[0]):
        block = f'\n{OUTPUT_START}\n{markdown}\n{OUTPUT_END}'
        lines.insert(after_line + 1, block)

    qmd_path.write_text('\n'.join(lines), encoding='utf-8')
    return len(injections)


def inject_html(project_root, notebooks_dir, img_base):
    """Inject outputs into all HTML .qmd files."""
    project_root = Path(project_root).resolve()
    notebooks_dir = Path(notebooks_dir).resolve()
    img_base = Path(img_base).resolve()

    qmd_files = sorted(project_root.glob('chapter_*/*.qmd'))
    total_files = 0
    total_cells = 0

    for qmd_path in qmd_files:
        n = inject_file_html(qmd_path, notebooks_dir, img_base, project_root)
        if n:
            rel = qmd_path.relative_to(project_root)
            print(f'  {rel}: {n} outputs')
            total_files += 1
            total_cells += n

    print(f'  Injected {total_cells} outputs across {total_files} files')
    return total_cells


# ── PDF Injection ────────────────────────────────────────────

def inject_file_pdf(qmd_path, framework, notebooks_dir, img_base,
                    project_root):
    """Inject single-framework outputs into a PDF .qmd file.

    Falls back through FALLBACK_ORDER if the primary framework has no
    notebook for this file.
    """
    text = qmd_path.read_text(encoding='utf-8')
    text = strip_outputs(text)
    cells = parse_qmd_cells(text)
    if not cells:
        return 0

    chapter = qmd_path.parent.name
    stem = qmd_path.stem
    rel_nb = Path(chapter) / f'{stem}.ipynb'

    # Find notebook: target framework first, then fallback order
    fw_order = [framework] + [f for f in FALLBACK_ORDER if f != framework]
    nb_path = None
    actual_fw = None
    for fw in fw_order:
        candidate = Path(notebooks_dir) / fw / rel_nb
        if candidate.exists():
            nb_path = candidate
            actual_fw = fw
            break

    if not nb_path:
        return 0

    ipynb_outputs = load_ipynb_outputs(nb_path)
    # PDF .qmd is single-framework: all cells are sequential
    cell_map = build_cell_map(cells, actual_fw)

    img_dir = Path(img_base) / chapter
    qmd_parent = qmd_path.parent.resolve()
    injections = []

    for i, cell in enumerate(cells):
        if i not in cell_map:
            continue
        idx = cell_map[i]
        if idx >= len(ipynb_outputs):
            continue
        outs = ipynb_outputs[idx]
        if not outs:
            continue
        rendered = format_cell_output(
            outs, img_dir, f'{stem}-c{idx}-{actual_fw}',
            qmd_parent, 'pdf')
        if rendered:
            injections.append((cell.end_line, rendered))

    if not injections:
        return 0

    lines = text.split('\n')
    for after_line, markdown in sorted(injections, key=lambda x: -x[0]):
        block = f'\n{OUTPUT_START}\n{markdown}\n{OUTPUT_END}'
        lines.insert(after_line + 1, block)

    qmd_path.write_text('\n'.join(lines), encoding='utf-8')
    return len(injections)


def inject_pdf(pdf_dir, framework, notebooks_dir, img_base, project_root):
    """Inject outputs into all PDF .qmd files for one framework."""
    pdf_dir = Path(pdf_dir).resolve()
    notebooks_dir = Path(notebooks_dir).resolve()
    img_base = Path(img_base).resolve()

    qmd_files = sorted(pdf_dir.glob('chapter_*/*.qmd'))
    total_files = 0
    total_cells = 0

    for qmd_path in qmd_files:
        n = inject_file_pdf(
            qmd_path, framework, notebooks_dir, img_base, project_root)
        if n:
            rel = qmd_path.relative_to(pdf_dir)
            print(f'  {rel}: {n} outputs')
            total_files += 1
            total_cells += n

    print(f'  Injected {total_cells} outputs across {total_files} files '
          f'(framework: {framework})')
    return total_cells


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Inject notebook outputs into .qmd files')
    sub = parser.add_subparsers(dest='mode', required=True)

    # html sub-command
    p_html = sub.add_parser('html',
        help='Inject all-framework outputs into chapter_*/*.qmd')
    p_html.add_argument('--project-dir', default='.',
        help='Project root (default: .)')
    p_html.add_argument('--notebooks-dir', default='_notebooks',
        help='Executed notebooks directory (default: _notebooks)')
    p_html.add_argument('--img-dir', default='img/outputs',
        help='Directory for extracted images (default: img/outputs)')

    # pdf sub-command
    p_pdf = sub.add_parser('pdf',
        help='Inject single-framework outputs into _pdf/*.qmd')
    p_pdf.add_argument('--framework', required=True,
        help='Target framework')
    p_pdf.add_argument('--pdf-dir', default='_pdf',
        help='PDF .qmd directory (default: _pdf)')
    p_pdf.add_argument('--project-dir', default='.',
        help='Project root (default: .)')
    p_pdf.add_argument('--notebooks-dir', default='_notebooks',
        help='Executed notebooks directory (default: _notebooks)')
    p_pdf.add_argument('--img-dir', default='img/outputs',
        help='Directory for extracted images (default: img/outputs)')

    args = parser.parse_args()

    if args.mode == 'html':
        print('=== Injecting notebook outputs (HTML) ===')
        inject_html(args.project_dir, args.notebooks_dir, args.img_dir)
    elif args.mode == 'pdf':
        print(f'=== Injecting notebook outputs (PDF: {args.framework}) ===')
        inject_pdf(args.pdf_dir, args.framework, args.notebooks_dir,
                   args.img_dir, args.project_dir)


if __name__ == '__main__':
    main()
