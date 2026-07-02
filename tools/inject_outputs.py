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
import textwrap
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

# Per-mode caps on text-output lines. Slides default tighter than the
# book / pdf since on-screen real estate is much smaller.
MAX_TEXT_LINES_BY_MODE = {
    'html': 40,
    'pdf': 40,
    'slides': 12,
}
MAX_TEXT_LINES = 40  # legacy default; some callers may pass mode='html'/'pdf'

# Strip ANSI CSI escape sequences (e.g. Keras 3 progress bars). Leaving them
# in the qmd corrupts Quarto's intermediate markdown — the ESC bytes confuse
# the panel-tabset Lua filter and silently truncate the document.
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_KERAS_PROGRESS_RE = re.compile(
    r'^\s*\d+/(?:\d+|Unknown)\b.*(?:/step|\sstep\b|━)')
_KERAS_EPOCH_RE = re.compile(r'^Epoch\s+\d+/\d+\s*$')
_NOISY_SLIDE_OUTPUT_RE = re.compile(
    r'^(WARNING:|WARNING:absl:|INFO:tensorflow:|I\d{4} |W\d{4} |'
    r'INFO:root:|'
    r'\[\d{2}:\d{2}:\d{2}\] |'
    r'All log messages before absl::InitializeLog\(\)|'
    r'Downloading .* from https?://|'
    r'Found \d+ files belonging to \d+ classes\.|'
    # MXNet oneDNN / storage-fallback C++ warning continuation lines. The
    # warning's first (timestamped) line is matched above / via the path
    # substring below; these are its field lines. Anchored at line start so
    # they can only match the C++ logger's fixed-format chatter, never
    # legitimate Python teaching output:
    #   operator = stack
    #   input/output storage types = [default, ...]
    #   params = {}
    #   context.dev_mask = cpu
    r'operator = \w|'
    r'(?:input|output) storage types = \[|'
    r'params = \{\}\s*$|'
    r'context\.dev_mask = |'
    r'<keras\.src\.callbacks\.history\.History at )')
_NOISY_SLIDE_OUTPUT_SUBSTRINGS = (
    'UserWarning:',
    'VisibleDeprecationWarning:',
    'AWS dependencies are not imported',
    '/home/smola/d2l/d2l-neu/',
    '/tmp/ipykernel_',
    'Storage type fallback detected',
    'StorageManager',
    # MXNet oneDNN / storage-fallback C++ warning (utils.h:521). Stderr
    # interleaving mangles the leading "[HH:MM:SS]" bracket on continuation
    # copies, so the anchored timestamp regex above misses the corrupted
    # variants; this source-path fragment is the logger's own and matches
    # every variant. The next two catch the verbose body + suppress hint:
    'mxnet/src/imperative',
    'Execution of the operator above will fallback',
    'MXNET_STORAGE_FALLBACK_LOG_VERBOSE',
)


# ── QMD Parsing ──────────────────────────────────────────────

class QmdCell:
    __slots__ = ('start_line', 'end_line', 'source', 'framework', 'cell_id')

    def __init__(self, start_line, end_line, source, framework, cell_id=None):
        self.start_line = start_line
        self.end_line = end_line
        self.source = source
        self.framework = framework  # None = shared / tab-all
        self.cell_id = cell_id      # `#| label: <id>` extracted from source


_LABEL_LINE_RE = re.compile(r'^#\|\s*label:\s*([a-z][a-z0-9-]*)\s*$')
# Hidden cell-id marker emitted by d2l_preprocess.py just before each
# non-primary (display-only) framework tab's code fence. Lets us match
# tf/jax/mxnet tabs to their notebook outputs even though they carry no
# `#| label:` (which would render as a Python comment).
_CELL_ID_MARKER_RE = re.compile(r'^<!--\s*cell-id:\s*([a-z][a-z0-9-]*)\s*-->\s*$')

# Suffix that gen_slides.py's `_dedupe_labels` adds to repeated cell
# labels in a single deck (e.g. `foo-fig2`, `foo-fig3`). Strip when
# looking up the canonical cell ID in the executed notebook.
_FIG_SUFFIX_RE = re.compile(r'-fig\d+$')


def _extract_cell_id(code_lines):
    """If first non-blank line is `#| label: <id>`, return the id.

    The `-fig{N}` dedup suffix added by gen_slides.py is stripped so
    the canonical cell ID is used for notebook-output lookup.
    """
    for ln in code_lines:
        s = ln.strip()
        if not s:
            continue
        m = _LABEL_LINE_RE.match(s)
        if not m:
            return None
        return _FIG_SUFFIX_RE.sub('', m.group(1))
    return None


def parse_qmd_cells(text):
    """Parse .qmd text and return ordered list of Python code cells.

    Tracks framework tab context via panel-tabset divs so each cell
    knows which framework it belongs to (or None for shared cells).

    Each cell's `cell_id` field is populated from the first source
    line if it matches `#| label: <id>`. Used by output injection to
    match cells to executed notebook cells by ID.
    """
    lines = text.split('\n')
    cells = []
    i = 0
    div_stack = []  # True = fw-tabset, False = other div
    current_fw = None
    # Sticky cell-id from the most recent `<!-- cell-id: ... -->` marker,
    # consumed by the next code fence. Reset on each non-blank/non-marker
    # line so a stray marker can't leak into a downstream fence.
    pending_cell_id = None

    while i < len(lines):
        line = lines[i]

        # Cell-id marker (used to tag display-only framework tabs).
        m = _CELL_ID_MARKER_RE.match(line.strip())
        if m:
            pending_cell_id = m.group(1)
            i += 1
            continue

        # Fenced div opener: ::: {.attrs}
        if re.match(r'^:{3,}\s*\{', line):
            is_fw_tabset = ('panel-tabset' in line
                            and 'group="framework"' in line)
            div_stack.append(is_fw_tabset)
            if is_fw_tabset:
                current_fw = None
            pending_cell_id = None
            i += 1
            continue

        # Fenced div closer: bare :::
        if re.match(r'^:{3,}\s*$', line) and div_stack:
            was_fw_tabset = div_stack.pop()
            if was_fw_tabset:
                current_fw = None
            pending_cell_id = None
            i += 1
            continue

        # Framework tab heading within a tabset
        if any(div_stack):
            m = re.match(r'^##\s+(PyTorch|TensorFlow|JAX|MXNet)\s*$', line)
            if m and True in div_stack:
                current_fw = DISPLAY_TO_FW[m.group(1)]
                pending_cell_id = None
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
            cid = _extract_cell_id(code_lines) or pending_cell_id
            cells.append(QmdCell(start, end, '\n'.join(code_lines), fw, cid))
            pending_cell_id = None
            i += 1
            continue

        # Anything other than blank lines clears the pending marker.
        if line.strip():
            pending_cell_id = None

        # Skip non-Python code blocks
        if re.match(r'^```', line) and not line.startswith('````'):
            i += 1
            while i < len(lines) and not re.match(r'^```\s*$', lines[i]):
                i += 1
            i += 1
            continue

        i += 1

    return cells


def index_ipynb_by_id(path):
    """Load executed .ipynb and return:
        ids: dict mapping cell.metadata.id (or cell.id) → outputs (list)
        cell_count: total number of code cells (for fallback warnings)
    """
    nb = json.loads(Path(path).read_bytes())
    ids = {}
    count = 0
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        count += 1
        cid = cell.get('id') or cell.get('metadata', {}).get('id')
        if cid:
            ids[cid] = cell.get('outputs', [])
    return ids, count


def _reconstruct_store_output(o, manifest_dir):
    """Rebuild one nbformat-style output dict from a manifest output entry.

    The reconstruction is shaped so the existing format_cell_output() /
    _output_fingerprint() code paths are byte-identical to the _notebooks/ path:
      - stream  → {'output_type':'stream','text':[<str>]}
      - inline  → {'output_type':..,'data':{'text/plain':<str>}}
      - asset   → re-read the file and re-encode into data{mime} (png→base64,
                  svg→text), exactly as it appeared in the .ipynb.
    """
    if o.get('type') == 'stream':
        return {'output_type': 'stream',
                'name': o.get('name', 'stdout'),
                'text': [o.get('text', '')]}
    mime = o.get('mime')
    if mime:
        asset = Path(manifest_dir) / o['asset']
        if mime == 'image/svg+xml':
            data = asset.read_text(encoding='utf-8')
        else:
            data = base64.b64encode(asset.read_bytes()).decode('ascii')
        return {'output_type': o.get('type', 'display_data'),
                'data': {mime: data}, 'metadata': {}}
    # inline text/plain
    return {'output_type': o.get('type', 'execute_result'),
            'data': {'text/plain': o.get('text', '')}, 'metadata': {}}


def index_store_by_id(store_dir, fw, chapter, stem):
    """Load a committed outputs manifest → (ids, count), same shape as
    index_ipynb_by_id. Returns None if no manifest exists for this notebook."""
    manifest_path = Path(store_dir) / fw / chapter / f'{stem}.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest_dir = manifest_path.parent
    ids = {}
    for cid, entry in manifest.get('cells', {}).items():
        ids[cid] = [_reconstruct_store_output(o, manifest_dir)
                    for o in entry.get('outputs', [])]
    return ids, len(ids)


def resolve_id_outputs(notebooks_dir, store_dir, fw, chapter, stem):
    """Prefer the committed outputs store; fall back to the executed notebook.

    This is the seam that decouples rendering from execution: with the store
    present, no _notebooks/ tree (and no framework venv / GPU) is needed.
    Returns (ids, count) or None if neither source has this notebook.
    """
    if store_dir:
        res = index_store_by_id(store_dir, fw, chapter, stem)
        if res is not None:
            return res
    nb_path = Path(notebooks_dir) / fw / chapter / f'{stem}.ipynb'
    if nb_path.exists():
        return index_ipynb_by_id(nb_path)
    return None


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
        if 'image/svg+xml' in data:
            svg = data['image/svg+xml']
            if isinstance(svg, list):
                svg = ''.join(svg)
            h.update(svg.encode())
        elif 'image/png' in data:
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
        mode: 'html' | 'pdf' | 'slides'

    Returns:
        Quarto markdown string, or '' if no renderable output.
    """
    if not raw_outputs:
        return ''

    max_lines = MAX_TEXT_LINES_BY_MODE.get(mode, MAX_TEXT_LINES)
    parts = []
    img_idx = 0
    slide_text_lines = []
    image_output_count = sum(
        1 for out in raw_outputs
        if 'image/png' in out.get('data', {})
        or 'image/svg+xml' in out.get('data', {}))
    image_output_seen = 0

    def clean_text_output(text):
        text = _ANSI_RE.sub('', text)
        # Drop carriage returns and the text they overwrite (terminal-style
        # progress-bar redraws). Each \r resets to the start of the line, so
        # only the segment after the last \r in any given line is what the
        # user would have seen.
        text = re.sub(r'[^\n]*\r', '', text)
        if mode == 'slides':
            kept = []
            for ln in text.split('\n'):
                stripped = ln.strip()
                if _NOISY_SLIDE_OUTPUT_RE.match(stripped):
                    continue
                if any(s in stripped for s in _NOISY_SLIDE_OUTPUT_SUBSTRINGS):
                    continue
                if len(ln) > 160:
                    ln = textwrap.shorten(ln, width=160, placeholder=' ...')
                kept.append(ln)
            text = '\n'.join(kept)
            text = collapse_keras_progress(text)
        return text.rstrip('\n')

    def collapse_keras_progress(text):
        lines = text.split('\n')
        if not any(_KERAS_PROGRESS_RE.match(ln) for ln in lines):
            return text

        collapsed = []
        last_progress = None
        for ln in lines:
            if _KERAS_PROGRESS_RE.match(ln):
                last_progress = ln.strip()
                continue
            if _KERAS_EPOCH_RE.match(ln):
                continue
            if ln.strip() == '...':
                continue
            collapsed.append(ln)

        if last_progress:
            metrics = re.sub(r'^\d+/(?:\d+|Unknown)\s+.*?/step\s+-\s*',
                             '', last_progress)
            metrics = re.sub(r'^\d+/(?:\d+|Unknown)\s+.*?\s+-\s*',
                             '', metrics)
            collapsed.append(f'Final epoch metrics: {metrics}')

        return '\n'.join(collapsed)

    def append_text_block(text):
        if not text:
            return
        text_lines = text.split('\n')
        n = len(text_lines)

        if n > max_lines:
            if mode in ('pdf', 'slides'):
                half = max_lines // 2
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

    for out in raw_outputs:
        otype = out.get('output_type', '')
        if otype == 'error':
            continue

        data = out.get('data', {})

        # Image output
        # Empty alt text — non-empty alt is rendered as a visible
        # caption by Pandoc's implicit_figures extension. We don't
        # want every notebook plot to display its cell-id underneath.
        if 'image/png' in data:
            image_output_seen += 1
            if mode == 'slides' and image_output_count > 2 and image_output_seen < image_output_count:
                continue
            img_idx += 1
            fname = f'{cell_id}-{img_idx}.png'
            fpath = Path(img_dir) / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_bytes(base64.b64decode(data['image/png']))
            rel = os.path.relpath(fpath, qmd_parent)
            parts.append(
                f'::: {{.cell-output-display}}\n![]({rel})\n:::')
            continue

        if 'image/svg+xml' in data:
            image_output_seen += 1
            if mode == 'slides' and image_output_count > 2 and image_output_seen < image_output_count:
                continue
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
                f'::: {{.cell-output-display}}\n![]({rel})\n:::')
            continue

        # Text output
        if otype == 'stream':
            text = ''.join(out.get('text', []))
            # Slides: drop stderr streams that are Python warnings wholesale
            # (path line, message line, and source-context line together) —
            # the per-line filters in clean_text_output only catch the first.
            if (mode == 'slides' and out.get('name') == 'stderr'
                    and re.search(r'\b[A-Za-z]*Warning:', text)):
                continue
        elif 'text/plain' in data:
            plain = data['text/plain']
            text = plain if isinstance(plain, str) else ''.join(plain)
        else:
            continue

        text = clean_text_output(text)
        if not text:
            continue

        if mode == 'slides':
            slide_text_lines.extend(text.split('\n'))
        else:
            append_text_block(text)

    if slide_text_lines:
        text = '\n'.join(slide_text_lines)
        append_text_block(text)

    return '\n'.join(parts)


# ── Output Stripping ─────────────────────────────────────────

def strip_outputs(text):
    """Remove previously injected output blocks for idempotent re-injection."""
    return re.sub(
        rf'\n?{re.escape(OUTPUT_START)}.*?{re.escape(OUTPUT_END)}\n?',
        '', text, flags=re.DOTALL)


# ── HTML Injection ───────────────────────────────────────────

def inject_file_html(qmd_path, notebooks_dir, img_base, project_root,
                      warnings, store_dir=None):
    """Inject all-framework outputs into one multi-framework .qmd file.

    Cells are matched to executed notebook cells by their `#| label: <id>`
    (in the .qmd) and `cell.metadata.id` (in the .ipynb). Cells without
    an ID are skipped; their absence is logged.

    Returns number of cells injected, or 0 if nothing changed.
    """
    text = qmd_path.read_text(encoding='utf-8')
    text = strip_outputs(text)
    cells = parse_qmd_cells(text)
    if not cells:
        return 0

    chapter = qmd_path.parent.name
    stem = qmd_path.stem

    # Per-framework: id → outputs (committed store preferred, _notebooks fallback)
    fw_id_outputs = {}
    for fw in FRAMEWORKS:
        res = resolve_id_outputs(notebooks_dir, store_dir, fw, chapter, stem)
        if res and res[0]:
            fw_id_outputs[fw] = res[0]

    if not fw_id_outputs:
        return 0

    img_dir = Path(img_base) / chapter
    qmd_parent = qmd_path.parent.resolve()
    injections = []  # (after_line, markdown)

    for i, cell in enumerate(cells):
        if not cell.cell_id:
            warnings.append(f'{qmd_path}:{cell.start_line+1}: missing #| label')
            continue

        if cell.framework is not None:
            fw = cell.framework
            if fw not in fw_id_outputs:
                continue
            outs = fw_id_outputs[fw].get(cell.cell_id)
            if not outs:
                continue
            rendered = format_cell_output(
                outs, img_dir, f'{stem}-{cell.cell_id}-{fw}',
                qmd_parent, 'html')
            if rendered:
                injections.append((cell.end_line, rendered))
        else:
            # Shared cell: collect outputs from each framework that has
            # this cell ID.
            fw_rendered = {}
            fw_fingerprints = {}
            for fw in FRAMEWORKS:
                if fw not in fw_id_outputs:
                    continue
                outs = fw_id_outputs[fw].get(cell.cell_id)
                if not outs:
                    continue
                rendered = format_cell_output(
                    outs, img_dir, f'{stem}-{cell.cell_id}-{fw}',
                    qmd_parent, 'html')
                if rendered:
                    fw_rendered[fw] = rendered
                    fw_fingerprints[fw] = _output_fingerprint(outs)

            if not fw_rendered:
                continue

            unique_fps = set(fw_fingerprints.values())
            if len(unique_fps) <= 1:
                injections.append(
                    (cell.end_line, next(iter(fw_rendered.values()))))
            else:
                parts = ['::: {.panel-tabset group="framework" '
                         '.d2l-output-tabs}']
                for fw in FRAMEWORKS:
                    if fw in fw_rendered:
                        parts.append(f'\n## {FRAMEWORK_DISPLAY[fw]}\n')
                        parts.append(fw_rendered[fw])
                parts.append('\n:::')
                injections.append((cell.end_line, '\n'.join(parts)))

    if not injections:
        return 0

    lines = text.split('\n')
    for after_line, markdown in sorted(injections, key=lambda x: -x[0]):
        block = f'\n{OUTPUT_START}\n{markdown}\n{OUTPUT_END}'
        lines.insert(after_line + 1, block)

    qmd_path.write_text('\n'.join(lines), encoding='utf-8')
    return len(injections)


def inject_html(project_root, notebooks_dir, img_base, store_dir=None):
    """Inject outputs into all HTML .qmd files."""
    project_root = Path(project_root).resolve()
    notebooks_dir = Path(notebooks_dir).resolve()
    img_base = Path(img_base).resolve()
    if store_dir:
        store_dir = Path(store_dir).resolve()

    qmd_files = sorted(project_root.glob('chapter_*/*.qmd'))
    total_files = 0
    total_cells = 0
    warnings = []

    for qmd_path in qmd_files:
        n = inject_file_html(qmd_path, notebooks_dir, img_base, project_root,
                              warnings, store_dir=store_dir)
        if n:
            rel = qmd_path.relative_to(project_root)
            print(f'  {rel}: {n} outputs')
            total_files += 1
            total_cells += n

    print(f'  Injected {total_cells} outputs across {total_files} files')
    if warnings:
        print(f'  {len(warnings)} cells without #| label '
              f'(skipped, no output injection):')
        for w in warnings[:10]:
            print(f'    {w}')
        if len(warnings) > 10:
            print(f'    ... and {len(warnings) - 10} more')
    return total_cells


# ── PDF Injection ────────────────────────────────────────────

def _inject_single_fw(qmd_path, framework, notebooks_dir, img_base,
                       mode, fallback=True, warnings=None, store_dir=None):
    """Shared helper for pdf and slides modes.

    Single-framework qmd → injects outputs from the matching framework's
    notebook by `#| label: <id>`. Cells without IDs are skipped. Outputs come
    from the committed store when present, else the executed notebook.

    `fallback`: pdf mode falls back through FALLBACK_ORDER if the target
    framework has no notebook for this file. slides mode does not.
    """
    if warnings is None:
        warnings = []
    text = qmd_path.read_text(encoding='utf-8')
    text = strip_outputs(text)
    cells = parse_qmd_cells(text)
    if not cells:
        return 0

    chapter = qmd_path.parent.name
    stem = qmd_path.stem

    fw_order = [framework]
    if fallback:
        fw_order += [f for f in FALLBACK_ORDER if f != framework]

    id_outputs = None
    actual_fw = None
    for fw in fw_order:
        res = resolve_id_outputs(notebooks_dir, store_dir, fw, chapter, stem)
        if res and res[0]:
            id_outputs = res[0]
            actual_fw = fw
            break

    if not id_outputs:
        return 0

    img_dir = Path(img_base) / chapter
    qmd_parent = qmd_path.parent.resolve()
    injections = []

    for cell in cells:
        if not cell.cell_id:
            warnings.append(f'{qmd_path}:{cell.start_line+1}: missing #| label')
            continue
        outs = id_outputs.get(cell.cell_id)
        if not outs:
            continue
        rendered = format_cell_output(
            outs, img_dir, f'{stem}-{cell.cell_id}-{actual_fw}',
            qmd_parent, mode)
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


def inject_file_pdf(qmd_path, framework, notebooks_dir, img_base,
                    project_root, warnings, store_dir=None):
    return _inject_single_fw(qmd_path, framework, notebooks_dir, img_base,
                              mode='pdf', fallback=True, warnings=warnings,
                              store_dir=store_dir)


def inject_pdf(pdf_dir, framework, notebooks_dir, img_base, project_root,
               store_dir=None):
    pdf_dir = Path(pdf_dir).resolve()
    notebooks_dir = Path(notebooks_dir).resolve()
    img_base = Path(img_base).resolve()
    if store_dir:
        store_dir = Path(store_dir).resolve()

    qmd_files = sorted(pdf_dir.glob('chapter_*/*.qmd'))
    total_files = 0
    total_cells = 0
    warnings = []

    for qmd_path in qmd_files:
        n = inject_file_pdf(
            qmd_path, framework, notebooks_dir, img_base, project_root,
            warnings, store_dir=store_dir)
        if n:
            rel = qmd_path.relative_to(pdf_dir)
            print(f'  {rel}: {n} outputs')
            total_files += 1
            total_cells += n

    print(f'  Injected {total_cells} outputs across {total_files} files '
          f'(framework: {framework})')
    if warnings:
        print(f'  {len(warnings)} cells without #| label')
    return total_cells


# ── Slides Injection ─────────────────────────────────────────

def inject_slides(slides_root, framework, notebooks_dir, img_base,
                  store_dir=None):
    """Inject outputs into all slide .qmd files for one framework.

    Slide qmd lives at `_slides/<fw>/<chapter>/<file>.qmd`. Outputs come from
    the committed store (`outputs/<fw>/...`) when present, else the executed
    `_notebooks/<fw>/<chapter>/<file>.ipynb`. No fallback across frameworks — if
    this framework has no outputs for a file, the deck has no outputs.
    """
    fw_dir = Path(slides_root).resolve() / framework
    notebooks_dir = Path(notebooks_dir).resolve()
    img_base = Path(img_base).resolve()
    if store_dir:
        store_dir = Path(store_dir).resolve()

    if not fw_dir.exists():
        print(f'  No slides directory for {framework}')
        return 0

    qmd_files = sorted(fw_dir.rglob('*.qmd'))
    total_files = 0
    total_cells = 0
    warnings = []

    for qmd_path in qmd_files:
        n = _inject_single_fw(qmd_path, framework, notebooks_dir, img_base,
                               mode='slides', fallback=False,
                               warnings=warnings, store_dir=store_dir)
        if n:
            rel = qmd_path.relative_to(fw_dir)
            print(f'  {rel}: {n} outputs')
            total_files += 1
            total_cells += n

    print(f'  Injected {total_cells} outputs across {total_files} files '
          f'(framework: {framework})')
    if warnings:
        print(f'  {len(warnings)} cells without #| label')
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
        help='Executed notebooks fallback directory (default: _notebooks)')
    p_html.add_argument('--store-dir', default='outputs',
        help='Committed outputs store, preferred source (default: outputs; '
             '"" to force _notebooks)')
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
        help='Executed notebooks fallback directory (default: _notebooks)')
    p_pdf.add_argument('--store-dir', default='outputs',
        help='Committed outputs store, preferred source (default: outputs)')
    p_pdf.add_argument('--img-dir', default='img/outputs',
        help='Directory for extracted images (default: img/outputs)')

    # slides sub-command
    p_sl = sub.add_parser('slides',
        help='Inject single-framework outputs into _slides/<fw>/*.qmd')
    p_sl.add_argument('--framework', required=True,
        help='Target framework')
    p_sl.add_argument('--slides-dir', default='_slides',
        help='Slides root directory (default: _slides)')
    p_sl.add_argument('--notebooks-dir', default='_notebooks',
        help='Executed notebooks fallback directory (default: _notebooks)')
    p_sl.add_argument('--store-dir', default='outputs',
        help='Committed outputs store, preferred source (default: outputs)')
    p_sl.add_argument('--img-dir', default='_slides/img/outputs',
        help='Directory for extracted images (default: _slides/img/outputs)')

    args = parser.parse_args()
    store_dir = args.store_dir or None

    if args.mode == 'html':
        print('=== Injecting notebook outputs (HTML) ===')
        inject_html(args.project_dir, args.notebooks_dir, args.img_dir,
                    store_dir=store_dir)
    elif args.mode == 'pdf':
        print(f'=== Injecting notebook outputs (PDF: {args.framework}) ===')
        inject_pdf(args.pdf_dir, args.framework, args.notebooks_dir,
                   args.img_dir, args.project_dir, store_dir=store_dir)
    elif args.mode == 'slides':
        print(f'=== Injecting notebook outputs (slides: {args.framework}) ===')
        inject_slides(args.slides_dir, args.framework,
                       args.notebooks_dir, args.img_dir, store_dir=store_dir)


if __name__ == '__main__':
    main()
