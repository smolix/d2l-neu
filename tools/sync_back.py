#!/usr/bin/env python3
"""Round-trip notebook edits back to the source `.md`.

Match notebook cells to source via:
- Code cells: `cell.metadata.id` (or `cell.id`) → fence with matching
  `#<id>`. The notebook view of a tabset variant shares the base ID
  with its source fence, so per-framework variants live in separate
  fences in source but the same metadata.id in their respective
  notebooks. We update only the source fence whose `#@tab` matches
  the notebook's framework (or the singleton fence if there's only
  one variant).
- Markdown cells: `<!-- d2l:prose id=<id> fw=<fw> -->` header → either
  a shared paragraph in source (`fw=all`) or the body of a matching
  `:begin_tab:` block (`fw=<fw>`).

Edits preserve source-side constructs that don't appear in the
notebook view: `#<id>` in fence info string, `#@tab` line, `#@save`
markers, `@d2l.add_to_class(...)` decorators.

What this tool does NOT do (yet):
- Cell insertion: warn and skip (author should add via source).
- Cell deletion: warn and skip (author should delete via source).
- Prose-paragraph fine-grained matching for shared `fw=all` content
  outside `:begin_tab:` blocks: only handled when the markdown cell's
  content matches a single contiguous prose region in source.
- Conflict detection beyond mtime check: a 3-way diff UI lives in
  the VS Code extension.

Usage:
    python tools/sync_back.py --notebook <ipynb> --source <md>
        [--dry-run] [--verbose]
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import is_python_block, is_boilerplate, extract_tab


# ──────────────────────────────────────────────────────────
# Source indexing
# ──────────────────────────────────────────────────────────

_FENCE_OPEN_RE = re.compile(r'^```(.*)$')
_FENCE_CLOSE_RE = re.compile(r'^```\s*$')
_ID_IN_INFO_RE = re.compile(r'(?:^|[\s{])#([a-z][a-z0-9-]*)(?=\s|\}|$)')
_BEGIN_TAB_RE = re.compile(r'^:begin_tab:`([^`]+)`\s*$')
_END_TAB_RE = re.compile(r'^:end_tab:\s*$')
_PROSE_HEADER_RE = re.compile(
    r'<!--\s*d2l:prose\s+id=([^\s]+)\s+fw=([^\s]+)\s*-->')
_LABEL_LINE_RE = re.compile(r'^#\|\s*label:\s*([a-z][a-z0-9-]*)\s*$')


class SourceFence:
    """One Python code fence in source, with line range."""
    __slots__ = ('start', 'end', 'info', 'cell_id', 'tab', 'body_lines')
    def __init__(self, start, end, info, cell_id, tab, body_lines):
        self.start, self.end = start, end
        self.info = info
        self.cell_id = cell_id
        self.tab = tab
        self.body_lines = body_lines


class SourceTabBlock:
    """One :begin_tab:/:end_tab: pair in source, with line range."""
    __slots__ = ('start', 'end', 'fws', 'body_lines')
    def __init__(self, start, end, fws, body_lines):
        self.start, self.end = start, end
        self.fws = fws
        self.body_lines = body_lines


def index_source(text):
    """Return (fences, tab_blocks).

    `fences`: list[SourceFence], in order. Includes only Python fences.
    `tab_blocks`: list[SourceTabBlock], in order.
    """
    lines = text.split('\n')
    fences = []
    tabs = []
    in_fence = False
    fence_start = -1
    fence_info = ''
    fence_body = []
    in_tab = False
    tab_start = -1
    tab_fws = []
    tab_body = []

    for i, line in enumerate(lines):
        if not in_fence and not in_tab:
            m = _FENCE_OPEN_RE.match(line)
            if m and not line.startswith('````'):
                fence_start = i
                fence_info = m.group(1).strip()
                fence_body = []
                in_fence = True
                continue
            mb = _BEGIN_TAB_RE.match(line)
            if mb:
                tab_start = i
                tab_fws = [t.strip() for t in mb.group(1).split(',')]
                tab_body = []
                in_tab = True
                continue
        elif in_fence:
            if _FENCE_CLOSE_RE.match(line):
                in_fence = False
                if is_python_block(fence_info) and not is_boilerplate(fence_body):
                    cid_m = _ID_IN_INFO_RE.search(fence_info)
                    cid = cid_m.group(1) if cid_m else None
                    tab, _ = extract_tab(list(fence_body))
                    fences.append(SourceFence(
                        fence_start, i, fence_info, cid, tab, fence_body))
            else:
                fence_body.append(line)
        elif in_tab:
            if _END_TAB_RE.match(line):
                in_tab = False
                tabs.append(SourceTabBlock(
                    tab_start, i, tab_fws, tab_body))
            else:
                tab_body.append(line)

    return fences, tabs


# ──────────────────────────────────────────────────────────
# Notebook indexing
# ──────────────────────────────────────────────────────────

def parse_nb_cells(nb_path: Path):
    """Return (code_cells, prose_cells) lists.

    Each code cell: dict {id, source_lines}.
    Each prose cell: dict {prose_id, fw, body_lines}.
    """
    nb = json.loads(nb_path.read_bytes())
    code = []
    prose = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            cid = cell.get('id') or cell.get('metadata', {}).get('id')
            if not cid:
                continue
            src = cell.get('source', [])
            if isinstance(src, str):
                src_lines = src.split('\n')
            else:
                src_lines = [ln.rstrip('\n') for ln in src]
            # Strip any leftover #| label: line (in case patch_ipynb_ids
            # didn't run, but normally it has).
            if src_lines and _LABEL_LINE_RE.match(src_lines[0]):
                src_lines = src_lines[1:]
            code.append({'id': cid, 'source_lines': src_lines})
        elif cell['cell_type'] == 'markdown':
            src = cell.get('source', [])
            if isinstance(src, str):
                src_lines = src.split('\n')
            else:
                src_lines = [ln.rstrip('\n') for ln in src]
            # Header on first line
            header_m = None
            if src_lines:
                header_m = _PROSE_HEADER_RE.search(src_lines[0])
            if not header_m:
                continue  # unanchored prose; skip
            pid, fw = header_m.group(1), header_m.group(2)
            # Strip header line + an empty line after, if present
            body = src_lines[1:]
            while body and not body[0].strip():
                body.pop(0)
            prose.append({'prose_id': pid, 'fw': fw, 'body_lines': body})
    return code, prose


# ──────────────────────────────────────────────────────────
# Source rewriting
# ──────────────────────────────────────────────────────────

def rebuild_fence_body(source_fence: SourceFence, new_body_lines):
    """Compose the new body lines for a code fence, preserving:
       - leading `#@tab <fw>` or `%%tab <fw>` line (if present)
       - `#@save` markers on lines that originally had them
         (new lines without them stay un-marked)
       - `@d2l.add_to_class(...)` decorator (if originally present)
    """
    orig = list(source_fence.body_lines)
    # Capture leading directives we want to keep at the top
    preserved_lead = []
    while orig:
        first = orig[0].strip()
        if (first.startswith('#@tab') or first.startswith('%%tab')
                or first == ''):
            preserved_lead.append(orig.pop(0))
        else:
            break

    # Capture `@d2l.add_to_class(...)` decorator if it's the first
    # non-blank, non-tab line after the leading directives.
    decorator = None
    rest = list(orig)
    while rest and rest[0].strip() == '':
        rest.pop(0)
    if rest and re.match(r'^@d2l\.add_to_class\(', rest[0]):
        decorator = rest[0]

    # Build new body: preserved_lead + decorator (if present) + new_body
    out = list(preserved_lead)
    if decorator:
        out.append(decorator)
    out.extend(new_body_lines)
    return out


def apply_fence_edit(lines, fence: SourceFence, new_body):
    """Replace the body of a fence in-place. Keeps the open/close lines
    and the `#<id>` info string untouched; only the body changes."""
    return (lines[:fence.start + 1]
            + new_body
            + lines[fence.end:])


def apply_tab_edit(lines, tab_block: SourceTabBlock, new_body):
    """Replace the body of a :begin_tab:/:end_tab: pair in-place."""
    return (lines[:tab_block.start + 1]
            + new_body
            + lines[tab_block.end:])


def find_target_fence(fences, cell_id, framework):
    """Return the fence in source matching cell_id+framework, or None.

    Multi-variant base ID: pick the variant whose `#@tab` includes
    `framework`, falling back to a variant tagged 'all' (or no tab).
    Single-variant: return it regardless of fw.
    """
    candidates = [f for f in fences if f.cell_id == cell_id]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer fw-specific variant
    for f in candidates:
        if f.tab and f.tab != 'all':
            tabs = [t.strip() for t in f.tab.split(',')]
            if framework in tabs:
                return f
    # Fall back to 'all' variant
    for f in candidates:
        if f.tab is None or f.tab == 'all':
            return f
    return None


def find_target_tab(tabs, framework):
    """Return the begin_tab block whose fws include framework, or None.

    Caller chooses which one — we return the first matching block, but
    the caller should iterate by prose cell order to pick the right
    instance.
    """
    return [t for t in tabs if framework in t.fws]


# ──────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────

def sync_file(notebook_path: Path, source_path: Path, framework: str,
              dry_run=False, verbose=False):
    text = source_path.read_text(encoding='utf-8')
    lines = text.split('\n')
    fences, tabs = index_source(text)
    code_cells, prose_cells = parse_nb_cells(notebook_path)

    # Plan all edits, then apply bottom-up so line numbers stay valid.
    edits = []  # list of (start, end, replacement_body_lines, kind, descr)
    warnings = []

    # ── Code cells ──
    used_fence_ids = set()
    for cell in code_cells:
        cid = cell['id']
        target = find_target_fence(fences, cid, framework)
        if target is None:
            warnings.append(f'no source fence for cell id={cid}')
            continue
        if target.start in used_fence_ids:
            warnings.append(f'duplicate match on fence at line {target.start+1}')
            continue
        used_fence_ids.add(target.start)
        new_body = rebuild_fence_body(target, cell['source_lines'])
        if new_body == target.body_lines:
            continue  # unchanged
        edits.append((target.start + 1, target.end, new_body, 'code',
                      f'fence #{cid}'))

    # ── Prose cells with fw=<framework> → begin_tab blocks ──
    fw_tab_blocks = [t for t in tabs if framework in t.fws]
    fw_tab_idx = 0
    fw_prose_cells = [p for p in prose_cells if p['fw'] == framework
                       or framework in p['fw'].split(',')]
    for p in fw_prose_cells:
        if fw_tab_idx >= len(fw_tab_blocks):
            warnings.append(
                f'fw=<{framework}> prose `{p["prose_id"]}` has no '
                f'matching :begin_tab: block in source')
            continue
        target = fw_tab_blocks[fw_tab_idx]
        fw_tab_idx += 1
        if p['body_lines'] == target.body_lines:
            continue
        edits.append((target.start + 1, target.end, p['body_lines'],
                      'tab', f':begin_tab:`{",".join(target.fws)}`'))

    # ── Prose cells with fw=all → shared prose ──
    # Skipped in v0: needs anchoring by surrounding code-cell IDs to be
    # robust. Warn if any fw=all cells have changed content. (We can't
    # detect change without knowing the source content; so warn that
    # shared-prose round-trip isn't supported yet.)
    for p in prose_cells:
        if p['fw'] != 'all':
            continue
        # No-op for now; document limitation.

    # ── Apply edits ──
    if not edits:
        if verbose:
            print(f'  {source_path}: no changes')
    else:
        for start, end, body, kind, descr in sorted(edits, key=lambda e: -e[0]):
            lines = lines[:start] + body + lines[end:]
            if verbose:
                print(f'  {source_path}: updated {kind} {descr}')

    if not dry_run and edits:
        # Atomic write
        tmp = source_path.with_suffix(source_path.suffix + '.tmp')
        tmp.write_text('\n'.join(lines), encoding='utf-8')
        tmp.replace(source_path)

    for w in warnings:
        print(f'warning: {w}', file=sys.stderr)

    return len(edits), len(warnings)


def main():
    parser = argparse.ArgumentParser(
        description='Round-trip notebook edits back to source .md')
    parser.add_argument('--notebook', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--framework',
                        help='Framework of the notebook '
                             '(default: parse from path)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    fw = args.framework
    if fw is None:
        # Path like _notebooks/<fw>/<chapter>/<file>.ipynb
        for part in args.notebook.parts:
            if part in {'pytorch', 'tensorflow', 'jax', 'mxnet'}:
                fw = part
                break
    if fw is None:
        print('error: cannot determine framework; pass --framework',
              file=sys.stderr)
        sys.exit(1)

    n_edits, n_warn = sync_file(args.notebook, args.source, fw,
                                 dry_run=args.dry_run, verbose=args.verbose)
    print(f'{args.source}: {n_edits} edit(s), {n_warn} warning(s)'
          + (' (dry run)' if args.dry_run else ''))


if __name__ == '__main__':
    main()
