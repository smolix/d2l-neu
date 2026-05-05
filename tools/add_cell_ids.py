#!/usr/bin/env python3
"""Assign stable #<id> attributes to Python code fences in chapter_*/*.md.

Idempotent. Skips fences that already have an ID. Section-slug-derived IDs
with a numeric suffix when the same heading slug appears more than once in
a file.

Form on a code fence (post-migration):

    ```{.python .input #linear-regression-vector-addition-1}
    c = a + b
    ```

The `n=NN` legacy d2lbook counter is preserved if present.

Per-framework variants share a base ID — multiple cells with different
`#@tab` directives that fall in the same section share the same base ID
plus per-variant numbering. Authors may renumber later by hand; this
script only assigns IDs the first time, never rewrites existing ones.

Usage:
    python tools/add_cell_ids.py [--check] [--dry-run] [--verbose] [files...]

By default, processes every chapter_*/*.md in CHAPTER_NUMBERING. With
explicit file arguments, processes only those.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import (CHAPTER_NUMBERING, FRAMEWORKS, is_python_block,
                             is_boilerplate, extract_tab)


# ──────────────────────────────────────────────────────────
# Slugification
# ──────────────────────────────────────────────────────────

def slugify(text):
    """Lowercase, non-alphanumerics → '-', trim. Strip {#anchor} suffix."""
    text = re.sub(r'\{#[^}]+\}', '', text)
    text = re.sub(r':label:`[^`]+`', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')


def file_slug(path: Path) -> str:
    """`chapter_linear-regression/linear-regression.md` → `linear-regression`.

    For `index.md`, use the chapter directory's slug (minus `chapter_` prefix).
    """
    if path.stem == 'index':
        return path.parent.name.removeprefix('chapter_')
    return path.stem


# ──────────────────────────────────────────────────────────
# Fence info string parsing & rewriting
# ──────────────────────────────────────────────────────────

# Match an existing #<id> attribute in a fence info string. We match
# `#word-with-dashes` only when followed by space, `}`, or end — to avoid
# matching `#hashtag` inside Python comments (which would never appear in
# an info string anyway, but be safe).
ID_IN_INFO_RE = re.compile(r'(?:^|\s)#([a-z][a-z0-9-]*)(?=\s|\}|$)')


def info_has_id(info: str) -> bool:
    return ID_IN_INFO_RE.search(info) is not None


def inject_id(info: str, cid: str) -> str:
    """Insert `#<cid>` into a Python fence info string.

    `{.python .input}` → `{.python .input #cid}`
    `{.python .input  n=12}` → `{.python .input #cid  n=12}`
    `python` → `{.python #cid}`
    """
    if info.startswith('{') and info.endswith('}'):
        # Insert right after `.input` (or after `.python` if no `.input`).
        body = info[1:-1].strip()
        if '.input' in body:
            new_body = re.sub(r'(\.input)', rf'\1 #{cid}', body, count=1)
        elif '.python' in body:
            new_body = re.sub(r'(\.python)', rf'\1 #{cid}', body, count=1)
        else:
            new_body = body + f' #{cid}'
        return '{' + new_body + '}'
    elif info == 'python' or info.startswith('python'):
        return '{.python #' + cid + '}' + info[len('python'):]
    else:
        # Already non-standard; just append
        return info + f' #{cid}'


# ──────────────────────────────────────────────────────────
# Fence walk
# ──────────────────────────────────────────────────────────

HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
FENCE_OPEN_RE = re.compile(r'^```(.*)$')
FENCE_CLOSE_RE = re.compile(r'^```\s*$')


def collect_fences(lines, chap_slug):
    """Walk lines, return list of dicts: {start, end, info, section, tab,
    has_nontrivial_text_before}.

    `start` is the line index of the opening ``` and `end` is the closing.
    `tab` is the result of extract_tab() on the cell body — None, 'all',
    'pytorch', 'pytorch, mxnet', etc.
    `has_nontrivial_text_before` is True iff there's any non-blank,
    non-fence content between this fence and the previous Python fence
    (used to break tabset accumulation).

    Section slug is the slug of the most recent heading (any level), with
    a numeric suffix appended when the same slug appears more than once
    in the file. When a heading slugifies to chap_slug (typical for the
    file's H1), section_slug is reset to '' so cells get IDs like
    `<chap>-<seq>` instead of `<chap>-<chap>-<seq>`.

    Boilerplate cells are skipped.
    Only Python code fences are collected.
    """
    fences = []
    section_slug = ''
    section_seen = {}
    in_fence = False
    fence_start = -1
    fence_info = ''
    fence_body = []
    saw_nontrivial = False  # cleared when a Python fence is added

    for i, line in enumerate(lines):
        if not in_fence:
            m = HEADING_RE.match(line)
            if m:
                slug = slugify(m.group(2))
                if slug:
                    if slug == chap_slug:
                        section_slug = ''
                    else:
                        n = section_seen.get(slug, 0) + 1
                        section_seen[slug] = n
                        section_slug = slug if n == 1 else f'{slug}-{n}'
                # A heading line is non-trivial content for tabset
                # accumulation purposes — breaks the run.
                saw_nontrivial = True
                continue
            m = FENCE_OPEN_RE.match(line)
            if m and not line.startswith('````'):
                fence_start = i
                fence_info = m.group(1).strip()
                fence_body = []
                in_fence = True
                continue
            if line.strip():
                saw_nontrivial = True
        else:
            if FENCE_CLOSE_RE.match(line):
                if is_python_block(fence_info) and not is_boilerplate(fence_body):
                    tab, _ = extract_tab(list(fence_body))
                    fences.append({
                        'start': fence_start,
                        'end': i,
                        'info': fence_info,
                        'section': section_slug,
                        'tab': tab,
                        'broken': saw_nontrivial,
                    })
                    saw_nontrivial = False
                else:
                    # Non-Python or boilerplate fence interrupts a tabset run
                    saw_nontrivial = True
                in_fence = False
            else:
                fence_body.append(line)

    return fences


def extract_existing_id(info):
    m = ID_IN_INFO_RE.search(info)
    return m.group(1) if m else None


def is_fw_specific(tab):
    """True if `tab` names one or more specific frameworks (not None, not 'all')."""
    if tab is None or tab == 'all':
        return False
    return any(t.strip() in FRAMEWORKS for t in tab.split(','))


def group_into_units(fences):
    """Group consecutive framework-specific fences into tabset units.

    A unit is a list of fence dicts that share a base ID:
    - A standalone fence (no tab, tab='all', or framework-specific but
      separated from neighbors by non-blank content) is its own unit.
    - Consecutive framework-specific fences with only blank lines
      between them form one unit, BUT a unit may contain at most one
      cell per framework — if a new cell's framework is already in the
      current unit, a new unit starts (matches group_code_tabs behavior
      in d2l_preprocess.py).

    The "broken" flag on a fence indicates non-blank content appeared
    between this fence and the previous Python fence; it forces a new
    unit even for framework-specific fences.
    """
    def fws_in_unit(unit):
        s = set()
        for f in unit:
            if is_fw_specific(f['tab']):
                for t in f['tab'].split(','):
                    t = t.strip()
                    if t in FRAMEWORKS:
                        s.add(t)
        return s

    units = []
    cur = []
    for f in fences:
        # A unit accumulates only consecutive framework-specific cells.
        # A non-fw-specific cell (`%%tab all`, untagged, etc.) is always
        # its own unit.
        if (is_fw_specific(f['tab']) and cur and not f['broken']
                and is_fw_specific(cur[-1]['tab'])):
            new_fws = {t.strip() for t in f['tab'].split(',')
                       if t.strip() in FRAMEWORKS}
            if new_fws & fws_in_unit(cur):
                # Collision: this framework is already in the unit. Start fresh.
                units.append(cur)
                cur = [f]
            else:
                cur.append(f)
        else:
            if cur:
                units.append(cur)
            cur = [f]
    if cur:
        units.append(cur)
    return units


def assign_ids(path: Path, fences, chap_slug):
    """Build the (line_idx → new_info) edit map for a file.

    Groups consecutive framework-specific fences into tabset units.
    Each unit gets one base ID; all member fences share it.
    Numbering is per-section, counting units (not individual cells).
    """
    units = group_into_units(fences)

    # File-global set of used IDs (existing or assigned). IDs must be
    # unique across sections within a file, since section-renumbering
    # could otherwise collide (e.g., `<chap>-model-2` from two different
    # collisions). Tabset variants share an ID — that's expected; we
    # only count distinct IDs.
    used_ids = set()
    for unit in units:
        for f in unit:
            eid = extract_existing_id(f['info'])
            if eid:
                used_ids.add(eid)

    # Section → ordered list of units in that section
    section_units = {}
    for unit in units:
        sec = unit[0]['section']
        section_units.setdefault(sec, []).append(unit)

    edits = {}
    for sec, sec_units in section_units.items():
        # Total units in this section (for singleton-suffix decision)
        total_units_in_section = len(sec_units)

        seq = 0
        for unit in sec_units:
            existing = next(
                (extract_existing_id(f['info']) for f in unit
                 if info_has_id(f['info'])), None)

            if existing:
                cid = existing  # propagate to siblings missing IDs
            else:
                seq += 1
                base = f'{chap_slug}-{sec}' if sec else chap_slug
                if total_units_in_section > 1:
                    cid = f'{base}-{seq}'
                else:
                    cid = base
                while cid in used_ids:
                    seq += 1
                    cid = f'{base}-{seq}'
                used_ids.add(cid)

            for f in unit:
                if not info_has_id(f['info']):
                    edits[f['start']] = inject_id(f['info'], cid)

    return edits


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def process_file(path: Path, dry_run=False, check=False, verbose=False):
    """Returns (n_changes, would_change). On check, doesn't write."""
    text = path.read_text(encoding='utf-8')
    lines = text.split('\n')
    chap_slug = file_slug(path)
    fences = collect_fences(lines, chap_slug)
    edits = assign_ids(path, fences, chap_slug)

    if not edits:
        if verbose:
            print(f'  {path}: no changes')
        return 0

    for idx, new_info in edits.items():
        old = lines[idx]
        lines[idx] = '```' + new_info
        if verbose:
            print(f'  {path}:{idx+1}: {old} → {lines[idx]}')

    if check:
        print(f'{path}: would assign {len(edits)} ID(s)')
        return len(edits)

    if not dry_run:
        new_text = '\n'.join(lines)
        path.write_text(new_text, encoding='utf-8')

    return len(edits)


def main():
    parser = argparse.ArgumentParser(
        description='Assign stable cell IDs to Python code fences')
    parser.add_argument('files', nargs='*', type=Path,
                        help='Specific .md files to process (default: all in CHAPTER_NUMBERING)')
    parser.add_argument('--check', action='store_true',
                        help='Report what would change but do not write')
    parser.add_argument('--dry-run', action='store_true',
                        help='Walk all files, count changes, but do not write')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print each ID assignment')
    parser.add_argument('--source', type=Path, default=Path('.'),
                        help='Source directory containing chapter_*/ (default: .)')
    args = parser.parse_args()

    if args.files:
        files = list(args.files)
    else:
        files = [args.source / rel for rel in CHAPTER_NUMBERING.keys()]
        files = [f for f in files if f.exists()]

    total_changes = 0
    files_changed = 0
    for f in files:
        n = process_file(f, dry_run=args.dry_run, check=args.check,
                         verbose=args.verbose)
        if n:
            files_changed += 1
            total_changes += n

    action = 'would assign' if (args.check or args.dry_run) else 'assigned'
    print(f'\nDone. {action} {total_changes} cell ID(s) across '
          f'{files_changed}/{len(files)} files.')


if __name__ == '__main__':
    main()
