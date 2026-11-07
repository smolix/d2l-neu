#!/usr/bin/env python3
"""Rewrite each notebook's imports cell to the standard preamble.

For every source file and framework tab, the imports cell (the cell whose
top level runs `from d2l import <fw> as d2l`) becomes

1. the canonical preamble of tools/preamble.py, byte for byte, keeping the
   cell's id and its `%%tab` line; and
2. if the notebook imports anything more, a second short cell right after
   the tab group with just those extra imports, sorted, under a new id from
   tools/add_cell_ids.py's scheme (one id shared by the frameworks' variants,
   like any tab group).

`%matplotlib inline` is dropped (the Jupyter default backend is inline). A
cell holding anything besides imports — comments, `#@save`, environment
settings, seeds, other statements — or importing a name the preamble binds
to a different object (MXNet's `from mxnet import np`) is left untouched and
listed for manual review; so is a second imports cell for the same
framework in one file. The teach-once page (preamble.TEACH_FILE) is skipped.

Dry run by default: prints the plan and a summary. `--apply` writes.

Usage:
    python tools/normalize_imports.py [--apply] [--verbose]
        [--frameworks pytorch,jax] [files...]
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import CHAPTER_NUMBERING, FRAMEWORKS, extract_tab
from gen_notebooks import file_supports_framework
from add_cell_ids import (assign_ids, collect_fences, extract_existing_id,
                          file_slug, group_into_units, inject_id)
import preamble

_SLIDES_MARKER_RE = re.compile(r'^<!--\s*slides\s*-->\s*$')


class Plan:
    """What normalize_file decided for one file."""

    def __init__(self, path):
        self.path = path
        self.canonical = []   # (fw, line) already canonical
        self.rewritten = []   # (fw, line, cell_id, extras: list[str], dropped)
        self.manual = []      # (fw, line, cell_id, reasons)
        self.extras_ids = []  # new ids
        self.extras_fences = 0  # extras cells written (after merging)
        self.new_text = None

    def changed_fws(self):
        return {fw for fw, *_ in self.rewritten}


def _cell_body(lines, f):
    """(tab_line or None, code_lines) of fence f."""
    body = lines[f['start'] + 1:f['end']]
    tab, code = extract_tab(list(body))
    tab_line = None
    if len(code) != len(body):
        tab_line = body[len(body) - len(code) - 1]
    return tab_line, code


def _merge_extras(unit_extras, file_fws):
    """One extras cell per distinct import list (code-guide §2: shared when
    identical): `%%tab a, b` for frameworks whose extras agree, untagged when
    they agree across all four (Basics; Advanced files spell out
    `%%tab pytorch, jax`)."""
    groups = {}
    for tab_line, fw, extras in unit_extras:
        groups.setdefault(tuple(extras), []).append((tab_line, fw))
    out = []
    for extras, members in groups.items():
        tab_line = members[0][0]
        fws = [fw for _, fw in members]
        if tab_line is not None and len(fws) > 1:
            if sorted(fws) == sorted(file_fws) == sorted(FRAMEWORKS):
                tab_line = None
            else:
                prefix = tab_line.split()[0]  # %%tab or #@tab
                tab_line = f'{prefix} {", ".join(fws)}'
        out.append((tab_line, list(extras)))
    return out


def normalize_file(path, frameworks=FRAMEWORKS):
    """Return a Plan; plan.new_text is the rewritten file text (or None when
    nothing changes)."""
    plan = Plan(path)
    text = path.read_text(encoding='utf-8')
    lines = text.split('\n')
    # Only cells above the slides marker are notebook cells.
    end = next((i for i, l in enumerate(lines) if _SLIDES_MARKER_RE.match(l)),
               len(lines))
    fences = [f for f in collect_fences(lines, file_slug(path))
              if f['end'] < end]
    units = group_into_units(fences)
    file_fws = [fw for fw in FRAMEWORKS if file_supports_framework(path, fw)]

    replace = {}       # fence start → new body lines (between the fences)
    insert_after = {}  # line index → list of (tab_line, extras lines)
    seen_fw = set()
    for unit in units:
        unit_extras = []
        for f in unit:
            tab_line, code = _cell_body(lines, f)
            fw = preamble.imports_cell_framework(code)
            if fw is None:
                continue
            line_no = f['start'] + 1
            cid = extract_existing_id(f['info'])
            tab = f['tab']
            if fw not in frameworks:
                seen_fw.add(fw)
                continue
            tab_fws = ([t.strip() for t in tab.split(',')]
                       if tab not in (None, 'all') else None)
            reasons = []
            if fw in seen_fw:
                reasons.append(f'second {fw} imports cell in this file')
            if tab_fws is not None and tab_fws != [fw]:
                reasons.append(f'cell tagged `{tab}` but imports d2l for {fw}')
            seen_fw.add(fw)
            if reasons:
                plan.manual.append((fw, line_no, cid, reasons))
                continue
            c = preamble.classify(code, fw)
            if c.status == 'canonical':
                plan.canonical.append((fw, line_no))
                continue
            if c.status == 'manual':
                plan.manual.append((fw, line_no, cid, c.reasons))
                continue
            new_body = ([tab_line] if tab_line is not None else []) + \
                preamble.CANONICAL[fw].split('\n')
            replace[f['start']] = (f['end'], new_body)
            extras = preamble.render_imports(c.extras)
            plan.rewritten.append((fw, line_no, cid, extras, c.dropped))
            if extras:
                unit_extras.append((tab_line, fw, extras))
        if unit_extras:
            insert_after[unit[-1]['end']] = _merge_extras(unit_extras,
                                                          file_fws)

    if not replace:
        return plan

    # Build the new text; remember where each inserted fence opens.
    out, inserted = [], []
    i = 0
    while i < len(lines):
        if i in replace:
            fence_end, body = replace[i]
            out.append(lines[i])
            out.extend(body)
            out.append(lines[fence_end])
            i = fence_end
        else:
            out.append(lines[i])
        if i in insert_after:
            group = []
            for tab_line, extras in insert_after[i]:
                plan.extras_fences += 1
                out.append('')
                group.append(len(out))
                out.append('```{.python .input}')
                if tab_line is not None:
                    out.append(tab_line)
                out.extend(extras)
                out.append('```')
            inserted.append(group)
        i += 1

    # Ids for the inserted extras cells from add_cell_ids' scheme; the
    # variants of one tab group share the first one's id.
    edits = assign_ids(path, collect_fences(out, file_slug(path)),
                       file_slug(path))
    for group in inserted:
        info = edits.get(group[0], '{.python .input}')
        cid = extract_existing_id(info)
        plan.extras_ids.append(cid)
        for idx in group:
            out[idx] = '```' + inject_id('{.python .input}', cid)

    plan.new_text = '\n'.join(out)
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('files', nargs='*', type=Path)
    ap.add_argument('--apply', action='store_true',
                    help='write the rewritten files (default: dry run)')
    ap.add_argument('--frameworks', default=','.join(FRAMEWORKS),
                    help='comma-separated frameworks to normalize')
    ap.add_argument('--verbose', '-v', action='store_true',
                    help='print every rewritten cell and its extras')
    ap.add_argument('--source', type=Path, default=Path('.'))
    args = ap.parse_args()
    fws = [f.strip() for f in args.frameworks.split(',') if f.strip()]

    if args.files:
        files = list(args.files)
    else:
        files = [args.source / rel for rel in CHAPTER_NUMBERING]
        files = [f for f in files if f.exists()]
    files = [f for f in files
             if f.resolve() != (args.source / preamble.TEACH_FILE).resolve()]

    stats = defaultdict(Counter)
    manual, stale = [], defaultdict(set)
    total_fences = [0]
    for path in files:
        plan = normalize_file(path, fws)
        for fw, _ in plan.canonical:
            stats[fw]['imports cells'] += 1
            stats[fw]['already canonical'] += 1
        for fw, line, cid, extras, dropped in plan.rewritten:
            stats[fw]['imports cells'] += 1
            stats[fw]['normalizable'] += 1
            stats[fw]['extras cells' if extras else 'no extras'] += 1
            if (args.source / 'outputs' / fw / path.relative_to(
                    args.source).with_suffix('.json')).exists():
                stale[fw].add(str(path))
            if args.verbose:
                ex = ('; '.join(extras)) if extras else '(no extras)'
                print(f'{path}:{line}: {fw} #{cid} → preamble + {ex}')
        total_fences[0] += plan.extras_fences
        for fw, line, cid, reasons in plan.manual:
            stats[fw]['imports cells'] += 1
            stats[fw]['manual review'] += 1
            manual.append((path, line, fw, cid, reasons))
        if plan.new_text is not None and args.apply:
            path.write_text(plan.new_text, encoding='utf-8')

    print('\nManual review (left unchanged):')
    for path, line, fw, cid, reasons in manual:
        print(f'  {path}:{line}: {fw} #{cid}: {"; ".join(reasons)}')
    print('\nSummary' + ('' if args.apply else ' (dry run; --apply writes)'))
    keys = ['imports cells', 'already canonical', 'normalizable', 'no extras',
            'extras cells', 'manual review']
    print(f'  {"framework":<11}' + ''.join(f'{k:>18}' for k in keys)
          + f'{"stored nbs staled":>18}')
    for fw in fws:
        print(f'  {fw:<11}' + ''.join(f'{stats[fw][k]:>18}' for k in keys)
              + f'{len(stale[fw]):>18}')
    total = Counter()
    for fw in fws:
        total.update(stats[fw])
    print(f'  {"total":<11}' + ''.join(f'{total[k]:>18}' for k in keys)
          + f'{sum(len(s) for s in stale.values()):>18}')
    print(f'  "extras cells" counts framework variants; merged where '
          f'identical they make {total_fences[0]} new cells in the source.')


if __name__ == '__main__':
    main()
