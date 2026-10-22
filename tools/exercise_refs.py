"""Labels and references for individual exercise problems.

A problem in a section's `## Exercises` list carries a label on a line of its own inside the item:

    1. [code] **Bigram baseline.** Compute ...
       :exlabel:`ex_gpt-bigram`

and any text in the book refers to it with `:exref:`ex_gpt-bigram`` ("Exercise :exref:`ex_gpt-bigram` of
:numref:`sec_gpt`", "problem :exref:`ex_gpt-bigram`b"). The reference renders as the problem's current number,
so reordering or inserting problems never breaks it. Numbers follow what a reader sees: an untabbed list counts
from its first marker, a list after a tab group starts at its literal number, and inside a :begin_tab: block the
items count from the block's first marker. A label may appear in several framework variants of one problem; all of
them must carry the same number.
"""
import re
import sys
from pathlib import Path

LABEL_LINE = re.compile(r'^[ \t]*:exlabel:`([^`]+)`[ \t]*$')
REF = re.compile(r':exref:`([^`]+)`')
ITEM = re.compile(r'^(\d+)\. ')
TAB_BEGIN = re.compile(r'^:begin_tab:`[^`]*`\s*$')

_INDEX = {}


def _block_range(lines):
    a = next((i for i, l in enumerate(lines) if l.startswith('## Exercises')), None)
    if a is None:
        return None
    for i in range(a + 1, len(lines)):
        l = lines[i]
        if l.startswith('<!-- slides -->') or l.startswith('[Discussions]') or (
                TAB_BEGIN.match(l) and i + 1 < len(lines) and lines[i + 1].startswith('[Discussions]')):
            return a, i
    return a, len(lines)


def labels_in(text, with_count=False):
    """{label: {numbers}} for the exercise block of one source file (and the number of label lines matched)."""
    lines = text.split('\n')
    r = _block_range(lines)
    if r is None:
        return ({}, 0) if with_count else {}
    out, cur, count, in_item, matched = {}, None, None, False, 0
    for l in lines[r[0] + 1:r[1]]:
        if TAB_BEGIN.match(l) or l.startswith(':end_tab:'):
            count, in_item = None, False
            continue
        m = ITEM.match(l)
        if m:
            count = int(m.group(1)) if count is None else count + 1
            cur, in_item = count, True
            continue
        if l.strip() and not l[0].isspace() and not l.startswith('*Adapted from'):
            count, in_item = None, False  # an unindented paragraph ends the list
            continue
        lm = LABEL_LINE.match(l)
        if lm and in_item:
            out.setdefault(lm.group(1), set()).add(cur)
            matched += 1
    return (out, matched) if with_count else out


def index(root):
    """{label: (relative .md path, number)} over every chapter_*/*.md under root."""
    root = Path(root).resolve()
    if root in _INDEX:
        return _INDEX[root]
    idx = {}
    for p in sorted(root.glob('chapter_*/*.md')):
        rel = p.relative_to(root).as_posix()
        for lab, nums in labels_in(p.read_text(encoding='utf-8')).items():
            if len(nums) > 1:
                print(f'exercise_refs: {rel}: label {lab} has different numbers {sorted(nums)}', file=sys.stderr)
            if lab in idx and idx[lab][0] != rel:
                print(f'exercise_refs: label {lab} defined in {idx[lab][0]} and {rel}', file=sys.stderr)
            idx[lab] = (rel, min(nums))
    _INDEX[root] = idx
    return idx


def render(text, src_path, mode='plain'):
    """Strip label lines and replace references by numbers.

    mode 'html': each labeled problem gets an anchor span after its bold title, and each reference becomes a link
    to it; mode 'plain' (notebooks, PDF): references become bare numbers.
    """
    src = Path(src_path).resolve()
    root = src.parent.parent
    idx = index(root)
    rel_self = src.relative_to(root).as_posix()
    lines = text.split('\n')

    if mode == 'html':
        # attach an anchor to the first line of each labeled item
        r = _block_range(lines)
        if r:
            item_start = None
            for i in range(r[0] + 1, r[1]):
                l = lines[i]
                if ITEM.match(l):
                    item_start = i
                    continue
                lm = LABEL_LINE.match(l)
                if lm and item_start is not None:
                    first = lines[item_start]
                    span = f'[]{{#{lm.group(1)}}}'
                    if span not in first:
                        lines[item_start] = re.sub(r'(\*\*.+?\*\*)', lambda m: m.group(1) + span, first, count=1) \
                            if '**' in first else first + ' ' + span
    lines = [l for l in lines if not LABEL_LINE.match(l)]
    text = '\n'.join(lines)

    def sub(m):
        lab = m.group(1)
        if lab not in idx:
            print(f'exercise_refs: {rel_self}: unresolved :exref:`{lab}`', file=sys.stderr)
            return '??'
        rel, n = idx[lab]
        if mode != 'html':
            return str(n)
        if rel == rel_self:
            return f'[{n}](#{lab})'
        target = Path(rel).with_suffix('.qmd')
        here = Path(rel_self).parent
        link = Path('..') / target if here != target.parent else Path(target.name)
        return f'[{n}]({link.as_posix()}#{lab})'

    return REF.sub(sub, text)


def check(root):
    """Problems: unresolved references, duplicate labels, labels outside exercise items. Returns a list of strings."""
    root = Path(root).resolve()
    _INDEX.pop(root, None)
    problems, seen = [], {}
    files = sorted(root.glob('chapter_*/*.md'))
    for p in files:
        rel = p.relative_to(root).as_posix()
        text = p.read_text(encoding='utf-8')
        labs, matched = labels_in(text, with_count=True)
        total = sum(1 for l in text.split('\n') if LABEL_LINE.match(l))
        if total != matched:
            problems.append(f'{rel}: {total - matched} :exlabel: line(s) outside an exercise item')
        for lab, nums in labs.items():
            if len(nums) > 1:
                problems.append(f'{rel}: label {lab} carries different numbers {sorted(nums)}')
            if lab in seen and seen[lab] != rel:
                problems.append(f'label {lab} defined in {seen[lab]} and {rel}')
            seen[lab] = rel
    for p in files:
        rel = p.relative_to(root).as_posix()
        for m in REF.finditer(p.read_text(encoding='utf-8')):
            if m.group(1) not in seen:
                problems.append(f'{rel}: unresolved :exref:`{m.group(1)}`')
    return problems


if __name__ == '__main__':
    probs = check(sys.argv[1] if len(sys.argv) > 1 else '.')
    for x in probs:
        print(x)
    print(f'exercise label problems: {len(probs)}')
    sys.exit(1 if probs else 0)
