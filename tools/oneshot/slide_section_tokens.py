#!/usr/bin/env python3
"""Rewrite hardcoded section numbers in slide text as §{label} tokens.

One-shot migration for the section-number tokens of tools/section_refs.py.
Slide sections (below `<!-- slides -->`) name other parts of the book by
number ("§9.3", "§10.1–10.3", "Chapter 13", "Chapters 14 and 15",
"Section 2.5"); those numbers rot whenever chapters are renumbered. This
script maps each number back to the label of the page that carries it under
the CURRENT numbering (CHAPTER_NUMBERING + each page's H1 `:label:`) and
writes the token instead:

    §9.3             → §{sec_sgd}
    §10.1–10.3       → §{sec_a}–§{num:sec_c}       (the second end has no §)
    §2.3–§2.4        → §{sec_a}–§{sec_b}
    §§1.2, 1.1       → §§{sec_a}, §{num:sec_b}
    Chapter 13       → Chapter §{num:chap_x}
    Chapters 14 and 15 → Chapters §{num:chap_a} and §{num:chap_b}
    Section 2.5      → Section §{num:sec_x}

So the generated slide text is byte-identical before and after (the token
resolves to the number it replaced). Only text below `<!-- slides -->` is
touched; the cover kicker "[Dive into Deep Learning · §N.M]{.kicker}" is left
alone (gen_slides.py already rewrites it from CHAPTER_NUMBERING), as are code
fences, inline code and math.

Refused (listed, never rewritten), because the right label cannot be told
from the number alone:
  - a number no numbered page carries (a stale or wrong reference);
  - a page that carries the number but has no H1 `:label:`;
  - a short range end ("§15.4–5"), a dotted "Chapter N.M", a bare
    "Section N";
  - a bare section-like number left beside a rewritten one ("§§2.3 -> 2.4");
  - a number within 60 characters of a citation or another work ("[@…]",
    "et al.", "paper", "survey", "thesis", "arXiv", "textbook"), which may name
    a section of that work.

Usage (from the repo root):
    python3 tools/oneshot/slide_section_tokens.py            # dry run: list changes
    python3 tools/oneshot/slide_section_tokens.py --check    # same; exit 1 if any change is pending
    python3 tools/oneshot/slide_section_tokens.py --apply    # rewrite the files
    ... [--root DIR] [files ...]

--root runs against another checkout (e.g. a scratch copy). Exit status of a
dry run or --apply is 0; --check exits 1 when it would change anything.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from d2l_preprocess import CHAPTER_NUMBERING  # noqa: E402
import section_refs  # noqa: E402

NUM = r'\d+(?:\.\d+)*'
DASH = r'\s*(?:–|—|--|-)\s*'
# "§9.3", "§10.1–10.3", "§2.3–§2.4". The lookarounds keep a number whole.
SECT_RE = re.compile(
    rf'(?<![\w.])§(?P<a>{NUM})(?![\w])'
    rf'(?:(?P<dash>{DASH})(?P<sign>§)?(?P<b>{NUM})(?![\w]))?')
# "Chapter 13", "Chapters 2–17", "Chapters 14 and 15", "Section 2.5",
# "Sections 16.1–16.3", "Ch. 11".
SEP = rf'(?:{DASH}|,\s*(?:and\s+|or\s+)?|\s+(?:and|or|to)\s+)'
WORD_RE = re.compile(
    rf'\b(?P<word>Chapters?|Sections?|Ch\.)\s+'
    rf'(?P<list>{NUM}(?:{SEP}{NUM})*)(?![\w])(?!\.\d)')
# "§§1.2, 1.1", "§§1.1 & 1.3", "§§2.3 -> 2.4 -> 2.5": a plural sign before a
# list. (A range "§§10.1–10.3" is left to SECT_RE.)
LIST_SEP = rf'(?:\s*(?:,|&|->|→)\s*(?:and\s+)?|\s+and\s+)'
LIST_RE = re.compile(
    rf'(?<![\w.])§§(?P<list>{NUM}(?:(?:{DASH})?{NUM})?'
    rf'(?:{LIST_SEP}{NUM}(?:{DASH}{NUM})?)+)(?![\w])(?!\.\d)')
BARE_RE = re.compile(r'(?<![\w.§$\\])\d{1,2}\.\d{1,2}(?![\w.%])')
KICKER_RE = re.compile(r'\[Dive into Deep Learning · §[0-9.]+\]\{\.kicker\}')
CODE_RE = re.compile(r'`[^`\n]*`')
MATH_RE = re.compile(r'\$\$.*?\$\$|\$[^$\n]+\$')
# Signs that a number may name a section of another work.
# ("book" is not one: in slide text it is this book.)
FOREIGN_RE = re.compile(
    r'\[@|:cite|et al\b|\b(?:paper|survey|thesis|arXiv|textbook)\b', re.I)


def mask(line):
    """The line with code spans, math and the cover kicker blanked out."""
    out = line
    for rx in (KICKER_RE, CODE_RE, MATH_RE):
        out = rx.sub(lambda m: '\0' * len(m.group(0)), out)
    return out


class Converter:
    def __init__(self, root):
        self.root = Path(root)
        by_label, by_number = section_refs.label_table(root)
        self.by_label = by_label
        self.by_number = by_number
        # Numbered pages with no H1 label: number → rel.
        self.unlabeled = {}
        for rel, nums in CHAPTER_NUMBERING.items():
            if nums and tuple(nums) not in by_number:
                self.unlabeled[tuple(nums)] = rel

    def title(self, label):
        rel = self.by_label[label][0]
        for line in (self.root / rel).read_text(encoding='utf-8').split('\n'):
            if line.startswith('# '):
                return line[2:].strip()
        return '?'

    def lookup(self, num_str):
        """(label, None) or (None, reason)."""
        nums = tuple(int(x) for x in num_str.split('.'))
        if len(nums) > 2:
            return None, f'{num_str} is a subsection number (no token form)'
        if nums in self.by_number:
            return self.by_number[nums], None
        if nums in self.unlabeled:
            return None, (f'{num_str} is {self.unlabeled[nums]}, which has '
                          f'no H1 :label:')
        return None, f'no page is numbered {num_str}'

    def convert_line(self, line):
        """(new_line, [change dicts], [refusal strings])."""
        masked = mask(line)
        edits, changes, refusals = [], [], []  # edits: (start, end, text)

        def foreign(m):
            ctx = masked[max(0, m.start() - 60):m.end() + 60]
            return FOREIGN_RE.search(ctx)

        for m in LIST_RE.finditer(masked):
            if foreign(m):
                refusals.append(f'"{m.group(0)}": next to a citation or '
                                f'another work ("{foreign(m).group(0)}")')
                continue
            pos = m.start('list')
            parts, targets, prev, bad = ['§'], [], pos, None
            for k, nm in enumerate(re.finditer(NUM, m.group('list'))):
                lab, why = self.lookup(nm.group(0))
                if lab is None:
                    bad = f'"{m.group(0)}": {why}'
                    break
                s, e = pos + nm.start(), pos + nm.end()
                parts.append(line[prev:s])
                parts.append(f'§{{{lab}}}' if k == 0 else f'§{{num:{lab}}}')
                prev = e
                targets.append((nm.group(0), lab))
            if bad:
                refusals.append(bad)
                continue
            new = ''.join(parts) + line[prev:m.end()]
            edits.append((m.start(), m.end(), new))
            changes.append((m.group(0), new, targets))
        in_list = [(s, e) for s, e, _ in edits]

        for m in SECT_RE.finditer(masked):
            if any(s <= m.start() < e for s, e in in_list):
                continue
            a, b = m.group('a'), m.group('b')
            if foreign(m):
                refusals.append(f'"{m.group(0)}": next to a citation or '
                                f'another work ("{foreign(m).group(0)}")')
                continue
            if b and '.' in a and '.' not in b and not m.group('sign'):
                refusals.append(f'"{m.group(0)}": short range end')
                continue
            la, why = self.lookup(a)
            if la is None:
                refusals.append(f'"{m.group(0)}": {why}')
                continue
            new = f'§{{{la}}}'
            targets = [(a, la)]
            if b:
                lb, why = self.lookup(b)
                if lb is None:
                    refusals.append(f'"{m.group(0)}": {why}')
                    continue
                new += m.group('dash') + (f'§{{{lb}}}' if m.group('sign')
                                          else f'§{{num:{lb}}}')
                targets.append((b, lb))
            edits.append((m.start(), m.end(), new))
            changes.append((m.group(0), new, targets))

        for m in WORD_RE.finditer(masked):
            word = m.group('word')
            if foreign(m):
                refusals.append(f'"{m.group(0)}": next to a citation or '
                                f'another work ("{foreign(m).group(0)}")')
                continue
            nums = list(re.finditer(NUM, m.group('list')))
            chapter = word.startswith('Ch')
            bad = None
            parts, targets = [], []
            pos = m.start('list')
            prev = pos
            for nm in nums:
                n = nm.group(0)
                if chapter and '.' in n:
                    bad = f'"{m.group(0)}": dotted number after "{word}"'
                elif not chapter and '.' not in n:
                    bad = f'"{m.group(0)}": bare number after "{word}"'
                if bad:
                    break
                lab, why = self.lookup(n)
                if lab is None:
                    bad = f'"{m.group(0)}": {why}'
                    break
                s, e = pos + nm.start(), pos + nm.end()
                parts.append(line[prev:s])
                parts.append(f'§{{num:{lab}}}')
                prev = e
                targets.append((n, lab))
            if bad:
                refusals.append(bad)
                continue
            new = line[m.start():m.start('list')] + ''.join(parts) + \
                line[prev:m.end()]
            edits.append((m.start(), m.end(), new))
            changes.append((m.group(0), new, targets))

        # A bare number left beside a rewritten one ("§§2.3 -> 2.4 -> 2.5")
        # stays hardcoded: list it for a hand edit, do not guess at it.
        chapters = {n.split('.')[0] for _, _, ts in changes for n, _ in ts}
        covered = [(s, e) for s, e, _ in edits]
        for m in BARE_RE.finditer(masked):
            if any(s <= m.start() < e for s, e in covered):
                continue
            n = m.group(0)
            nums = tuple(int(x) for x in n.split('.'))
            if n.split('.')[0] in chapters and nums in self.by_number:
                refusals.append(f'"{n}": bare section-like number beside a '
                                f'rewritten reference; left as text')

        edits.sort()
        for (s1, e1, _), (s2, _, _) in zip(edits, edits[1:]):
            if s2 < e1:
                raise RuntimeError(f'overlapping rewrites in: {line}')
        out = line
        for s, e, new in reversed(edits):
            out = out[:s] + new + out[e:]
        return out, changes, refusals

    def convert_text(self, text):
        lines = text.split('\n')
        start = next((i for i, l in enumerate(lines)
                      if section_refs.SLIDES_MARKER_RE.match(l)), None)
        if start is None:
            return text, [], []
        changes, refusals = [], []
        in_fence = False
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if re.match(r'^\s*(```|~~~)', line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            new, ch, rf = self.convert_line(line)
            changes += [(i + 1, *c) for c in ch]
            refusals += [(i + 1, r) for r in rf]
            lines[i] = new
        return '\n'.join(lines), changes, refusals


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*')
    ap.add_argument('--root', default='.')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true',
                      help='dry run; exit 1 if any change is pending')
    mode.add_argument('--apply', action='store_true',
                      help='rewrite the files in place')
    args = ap.parse_args()

    root = Path(args.root)
    conv = Converter(root)
    rels = args.files or sorted(p.relative_to(root).as_posix()
                                for p in root.glob('chapter_*/*.md'))
    n_changes = n_refusals = n_files = 0
    for rel in rels:
        path = root / rel
        text = path.read_text(encoding='utf-8')
        new, changes, refusals = conv.convert_text(text)
        if not changes and not refusals:
            continue
        for line_no, old, rep, targets in changes:
            tgt = '; '.join(f'{n} = {lab} "{conv.title(lab)}"'
                            for n, lab in targets)
            print(f'{rel}:{line_no}: {old!r} -> {rep!r}  [{tgt}]')
        for line_no, why in refusals:
            print(f'{rel}:{line_no}: REFUSED {why}')
        n_changes += len(changes)
        n_refusals += len(refusals)
        if changes:
            n_files += 1
        if args.apply and new != text:
            path.write_text(new, encoding='utf-8')
    verb = 'rewrote' if args.apply else 'would rewrite'
    print(f'\n{verb} {n_changes} reference(s) in {n_files} file(s); '
          f'refused {n_refusals}.')
    if args.check and n_changes:
        sys.exit(1)


if __name__ == '__main__':
    main()
