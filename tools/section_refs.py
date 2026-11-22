#!/usr/bin/env python3
"""Section-number tokens for slide text.

Slide text (the `<!-- slides -->` section of a source `.md`) names other
parts of the book by number: "see §9.3", "Chapter 13". Hardcoded, those
numbers rot whenever chapters are renumbered. A token names the target by its
`:label:` instead, and `gen_slides.py` replaces it with the current number:

    §{sec_sgd}           → §9.3     (the number, with the section sign)
    §{chap_optimization} → §9
    §{num:sec_sgd}       → 9.3      (the bare number: "Chapter §{num:chap_x}",
                                     the second end of "§{sec_a}–§{num:sec_b}")

A label resolves when it is the `:label:` of the H1 of a file that
`CHAPTER_NUMBERING` (tools/d2l_preprocess.py) numbers: a chapter's `index.md`
(`[N]` → "N") or a section (`[N, k]` → "N.k"). This is the same map the HTML
book, the PDFs and the slide cover kickers number from. Subsection labels
(`##` headings) and labels of unnumbered pages do not resolve; neither do
unknown labels. Every such token is an error, reported with file and line:
`gen_slides.py` exits non-zero and `tools/lint_source.py` reports it. A token
above `<!-- slides -->` is an error too, since the HTML/PDF/notebook renders
do not resolve it (book prose uses :numref:).
"""

import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import CHAPTER_NUMBERING  # noqa: E402

# `§{label}` / `§{num:label}`. The label group is permissive so a malformed
# token ("§{sec x}", "§{}") is still matched and reported, never left as text.
TOKEN_RE = re.compile(r'§\{(?P<num>num:)?(?P<label>[^{}\n]*)\}')

_FENCE_RE = re.compile(r'^\s*(```|~~~)')
_H1_RE = re.compile(r'^# \S')
_LABEL_LINE_RE = re.compile(r'^:label:`([^`]+)`\s*$')
SLIDES_MARKER_RE = re.compile(r'^<!--\s*slides\s*-->\s*$')


class SectionRefError(Exception):
    """One or more unresolvable tokens; .problems is [(path, line, msg)]."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__('\n'.join(f'{p}:{n}: error: {m}'
                                   for p, n, m in problems))


def h1_label(text):
    """The `:label:` of the file's H1 (blank lines may separate them), or None.

    The H1 is the first `# ` heading outside code fences and above
    `<!-- slides -->`.
    """
    lines = text.split('\n')
    in_fence = False
    for i, line in enumerate(lines):
        if SLIDES_MARKER_RE.match(line):
            return None
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not _H1_RE.match(line):
            continue
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                continue
            m = _LABEL_LINE_RE.match(nxt)
            return m.group(1) if m else None
        return None
    return None


def format_number(nums):
    return '.'.join(str(n) for n in nums)


@lru_cache(maxsize=None)
def _label_table(root):
    """({label: (rel, nums or None)}, {(nums tuple): label}) for the corpus."""
    by_label, by_number = {}, {}
    for rel, nums in CHAPTER_NUMBERING.items():
        p = Path(root) / rel
        if not p.exists():
            continue
        label = h1_label(p.read_text(encoding='utf-8'))
        if label is None:
            continue
        by_label[label] = (rel, tuple(nums) if nums else None)
        if nums:
            by_number[tuple(nums)] = label
    return by_label, by_number


def label_table(root='.'):
    return _label_table(str(Path(root).resolve()))


def resolve_label(label, root='.'):
    """Number string for a chapter/section label; raises KeyError(reason)."""
    by_label, _ = label_table(root)
    hit = by_label.get(label)
    if hit is None:
        raise KeyError(f'unknown section label `{label}` (only the H1 '
                       f'labels of chapter and section pages resolve)')
    rel, nums = hit
    if nums is None:
        raise KeyError(f'label `{label}` names {rel}, which '
                       f'CHAPTER_NUMBERING leaves unnumbered')
    return format_number(nums)


def _slides_start(lines):
    return next((i for i, l in enumerate(lines) if SLIDES_MARKER_RE.match(l)),
                len(lines))


def check(text, path, root='.'):
    """[(path, line, message)] for every bad token in text."""
    problems = []
    lines = text.split('\n')
    start = _slides_start(lines)
    for i, line in enumerate(lines):
        for m in TOKEN_RE.finditer(line):
            if i < start:
                problems.append((path, i + 1,
                                 f'section token `{m.group(0)}` outside the '
                                 f'slides section renders literally; use '
                                 f':numref: in book prose'))
                continue
            try:
                resolve_label(m.group('label'), root)
            except KeyError as e:
                problems.append((path, i + 1,
                                 f'`{m.group(0)}`: {e.args[0]}'))
    return problems


def resolve(text, path, root='.'):
    """Replace every token in text; raise SectionRefError on any bad one."""
    problems = check(text, path, root)
    if problems:
        raise SectionRefError(problems)
    if '§{' not in text:
        return text

    def sub(m):
        num = resolve_label(m.group('label'), root)
        return num if m.group('num') else '§' + num
    return TOKEN_RE.sub(sub, text)
