#!/usr/bin/env python3
"""Convert inline slide markers to ::: {.slide} divs in chapter_*/*.md.

Inline markers being migrated:
  [**text**]   → starts a NEW slide; book-visible
  (**text**)   → continues current slide; book-visible
  [~~text~~]   → starts a NEW slide; slide-only (hidden from book)
  (~~text~~)   → continues current slide; slide-only (hidden from book)

Migration strategy:
  1. Parse the source into MarkdownBlocks and CodeBlocks (with cell IDs
     from add_cell_ids.py).
  2. Walk blocks; group into "slide groups". A new group starts at any
     `[**...**]` or `[~~...~~]` marker in a markdown block. Code cells
     between markers belong to the current group.
  3. For each slide group, emit `::: {.slide title="..."} ... :::` divs.
     The title (when present) comes from the file's H1 — only the
     first slide gets it. Code cells become `@<id>` placeholders.
  4. Strip markers from source prose:
     - `[**X**]` and `(**X**)` → `X` (book-visible content stays)
     - `[~~X~~]` and `(~~X~~)` → `` (slide-only, removed from prose)
  5. Append slide divs to the end of the file under a `<!-- slides -->`
     comment marker.

The book renderer strips `::: {.slide}` divs (handled in
d2l_preprocess.py post-migration). Slide rendering reads the divs.

Idempotent: skips files that already have a `<!-- slides -->` block.

Usage:
    python tools/migrate_slide_markers.py [--check] [--verbose] [files...]
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import (CHAPTER_NUMBERING, is_python_block, is_boilerplate,
                             extract_tab)


# ──────────────────────────────────────────────────────────
# Block parsing (line-aware)
# ──────────────────────────────────────────────────────────

class MdBlock:
    __slots__ = ('start', 'end', 'text')
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text

class CodeBlk:
    __slots__ = ('start', 'end', 'info', 'cell_id', 'tab')
    def __init__(self, start, end, info, cell_id, tab):
        self.start, self.end = start, end
        self.info, self.cell_id, self.tab = info, cell_id, tab


_FENCE_OPEN_RE = re.compile(r'^```(.*)$')
_FENCE_CLOSE_RE = re.compile(r'^```\s*$')
_ID_IN_INFO_RE = re.compile(r'(?:^|[\s{])#([a-z][a-z0-9-]*)(?=\s|\}|$)')


def parse_blocks_with_lines(lines):
    """Return list[MdBlock | CodeBlk]. Boilerplate fences are dropped."""
    blocks = []
    md_start = 0
    md_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE_OPEN_RE.match(line)
        if m and not line.startswith('````'):
            # Flush md
            if md_lines:
                blocks.append(MdBlock(md_start, i - 1, '\n'.join(md_lines)))
                md_lines = []
            info = m.group(1).strip()
            fence_start = i
            body = []
            i += 1
            while i < len(lines) and not _FENCE_CLOSE_RE.match(lines[i]):
                body.append(lines[i])
                i += 1
            fence_end = i if i < len(lines) else len(lines) - 1
            i += 1
            md_start = i

            if info == 'toc':
                continue
            if is_python_block(info):
                if is_boilerplate(body):
                    continue
                cid_m = _ID_IN_INFO_RE.search(info)
                cid = cid_m.group(1) if cid_m else None
                tab, _ = extract_tab(body)
                blocks.append(CodeBlk(fence_start, fence_end, info, cid, tab))
            else:
                # Non-Python (bash, etc.) — not in slides
                blocks.append(CodeBlk(fence_start, fence_end, info, None,
                                       'non-python'))
        else:
            if not md_lines:
                md_start = i
            md_lines.append(line)
            i += 1
    if md_lines:
        blocks.append(MdBlock(md_start, len(lines) - 1, '\n'.join(md_lines)))
    return blocks


# ──────────────────────────────────────────────────────────
# Marker extraction
# ──────────────────────────────────────────────────────────

PAIRS = (('[**', '**]', True, True),    # new slide, book-visible
         ('(**', '**)', False, True),   # continue, book-visible
         ('[~~', '~~]', True, False),   # new slide, slide-only
         ('(~~', '~~)', False, False))  # continue, slide-only


def find_markers(text):
    """Return list of (pos, inner_text, new_slide, book_visible)."""
    matches = []
    for open_mark, close_mark, new_slide, book_visible in PAIRS:
        start = 0
        while True:
            s = text.find(open_mark, start)
            if s == -1:
                break
            e = text.find(close_mark, s + len(open_mark))
            if e == -1:
                break
            inner = text[s + len(open_mark):e].strip()
            matches.append((s, inner, new_slide, book_visible))
            start = e + len(close_mark)
    matches.sort(key=lambda x: x[0])
    return matches


def extract_h1(text):
    """First H1 in the markdown text, with :label:/anchors stripped."""
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('# ') and not line.startswith('## '):
            heading = re.sub(r'\s*\{#[^}]+\}', '', line[2:]).strip()
            return heading
    return None


# ──────────────────────────────────────────────────────────
# Slide-group construction
# ──────────────────────────────────────────────────────────

class Slide:
    """One emitted ::: {.slide ...} div.

    Entries: ordered list of ('text', str) | ('code', cell_id) tuples.
    """
    __slots__ = ('title', 'entries')
    def __init__(self, title=None):
        self.title = title
        self.entries = []


def build_slides(blocks):
    """Group source blocks into slide objects.

    Walks blocks in order, mirroring the existing gen_slides.py extract
    logic. Each MD block produces (at most) one text entry combining all
    its markers; if any marker is `[`-style, it starts a new slide.
    Code cells append to the current slide.

    Slide titles are NOT set during migration — the file's H1 acts as
    the implicit deck title in gen_slides.py emission, and slides
    without explicit titles render with `---` separators. Authors can
    add `title="..."` attributes to specific slides later.
    """
    slides = []
    current = None

    def ensure_slide():
        nonlocal current
        if current is None:
            current = Slide(title=None)
            slides.append(current)

    for block in blocks:
        if isinstance(block, MdBlock):
            markers = find_markers(block.text)
            if not markers:
                continue
            new_slide_in_block = any(m[2] for m in markers)
            texts = [m[1] for m in markers]
            combined = ' '.join(t for t in texts if t).strip().rstrip(',. \n:')
            if not combined:
                continue
            if new_slide_in_block:
                current = Slide(title=None)
                slides.append(current)
            else:
                ensure_slide()
            current.entries.append(('text', combined))
        elif isinstance(block, CodeBlk):
            if block.tab == 'non-python':
                continue
            if block.cell_id is None:
                continue
            ensure_slide()
            # Dedupe: tabset variants share a base ID. One @<id> placeholder
            # is enough — the slide builder picks the framework's variant.
            if (current.entries
                    and current.entries[-1] == ('code', block.cell_id)):
                continue
            current.entries.append(('code', block.cell_id))

    return slides


# ──────────────────────────────────────────────────────────
# Emission
# ──────────────────────────────────────────────────────────

def emit_slide(slide):
    attrs = []
    if slide.title:
        # Quote-escape any " in the title
        attrs.append(f'title="{slide.title.replace("\"", "\\\"")}"')
    header = f'::: {{.slide{(" " + " ".join(attrs)) if attrs else ""}}}'
    parts = [header]
    for kind, payload in slide.entries:
        if kind == 'text':
            parts.append(payload)
        elif kind == 'code':
            parts.append(f'@{payload}')
    parts.append(':::')
    return '\n\n'.join(parts) + '\n'


def strip_inline_markers(text):
    """Remove inline slide markers from prose."""
    # Order matters: book-visible markers leave inner text behind;
    # slide-only markers remove the entire match.
    text = re.sub(r'\[\*\*(.*?)\*\*\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\(\*\*(.*?)\*\*\)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\[~~.*?~~\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\(~~.*?~~\)', '', text, flags=re.DOTALL)
    return text


SLIDES_MARKER = '<!-- slides -->'


def has_slides_section(text):
    return SLIDES_MARKER in text


# ──────────────────────────────────────────────────────────
# Per-file driver
# ──────────────────────────────────────────────────────────

def process_file(path: Path, dry_run=False, verbose=False):
    text = path.read_text(encoding='utf-8')
    if has_slides_section(text):
        if verbose:
            print(f'  {path}: already migrated')
        return 0

    lines = text.split('\n')
    blocks = parse_blocks_with_lines(lines)
    slides = build_slides(blocks)
    if not slides:
        if verbose:
            print(f'  {path}: no slide markers')
        return 0

    # Strip markers from source body
    new_body = strip_inline_markers(text)
    # Trim trailing whitespace before appending slides
    new_body = new_body.rstrip() + '\n\n'

    # Build slides section
    sections = [SLIDES_MARKER, '']
    for slide in slides:
        sections.append(emit_slide(slide))
    new_body += '\n'.join(sections).rstrip() + '\n'

    if dry_run:
        print(f'{path}: would emit {len(slides)} slide(s)')
        return len(slides)

    path.write_text(new_body, encoding='utf-8')
    if verbose:
        print(f'  {path}: emitted {len(slides)} slide(s)')
    return len(slides)


def main():
    parser = argparse.ArgumentParser(
        description='Convert inline slide markers to ::: {.slide} divs')
    parser.add_argument('files', nargs='*', type=Path)
    parser.add_argument('--check', action='store_true',
                        help='Walk files, count what would change, do not write')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--source', type=Path, default=Path('.'))
    args = parser.parse_args()

    if args.files:
        files = list(args.files)
    else:
        files = [args.source / rel for rel in CHAPTER_NUMBERING.keys()]
        files = [f for f in files if f.exists()]

    total = 0
    files_changed = 0
    for f in files:
        n = process_file(f, dry_run=args.check, verbose=args.verbose)
        if n:
            files_changed += 1
            total += n

    action = 'would emit' if args.check else 'emitted'
    print(f'\nDone. {action} {total} slide div(s) across '
          f'{files_changed}/{len(files)} files.')


if __name__ == '__main__':
    main()
