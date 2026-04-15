#!/usr/bin/env python3
"""Generate per-framework Reveal.js slides from d2l-en source.

The d2l source uses marker pairs to annotate what goes into slides:
  [**text**]  → included in slides, starts a NEW slide
  (**text**)  → included in slides, continues current slide
  [~~text~~]  → included in slides (slide-only), starts a NEW slide
  (~~text~~)  → included in slides (slide-only), continues current slide
  unmarked    → NOT included in slides (book-only)

Code cells are included as-is. Level-1 headings are always included.

Usage:
    python tools/gen_slides.py <source_dir> <output_dir> [--frameworks pytorch jax]
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import (
    FRAMEWORKS, FRAMEWORK_DISPLAY, CHAPTER_NUMBERING,
    extract_tab, is_boilerplate, is_python_block,
    translate_directives,
)
from build_lib import flatten_tab_branches


# ──────────────────────────────────────────────────────────
# Slide marker extraction
# ──────────────────────────────────────────────────────────

# The four marker pairs
PAIRS = (('[**', '**]'), ('(**', '**)'), ('[~~', '~~]'), ('(~~', '~~)'))


def extract_slide_text(text):
    """Extract slide-marked text from a markdown block.

    Returns list of (text, new_slide) tuples. new_slide is True if the
    marker starts with '[' (new slide) vs '(' (continue).
    """
    matches = []
    for open_mark, close_mark in PAIRS:
        start = 0
        while True:
            s = text.find(open_mark, start)
            if s == -1:
                break
            e = text.find(close_mark, s + len(open_mark))
            if e == -1:
                break
            inner = text[s + len(open_mark):e]
            new_slide = open_mark.startswith('[')
            matches.append((s, inner.strip(), new_slide))
            start = e + len(close_mark)

    matches.sort(key=lambda x: x[0])
    return [(inner, ns) for _, inner, ns in matches]


def extract_heading(text):
    """Extract level-1 heading from a markdown block, if any."""
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('# ') and not line.startswith('## '):
            # Strip label
            heading = re.sub(r'\s*\{#[^}]+\}', '', line)
            return heading
    return None


# ──────────────────────────────────────────────────────────
# Prose tab filtering for slides (single framework)
# ──────────────────────────────────────────────────────────

def filter_prose_tabs(text, framework):
    """Keep only the target framework's prose tab content."""
    tab_pattern = re.compile(
        r':begin_tab:`([^`]+)`\s*\n(.*?)\n?:end_tab:', re.DOTALL)

    result = []
    last_end = 0
    for m in tab_pattern.finditer(text):
        result.append(text[last_end:m.start()])
        fw_key = m.group(1)
        content = m.group(2).strip()
        fws = [f.strip() for f in fw_key.split(',')]
        if framework in fws:
            result.append(content)
        last_end = m.end()
    result.append(text[last_end:])
    return ''.join(result)


# ──────────────────────────────────────────────────────────
# Slide generation from source .md
# ──────────────────────────────────────────────────────────

def generate_slides_qmd(src_path, framework):
    """Generate a Reveal.js .qmd slide deck from a d2l source .md file.

    Returns the .qmd content string, or None if no slide markers found.
    """
    text = Path(src_path).read_text(encoding='utf-8')

    # Filter prose tabs
    text = filter_prose_tabs(text, framework)

    # Parse into code/markdown blocks
    lines = text.split('\n')
    blocks = []  # list of ('md', text) or ('code', text, tab)
    md_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        m = re.match(r'^```(.*)$', line)
        if m and not line.startswith('````'):
            if md_lines:
                blocks.append(('md', '\n'.join(md_lines)))
                md_lines = []
            info = m.group(1).strip()
            code_lines = []
            i += 1
            while i < len(lines) and not re.match(r'^```\s*$', lines[i]):
                code_lines.append(lines[i])
                i += 1
            i += 1

            if info == 'toc' or info.startswith('eval_rst'):
                continue
            if is_python_block(info):
                if is_boilerplate(code_lines):
                    continue
                tab, cleaned = extract_tab(code_lines)
                # Filter by framework
                if tab is not None and tab != 'all':
                    tabs = [t.strip() for t in tab.split(',')]
                    if framework not in tabs:
                        continue
                # Strip #@save markers and %%tab lines that leaked through
                cleaned = [re.sub(r'\s*#@save\b', '', l) for l in cleaned]
                cleaned = [l for l in cleaned if not re.match(r'^%%tab\s+', l)]
                code_str = '\n'.join(cleaned)
                # Flatten tab.selected() branches
                code_str = flatten_tab_branches(code_str, framework)
                blocks.append(('code', code_str, 'python'))
            else:
                blocks.append(('code', '\n'.join(code_lines), info or ''))
        else:
            md_lines.append(line)
            i += 1

    if md_lines:
        blocks.append(('md', '\n'.join(md_lines)))

    # Extract slide content from markdown blocks
    slides = []  # list of slide parts: ('text', str) or ('code', str, lang)
    has_slide_markers = False
    title = None

    for block in blocks:
        if block[0] == 'md':
            md_text = block[1]

            # Always grab the title
            h = extract_heading(md_text)
            if h and title is None:
                title = h

            # Extract slide-marked text from this markdown block.
            # All marked text in one block is joined into a single slide entry.
            # If ANY marker uses [, it starts a new slide.
            marked = extract_slide_text(md_text)
            if marked:
                has_slide_markers = True
                texts = []
                new_slide = False
                for text, ns in marked:
                    if ns:
                        new_slide = True
                    texts.append(text)

                combined = ' '.join(texts)
                combined = combined.rstrip(',. \n:')
                if combined:
                    if new_slide:
                        slides.append(('break',))
                    slides.append(('text', combined))

        elif block[0] == 'code':
            code = block[1]
            lang = block[2]
            if code.strip():
                slides.append(('code', code, lang))

    if not has_slide_markers:
        return None

    # Build the .qmd slide deck
    out = []
    out.append('---')
    out.append('format:')
    out.append('  revealjs:')
    out.append('    theme: simple')
    out.append('    slide-number: true')
    out.append('    chalkboard: true')
    out.append('    scrollable: true')
    out.append('    code-line-numbers: false')
    out.append('execute:')
    out.append('  echo: true')
    out.append('  eval: false')
    out.append('---')
    out.append('')

    if title:
        # Apply directive translation to the title
        clean_title = translate_directives(title)
        out.append(clean_title)
        out.append('')

    in_slide = False
    for part in slides:
        if part[0] == 'break':
            out.append('')
            out.append('---')
            out.append('')
            in_slide = True
        elif part[0] == 'text':
            text = translate_directives(part[1])
            out.append(text)
            out.append('')
        elif part[0] == 'code':
            code = part[1]
            lang = part[2]
            if lang == 'python':
                out.append(f'```{{python}}')
            else:
                out.append(f'```{lang}')
            out.append(code)
            out.append('```')
            out.append('')

    return '\n'.join(out)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate per-framework Reveal.js slides')
    parser.add_argument('source', type=Path, help='Source d2l-en directory')
    parser.add_argument('output', type=Path, help='Output directory')
    parser.add_argument('--frameworks', nargs='*',
                        default=['pytorch', 'tensorflow', 'jax', 'mxnet'],
                        help='Frameworks to generate')
    parser.add_argument('--render', action='store_true',
                        help='Also render slides to HTML')
    args = parser.parse_args()

    src = args.source
    files = list(CHAPTER_NUMBERING.keys())

    for fw in args.frameworks:
        fw_dir = args.output / fw
        print(f'\n=== {FRAMEWORK_DISPLAY.get(fw, fw)} slides ===')

        generated = 0
        rendered = 0
        for rel in files:
            src_file = src / rel
            if not src_file.exists():
                continue

            content = generate_slides_qmd(src_file, fw)
            if content is None:
                continue  # no slide markers in this file

            dst_file = fw_dir / rel.replace('.md', '.qmd')
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_text(content, encoding='utf-8')
            generated += 1

            if args.render:
                try:
                    result = subprocess.run(
                        ['quarto', 'render', str(dst_file), '--to', 'revealjs'],
                        capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        rendered += 1
                    else:
                        print(f'    WARN: {dst_file.name}: {result.stderr.strip()[:80]}')
                except subprocess.TimeoutExpired:
                    print(f'    WARN: {dst_file.name}: timeout')

        print(f'  Generated {generated} slide decks')
        if args.render:
            print(f'  Rendered {rendered} to HTML')

    print(f'\nDone. Slides in {args.output}/')


import argparse

if __name__ == '__main__':
    main()
