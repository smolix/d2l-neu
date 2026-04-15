#!/usr/bin/env python3
"""Generate single-framework .qmd files for PDF rendering.

Unlike the HTML book (which shows all frameworks in tabs), the PDF version
shows only one framework. This script generates .qmd files with:
- Only the selected framework's code (no tabs)
- Only the selected framework's prose
- Discussion sections stripped (website-only)
- Slide markers stripped

Usage:
    python tools/gen_pdf.py <source_dir> <output_dir> [--framework pytorch]
    Then: quarto render <output_dir> --to pdf
"""

import re
import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import (
    FRAMEWORKS, FRAMEWORK_DISPLAY, CHAPTER_NUMBERING,
    parse_blocks, extract_tab, is_boilerplate, is_python_block,
    clean_save_markers, translate_directives,
    CodeBlock, MarkdownBlock, CodeTabSet, TocBlock,
)
from build_lib import flatten_tab_branches


def convert_prose_tabs_single(text, framework):
    """Keep only the target framework's prose tab content."""
    tab_pattern = re.compile(
        r':begin_tab:`([^`]+)`\s*\n(.*?)\n?:end_tab:', re.DOTALL)

    matches = list(tab_pattern.finditer(text))
    if not matches:
        return text

    groups = []
    current_group = [matches[0]]
    for i in range(1, len(matches)):
        between = text[matches[i-1].end():matches[i].start()]
        if between.strip() == '':
            current_group.append(matches[i])
        else:
            groups.append(current_group)
            current_group = [matches[i]]
    groups.append(current_group)

    parts = []
    last_end = 0
    for group in groups:
        group_start = group[0].start()
        parts.append(text[last_end:group_start])

        for match in group:
            fw_key = match.group(1)
            content = match.group(2).strip()
            fws = [f.strip() for f in fw_key.split(',')]
            if framework in fws:
                parts.append(content + '\n')
                break

        last_end = group[-1].end()

    parts.append(text[last_end:])
    return ''.join(parts)


def localize_external_images(text):
    """Replace external image URLs with local paths."""
    # GitHub user-images SVGs → local copies
    def replace_github_svg(m):
        url = m.group(0)
        fname = url.split('/')[-1]
        base = fname.replace('.svg', '')
        return f'../img/gp-{base}.svg'

    text = re.sub(
        r'https://user-images\.githubusercontent\.com/[^\s)]+\.svg',
        replace_github_svg, text)
    return text


def strip_discussions(text):
    """Remove Discussion sections and links (website-only content)."""
    # Remove ## Discussion headings and content until next heading or EOF
    text = re.sub(
        r'^##\s+Discussions?\s*\n.*?(?=^##\s|\Z)',
        '', text, flags=re.MULTILINE | re.DOTALL)
    # Remove standalone discussion links: [Discussions](https://discuss.d2l.ai/...)
    text = re.sub(
        r'^\[Discussions?\]\(https://discuss\.d2l\.ai/[^)]*\)\s*$',
        '', text, flags=re.MULTILINE)
    return text


def emit_pdf_qmd(blocks, framework):
    """Emit single-framework .qmd for PDF rendering."""
    parts = []

    for block in blocks:
        if isinstance(block, MarkdownBlock):
            text = '\n'.join(block.lines)
            text = translate_directives(text)
            text = strip_discussions(text)
            parts.append(text)

        elif isinstance(block, CodeBlock):
            code = '\n'.join(block.lines)

            if not is_python_block(block.info) and block.tab is None:
                lang = block.info or ''
                parts.append(f'\n```{lang}\n{code}\n```\n')
            elif block.tab is None or block.tab == 'all':
                # Flatten tab.selected() branches
                code = flatten_tab_branches(code, framework)
                parts.append(f'\n```{{python}}\n{code}\n```\n')
            elif framework in (block.tab or ''):
                code = flatten_tab_branches(code, framework)
                parts.append(f'\n```{{python}}\n{code}\n```\n')
            # else: different framework, skip

        elif isinstance(block, CodeTabSet):
            if framework in block.tabs:
                code = '\n'.join(block.tabs[framework])
                code = flatten_tab_branches(code, framework)
                parts.append(f'\n```{{python}}\n{code}\n```\n')
            # else: framework has no code, skip

        elif isinstance(block, TocBlock):
            pass

    output = '\n'.join(parts)
    output = re.sub(r'\n{4,}', '\n\n\n', output)

    # Post-process table captions (same as HTML preprocessor)
    from d2l_preprocess import postprocess_table_captions
    output = postprocess_table_captions(output)

    return output


def convert_file_pdf(src_path, framework, chapter_number=None):
    """Convert a d2l .md file to single-framework .qmd for PDF."""
    text = Path(src_path).read_text(encoding='utf-8')

    # Localize external images and filter prose tabs
    text = localize_external_images(text)
    text = convert_prose_tabs_single(text, framework)

    # Parse into blocks
    blocks = parse_blocks(text)

    # Group code tabs (reuse preprocessor logic)
    from d2l_preprocess import group_code_tabs
    blocks = group_code_tabs(blocks, framework)

    # Emit single-framework output
    output = emit_pdf_qmd(blocks, framework)

    # Add front matter for unnumbered chapters
    if chapter_number is None:
        output = '---\nnumber-sections: false\n---\n' + output

    return output


def main():
    parser = argparse.ArgumentParser(
        description='Generate single-framework .qmd for PDF rendering')
    parser.add_argument('source', type=Path, help='Source d2l-en directory')
    parser.add_argument('output', type=Path, help='Output directory')
    parser.add_argument('--framework', default='pytorch',
                        help='Framework (default: pytorch)')
    args = parser.parse_args()

    src = args.source
    fw = args.framework
    dst = args.output
    files = list(CHAPTER_NUMBERING.keys())

    print(f'Generating PDF sources for {FRAMEWORK_DISPLAY.get(fw, fw)}...')

    # Generate .qmd files
    converted = 0
    for rel in files:
        src_file = src / rel
        if not src_file.exists():
            continue

        dst_file = dst / rel.replace('.md', '.qmd')
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        ch_num = CHAPTER_NUMBERING.get(rel)
        output = convert_file_pdf(src_file, fw, chapter_number=ch_num)
        dst_file.write_text(output, encoding='utf-8')
        converted += 1

    # Generate index.qmd
    index_dst = dst / 'index.qmd'
    index_dst.write_text(
        '---\ntitle: "Dive into Deep Learning"\n---\n\n'
        'An interactive deep learning book with code, math, and discussions.\n',
        encoding='utf-8')

    # Generate references.qmd
    refs_dst = dst / 'references.qmd'
    refs_dst.write_text('# References {.unnumbered}\n\n::: {#refs}\n:::\n',
                        encoding='utf-8')

    # Copy assets from the project (d2l-neu), not from the source (d2l-en)
    project_dir = Path(__file__).parent.parent
    for asset in ['d2l.bib', 'img']:
        asset_src = project_dir / asset
        if not asset_src.exists():
            # Fallback to source
            asset_src = src / asset
        asset_dst = dst / asset
        if asset_dst.exists() or asset_dst.is_symlink():
            continue
        if asset_src.is_dir():
            shutil.copytree(asset_src, asset_dst,
                           ignore=shutil.ignore_patterns('*.pdf'))
        else:
            shutil.copy2(asset_src, asset_dst)

    # Copy static files
    static_src = Path(__file__).parent.parent / 'static'
    static_dst = dst / 'static'
    if not static_dst.exists():
        shutil.copytree(static_src, static_dst, dirs_exist_ok=True)

    # Copy SVG→PDF lua filter
    lua_src = Path(__file__).parent.parent / '_svg-to-pdf.lua'
    if lua_src.exists():
        shutil.copy2(lua_src, dst / '_svg-to-pdf.lua')

    # Generate _quarto.yml for PDF
    _write_quarto_yml(dst, fw, files)

    # Pre-convert SVGs to PDF
    import subprocess
    svg_script = Path(__file__).parent / 'convert_svg.sh'
    if svg_script.exists():
        subprocess.run(['bash', str(svg_script), str(dst / 'img')],
                      capture_output=True)

    print(f'Generated {converted} .qmd files in {dst}/')
    print(f'To render: cd {dst} && quarto render --to pdf')


def _write_quarto_yml(dst, framework, files):
    """Write a PDF-specific _quarto.yml."""
    fw_display = FRAMEWORK_DISPLAY.get(framework, framework)

    # Build chapters list
    chapters = ['    - index.qmd\n']
    for rel in files:
        qmd = rel.replace('.md', '.qmd')
        chapters.append(f'    - {qmd}\n')
    chapters.append('    - references.qmd\n')

    yml = f"""project:
  type: book
  output-dir: _pdf

book:
  title: "Dive into Deep Learning"
  subtitle: "{fw_display} Edition"
  author:
    - Aston Zhang
    - Zachary C. Lipton
    - Mu Li
    - Alexander J. Smola
  date: today
  chapters:
{''.join(chapters)}
bibliography: d2l.bib
link-citations: true
number-sections: true
crossref:
  chapters: true

filters:
  - _svg-to-pdf.lua

format:
  pdf:
    pdf-engine: xelatex
    documentclass: book
    papersize: a4
    fontsize: 11pt
    classoption:
      - twoside
      - openright
    mainfont: "SourceSerif4"
    mainfontoptions:
      - Path=static/fonts/
      - Extension=.ttf
      - UprightFont=*-Regular
      - BoldFont=*-Bold
      - ItalicFont=*-Italic
    sansfont: "SourceSans3"
    sansfontoptions:
      - Path=static/fonts/
      - Extension=.ttf
      - UprightFont=*-Regular
      - BoldFont=*-Bold
      - ItalicFont=*-Italic
    monofont: "Inconsolata"
    monofontoptions:
      - Path=static/fonts/
      - Extension=.ttf
      - UprightFont=*-Regular
      - BoldFont=*-Bold
      - Scale=0.9
    geometry:
      - left=1in
      - right=1in
      - top=1in
      - bottom=1in
      - twoside
    linkcolor: black
    citecolor: black
    urlcolor: black
    include-in-header: static/d2l-preamble.tex
    toc: true
    toc-depth: 2
    code-block-bg: "#f2f2f2"
    fig-pos: "htbp"
    keep-tex: true
    latex-auto-install: false

execute:
  enabled: false
  echo: true
"""
    (dst / '_quarto.yml').write_text(yml, encoding='utf-8')


if __name__ == '__main__':
    main()
