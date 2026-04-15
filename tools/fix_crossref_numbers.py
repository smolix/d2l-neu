#!/usr/bin/env python3
"""Post-render script: fix section/figure/equation numbers and citation links.

Quarto numbers chapters by file position (54th file → chapter 54). This script
replaces those with logical chapter numbers (e.g., section 8.1) everywhere:
headings, sidebar, figures, tables, equations.

Also fixes narrative citations so the full author name is hyperlinked,
not just the year.

Run after `quarto render`:
    python tools/fix_crossref_numbers.py [book_dir]
"""
import re
import sys
from pathlib import Path


def build_chapter_map(book_dir):
    """Build pandoc_chapter → logical_prefix mapping.

    Returns dict like {54: '8.1', 5: '1', 10: '2.4', ...}
    """
    import yaml
    sys.path.insert(0, str(book_dir / 'tools'))
    from d2l_preprocess import CHAPTER_NUMBERING

    with open(book_dir / '_quarto.yml') as f:
        cfg = yaml.safe_load(f)

    flat = []
    for item in cfg['book']['chapters']:
        if isinstance(item, str):
            flat.append(item)
        elif isinstance(item, dict):
            if 'chapters' in item:
                for ch in item['chapters']:
                    flat.append(ch)

    ch_map = {}        # pandoc_ch → logical prefix string (for numbered files)
    unnumbered = set()  # pandoc chapter numbers for frontmatter (to strip)
    for i, qmd in enumerate(flat):
        pandoc_ch = i + 1
        qmd_as_md = qmd.replace('.qmd', '.md')
        logical = CHAPTER_NUMBERING.get(qmd_as_md)
        if logical is not None:
            logical_prefix = '.'.join(str(n) for n in logical)
            if str(pandoc_ch) != logical_prefix:
                ch_map[pandoc_ch] = logical_prefix
        else:
            # Frontmatter or references — these should have no chapter number
            unnumbered.add(pandoc_ch)

    return ch_map, unnumbered, flat


def fix_section_numbers(content, ch_map):
    """Fix ALL section/figure/table/equation numbers simultaneously."""
    count = 0

    def replace_number(m):
        """Replace a Pandoc chapter prefix with the logical one."""
        nonlocal count
        before = m.group(1)
        pandoc = int(m.group(2))
        after = m.group(3)
        logical = ch_map.get(pandoc)
        if logical is not None:
            count += 1
            return f'{before}{logical}{after}'
        return m.group(0)

    # Fix header-section-number spans (headings + in-page TOC):
    #   <span class="header-section-number">54.1</span>
    #   → <span class="header-section-number">8.1.1</span>
    content = re.sub(
        r'(header-section-number">)(\d+)((?:\.\d+)*</span>)',
        replace_number, content)

    # Fix data-number attributes on headings:
    #   data-number="54.1" → data-number="8.1.1"
    content = re.sub(
        r'(data-number=")(\d+)((?:\.\d+)*")',
        replace_number, content)

    # Fix sidebar chapter-number spans:
    #   <span class="chapter-number">54</span>
    #   → <span class="chapter-number">8.1</span>
    content = re.sub(
        r'(chapter-number">)(\d+)(</span>)',
        replace_number, content)

    # Fix section/chapter cross-references (quarto-xref links):
    #   <span>Chapter 54</span>  → <span>Chapter 8.1</span>
    #   <span>Section 54.1.2</span> → <span>Section 8.1.1.2</span>
    for label in ['Chapter', 'Section']:
        content = re.sub(
            rf'({label} )(\d+)((?:\.\d+)*</span>)',
            replace_number, content)

    # Fix figure/table/listing references and captions:
    #   Figure&nbsp;54.1  or  Figure&nbsp;<span>54.1
    for label in ['Figure', 'Table', 'Listing']:
        content = re.sub(
            rf'({label}&nbsp;(?:<span>)?)(\d+)(\.\d+)',
            replace_number, content)

    # Fix equation tags: \tag{54.1}
    content = re.sub(
        r'(\\tag\{)(\d+)(\.\d+)',
        replace_number, content)

    # Fix equation references: Equation&nbsp;(54.1)
    content = re.sub(
        r'(Equation&nbsp;\()(\d+)(\.\d+)',
        replace_number, content)

    return content, count


def strip_frontmatter_numbers(content, unnumbered):
    """Remove chapter numbers from frontmatter sidebar entries.

    Frontmatter files (index, Preface, Installation, Notation, References)
    get Pandoc chapter numbers that should not be displayed. This ONLY strips
    the sidebar <span class="chapter-number">N</span> entries for these files.
    (Heading numbers in frontmatter files don't exist because they have
    number-sections: false in their front matter.)
    """
    count = 0
    for pandoc_ch in unnumbered:
        # Sidebar/breadcrumb: <span class="chapter-number">N</span>&nbsp;
        old = f'<span class="chapter-number">{pandoc_ch}</span>&nbsp; '
        if old in content:
            n = content.count(old)
            content = content.replace(old, '')
            count += n

    return content, count


def fix_narrative_citations(content):
    """Fix narrative citations so the full text is hyperlinked.

    Before: <span class="citation" ...>Author Names (<a href="URL">Year</a>)</span>
    After:  <span class="citation" ...><a href="URL">Author Names (Year)</a></span>

    Only touches narrative citations (where author names appear before the link).
    Parenthetical citations (everything already inside <a>) are left alone.
    """
    count = 0

    def fix_cite(m):
        nonlocal count
        key = m.group(1)
        inner = m.group(2)

        # Narrative citation: "Author Names (<a href="URL" role="doc-biblioref">Year</a>)"
        narrative = re.match(
            r'([^<]+)\(<a\s+href="([^"]+)"([^>]*)>([^<]+)</a>\)',
            inner)
        if narrative:
            authors = narrative.group(1).strip()
            url = narrative.group(2)
            attrs = narrative.group(3)
            year = narrative.group(4).strip()
            count += 1
            return (f'<span class="citation" data-cites="{key}">'
                    f'<a href="{url}"{attrs}>{authors} ({year})</a></span>')
        return m.group(0)

    content = re.sub(
        r'<span class="citation" data-cites="([^"]+)">(.*?)</span>',
        fix_cite, content)

    return content, count


def main():
    book_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    output_dir = book_dir / '_book'

    if not output_dir.exists():
        print("Error: _book/ not found. Run `quarto render` first.")
        sys.exit(1)

    ch_map, unnumbered, flat = build_chapter_map(book_dir)
    print(f"Chapter mapping: {len(ch_map)} entries to fix, "
          f"{len(unnumbered)} frontmatter to strip")

    total_num_fixes = 0
    total_cite_fixes = 0
    total_strip = 0
    files_modified = 0

    for html_path in sorted(output_dir.rglob('*.html')):
        with open(html_path) as f:
            content = f.read()

        new_content, num_fixes = fix_section_numbers(content, ch_map)
        new_content, cite_fixes = fix_narrative_citations(new_content)
        new_content, strip_fixes = strip_frontmatter_numbers(
            new_content, unnumbered)

        total = num_fixes + cite_fixes + strip_fixes
        if total > 0:
            with open(html_path, 'w') as f:
                f.write(new_content)
            rel = html_path.relative_to(output_dir)
            parts = []
            if num_fixes:
                parts.append(f'{num_fixes} numbers')
            if cite_fixes:
                parts.append(f'{cite_fixes} citations')
            if strip_fixes:
                parts.append(f'{strip_fixes} stripped')
            print(f'  {rel}: {", ".join(parts)}')
            total_num_fixes += num_fixes
            total_cite_fixes += cite_fixes
            total_strip += strip_fixes
            files_modified += 1

    print(f'\nFixed {total_num_fixes} section/figure numbers, '
          f'{total_cite_fixes} narrative citations, '
          f'stripped {total_strip} frontmatter numbers '
          f'in {files_modified} files')


if __name__ == '__main__':
    main()
