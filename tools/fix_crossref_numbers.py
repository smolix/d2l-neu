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
            # Quarto renders a file-backed part page but does not allocate it a
            # Pandoc chapter position. Only its child chapters enter this map.
            if 'chapters' in item:
                for ch in item['chapters']:
                    flat.append(ch)

    ch_map = {}        # pandoc_ch → logical prefix string (for numbered files)
    section_files = set()  # pandoc chapter numbers that are section-level (not chapter-level)
    unnumbered = set()  # pandoc chapter numbers for frontmatter (to strip)
    for i, qmd in enumerate(flat):
        pandoc_ch = i + 1
        qmd_as_md = qmd.replace('.qmd', '.md')
        logical = CHAPTER_NUMBERING.get(qmd_as_md)
        if logical is not None:
            logical_prefix = '.'.join(str(n) for n in logical)
            if str(pandoc_ch) != logical_prefix:
                ch_map[pandoc_ch] = logical_prefix
            if len(logical) >= 2:
                section_files.add(pandoc_ch)
        else:
            # Frontmatter or references — these should have no chapter number
            unnumbered.add(pandoc_ch)

    return ch_map, unnumbered, section_files, flat


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

    # Fix inline chapter cross-refs in quarto-xref links:
    #   <span>54&nbsp; Title</span> → <span>8.1&nbsp; Title</span>
    content = re.sub(
        r'(<span>)(\d+)(&nbsp; )',
        replace_number, content)

    # Fix figure/table/listing references and captions:
    #   Figure&nbsp;54.1  or  Figure&nbsp;<span>54.1
    for label in ['Figure', 'Table', 'Listing']:
        content = re.sub(
            rf'({label}&nbsp;(?:<span>)?)(\d+)(\.\d+)',
            replace_number, content)

    # Fix <title> tag: <title>54&nbsp; Title</title>
    content = re.sub(
        r'(<title>)(\d+)(&nbsp; )',
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


def fix_chapter_to_section(content, ch_map, section_files):
    """Replace 'Chapter X.Y' with 'Section X.Y' for section-level files.

    Quarto renders all cross-refs as 'Chapter N' since each file is a chapter.
    Section-level files (e.g., 3.1, 8.6) should use 'Section' not 'Chapter'.
    """
    count = 0

    # Build set of logical prefixes that are section-level
    section_prefixes = set()
    for pandoc_ch, logical in ch_map.items():
        if pandoc_ch in section_files:
            section_prefixes.add(logical)

    def replace_chapter_with_section(m):
        nonlocal count
        num = m.group(1)
        after = m.group(2)
        # Check if this number matches a section-level file
        # The number could be "8.1" or "8.1.2" — check if the base matches
        base = num.split('.')[0] + '.' + num.split('.')[1] if '.' in num else num
        if base in section_prefixes or num in section_prefixes:
            count += 1
            return f'Section {num}{after}'
        return m.group(0)

    content = re.sub(
        r'Chapter (\d+\.\d+(?:\.\d+)*)(</span>)',
        replace_chapter_with_section, content)

    return content, count


def fix_equation_crossrefs(content, ch_map):
    """Fix equation cross-reference link text.

    Quarto renders equation refs as 'Equation&nbsp;N.M' where N is the
    file-position chapter number. Fix to use the logical chapter number.
    Also handles 'Equation&nbsp;(N.M)' and 'Equation&nbsp;<span>N.M</span>'.
    """
    count = 0

    def replace_eq_number(m):
        nonlocal count
        before = m.group(1)
        pandoc = int(m.group(2))
        after = m.group(3)
        logical = ch_map.get(pandoc)
        if logical is not None:
            count += 1
            return f'{before}{logical}{after}'
        return m.group(0)

    # Equation&nbsp;N.M  or  Equation&nbsp;<span>N.M
    content = re.sub(
        r'(Equation&nbsp;(?:<span>)?)(\d+)(\.\d+)',
        replace_eq_number, content)

    # Equation&nbsp;(N.M)
    content = re.sub(
        r'(Equation&nbsp;\()(\d+)(\.\d+)',
        replace_eq_number, content)

    return content, count


def strip_frontmatter_figures(content, unnumbered):
    """Strip figure numbers from frontmatter pages.

    Figures in unnumbered chapters (Preface, etc.) shouldn't have chapter-based
    numbers like 'Figure 2.1'. Replace with just 'Figure' or strip the number.
    """
    count = 0
    for pandoc_ch in unnumbered:
        # Figure&nbsp;N.M → Figure
        old = f'Figure&nbsp;{pandoc_ch}.'
        while old in content:
            # Find the full number (e.g., "Figure&nbsp;2.1")
            m = re.search(rf'Figure&nbsp;{pandoc_ch}\.\d+', content)
            if m:
                content = content.replace(m.group(), 'Figure', 1)
                count += 1
            else:
                break

        # Also in <span> form: Figure&nbsp;<span>N.M</span> → Figure
        old_span = f'Figure&nbsp;<span>{pandoc_ch}.'
        while old_span in content:
            m = re.search(rf'Figure&nbsp;<span>{pandoc_ch}\.\d+</span>', content)
            if m:
                content = content.replace(m.group(), 'Figure', 1)
                count += 1
            else:
                break

    return content, count


def strip_frontmatter_numbers(content, unnumbered):
    """Remove chapter numbers from frontmatter sidebar entries.

    Frontmatter files (index, Preface, Installation, Notation, Introduction,
    References)
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


def fix_frontmatter_crossrefs(content):
    """Use page titles, not raw Pandoc chapter numbers, for frontmatter xrefs."""
    count = 0
    titles = {
        'chapter_preface/index.html': 'Preface',
        'chapter_installation/index.html': 'Installation',
        'chapter_notation/index.html': 'Notation',
        'chapter_introduction/index.html': 'Introduction',
    }
    for target, title in titles.items():
        pattern = re.compile(
            rf'(<a href="[^"]*{re.escape(target)}" class="quarto-xref">'
            rf'<span>).*?(</span></a>)')

        def repl(m):
            nonlocal count
            replacement = f'{m.group(1)}{title}{m.group(2)}'
            if replacement != m.group(0):
                count += 1
            return replacement

        content = pattern.sub(repl, content)
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


def fix_search_json(search_path, ch_map, unnumbered, section_files):
    """Fix chapter numbers in search.json.

    Quarto's search index contains plain-text references (Figure 54.1, etc.)
    and HTML crumbs with chapter-number spans — both use raw Pandoc numbers.
    """
    import json

    with open(search_path) as f:
        data = json.load(f)

    section_prefixes = set()
    for pandoc_ch, logical in ch_map.items():
        if pandoc_ch in section_files:
            section_prefixes.add(logical)

    count = 0

    def replace_text_number(m):
        nonlocal count
        label = m.group(1)
        sep = m.group(2)
        pandoc = int(m.group(3))
        rest = m.group(4)
        logical = ch_map.get(pandoc)
        if logical is not None:
            count += 1
            return f'{label}{sep}{logical}{rest}'
        return m.group(0)

    def replace_crumb_number(m):
        nonlocal count
        before = m.group(1)
        pandoc = int(m.group(2))
        after = m.group(3)
        logical = ch_map.get(pandoc)
        if logical is not None:
            count += 1
            return f'{before}{logical}{after}'
        return m.group(0)

    def strip_unnumbered_crumb(text):
        nonlocal count
        for pandoc_ch in unnumbered:
            old = f"<span class='chapter-number'>{pandoc_ch}</span>\xa0 "
            if old in text:
                n = text.count(old)
                text = text.replace(old, '')
                count += n
        return text

    def fix_chapter_label(text):
        nonlocal count
        def repl(m):
            nonlocal count
            num = m.group(1)
            base = num.split('.')[0] + '.' + num.split('.')[1] if '.' in num else num
            if base in section_prefixes or num in section_prefixes:
                count += 1
                return f'Section {num}'
            return m.group(0)
        return re.sub(r'Chapter[ \xa0](\d+\.\d+(?:\.\d+)*)', repl, text)

    def fix_leading_number(text):
        # A search entry's own title/section carries the chapter number as a
        # bare leading prefix ("42\xa0 Numerics", "42.2 Dtype Rules") — Quarto's
        # flat file-position number. Remap that leading number to the logical
        # one (→ "5.5\xa0 Numerics", "5.5.2 …"), mirroring the HTML
        # fix_section_numbers, or strip it for frontmatter pages. This is the
        # field the earlier reference/crumb passes never touched.
        nonlocal count
        m = re.match(r'(\d+)((?:\.\d+)*)([\xa0 ]+)', text)
        if not m:
            return text
        pandoc = int(m.group(1))
        if pandoc in unnumbered:
            count += 1
            return text[m.end():]
        logical = ch_map.get(pandoc)
        if logical is not None:
            count += 1
            return f'{logical}{m.group(2)}{m.group(3)}{text[m.end():]}'
        return text

    for entry in data:
        for field in ['text', 'title', 'section']:
            val = entry.get(field, '')
            if not val:
                continue
            new_val = re.sub(
                r'(Figure|Table|Section|Equation|Listing)([ \xa0])(\d+)(\.\d+(?:\.\d+)*)',
                replace_text_number, val)
            new_val = fix_chapter_label(new_val)
            if field in ('title', 'section'):
                new_val = fix_leading_number(new_val)
            if new_val != val:
                entry[field] = new_val

        crumbs = entry.get('crumbs', [])
        if crumbs:
            new_crumbs = []
            for c in crumbs:
                c = strip_unnumbered_crumb(c)
                c = re.sub(
                    r"(chapter-number['\"]?>)(\d+)(<)",
                    replace_crumb_number, c)
                new_crumbs.append(c)
            entry['crumbs'] = new_crumbs

    with open(search_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    return count


def fix_breadcrumbs(output_dir):
    """Rewrite the 2nd breadcrumb crumb from the current *section* to its parent
    *chapter*: "Basics > 1.1 Data Manipulation" -> "Basics > 1 Preliminaries".

    A section page sits in the same directory as its chapter's index.html
    (chapter_preliminaries/ndarray.html -> chapter_preliminaries/index.html), so
    we harvest each chapter index's own crumb-2 (its "<num> <title>" anchor) and
    stamp it onto the section pages in that directory, linking to index.html.
    The part crumb (crumb 1) already links to the part page and is left as is.
    """
    CRUMB = re.compile(r'(<ol class="breadcrumb">)(.*?)(</ol>)', re.S)
    ITEM = re.compile(r'<li class="breadcrumb-item">.*?</li>', re.S)
    AINNER = re.compile(r'<a\b[^>]*>(.*?)</a>', re.S)
    NUM = re.compile(r'chapter-number">([\d.]+)<')

    # Pass 1: harvest each chapter index's crumb-2 (the "<num> <title>" anchor).
    chapter_inner = {}
    for p in output_dir.rglob('index.html'):
        m = CRUMB.search(p.read_text())
        if not m:
            continue
        items = ITEM.findall(m.group(2))
        if len(items) < 2:
            continue
        a = AINNER.search(items[1])
        if a and 'chapter-number' in a.group(1):
            chapter_inner[str(p.parent.relative_to(output_dir))] = a.group(1)

    # Pass 2: stamp it onto the section (non-index) pages in that directory.
    count = 0
    for p in output_dir.rglob('*.html'):
        if p.name == 'index.html':
            continue
        inner = chapter_inner.get(str(p.parent.relative_to(output_dir)))
        if not inner:
            continue
        html = p.read_text()

        # Quarto emits the breadcrumb TWICE per page: once in the mobile
        # secondary-nav (<nav class="quarto-secondary-nav">) and once in the
        # desktop title-block header (<nav class="... d-none d-lg-block">). We
        # must rewrite crumb-2 in BOTH — a single re.search() only ever hit the
        # first (mobile) one, leaving the desktop breadcrumb showing Quarto's raw
        # self-linked section title ("5.6 Saving, loading…") with the wrong link.
        def _rewrite(m, inner=inner):
            items = ITEM.findall(m.group(2))
            if len(items) < 2:
                return m.group(0)
            # Only rewrite genuine sections: their crumb-2 number is dotted
            # ("1.1"); a chapter's is not ("1"). Guards directories holding
            # several chapters, and leaves the chapter index's own crumb alone.
            cur = AINNER.search(items[1])
            nm = cur and NUM.search(cur.group(1))
            if not (nm and '.' in nm.group(1)):
                return m.group(0)
            new_crumb2 = ('<li class="breadcrumb-item"><a href="index.html">'
                          + inner + '</a></li>')
            return m.group(1) + items[0] + new_crumb2 + m.group(3)

        html2 = CRUMB.sub(_rewrite, html)
        if html2 != html:
            p.write_text(html2)
            count += 1
    return count


def main():
    book_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    output_dir = book_dir / '_book'

    if not output_dir.exists():
        print("Error: _book/ not found. Run `quarto render` first.")
        sys.exit(1)

    ch_map, unnumbered, section_files, flat = build_chapter_map(book_dir)
    print(f"Chapter mapping: {len(ch_map)} entries, "
          f"{len(section_files)} section-level, "
          f"{len(unnumbered)} frontmatter")

    totals = {'numbers': 0, 'chap2sec': 0, 'eqrefs': 0,
              'frontrefs': 0, 'citations': 0, 'strip': 0, 'figstrip': 0}
    files_modified = 0

    for html_path in sorted(output_dir.rglob('*.html')):
        with open(html_path) as f:
            content = f.read()

        # Order matters: strip frontmatter BEFORE renumbering,
        # because renumbering maps e.g. pandoc 5→1, and pandoc 1 is in the
        # unnumbered set (index.qmd). Stripping after would hit the wrong files.
        c, n = strip_frontmatter_numbers(content, unnumbered); content = c; totals['strip'] += n
        c, n = strip_frontmatter_figures(content, unnumbered); content = c; totals['figstrip'] += n
        c, n = fix_frontmatter_crossrefs(content); content = c; totals['frontrefs'] += n
        c, n = fix_section_numbers(content, ch_map);      content = c; totals['numbers'] += n
        c, n = fix_chapter_to_section(content, ch_map, section_files); content = c; totals['chap2sec'] += n
        c, n = fix_equation_crossrefs(content, ch_map);   content = c; totals['eqrefs'] += n
        c, n = fix_narrative_citations(content);           content = c; totals['citations'] += n

        total = sum(totals.values()) - files_modified  # crude check if anything changed
        if content != open(html_path).read():
            with open(html_path, 'w') as f:
                f.write(content)
            files_modified += 1

    # Fix search.json (plain-text references + crumb HTML)
    search_json = output_dir / 'search.json'
    search_fixes = 0
    if search_json.exists():
        search_fixes = fix_search_json(search_json, ch_map, unnumbered, section_files)

    breadcrumb_fixes = fix_breadcrumbs(output_dir)

    print(f'\nFixed: {totals["numbers"]} numbers, '
          f'{totals["chap2sec"]} Chapter→Section, '
          f'{totals["eqrefs"]} equation refs, '
          f'{totals["frontrefs"]} frontmatter refs, '
          f'{totals["citations"]} citations, '
          f'stripped {totals["strip"]} sidebar + {totals["figstrip"]} frontmatter figs '
          f'in {files_modified} files, '
          f'{search_fixes} search.json entries, '
          f'{breadcrumb_fixes} breadcrumbs')


if __name__ == '__main__':
    main()
