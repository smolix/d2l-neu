#!/usr/bin/env python3
"""Convert d2l markdown files to Quarto .qmd format.

Usage:
    python d2l_preprocess.py <source_dir> <output_dir> [--primary pytorch]

Converts d2l-en markdown files to Quarto .qmd format with:
- Directive translation (labels, refs, citations, equations)
- Multi-framework tab display (primary framework executes, others display-only)
- Code block conversion to Quarto executable cells
- Prose tab sections (:begin_tab:/:end_tab:) to Quarto panel-tabsets
"""

import re
import argparse
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────

FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'mxnet']
FRAMEWORK_DISPLAY = {
    'pytorch': 'PyTorch',
    'mxnet': 'MXNet',
    'tensorflow': 'TensorFlow',
    'jax': 'JAX',
}

# ──────────────────────────────────────────────────────────
# Block types
# ──────────────────────────────────────────────────────────

class MarkdownBlock:
    def __init__(self, lines):
        self.lines = lines

class CodeBlock:
    def __init__(self, info, lines, tab=None, cell_id=None):
        self.info = info       # info string after opening ```
        self.lines = lines     # code content lines
        self.tab = tab         # None, 'all', 'pytorch', etc.
        self.cell_id = cell_id # stable #<id> from info string, or None

class CodeTabSet:
    """Grouped consecutive framework-specific code blocks."""
    def __init__(self):
        self.tabs = {}         # framework -> list of code lines
        self.ids = {}          # framework -> cell_id (or None)

class TocBlock:
    """Table of contents block - handled by _quarto.yml, skipped in output."""
    def __init__(self, lines):
        self.lines = lines

# ──────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────

def extract_tab(lines):
    """Extract tab marker from first line(s) of code.

    Returns (tab_string, cleaned_lines). Untagged Python cells default
    to 'all' (was None in the legacy convention) — every Python cell
    has a tab value after this call. Non-Python blocks bypass this
    function and keep tab=None.
    """
    if not lines:
        return 'all', lines

    # Skip leading blank lines
    start = 0
    while start < len(lines) and lines[start].strip() == '':
        start += 1
    if start >= len(lines):
        return 'all', lines

    first = lines[start]

    # %%tab style (IPython magic)
    m = re.match(r'^%%tab\s+([\w,\s]+)$', first)
    if m:
        return m.group(1).strip(), lines[start + 1:]

    # #@tab style (comment)
    m = re.match(r'^#@tab\s+([\w,\s]+)$', first)
    if m:
        return m.group(1).strip(), lines[start + 1:]

    return 'all', lines


def is_boilerplate(lines):
    """Check if code block is d2lbook tab setup boilerplate."""
    text = '\n'.join(lines)
    return ('%load_ext d2lbook.tab' in text
            or 'tab.interact_select' in text)


def is_python_block(info):
    """Check if a code fence info string indicates Python code."""
    return ('.python' in info or info == 'python'
            or info.startswith('{.python'))


# Match `#<id>` inside a fence info string (e.g. `{.python .input #foo}`).
# Anchored on whitespace or `{` so it doesn't confuse `#@tab` (which
# appears inside the fence body, never in the info string).
_CELL_ID_RE = re.compile(r'(?:^|[\s{])#([a-z][a-z0-9-]*)(?=\s|\}|$)')


def extract_cell_id(info):
    """Return the `#<id>` from a fence info string, or None."""
    m = _CELL_ID_RE.search(info)
    return m.group(1) if m else None


def clean_save_markers(lines):
    """Remove #@save markers from code lines."""
    return [re.sub(r'\s*#@save\b', '', line) for line in lines]


_SLIDES_BLOCK_RE = re.compile(
    r'(?ms)^<!--\s*slides\s*-->\s*\n.*\Z')


def strip_slide_divs(text):
    """Strip the trailing `<!-- slides -->` section from a source .md.

    Slide divs (`::: {.slide}` etc.) live below this marker and are
    consumed by `gen_slides.py`. The HTML book / PDF / per-fw notebook
    renders should not see them.
    """
    return _SLIDES_BLOCK_RE.sub('', text)


def parse_blocks(text):
    """Parse d2l markdown text into a list of blocks."""
    src_lines = text.split('\n')
    blocks = []
    md_lines = []
    i = 0

    while i < len(src_lines):
        line = src_lines[i]

        # Detect opening code fence
        m = re.match(r'^```(.*)$', line)
        if m and not line.startswith('````'):
            # Flush accumulated markdown
            if md_lines:
                blocks.append(MarkdownBlock(list(md_lines)))
                md_lines = []

            info = m.group(1).strip()
            code_lines = []
            i += 1

            # Collect until closing fence
            while i < len(src_lines):
                if re.match(r'^```\s*$', src_lines[i]):
                    i += 1
                    break
                code_lines.append(src_lines[i])
                i += 1

            # Classify the block
            if info == 'toc':
                blocks.append(TocBlock(code_lines))
            elif is_python_block(info):
                if is_boilerplate(code_lines):
                    # Drop the d2lbook tab boilerplate cell
                    continue
                tab, cleaned = extract_tab(code_lines)
                cleaned = clean_save_markers(cleaned)
                cell_id = extract_cell_id(info)
                blocks.append(CodeBlock(info, cleaned, tab, cell_id))
            else:
                # Non-Python code (bash, etc.) - keep as-is
                blocks.append(CodeBlock(info, code_lines, None))
        else:
            md_lines.append(line)
            i += 1

    if md_lines:
        blocks.append(MarkdownBlock(md_lines))

    return blocks


def group_code_tabs(blocks, primary):
    """Group consecutive framework-specific code blocks into CodeTabSets.

    Handles single-framework blocks (%%tab pytorch), multi-framework blocks
    (%%tab mxnet, pytorch), and skips blank lines between them.

    ALL framework-specific blocks end up in a CodeTabSet (even if only one
    framework has content), so that the tab visibility JS can hide them
    when a different framework is selected.
    """
    result = []
    i = 0

    def is_fw_specific(block):
        """Is this a code block tagged with specific framework(s)?"""
        if not isinstance(block, CodeBlock):
            return False
        if block.tab is None or block.tab == 'all':
            return False
        tabs = [t.strip() for t in block.tab.split(',')]
        return any(t in FRAMEWORKS for t in tabs)

    def is_blank_md(block):
        """Is this a MarkdownBlock containing only whitespace?"""
        return (isinstance(block, MarkdownBlock)
                and all(line.strip() == '' for line in block.lines))

    while i < len(blocks):
        block = blocks[i]

        if is_fw_specific(block):
            # Accumulate consecutive framework-specific blocks into a tabset,
            # skipping blank markdown lines between them.
            # Multi-framework blocks (e.g. "mxnet, pytorch") contribute their
            # code to each listed framework's tab.
            tabset = CodeTabSet()
            while i < len(blocks):
                if is_fw_specific(blocks[i]):
                    cb = blocks[i]
                    fws = [t.strip() for t in cb.tab.split(',')]
                    # If any listed framework already has content in this
                    # tabset, start a new one (separate code cell).
                    if any(fw in tabset.tabs for fw in fws if fw in FRAMEWORKS):
                        result.append(tabset)
                        tabset = CodeTabSet()
                    for fw in fws:
                        if fw in FRAMEWORKS:
                            tabset.tabs[fw] = cb.lines
                            tabset.ids[fw] = cb.cell_id
                    i += 1
                elif (is_blank_md(blocks[i])
                      and i + 1 < len(blocks)
                      and is_fw_specific(blocks[i + 1])):
                    # Skip blank lines between tab blocks
                    i += 1
                else:
                    break

            # Always keep as a tabset (even with 1 tab) so the
            # visibility JS can hide it when another framework is selected
            result.append(tabset)
        else:
            result.append(block)
            i += 1

    return result

# ──────────────────────────────────────────────────────────
# Directive translation
# ──────────────────────────────────────────────────────────

def convert_label_id(label):
    """Convert d2l label to Quarto identifier.

    sec_linear_regression -> sec-linear-regression
    fig_wake_word         -> fig-wake-word
    tab_intro_decade      -> tbl-intro-decade
    eq_price-area         -> eq-price-area
    chap_introduction     -> sec-chap-introduction
    subsec_linear_model   -> sec-linear-model
    eq_sgd-xt+1-xstar     -> eq-sgd-xt-plus-1-xstar
    """
    prefix_map = [
        ('fig_',    'fig-'),
        ('img_',    'fig-'),     # d2l uses img_ for some figures
        ('tab_',    'tbl-'),
        ('table_',  'tbl-'),     # d2l uses table_ for some tables
        ('eq_',     'eq-'),
        ('eqref_',  'eq-'),      # some d2l eqlabels have eqref_ prefix
        ('sec_',    'sec-'),
        ('chap_',   'sec-chap-'),     # chapter index labels (distinct from sec_)
        ('subsec_', 'sec-'),
    ]
    for old, new in prefix_map:
        if label.startswith(old):
            rest = label[len(old):]
            return _sanitize_id(new + rest.replace('_', '-'))
    # No recognized prefix — use explicit mapping for known bare labels
    bare_label = label.replace('_', '-')
    return _sanitize_id(BARE_LABEL_MAP.get(bare_label, bare_label))


def _sanitize_id(ident):
    """Replace characters pandoc-crossref rejects in identifiers."""
    return ident.replace('+', '-plus-')


# Known bare labels (no standard prefix in original d2l source)
# mapped to their correct Quarto type prefix.
BARE_LABEL_MAP = {
    'asha': 'fig-asha',
    'field-visual': 'fig-field-visual',
    'distributed-scheduling': 'fig-distributed-scheduling',
    'synchronous-sh': 'fig-synchronous-sh',
    'oo-design-data': 'sec-oo-design-data',
    'oo-design-training': 'sec-oo-design-training',
    'oo-design-utilities': 'sec-oo-design-utilities',
    'subsec-connection-to-mat-transposition': 'sec-connection-to-mat-transposition',
}


def translate_directives(text):
    """Translate all d2l directives in markdown text to Quarto equivalents."""

    # ── Emphasis-wrapped display equations ──
    # (**$$equation$$**) → $$equation$$
    text = re.sub(r'\(\*\*(\$\$.*?\$\$)\*\*\)', r'\1', text)

    # ── Heading + label ──
    # # Heading\n:label:`sec_foo`  →  # Heading {#sec-foo}
    text = re.sub(
        r'^(#{1,6}\s+.+?)\s*\n:label:`([^`]+)`',
        lambda m: f'{m.group(1)} {{#{convert_label_id(m.group(2))}}}',
        text, flags=re.MULTILINE)

    # ── Figure + width + label ──
    # ![caption](path)\n:width:`Npx`\n:label:`fig_foo`
    #   → ![caption](path){#fig-foo width="Npx"}
    text = re.sub(
        r'^(!\[[^\]]*\]\([^)]+\))\s*\n:width:`([^`]+)`\s*\n:label:`([^`]+)`',
        lambda m: (f'{m.group(1)}'
                   f'{{#{convert_label_id(m.group(3))} width="{m.group(2)}"}}'),
        text, flags=re.MULTILINE)

    # ── Figure + label (no width) ──
    # ![caption](path)\n:label:`fig_foo`  →  ![caption](path){#fig-foo}
    text = re.sub(
        r'^(!\[[^\]]*\]\([^)]+\))\s*\n:label:`([^`]+)`',
        lambda m: f'{m.group(1)}{{#{convert_label_id(m.group(2))}}}',
        text, flags=re.MULTILINE)

    # ── Standalone :width: (figure width without label, just strip it) ──
    text = re.sub(r'^:width:`[^`]+`\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^:height:`[^`]+`\s*$', '', text, flags=re.MULTILINE)

    # ── Display equation + label (same line) ──
    # $$...$$ :eqlabel:`eq_foo`  →  $$...$$ {#eq-foo}
    # eqlabel ALWAYS produces eq- prefix regardless of the label's own prefix
    def eqlabel_to_eq(label):
        """Force eq- prefix for equation labels."""
        converted = convert_label_id(label)
        # Strip any non-eq prefix that convert_label_id might have added
        for pfx in ('fig-', 'tbl-', 'sec-'):
            if converted.startswith(pfx):
                converted = 'eq-' + converted[len(pfx):]
                break
        if not converted.startswith('eq-'):
            converted = 'eq-' + converted
        return converted

    text = re.sub(
        r'(\$\$[^\n]*\$\$)\s*:eqlabel:`([^`]+)`',
        lambda m: f'{m.group(1)} {{#{eqlabel_to_eq(m.group(2))}}}',
        text)

    # ── Display equation + label (next line) ──
    # $$..$$\n:eqlabel:`eq_foo`  →  $$...$$ {#eq-foo}
    text = re.sub(
        r'(\$\$[^\n]*\$\$)\s*\n:eqlabel:`([^`]+)`',
        lambda m: f'{m.group(1)} {{#{eqlabel_to_eq(m.group(2))}}}',
        text, flags=re.MULTILINE)

    # ── Multi-line display equation labels ──
    # $$\n...\n$$\n:eqlabel:`eq_foo`
    text = re.sub(
        r'^(\$\$)\s*\n:eqlabel:`([^`]+)`',
        lambda m: f'{m.group(1)} {{#{eqlabel_to_eq(m.group(2))}}}',
        text, flags=re.MULTILINE)

    # ── Table caption + label ──
    # :Caption Text\n:label:`tab_foo`\n\n|...|
    # → <!-- tbl-caption:Caption Text {#tbl-foo} -->\n\n|...|
    # Post-processing will move this to after the table.
    def replace_table_header(m):
        caption = m.group(1).strip()
        label = m.group(2)
        qlabel = convert_label_id(label)
        return f'<!-- tbl-caption:{caption} {{#{qlabel}}} -->'

    text = re.sub(
        r'^:(?!label:|width:|height:|begin_tab:|end_tab:|eqlabel:|numref:|eqref:|ref:|cite|bibliography:|class:|func:|mod:)([^\n]+)\n:label:`([^`]+)`',
        replace_table_header,
        text, flags=re.MULTILINE)

    # ── Inline equation references ──
    # eqref always produces eq- prefix
    text = re.sub(
        r':eqref:`([^`]+)`',
        lambda m: f'@{eqlabel_to_eq(m.group(1))}',
        text)

    # ── Numbered cross-references (single or double backticks) ──
    text = re.sub(
        r':numref:`{1,2}([^`]+)`{1,2}',
        lambda m: f'@{convert_label_id(m.group(1))}',
        text)

    # ── Section range references ──
    # @sec-foo--@sec-bar → @sec-foo to @sec-bar
    text = re.sub(
        r'@(sec-[a-zA-Z0-9-]+)--@(sec-[a-zA-Z0-9-]+)',
        r'@\1 to @\2', text)
    # @sec-foo---word → @sec-foo --- word (em-dash separated from ref)
    text = re.sub(
        r'@(sec-[a-zA-Z0-9-]+)---(\w)',
        r'@\1 --- \2', text)

    # ── Named cross-references ──
    text = re.sub(
        r':ref:`([^`]+)`',
        lambda m: f'@{convert_label_id(m.group(1))}',
        text)

    # ── Citations (parenthetical) ──
    # :cite:`A.2020,B.2021` → [@A.2020; @B.2021]
    def replace_cite(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        inner = '; '.join(f'@{k}' for k in keys)
        return f'[{inner}]'

    text = re.sub(r':cite:`([^`]+)`', replace_cite, text)

    # ── Citations (textual) ──
    # :citet:`Key.2023` → @Key.2023
    # Ensure trailing sentence punctuation doesn't attach to the citation key
    def replace_citet(m):
        keys = [k.strip() for k in m.group(1).split(',')]
        return '; '.join(f'@{k}' for k in keys)

    text = re.sub(r':citet:`([^`]+)`', replace_citet, text)

    # Fix citation keys with trailing sentence period: @Key.2023. → @Key.2023 .
    # Pandoc would otherwise treat "Key.2023." as the citation key.
    text = re.sub(r'(@[A-Z][A-Za-z.]+\.\d{4}[a-z]?)\.(\s)', r'\1 .\2', text)
    text = re.sub(r'(@[A-Z][A-Za-z.]+\.\d{4}[a-z]?)\.\n', r'\1 .\n', text)

    # ── Slide/highlight markers ──
    # [**text**] and (**text**) → text (keep in book, strip wrappers)
    # [~~text~~] and (~~text~~) → removed (slide-only, hidden from book)
    # Uses DOTALL since markers can span multiple lines.
    text = re.sub(r'\[\*\*(.*?)\*\*\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\(\*\*(.*?)\*\*\)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\[~~.*?~~\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\(~~.*?~~\)', '', text, flags=re.DOTALL)

    # ── LaTeX math fixes ──
    # _\mathcal{X} → _{\mathcal{X}} (required by unicode-math / XeTeX)
    text = re.sub(
        r'_\\(math(?:cal|bf|sf|it|rm|bb))\{([^}]*)\}',
        r'_{\\\1{\2}}', text)

    # ── Standalone :label: (catch-all for any remaining) ──
    text = re.sub(
        r'^:label:`([^`]+)`\s*$',
        lambda m: f'{{#{convert_label_id(m.group(1))}}}',
        text, flags=re.MULTILINE)

    # ── :bibliography: directive → handled by _quarto.yml, drop ──
    text = re.sub(r'^:bibliography:`[^`]+`\s*$', '', text, flags=re.MULTILINE)

    # ── Python cross-references ──
    text = re.sub(r':class:`([^`]+)`', r'`\1`', text)
    text = re.sub(r':func:`([^`]+)`', r'`\1`', text)
    text = re.sub(r':mod:`([^`]+)`', r'`\1`', text)

    return text


def convert_prose_tabs(text, primary):
    """Convert :begin_tab:/:end_tab: sections to Quarto panel-tabsets.

    IMPORTANT: Must be called on raw text BEFORE block parsing, because
    tab blocks can span code fences (e.g. installation instructions with
    embedded bash blocks). The regex uses DOTALL to match across code fences.
    """
    tab_pattern = re.compile(
        r':begin_tab:`([^`]+)`\s*\n(.*?)\n?:end_tab:',
        re.DOTALL)

    matches = list(tab_pattern.finditer(text))
    if not matches:
        return text

    # Group consecutive tab blocks (separated by only whitespace)
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

    # Rebuild text with panel-tabsets
    parts = []
    last_end = 0

    for group in groups:
        group_start = group[0].start()
        parts.append(text[last_end:group_start])

        # Collect tabs in this group.
        # Keys can be multi-framework: "mxnet, pytorch" → applies to both.
        tab_contents = {}
        for match in group:
            fw_key = match.group(1)
            content = match.group(2).strip()
            fws = [f.strip() for f in fw_key.split(',')]
            for fw in fws:
                if fw not in tab_contents:
                    tab_contents[fw] = content

        # Emit panel-tabset with primary framework first
        parts.append('::: {.panel-tabset group="framework"}\n')
        order = [primary] + [f for f in FRAMEWORKS if f != primary]
        for fw in order:
            if fw in tab_contents:
                display = FRAMEWORK_DISPLAY.get(fw, fw)
                parts.append(f'\n## {display}\n\n{tab_contents[fw]}\n')
        parts.append('\n:::\n')

        last_end = group[-1].end()

    parts.append(text[last_end:])
    return ''.join(parts)

# ──────────────────────────────────────────────────────────
# Table caption post-processing
# ──────────────────────────────────────────────────────────

def postprocess_table_captions(text):
    """Move table captions from before the table to after (Quarto style).

    Converts:
        <!-- tbl-caption:Caption {#tbl-foo} -->
        <blank lines>
        |...|...|
        |---|---|
        |...|...|

    To:
        |...|...|
        |---|---|
        |...|...|

        : Caption {#tbl-foo}
    """
    pattern = re.compile(
        r'<!-- tbl-caption:(.*?) -->\s*\n'
        r'((?:\s*\n)*)'          # optional blank lines
        r'((?:\|[^\n]+\n)+)',    # the table rows
        re.MULTILINE)

    def replace(m):
        caption_attr = m.group(1).strip()
        table = m.group(3).rstrip('\n')
        return f'{table}\n\n: {caption_attr}\n'

    text = pattern.sub(replace, text)

    # Fix bare {#tbl-*} on its own line (from standalone :label: conversion).
    # Convert to Quarto table caption format: `: Caption {#tbl-*}`
    # Use the PRECEDING :Caption line (d2l puts caption before the table).
    # Search backwards for d2l-style `:Caption.` line before each table.
    lines = text.split('\n')
    # First pass: find all bare {#tbl-*} lines and the caption that precedes their table
    tbl_labels = {}  # line_index → label
    for i, l in enumerate(lines):
        m = re.match(r'^\{#(tbl-[a-zA-Z0-9-]+)\}$', l.strip())
        if m:
            tbl_labels[i] = m.group(1)

    # For each table label, find the d2l-style :Caption line before the table
    for label_line, label in sorted(tbl_labels.items(), reverse=True):
        # Search backwards for the start of the table (first | line)
        table_start = label_line
        for j in range(label_line - 1, -1, -1):
            if lines[j].startswith('|'):
                table_start = j
            elif lines[j].strip() == '':
                continue
            else:
                break
        # Search backwards from table_start for :Caption line
        caption = ''
        caption_line = None
        for j in range(table_start - 1, max(table_start - 5, -1), -1):
            if j < 0:
                break
            cl = lines[j].strip()
            if cl.startswith(':') and not cl.startswith(':label') and not cl.startswith(':::') and len(cl) > 2:
                caption = cl[1:].strip().rstrip('.')
                caption_line = j
                break
            elif cl == '':
                continue
            else:
                break

        if not caption:
            caption = label.replace('tbl-', '').replace('-', ' ').title()

        # Replace the {#tbl-*} line with `: Caption {#tbl-*}`
        lines[label_line] = f'\n: {caption} {{#{label}}}\n'
        # Remove the d2l :Caption line if we found one
        if caption_line is not None:
            lines[caption_line] = ''

    text = '\n'.join(lines)

    return text

# ──────────────────────────────────────────────────────────
# Output generation
# ──────────────────────────────────────────────────────────

def emit_qmd(blocks, primary='pytorch'):
    """Generate .qmd content from processed blocks."""
    parts = []

    for block in blocks:
        if isinstance(block, MarkdownBlock):
            text = '\n'.join(block.lines)
            # Prose tabs already converted in convert_file before parsing
            text = translate_directives(text)
            parts.append(text)

        elif isinstance(block, CodeBlock):
            code = '\n'.join(block.lines)
            id_prefix = f'#| label: {block.cell_id}\n' if block.cell_id else ''

            if not is_python_block(block.info) and block.tab is None:
                # Non-Python block (bash, etc.) - keep as-is, no label
                lang = block.info or ''
                parts.append(f'\n```{lang}\n{code}\n```\n')

            elif block.tab is None or block.tab == 'all':
                # Shared or un-tabbed Python → executable
                parts.append(f'\n```{{python}}\n{id_prefix}{code}\n```\n')

            else:
                # Framework-specific block not in a tabset (shouldn't happen
                # after group_code_tabs, but handle gracefully)
                if primary in (block.tab or ''):
                    parts.append(f'\n```{{python}}\n{id_prefix}{code}\n```\n')
                else:
                    # Display-only: no label (Quarto would render `#| label:`
                    # as a literal comment in non-executed code blocks).
                    parts.append(f'\n```python\n{code}\n```\n')

        elif isinstance(block, CodeTabSet):
            parts.append('\n::: {.panel-tabset group="framework"}\n')
            order = [primary] + [f for f in FRAMEWORKS if f != primary]

            for fw in order:
                if fw in block.tabs:
                    display = FRAMEWORK_DISPLAY.get(fw, fw)
                    code = '\n'.join(block.tabs[fw])
                    cid = block.ids.get(fw)

                    if fw == primary:
                        # Primary framework is executable; Quarto uses
                        # `#| label:` as the cell tag.
                        id_prefix = f'#| label: {cid}\n' if cid else ''
                        parts.append(
                            f'\n## {display}\n\n'
                            f'```{{python}}\n{id_prefix}{code}\n```\n')
                    else:
                        # Non-primary tabs are display-only (Quarto doesn't
                        # execute them), so `#| label:` would render as a
                        # literal Python comment. Instead, emit a hidden HTML
                        # comment before the fence so inject_outputs.py can
                        # still pair this tab to its framework's notebook
                        # cell by ID and inject its outputs into the right
                        # tab.
                        id_marker = f'<!-- cell-id: {cid} -->\n' if cid else ''
                        parts.append(
                            f'\n## {display}\n\n'
                            f'{id_marker}'
                            f'```python\n{code}\n```\n')

            parts.append('\n:::\n')

        elif isinstance(block, TocBlock):
            # Handled by _quarto.yml - skip
            pass

    output = '\n'.join(parts)

    # Clean up excessive blank lines (more than 2 consecutive)
    output = re.sub(r'\n{4,}', '\n\n\n', output)

    # Post-process table captions
    output = postprocess_table_captions(output)

    return output

# ──────────────────────────────────────────────────────────
# File conversion
# ──────────────────────────────────────────────────────────

def convert_file(src_path, primary='pytorch', chapter_number=None,
                 pandoc_chapter=None):
    """Convert a single d2l .md file to Quarto .qmd string.

    Args:
        chapter_number: Logical number, e.g. [2,1] for section 2.1
        pandoc_chapter: Pandoc's auto-assigned chapter number (file position).
            Used to compute number-offset so figure/equation numbering matches
            the logical chapter.
    """
    text = Path(src_path).read_text(encoding='utf-8')

    # Step 0: Strip the trailing `<!-- slides -->` block (and slide divs
    # below it). The book renderer should not see slide-only content.
    text = strip_slide_divs(text)

    # Step 1: Convert prose tabs on raw text BEFORE block parsing
    text = convert_prose_tabs(text, primary)

    # Step 2: Parse into blocks (code vs markdown)
    blocks = parse_blocks(text)

    # Step 3: Group consecutive framework-specific code blocks into tabsets
    blocks = group_code_tabs(blocks, primary)

    # Step 4: Emit .qmd output (directive translation happens per markdown block)
    output = emit_qmd(blocks, primary)

    # Step 5: Auto-number all display equations that don't have labels
    rel_path = str(src_path).split('/')[-1].replace('.md', '')
    output = number_all_equations(output, rel_path)

    # Step 6: Add YAML front matter
    # Frontmatter files: suppress Pandoc auto-numbering.
    # Numbered files: no front matter needed — Pandoc auto-numbers headings
    # and the post-render script (fix_crossref_numbers.py) corrects the numbers.
    if chapter_number is None:
        output = '---\nnumber-sections: false\n---\n' + output

    return output


def number_all_equations(text, file_slug):
    """Add {#eq-*} labels to all display equations that don't have one.

    Quarto only numbers equations with explicit labels. d2l.ai numbers all
    display equations. This function auto-assigns labels like
    {#eq-linear-regression-auto-1}, {#eq-linear-regression-auto-2}, etc.
    """
    lines = text.split('\n')
    result = []
    eq_counter = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Single-line display equation: $$...$$ (opens and closes on one line)
        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
            if '{#eq-' in line:
                result.append(line)
            else:
                eq_counter += 1
                label = f'{{#eq-{file_slug}-auto-{eq_counter}}}'
                result.append(f'{line} {label}')
            i += 1
            continue

        # Multi-line display equation: line starts with $$ (possibly with content after)
        # The closing is a line that ends with $$ (possibly with content before)
        if stripped.startswith('$$') and not stripped.endswith('$$'):
            eq_lines = [line]
            i += 1
            while i < len(lines):
                l = lines[i]
                eq_lines.append(l)
                # Closing: line ends with $$ (possibly preceded by content)
                if l.strip().endswith('$$'):
                    i += 1
                    break
                i += 1

            # Check if closing line or next line has a label
            has_label = any('{#eq-' in el for el in eq_lines)
            if not has_label and i < len(lines) and '{#eq-' in lines[i]:
                has_label = True

            if not has_label:
                eq_counter += 1
                label = f'{{#eq-{file_slug}-auto-{eq_counter}}}'
                # Append label AFTER the closing $$ line
                eq_lines[-1] = f'{eq_lines[-1]} {label}'

            result.extend(eq_lines)
            continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def number_headings(text, chapter_number):
    """Add hierarchical section numbers to markdown headings.

    chapter_number = [2, 1] means top-level heading is "2.1",
    sub-headings are "2.1.1", "2.1.2", etc.

    Skips headings inside ::: panel-tabset blocks (those are tab labels,
    not section headings).
    """
    prefix = '.'.join(str(n) for n in chapter_number)
    sub_counters = {}
    lines = text.split('\n')
    result = []
    in_tabset = 0  # nesting depth of ::: blocks

    for line in lines:
        # Track ::: div nesting
        stripped = line.strip()
        if stripped.startswith('::: '):
            in_tabset += 1
            result.append(line)
            continue
        elif stripped == ':::':
            in_tabset = max(0, in_tabset - 1)
            result.append(line)
            continue

        # Only number headings outside panel-tabsets
        m = re.match(r'^(#{1,6})\s+(.+)$', line)
        if m and in_tabset == 0:
            hashes = m.group(1)
            title = m.group(2)
            level = len(hashes)

            # Pandoc-style unnumbered headings (e.g. "## Resources {.unnumbered}")
            # get no number and do not advance the section counters.
            if re.search(r'\{[^}]*\.unnumbered[^}]*\}\s*$|\{\s*-\s*\}\s*$', title):
                result.append(line)
                continue

            if level == 1:
                # Top-level heading: use chapter_number directly
                sub_counters = {}
                result.append(f'{hashes} {prefix} {title}')
            else:
                # Sub-heading: chapter_number.counter...
                sub_level = level - 1
                sub_counters[sub_level] = sub_counters.get(sub_level, 0) + 1
                for k in range(sub_level + 1, 7):
                    sub_counters.pop(k, None)

                parts = [prefix]
                for i in range(1, sub_level + 1):
                    parts.append(str(sub_counters.get(i, 0)))
                num_str = '.'.join(parts)
                result.append(f'{hashes} {num_str} {title}')
        else:
            result.append(line)

    return '\n'.join(result)


def convert_index(src_path):
    """Convert the main book index.md.

    The TOC ```toc``` blocks are dropped (handled by _quarto.yml).
    All other content (markdown, raw HTML, divs) passes through so the
    landing page can be richly designed in index.md.
    """
    text = Path(src_path).read_text(encoding='utf-8')

    # Strip ```toc ... ``` fenced blocks (TOC is driven by _quarto.yml).
    text = re.sub(r'(?ms)^```toc\s*\n.*?^```\s*\n?', '', text)

    # If the source already starts with YAML front matter, keep it as-is;
    # otherwise inject one with the title + suppressed numbering.
    if text.lstrip().startswith('---'):
        return text

    front = (
        '---\n'
        'title: "Dive into Deep Learning"\n'
        'pagetitle: "Dive into Deep Learning"\n'
        'number-sections: false\n'
        'toc: false\n'
        'page-layout: full\n'
        '---\n\n'
    )
    return front + text.lstrip()

# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

# Chapter numbering: maps source file to [chapter, section, ...] or None
# None = unnumbered (frontmatter/references). [1] = chapter 1. [2,3] = section 2.3.
# Generated from d2l-en/index.md and each chapter's toc block.
CHAPTER_NUMBERING = {
    'chapter_preface/index.md': None,
    'chapter_installation/index.md': None,
    'chapter_notation/index.md': None,
    'chapter_introduction/index.md': [1],
    'chapter_preliminaries/index.md': [2],
    'chapter_preliminaries/ndarray.md': [2, 1],
    'chapter_preliminaries/pandas.md': [2, 2],
    'chapter_preliminaries/linear-algebra.md': [2, 3],
    'chapter_preliminaries/calculus.md': [2, 4],
    'chapter_preliminaries/autograd.md': [2, 5],
    'chapter_preliminaries/probability.md': [2, 6],
    'chapter_preliminaries/lookup-api.md': [2, 7],
    'chapter_linear-regression/index.md': [3],
    'chapter_linear-regression/linear-regression.md': [3, 1],
    'chapter_linear-regression/oo-design.md': [3, 2],
    'chapter_linear-regression/synthetic-regression-data.md': [3, 3],
    'chapter_linear-regression/linear-regression-scratch.md': [3, 4],
    'chapter_linear-regression/linear-regression-concise.md': [3, 5],
    'chapter_linear-regression/generalization.md': [3, 6],
    'chapter_linear-regression/weight-decay.md': [3, 7],
    'chapter_linear-classification/index.md': [4],
    'chapter_linear-classification/softmax-regression.md': [4, 1],
    'chapter_linear-classification/image-classification-dataset.md': [4, 2],
    'chapter_linear-classification/classification.md': [4, 3],
    'chapter_linear-classification/softmax-regression-scratch.md': [4, 4],
    'chapter_linear-classification/softmax-regression-concise.md': [4, 5],
    'chapter_linear-classification/generalization-classification.md': [4, 6],
    'chapter_linear-classification/environment-and-distribution-shift.md': [4, 7],
    'chapter_multilayer-perceptrons/index.md': [5],
    'chapter_multilayer-perceptrons/mlp.md': [5, 1],
    'chapter_multilayer-perceptrons/mlp-implementation.md': [5, 2],
    'chapter_multilayer-perceptrons/backprop.md': [5, 3],
    'chapter_multilayer-perceptrons/numerical-stability-and-init.md': [5, 4],
    'chapter_multilayer-perceptrons/generalization-deep.md': [5, 5],
    'chapter_multilayer-perceptrons/dropout.md': [5, 6],
    'chapter_multilayer-perceptrons/kaggle-house-price.md': [5, 7],
    'chapter_builders-guide/index.md': [6],
    'chapter_builders-guide/model-construction.md': [6, 1],
    'chapter_builders-guide/parameters-state-memory.md': [6, 2],
    'chapter_builders-guide/init.md': [6, 3],
    'chapter_builders-guide/custom-layers.md': [6, 4],
    'chapter_builders-guide/numerics.md': [6, 5],
    'chapter_builders-guide/saving-loading.md': [6, 6],
    'chapter_builders-guide/gpus-devices-memory.md': [6, 7],
    'chapter_builders-guide/reproducibility-inspection.md': [6, 8],
    'chapter_convolutional-neural-networks/index.md': [7],
    'chapter_convolutional-neural-networks/why-conv.md': [7, 1],
    'chapter_convolutional-neural-networks/conv-layer.md': [7, 2],
    'chapter_convolutional-neural-networks/padding-and-strides.md': [7, 3],
    'chapter_convolutional-neural-networks/channels.md': [7, 4],
    'chapter_convolutional-neural-networks/pooling.md': [7, 5],
    'chapter_convolutional-neural-networks/lenet.md': [7, 6],
    'chapter_convolutional-modern/index.md': [8],
    'chapter_convolutional-modern/alexnet.md': [8, 1],
    'chapter_convolutional-modern/vgg.md': [8, 2],
    'chapter_convolutional-modern/nin.md': [8, 3],
    'chapter_convolutional-modern/googlenet.md': [8, 4],
    'chapter_convolutional-modern/batch-norm.md': [8, 5],
    'chapter_convolutional-modern/resnet.md': [8, 6],
    'chapter_convolutional-modern/densenet.md': [8, 7],
    'chapter_convolutional-modern/cnn-design.md': [8, 8],
    'chapter_recurrent-neural-networks/index.md': [9],
    'chapter_recurrent-neural-networks/sequence.md': [9, 1],
    'chapter_recurrent-neural-networks/text-sequence.md': [9, 2],
    'chapter_recurrent-neural-networks/language-model.md': [9, 3],
    'chapter_recurrent-neural-networks/rnn.md': [9, 4],
    'chapter_recurrent-neural-networks/rnn-scratch.md': [9, 5],
    'chapter_recurrent-neural-networks/rnn-concise.md': [9, 6],
    'chapter_recurrent-neural-networks/bptt.md': [9, 7],
    'chapter_recurrent-modern/index.md': [10],
    'chapter_recurrent-modern/lstm.md': [10, 1],
    'chapter_recurrent-modern/gru.md': [10, 2],
    'chapter_recurrent-modern/deep-rnn.md': [10, 3],
    'chapter_recurrent-modern/bi-rnn.md': [10, 4],
    'chapter_recurrent-modern/machine-translation-and-dataset.md': [10, 5],
    'chapter_recurrent-modern/encoder-decoder.md': [10, 6],
    'chapter_recurrent-modern/seq2seq.md': [10, 7],
    'chapter_recurrent-modern/beam-search.md': [10, 8],
    'chapter_attention-mechanisms-and-transformers/index.md': [11],
    'chapter_attention-mechanisms-and-transformers/queries-keys-values.md': [11, 1],
    'chapter_attention-mechanisms-and-transformers/attention-pooling.md': [11, 2],
    'chapter_attention-mechanisms-and-transformers/attention-scoring-functions.md': [11, 3],
    'chapter_attention-mechanisms-and-transformers/bahdanau-attention.md': [11, 4],
    'chapter_attention-mechanisms-and-transformers/multihead-attention.md': [11, 5],
    'chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.md': [11, 6],
    'chapter_attention-mechanisms-and-transformers/transformer.md': [11, 7],
    'chapter_attention-mechanisms-and-transformers/vision-transformer.md': [11, 8],
    'chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.md': [11, 9],
    'chapter_optimization/index.md': [12],
    'chapter_optimization/optimization-intro.md': [12, 1],
    'chapter_optimization/convexity.md': [12, 2],
    'chapter_optimization/gd.md': [12, 3],
    'chapter_optimization/sgd.md': [12, 4],
    'chapter_optimization/minibatch-sgd.md': [12, 5],
    'chapter_optimization/momentum.md': [12, 6],
    'chapter_optimization/adagrad.md': [12, 7],
    'chapter_optimization/rmsprop.md': [12, 8],
    'chapter_optimization/adadelta.md': [12, 9],
    'chapter_optimization/adam.md': [12, 10],
    'chapter_optimization/lr-scheduler.md': [12, 11],
    'chapter_computational-performance/index.md': [13],
    'chapter_computational-performance/hybridize.md': [13, 1],
    'chapter_computational-performance/async-computation.md': [13, 2],
    'chapter_computational-performance/auto-parallelism.md': [13, 3],
    'chapter_computational-performance/hardware.md': [13, 4],
    'chapter_computational-performance/multiple-gpus.md': [13, 5],
    'chapter_computational-performance/multiple-gpus-concise.md': [13, 6],
    'chapter_computational-performance/parameterserver.md': [13, 7],
    'chapter_computer-vision/index.md': [14],
    'chapter_computer-vision/image-augmentation.md': [14, 1],
    'chapter_computer-vision/fine-tuning.md': [14, 2],
    'chapter_computer-vision/bounding-box.md': [14, 3],
    'chapter_computer-vision/anchor.md': [14, 4],
    'chapter_computer-vision/multiscale-object-detection.md': [14, 5],
    'chapter_computer-vision/object-detection-dataset.md': [14, 6],
    'chapter_computer-vision/ssd.md': [14, 7],
    'chapter_computer-vision/rcnn.md': [14, 8],
    'chapter_computer-vision/semantic-segmentation-and-dataset.md': [14, 9],
    'chapter_computer-vision/transposed-conv.md': [14, 10],
    'chapter_computer-vision/fcn.md': [14, 11],
    'chapter_computer-vision/neural-style.md': [14, 12],
    'chapter_computer-vision/kaggle-cifar10.md': [14, 13],
    'chapter_computer-vision/kaggle-dog.md': [14, 14],
    'chapter_natural-language-processing-pretraining/index.md': [15],
    'chapter_natural-language-processing-pretraining/word2vec.md': [15, 1],
    'chapter_natural-language-processing-pretraining/approx-training.md': [15, 2],
    'chapter_natural-language-processing-pretraining/word-embedding-dataset.md': [15, 3],
    'chapter_natural-language-processing-pretraining/word2vec-pretraining.md': [15, 4],
    'chapter_natural-language-processing-pretraining/glove.md': [15, 5],
    'chapter_natural-language-processing-pretraining/subword-embedding.md': [15, 6],
    'chapter_natural-language-processing-pretraining/similarity-analogy.md': [15, 7],
    'chapter_natural-language-processing-pretraining/bert.md': [15, 8],
    'chapter_natural-language-processing-pretraining/bert-dataset.md': [15, 9],
    'chapter_natural-language-processing-pretraining/bert-pretraining.md': [15, 10],
    'chapter_natural-language-processing-applications/index.md': [16],
    'chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md': [16, 1],
    'chapter_natural-language-processing-applications/sentiment-analysis-rnn.md': [16, 2],
    'chapter_natural-language-processing-applications/sentiment-analysis-cnn.md': [16, 3],
    'chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md': [16, 4],
    'chapter_natural-language-processing-applications/natural-language-inference-attention.md': [16, 5],
    'chapter_natural-language-processing-applications/finetuning-bert.md': [16, 6],
    'chapter_natural-language-processing-applications/natural-language-inference-bert.md': [16, 7],
    'chapter_reinforcement-learning/index.md': [17],
    'chapter_reinforcement-learning/mdp.md': [17, 1],
    'chapter_reinforcement-learning/value-iter.md': [17, 2],
    'chapter_reinforcement-learning/qlearning.md': [17, 3],
    'chapter_gaussian-processes/index.md': [18],
    'chapter_gaussian-processes/gp-intro.md': [18, 1],
    'chapter_gaussian-processes/gp-priors.md': [18, 2],
    'chapter_gaussian-processes/gp-inference.md': [18, 3],
    'chapter_hyperparameter-optimization/index.md': [19],
    'chapter_hyperparameter-optimization/hyperopt-intro.md': [19, 1],
    'chapter_hyperparameter-optimization/hyperopt-api.md': [19, 2],
    'chapter_hyperparameter-optimization/rs-async.md': [19, 3],
    'chapter_hyperparameter-optimization/sh-intro.md': [19, 4],
    'chapter_hyperparameter-optimization/sh-async.md': [19, 5],
    'chapter_generative-adversarial-networks/index.md': [20],
    'chapter_generative-adversarial-networks/gan.md': [20, 1],
    'chapter_generative-adversarial-networks/dcgan.md': [20, 2],
    'chapter_recommender-systems/index.md': [21],
    'chapter_recommender-systems/recsys-intro.md': [21, 1],
    'chapter_recommender-systems/movielens.md': [21, 2],
    'chapter_recommender-systems/mf.md': [21, 3],
    'chapter_recommender-systems/autorec.md': [21, 4],
    'chapter_recommender-systems/ranking.md': [21, 5],
    'chapter_recommender-systems/neumf.md': [21, 6],
    'chapter_recommender-systems/seqrec.md': [21, 7],
    'chapter_recommender-systems/ctr.md': [21, 8],
    'chapter_recommender-systems/fm.md': [21, 9],
    'chapter_recommender-systems/deepfm.md': [21, 10],
    'chapter_mdl-linear-algebra/index.md': [22],
    'chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md': [22, 1],
    'chapter_mdl-linear-algebra/mdl-eigendecomposition.md': [22, 2],
    'chapter_mdl-linear-algebra/mdl-svd-low-rank.md': [22, 3],
    'chapter_mdl-calculus/index.md': [23],
    'chapter_mdl-calculus/mdl-single-variable-calculus.md': [23, 1],
    'chapter_mdl-calculus/mdl-multivariable-calculus.md': [23, 2],
    'chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md': [23, 3],
    'chapter_mdl-calculus/mdl-integral-calculus.md': [23, 4],
    'chapter_mdl-optimization/index.md': [24],
    'chapter_mdl-optimization/mdl-gradient-based-optimization.md': [24, 1],
    'chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md': [24, 2],
    'chapter_mdl-optimization/mdl-convexity.md': [24, 3],
    'chapter_mdl-optimization/mdl-constrained-optimization-duality.md': [24, 4],
    'chapter_mdl-optimization/mdl-numerical-stability-conditioning.md': [24, 5],
    'chapter_mdl-probability-statistics/index.md': [25],
    'chapter_mdl-probability-statistics/mdl-random-variables.md': [25, 1],
    'chapter_mdl-probability-statistics/mdl-distributions.md': [25, 2],
    'chapter_mdl-probability-statistics/mdl-maximum-likelihood.md': [25, 3],
    'chapter_mdl-probability-statistics/mdl-statistics.md': [25, 4],
    'chapter_mdl-probability-statistics/mdl-concentration-generalization.md': [25, 5],
    'chapter_mdl-probability-statistics/mdl-naive-bayes.md': [25, 6],
    'chapter_mdl-information-theory/index.md': [26],
    'chapter_mdl-information-theory/mdl-information-theory.md': [26, 1],
    'chapter_mdl-information-theory/mdl-divergences-distances.md': [26, 2],
    'chapter_mdl-information-theory/mdl-mutual-information.md': [26, 3],
    'chapter_mdl-dynamics/index.md': [27],
    'chapter_mdl-dynamics/mdl-odes-solvers.md': [27, 1],
    'chapter_mdl-dynamics/mdl-sdes.md': [27, 2],
    'chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md': [27, 3],
    'chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md': [27, 4],
    'chapter_appendix-tools-for-deep-learning/index.md': [28],
    'chapter_appendix-tools-for-deep-learning/jupyter.md': [28, 1],
    'chapter_appendix-tools-for-deep-learning/sagemaker.md': [28, 2],
    'chapter_appendix-tools-for-deep-learning/aws.md': [28, 3],
    'chapter_appendix-tools-for-deep-learning/colab.md': [28, 4],
    'chapter_appendix-tools-for-deep-learning/selecting-servers-gpus.md': [28, 5],
    'chapter_appendix-tools-for-deep-learning/contributing.md': [28, 6],
    'chapter_appendix-tools-for-deep-learning/utils.md': [28, 7],
    'chapter_appendix-tools-for-deep-learning/d2l.md': [28, 8],
    'chapter_references/zreferences.md': None,
}

CHAPTER_FILES = list(CHAPTER_NUMBERING.keys())


def main():
    parser = argparse.ArgumentParser(
        description='Convert d2l-en markdown to Quarto .qmd format')
    parser.add_argument('source', type=Path,
                        help='Source d2l-en directory')
    parser.add_argument('output', type=Path,
                        help='Output directory for .qmd files')
    parser.add_argument('--primary', default='pytorch',
                        help='Primary framework (default: pytorch)')
    parser.add_argument('--files', nargs='*',
                        help='Specific files to convert (default: all chapters)')
    args = parser.parse_args()

    files = args.files or CHAPTER_FILES
    src = args.source
    dst = args.output

    def _write_if_changed(path, content):
        """Preserve mtime when content unchanged so downstream Make rules
        (HTML/PDF render) don't re-fire unnecessarily."""
        new_bytes = content.encode('utf-8')
        if path.exists() and path.read_bytes() == new_bytes:
            return False
        path.write_bytes(new_bytes)
        return True

    # Convert main index
    index_dst = dst / 'index.qmd'
    index_dst.parent.mkdir(parents=True, exist_ok=True)
    if _write_if_changed(index_dst, convert_index(src / 'index.md')):
        print(f'  index.md -> index.qmd')

    # Convert chapter files. Compute pandoc_chapter (file position in book,
    # 1-indexed) for number-offset calculation.
    n_changed = 0
    for pos, rel in enumerate(files, start=1):
        src_file = src / rel
        if not src_file.exists():
            print(f'  SKIP {rel} (not found)')
            continue

        dst_file = dst / rel.replace('.md', '.qmd')
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        ch_num = CHAPTER_NUMBERING.get(rel)
        output = convert_file(src_file, args.primary,
                              chapter_number=ch_num, pandoc_chapter=pos)
        if _write_if_changed(dst_file, output):
            n_changed += 1

    print(f'\nConverted {len(files)} files to {dst} ({n_changed} updated)')


if __name__ == '__main__':
    main()
