"""The book's own text at a pinned commit, as plain prose.

Used by the Language Models authoring scripts that turn book sources into
fixed inputs (tools/lm/make_port_fixtures.py,
tools/oneshot/make_lm_heldout_texts.py). Files are read with `git show
<commit>:<path>`, so a fixture depends on the pinned commit only, not on the
working tree.

`prose(md)` keeps the running text of a section source and removes: the slide
section (from `<!-- slides -->` on), HTML comments (author briefs and
placeholders), code fences with their contents, image includes (also captions
that span several lines), and directive lines (a line holding only one directive:
`:label:`, `:eqlabel:`, `:begin_tab:`, fenced divs `:::`) and the forum
link line `[Discussions](...)`. Display math and
inline markup (`:numref:` and the like) stay; runs of blank lines collapse to
one.
"""
import re
import subprocess

# The book's licences (LICENSE, LICENSE-SAMPLECODE).
PROSE_LICENCE = 'CC BY-SA 4.0 (the book\'s text licence, LICENSE)'
CODE_LICENCE = 'MIT-0 (the book\'s sample-code licence, LICENSE-SAMPLECODE)'


def git_show(root, commit, path):
    """The bytes of path at commit, decoded as UTF-8."""
    return subprocess.run(['git', '-C', str(root), 'show', f'{commit}:{path}'],
                          check=True, capture_output=True).stdout.decode()


def git_commit(root, rev):
    """The full SHA and the committer date (ISO) of rev."""
    out = subprocess.run(['git', '-C', str(root), 'log', '-1',
                          '--format=%H %cI', rev], check=True,
                         capture_output=True, text=True).stdout.split()
    return out[0], out[1]


def _drop_images(text):
    """Remove `![caption](target)` includes; captions may hold one level of
    balanced brackets and span lines; a `{...}` attribute block after the
    target goes too."""
    out, i = [], 0
    while True:
        j = text.find('![', i)
        if j < 0:
            out.append(text[i:])
            return ''.join(out)
        depth, k = 0, j + 1
        while k < len(text):
            if text[k] == '[':
                depth += 1
            elif text[k] == ']':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if k + 1 < len(text) and text[k + 1] == '(':
            end = text.index(')', k + 1) + 1
            if end < len(text) and text[end] == '{':
                end = text.index('}', end) + 1
            out.append(text[i:j])
            i = end
        else:  # not an image include: keep the text as is
            out.append(text[i:k + 1])
            i = k + 1


# A line that is one directive and nothing else (`:label:`x``, `:end_tab:`,
# `::: {.callout}`); prose lines that start with a role (`:cite:`x`, ...`) stay.
_DIRECTIVE = re.compile(r'^\s*(:::.*|:[a-z_]+:(`[^`]*`)?\s*)$')


def prose(md):
    """Running text of a section source (see the module docstring)."""
    md = md.split('<!-- slides -->')[0]
    md = re.sub(r'<!--.*?-->', '', md, flags=re.S)
    md = re.sub(r'^```.*?^```[ \t]*$', '', md, flags=re.S | re.M)
    md = _drop_images(md)
    lines = [ln.rstrip() for ln in md.splitlines()
             if not _DIRECTIVE.match(ln) and not ln.startswith('[Discussions](')]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() + '\n'


def paragraphs(text):
    """Prose paragraphs (blank-line separated), skipping headings, display
    math, lists and tables."""
    for block in text.split('\n\n'):
        b = block.strip()
        if b and not b.startswith(('#', '$$', '-', '*', '|', '1.', '>')):
            yield ' '.join(ln.strip() for ln in b.splitlines())


def display_math(md):
    """The bodies of the `$$ ... $$` display equations of a source file
    (code fences excluded), stripped."""
    md = re.sub(r'^```.*?^```[ \t]*$', '', md, flags=re.S | re.M)
    return [m.strip() for m in re.findall(r'\$\$(.*?)\$\$', md, flags=re.S)]
