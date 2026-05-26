#!/usr/bin/env python3
"""Generate per-framework Reveal.js slides from d2l-en source.

The d2l source uses `::: {.slide}` fenced divs to define slides.
Each slide div contains text and `@<id>` (or `@<id>@<fw>`) placeholders
that reference code cells by their stable `#<id>` attributes.

  ::: {.slide title="Why vectorize?" layout="2col"}
  A naïve Python loop pays interpreter overhead per element:

  @vec-loop

  . . .

  A vectorized call hands the work to a C kernel:

  @vec-add
  :::

The slide builder:
  - Indexes code cells in the source by `#<id>` (per-framework variants
    differ via `#@tab`).
  - Resolves `@<id>` to the deck's framework variant (or `@<id>@<fw>`
    to a specific variant).
  - Strips `@d2l.add_to_class(...)` decorators from emitted code; slides
    do not add to the library.
  - Strips `#@save` markers and `%%tab` lines.
  - Flattens `tab.selected()` branches for the deck's framework.

The output `.qmd` has `eval: false` — outputs are injected post-build
by `tools/inject_outputs.py slides`.

Rendering is CPU-only (no GPU env vars, no kernel cleanup) and parallel
across frameworks (`make -j4 slides`) and within a framework (thread
pool of 8 quarto-render subprocesses).

Usage:
    python tools/gen_slides.py <source_dir> <output_dir> [--render]
        [--frameworks pytorch jax] [--workers 8]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d2l_preprocess import (
    FRAMEWORKS, FRAMEWORK_DISPLAY, CHAPTER_NUMBERING,
    extract_tab, is_boilerplate, is_python_block,
    extract_cell_id, translate_directives,
)
from build_lib import flatten_tab_branches


# ──────────────────────────────────────────────────────────
# Source parsing
# ──────────────────────────────────────────────────────────

class SourceCell:
    """One Python code fence in source, indexed by cell_id+framework."""
    __slots__ = ('cell_id', 'tab', 'lines')
    def __init__(self, cell_id, tab, lines):
        self.cell_id, self.tab, self.lines = cell_id, tab, lines


_FENCE_OPEN_RE = re.compile(r'^```(.*)$')
_FENCE_CLOSE_RE = re.compile(r'^```\s*$')
_SLIDE_OPEN_RE = re.compile(r'^:{3,}\s*\{\.slide(?:\s+([^}]*))?\}\s*$')
_SUBSLIDE_OPEN_RE = re.compile(r'^:{3,}\s*\{\.subslide(?:\s+([^}]*))?\}\s*$')
_DIV_OPEN_RE = re.compile(r'^:{3,}\s*\{')
_DIV_CLOSE_RE = re.compile(r'^:{3,}\s*$')
# Placeholder forms (each on its own line):
#   @cell-id              → render code + injected output
#   @cell-id@pytorch      → force a specific framework variant
#   @!cell-id             → output-only (echo: false) — useful on
#                           cover/teaser slides where the figure
#                           matters but the code doesn't
#   @!cell-id@pytorch     → output-only with framework override
_PLACEHOLDER_RE = re.compile(
    r'^@(?P<echo>!)?(?P<id>[a-z][a-z0-9-]*)(?:@(?P<fw>\w+))?\s*$')

# Match @d2l.add_to_class(...) decorator on its own line.
_ADD_TO_CLASS_RE = re.compile(r'^@d2l\.add_to_class\([^)]*\)\s*$')


def index_source_cells(text):
    """Walk source, return a dict: cell_id → list[SourceCell].

    Each list contains one entry per framework variant. A cell with
    `#@tab pytorch` produces a SourceCell with tab='pytorch' (not 'all').
    Untagged cells get tab='all' from extract_tab.

    Boilerplate cells and non-Python fences are ignored.
    """
    lines = text.split('\n')
    cells = {}
    in_fence = False
    info = ''
    body = []
    for line in lines:
        if not in_fence:
            m = _FENCE_OPEN_RE.match(line)
            if m and not line.startswith('````'):
                info = m.group(1).strip()
                body = []
                in_fence = True
        else:
            if _FENCE_CLOSE_RE.match(line):
                if is_python_block(info) and not is_boilerplate(body):
                    cid = extract_cell_id(info)
                    if cid:
                        tab, cleaned = extract_tab(list(body))
                        cells.setdefault(cid, []).append(
                            SourceCell(cid, tab, cleaned))
                in_fence = False
            else:
                body.append(line)
    return cells


def select_variant(variants, framework, forced_fw=None):
    """Pick the SourceCell for the deck's framework.

    Resolution order:
      1. If forced_fw is set: a variant whose tab includes forced_fw.
      2. A variant whose tab includes the deck's framework.
      3. A variant tagged 'all'.
    Returns None if no match.
    """
    target = forced_fw or framework
    for v in variants:
        if v.tab == 'all':
            continue
        tabs = [t.strip() for t in v.tab.split(',')]
        if target in tabs:
            return v
    if forced_fw:
        return None  # explicit @<id>@<fw> doesn't fall back to 'all'
    for v in variants:
        if v.tab == 'all':
            return v
    return None


def clean_cell_lines(lines, framework):
    """Apply slide-emit cleanup to a code cell's source lines.

    - Drop `@d2l.add_to_class(...)` decorators (slides never extend the library)
    - Strip `#@save` markers
    - Drop residual `%%tab` lines
    - Flatten `tab.selected()` branches for the deck framework
    """
    out = []
    for line in lines:
        if _ADD_TO_CLASS_RE.match(line.strip()):
            continue
        if re.match(r'^%%tab\s+', line):
            continue
        line = re.sub(r'\s*#@save\b', '', line)
        out.append(line)
    code = '\n'.join(out)
    code = flatten_tab_branches(code, framework)
    return code


# ──────────────────────────────────────────────────────────
# Slide div parsing
# ──────────────────────────────────────────────────────────

class Slide:
    __slots__ = ('attrs', 'body', 'subslides')
    def __init__(self, attrs):
        self.attrs = attrs           # parsed key=value dict
        self.body = []               # list of lines
        self.subslides = []          # list[Slide]


def _parse_attrs(attr_str):
    """Parse `title="…" layout=foo transition="bar"` style attribute string."""
    if not attr_str:
        return {}
    attrs = {}
    # Handle quoted values: key="…"
    for m in re.finditer(r'(\w[\w-]*)\s*=\s*"([^"]*)"', attr_str):
        attrs[m.group(1)] = m.group(2)
    # Handle unquoted: key=value
    for m in re.finditer(r'(\w[\w-]*)\s*=\s*([^\s"}]+)', attr_str):
        if m.group(1) not in attrs:
            attrs[m.group(1)] = m.group(2)
    return attrs


def parse_slide_blocks(text):
    """Return list[Slide]. Skips content outside slide divs.

    Handles nested `::: {.subslide}` divs by collecting them on the
    enclosing slide's `subslides` list (rendered as vertical sub-slides).
    Other nested divs (e.g. `::: {.fragment}`) pass through verbatim
    in the body.
    """
    lines = text.split('\n')
    slides = []
    i = 0
    while i < len(lines):
        m = _SLIDE_OPEN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        slide = Slide(_parse_attrs(m.group(1) or ''))
        i += 1
        depth = 1  # depth of fenced divs we entered
        while i < len(lines):
            line = lines[i]
            sub = _SUBSLIDE_OPEN_RE.match(line)
            slide_open_inner = _SLIDE_OPEN_RE.match(line)
            other_open = (not sub and not slide_open_inner
                          and _DIV_OPEN_RE.match(line))
            close = _DIV_CLOSE_RE.match(line)
            if sub and depth == 1:
                # Subslide nested directly inside this slide
                ss = Slide(_parse_attrs(sub.group(1) or ''))
                i += 1
                ss_depth = 1
                while i < len(lines):
                    inner = lines[i]
                    if _DIV_OPEN_RE.match(inner):
                        ss_depth += 1
                    elif _DIV_CLOSE_RE.match(inner):
                        ss_depth -= 1
                        if ss_depth == 0:
                            i += 1
                            break
                    ss.body.append(inner)
                    i += 1
                slide.subslides.append(ss)
                continue
            if slide_open_inner or other_open:
                depth += 1
                slide.body.append(line)
            elif close:
                depth -= 1
                if depth == 0:
                    i += 1
                    break
                slide.body.append(line)
            else:
                slide.body.append(line)
            i += 1
        slides.append(slide)
    return slides


def has_framework_code(src_path, framework):
    """Same semantics as before: skip files with no fw-applicable code."""
    text = Path(src_path).read_text(encoding='utf-8')
    tabs_found = set()
    for m in re.finditer(r'^#@tab\s+(.+)$', text, re.MULTILINE):
        for t in m.group(1).split(','):
            tabs_found.add(t.strip())
    for m in re.finditer(r'^%%tab\s+(.+)$', text, re.MULTILINE):
        for t in m.group(1).split(','):
            tabs_found.add(t.strip())
    if not tabs_found:
        return True  # no tabs anywhere — applies to all frameworks
    fw_specific = tabs_found - {'all'}
    if not fw_specific:
        return True
    return framework in fw_specific


# ──────────────────────────────────────────────────────────
# Slide qmd emission
# ──────────────────────────────────────────────────────────

def _emit_attrs(attrs):
    """Build Quarto-style attribute string for a slide heading."""
    parts = []
    if 'transition' in attrs:
        parts.append(f'data-transition="{attrs["transition"]}"')
    if 'layout' in attrs:
        parts.append(f'.slide-{attrs["layout"]}')
    for k, v in attrs.items():
        if k.startswith('data-background'):
            parts.append(f'{k}="{v}"')
    return parts


def _resolve_body(body_lines, source_cells, framework, slide_path,
                  warnings):
    """Replace `@<id>` and `@<id>@<fw>` placeholder lines with code fences.

    Other lines pass through verbatim. Returns a list of output lines.
    """
    out = []
    for line in body_lines:
        m = _PLACEHOLDER_RE.match(line)
        if not m:
            out.append(line)
            continue
        cid = m.group('id')
        forced_fw = m.group('fw')
        echo_off = bool(m.group('echo'))
        variants = source_cells.get(cid)
        if not variants:
            warnings.append(f'{slide_path}: unknown @{cid}')
            out.append(line)
            continue
        cell = select_variant(variants, framework, forced_fw)
        if cell is None:
            warnings.append(
                f'{slide_path}: @{cid}'
                + (f'@{forced_fw}' if forced_fw else '')
                + f' has no variant for {framework}')
            continue
        code = clean_cell_lines(cell.lines, framework)
        if not code.strip():
            continue
        out.append('```{python}')
        out.append(f'#| label: {cid}')
        if echo_off:
            out.append('#| echo: false')
        out.append(code)
        out.append('```')
    return out


def _emit_slide(slide, source_cells, framework, slide_path, warnings,
                is_first):
    """Emit one slide as Quarto reveal.js markdown.

    The deck has the file H1 as its title (emitted by the caller). The
    first slide's content follows the H1 directly. Subsequent slides
    are separated by `---` (no title) or by `## <title>` if the slide
    has an explicit `title=` attribute.
    """
    out = []
    title = slide.attrs.get('title')
    extras = _emit_attrs(slide.attrs)
    attr_str = (' {' + ' '.join(extras) + '}') if extras else ''
    if title:
        # Emit `## Title` separator. (Implicitly starts a new slide.)
        out.append('')
        out.append(f'## {title}{attr_str}')
        out.append('')
    elif not is_first:
        # Plain horizontal-rule slide separator
        if extras:
            out.append('')
            out.append(f'## {{{" ".join(extras)}}}')
            out.append('')
        else:
            out.append('')
            out.append('---')
            out.append('')
    out.extend(_resolve_body(slide.body, source_cells, framework,
                              slide_path, warnings))
    # Subslides (vertical sub-slides under the current slide)
    for ss in slide.subslides:
        out.append('')
        ss_title = ss.attrs.get('title', '')
        ss_extras = _emit_attrs(ss.attrs)
        ss_attr = (' {' + ' '.join(ss_extras) + '}') if ss_extras else ''
        out.append(f'### {ss_title}{ss_attr}'.rstrip())
        out.append('')
        out.extend(_resolve_body(ss.body, source_cells, framework,
                                  slide_path, warnings))
    return out


def generate_slides_qmd(src_path, framework, warnings):
    """Generate a Reveal.js .qmd slide deck from a source .md file.

    Returns the .qmd content string, or None if the file has no slide
    divs or has no framework-applicable code.
    """
    if not has_framework_code(src_path, framework):
        return None

    text = Path(src_path).read_text(encoding='utf-8')
    slides = parse_slide_blocks(text)
    if not slides:
        return None

    source_cells = index_source_cells(text)

    # File-level title from H1
    file_title = None
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('# ') and not s.startswith('## '):
            file_title = re.sub(r'\s*\{#[^}]+\}', '', s[2:]).strip()
            break

    out = []
    out.append('---')
    out.append('format:')
    out.append('  revealjs:')
    out.append('    theme: [simple, ../../../_d2l-slides.scss]')
    out.append('    width: 1280')
    out.append('    height: 720')
    out.append('    margin: 0.06')
    out.append('    slide-number: true')
    out.append('    chalkboard: true')
    out.append('    scrollable: true')
    out.append('    code-line-numbers: false')
    out.append('    include-after-body: ../../../_d2l-slides-overlay.html')
    # KaTeX renders math with bundled fonts (reliable for `\mathcal`,
    # `\mathbb`, etc.); MathJax 2.7 — Quarto's default for revealjs —
    # ships the math output via dynamic font requests that occasionally
    # render as boxes when the caligraphic / blackboard-bold fonts
    # don't load on time.
    out.append('    html-math-method: katex')
    out.append('execute:')
    out.append('  echo: true')
    out.append('  eval: false')
    out.append('---')
    out.append('')

    if file_title:
        out.append(f'# {translate_directives(file_title)}')
        out.append('')

    for i, slide in enumerate(slides):
        emitted = _emit_slide(slide, source_cells, framework,
                              str(src_path), warnings,
                              is_first=(i == 0))
        out.extend(emitted)
        out.append('')

    qmd = '\n'.join(out)
    # Cell labels must be unique per Quarto. When the same cell ID
    # appears in multiple placeholders (typically `@!id` on the cover
    # plus `@id` on a content slide), append `-fig{N}` to repeats so
    # quarto render accepts them. inject_outputs.py strips the
    # `-fig{N}` suffix when looking up the cell in the executed
    # notebook.
    qmd = _dedupe_labels(qmd)
    return qmd


_LABEL_RE = re.compile(r'^(\s*#\|\s*label:\s*)([a-z][a-z0-9-]*)(\s*)$',
                        re.MULTILINE)


def _dedupe_labels(text: str) -> str:
    """Suffix repeated `#| label: <id>` lines with `-fig{N}`."""
    seen = {}
    def sub(m):
        prefix, label, suffix = m.group(1), m.group(2), m.group(3)
        seen[label] = seen.get(label, 0) + 1
        if seen[label] == 1:
            return m.group(0)
        return f'{prefix}{label}-fig{seen[label]}{suffix}'
    return _LABEL_RE.sub(sub, text)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate per-framework Reveal.js slides')
    parser.add_argument('source', type=Path, help='Source d2l-en directory')
    parser.add_argument('output', type=Path, help='Output directory')
    parser.add_argument('--frameworks', nargs='*',
                        default=list(FRAMEWORKS),
                        help='Frameworks to generate (default: all)')
    parser.add_argument('--render', action='store_true',
                        help='Also render slides to HTML')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of CPU workers for parallel rendering')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Per-slide render timeout in seconds')
    parser.add_argument('--files', nargs='*', default=None,
                        help='Only process these source paths '
                             '(e.g. chapter_foo/bar.md)')
    args = parser.parse_args()

    src = args.source
    file_filter = set(args.files) if args.files else None
    files = list(CHAPTER_NUMBERING.keys())

    warnings = []
    for fw in args.frameworks:
        fw_dir = args.output / fw
        print(f'\n=== {FRAMEWORK_DISPLAY.get(fw, fw)} slides ===')

        generated = 0
        for rel in files:
            if file_filter and rel not in file_filter:
                continue
            src_file = src / rel
            if not src_file.exists():
                continue

            content = generate_slides_qmd(src_file, fw, warnings)
            if content is None:
                continue

            dst_file = fw_dir / rel.replace('.md', '.qmd')
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.write_text(content, encoding='utf-8')
            generated += 1

        if args.render and generated:
            # img / data symlinks (unchanged from previous behavior)
            img_link = fw_dir / 'img'
            if not img_link.exists():
                img_link.symlink_to(Path('../../img'))
            data_link = fw_dir / 'data'
            if not data_link.exists():
                data_dir = Path.cwd() / 'data'
                data_dir.mkdir(exist_ok=True)
                data_link.symlink_to(data_dir)

            # Place a minimal _quarto.yml per-framework so Quarto's
            # `.quarto/` cache (cites index, crossref index) is isolated
            # between frameworks. A shared cache at _slides/.quarto/
            # gets corrupted when one framework's render reads cached
            # state populated by a previous framework's run, producing
            # "Unexpected non-whitespace character after JSON" errors.
            fw_yml = fw_dir / '_quarto.yml'
            if not fw_yml.exists():
                fw_yml.write_text('project:\n  type: default\n',
                                   encoding='utf-8')
            # Remove any legacy shared _quarto.yml at _slides/ root and
            # its accompanying .quarto/ cache (left over from before
            # this fix).
            stale_root_yml = args.output / '_quarto.yml'
            if stale_root_yml.exists():
                stale_root_yml.unlink()
            stale_root_cache = args.output / '.quarto'
            if stale_root_cache.exists():
                shutil.rmtree(stale_root_cache)

            # Per-framework Quarto cache (xref/INDEX, idx/, cites/, …).
            # Prior builds of this tool ran 16 parallel `quarto render`
            # subprocesses that raced on INDEX without locking, producing
            # a JSON document followed by an orphan tail fragment. Once
            # corrupted, every subsequent render in the same project
            # dies with `SyntaxError: Unexpected non-whitespace
            # character after JSON …`. Wipe the cache so the new
            # single-project render starts from a clean slate.
            fw_cache = fw_dir / '.quarto'
            if fw_cache.exists():
                shutil.rmtree(fw_cache)

            # Inject executed notebook outputs into the slide qmds before
            # rendering, so the rendered HTML carries cached outputs.
            notebooks_dir = Path('_notebooks')
            if notebooks_dir.exists() and (notebooks_dir / fw).exists():
                from inject_outputs import inject_slides
                img_outputs_dir = args.output / 'img' / 'outputs'
                inject_slides(str(args.output), fw, str(notebooks_dir),
                               str(img_outputs_dir))

            qmd_files = sorted(fw_dir.rglob('*.qmd'))
            # Honor the --files filter at render time too, so a one-shot
            # rebuild of a single deck doesn't re-render every existing
            # .qmd in the framework directory.
            if file_filter:
                stems = {Path(f).with_suffix('').as_posix()
                         for f in file_filter}
                qmd_files = [
                    q for q in qmd_files
                    if q.relative_to(fw_dir).with_suffix('').as_posix()
                       in stems]

            # Quarto needs nbformat (Python) to parse code chunks even
            # with eval: false. Use any framework's venv that exists.
            base_env = {**os.environ}
            for candidate in (Path(f'.venv-{fw}'), Path('.venv-pytorch'),
                              Path('.venv-jax'), Path('.venv-mxnet'),
                              Path('.venv-tensorflow')):
                py = candidate.absolute() / 'bin' / 'python'
                if py.exists():
                    base_env['QUARTO_PYTHON'] = str(py)
                    break

            error_dir = fw_dir.parent / 'errors' / fw
            error_dir.mkdir(parents=True, exist_ok=True)

            # Render the framework as a single Quarto project. Quarto
            # renders files sequentially within one process, so its
            # writes to .quarto/xref/INDEX are not racing. Spawning N
            # parallel `quarto render <single.qmd>` subprocesses (the
            # previous design) was unsafe — the per-file workers all did
            # read-modify-write on the shared INDEX without locking, and
            # under contention a slow writer's bytes were truncated to
            # the fast writer's length, leaving a valid JSON document
            # followed by an orphan tail fragment. Subsequent renders
            # then died with `SyntaxError: Unexpected non-whitespace
            # character after JSON at position …` from
            # `crossrefIndexForOutputFile`. --workers is preserved as a
            # CLI knob for backwards compatibility but is now ignored.
            total = len(qmd_files)
            print(f'  Rendering {total} deck(s) via single-project quarto')
            rel_paths = [str(q.relative_to(fw_dir)) for q in qmd_files]

            # --no-execute skips Quarto's per-deck Python-kernel startup
            # (~3-5s × N decks). Code-cell outputs are already injected
            # into the .qmd by inject_outputs.py before this render, so
            # there's nothing to execute — kernel startup was pure
            # overhead. eval: false in each deck's frontmatter is not
            # enough; Quarto still spins up nbconvert.
            cmd = ['quarto', 'render', *rel_paths,
                   '--to', 'revealjs', '--no-execute']
            t0 = time.time()
            proc = subprocess.Popen(
                cmd, cwd=fw_dir, env=base_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)

            current_file = None
            current_buf = []
            file_errors = {}
            line_re = re.compile(r'^\s*\[\d+/\d+\]\s+(\S+\.qmd)\s*$')
            fail_re = re.compile(r'^\s*FAIL\s+\([\d.]+s\)\s+(\S+\.qmd)\s*$')

            def _flush():
                if current_file and current_buf:
                    file_errors.setdefault(current_file, []).extend(current_buf)

            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip('\n')
                print(f'  {line}', flush=True)
                m = line_re.match(line)
                if m:
                    _flush()
                    current_file = m.group(1)
                    current_buf = []
                    continue
                m = fail_re.match(line)
                if m:
                    _flush()
                    current_file = m.group(1)
                    current_buf = [line]
                    file_errors.setdefault(current_file, []).append(line)
                    continue
                if current_file:
                    current_buf.append(line)
            _flush()
            rc = proc.wait()
            elapsed = time.time() - t0

            # Anything still in errors/* without a corresponding .html
            # output is treated as a failure (Quarto sometimes emits
            # FAIL without exiting non-zero on the first try).
            failed_paths = set(file_errors.keys())
            for rel in rel_paths:
                html = fw_dir / Path(rel).with_suffix('.html')
                if not html.exists():
                    failed_paths.add(rel)

            # A non-zero exit from quarto that didn't surface as a
            # per-file FAIL means the project-level render failed
            # before any file got re-rendered (typical cause: a theme
            # SCSS compile error). The old .html files from a prior
            # build still exist on disk, so the per-file existence
            # check above would otherwise spuriously report 0 failures.
            # Treat this as "every file failed" so the build is
            # visibly broken.
            if rc != 0 and not failed_paths:
                for rel in rel_paths:
                    failed_paths.add(rel)
                    file_errors.setdefault(rel, []).append(
                        f'quarto exited rc={rc} before any per-file render '
                        f'completed (likely theme SCSS compile error)')

            for rel in sorted(failed_paths):
                log_path = error_dir / Path(rel).with_suffix('.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text('\n'.join(file_errors.get(rel, [])) or
                                    'no output written')

            results = [
                (fw_dir / rel, rel not in failed_paths,
                 '\n'.join(file_errors.get(rel, [])) if rel in failed_paths
                 else None)
                for rel in rel_paths
            ]
            failures = [(q, err) for q, ok, err in results if not ok]
            print(f'  Rendered {total - len(failures)} / {total} '
                  f'({len(failures)} failed) in {elapsed:.0f}s (rc={rc})')

            # Dedupe per-deck `<deck>_files/libs/` into a shared
            # `_slides/libs/`. Quarto writes a full copy of revealjs +
            # plugins (~115 files) next to every deck; with 261 decks
            # that balloons the deployed tree to ~30k extra files,
            # which makes incremental upload painfully slow. The libs
            # are byte-identical across decks (and across frameworks),
            # so we keep one copy and rewrite each deck's HTML to
            # point at it via a relative path.
            shared_libs = args.output / 'libs'
            for deck_html in sorted(fw_dir.rglob('*.html')):
                # Quarto's revealjs plugins ship .html assets (e.g.
                # speaker-view.html) under <deck>_files/libs/. Skip
                # anything inside a *_files/ tree — only top-level
                # deck HTMLs need rewriting.
                if any(p.endswith('_files') for p in deck_html.parts):
                    continue
                files_dir = deck_html.parent / f'{deck_html.stem}_files'
                per_deck_libs = files_dir / 'libs'
                # Merge per-deck libs into the shared tree on every
                # deck (not just when shared doesn't exist). Quarto
                # bakes a content-hash into the compiled theme CSS
                # filename (e.g. `quarto-<sha>.css`), and different
                # SCSS compiles across frameworks / across rebuilds
                # produce different hashes. The HTML rewrite below
                # only changes the *directory* prefix, so the
                # filename has to exist in shared — otherwise the
                # deck loads no theme CSS at all. Copy adds new
                # hashed siblings without clobbering existing ones.
                if per_deck_libs.exists():
                    shutil.copytree(per_deck_libs, shared_libs,
                                    dirs_exist_ok=True)
                rel_libs = os.path.relpath(
                    shared_libs, deck_html.parent) + '/'
                old_ref = f'{deck_html.stem}_files/libs/'
                text = deck_html.read_text(encoding='utf-8')
                if old_ref in text:
                    deck_html.write_text(
                        text.replace(old_ref, rel_libs),
                        encoding='utf-8')
                if per_deck_libs.exists():
                    shutil.rmtree(per_deck_libs)
                if files_dir.exists():
                    try:
                        files_dir.rmdir()
                    except OSError:
                        pass  # not empty — keep figure-revealjs/ etc.

        print(f'  Generated {generated} slide deck(s)')

    if warnings:
        print(f'\n{len(warnings)} placeholder warning(s):')
        for w in warnings[:30]:
            print(f'  {w}')
        if len(warnings) > 30:
            print(f'  ... and {len(warnings) - 30} more')

    print(f'\nDone. Slides in {args.output}/')


if __name__ == '__main__':
    main()
