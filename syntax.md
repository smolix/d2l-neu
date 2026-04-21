# D2L Source Directive Syntax Reference

This document catalogs every special directive used in the `.md` source files
and how each is parsed by the build tools. The `.md` files in `chapter_*/` are
the source of truth — `.qmd` files are generated from them.

## Framework Selection

Three directive syntaxes select which framework(s) a code block or prose section
belongs to. All three ultimately serve the same purpose: splitting content into
per-framework views.

### `%%tab` — Cell-Level Framework Tag

Appears as the first line inside a code fence. Consumed by `extract_tab()`.

```
%%tab <framework-list>
```

**Examples:**
```
%%tab pytorch
%%tab all
%%tab pytorch, mxnet, tensorflow
%%tab mxnet, pytorch, jax
```

**Occurrences:** 1,691 total. `jax` (380), `tensorflow` (375), `pytorch` (374),
`mxnet` (343), `all` (141), multi-framework combos (78).

**Parser:** `d2l_preprocess.py:extract_tab()` — regex `^%%tab\s+([\w,\s]+)$`.
The framework list is split on `,` and each token is `.strip()`ed.

**Edge cases:** Comma spacing is inconsistent in 3 instances (e.g.
`mxnet,pytorch` without space). The parser handles this because it splits on
`,` then strips.


### `#@tab` — Inline Comment Framework Tag

Same semantics as `%%tab`, used in a different style. Appears as the first
non-blank line inside a code fence.

```
#@tab <framework-list>
```

**Examples:**
```
#@tab pytorch
#@tab all
#@tab mxnet, pytorch
```

**Occurrences:** 1,173 total. `mxnet` (432), `pytorch` (368), `all` (197),
`tensorflow` (160), combos (16).

**Parser:** `d2l_preprocess.py:extract_tab()` — regex `^#@tab\s+([\w,\s]+)$`.
Same split-and-strip logic as `%%tab`.


### `:begin_tab:` / `:end_tab:` — Prose Framework Blocks

Wraps prose (not code) in framework-specific sections. Appears in running
markdown text, outside code fences.

```
:begin_tab:`<framework-list>`
Content visible only for the listed framework(s).
:end_tab:
```

**Examples:**
```
:begin_tab:`pytorch`
This paragraph appears only in the PyTorch tab.
:end_tab:

:begin_tab:`mxnet, pytorch, tensorflow`
Shared across three frameworks.
:end_tab:
```

**Occurrences:** 632 `:begin_tab:` / 631 `:end_tab:`. `mxnet` (203),
`pytorch` (187), `tensorflow` (129), `jax` (93), multi-framework (19).

**Parser:** `d2l_preprocess.py:convert_prose_tabs()` — regex
`:begin_tab:\`([^\`]+)\``. The framework key is split on `,` and stripped.
Consecutive tab blocks are grouped into Quarto `panel-tabset` divs.
In `gen_notebooks.py:convert_prose_tabs_single()`, only the target
framework's content is kept.


## Framework Branching (Code-Level)

### `tab.interact_select()` — Framework Declaration

Declares which frameworks a file supports. Appears inside a boilerplate code
cell at the top of the file (consumed and stripped by the parser).

```python
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

Uses variadic form with single quotes. Single-framework declarations like
`tab.interact_select('pytorch')` mark framework-specific-only files
(e.g. Gaussian processes, hyperparameter optimization).

**Parser:** `d2l_preprocess.py:is_boilerplate()` detects and strips the entire
cell. `gen_notebooks.py:file_supports_framework()` reads it to decide which
notebooks to generate — regex `tab\.interact_select\(([^)]+)\)` with
`re.findall(r"['\"](\w+)['\"]", ...)` to extract framework names.


### `tab.selected()` — Conditional Framework Branch

Selects code within a cell based on framework. Used in d2l-en's contrib
directory; **not present in d2l-neu source files**.

```python
if tab.selected('pytorch'):
    x = torch.tensor(1)
if tab.selected('tensorflow'):
    x = tf.constant(1)
```

**Parser:** `build_lib.py:flatten_tab_branches()` — regex
`^(\s*)if (tab\.selected\(.+)\):\s*$`. Resolves branches for the target
framework and de-indents the body. Handles nesting via
`_resolve_nested_tabs()`. This is effectively dead code for d2l-neu but
remains for d2l-en compatibility.


## Library Extraction

### `#@save` — Mark Code for d2l Package

Marks a function, class, or statement block for extraction into the generated
`d2l` Python package (`d2l/torch.py`, `d2l/jax.py`, etc.).

Two placement styles:

```python
# Inline (most common): marker on the def/class line
def use_svg_display():  #@save
    """Use the svg format to display a plot."""
    ...

class HyperParameters:  #@save
    ...

# Also works on decorator lines
@d2l.add_to_class(Trainer)  #@save
def fit_epoch(self):
    ...

# Standalone: marker on its own line, block follows
#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

**Occurrences:** 506.

**Parser:** `build_lib.py:save_blocks()` — regex `#\s*@save` (tolerates
optional space). Collection uses indentation: for `def`/`class`/`@` blocks,
collects all indented and continuation lines; for bare statements, collects
until blank line.

**Post-processing:**
- `deduplicate_blocks()` removes earlier definitions when a name is redefined
  later (placeholder → real implementation pattern).
- `_dedup_class_methods()` deduplicates methods within merged classes, including
  their decorator lines.
- `merge_add_to_class()` splices `@d2l.add_to_class` methods into their target
  class body.


### `@d2l.add_to_class()` — Monkey-Patch a Class

Adds or replaces a method on an existing d2l class. Always paired with
`#@save`.

```python
@d2l.add_to_class(d2l.Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    ...
```

**Occurrences:** 178.

**Parser:** `build_lib.py:merge_add_to_class()` extracts the target class name
from the decorator, removes the decorator line, indents the method body, and
appends it to the target class block. The resulting merged class is then
subject to `_dedup_class_methods()`.


## Cross-References

All RST-style cross-reference directives are translated to Quarto equivalents
by `d2l_preprocess.py:translate_directives()`. The label ID convention is
`type_name` in source (underscores), translated to `type-name` (hyphens)
for Quarto.

### `:label:` — Anchor

Defines a referenceable anchor. Usually appears immediately after a heading,
figure, or equation.

```markdown
# Linear Regression
:label:`sec_linear-regression`

![Caption](img/fig.svg)
:label:`fig_example`
```

**Occurrences:** 487. **Translated to:** `{#sec-linear-regression}` (Quarto ID).


### `:numref:` — Numbered Reference

```markdown
As shown in :numref:`fig_example`, the model...
See :numref:`sec_linear-regression` for details.
```

**Occurrences:** 897. **Translated to:** `@fig-example`, `@sec-linear-regression`.


### `:eqlabel:` — Equation Label

Placed after a display equation.

```markdown
$$y = Xw + b$$
:eqlabel:`eq_linear`
```

Can appear on the same line as the closing `$$` or on the next line.

**Occurrences:** 130. **Translated to:** `{#eq-linear}`.


### `:eqref:` — Equation Reference

```markdown
From :eqref:`eq_linear` we see...
```

**Occurrences:** 193. **Translated to:** `@eq-linear`.


### `:ref:` — Generic Reference

Rarely used (`:numref:` preferred).

```markdown
See :ref:`sec_installation` for setup.
```

**Occurrences:** 10. **Translated to:** `@sec-installation`.


### `:cite:` — Parenthetical Citation

```markdown
Recent work :cite:`Vaswani.Shazeer.Parmar.ea.2017` introduced...
```

Author format: `LastName.LastName.ea.YYYY` where `ea` = et al.
Multiple citations: `:cite:`Key1,Key2``.

**Occurrences:** 481. **Translated to:** `[@Vaswani.Shazeer.Parmar.ea.2017]`.


### `:citet:` — Textual Citation

```markdown
:citet:`Bahdanau.Cho.Bengio.2014` proposed...
```

**Occurrences:** 131. **Translated to:** `@Bahdanau.Cho.Bengio.2014`.


### `:width:` — Image Width

```markdown
![Caption](img/fig.svg)
:width:`400px`
:label:`fig_example`
```

**Occurrences:** 63. **Translated to:** `width="400px"` attribute on the
image's Quarto attribute block.


## Code Fences

### `` ```{.python .input} ``

Standard Python code block. The `.input` class is a d2l-book convention;
both `.python .input` and `.python .input  n=NN` variants exist.

```
```{.python .input}
import torch
```                                   ← closing fence
```

**Occurrences:** 2,980.

**Parser:** `d2l_preprocess.py:parse_blocks()` uses `is_python_block()` to
detect any fence with `.python` in the info string. The `n=NN` attribute is
ignored.


### `` ```toc ``

Table of contents block. Lists child pages.

```
```toc
:maxdepth: 2

chapter_foo/index
chapter_bar/index
```                                   ← closing fence
```

**Occurrences:** 22. Parsed into `TocBlock` and dropped (TOC is handled by
`_quarto.yml`).


## API Documentation

`chapter_appendix-tools-for-deep-learning/d2l.md` contains only anchor
headings (`## Classes`, `## Functions`). API documentation is generated
by `gen_api_doc.py`, which reads the generated `d2l/*.py` files via the
Python `ast` module and injects per-framework tabbed docs into `d2l.qmd`.


## Slide/Highlight Markers

Inline markers for slide presentation vs. book rendering:

| Marker | Meaning | Translation |
|---|---|---|
| `[**text**]` | Show in book (visible in slides) | → `text` |
| `(**text**)` | Show in book (visible in slides) | → `text` |
| `[~~text~~]` | Slide-only (hidden from book) | → removed |
| `(~~text~~)` | Slide-only (hidden from book) | → removed |

Handled by `translate_directives()` with DOTALL regexes (markers can span
lines).


## Build Pipeline Flow

```
.md source files
    │
    ├─→ d2l_preprocess.py ─→ .qmd (multi-framework tabs for HTML)
    │     Consumes: all directives above
    │     Outputs: Quarto-native syntax
    │
    ├─→ gen_notebooks.py ─→ .ipynb (single-framework notebooks)
    │     Uses: parse_blocks, extract_tab, file_supports_framework
    │     Consumes: %%tab, #@tab, :begin_tab:/:end_tab:, tab.interact_select
    │
    ├─→ build_lib.py ─→ d2l/*.py (library package)
    │     Uses: parse_blocks, extract_tab, save_blocks, flatten_tab_branches
    │     Consumes: #@save, @d2l.add_to_class, %%tab/#@tab, tab.selected
    │
    ├─→ gen_pdf.py ─→ single-framework .qmd (for PDF rendering)
    │     Uses: parse_blocks, extract_tab
    │
    ├─→ gen_slides.py ─→ single-framework Reveal.js slides
    │     Uses: parse_blocks, extract_tab
    │
    └─→ gen_api_doc.py ─→ API documentation in d2l.qmd
          Consumes: .. autoclass::, .. autofunction::, .. currentmodule::
```

All tools except `gen_api_doc.py` share the core parsing functions from
`d2l_preprocess.py`: `parse_blocks()`, `extract_tab()`, `is_boilerplate()`,
`is_python_block()`, `clean_save_markers()`.
