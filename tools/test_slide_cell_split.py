#!/usr/bin/env python3
"""Slide placeholders that show part of a cell: `@<id>[lines, out=items]`.

Covers gen_slides.parse_selector, the code-line slice (_resolve_body), the
`#| d2l-output-items:` hand-off to tools/inject_outputs.py, output-item
selection from a committed-store manifest (image files of two fragments of
one cell never collide), the build failures for bad or out-of-range
selectors, and one real `gen_slides.py --render` of a synthetic deck
(skipped when .venv-build/bin/quarto is absent). Plain asserts on a
synthetic corpus; run directly:

    python3 tools/test_slide_cell_split.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import gen_slides  # noqa: E402
import inject_outputs as inj  # noqa: E402

# `%%tab`, `#@save` and the add_to_class decorator are stripped before lines
# are counted, so line 1 of the pytorch slide code is `def f(self):`.
SGD = """# Stochastic Gradient Descent
:label:`sec_sgd`

```{.python .input #split-a}
%%tab pytorch, jax
@d2l.add_to_class(Foo)  #@save
def f(self):
    return 1

x = f(None)
y = x + 1
y
```

<!-- slides -->

::: {.slide title="Head"}
@-split-a[1-2]
:::

::: {.slide title="Tail"}
@split-a[3-]
:::

::: {.slide title="First plot"}
@!split-a[out=1]
:::

::: {.slide title="Second plot"}
@!split-a[out=2-3]
:::
"""

PLOT1 = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"><rect width="40" height="20" fill="red"/></svg>'
PLOT2 = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"><rect width="40" height="20" fill="blue"/></svg>'


def corpus(tmp, sgd=SGD):
    """A source tree + committed-store manifest for chapter_optimization/sgd."""
    root = Path(tmp)
    src = root / 'chapter_optimization' / 'sgd.md'
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(sgd, encoding='utf-8')
    store = root / 'outputs' / 'pytorch' / 'chapter_optimization'
    (store / 'sgd').mkdir(parents=True, exist_ok=True)
    (store / 'sgd' / 'split-a-1.svg').write_text(PLOT1, encoding='utf-8')
    (store / 'sgd' / 'split-a-2.svg').write_text(PLOT2, encoding='utf-8')
    manifest = {'schema': 1, 'cells': {'split-a': {'outputs': [
        {'type': 'display_data', 'mime': 'image/svg+xml',
         'asset': 'sgd/split-a-1.svg'},
        {'type': 'display_data', 'mime': 'image/svg+xml',
         'asset': 'sgd/split-a-2.svg'},
        {'type': 'execute_result', 'text': '2'},
    ]}}}
    (store / 'sgd.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root


def fences(qmd):
    """The ```{python} fences of a deck, as lists of lines."""
    return [m.group(1).split('\n') for m in
            re.finditer(r'```\{python\}\n(.*?)\n```', qmd, re.S)]


def test_parse_selector():
    ps = gen_slides.parse_selector
    assert ps('1-24') == ((1, 24), None)
    assert ps('25-') == ((25, None), None)
    assert ps('-24') == ((1, 24), None)
    assert ps('7') == ((7, 7), None)
    assert ps('out=2') == (None, (2, 2))
    assert ps('out=2-') == (None, (2, None))
    assert ps(' 25- , out=2-3 ') == ((25, None), (2, 3))
    assert ps('out=1, 3-4') == ((3, 4), (1, 1))
    for bad in ('', ' ', '0-3', '5-2', 'a-b', '1-2, 3-4', 'out=1,out=2',
                'out=', 'out=x', '1--2', '-'):
        try:
            ps(bad)
        except ValueError:
            continue
        raise AssertionError(f'parse_selector({bad!r}) accepted')
    assert gen_slides.format_range((2, 2)) == '2'
    assert gen_slides.format_range((2, None)) == '2-'
    assert gen_slides.format_range((2, 5)) == '2-5'


def test_code_lines_and_labels():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        qmd = gen_slides.generate_slides_qmd(
            root / 'chapter_optimization/sgd.md', 'pytorch', [])
        head, tail, first, second = fences(qmd)
        # Code-only head: no label (inject_outputs leaves it bare).
        assert head == ['def f(self):', '    return 1'], head
        # Lines 3- start on a blank line, which is dropped; with output.
        assert tail == ['#| label: split-a', 'x = f(None)', 'y = x + 1',
                        'y'], tail
        # Output-only fragments: deduped labels, item option, full code.
        assert first[:3] == ['#| label: split-a-fig2', '#| echo: false',
                             '#| d2l-output-items: "1"'], first
        assert second[:3] == ['#| label: split-a-fig3', '#| echo: false',
                              '#| d2l-output-items: "2-3"'], second
        assert first[3:] == ['def f(self):', '    return 1', '',
                             'x = f(None)', 'y = x + 1', 'y']


def test_no_selector_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        plain = SGD[:SGD.index('<!-- slides -->')] + (
            '<!-- slides -->\n\n::: {.slide title="All"}\n@split-a\n:::\n')
        root = corpus(tmp, plain)
        qmd = gen_slides.generate_slides_qmd(
            root / 'chapter_optimization/sgd.md', 'pytorch', [])
        (cell,) = fences(qmd)
        assert cell[0] == '#| label: split-a'
        assert 'd2l-output-items' not in qmd


def _spec_error(slide_line):
    with tempfile.TemporaryDirectory() as tmp:
        text = SGD.replace('@-split-a[1-2]', slide_line)
        root = corpus(tmp, text)
        try:
            gen_slides.generate_slides_qmd(
                root / 'chapter_optimization/sgd.md', 'pytorch', [])
        except gen_slides.SlideSpecError as e:
            return str(e)
    raise AssertionError(f'{slide_line} accepted')


def test_bad_selectors_fail():
    msg = _spec_error('@split-a[5-9]')
    assert 'out of range' in msg and 'has 6 lines' in msg, msg
    assert 'sgd.md: @split-a[5-9]' in msg, msg
    assert 'out of range' in _spec_error('@split-a[7-]')
    assert 'all blank' in _spec_error('@split-a[3]')
    assert 'output-only' in _spec_error('@!split-a[1-2]')
    assert 'code-only' in _spec_error('@-split-a[out=1]')
    assert 'bad range' in _spec_error('@split-a[x]')
    assert 'malformed' in _spec_error('@split-a[1-2')
    assert 'malformed' in _spec_error('@split-a[1-2] more')


def test_gen_slides_exits_on_bad_selector():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp, SGD.replace('@-split-a[1-2]', '@-split-a[1-99]'))
        r = subprocess.run(
            [sys.executable, str(TOOLS / 'gen_slides.py'), str(root),
             str(root / '_slides'), '--frameworks', 'pytorch',
             '--files', 'chapter_optimization/sgd.md'],
            capture_output=True, text=True, cwd=root)
        assert r.returncode == 1, r.stdout + r.stderr
        assert 'line range 1-99 is out of range' in r.stderr, r.stderr
        assert 'bad placeholder selector' in r.stderr, r.stderr


def _inject(root, sgd=SGD):
    """Generate + inject the pytorch deck; return (qmd text, img dir)."""
    qmd = gen_slides.generate_slides_qmd(
        root / 'chapter_optimization/sgd.md', 'pytorch', [])
    deck = root / '_slides/pytorch/chapter_optimization/sgd.qmd'
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_text(qmd, encoding='utf-8')
    img = root / '_slides/img/outputs'
    n = inj._inject_single_fw(deck, 'pytorch', root / '_notebooks', img,
                               mode='slides', fallback=False,
                               store_dir=root / 'outputs')
    return deck.read_text(encoding='utf-8'), img, n


def _blocks(text):
    """Injected output blocks, in deck order."""
    return re.findall(re.escape(inj.OUTPUT_START) + r'\n(.*?)\n'
                      + re.escape(inj.OUTPUT_END), text, re.S)


def test_inject_output_items():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        text, img, n = _inject(root)
        assert n == 3, n          # head is code-only
        tail, first, second = _blocks(text)
        # Full outputs on the tail fragment: both plots and the result.
        assert 'sgd-split-a-pytorch-1.svg' in tail
        assert 'sgd-split-a-pytorch-2.svg' in tail and '\n2\n' in tail
        # Plot 1 alone; plot 2 + result. Image files carry the plot's ordinal
        # in the whole cell under their own prefix, so they never collide.
        assert first.count('![]') == 1 and '```' not in first, first
        assert 'sgd-split-a-pytorch-items-1.svg' in first, first
        assert second.count('![]') == 1 and '\n2\n' in second, second
        assert 'sgd-split-a-pytorch-items-2.svg' in second, second
        d = img / 'chapter_optimization'
        assert (d / 'sgd-split-a-pytorch-items-1.svg').read_text() == PLOT1
        assert (d / 'sgd-split-a-pytorch-items-2.svg').read_text() == PLOT2
        # Re-injection is idempotent: the option survives in the fence.
        deck = root / '_slides/pytorch/chapter_optimization/sgd.qmd'
        inj._inject_single_fw(deck, 'pytorch', root / '_notebooks', img,
                              mode='slides', fallback=False,
                              store_dir=root / 'outputs')
        assert deck.read_text(encoding='utf-8') == text


def test_inject_out_of_range_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp, SGD.replace('[out=2-3]', '[out=2-4]'))
        try:
            _inject(root)
        except inj.OutputSelectionError as e:
            msg = str(e)
        else:
            raise AssertionError('out=2-4 accepted')
        assert 'cell split-a: out=2-4 is out of range' in msg, msg
        assert 'the cell has 3 output item(s)' in msg, msg
        assert 'out=1: display_data image/svg+xml' in msg, msg
        assert 'out=3: execute_result' in msg, msg


def test_stream_items_coalesce():
    raw = [{'output_type': 'stream', 'name': 'stdout', 'text': ['a\n']},
           {'output_type': 'stream', 'name': 'stdout', 'text': ['b\n']},
           {'output_type': 'display_data',
            'data': {'image/svg+xml': PLOT1}},
           {'output_type': 'error', 'ename': 'E'},
           {'output_type': 'stream', 'name': 'stdout', 'text': ['c\n']}]
    items = inj.output_items(raw)
    assert len(items) == 3, items      # a+b merged; error dropped
    with tempfile.TemporaryDirectory() as tmp:
        md = inj.format_cell_output(raw, Path(tmp), 'x', Path(tmp), 'slides',
                                    item_range=(1, 1))
        assert md.count('```') == 2 and 'a\nb' in md and '\nc\n' not in md, md
        md = inj.format_cell_output(raw, Path(tmp), 'x', Path(tmp), 'slides',
                                    item_range=(2, None))
        assert '![](x-1.svg)' in md and '\nc\n' in md and '\na\n' not in md, md


def test_render_split_deck():
    """One real render: gen_slides --render, as `make slides` runs it."""
    quarto = REPO / '.venv-build' / 'bin' / 'quarto'
    if not quarto.exists():
        print('  (skipped: no .venv-build/bin/quarto)')
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        shutil.copy(REPO / '_d2l-slides.scss', root / '_d2l-slides.scss')
        # Decks include the rendered overlay (tools/i18n.py render).
        sys.path.insert(0, str(TOOLS))
        import i18n
        (root / '_i18n').mkdir()
        (root / '_i18n' / '_d2l-slides-overlay.html').write_text(
            i18n.render_template(
                (REPO / '_d2l-slides-overlay.html').read_text()))
        (root / 'img').mkdir()
        (root / '.venv-build').symlink_to(REPO / '.venv-build')
        env = {**os.environ,
               'PATH': f'{quarto.parent}{os.pathsep}{os.environ["PATH"]}'}
        r = subprocess.run(
            [sys.executable, str(TOOLS / 'gen_slides.py'), '.', '_slides',
             '--frameworks', 'pytorch', '--render', '--workers', '1',
             '--files', 'chapter_optimization/sgd.md'],
            capture_output=True, text=True, cwd=root, env=env, timeout=600)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        html = (root / '_slides/pytorch/chapter_optimization/sgd.html'
                ).read_text(encoding='utf-8')
        sections = re.split(r'<section id="', html)

        def section(title):
            (s,) = [s for s in sections if f'>{title}</h2>' in s]
            return s

        def text(s):            # highlighting splits code into token spans
            return re.sub(r'<[^>]+>', '', s)
        head, tail = section('Head'), section('Tail')
        first, second = section('First plot'), section('Second plot')
        # Fragments keep Python highlighting; no option text leaks.
        assert 'sourceCode python' in head and 'sourceCode python' in tail
        assert 'return' in text(head) and 'x + 1' not in text(head)
        assert 'x + 1' in text(tail) and 'return' not in text(tail)
        assert 'd2l-output-items' not in html
        assert 'sgd-split-a-pytorch-items-1.svg' in first, first
        assert 'items-2.svg' not in first and 'sourceCode' not in first
        assert 'sgd-split-a-pytorch-items-2.svg' in second, second
        assert 'items-1.svg' not in second


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')
