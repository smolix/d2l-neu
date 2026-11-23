#!/usr/bin/env python3
"""Slide text outputs are capped at the rows their output box holds.

Covers inject_outputs.fit_slide_rows (head + `...` + tail, wrapped lines
count as several rows, the block never exceeds the cap), the per-slide
`#| d2l-output-lines:` / `#| d2l-output-width:` hand-off from gen_slides.py
(slide layout and enclosing column), that book/PDF truncation is unchanged,
and one inject round trip on a synthetic deck. Plain asserts; run directly:

    python3 tools/test_slide_output_fit.py
"""
import json
import random
import re
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import gen_slides  # noqa: E402
import inject_outputs as inj  # noqa: E402

LINES = [f'row {i}' for i in range(1, 16)]       # 15 short lines


def _rows(lines, width):
    return sum(inj._row_count(ln, width) for ln in lines)


def test_fit_short_unchanged():
    assert inj.fit_slide_rows(LINES[:10], 10, 100) == LINES[:10]


def test_fit_head_dots_tail():
    got = inj.fit_slide_rows(LINES[:11], 10, 100)
    # ceil(9/2) = 5 head rows, the `...`, 4 tail rows: 10 rows in all.
    assert got == LINES[:5] + ['...'] + LINES[7:11], got
    got = inj.fit_slide_rows(LINES, 13, 100)
    assert got == LINES[:6] + ['...'] + LINES[9:], got


def test_fit_counts_wrapped_rows():
    long = 'x' * 150                                  # 2 rows at 100 chars
    lines = [long] * 3 + LINES[:4]                    # 7 lines, 10 rows
    assert inj.fit_slide_rows(lines, 10, 100) == lines
    lines = [long] * 4 + LINES[:4]                    # 8 lines, 12 rows
    got = inj.fit_slide_rows(lines, 10, 100)
    assert got == [long] * 2 + ['...'] + LINES[:4], got
    # In a 50-character column the same line is 3 rows.
    assert inj._row_count(long, 50) == 3
    assert inj._row_count('', 50) == 1


def test_fit_never_exceeds_cap():
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(1, 30)
        lines = ['y' * rng.choice([0, 5, 60, 99, 100, 101, 160, 260])
                 for _ in range(n)]
        cap, width = rng.randint(3, 16), rng.choice([31, 47, 100, 166])
        got = inj.fit_slide_rows(lines, cap, width)
        assert _rows(got, width) <= cap or got == lines, (lines, cap, width)
        if _rows(lines, width) <= cap:
            assert got == lines
        else:
            assert '...' in got and got[0] == lines[0] or got[0] == '...'


def _text_block(md):
    body = md.split('```\n', 1)[1].rsplit('\n```', 1)[0]
    return body.split('\n')


def test_format_slides_vs_book():
    text = '\n'.join(f'line {i}' for i in range(1, 51)) + '\n'
    raw = [{'output_type': 'stream', 'name': 'stdout', 'text': [text]}]
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        md = inj.format_cell_output(raw, t, 'x', t, 'slides')
        lines = _text_block(md)
        assert len(lines) == 10 and lines[5] == '...', lines
        assert lines[0] == 'line 1' and lines[-1] == 'line 50', lines
        md = inj.format_cell_output(raw, t, 'x', t, 'slides', max_rows=13,
                                    row_width=40)
        assert len(_text_block(md)) == 13
        # PDF: unchanged 20 + `...` + 20; HTML: whole output, scroll box.
        pdf = _text_block(inj.format_cell_output(raw, t, 'x', t, 'pdf'))
        assert pdf == ([f'line {i}' for i in range(1, 21)] + ['...']
                       + [f'line {i}' for i in range(31, 51)]), pdf
        html = inj.format_cell_output(raw, t, 'x', t, 'html')
        assert '.d2l-output-scroll' in html
        assert len(_text_block(html)) == 50


def test_fit_options_by_layout():
    f = gen_slides.output_fit_options
    assert f(None, None) == []
    assert f('2col', None) == []                     # no box change
    assert f('tall-output', None) == ['#| d2l-output-lines: 13']
    assert f('tight', None) == ['#| d2l-output-lines: 6']
    assert f('code', None) == ['#| d2l-output-lines: 14',
                               '#| d2l-output-width: 166']
    assert f(None, 47) == ['#| d2l-output-width: 47']
    assert f('tall-output', 47) == ['#| d2l-output-lines: 15',
                                    '#| d2l-output-width: 47']


def test_column_widths():
    body = ['intro',
            '::: {.cols .vc}',
            '::: {.col .fig}', '@a', ':::',
            '::: {.col}', '@b', '::: {.callout}', '@c', ':::', ':::',
            ':::',
            '@d',
            '::: {.cols}',
            '::: {.col}', '@e', ':::',
            '::: {.col}', '@f', ':::',
            ':::']
    w = dict(zip(body, gen_slides._column_widths(body)))
    assert w['intro'] is None and w['@d'] is None
    assert w['@a'] == int(123 * 0.44 - 12) - 1          # 42
    assert w['@b'] == w['@c'] == int((123 * 0.56 - 4) - 12) - 1   # 52
    assert w['@e'] == w['@f'] == int((123 - 4) / 2 - 12) - 1      # 46


SRC = """# Demo
:label:`sec_demo`

```{.python .input #fit-a}
print(table)
```

<!-- slides -->

::: {.slide title="Plain"}
@fit-a
:::

::: {.slide title="Tall" layout="tall-output"}
@fit-a
:::

::: {.slide title="Column"}
::: {.cols}
::: {.col}
Some prose.
:::
::: {.col}
@fit-a
:::
:::
:::

::: {.slide title="Code only" layout="tall-output"}
@-fit-a
:::
"""


def test_deck_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / 'chapter_demo' / 'demo.md'
        src.parent.mkdir(parents=True)
        src.write_text(SRC, encoding='utf-8')
        store = root / 'outputs' / 'pytorch' / 'chapter_demo'
        store.mkdir(parents=True)
        out = '\n'.join(f'{i:2d} ' + 'z' * 60 for i in range(1, 21))
        (store / 'demo.json').write_text(json.dumps(
            {'schema': 1, 'cells': {'fit-a': {'outputs': [
                {'type': 'stream', 'name': 'stdout', 'text': out}]}}}),
            encoding='utf-8')
        qmd = gen_slides.generate_slides_qmd(src, 'pytorch', [])
        assert qmd.count('#| d2l-output-lines: 13') == 1, qmd
        assert qmd.count('#| d2l-output-width: 46') == 1, qmd
        deck = root / '_slides/pytorch/chapter_demo/demo.qmd'
        deck.parent.mkdir(parents=True)
        deck.write_text(qmd, encoding='utf-8')
        n = inj._inject_single_fw(deck, 'pytorch', root / '_notebooks',
                                  root / '_slides/img/outputs', mode='slides',
                                  fallback=False, store_dir=root / 'outputs')
        assert n == 3, n
        text = deck.read_text(encoding='utf-8')
        blocks = [_text_block(b) for b in re.findall(
            re.escape(inj.OUTPUT_START) + r'\n(.*?)\n'
            + re.escape(inj.OUTPUT_END), text, re.S)]
        plain, tall, col = blocks
        assert len(plain) == 10 and len(tall) == 13, (plain, tall)
        # 63-character lines wrap to 2 rows in a 46-character column.
        assert col[:2] == out.split('\n')[:2] and col[2] == '...', col
        assert len(col) == 5 and _rows(col, 46) <= 10, col


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')
