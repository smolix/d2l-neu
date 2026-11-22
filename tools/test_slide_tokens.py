#!/usr/bin/env python3
"""Section-number tokens in slide text, and per-chapter deck frameworks.

Covers tools/section_refs.py (§{label} / §{num:label}), its use in
gen_slides.py (fail loudly; resolved text), the one-shot converter
tools/oneshot/slide_section_tokens.py (round trip), and
d2l_preprocess.page_frameworks (decks and exercise panes of untagged pages
follow the chapter). Plain asserts on a synthetic corpus; run directly:

    python3 tools/test_slide_tokens.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'oneshot'))

import d2l_preprocess as pre  # noqa: E402
import gen_slides  # noqa: E402
import section_refs  # noqa: E402
import slide_section_tokens as conv  # noqa: E402

INDEX = """# Optimization
:label:`chap_optimization`

Prose.

<!-- slides -->

::: {.slide title="Roadmap of Chapter §{num:chap_optimization}"}
Start with §{sec_sgd}; then §{sec_sgd}–§{num:sec_momentum}.
:::
"""

SGD = """# Stochastic Gradient Descent
:label:`sec_sgd`

```{.python .input #sgd-a}
%%tab pytorch, jax
x = 1
```

## Exercises

1. Shared problem.

:begin_tab:`pytorch`
2. PyTorch problem.
:end_tab:

:begin_tab:`tensorflow`
2. TensorFlow problem.
:end_tab:

<!-- slides -->

::: {.slide title="SGD"}
@sgd-a
:::
"""

MOMENTUM = """# Momentum
:label:`sec_momentum`

```{.python .input #mom-a}
%%tab pytorch, jax
y = 2
```
"""

PRELIM_INDEX = """# Preliminaries
:label:`chap_preliminaries`

<!-- slides -->

::: {.slide title="Roadmap"}
Text.
:::
"""

NDARRAY = """# Data Manipulation
:label:`sec_ndarray`

```{.python .input #nd-a}
%%tab pytorch, tensorflow, jax, mxnet
z = 3
```
"""


def corpus(tmp):
    files = {
        'chapter_optimization/index.md': INDEX,
        'chapter_optimization/sgd.md': SGD,
        'chapter_optimization/momentum.md': MOMENTUM,
        'chapter_preliminaries/index.md': PRELIM_INDEX,
        'chapter_preliminaries/ndarray.md': NDARRAY,
    }
    for rel, text in files.items():
        p = Path(tmp) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
    return Path(tmp)


def numbers():
    n = pre.CHAPTER_NUMBERING
    return (n['chapter_optimization/index.md'][0],
            '.'.join(map(str, n['chapter_optimization/sgd.md'])),
            '.'.join(map(str, n['chapter_optimization/momentum.md'])))


def test_resolve():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        ch, sgd, mom = numbers()
        out = section_refs.resolve(INDEX, 'x.md', root)
        assert f'Roadmap of Chapter {ch}"' in out, out
        assert f'Start with §{sgd}; then §{sgd}–{mom}.' in out, out
        assert '§{' not in out


def test_renumbering_follows_the_map():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        saved = dict(section_refs.CHAPTER_NUMBERING)
        try:
            section_refs.CHAPTER_NUMBERING['chapter_optimization/index.md'] = [99]
            section_refs.CHAPTER_NUMBERING['chapter_optimization/sgd.md'] = [99, 4]
            section_refs._label_table.cache_clear()
            out = section_refs.resolve(INDEX, 'x.md', root)
            assert 'Roadmap of Chapter 99"' in out and '§99.4' in out, out
        finally:
            section_refs.CHAPTER_NUMBERING.clear()
            section_refs.CHAPTER_NUMBERING.update(saved)
            section_refs._label_table.cache_clear()


def test_bad_tokens_are_errors():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        text = ('# T\n:label:`sec_t`\n\nSee §{sec_sgd} above.\n\n'
                '<!-- slides -->\n\n::: {.slide}\n'
                'Bad §{sec_nope}, empty §{}, sub §{num:sec_x_sub}.\n:::\n')
        probs = section_refs.check(text, 'f.md', root)
        assert [p[1] for p in probs] == [4, 9, 9, 9], probs
        assert 'outside the slides section' in probs[0][2]
        assert 'unknown section label `sec_nope`' in probs[1][2]
        try:
            section_refs.resolve(text, 'f.md', root)
        except section_refs.SectionRefError as e:
            assert 'f.md:9: error:' in str(e)
        else:
            raise AssertionError('no SectionRefError')


def test_gen_slides_fails_loudly():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        p = root / 'chapter_optimization/index.md'
        p.write_text(INDEX.replace('§{sec_sgd};', '§{sec_typo};'),
                     encoding='utf-8')
        r = subprocess.run(
            [sys.executable, str(TOOLS / 'gen_slides.py'), str(root),
             str(root / '_slides'), '--frameworks', 'pytorch'],
            capture_output=True, text=True)
        assert r.returncode != 0
        assert 'chapter_optimization/index.md:9: error:' in r.stderr, r.stderr
        assert 'sec_typo' in r.stderr
        assert not (root / '_slides').exists()


def test_page_frameworks():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        opt = root / 'chapter_optimization'
        assert pre.page_frameworks(opt / 'index.md') == ['pytorch', 'jax']
        assert pre.page_frameworks(opt / 'sgd.md') == ['pytorch', 'jax']
        four = ['pytorch', 'tensorflow', 'jax', 'mxnet']
        assert pre.page_frameworks(
            root / 'chapter_preliminaries/index.md') == four
        lone = root / 'chapter_nowhere/x.md'
        lone.parent.mkdir()
        lone.write_text('# X\n', encoding='utf-8')
        assert pre.page_frameworks(lone) == four
        assert not gen_slides.has_framework_code(opt / 'index.md', 'mxnet')
        assert gen_slides.has_framework_code(opt / 'index.md', 'jax')


def test_stale_decks_removed():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        out = root / '_slides'
        stale = [out / 'tensorflow/chapter_optimization/index.qmd',
                 out / 'tensorflow/chapter_optimization/index.html',
                 out / 'tensorflow/chapter_optimization/gone.html',
                 out / 'errors/tensorflow/chapter_optimization/index.log']
        for f in stale:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text('old', encoding='utf-8')
        (out / 'tensorflow/chapter_optimization/index_files').mkdir()
        r = subprocess.run(
            [sys.executable, str(TOOLS / 'gen_slides.py'), str(root),
             str(out), '--frameworks', 'tensorflow', 'pytorch'],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert not any(f.exists() for f in stale), r.stdout
        assert not (out / 'tensorflow/chapter_optimization/index_files').exists()
        assert (out / 'tensorflow/chapter_preliminaries/index.qmd').exists()
        assert (out / 'pytorch/chapter_optimization/index.qmd').exists()


def test_exercise_panes_follow_page():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        q = pre.convert_file(root / 'chapter_optimization/sgd.md', 'pytorch',
                             [9, 3])
        ex = q[q.index('Exercises'):]
        assert '## PyTorch' in ex and '## JAX' in ex
        assert '## TensorFlow' not in ex and '## MXNet' not in ex, ex
        text = SGD.replace('%%tab pytorch, jax',
                           '%%tab pytorch, tensorflow, jax, mxnet')
        (root / 'chapter_optimization/sgd.md').write_text(text)
        q = pre.convert_file(root / 'chapter_optimization/sgd.md', 'pytorch',
                             [9, 3])
        ex = q[q.index('Exercises'):]
        assert all(f'## {d}' in ex for d in
                   ('PyTorch', 'TensorFlow', 'JAX', 'MXNet')), ex


def test_converter_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        root = corpus(tmp)
        ch, sgd, mom = numbers()
        text = ('# T\n:label:`sec_t`\n\nAbove the marker: §{0}.\n\n'
                '<!-- slides -->\n\n'
                '::: {{.cover}}\n[Dive into Deep Learning · §{0}]{{.kicker}}\n'
                ':::\n\n'
                '::: {{.slide title="Recap of Chapter {1}"}}\n'
                'See §{0}–{2}, §{0}–§{2}, Chapters {1} and {1}, '
                'Section {0}, `§{0}` and $§{0}$; §§{0}, {2}.\n'
                'Unknown §77.77; short §{0}–5; cited §{0} [@Doe.2020].\n'
                ':::\n').format(sgd, ch, mom)
        c = conv.Converter(root)
        new, changes, refusals = c.convert_text(text)
        assert len(changes) == 6, changes
        assert len(refusals) == 3, refusals
        assert f'Above the marker: §{sgd}.' in new
        assert f'[Dive into Deep Learning · §{sgd}]{{.kicker}}' in new
        assert '`§' + sgd + '`' in new and '$§' + sgd + '$' in new
        assert 'Recap of Chapter §{num:chap_optimization}' in new
        assert '§{sec_sgd}–§{num:sec_momentum}' in new
        assert '§{sec_sgd}–§{sec_momentum}' in new
        assert '§§{sec_sgd}, §{num:sec_momentum}' in new
        # Resolving the converted slide text gives back the original.
        start = text.index('<!-- slides -->')
        assert section_refs.resolve(new[start:], 'x.md', root) == text[start:]


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')
