"""Single-framework parts render without the framework tab bar (HTML only).

SINGLE_FRAMEWORK_PARTS in tools/d2l_preprocess.py names the parts (the
Language Models part); docs/build-system.md §6.11 describes the behavior.
"""
import importlib.util
import pathlib
import sys

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

SOURCE = """# A Section
:label:`sec_x`

Some prose.

```{.python .input #x-first}
%%tab pytorch
import torch
x = torch.ones(3)
x
```

:begin_tab:`pytorch`
A tab note for PyTorch readers.
:end_tab:

```{.python .input #x-shared}
y = 1 + 1
```

## Exercises

1. Do something.
"""


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _convert(tmp_path, chapter):
    md = tmp_path / chapter / 'x.md'
    md.parent.mkdir(parents=True)
    md.write_text(SOURCE, encoding='utf-8')
    return _load('d2l_preprocess').convert_file(str(md), 'pytorch',
                                                chapter_number=[1, 1])


def test_lm_page_has_no_tabset(tmp_path):
    qmd = _convert(tmp_path, 'chapter_lm-example')
    assert 'panel-tabset' not in qmd
    assert '## PyTorch' not in qmd
    # The code stays an executable cell with its id, so outputs still inject.
    assert '```{python}\n#| label: x-first\nimport torch' in qmd
    assert '#| label: x-shared' in qmd
    assert 'A tab note for PyTorch readers.' in qmd
    assert 'id="d2l-single-framework" data-framework="pytorch"' in qmd


def test_other_chapters_keep_tabsets(tmp_path):
    qmd = _convert(tmp_path, 'chapter_linear-regression')
    assert qmd.count('::: {.panel-tabset group="framework"}') == 2
    assert 'd2l-single-framework' not in qmd


def test_lm_outputs_inject_as_untabbed_cells(tmp_path):
    inject = _load('inject_outputs')
    qmd = _convert(tmp_path, 'chapter_lm-example')
    cells = inject.parse_qmd_cells(qmd)
    assert [(c.cell_id, c.framework) for c in cells] == [
        ('x-first', None), ('x-shared', None)]


def test_other_framework_content_dropped_with_warning(capsys):
    pre = _load('d2l_preprocess')
    text = (':begin_tab:`pytorch`\nTorch note.\n:end_tab:\n\n'
            ':begin_tab:`jax`\nJAX note.\n:end_tab:\n')
    out = pre.convert_prose_tabs(text, 'pytorch', solo='pytorch')
    assert 'Torch note.' in out and 'JAX note.' not in out
    assert 'panel-tabset' not in out
    assert 'jax' in capsys.readouterr().err


def test_parts_switched_on():
    pre = _load('d2l_preprocess')
    assert pre.single_framework('chapter_lm-data/crawl.md') == 'pytorch'
    assert pre.single_framework('chapter_lm-pretrained/index.md') == 'pytorch'
    # Not (yet) switched on: Image Models, and all other parts.
    assert pre.single_framework('chapter_image-models/index.md') is None
    assert pre.single_framework('chapter_reinforcement-learning/x.md') is None
    assert pre.single_framework('chapter_linear-regression/x.md') is None
