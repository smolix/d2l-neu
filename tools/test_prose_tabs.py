"""Adjacent prose tab groups must not lose a block when a framework repeats."""
import importlib.util
import pathlib
import sys

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

TEXT = """Intro.

:begin_tab:`pytorch`
First pytorch.
:end_tab:

:begin_tab:`jax`
First jax.
:end_tab:

:begin_tab:`pytorch`
Second pytorch.
:end_tab:

:begin_tab:`jax`
Second jax.
:end_tab:

Outro.
"""


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_html_keeps_both_groups():
    out = _load('d2l_preprocess').convert_prose_tabs(TEXT, 'pytorch')
    assert out.count('panel-tabset') == 2
    for s in ('First pytorch.', 'First jax.', 'Second pytorch.', 'Second jax.'):
        assert s in out


def test_single_framework_keeps_both_blocks():
    for name in ('gen_pdf', 'gen_notebooks'):
        out = _load(name).convert_prose_tabs_single(TEXT, 'jax')
        assert 'First jax.' in out and 'Second jax.' in out, name
        assert 'pytorch.' not in out, name


def test_one_group_still_merges():
    text = TEXT.replace('Second pytorch.', 'x').split(':begin_tab:`pytorch`\nx')[0]
    out = _load('d2l_preprocess').convert_prose_tabs(text, 'pytorch')
    assert out.count('panel-tabset') == 1
