# HyperParameters Base Class (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`. It is in `LIB_ONLY_EARLY_FILES`, so the class is emitted
before the numbered chapters: `Module`, `DataModule`, `Trainer`, and `SGD`
in :numref:`sec_oo-design` inherit from `d2l.HyperParameters` at import
time. The book does not show this class (review finding OO-2: no stub
shown only to be replaced). Its one method, `save_hyperparameters`, is
taught and saved in `chapter_appendix-tools-for-deep-learning/utils.md`
with `@d2l.add_to_class(d2l.HyperParameters)`, which `build_lib.py` merges
into this class body.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the class docstring. -->
:label:`sec_utils`

```{.python .input}
class HyperParameters:  #@save
    """The base class of hyperparameters."""
```
