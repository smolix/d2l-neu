# Pretraining: Import-Time Library Code (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES`. It is in `LIB_ONLY_EARLY_FILES`, so its blocks are emitted
before the numbered chapters: `d2l.PretrainConfig`
(:numref:`sec_lm_reference_model`) is a `dataclasses.dataclass`, so the
library must import `dataclasses` before that class is defined. The page
imports the module in its own imports cell, without `#@save`: a saved
import that defines no name would land in the `_preamble` freshness shard,
which every notebook of the book depends on, and stale all of them.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_data_path`

## Plain JSON Values

`_jsonable` turns a configuration (a dataclass or namedtuple, with tuples
inside) into plain JSON values; the chapter stores every result's
configuration this way and compares it on load.

```{.python .input #early-lib-plain-json-values}
%%tab pytorch
import dataclasses  #@save

def _jsonable(x):
    """x as plain JSON values (tuples become lists, namedtuples dicts)."""
    if hasattr(x, '_asdict'):
        x = x._asdict()
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        x = {f.name: getattr(x, f.name) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x
```
