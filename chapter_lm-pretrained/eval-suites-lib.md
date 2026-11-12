# Evaluation Suite Registry (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, not early: nothing
uses these symbols at import time). It will hold the library code that
:numref:`sec_lm_eval_suites` explains in one sentence and does not show:
`d2l.EVAL_ADAPTERS`, `d2l.EVAL_TASKS`, the frozen suite manifests
`d2l.EVAL_SUITES`, `_eval_load_rows`, the `_eval_cache_*` helpers and the
manifest/save/load methods of `d2l.EvalSuite`. Every block is tagged
`%%tab pytorch` and imports third-party modules inside its functions.
Stub until the section's library code is written (stage 2, step 6).
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_eval_suites`
