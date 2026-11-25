# Low-Precision Training: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is
never rendered or executed as a notebook: `make lib` only scans it for
`#@save` blocks, and `gen_notebooks.py` excludes it via `LIB_ONLY_FILES`.
It holds the library code of :numref:`sec_lm_low_precision`
that the section explains in a sentence but does not show
(docs/code-guide.md 7.1). For now: 21.3's committed results
(`PRETRAIN_ARTIFACTS`; the `DATA_HUB` entries follow the uploads). The
section's build adds its internals here.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_low_precision`

## Registries

```{.python .input #low-precision-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['gemm-sweep'] = (
    'lm-fp8-gemm-sweep-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['lowp-runs'] = (
    'lm-lowp-runs-v1', 'jsonl')
```
