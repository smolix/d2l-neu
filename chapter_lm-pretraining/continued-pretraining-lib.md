# Continued Pretraining: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is
never rendered or executed as a notebook: `make lib` only scans it for
`#@save` blocks, and `gen_notebooks.py` excludes it via `LIB_ONLY_FILES`.
It holds the library code of :numref:`sec_lm_continued_pretraining`
that the section explains in a sentence but does not show
(docs/code-guide.md 7.1). For now: 21.4's committed results
(`PRETRAIN_ARTIFACTS`; the `DATA_HUB` entries follow the uploads). The
section's build adds its internals here.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_continued_pretraining`

## Registries

```{.python .input #continued-pretraining-lib-artifacts}
%%tab pytorch
#@save
d2l.PRETRAIN_ARTIFACTS['cpt-qwen-manifest'] = (
    'lm-cpt-qwen-manifest-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['cpt-t1'] = (
    'lm-cpt-t1-v1', 'jsonl')
d2l.PRETRAIN_ARTIFACTS['cpt-runs'] = (
    'lm-cpt-runs-v1', 'json-table')
d2l.PRETRAIN_ARTIFACTS['smollm3-trajectory'] = (
    'lm-smollm3-trajectory-v1', 'json-table')
```
