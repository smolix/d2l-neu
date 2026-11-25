# Memory and Storage Tiers: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_memory_tiers` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1). For now:
the section's registry entries, one statement each (the tier table, the box's
storage sweeps and the campaign log C-T3; the `DATA_HUB` entries follow the
uploads) and its regeneration job. The section's build adds
`HW_OFFLOAD_CONFIG`, the sweeps (host links, DRAM, fio, object store, CPU
optimizer), the offloaded run, scratch-file handling and the disk check, and
displays. Blocks are tagged `%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_memory_tiers`

## Registries

```{.python .input #memory-tiers-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['tiers'] = ('lm-hw-tiers-v1', 'csv')
d2l.HW_ARTIFACTS['box-storage'] = ('lm-hw-buildbox-storage-v1', 'tar')
d2l.HW_CAMPAIGNS['node-storage'] = (
    'lm-hw-node-storage-v1', 'tar', 'C-T3', 'one 8 x A100 40 GB node')
```

```{.python .input #memory-tiers-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-storage'] = (
    'regen_storage', lambda: dict(config='box'))
```
