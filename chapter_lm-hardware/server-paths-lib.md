# Data Paths inside a Server: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_server_paths` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1). For now:
the section's registry entries, one statement each (the link table, the box's
topology and NCCL sweeps and the campaign logs C-T3 and C-T2pair; the
`DATA_HUB` entries follow the uploads) and its regeneration job. The section's
build adds `nccl_sweep(s)`, `copy_engine_effect`, the torch.distributed
fallback script and its timeout wrapper, `fit_collectives`, `box_alpha_beta`,
the `alpha_beta.json` writer and displays. Blocks are tagged `%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_server_paths`

## Registries

```{.python .input #server-paths-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['links'] = ('lm-hw-links-v1', 'csv')
d2l.HW_ARTIFACTS['box-p2p'] = ('lm-hw-buildbox-p2p-v1', 'tar')
d2l.HW_CAMPAIGNS['node-p2p'] = (
    'lm-hw-node-p2p-v1', 'tar', 'C-T3', 'one 8 x A100 40 GB node')
d2l.HW_CAMPAIGNS['t2pair-p2p'] = (
    'lm-hw-t2pair-p2p-v1', 'tar', 'C-T2pair',
    'two A100 40 GB with an NVLink bridge')
```

```{.python .input #server-paths-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-p2p'] = (
    'regen_p2p', lambda: dict(config='box'))
```
