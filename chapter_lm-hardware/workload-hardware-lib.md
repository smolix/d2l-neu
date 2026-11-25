# Training, Prefill and Decode Workloads: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_workload_hardware` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1). For now:
the section's registry entries, one statement each (the box's workload grid
and MoE coverage, the InferenceX snapshot and the campaign logs C-T2 and
C-T2pair; the `DATA_HUB` entries follow the uploads) and its regeneration job.
The section's build adds `HW_WORKLOAD_CONFIGS`, the grid runner, capacity cut,
live checks, `hw_weight_control`, `show_calib`, the MoE loader with router
hooks and its input builder, `fit_workload_models`, `plot_frontier`,
`plot_coverage`, `inferencex_snapshot`, `rank_against`. Blocks are tagged
`%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_workload_hardware`

## Registries

```{.python .input #workload-hardware-lib-entries}
%%tab pytorch
#@save
d2l.HW_ARTIFACTS['box-workloads'] = ('lm-hw-buildbox-workloads-v1', 'jsonl')
d2l.HW_ARTIFACTS['box-moe'] = ('lm-hw-buildbox-moe-v1', 'jsonl')
d2l.HW_STATIC['inferencex-snapshot'] = ('lm-hw-inferencex-snapshot-v1', 'tar')
d2l.HW_CAMPAIGNS['a100-workloads'] = (
    'lm-hw-a100-workloads-v1', 'jsonl', 'C-T2', 'A100 PCIe 40 GB')
d2l.HW_CAMPAIGNS['t2pair-moe'] = (
    'lm-hw-t2pair-moe-v1', 'jsonl', 'C-T2pair',
    'two A100 40 GB with an NVLink bridge')
```

```{.python .input #workload-hardware-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-workloads'] = (
    'regen_workloads', lambda: dict(config='box'))
d2l.HW_JOBS['box-moe'] = (
    'regen_moe', lambda: dict(config='box'))
```
