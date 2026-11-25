# Collective Communication at Scale: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_collectives` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1). For now:
the section's registry entries, one statement each (the box's forced algorithm
and protocol sweeps; 22.7 also reads the C-T4 log registered by 22.6; the
`DATA_HUB` entries follow the uploads) and its regeneration job. The section's
build adds `nccl_info`, `nccl_algo_proto_sweeps` (drop reasons per pair),
`plot_algo_crossover`, `tuner_regret`, `hier_vs_flat` (its torchrun script),
`coll_symptoms` and `sim_topology`. Blocks are tagged `%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_collectives`

## Registries

```{.python .input #collectives-lib-entries}
%%tab pytorch
#@save
d2l.HW_ARTIFACTS['box-coll'] = ('lm-hw-buildbox-coll-v1', 'tar')
```

```{.python .input #collectives-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-coll'] = (
    'regen_coll', lambda: dict(config='box'))
```
