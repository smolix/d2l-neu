# Scale-Up Domains and Scale-Out Networks: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_networks` that the section explains
in a sentence but does not show (docs/code-guide.md 7.1). For now: the
section's registry entries, one statement each (the systems table, the box's
loopback sweeps and the campaign log C-T4; the `DATA_HUB` entries follow the
uploads) and its regeneration job. The section's build adds `loopback_sweep`,
`transport_ladder`, `link_betas`, `quoted_latencies` (with sources),
`fit_perftest`, `plot_layout_fit` and `SYSTEMS_SHORT`. Blocks are tagged
`%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_networks`

## Registries

```{.python .input #networks-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['systems'] = ('lm-hw-systems-v1', 'json')
d2l.HW_ARTIFACTS['box-net'] = ('lm-hw-buildbox-net-v1', 'tar')
d2l.HW_CAMPAIGNS['cluster-nccl'] = (
    'lm-hw-cluster-nccl-v1', 'tar', 'C-T4',
    'two 8 x A100 nodes with RDMA')
```

```{.python .input #networks-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-net'] = (
    'regen_net', lambda: dict(config='box'))
```
