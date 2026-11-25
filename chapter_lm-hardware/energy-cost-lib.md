# Energy, Power and Cost: Library Internals (build-only, unlisted)

<!--
This file is not listed in `_quarto.yml`/`CHAPTER_NUMBERING` and is never
rendered or executed as a notebook: `make lib` (`tools/build_lib.py`) only
scans it for `#@save` blocks, and `gen_notebooks.py` excludes it via
`LIB_ONLY_FILES` (appended after the numbered chapters, after
`accelerator-design-lib.md`, whose empty registries these entries fill). It
holds the library code of :numref:`sec_lm_energy_cost` that the section
explains in a sentence but does not show (docs/code-guide.md 7.1). For now:
the section's registry entries, one statement each (the price table, the box's
power traces, the root clock sweep and chapter 23's measurements; the
`DATA_HUB` entries follow the uploads) and its regeneration job. The section's
build adds `power_profile`, `decode_energy`, `train_energy`,
`hw_clock_sweep_log`, `training_mfu`, `plot_dvfs`, `cap_effect`, `hw_prices`,
`plot_tornado` and `check_cost_units`. Blocks are tagged `%%tab pytorch`.
-->

<!-- Not rendered; it sets the "Defined in" suffix of the docstrings. -->
:label:`sec_lm_energy_cost`

## Registries

```{.python .input #energy-cost-lib-entries}
%%tab pytorch
#@save
d2l.HW_STATIC['prices'] = ('lm-hw-prices-v1', 'csv')
d2l.HW_ARTIFACTS['box-power'] = ('lm-hw-buildbox-power-v1', 'tar')
d2l.HW_CAMPAIGNS['box-clocksweep'] = (
    'lm-hw-buildbox-clocksweep-v1', 'tar', 'C-box-root',
    'RTX 4090 (this box, with root)')
d2l.HW_CAMPAIGNS['mfu-table'] = (
    'lm-mfu-table-v1', 'json', 'chapter 23', 'measured MFU per layout')
d2l.HW_CAMPAIGNS['par-goodput'] = (
    'lm-par-goodput-v1', 'json', 'chapter 23', 'goodput per layout')
```

```{.python .input #energy-cost-lib-jobs}
%%tab pytorch
#@save
d2l.HW_JOBS['box-power'] = (
    'regen_power', lambda: dict(config='box'))
```
