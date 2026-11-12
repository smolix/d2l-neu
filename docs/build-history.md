# Build history

**What this is:** the record of how the build reached its current shape:
one-time migrations, implementation checklists and designs that were replaced.
**When to read it:** when you wonder why something is the way it is, or before
reintroducing an idea that was tried. Nothing here describes how to build
today; for that see [`build-guide.md`](build-guide.md) and
[`build-system.md`](build-system.md). Moved out of `build-system.md` (former
§10, §16 and history paragraphs) on 2026-11-12.

## Migration to the outputs store (2026-06)

Before June 2026 every render target read executed notebooks from
`_notebooks/`, so a CSS change needed all four framework venvs and about 110
GPU-minutes (build-system.md §1). The migration plan, as executed:

The current `_notebooks/` tree is green (129/129 MXNet, all frameworks). Bless it
into the store once, then flip the readers.

```bash
# 0. Prereqs (git-lfs/3.4.1 already installed; just wire the clone hooks)
git lfs install
git checkout -b decouple-outputs

# 1. Land .gitattributes (§7) BEFORE adding any asset, so blobs go to LFS, not git.
git add .gitattributes && git commit -m "lfs: track outputs/ image assets"

# 2. Build the new tools (capture_outputs.py, audit_outputs.py) and repoint
#    inject_outputs.py per §8.

# 3. Bless the current executed tree wholesale.
make capture-outputs                      # full sweep of _notebooks/ → outputs/
make audit-outputs                        # expect: integrity clean, 0 stale
git add outputs/ && git commit -m "outputs: initial capture of current green build"

# 4. Rewire site targets to depend on outputs/ + verify-outputs-fresh (§6.1);
#    drop the implicit _notebooks/ dependency.

# 5. Prove the decoupling: in a clean worktree with ONLY .venv-build,
#    make html && make -j4 slides  must succeed.

# 6. PR via gh; squash-merge.
```

After migration, `_notebooks/` remains a gitignored scratch dir; deleting it
costs nothing (a render no longer needs it). Re-executing is only needed to
*refresh* outputs, never to *render* them.

## Implementation status

All of this has **landed** — the table below is now a verification checklist, not
a plan. (Verified 2026-06-06.)

| Piece | State |
|---|---|
| `execute.enabled: false`, cell-id-keyed injection, `_output_fingerprint`, stable ids, `.d` deps | **Landed** — the seam this design builds on. |
| `outputs/` store, manifest schema, split (§2) | **Landed** — `outputs/` committed (≈1400 manifests; `git ls-files outputs/`); images in LFS. |
| `tools/capture_outputs.py`, `tools/audit_outputs.py` (§8) | **Landed** — both present and used by the targets below. |
| `inject_outputs.py` repoint to `outputs/` (§8) | **Landed** — `resolve_id_outputs()` prefers the store, falls back to `_notebooks/`. |
| Make rewiring + new targets (§6) | **Landed** — `capture-outputs`, `audit-outputs`, `verify-outputs-fresh`, `refresh-stale`, `render-fresh` all exist; `_book/index.html` runs `audit_outputs.py --verify-fresh`. |
| Git LFS + `.gitattributes` (§7) | **Landed** — `git-lfs/3.4.1` installed; `.gitattributes` tracks `outputs/**/*.{png,jpg,jpeg,svg,gif}`. Per-clone `git lfs install` still required once. |
| Initial bless + reader flip (see the migration above) | **Done** — `bootstrap.sh` exists; render reads `outputs/`. |
| Host-capability-aware gate (§3.3a) | **Landed** — `audit_outputs.py --verify-fresh` tiers by `nvidia-smi` GPU count. |
| Locked notebooks (§3.7) | **Landed** 2026-11 — `<!-- execution: locked; … -->`, materialized from the store, `make locks`, `make rerun-locked`, `capture-outputs UNLOCK=1`. |

## Execution before the unified scheduler

Until the unified scheduler (`tools/notebook_scheduler.py`) replaced it, the
Makefile was the scheduler: `tools/scan_notebook_manifests.py` split each
framework's stamps into `EXECUTED_CPU_<fw>`, `EXECUTED_GPU_<fw>` and
`EXECUTED_MULTI_GPU_<fw>`; `run-notebooks-<fw>` ran a `-j$(CPU_SLOTS)` sub-make
over the CPU list next to a `-j$(GPU_SLOTS)` sub-make over the single-GPU
list, then the multi-GPU list at `-j1`, with flock-based slot locks in
`tools/run_one_notebook.py`. The per-notebook processes were not pinned to
cores. The lists and the `run-notebooks-{cpu,gpu,multigpu}-<fw>` targets
still exist in `make/notebooks.mk` and the flock locks still guard a direct
`make <stamp>`; notebook-scheduler.md §7 records why the rest was removed.

## Library dependency tracking before #94

Until then the list was not transitive (the
JAX legend fix in `_trainer_validation_step` flagged none of the 27 JAX
notebooks that train with `d2l.Trainer`; a `d2l.Residual` change missed the
TF/MXNet users of `d2l.resnet18`), and a shard was named after
`sorted(names)[0]` over all names including nested `def`s: the `plot` block
became `has_one_axis.py`, so no notebook depended on `d2l.plot`, and two
blocks could collide on one shard (PyTorch `train_ch6` and
`train_concise_ch11` both became `init_weights.py`). That hid 8 PyTorch, 7
TensorFlow, 6 MXNet and 18 JAX symbols. The fix changed the dependency list
of all 622 notebooks (every notebook now also depends on `_preamble.py`;
the renaming and the closure alone changed 481), so the first audit after
`make lib` reports all of them stale once. The full re-run planned with the
fix refreshes them.

## GNU Make 3.81 and `.preprocess.stamp`

The Makefile has refused GNU Make older than 4.3 since 2026-06-15, so this can no
longer happen; it explains old advice to avoid deleting stamps by hand.

The `.d` auto-dependencies are `-include`d, so every `make` first remakes those
files (the `scan_d2l_usage` lines) and **re-execs itself**. Under the macOS
system **GNU Make 3.81**, if you manually `rm .preprocess.stamp`, that re-exec
can leave `make html` deciding `_book/index.html` is already up to date and skip
preprocessing **and** rendering entirely — it prints only the trailing
`Output:`/`Log:` echoes and exits 0, fooling you into thinking it rebuilt. Force
the stamp explicitly first (`make .preprocess.stamp`) or touch a real
prerequisite (`touch _quarto.yml`); then `make html` renders normally. Better
still, use `make -B <target>` when you deliberately want a forced rebuild rather
than deleting stamps by hand. (GNU Make ≥ 4.3 resolves the missing stamp
correctly; 3.81 also emits harmless `overriding commands for target '&'` warnings
because it predates the `&:` grouped-target syntax the Makefile uses.)

Preprocessing used to have a second pass that re-ran the `chapter_mdl-*` files,
a workaround for their absence from `CHAPTER_NUMBERING`; it was removed once
they were added to the map.

## refresh-stale before the scheduler

`make refresh-stale` used to loop `make -B …executed` one notebook at a time,
with the GPU pool mostly idle, and capture per file. It now deletes the audit's
stale stamps and hands the whole set to the scheduler (build-system.md §3.5).

## Makefile cleanup (2026-07-22)

The Makefile was split from 1,032 lines into a thin top-level file plus
`make/*.mk` modules (build-system.md §6.9), and the deploy targets replaced a
throwaway `scratchpad/fullrun.sh`. `build.sh` (superseded legacy wrapper) and
`tools/mkstamp.py` (unused) were deleted; already-applied one-shot migrations and
ad-hoc probes moved to `tools/oneshot/` (see its README). University-adoption and
diagnostic tools (`repro_mxnet_*`, `audit_slides.py`, …) are kept as **manual**
tools — see `reviews/build-system-cleanup-proposal-2026-07-22.md` for the full
inventory and rationale.
