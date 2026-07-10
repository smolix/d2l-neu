# v2 → promotion checklist (collected from the writing pass, 2026-07-09)

## GPU-box verification / capture (cells guarded on CPU)

- **6.7 gpus-devices-memory**: four guarded cells need real GPU capture:
  allocated-vs-reserved demo (~1 GiB), four-plateau instrumented step
  (38M-param net, batch 4096), activation-checkpointing peak-memory
  comparison (16×1024-wide blocks, batch 8192, ≈2 GiB peak), matmul-queue +
  pinned-copy timings. All sized to fit one ~11 GB scheduler slot.
- **6.6 saving-loading**: `save_checkpoint`/`load_checkpoint` CUDA RNG
  branches are untested on this laptop; verify on GPU.
- **6.8 reproducibility-inspection**: guarded determinism cell should raise
  the documented `RuntimeError` (`kthvalue` under
  `use_deterministic_algorithms(True)` on CUDA); check the captured output
  reads well.
- **6.2 / 6.4**: guarded `.to('cuda')` cells print the CUDA path on GPU;
  prose reads correctly either way.
- **6.5 numerics**: nothing; all cells CPU-by-design (`autocast('cpu', ...)`).

## Promotion-time edits (when v2 replaces chapter_builders-guide/)

1. **Adopt the old labels.** Earlier chapters cite `sec_use_gpu`
   (ndarray.md), `sec_model_construction` + `sec_lazy_init` (oo-design.md),
   `chap_computation` (preface, numerical-stability). At promotion, rename
   the v2 `*_v2` labels to the old names (sec_model_construction_v2 →
   sec_model_construction, etc.) so inbound refs keep resolving.
   `sec_lazy_init` and `subsec_param-access` /
   `subsec_model-construction-sequential` already use the old names.
   `fig_blocks_v2` can revert to `fig_blocks` once the old chapter is gone.
2. **Revert the try_gpu cell to untagged.** The old `try_gpu`/`try_all_gpus`
   cell is untagged (all frameworks share it via d2l dispatch); the v2 pass
   tagged it `%%tab pytorch`. When the other tabs land, restore the untagged
   form or the other frameworks' d2l libs lose the helpers. Function bodies
   verified byte-identical.
3. **Byte-identity holds** for every carried pytorch `#@save`
   (cpu/gpu, num_gpus, try_gpu/try_all_gpus, three Trainer patches,
   apply_init) — verified programmatically; zero downstream recapture needed
   for these on promotion. `set_scratch_params_device` (mxnet-only, zero
   users) intentionally dropped.
4. **New d2l symbols**: `save_checkpoint`/`load_checkpoint` (6.6, pytorch
   tab). `safetensors` added to the `run` extra in pyproject (all four venvs
   get it on next `uv sync`).
5. **Update CHAPTER_NUMBERING + _quarto.yml** to the new file set (8 sections
   vs 7; lazy-init.md folded into model-construction.md).
6. **Pending tabs**: JAX first, then TF; MXNet last with the reduced 6.5
   variant per the framework-coverage review (see _outline/index.md for the
   full matrix and skip lists).
7. **Slides**: no `<!-- slides -->` blocks yet; add during the slide pass.
