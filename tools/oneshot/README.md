# tools/oneshot/ — archived one-shot & investigation scripts

These scripts ran **once** to perform a migration or an ad-hoc investigation and
are **not part of the build**. They are kept for provenance (what was done, and
how) but are not wired into any `make` target and are not expected to run again
as-is. Nothing in the live build imports or shells out to them.

| Script | What it did (once) |
|---|---|
| `migrate_slide_markers.py` | Converted inline `[**…**]` slide markers → `::: {.slide}` divs (slides refactor). |
| `strip_tab_all.py` | Removed `#@tab all` / `%%tab all` from the corpus (untagged = all frameworks). |
| `split_tab_selected.py` | Split `tab.selected()` cells during the same tab-marker cleanup. |
| `patch_slides_navlink_r2.sh` | One-time remediation of a stale nav href on ~520 already-live R2 slide decks. |
| `compare_convergence.py` / `extract_convergence.py` | One-shot analysis pair comparing training-convergence across a store recapture. |
| `test_jax_threads.py` / `test_jax_threads2.py` | Ad-hoc probes measuring JAX process thread counts; conclusions are baked into the Makefile's `EXTRA_ENV_jax` commentary. |
| `make_lm_eval_arc_fixture.py` | Made `lm-eval-reference-arc-v1` (18.4): lm-evaluation-harness 0.4.13 on 200 ARC-Challenge items, Qwen3-0.6B-Base fp32, in a throwaway env (2026-11-13). |
| `make_lm_heldout_texts.py` | Made `lm-heldout-texts-v1` (18.4): book prose and code written after the Qwen3.5 release, at a pinned commit (2026-11-13). |
| `build_lm_recall_pairs.py` | Made `lm-interp-recall-v1` (18.6): 40 landmark/city recall pairs filtered by Qwen3-0.6B's clean top-1 (2026-11-13). |

If you need one again, review it before running — paths and APIs have moved on
since it was written.
